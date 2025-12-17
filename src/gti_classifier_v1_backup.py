"""
GTI Classifier v1.0
Game Theory Index — Deterministic classification from orthogonal fields
"""

from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
import json
import yaml


class GameType(Enum):
    PRISONERS_DILEMMA = "Prisoners_Dilemma"
    CHICKEN = "Chicken"
    STAG_HUNT = "Stag_Hunt"
    COORDINATION = "Coordination"
    BATTLE_OF_SEXES = "Battle_of_Sexes"
    DEADLOCK = "Deadlock"
    HARMONY = "Harmony"
    ASSURANCE = "Assurance"
    SEQUENTIAL = "Sequential"
    ASYMMETRIC = "Asymmetric"
    INCOMPLETE_INFO = "Incomplete_Info"
    N_PLAYER = "N_Player"
    UNKNOWN = "Unknown"


@dataclass
class ClassificationResult:
    game_type: GameType
    confidence: int
    reasoning: str
    fingerprint: Dict


def normalize_matrix(matrix: Dict) -> Dict:
    """Normalize matrix keys to A_A, A_B, B_A, B_B format."""
    if 'A_A' in matrix:
        return matrix

    # Get keys and sort them to determine order
    keys = list(matrix.keys())

    if len(keys) != 4:
        return None

    # Try to identify pattern: look for keys with same prefix/suffix
    # Common patterns: Option1_Option1, Cooperate_Cooperate, Swerve_Swerve, etc.
    # Split by underscore and find unique first/second parts
    first_parts = []
    second_parts = []

    for key in keys:
        parts = key.split('_')
        if len(parts) >= 2:
            first_parts.append(parts[0])
            second_parts.append('_'.join(parts[1:]))

    unique_first = list(dict.fromkeys(first_parts))
    unique_second = list(dict.fromkeys(second_parts))

    if len(unique_first) != 2 or len(unique_second) != 2:
        return None

    # Map to A/B (first option = A = cooperate, second = B = defect)
    opt_a_first = unique_first[0]
    opt_b_first = unique_first[1]
    opt_a_second = unique_second[0]
    opt_b_second = unique_second[1]

    # Build normalized matrix
    normalized = {}
    for key, value in matrix.items():
        parts = key.split('_')
        first = parts[0]
        second = '_'.join(parts[1:])

        new_first = 'A' if first == opt_a_first else 'B'
        new_second = 'A' if second == opt_a_second else 'B'
        new_key = f"{new_first}_{new_second}"
        normalized[new_key] = value

    return normalized


def extract_fingerprint(matrix: Dict) -> Dict:
    """Extract structural fingerprint from 2x2 payoff matrix."""
    # Normalize matrix keys first
    matrix = normalize_matrix(matrix)
    if not matrix:
        return None

    CC = matrix['A_A'][0]
    CD = matrix['A_B'][0]
    DC = matrix['B_A'][0]
    DD = matrix['B_B'][0]

    T, R, P, S = DC, CC, DD, CD

    return {
        'CC': CC, 'CD': CD, 'DC': DC, 'DD': DD,
        'T': T, 'R': R, 'P': P, 'S': S,
        'defect_dominates': (T > R) and (P > S),
        'cooperate_dominates': (R > T) and (S > P),
        'no_dominant': not ((T > R and P > S) or (R > T and S > P)),
        'mutual_cooperate_best': R == max(T, R, P, S),
        'mutual_defect_worst': P == min(T, R, P, S),
        'P_gt_S': P > S,
    }


def classify_2x2_symmetric(fingerprint: Dict) -> ClassificationResult:
    """Classify 2x2 symmetric games using fingerprint patterns."""
    T, R, P, S = fingerprint['T'], fingerprint['R'], fingerprint['P'], fingerprint['S']

    # COORDINATION (check first)
    if (R > T and R > S and P > T and P > S and abs(R - P) <= 1):
        return ClassificationResult(
            game_type=GameType.COORDINATION,
            confidence=95,
            reasoning=f"R({R}) ≈ P({P}) >> T({T}), S({S}): Pure coordination",
            fingerprint=fingerprint
        )

    # BATTLE OF SEXES
    if (R > T and R > S and P > T and P > S and R != P):
        return ClassificationResult(
            game_type=GameType.BATTLE_OF_SEXES,
            confidence=90,
            reasoning=f"R({R}) ≠ P({P}), both >> T({T}), S({S}): Coordination with conflict",
            fingerprint=fingerprint
        )

    # PRISONER'S DILEMMA: T > R > P > S
    if (fingerprint['defect_dominates'] and R > P and T > R):
        return ClassificationResult(
            game_type=GameType.PRISONERS_DILEMMA,
            confidence=95,
            reasoning=f"T({T}) > R({R}) > P({P}) > S({S}): Defect dominates, social dilemma",
            fingerprint=fingerprint
        )

    # CHICKEN: T > R > S > P
    if (fingerprint['no_dominant'] and fingerprint['mutual_defect_worst'] and T > R):
        return ClassificationResult(
            game_type=GameType.CHICKEN,
            confidence=95,
            reasoning=f"T({T}) > R({R}) > S({S}) > P({P}): No dominant, mutual extreme = disaster",
            fingerprint=fingerprint
        )

    # STAG HUNT: R > T > P > S
    if (fingerprint['mutual_cooperate_best'] and fingerprint['P_gt_S'] and not fingerprint['defect_dominates']):
        return ClassificationResult(
            game_type=GameType.STAG_HUNT,
            confidence=95,
            reasoning=f"R({R}) > T({T}) > P({P}) > S({S}): Cooperate best, defect safe",
            fingerprint=fingerprint
        )

    # DEADLOCK: T > P > R > S
    if (fingerprint['defect_dominates'] and P > R):
        return ClassificationResult(
            game_type=GameType.DEADLOCK,
            confidence=90,
            reasoning=f"T({T}) > P({P}) > R({R}) > S({S}): Defect dominates, no dilemma",
            fingerprint=fingerprint
        )

    # HARMONY: R > T > S > P
    if fingerprint['cooperate_dominates']:
        return ClassificationResult(
            game_type=GameType.HARMONY,
            confidence=90,
            reasoning=f"Cooperate dominates: No conflict",
            fingerprint=fingerprint
        )

    return ClassificationResult(
        game_type=GameType.UNKNOWN,
        confidence=50,
        reasoning=f"T={T}, R={R}, P={P}, S={S}: No standard pattern",
        fingerprint=fingerprint
    )


def classify_case(case: Dict) -> ClassificationResult:
    """Main classification entry point."""
    flags = case.get('notes', {}).get('flags', [])

    if 'incomplete-info' in flags:
        return ClassificationResult(GameType.INCOMPLETE_INFO, 30, "Incomplete information game", {})

    if 'n-player' in flags:
        return ClassificationResult(GameType.N_PLAYER, 30, "N-player game", {})

    if case.get('information', {}).get('timing') == 'sequential':
        return ClassificationResult(GameType.SEQUENTIAL, 70, "Sequential game", {})

    matrix = case.get('payoffs', {}).get('matrix', {})
    if not matrix:
        return ClassificationResult(GameType.UNKNOWN, 0, "No valid payoff matrix", {})

    # Normalize matrix keys
    normalized = normalize_matrix(matrix)
    if not normalized:
        return ClassificationResult(GameType.UNKNOWN, 0, "Could not normalize payoff matrix", {})

    fingerprint = extract_fingerprint(matrix)
    if not fingerprint:
        return ClassificationResult(GameType.UNKNOWN, 0, "Could not extract fingerprint", {})

    # Check for Battle of Sexes (asymmetric coordination)
    is_asymmetric = case.get('symmetry', {}).get('symmetric') == False

    if is_asymmetric:
        # Check if it's Battle of Sexes pattern
        # Both prefer coordinating, but on different outcomes
        aa = normalized['A_A']
        bb = normalized['B_B']
        ab = normalized['A_B']
        ba = normalized['B_A']

        # BoS: coordination payoffs >> miscoordination
        coord_good_p1 = (aa[0] > ab[0] and aa[0] > ba[0] and bb[0] > ab[0] and bb[0] > ba[0])
        coord_good_p2 = (aa[1] > ab[1] and aa[1] > ba[1] and bb[1] > ab[1] and bb[1] > ba[1])
        # Players prefer different coordination points
        p1_prefers_different = aa[0] != bb[0]
        p2_prefers_different = aa[1] != bb[1]

        if coord_good_p1 and coord_good_p2 and (p1_prefers_different or p2_prefers_different):
            return ClassificationResult(
                game_type=GameType.BATTLE_OF_SEXES,
                confidence=90,
                reasoning=f"Asymmetric coordination: both prefer matching, different favorites",
                fingerprint=fingerprint
            )

        return ClassificationResult(GameType.ASYMMETRIC, 70, "Asymmetric game", fingerprint)

    return classify_2x2_symmetric(fingerprint)


def process_file(input_path: str, output_path: str = None):
    """Process a YAML case file and output classification."""
    with open(input_path, 'r', encoding='utf-8') as f:
        case = yaml.safe_load(f)

    result = classify_case(case)

    output = {
        'case_id': case.get('case_id', 'unknown'),
        'title': case.get('title', ''),
        'game_type': result.game_type.value,
        'confidence': result.confidence,
        'reasoning': result.reasoning
    }

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

    return output


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = process_file(sys.argv[1])
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python gti_classifier.py <case.yaml>")
