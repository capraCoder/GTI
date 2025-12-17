"""
GTI Classifier v2.0
Game Theory Index — Extended coverage: symmetric, asymmetric, sequential games
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import yaml


class GameType(Enum):
    # Symmetric simultaneous
    PRISONERS_DILEMMA = "Prisoners_Dilemma"
    CHICKEN = "Chicken"
    STAG_HUNT = "Stag_Hunt"
    COORDINATION = "Coordination"
    BATTLE_OF_SEXES = "Battle_of_Sexes"
    DEADLOCK = "Deadlock"
    HARMONY = "Harmony"
    ASSURANCE = "Assurance"

    # Sequential
    ULTIMATUM = "Ultimatum"
    STACKELBERG = "Stackelberg"
    ENTRY_DETERRENCE = "Entry_Deterrence"
    CENTIPEDE = "Centipede"
    TRUST_GAME = "Trust_Game"
    SEQUENTIAL_BARGAINING = "Sequential_Bargaining"

    # Asymmetric
    PRINCIPAL_AGENT = "Principal_Agent"
    INSPECTION_GAME = "Inspection_Game"
    MATCHING_PENNIES = "Matching_Pennies"
    HAWK_DOVE_ASYMMETRIC = "Hawk_Dove_Asymmetric"

    # Special
    INCOMPLETE_INFO = "Incomplete_Info"
    N_PLAYER = "N_Player"
    MIXED_MOTIVE = "Mixed_Motive"
    UNKNOWN = "Unknown"


@dataclass
class ClassificationResult:
    game_type: GameType
    confidence: int
    reasoning: str
    fingerprint: Dict
    subcategory: Optional[str] = None


# ============================================================
# MATRIX NORMALIZATION
# ============================================================

def normalize_matrix(matrix: Dict) -> Dict:
    """Normalize matrix keys to A_A, A_B, B_A, B_B format."""
    if 'A_A' in matrix:
        return matrix

    keys = list(matrix.keys())
    if len(keys) != 4:
        return None

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

    opt_a_first = unique_first[0]
    opt_a_second = unique_second[0]

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


# ============================================================
# SYMMETRIC 2x2 GAMES
# ============================================================

def extract_fingerprint(matrix: Dict) -> Dict:
    """Extract structural fingerprint from 2x2 payoff matrix."""
    matrix = normalize_matrix(matrix)
    if not matrix:
        return None

    CC = matrix['A_A'][0]
    CD = matrix['A_B'][0]
    DC = matrix['B_A'][0]
    DD = matrix['B_B'][0]

    # Standard notation: T=temptation, R=reward, P=punishment, S=sucker
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
        'social_dilemma': (T > R > P > S),
    }


def classify_2x2_symmetric(fingerprint: Dict) -> ClassificationResult:
    """Classify 2x2 symmetric games using fingerprint patterns."""
    T, R, P, S = fingerprint['T'], fingerprint['R'], fingerprint['P'], fingerprint['S']
    CC, CD, DC, DD = fingerprint['CC'], fingerprint['CD'], fingerprint['DC'], fingerprint['DD']

    # Check for MATCHING PENNIES / anti-coordination pattern first
    # Pattern: diagonal outcomes equal, off-diagonal outcomes equal, but different from diagonal
    # This detects zero-sum-like games even when marked as symmetric
    if CC == DD and CD == DC and CC != CD:
        return ClassificationResult(
            game_type=GameType.MATCHING_PENNIES,
            confidence=85,
            reasoning=f"Anti-coordination: diagonal={CC}, off-diagonal={CD}",
            fingerprint=fingerprint,
            subcategory="zero_sum"
        )

    # COORDINATION (R ≈ P >> T, S) - check first
    if (R > T and R > S and P > T and P > S and abs(R - P) <= 1):
        return ClassificationResult(
            game_type=GameType.COORDINATION,
            confidence=95,
            reasoning=f"R({R}) ≈ P({P}) >> T({T}), S({S}): Pure coordination",
            fingerprint=fingerprint
        )

    # BATTLE OF SEXES (R ≠ P, both >> T, S)
    if (R > T and R > S and P > T and P > S and R != P):
        return ClassificationResult(
            game_type=GameType.BATTLE_OF_SEXES,
            confidence=90,
            reasoning=f"R({R}) ≠ P({P}), both >> T({T}), S({S}): Coordination with preference conflict",
            fingerprint=fingerprint
        )

    # PRISONER'S DILEMMA: T > R > P > S
    if fingerprint['social_dilemma']:
        return ClassificationResult(
            game_type=GameType.PRISONERS_DILEMMA,
            confidence=95,
            reasoning=f"T({T}) > R({R}) > P({P}) > S({S}): Defect dominates, social dilemma",
            fingerprint=fingerprint
        )

    # CHICKEN: T > R > S > P (mutual defect = worst)
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

    # DEADLOCK: T > P > R > S (defect dominates, no dilemma)
    if (fingerprint['defect_dominates'] and P > R):
        return ClassificationResult(
            game_type=GameType.DEADLOCK,
            confidence=90,
            reasoning=f"T({T}) > P({P}) > R({R}) > S({S}): Defect dominates, no dilemma",
            fingerprint=fingerprint
        )

    # HARMONY: R > T > S > P (cooperate dominates)
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
        reasoning=f"T={T}, R={R}, P={P}, S={S}: No standard symmetric pattern",
        fingerprint=fingerprint
    )


# ============================================================
# SEQUENTIAL GAMES
# ============================================================

def classify_sequential(case: Dict) -> ClassificationResult:
    """Classify sequential (extensive form) games."""

    title = case.get('title', '').lower()
    notes = case.get('notes', {})
    flags = notes.get('flags', [])
    assumptions = notes.get('assumptions', [])
    all_text = f"{title} {' '.join(flags)} {' '.join(assumptions)}".lower()

    # Check for specific sequential game patterns

    # ULTIMATUM: Proposer/Responder, accept/reject, fairness
    if any(kw in all_text for kw in ['ultimatum', 'proposer', 'responder', 'accept or reject', 'take it or leave']):
        return ClassificationResult(
            game_type=GameType.ULTIMATUM,
            confidence=85,
            reasoning="Proposer-Responder structure with accept/reject decision",
            fingerprint={},
            subcategory="bargaining"
        )

    # TRUST GAME: Send money, multiply, return
    if any(kw in all_text for kw in ['trust game', 'send money', 'multiplied', 'trustor', 'trustee']):
        return ClassificationResult(
            game_type=GameType.TRUST_GAME,
            confidence=85,
            reasoning="Investment with multiplication and voluntary return",
            fingerprint={},
            subcategory="trust"
        )

    # STACKELBERG: Leader/follower, quantity/price competition
    if any(kw in all_text for kw in ['stackelberg', 'leader', 'follower', 'first mover', 'quantity leader']):
        return ClassificationResult(
            game_type=GameType.STACKELBERG,
            confidence=85,
            reasoning="Leader-follower sequential competition",
            fingerprint={},
            subcategory="competition"
        )

    # ENTRY DETERRENCE: Incumbent/entrant, enter/stay out, fight/accommodate
    if any(kw in all_text for kw in ['entry', 'entrant', 'incumbent', 'deter', 'market entry', 'fight or accommodate']):
        return ClassificationResult(
            game_type=GameType.ENTRY_DETERRENCE,
            confidence=85,
            reasoning="Market entry with deterrence option",
            fingerprint={},
            subcategory="competition"
        )

    # CENTIPEDE: Pass/take, growing pot, finite rounds
    if any(kw in all_text for kw in ['centipede', 'pass or take', 'growing pot', 'continue or stop']):
        return ClassificationResult(
            game_type=GameType.CENTIPEDE,
            confidence=85,
            reasoning="Sequential pass-or-take with growing stakes",
            fingerprint={},
            subcategory="cooperation"
        )

    # SEQUENTIAL BARGAINING: Alternating offers
    if any(kw in all_text for kw in ['alternating offer', 'bargaining', 'counteroffer', 'rubinstein']):
        return ClassificationResult(
            game_type=GameType.SEQUENTIAL_BARGAINING,
            confidence=80,
            reasoning="Alternating offer bargaining structure",
            fingerprint={},
            subcategory="bargaining"
        )

    # PRINCIPAL-AGENT (can be sequential): monitoring, moral hazard, effort
    if any(kw in all_text for kw in ['principal', 'agent', 'moral hazard', 'moral-hazard', 'monitoring', 'effort', 'shirk']):
        return ClassificationResult(
            game_type=GameType.PRINCIPAL_AGENT,
            confidence=85,
            reasoning="Principal-agent structure with monitoring/effort decision",
            fingerprint={},
            subcategory="contract"
        )

    # ULTIMATUM-like: offer + accept/reject pattern
    if ('offer' in all_text and any(kw in all_text for kw in ['reject', 'accept', 'walk away', 'refuse'])):
        return ClassificationResult(
            game_type=GameType.ULTIMATUM,
            confidence=80,
            reasoning="Take-it-or-leave-it offer structure",
            fingerprint={},
            subcategory="bargaining"
        )

    # Generic sequential - try to classify by payoff structure if available
    matrix = case.get('payoffs', {}).get('matrix', {})
    normalized = normalize_matrix(matrix) if matrix else None

    if normalized:
        fingerprint = extract_fingerprint(matrix)
        if fingerprint:
            base_result = classify_2x2_symmetric(fingerprint)
            return ClassificationResult(
                game_type=base_result.game_type,
                confidence=base_result.confidence - 10,
                reasoning=f"Sequential variant of {base_result.game_type.value}: {base_result.reasoning}",
                fingerprint=fingerprint,
                subcategory="sequential"
            )

    return ClassificationResult(
        game_type=GameType.UNKNOWN,
        confidence=40,
        reasoning="Sequential game - insufficient pattern match",
        fingerprint={},
        subcategory="sequential"
    )


# ============================================================
# ASYMMETRIC GAMES
# ============================================================

def check_asymmetry(matrix: Dict) -> Tuple[bool, str]:
    """Check if game is asymmetric and characterize the asymmetry."""
    matrix = normalize_matrix(matrix)
    if not matrix:
        return True, "unknown"

    # Check if P1 and P2 payoffs are symmetric
    sym_check_1 = (matrix['A_B'][0] == matrix['B_A'][1])
    sym_check_2 = (matrix['B_A'][0] == matrix['A_B'][1])

    is_symmetric = sym_check_1 and sym_check_2

    if is_symmetric:
        return False, "symmetric"

    # Zero-sum check
    sums = [
        matrix['A_A'][0] + matrix['A_A'][1],
        matrix['A_B'][0] + matrix['A_B'][1],
        matrix['B_A'][0] + matrix['B_A'][1],
        matrix['B_B'][0] + matrix['B_B'][1]
    ]
    if len(set(sums)) == 1:
        return True, "zero_sum"

    # Constant sum check
    if max(sums) - min(sums) <= 1:
        return True, "constant_sum"

    return True, "general_asymmetric"


def classify_asymmetric(case: Dict) -> ClassificationResult:
    """Classify asymmetric games."""

    title = case.get('title', '').lower()
    notes = case.get('notes', {})
    flags = notes.get('flags', [])
    all_text = f"{title} {' '.join(flags)}".lower()

    matrix = case.get('payoffs', {}).get('matrix', {})

    if matrix:
        normalized = normalize_matrix(matrix)
        if normalized:
            is_asym, asym_type = check_asymmetry(matrix)
        else:
            is_asym, asym_type = True, "unknown"
    else:
        is_asym, asym_type = True, "unknown"

    # MATCHING PENNIES: Zero-sum, pure conflict
    if asym_type == "zero_sum" or any(kw in all_text for kw in ['matching pennies', 'zero sum', 'zero-sum', 'pure conflict', 'pure opposition', 'opposite goal']):
        return ClassificationResult(
            game_type=GameType.MATCHING_PENNIES,
            confidence=90,
            reasoning=f"Zero-sum game: Pure conflict of interest",
            fingerprint={'asymmetry_type': asym_type},
            subcategory="zero_sum"
        )

    # PRINCIPAL-AGENT: Employer/employee, owner/manager, incentive
    if any(kw in all_text for kw in ['principal', 'agent', 'employer', 'employee', 'manager', 'incentive', 'moral hazard']):
        return ClassificationResult(
            game_type=GameType.PRINCIPAL_AGENT,
            confidence=85,
            reasoning="Principal-agent structure with incentive alignment problem",
            fingerprint={'asymmetry_type': asym_type},
            subcategory="contract"
        )

    # INSPECTION GAME: Inspector/inspected, audit, compliance
    if any(kw in all_text for kw in ['inspection', 'inspector', 'audit', 'compliance', 'monitor', 'enforcement']):
        return ClassificationResult(
            game_type=GameType.INSPECTION_GAME,
            confidence=85,
            reasoning="Inspection/monitoring structure",
            fingerprint={'asymmetry_type': asym_type},
            subcategory="monitoring"
        )

    # HAWK-DOVE ASYMMETRIC: Territory holder vs intruder
    if any(kw in all_text for kw in ['territory', 'intruder', 'owner', 'challenger', 'incumbent']):
        return ClassificationResult(
            game_type=GameType.HAWK_DOVE_ASYMMETRIC,
            confidence=80,
            reasoning="Asymmetric contest (owner vs challenger)",
            fingerprint={'asymmetry_type': asym_type},
            subcategory="contest"
        )

    # Check for Battle of Sexes pattern in asymmetric games
    if matrix:
        normalized = normalize_matrix(matrix)
        if normalized:
            aa = normalized['A_A']
            bb = normalized['B_B']
            ab = normalized['A_B']
            ba = normalized['B_A']

            coord_good_p1 = (aa[0] > ab[0] and aa[0] > ba[0] and bb[0] > ab[0] and bb[0] > ba[0])
            coord_good_p2 = (aa[1] > ab[1] and aa[1] > ba[1] and bb[1] > ab[1] and bb[1] > ba[1])
            p1_prefers_different = aa[0] != bb[0]
            p2_prefers_different = aa[1] != bb[1]

            if coord_good_p1 and coord_good_p2 and (p1_prefers_different or p2_prefers_different):
                return ClassificationResult(
                    game_type=GameType.BATTLE_OF_SEXES,
                    confidence=90,
                    reasoning=f"Asymmetric coordination: both prefer matching, different favorites",
                    fingerprint={'asymmetry_type': asym_type},
                    subcategory="coordination"
                )

            # Try symmetric classification as baseline
            fingerprint = extract_fingerprint(matrix)
            if fingerprint:
                base_result = classify_2x2_symmetric(fingerprint)
                if base_result.game_type != GameType.UNKNOWN:
                    return ClassificationResult(
                        game_type=base_result.game_type,
                        confidence=base_result.confidence - 15,
                        reasoning=f"Asymmetric variant of {base_result.game_type.value}",
                        fingerprint=fingerprint,
                        subcategory="asymmetric"
                    )

    return ClassificationResult(
        game_type=GameType.MIXED_MOTIVE,
        confidence=60,
        reasoning=f"Asymmetric game ({asym_type}) - no specific pattern matched",
        fingerprint={'asymmetry_type': asym_type},
        subcategory="asymmetric"
    )


# ============================================================
# MAIN CLASSIFIER
# ============================================================

def classify_case(case: Dict) -> ClassificationResult:
    """Main classification entry point - routes to appropriate handler."""

    notes = case.get('notes', {})
    flags = notes.get('flags', [])
    info = case.get('information', {})
    symmetry = case.get('symmetry', {})

    # Check flags first
    if 'incomplete-info' in flags:
        return ClassificationResult(
            game_type=GameType.INCOMPLETE_INFO,
            confidence=30,
            reasoning="Incomplete information game - requires Bayesian analysis",
            fingerprint={}
        )

    if 'n-player' in flags:
        return ClassificationResult(
            game_type=GameType.N_PLAYER,
            confidence=30,
            reasoning="N-player game - beyond 2-player scope",
            fingerprint={}
        )

    # Route by game structure
    is_sequential = info.get('timing') == 'sequential'
    is_asymmetric = symmetry.get('symmetric') == False

    # Sequential games
    if is_sequential:
        return classify_sequential(case)

    # Asymmetric games
    if is_asymmetric:
        return classify_asymmetric(case)

    # Symmetric simultaneous games (default path)
    matrix = case.get('payoffs', {}).get('matrix', {})
    if not matrix:
        return ClassificationResult(
            game_type=GameType.UNKNOWN,
            confidence=0,
            reasoning="No valid payoff matrix provided",
            fingerprint={}
        )

    normalized = normalize_matrix(matrix)
    if not normalized:
        return ClassificationResult(
            game_type=GameType.UNKNOWN,
            confidence=0,
            reasoning="Could not normalize payoff matrix",
            fingerprint={}
        )

    fingerprint = extract_fingerprint(matrix)
    if not fingerprint:
        return ClassificationResult(
            game_type=GameType.UNKNOWN,
            confidence=0,
            reasoning="Could not extract fingerprint",
            fingerprint={}
        )

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
        'reasoning': result.reasoning,
        'subcategory': result.subcategory
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
        print("GTI Classifier v2.0")
        print("Usage: python gti_classifier.py <case.yaml>")
        print("")
        print("Supported game types:")
        for gt in GameType:
            print(f"  - {gt.value}")
