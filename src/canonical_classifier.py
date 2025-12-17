#!/usr/bin/env python3
"""
canonical_classifier.py - Group-Theoretic 2x2 Game Classification

Mathematical Foundation:
- 2x2 games have 4! x 4! = 576 possible payoff orderings
- D4 symmetry group (8 operations) reduces to 72 strategic equivalence classes
- Robinson-Goforth further collapses to 12 topological classes

The 8 D4 symmetries:
1. Identity (e)
2. Swap Rows (r) - player 1 strategy relabeling
3. Swap Columns (c) - player 2 strategy relabeling
4. Swap Both (rc)
5. Transpose (t) - swap players
6. Transpose + Swap Rows (tr)
7. Transpose + Swap Columns (tc)
8. Transpose + Swap Both (trc)
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


class RGClass(Enum):
    """Robinson-Goforth 12-class topology"""
    # Symmetric Classes (both players have same structure)
    CLASS_1_WIN_WIN = 1        # Coordination (pure)
    CLASS_2_DILEMMA = 2        # Prisoner's Dilemma
    CLASS_3_CHICKEN = 3        # Chicken/Hawk-Dove
    CLASS_4_STAG_HUNT = 4      # Stag Hunt (Assurance)
    CLASS_5_BATTLE = 5         # Battle of Sexes
    CLASS_6_DEADLOCK = 6       # Deadlock
    CLASS_7_HARMONY = 7        # Harmony Game
    CLASS_8_ZERO_SUM = 8       # Zero-sum (Matching Pennies)
    # Asymmetric Classes (players have different structures)
    CLASS_9_CYCLIC = 9         # No pure NE, cyclic dominance
    CLASS_10_INTERMEDIATE = 10 # One pure NE, asymmetric dominance
    CLASS_11_BIASED = 11       # Two pure NEs, unequal preferences
    CLASS_12_DOMINANCE = 12    # One player has dominant strategy
    UNKNOWN = 0


@dataclass
class GameMatrix:
    """2x2 game in normal form"""
    # Payoffs: matrix[row][col] = (player1_payoff, player2_payoff)
    AA: Tuple[float, float]  # Both choose A
    AB: Tuple[float, float]  # P1 chooses A, P2 chooses B
    BA: Tuple[float, float]  # P1 chooses B, P2 chooses A
    BB: Tuple[float, float]  # Both choose B

    def to_array(self) -> np.ndarray:
        """Convert to 2x2x2 numpy array [row, col, player]"""
        return np.array([
            [[self.AA[0], self.AA[1]], [self.AB[0], self.AB[1]]],
            [[self.BA[0], self.BA[1]], [self.BB[0], self.BB[1]]]
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'GameMatrix':
        """Create from numpy array"""
        return cls(
            AA=(float(arr[0, 0, 0]), float(arr[0, 0, 1])),
            AB=(float(arr[0, 1, 0]), float(arr[0, 1, 1])),
            BA=(float(arr[1, 0, 0]), float(arr[1, 0, 1])),
            BB=(float(arr[1, 1, 0]), float(arr[1, 1, 1]))
        )


def to_ordinal(payoffs: np.ndarray) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Convert cardinal payoffs to ordinal ranks (0,1,2,3) for each player.
    Handles ties by assigning equal ranks.

    Returns: (player1_ranks, player2_ranks) as flat tuples
    """
    def rank_payoffs(p: np.ndarray) -> Tuple[int, ...]:
        flat = p.flatten()
        # argsort twice gives ranks
        order = flat.argsort().argsort()
        return tuple(order)

    p1 = payoffs[:, :, 0]  # Player 1 payoffs
    p2 = payoffs[:, :, 1]  # Player 2 payoffs

    return rank_payoffs(p1), rank_payoffs(p2)


def apply_symmetry(p1_ranks: Tuple, p2_ranks: Tuple, sym: str) -> Tuple[Tuple, Tuple]:
    """
    Apply one of 8 D4 symmetries to ordinal game representation.

    Symmetries:
    - 'e': identity
    - 'r': swap rows (P1 strategy relabel)
    - 'c': swap columns (P2 strategy relabel)
    - 'rc': swap both
    - 't': transpose (swap players)
    - 'tr': transpose + swap rows
    - 'tc': transpose + swap columns
    - 'trc': transpose + swap both
    """
    # Convert tuples to 2x2 for manipulation
    m1 = np.array(p1_ranks).reshape(2, 2)
    m2 = np.array(p2_ranks).reshape(2, 2)

    if sym == 'e':
        pass
    elif sym == 'r':
        m1 = m1[::-1, :]
        m2 = m2[::-1, :]
    elif sym == 'c':
        m1 = m1[:, ::-1]
        m2 = m2[:, ::-1]
    elif sym == 'rc':
        m1 = m1[::-1, ::-1]
        m2 = m2[::-1, ::-1]
    elif sym == 't':
        m1, m2 = m2.T, m1.T
    elif sym == 'tr':
        m1, m2 = m2.T[::-1, :], m1.T[::-1, :]
    elif sym == 'tc':
        m1, m2 = m2.T[:, ::-1], m1.T[:, ::-1]
    elif sym == 'trc':
        m1, m2 = m2.T[::-1, ::-1], m1.T[::-1, ::-1]

    return tuple(m1.flatten()), tuple(m2.flatten())


def get_canonical_form(payoffs: np.ndarray) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute canonical form using lexicographic minimum over D4 orbit.

    This is the mathematically rigorous way to identify strategic equivalence.
    """
    p1_ranks, p2_ranks = to_ordinal(payoffs)

    symmetries = ['e', 'r', 'c', 'rc', 't', 'tr', 'tc', 'trc']

    candidates = []
    for sym in symmetries:
        transformed = apply_symmetry(p1_ranks, p2_ranks, sym)
        candidates.append(transformed)

    # Canonical form = lexicographically minimal
    return min(candidates)


def canonical_fingerprint(payoffs: np.ndarray) -> str:
    """
    Generate unique hash fingerprint for canonical form.
    Two games with same fingerprint are strategically equivalent.
    """
    p1, p2 = get_canonical_form(payoffs)
    return f"{p1}|{p2}"


# Topology map will be loaded from JSON if available
_TOPOLOGY_MAP: Optional[Dict[str, Dict]] = None


def load_topology_map() -> Dict[str, Dict]:
    """Load topology map from JSON file."""
    global _TOPOLOGY_MAP
    if _TOPOLOGY_MAP is not None:
        return _TOPOLOGY_MAP

    map_path = Path(__file__).parent.parent / "topology_map.json"
    if map_path.exists():
        with open(map_path, 'r') as f:
            data = json.load(f)
            _TOPOLOGY_MAP = data.get('topology_map', {})
    else:
        _TOPOLOGY_MAP = {}

    return _TOPOLOGY_MAP


def classify_game(payoffs: np.ndarray) -> RGClass:
    """
    Classify a 2x2 game into Robinson-Goforth topology class.
    O(1) lookup after canonical form computation.
    """
    fingerprint = canonical_fingerprint(payoffs)
    topology_map = load_topology_map()

    if fingerprint in topology_map:
        class_name = topology_map[fingerprint].get('class', 'UNKNOWN')
        try:
            return RGClass[class_name]
        except KeyError:
            return RGClass.UNKNOWN

    # Fallback: classify by structure
    nash = analyze_nash_equilibria(payoffs)
    return classify_by_nash_structure(nash)


def analyze_nash_equilibria(payoffs: np.ndarray) -> Dict:
    """
    Compute Nash equilibria for classification validation.
    """
    p1 = payoffs[:, :, 0]
    p2 = payoffs[:, :, 1]

    pure_ne = []

    # Check each cell for NE
    for i in range(2):
        for j in range(2):
            # P1 best response check (comparing rows)
            p1_br = p1[i, j] >= p1[1-i, j]
            # P2 best response check (comparing cols)
            p2_br = p2[i, j] >= p2[i, 1-j]

            if p1_br and p2_br:
                pure_ne.append((i, j))

    # Dominance check
    p1_dom = None
    if all(p1[0, :] > p1[1, :]):
        p1_dom = 0
    elif all(p1[1, :] > p1[0, :]):
        p1_dom = 1

    p2_dom = None
    if all(p2[:, 0] > p2[:, 1]):
        p2_dom = 0
    elif all(p2[:, 1] > p2[:, 0]):
        p2_dom = 1

    # Check symmetry
    is_symmetric = np.allclose(p1, p2.T)

    # Pareto analysis
    all_outcomes = [(p1[i, j], p2[i, j]) for i in range(2) for j in range(2)]
    pareto_optimal = []
    for idx, (u1, u2) in enumerate(all_outcomes):
        dominated = False
        for other_idx, (v1, v2) in enumerate(all_outcomes):
            if other_idx != idx and v1 >= u1 and v2 >= u2 and (v1 > u1 or v2 > u2):
                dominated = True
                break
        if not dominated:
            pareto_optimal.append(idx)

    return {
        'pure_ne_count': len(pure_ne),
        'pure_ne_cells': pure_ne,
        'p1_dominant_strategy': p1_dom,
        'p2_dominant_strategy': p2_dom,
        'is_symmetric': is_symmetric,
        'pareto_optimal_count': len(pareto_optimal),
        'pareto_optimal_indices': pareto_optimal
    }


def classify_by_nash_structure(nash_info: Dict) -> RGClass:
    """
    Classify game based on Nash equilibrium structure.
    Fallback when topology map doesn't have the fingerprint.
    """
    ne_count = nash_info['pure_ne_count']
    p1_dom = nash_info['p1_dominant_strategy']
    p2_dom = nash_info['p2_dominant_strategy']
    is_sym = nash_info['is_symmetric']
    ne_cells = nash_info['pure_ne_cells']

    # No pure NE - cyclic/zero-sum
    if ne_count == 0:
        return RGClass.CLASS_9_CYCLIC

    # Both have dominant strategies
    if p1_dom is not None and p2_dom is not None:
        if ne_count == 1:
            if is_sym:
                # Dominant-strategy equilibrium on anti-diagonal suggests PD
                if ne_cells[0] == (1, 1) or ne_cells[0] == (0, 0):
                    # Check if it's suboptimal (PD) or optimal (Harmony)
                    return RGClass.CLASS_2_DILEMMA
                return RGClass.CLASS_7_HARMONY
            return RGClass.CLASS_12_DOMINANCE

    # Two pure NE
    if ne_count == 2:
        if is_sym:
            # Check coordination vs anti-coordination
            if (0, 0) in ne_cells and (1, 1) in ne_cells:
                return RGClass.CLASS_4_STAG_HUNT
            elif (0, 1) in ne_cells and (1, 0) in ne_cells:
                return RGClass.CLASS_5_BATTLE
            else:
                return RGClass.CLASS_3_CHICKEN
        return RGClass.CLASS_11_BIASED

    # One pure NE
    if ne_count == 1:
        if p1_dom is not None or p2_dom is not None:
            return RGClass.CLASS_12_DOMINANCE
        return RGClass.CLASS_10_INTERMEDIATE

    return RGClass.UNKNOWN


def get_game_type_name(rg_class: RGClass) -> str:
    """Map RG class to common game name."""
    mapping = {
        RGClass.CLASS_1_WIN_WIN: "Coordination",
        RGClass.CLASS_2_DILEMMA: "Prisoners_Dilemma",
        RGClass.CLASS_3_CHICKEN: "Chicken",
        RGClass.CLASS_4_STAG_HUNT: "Stag_Hunt",
        RGClass.CLASS_5_BATTLE: "Battle_of_Sexes",
        RGClass.CLASS_6_DEADLOCK: "Deadlock",
        RGClass.CLASS_7_HARMONY: "Harmony",
        RGClass.CLASS_8_ZERO_SUM: "Matching_Pennies",
        RGClass.CLASS_9_CYCLIC: "Cyclic",
        RGClass.CLASS_10_INTERMEDIATE: "Intermediate",
        RGClass.CLASS_11_BIASED: "Biased_Coordination",
        RGClass.CLASS_12_DOMINANCE: "Dominance_Solvable",
        RGClass.UNKNOWN: "Unknown"
    }
    return mapping.get(rg_class, "Unknown")


# ============================================================================
# TEST SUITE
# ============================================================================

def test_canonical_classifier():
    """Test the canonical classifier with known games."""
    print("=" * 60)
    print("CANONICAL CLASSIFIER TEST SUITE")
    print("=" * 60)

    # Prisoner's Dilemma: T=5, R=3, P=1, S=0
    # Format: [row, col, player]
    pd = np.array([
        [[3, 3], [0, 5]],  # (R,R), (S,T)
        [[5, 0], [1, 1]]   # (T,S), (P,P)
    ])

    # Chicken: T=4, R=3, S=2, P=1
    chicken = np.array([
        [[3, 3], [2, 4]],  # (R,R), (S,T)
        [[4, 2], [1, 1]]   # (T,S), (P,P)
    ])

    # Stag Hunt: R=4, T=3, P=2, S=1
    stag = np.array([
        [[4, 4], [1, 3]],  # (R,R), (S,T)
        [[3, 1], [2, 2]]   # (T,S), (P,P)
    ])

    # Battle of Sexes
    bos = np.array([
        [[3, 2], [0, 0]],
        [[0, 0], [2, 3]]
    ])

    # Matching Pennies (zero-sum)
    mp = np.array([
        [[1, -1], [-1, 1]],
        [[-1, 1], [1, -1]]
    ])

    test_games = [
        ("Prisoner's Dilemma", pd),
        ("Chicken", chicken),
        ("Stag Hunt", stag),
        ("Battle of Sexes", bos),
        ("Matching Pennies", mp)
    ]

    for name, game in test_games:
        print(f"\n{name}:")
        print(f"  Canonical: {canonical_fingerprint(game)}")
        rg_class = classify_game(game)
        print(f"  RG Class: {rg_class.name}")
        print(f"  Game Type: {get_game_type_name(rg_class)}")
        nash = analyze_nash_equilibria(game)
        print(f"  Nash Info: {nash['pure_ne_count']} pure NE, symmetric={nash['is_symmetric']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_canonical_classifier()
