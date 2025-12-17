#!/usr/bin/env python3
"""
build_topology_map.py - Enumerate all 144 distinct 2x2 games

This script generates the complete TOPOLOGY_MAP by:
1. Enumerating all 4! x 4! = 576 ordinal payoff combinations
2. Computing canonical forms (reduces to 72-144 equivalence classes)
3. Classifying each into Robinson-Goforth topology
4. Exporting to JSON for O(1) lookup
"""

import itertools
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from canonical_classifier import (
    get_canonical_form, analyze_nash_equilibria, RGClass
)


def enumerate_ordinal_games():
    """
    Generate all possible ordinal 2x2 games.

    Each player has 4 outcomes ranked 0-3 (worst to best).
    Total: 4! x 4! = 576 combinations.
    """
    p1_perms = list(itertools.permutations([0, 1, 2, 3]))
    p2_perms = list(itertools.permutations([0, 1, 2, 3]))

    games = []
    for p1 in p1_perms:
        for p2 in p2_perms:
            # Layout: [[AA, AB], [BA, BB]] with [p1_payoff, p2_payoff]
            matrix = np.array([
                [[p1[0], p2[0]], [p1[1], p2[1]]],
                [[p1[2], p2[2]], [p1[3], p2[3]]]
            ])
            games.append({
                'matrix': matrix,
                'p1_ordinal': p1,
                'p2_ordinal': p2
            })

    return games


def classify_by_structure(nash_info: dict, p1_ordinal: tuple, p2_ordinal: tuple) -> RGClass:
    """
    Classify game based on Nash equilibrium structure and payoff patterns.

    Uses Robinson-Goforth topology rules with refinements for
    distinguishing similar structures.

    Key insight: Structure alone doesn't distinguish all games.
    We need to check payoff patterns (T/R/P/S relationships).
    """
    ne_count = nash_info['pure_ne_count']
    p1_dom = nash_info['p1_dominant_strategy']
    p2_dom = nash_info['p2_dominant_strategy']
    is_sym = nash_info['is_symmetric']
    ne_cells = nash_info['pure_ne_cells']
    pareto_indices = nash_info.get('pareto_optimal_indices', [])

    # For symmetric games, extract T/R/P/S from ordinal
    # Layout: [[AA=R, AB=S], [BA=T, BB=P]] where position maps to outcome
    # p1_ordinal = (rank_AA, rank_AB, rank_BA, rank_BB)
    R, S, T, P = p1_ordinal  # For player 1 in symmetric game

    # CASE 1: No pure Nash equilibrium - cyclic/zero-sum games
    if ne_count == 0:
        # Zero-sum: gains are opposite (one's win is other's loss)
        # Check if ordinal pattern is anti-symmetric
        p2_reversed = tuple(3 - x for x in p2_ordinal)
        if p1_ordinal == p2_reversed or is_matching_pennies_structure(p1_ordinal, p2_ordinal):
            return RGClass.CLASS_8_ZERO_SUM
        return RGClass.CLASS_9_CYCLIC

    # CASE 2: Both players have dominant strategies (single NE)
    if p1_dom is not None and p2_dom is not None:
        # Dominant strategy leads to single NE
        # Distinguish PD, Harmony, Deadlock by Pareto properties

        if is_sym:
            # PD: T > R > P > S - dominant defection, but mutual cooperation Pareto-dominates
            if T > R > P > S:
                return RGClass.CLASS_2_DILEMMA

            # Harmony: R > T > S > P or R > T > P > S with R dominant
            # Cooperation is both dominant AND Pareto-optimal
            if R > T and R == max(p1_ordinal):
                return RGClass.CLASS_7_HARMONY

            # Deadlock: P > S and T > R, defection dominant and efficient
            if T > R and P > S:
                return RGClass.CLASS_6_DEADLOCK

        return RGClass.CLASS_12_DOMINANCE

    # CASE 3: Two pure Nash equilibria
    if ne_count == 2:
        ne_set = set(ne_cells)

        if is_sym:
            # Diagonal NEs (0,0) and (1,1): Coordination type games
            if ne_set == {(0, 0), (1, 1)}:
                # Stag Hunt: R > T > P > S
                # Both equilibria on diagonal, but one payoff-dominates
                if R > T > P > S:
                    return RGClass.CLASS_4_STAG_HUNT
                # Pure Coordination: Both diagonal outcomes equally preferred
                return RGClass.CLASS_1_WIN_WIN

            # Anti-diagonal NEs (0,1) and (1,0): Battle of Sexes
            elif ne_set == {(0, 1), (1, 0)}:
                return RGClass.CLASS_5_BATTLE

            # Other 2-NE patterns
            else:
                # Chicken: T > R > S > P (swerve better than crash)
                if T > R > S > P:
                    return RGClass.CLASS_3_CHICKEN
                return RGClass.CLASS_11_BIASED

        # Asymmetric with 2 NEs
        return RGClass.CLASS_11_BIASED

    # CASE 4: Exactly one pure Nash equilibrium (no dominant strategies)
    if ne_count == 1:
        ne_cell = ne_cells[0]

        if is_sym:
            # Chicken: T > R > S > P, single NE (often mixed in theory)
            # But strict ordinal may give 1 pure NE
            if T > R > S > P:
                return RGClass.CLASS_3_CHICKEN

        return RGClass.CLASS_10_INTERMEDIATE

    # CASE 5: More than 2 NEs (rare in strict ordinal games)
    if ne_count > 2:
        return RGClass.CLASS_1_WIN_WIN

    return RGClass.UNKNOWN


def is_matching_pennies_structure(p1_ord: tuple, p2_ord: tuple) -> bool:
    """Check if game has matching pennies / zero-sum structure."""
    # In matching pennies, diagonal and anti-diagonal outcomes are opposite
    # P1 wants to match, P2 wants to mismatch (or vice versa)
    # Check if P1's best is P2's worst and vice versa
    p1_arr = np.array(p1_ord).reshape(2, 2)
    p2_arr = np.array(p2_ord).reshape(2, 2)

    # Check anti-correlation: where P1 is high, P2 is low
    p1_high = p1_arr >= 2
    p2_high = p2_arr >= 2

    # Perfect anti-correlation
    return np.sum(p1_high == ~p2_high) >= 3


def detect_classic_patterns(p1_ordinal: tuple, p2_ordinal: tuple, is_sym: bool) -> RGClass:
    """
    Detect classic game patterns by ordinal fingerprint.

    Standard ordinal patterns (with 0=worst, 3=best):
    - PD: T>R>P>S means ranks (T=3, R=2, P=1, S=0)
    - Chicken: T>R>S>P means ranks (T=3, R=2, S=1, P=0)
    - Stag Hunt: R>T>P>S means ranks (R=3, T=2, P=1, S=0)
    """
    if not is_sym:
        return None

    # Extract T, R, P, S for symmetric games
    # Layout: [[AA=R, AB=S], [BA=T, BB=P]] for row player
    R, S, T, P = p1_ordinal

    # Prisoner's Dilemma: T > R > P > S (3 > 2 > 1 > 0)
    if T > R > P > S:
        return RGClass.CLASS_2_DILEMMA

    # Chicken/Hawk-Dove: T > R > S > P (3 > 2 > 1 > 0)
    if T > R > S > P:
        return RGClass.CLASS_3_CHICKEN

    # Stag Hunt: R > T > P > S
    if R > T > P > S:
        return RGClass.CLASS_4_STAG_HUNT

    # Pure Coordination: R > T and R > S and P > T and P > S
    if R == max(p1_ordinal) and P == sorted(p1_ordinal)[-2]:
        return RGClass.CLASS_1_WIN_WIN

    # Harmony: R > T > S > P (cooperation dominant and best)
    if R > T > S > P:
        return RGClass.CLASS_7_HARMONY

    return None


def build_map():
    """Build complete topology map."""
    print("=" * 60)
    print("TOPOLOGY MAP BUILDER")
    print("=" * 60)

    print("\n[Step 1] Enumerating all 576 ordinal games...")
    games = enumerate_ordinal_games()
    print(f"  Generated {len(games)} games")

    print("\n[Step 2] Computing canonical forms...")
    canonical_map = defaultdict(list)

    for game in games:
        p1, p2 = get_canonical_form(game['matrix'])
        fingerprint = f"{p1}|{p2}"
        canonical_map[fingerprint].append(game)

    print(f"  Found {len(canonical_map)} unique canonical forms")

    print("\n[Step 3] Classifying each canonical form...")
    topology_map = {}
    class_counts = defaultdict(int)
    unclassified = []

    for fingerprint, examples in canonical_map.items():
        example = examples[0]
        matrix = example['matrix']
        p1_ord = example['p1_ordinal']
        p2_ord = example['p2_ordinal']

        nash = analyze_nash_equilibria(matrix)

        # Try classic pattern detection first
        rg_class = detect_classic_patterns(p1_ord, p2_ord, nash['is_symmetric'])

        # Fall back to structure-based classification
        if rg_class is None:
            rg_class = classify_by_structure(nash, p1_ord, p2_ord)

        if rg_class == RGClass.UNKNOWN:
            unclassified.append({
                'fingerprint': fingerprint,
                'nash': nash,
                'p1_ordinal': p1_ord,
                'p2_ordinal': p2_ord
            })

        # Convert nash info to JSON-serializable format
        nash_serializable = {
            'pure_ne_count': nash['pure_ne_count'],
            'pure_ne_cells': [list(c) for c in nash['pure_ne_cells']],
            'p1_dominant': nash['p1_dominant_strategy'],
            'p2_dominant': nash['p2_dominant_strategy'],
            'is_symmetric': bool(nash['is_symmetric']),
            'pareto_optimal': nash['pareto_optimal_count']
        }

        topology_map[fingerprint] = {
            'class': rg_class.name,
            'class_id': rg_class.value,
            'orbit_size': len(examples),
            'nash_info': nash_serializable,
            'example_p1': list(p1_ord),
            'example_p2': list(p2_ord)
        }
        class_counts[rg_class.name] += 1

    print("\n[Step 4] Class distribution:")
    print("-" * 40)
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        pct = count / len(canonical_map) * 100
        bar = "#" * int(pct / 2)
        print(f"  {cls:25} {count:3} ({pct:5.1f}%) {bar}")
    print("-" * 40)
    print(f"  Total canonical forms: {len(canonical_map)}")

    if unclassified:
        print(f"\n  WARNING: {len(unclassified)} forms unclassified")

    # Export
    print("\n[Step 5] Exporting topology map...")
    output = {
        'metadata': {
            'total_games': 576,
            'canonical_forms': len(canonical_map),
            'classified': len(canonical_map) - len(unclassified),
            'unclassified': len(unclassified)
        },
        'class_distribution': dict(class_counts),
        'topology_map': topology_map
    }

    output_path = Path(__file__).parent.parent / 'topology_map.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Exported to {output_path}")

    # Also export unclassified for analysis
    if unclassified:
        unclassified_path = Path(__file__).parent.parent / 'unclassified_games.json'
        with open(unclassified_path, 'w') as f:
            json.dump(unclassified, f, indent=2, default=str)
        print(f"  Unclassified games exported to {unclassified_path}")

    print("\n" + "=" * 60)
    print("TOPOLOGY MAP BUILD COMPLETE")
    print("=" * 60)

    return topology_map


def classify_by_trps_ordering(matrix: np.ndarray) -> RGClass:
    """
    Classify game directly by T/R/P/S ordering - the 'Game DNA'.

    For symmetric 2x2 games:
    - R (Reward): Mutual cooperation (AA)
    - S (Sucker): Cooperate while other defects (AB for row player)
    - T (Temptation): Defect while other cooperates (BA for row player)
    - P (Punishment): Mutual defection (BB)

    The ordering T/R/P/S determines the game type:
    - Prisoner's Dilemma: T > R > P > S
    - Chicken (Hawk-Dove): T > R > S > P
    - Stag Hunt: R > T > P > S
    - Harmony: R > T > S > P
    - Deadlock: T > P > R > S
    """
    p1 = matrix[:, :, 0]  # Player 1 payoffs
    p2 = matrix[:, :, 1]  # Player 2 payoffs

    # Extract T, R, P, S for player 1
    R = p1[0, 0]  # Mutual cooperation
    S = p1[0, 1]  # Sucker payoff
    T = p1[1, 0]  # Temptation payoff
    P = p1[1, 1]  # Mutual defection

    # Check symmetry
    is_sym = np.allclose(p1, p2.T)

    # Compute Nash equilibria
    nash = analyze_nash_equilibria(matrix)
    ne_count = nash['pure_ne_count']
    ne_cells = nash['pure_ne_cells']

    # No pure NE - zero-sum / cyclic
    if ne_count == 0:
        # Check for matching pennies structure
        if is_matching_pennies_structure(
            tuple(p1.flatten().argsort().argsort()),
            tuple(p2.flatten().argsort().argsort())
        ):
            return RGClass.CLASS_8_ZERO_SUM
        return RGClass.CLASS_9_CYCLIC

    # Classification by T/R/P/S ordering (Game DNA)
    if is_sym:
        # Prisoner's Dilemma: T > R > P > S
        if T > R > P > S:
            return RGClass.CLASS_2_DILEMMA

        # Chicken / Hawk-Dove: T > R > S > P (crash is worst)
        if T > R > S > P:
            return RGClass.CLASS_3_CHICKEN

        # Stag Hunt: R > T > P > S (cooperation best but risky)
        if R > T > P > S:
            return RGClass.CLASS_4_STAG_HUNT

        # Harmony: R > T > S > P (cooperation dominant and best)
        if R > T > S > P:
            return RGClass.CLASS_7_HARMONY

        # Deadlock: T > P > R > S (defection dominant and efficient)
        if T > P > R > S:
            return RGClass.CLASS_6_DEADLOCK

        # Pure Coordination: R = P > T = S (only care about matching)
        if R == P and T == S and R > T:
            return RGClass.CLASS_1_WIN_WIN

    # Two NEs on diagonal - coordination games
    if ne_count == 2 and set(ne_cells) == {(0, 0), (1, 1)}:
        # Battle of Sexes: Players prefer DIFFERENT equilibria
        # P1 prefers (0,0), P2 prefers (1,1) or vice versa
        p1_pref_00 = p1[0, 0] > p1[1, 1]  # P1 prefers AA over BB
        p2_pref_00 = p2[0, 0] > p2[1, 1]  # P2 prefers AA over BB

        if p1_pref_00 != p2_pref_00:
            # Different preferences = Battle of Sexes
            return RGClass.CLASS_5_BATTLE

        # Same preferences = Stag Hunt or coordination
        if is_sym:
            return RGClass.CLASS_4_STAG_HUNT
        return RGClass.CLASS_11_BIASED

    # Anti-diagonal NEs (0,1) and (1,0) - anti-coordination
    if ne_count == 2 and set(ne_cells) == {(0, 1), (1, 0)}:
        return RGClass.CLASS_11_BIASED

    # Fallback classifications
    if nash['p1_dominant_strategy'] is not None and nash['p2_dominant_strategy'] is not None:
        return RGClass.CLASS_12_DOMINANCE

    if ne_count == 2:
        return RGClass.CLASS_11_BIASED

    return RGClass.CLASS_10_INTERMEDIATE


def verify_classic_games():
    """Verify that classic games are correctly classified using direct T/R/P/S analysis."""
    print("\n" + "=" * 60)
    print("CLASSIC GAME VERIFICATION (Direct T/R/P/S Classification)")
    print("=" * 60)

    # Classic games with known classifications
    # Matrix format: [[[R,R], [S,T]], [[T,S], [P,P]]] for symmetric games
    classic_tests = [
        {
            'name': "Prisoner's Dilemma (T>R>P>S)",
            'matrix': np.array([[[3, 3], [0, 5]], [[5, 0], [1, 1]]]),  # T=5,R=3,P=1,S=0
            'expected': 'CLASS_2_DILEMMA'
        },
        {
            'name': "Chicken (T>R>S>P)",
            'matrix': np.array([[[3, 3], [1, 4]], [[4, 1], [0, 0]]]),  # T=4,R=3,S=1,P=0
            'expected': 'CLASS_3_CHICKEN'
        },
        {
            'name': "Stag Hunt (R>T>P>S)",
            'matrix': np.array([[[4, 4], [0, 3]], [[3, 0], [1, 1]]]),  # R=4,T=3,P=1,S=0
            'expected': 'CLASS_4_STAG_HUNT'
        },
        {
            'name': "Harmony (R>T>S>P)",
            'matrix': np.array([[[4, 4], [2, 3]], [[3, 2], [1, 1]]]),  # R=4,T=3,S=2,P=1
            'expected': 'CLASS_7_HARMONY'
        },
        {
            'name': "Battle of Sexes",
            'matrix': np.array([[[3, 2], [0, 0]], [[0, 0], [2, 3]]]),
            'expected': 'CLASS_5_BATTLE'
        },
        {
            'name': "Matching Pennies (Zero-Sum)",
            'matrix': np.array([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]]),
            'expected': 'CLASS_8_ZERO_SUM'
        },
        {
            'name': "Deadlock (T>P>R>S)",
            'matrix': np.array([[[2, 2], [0, 4]], [[4, 0], [3, 3]]]),  # T=4,P=3,R=2,S=0
            'expected': 'CLASS_6_DEADLOCK'
        }
    ]

    print("\nTest Results:")
    print("-" * 60)

    passed = 0
    for test in classic_tests:
        actual_class = classify_by_trps_ordering(test['matrix'])
        actual = actual_class.name

        status = "PASS" if actual == test['expected'] else "FAIL"
        if status == "PASS":
            passed += 1

        # Extract T/R/P/S for display
        p1 = test['matrix'][:, :, 0]
        R, S, T, P = p1[0,0], p1[0,1], p1[1,0], p1[1,1]

        print(f"  {status}: {test['name']}")
        print(f"         T={T}, R={R}, P={P}, S={S}")
        print(f"         Expected: {test['expected']}")
        print(f"         Actual:   {actual}")
        print()

    print("-" * 60)
    print(f"Results: {passed}/{len(classic_tests)} tests passed")

    return passed == len(classic_tests)


if __name__ == "__main__":
    topology_map = build_map()
    verify_classic_games()
