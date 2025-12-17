"""
Robinson-Goforth 2x2 Game Topology Classifier

Based on Robinson & Goforth (2005) "The Topology of 2x2 Games"
and the games2p2k dataset from Zhu, Peterson, Enke & Griffiths (2024)

This module classifies 2x2 matrix games into the 12 Robinson-Goforth
ordinal types based on payoff structure.
"""

from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import json


@dataclass
class GameMatrix:
    """
    2x2 Game Matrix representation
    
    Row player chooses A or B
    Column player chooses C or D
    
    Payoffs stored as:
        Row player:    (A,C)=a, (A,D)=b, (B,C)=c, (B,D)=d
        Column player: (A,C)=x, (A,D)=y, (B,C)=z, (B,D)=w
    """
    a: float  # Row player: (A,C)
    b: float  # Row player: (A,D)
    c: float  # Row player: (B,C)
    d: float  # Row player: (B,D)
    x: float  # Col player: (A,C)
    y: float  # Col player: (A,D)
    z: float  # Col player: (B,C)
    w: float  # Col player: (B,D)
    
    @classmethod
    def from_vector(cls, v: List[float]) -> 'GameMatrix':
        """Create from 8-element vector [a,b,c,d,x,y,z,w]"""
        if len(v) != 8:
            raise ValueError(f"Expected 8 elements, got {len(v)}")
        return cls(a=v[0], b=v[1], c=v[2], d=v[3], 
                   x=v[4], y=v[5], z=v[6], w=v[7])
    
    def to_vector(self) -> List[float]:
        return [self.a, self.b, self.c, self.d, self.x, self.y, self.z, self.w]
    
    def row_payoffs(self) -> Tuple[float, float, float, float]:
        """Return (a, b, c, d) - row player's payoffs"""
        return (self.a, self.b, self.c, self.d)
    
    def col_payoffs(self) -> Tuple[float, float, float, float]:
        """Return (x, y, z, w) - column player's payoffs"""
        return (self.x, self.y, self.z, self.w)
    
    def to_bimatrix_str(self) -> str:
        """Pretty print as bimatrix"""
        return f"""
        |   C       |   D       |
    ----|-----------|-----------|
    A   | {self.a:3g},{self.x:3g}  | {self.b:3g},{self.y:3g}  |
    B   | {self.c:3g},{self.z:3g}  | {self.d:3g},{self.w:3g}  |
    """


# Robinson-Goforth order graphs for row player
# Each type is defined by the ordinal ranking of payoffs a, b, c, d
# where a=(A,C), b=(A,D), c=(B,C), d=(B,D)

ROBINSON_GOFORTH_TYPES = {
    # Format: (ranking) -> type_name
    # Ranking is tuple of positions when sorted descending
    # e.g., if c > a > b > d, then ranking positions are c=0, a=1, b=2, d=3
    
    # The 12 ordinal types based on payoff ordering
    "Chicken":      "c > a > b > d",  # Hawk-Dove, Snowdrift
    "Leader":       "c > b > a > d",  # Battle of the Sexes
    "Hero":         "c > b > d > a",
    "Compromise":   "c > d > b > a",
    "Deadlock":     "c > d > a > b",  # Altruist's Dilemma
    "Dilemma":      "c > a > d > b",  # Prisoner's Dilemma
    "Staghunt":     "a > c > d > b",  # Trust
    "Assurance":    "a > d > c > b",
    "Safecoord":    "a > d > b > c",  # Safe Coordination
    "Peace":        "a > b > d > c",  # Club
    "Harmony":      "a > b > c > d",
    "Concord":      "a > c > b > d",
}

# GTI Mapping: Robinson-Goforth type -> GTI canonical type
RG_TO_GTI = {
    "Chicken":      "Chicken",
    "Leader":       "Battle_of_the_Sexes",
    "Hero":         "Hero",
    "Compromise":   "Compromise",
    "Deadlock":     "Deadlock",
    "Dilemma":      "Prisoners_Dilemma",
    "Staghunt":     "Stag_Hunt",
    "Assurance":    "Assurance_Game",
    "Safecoord":    "Coordination_Game",
    "Peace":        "Peace",
    "Harmony":      "Harmony",
    "Concord":      "Concord",
}

# Reverse mapping
GTI_TO_RG = {v: k for k, v in RG_TO_GTI.items()}


def get_ordinal_ranking(payoffs: Tuple[float, float, float, float]) -> str:
    """
    Get the ordinal ranking string for a set of 4 payoffs.
    
    Args:
        payoffs: (a, b, c, d) where a=(A,C), b=(A,D), c=(B,C), d=(B,D)
    
    Returns:
        String like "c > a > b > d" representing the ordering
    """
    a, b, c, d = payoffs
    labels = ['a', 'b', 'c', 'd']
    values = [a, b, c, d]
    
    # Sort by value descending, preserving label
    sorted_pairs = sorted(zip(values, labels), key=lambda x: -x[0])
    
    # Build ordering string
    ranking = " > ".join([label for _, label in sorted_pairs])
    return ranking


def classify_robinson_goforth(game: GameMatrix, player: str = "row") -> Optional[str]:
    """
    Classify a 2x2 game into one of the 12 Robinson-Goforth types.
    
    Args:
        game: GameMatrix object
        player: "row" or "col" - which player's perspective
    
    Returns:
        Robinson-Goforth type name, or None if classification fails
    """
    if player == "row":
        payoffs = game.row_payoffs()
    else:
        payoffs = game.col_payoffs()
    
    ranking = get_ordinal_ranking(payoffs)
    
    # Match against known types
    for rg_type, rg_ranking in ROBINSON_GOFORTH_TYPES.items():
        if ranking == rg_ranking:
            return rg_type
    
    # Handle ties or unusual orderings
    return None


def classify_to_gti(game: GameMatrix, player: str = "row") -> Optional[str]:
    """
    Classify a 2x2 game into GTI taxonomy.
    
    Args:
        game: GameMatrix object
        player: "row" or "col" - which player's perspective
    
    Returns:
        GTI type name, or None if classification fails
    """
    rg_type = classify_robinson_goforth(game, player)
    if rg_type is None:
        return None
    return RG_TO_GTI.get(rg_type)


def analyze_game_properties(game: GameMatrix) -> Dict:
    """
    Analyze strategic properties of a 2x2 game.
    
    Returns dict with:
        - dominant_strategy_row: bool
        - dominant_strategy_col: bool
        - nash_equilibria: list of (row, col) tuples
        - pareto_optimal: list of (row, col) tuples
        - game_type_row: R-G type from row perspective
        - game_type_col: R-G type from col perspective
    """
    a, b, c, d = game.row_payoffs()
    x, y, z, w = game.col_payoffs()
    
    result = {}
    
    # Dominant strategies
    # Row player: A dominates B if a >= c AND b >= d (with at least one strict)
    row_A_dom = (a >= c and b >= d) and (a > c or b > d)
    row_B_dom = (c >= a and d >= b) and (c > a or d > b)
    result['dominant_row'] = 'A' if row_A_dom else ('B' if row_B_dom else None)
    
    # Col player: C dominates D if x >= y AND z >= w (with at least one strict)
    col_C_dom = (x >= y and z >= w) and (x > y or z > w)
    col_D_dom = (y >= x and w >= z) and (y > x or w > z)
    result['dominant_col'] = 'C' if col_C_dom else ('D' if col_D_dom else None)
    
    # Pure strategy Nash equilibria
    # (A,C) is NE if a >= c (row prefers A given C) and x >= y (col prefers C given A)
    nash = []
    if a >= c and x >= y:
        nash.append(('A', 'C'))
    if b >= d and y >= x:
        nash.append(('A', 'D'))
    if c >= a and z >= w:
        nash.append(('B', 'C'))
    if d >= b and w >= z:
        nash.append(('B', 'D'))
    result['nash_equilibria'] = nash
    
    # Pareto optimal outcomes
    outcomes = [
        ('A', 'C', a + x),
        ('A', 'D', b + y),
        ('B', 'C', c + z),
        ('B', 'D', d + w),
    ]
    max_welfare = max(o[2] for o in outcomes)
    result['pareto_optimal'] = [(o[0], o[1]) for o in outcomes if o[2] == max_welfare]
    
    # R-G classification
    result['rg_type_row'] = classify_robinson_goforth(game, 'row')
    result['rg_type_col'] = classify_robinson_goforth(game, 'col')
    result['gti_type_row'] = classify_to_gti(game, 'row')
    result['gti_type_col'] = classify_to_gti(game, 'col')
    
    return result


# Classic game examples for testing
CLASSIC_GAMES = {
    "Prisoners_Dilemma": GameMatrix(
        a=3, b=0, c=5, d=1,  # Row: T > R > P > S (c > a > d > b = Dilemma)
        x=3, y=5, z=0, w=1   # Col: symmetric
    ),
    "Chicken": GameMatrix(
        a=0, b=-1, c=1, d=-10,  # Row: c > a > b > d
        x=0, y=1, z=-1, w=-10   # Col: symmetric
    ),
    "Stag_Hunt": GameMatrix(
        a=4, b=0, c=3, d=2,  # Row: a > c > d > b (Staghunt)
        x=4, y=3, z=0, w=2   # Col: symmetric
    ),
    "Battle_of_the_Sexes": GameMatrix(
        a=3, b=0, c=0, d=2,  # Row: This is actually coordination
        x=2, y=0, z=0, w=3   # Col: asymmetric preferences
    ),
    "Coordination": GameMatrix(
        a=2, b=0, c=0, d=1,  # Row: a > d > b = c 
        x=2, y=0, z=0, w=1   # Col: symmetric
    ),
    "Harmony": GameMatrix(
        a=3, b=2, c=1, d=0,  # Row: a > b > c > d
        x=3, y=2, z=1, w=0   # Col: symmetric
    ),
}


def test_classifier():
    """Test the classifier on classic games"""
    print("Testing Robinson-Goforth Classifier\n" + "="*50)
    
    for name, game in CLASSIC_GAMES.items():
        print(f"\n{name}:")
        print(game.to_bimatrix_str())
        
        props = analyze_game_properties(game)
        print(f"  R-G Type (row): {props['rg_type_row']}")
        print(f"  R-G Type (col): {props['rg_type_col']}")
        print(f"  GTI Type: {props['gti_type_row']}")
        print(f"  Nash Equilibria: {props['nash_equilibria']}")
        print(f"  Dominant Row: {props['dominant_row']}")
        print(f"  Dominant Col: {props['dominant_col']}")


if __name__ == "__main__":
    test_classifier()
