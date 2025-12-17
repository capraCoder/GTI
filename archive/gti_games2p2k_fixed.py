"""
GTI games2p2k Parser - Fixed for actual CSV format

The games2p2k_main.csv has this structure:
- row_form_matrix: '[21 10 24 17  2 19  6  4]' - payoffs as string
- topology: "('dilemma', 'hero')" - GROUND TRUTH R-G types
- role: 'rowplayer' or 'colplayer'  
- up_choice: human choice frequency for "up" (row A)

This script correctly parses these columns.
"""

import os
import sys
import json
import csv
import yaml
import time
import argparse
import random
import re
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from collections import Counter, defaultdict
from datetime import datetime

# Try to import anthropic
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# =============================================================================
# ROBINSON-GOFORTH MAPPINGS
# =============================================================================

# R-G type names as they appear in topology column -> GTI canonical names
RG_TO_GTI = {
    "chicken": "Chicken",
    "leader": "Battle_of_the_Sexes",
    "hero": "Hero",
    "compromise": "Compromise",
    "deadlock": "Deadlock",
    "dilemma": "Prisoners_Dilemma",
    "staghunt": "Stag_Hunt",
    "assurance": "Assurance_Game",
    "safecoord": "Coordination_Game",
    "peace": "Peace",
    "harmony": "Harmony",
    "concord": "Concord",
}

# Normalization for Claude outputs
NORMALIZATION_MAP = {
    "prisoners dilemma": "Prisoners_Dilemma",
    "prisoner's dilemma": "Prisoners_Dilemma",
    "prisoners_dilemma": "Prisoners_Dilemma",
    "pd": "Prisoners_Dilemma",
    "dilemma": "Prisoners_Dilemma",
    "chicken": "Chicken",
    "chicken game": "Chicken",
    "hawk-dove": "Chicken",
    "hawk dove": "Chicken",
    "snowdrift": "Chicken",
    "stag hunt": "Stag_Hunt",
    "stag_hunt": "Stag_Hunt",
    "staghunt": "Stag_Hunt",
    "trust game": "Stag_Hunt",
    "battle of the sexes": "Battle_of_the_Sexes",
    "battle_of_the_sexes": "Battle_of_the_Sexes",
    "bos": "Battle_of_the_Sexes",
    "leader": "Battle_of_the_Sexes",
    "coordination": "Coordination_Game",
    "coordination game": "Coordination_Game",
    "coordination_game": "Coordination_Game",
    "pure coordination": "Coordination_Game",
    "safecoord": "Coordination_Game",
    "assurance": "Assurance_Game",
    "assurance game": "Assurance_Game",
    "assurance_game": "Assurance_Game",
    "harmony": "Harmony",
    "harmony game": "Harmony",
    "deadlock": "Deadlock",
    "hero": "Hero",
    "hero game": "Hero",
    "compromise": "Compromise",
    "compromise game": "Compromise",
    "peace": "Peace",
    "peace game": "Peace",
    "club": "Peace",
    "concord": "Concord",
    "concord game": "Concord",
    "matching pennies": "Matching_Pennies",
    "zero-sum": "Matching_Pennies",
    "zero sum": "Matching_Pennies",
    "common pool": "Common_Pool_Resource",
    "common pool resource": "Common_Pool_Resource",
    "commons": "Common_Pool_Resource",
    "public goods": "Public_Goods_Game",
}


def normalize_game_type(raw: str) -> str:
    """Normalize a game type string to canonical form"""
    if not raw:
        return raw
    cleaned = raw.strip().lower().replace("_", " ").replace("-", " ")
    if cleaned in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[cleaned]
    return raw.strip().replace(" ", "_").title()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GameEntry:
    """Single game from games2p2k dataset"""
    unique_id: str
    game_id: str
    payoffs: List[int]  # [a, b, c, d, x, y, z, w] from row_form_matrix
    role: str  # 'rowplayer' or 'colplayer'
    topology_row: str  # R-G type for row player
    topology_col: str  # R-G type for column player
    gti_type: str  # GTI type based on role
    up_choice: float  # Human choice frequency
    
    def to_matrix_str(self) -> str:
        """Format as payoff matrix string"""
        a, b, c, d, x, y, z, w = self.payoffs
        return f"""       C           D
A   ({a:4},{x:4})  ({b:4},{y:4})
B   ({c:4},{z:4})  ({d:4},{w:4})"""


def parse_matrix_string(matrix_str: str) -> List[int]:
    """
    Parse '[21 10 24 17  2 19  6  4]' into [21, 10, 24, 17, 2, 19, 6, 4]
    """
    # Remove brackets and split on whitespace
    cleaned = matrix_str.strip().strip('[]')
    # Handle multiple spaces
    values = [int(x) for x in cleaned.split() if x]
    if len(values) != 8:
        raise ValueError(f"Expected 8 payoffs, got {len(values)}: {matrix_str}")
    return values


def parse_topology(topology_str: str) -> Tuple[str, str]:
    """
    Parse "('dilemma', 'hero')" into ('dilemma', 'hero')
    """
    try:
        # Use ast.literal_eval for safe parsing
        result = ast.literal_eval(topology_str)
        if isinstance(result, tuple) and len(result) == 2:
            return result[0].lower(), result[1].lower()
    except:
        pass
    
    # Fallback: regex parsing
    match = re.search(r"\('(\w+)',\s*'(\w+)'\)", topology_str)
    if match:
        return match.group(1).lower(), match.group(2).lower()
    
    raise ValueError(f"Could not parse topology: {topology_str}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_games2p2k(filepath: Path, max_games: Optional[int] = None) -> List[GameEntry]:
    """Load games2p2k_main.csv with correct column parsing"""
    entries = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            if max_games and i >= max_games:
                break
            
            try:
                # Parse payoff matrix
                payoffs = parse_matrix_string(row['row_form_matrix'])
                
                # Parse topology (ground truth!)
                topo_row, topo_col = parse_topology(row['topology'])
                
                # Get role
                role = row['role'].strip().lower()
                is_row = 'row' in role
                
                # Determine GTI type based on role
                if is_row:
                    rg_type = topo_row
                else:
                    rg_type = topo_col
                
                gti_type = RG_TO_GTI.get(rg_type, rg_type.title())
                
                # Get choice frequency
                up_choice = float(row['up_choice'])
                
                entry = GameEntry(
                    unique_id=row['unique_id'],
                    game_id=row['game_id'],
                    payoffs=payoffs,
                    role='row' if is_row else 'col',
                    topology_row=topo_row,
                    topology_col=topo_col,
                    gti_type=gti_type,
                    up_choice=up_choice,
                )
                entries.append(entry)
                
            except Exception as e:
                if i < 5:
                    print(f"  Warning row {i}: {e}")
                continue
    
    return entries


# =============================================================================
# STATISTICS
# =============================================================================

def show_statistics(games: List[GameEntry]):
    """Display comprehensive statistics"""
    print("\n" + "=" * 60)
    print("GAMES2P2K DATASET STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal entries: {len(games)}")
    
    # Unique games (by game_id)
    unique_games = len(set(g.game_id for g in games))
    print(f"Unique games: {unique_games}")
    print(f"(Each game appears twice: once for row, once for column player)")
    
    # Role distribution
    role_counts = Counter(g.role for g in games)
    print(f"\nBy role: {dict(role_counts)}")
    
    # R-G type distribution (row topology)
    print("\n" + "-" * 40)
    print("ROBINSON-GOFORTH TYPES (from topology column)")
    print("-" * 40)
    
    rg_row_counts = Counter(g.topology_row for g in games)
    print(f"\nRow player types ({len(rg_row_counts)} unique):")
    for rg, count in sorted(rg_row_counts.items(), key=lambda x: -x[1]):
        gti = RG_TO_GTI.get(rg, "?")
        pct = 100 * count / len(games)
        print(f"  {rg:15} -> {gti:25} : {count:5} ({pct:5.1f}%)")
    
    rg_col_counts = Counter(g.topology_col for g in games)
    print(f"\nColumn player types ({len(rg_col_counts)} unique):")
    for rg, count in sorted(rg_col_counts.items(), key=lambda x: -x[1]):
        gti = RG_TO_GTI.get(rg, "?")
        pct = 100 * count / len(games)
        print(f"  {rg:15} -> {gti:25} : {count:5} ({pct:5.1f}%)")
    
    # GTI distribution (based on role)
    print("\n" + "-" * 40)
    print("GTI TYPES (perspective-dependent)")
    print("-" * 40)
    
    gti_counts = Counter(g.gti_type for g in games)
    print(f"\nGTI types ({len(gti_counts)} unique):")
    for gti, count in sorted(gti_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(games)
        print(f"  {gti:25} : {count:5} ({pct:5.1f}%)")
    
    # Topology pairs
    print("\n" + "-" * 40)
    print("TOPOLOGY PAIRS (row, col)")
    print("-" * 40)
    
    pair_counts = Counter((g.topology_row, g.topology_col) for g in games)
    print(f"\nTop 20 topology pairs:")
    for (row, col), count in pair_counts.most_common(20):
        gti_r = RG_TO_GTI.get(row, row)[:15]
        gti_c = RG_TO_GTI.get(col, col)[:15]
        print(f"  ({row:12}, {col:12}) : {count:4}")
    
    # Human choice stats
    print("\n" + "-" * 40)
    print("HUMAN CHOICE FREQUENCIES")
    print("-" * 40)
    
    choices = [g.up_choice for g in games]
    print(f"  Mean choice(A): {sum(choices)/len(choices):.3f}")
    print(f"  Min: {min(choices):.3f}, Max: {max(choices):.3f}")


# =============================================================================
# CLAUDE CLASSIFIER
# =============================================================================

GTI_PROMPT = """You are a game theory expert. Classify this 2x2 strategic game.

GAME TYPES (Robinson-Goforth topology):
- Prisoners_Dilemma: Dominant defection, mutual cooperation better but unstable
- Chicken: Anti-coordination, mutual aggression worst
- Stag_Hunt: Coordination with risky high-payoff option  
- Battle_of_the_Sexes: Coordination with conflicting preferences
- Coordination_Game: Pure coordination
- Assurance_Game: Coordination requiring trust
- Harmony: No conflict, cooperation dominates
- Deadlock: Defection dominates and is Pareto optimal
- Hero: One player sacrifices for mutual benefit
- Compromise: Mutual concession preferred
- Peace: Cooperation dominates, stable
- Concord: Strong mutual cooperation preference

Payoff matrix (Row player, Column player):
{matrix}

The {role} player is making the decision.

Respond with ONLY the game type name (e.g., "Prisoners_Dilemma"):"""


class ClaudeClassifier:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic()
        self.model = model
        self.calls = 0
        self.tokens = 0
    
    def classify(self, game: GameEntry) -> str:
        prompt = GTI_PROMPT.format(
            matrix=game.to_matrix_str(),
            role=game.role
        )
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )
        
        self.calls += 1
        self.tokens += response.usage.input_tokens + response.usage.output_tokens
        
        raw = response.content[0].text.strip()
        return normalize_game_type(raw)


# =============================================================================
# VALIDATION
# =============================================================================

@dataclass
class ValidationResult:
    unique_id: str
    expected: str
    predicted: str
    correct: bool
    topology: Tuple[str, str]
    role: str


def run_validation(games: List[GameEntry], classifier: ClaudeClassifier, 
                   delay: float = 0.5) -> Dict:
    """Run validation with progress display"""
    results = []
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(lambda: defaultdict(int))
    
    start = time.time()
    
    for i, game in enumerate(games):
        print(f"\r  Validating {i+1}/{len(games)}...", end="", flush=True)
        
        try:
            predicted = classifier.classify(game)
            expected = game.gti_type
            correct = (predicted == expected)
            
            results.append(ValidationResult(
                unique_id=game.unique_id,
                expected=expected,
                predicted=predicted,
                correct=correct,
                topology=(game.topology_row, game.topology_col),
                role=game.role,
            ))
            
            by_type[expected]["total"] += 1
            if correct:
                by_type[expected]["correct"] += 1
            
            confusion[expected][predicted] += 1
            
            time.sleep(delay)
            
        except Exception as e:
            print(f"\n  Error: {e}")
    
    print()
    elapsed = time.time() - start
    n_correct = sum(1 for r in results if r.correct)
    
    return {
        'n_games': len(results),
        'n_correct': n_correct,
        'accuracy': n_correct / len(results) if results else 0,
        'by_type': dict(by_type),
        'confusion': {k: dict(v) for k, v in confusion.items()},
        'results': results,
        'api_calls': classifier.calls,
        'tokens': classifier.tokens,
        'elapsed': elapsed,
    }


def print_validation_report(report: Dict):
    """Print validation results"""
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    
    print(f"\nOverall: {report['n_correct']}/{report['n_games']} ({100*report['accuracy']:.1f}%)")
    print(f"API calls: {report['api_calls']}, Tokens: {report['tokens']}")
    print(f"Time: {report['elapsed']:.1f}s")
    
    print("\nBy Game Type:")
    print("-" * 50)
    for gti, stats in sorted(report['by_type'].items()):
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {gti:25} : {stats['correct']:3}/{stats['total']:3} ({100*acc:5.1f}%)")
    
    print("\nMisclassifications (first 15):")
    print("-" * 50)
    misses = [r for r in report['results'] if not r.correct]
    for r in misses[:15]:
        print(f"  {r.unique_id}: {r.expected} -> {r.predicted} (topo: {r.topology})")


# =============================================================================
# CORPUS GENERATION
# =============================================================================

def generate_corpus(games: List[GameEntry], output_path: Path):
    """Generate GTI YAML corpus with ground truth labels"""
    cases = []
    
    for g in games:
        cases.append({
            'case_id': f"G2P2K-{g.unique_id}",
            'expected': g.gti_type,
            'rg_type_row': g.topology_row,
            'rg_type_col': g.topology_col,
            'role': g.role,
            'payoffs': g.payoffs,
            'human_choice_freq': round(g.up_choice, 4),
            'matrix': g.to_matrix_str(),
        })
    
    corpus = {
        'metadata': {
            'source': 'games2p2k (Zhu, Peterson, Enke, Griffiths 2024)',
            'generated': datetime.now().isoformat(),
            'n_cases': len(cases),
            'ground_truth': 'Robinson-Goforth topology from dataset',
        },
        'cases': cases,
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(corpus, f, default_flow_style=False, width=120)
    
    print(f"\nGenerated corpus: {output_path}")
    print(f"  {len(cases)} cases with ground truth labels")
    
    # Distribution
    type_counts = Counter(c['expected'] for c in cases)
    print("\nType distribution:")
    for t, c in type_counts.most_common():
        print(f"  {t:25} : {c:5}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GTI games2p2k Validation (Fixed Parser)")
    parser.add_argument('--data', type=str, required=True, help='Path to games2p2k_main.csv')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--validate', type=str, metavar='N', help='Validate N games (or "all")')
    parser.add_argument('--corpus', action='store_true', help='Generate corpus')
    parser.add_argument('--output', type=str, default='gti_corpus_games2p2k.yaml')
    parser.add_argument('--delay', type=float, default=0.5)
    parser.add_argument('--model', type=str, default='claude-sonnet-4-20250514')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.data}...")
    games = load_games2p2k(Path(args.data))
    print(f"Loaded {len(games)} entries")
    
    if args.stats:
        show_statistics(games)
    
    if args.corpus:
        generate_corpus(games, Path(args.output))
    
    if args.validate:
        if not HAS_ANTHROPIC:
            print("Error: pip install anthropic")
            return
        
        if args.validate.lower() == 'all':
            sample = games
        else:
            n = min(int(args.validate), len(games))
            sample = random.sample(games, n)
        
        print(f"\nValidating {len(sample)} games...")
        classifier = ClaudeClassifier(model=args.model)
        report = run_validation(sample, classifier, args.delay)
        print_validation_report(report)
        
        # Save report
        out_file = f"gti_validation_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(out_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'n_games': report['n_games'],
                'accuracy': report['accuracy'],
                'by_type': report['by_type'],
                'confusion': report['confusion'],
            }, f, indent=2)
        print(f"\nSaved: {out_file}")
    
    if not any([args.stats, args.validate, args.corpus]):
        show_statistics(games)


if __name__ == "__main__":
    main()
