"""
GTI Validation Pipeline with games2p2k Dataset

This script validates the Claude-based GTI classifier against the games2p2k
dataset which provides ground-truth labels via Robinson-Goforth topology.

Usage:
    # First, discover the data structure:
    python gti_validate_games2p2k.py --discover --data path/to/games2p2k.csv
    
    # Run validation on N random samples:
    python gti_validate_games2p2k.py --validate 100 --data path/to/games2p2k.csv
    
    # Run full validation:
    python gti_validate_games2p2k.py --validate all --data path/to/games2p2k.csv
    
    # Generate corpus for manual review:
    python gti_validate_games2p2k.py --corpus --data path/to/games2p2k.csv

Requirements:
    pip install anthropic pyyaml pandas
"""

import os
import sys
import json
import csv
import yaml
import time
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from collections import Counter, defaultdict
from datetime import datetime

# Try to import optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Note: pandas not available, using csv module")

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("Warning: anthropic not installed. Install with: pip install anthropic")


# =============================================================================
# ROBINSON-GOFORTH CLASSIFIER (inline to avoid import issues)
# =============================================================================

ROBINSON_GOFORTH_TYPES = {
    "Chicken":      "c > a > b > d",
    "Leader":       "c > b > a > d",
    "Hero":         "c > b > d > a",
    "Compromise":   "c > d > b > a",
    "Deadlock":     "c > d > a > b",
    "Dilemma":      "c > a > d > b",
    "Staghunt":     "a > c > d > b",
    "Assurance":    "a > d > c > b",
    "Safecoord":    "a > d > b > c",
    "Peace":        "a > b > d > c",
    "Harmony":      "a > b > c > d",
    "Concord":      "a > c > b > d",
}

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

# Normalization map for Claude's outputs
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


def get_ordinal_ranking(payoffs: Tuple[float, float, float, float]) -> str:
    """Get ordinal ranking string for 4 payoffs (a, b, c, d)"""
    a, b, c, d = payoffs
    labels = ['a', 'b', 'c', 'd']
    values = [a, b, c, d]
    sorted_pairs = sorted(zip(values, labels), key=lambda x: -x[0])
    return " > ".join([label for _, label in sorted_pairs])


def classify_rg(payoffs: List[float], player: str = "row") -> Optional[str]:
    """Classify payoffs into Robinson-Goforth type"""
    if player == "row":
        p = (payoffs[0], payoffs[1], payoffs[2], payoffs[3])
    else:
        p = (payoffs[4], payoffs[5], payoffs[6], payoffs[7])
    
    ranking = get_ordinal_ranking(p)
    
    for rg_type, rg_ranking in ROBINSON_GOFORTH_TYPES.items():
        if ranking == rg_ranking:
            return rg_type
    return None


def classify_gti(payoffs: List[float], player: str = "row") -> Optional[str]:
    """Classify payoffs into GTI type"""
    rg_type = classify_rg(payoffs, player)
    if rg_type:
        return RG_TO_GTI.get(rg_type)
    return None


def normalize_game_type(raw: str) -> str:
    """Normalize a game type string to canonical form"""
    if not raw:
        return raw
    
    # Clean and lowercase
    cleaned = raw.strip().lower()
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    
    # Check normalization map
    if cleaned in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[cleaned]
    
    # Try with underscores
    with_underscores = cleaned.replace(" ", "_")
    for key, value in NORMALIZATION_MAP.items():
        if key.replace(" ", "_") == with_underscores:
            return value
    
    # Title case as fallback
    return raw.strip().replace(" ", "_").title()


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class GameEntry:
    """Single game from games2p2k dataset"""
    game_id: str
    payoffs: List[float]  # [a, b, c, d, x, y, z, w]
    rg_type_row: Optional[str] = None
    rg_type_col: Optional[str] = None
    gti_type: Optional[str] = None
    player_role: str = "row"
    choice_freq: Optional[float] = None
    n_obs: Optional[int] = None
    
    def to_matrix_str(self) -> str:
        """Format as payoff matrix string"""
        a, b, c, d, x, y, z, w = self.payoffs
        return f"""       C           D
A   ({a:4.0f},{x:4.0f})  ({b:4.0f},{y:4.0f})
B   ({c:4.0f},{z:4.0f})  ({d:4.0f},{w:4.0f})"""


def discover_csv_structure(filepath: Path) -> Dict:
    """Analyze CSV to understand structure"""
    with open(filepath, 'r', encoding='utf-8') as f:
        # Try to detect delimiter
        sample = f.read(4096)
        f.seek(0)
        
        # Count potential delimiters
        delimiters = {',': sample.count(','), '\t': sample.count('\t'), ';': sample.count(';')}
        delimiter = max(delimiters, key=delimiters.get)
        
        reader = csv.reader(f, delimiter=delimiter)
        headers = next(reader)
        
        # Get sample rows
        sample_rows = []
        for i, row in enumerate(reader):
            if i >= 5:
                break
            sample_rows.append(row)
        
        # Count total rows
        f.seek(0)
        total_rows = sum(1 for _ in f) - 1  # Exclude header
    
    return {
        'filepath': str(filepath),
        'delimiter': delimiter,
        'n_columns': len(headers),
        'n_rows': total_rows,
        'headers': headers,
        'sample_rows': sample_rows,
    }


def load_games2p2k(filepath: Path, max_games: Optional[int] = None) -> List[GameEntry]:
    """
    Load games2p2k dataset with automatic format detection.
    
    The dataset may have various formats. This function tries to detect:
    - Separate columns for each payoff (a, b, c, d, x, y, z, w)
    - Combined payoff columns
    - Game ID columns
    - Choice frequency columns
    """
    entries = []
    
    # Discover structure
    info = discover_csv_structure(filepath)
    headers = [h.lower().strip() for h in info['headers']]
    original_headers = info['headers']
    delimiter = info['delimiter']
    
    print(f"Loading {filepath}")
    print(f"  Columns: {len(headers)}")
    print(f"  Rows: {info['n_rows']}")
    print(f"  Headers: {original_headers[:10]}{'...' if len(headers) > 10 else ''}")
    
    # Identify payoff columns
    payoff_patterns = {
        'individual': ['a', 'b', 'c', 'd', 'x', 'y', 'z', 'w'],
        'row_col': ['row_a', 'row_b', 'row_c', 'row_d', 'col_a', 'col_b', 'col_c', 'col_d'],
        'p1_p2': ['p1_aa', 'p1_ab', 'p1_ba', 'p1_bb', 'p2_aa', 'p2_ab', 'p2_ba', 'p2_bb'],
        'numbered': ['payoff1', 'payoff2', 'payoff3', 'payoff4', 'payoff5', 'payoff6', 'payoff7', 'payoff8'],
    }
    
    payoff_cols = None
    for pattern_name, pattern in payoff_patterns.items():
        if all(p in headers for p in pattern):
            payoff_cols = [headers.index(p) for p in pattern]
            print(f"  Payoff pattern: {pattern_name}")
            break
    
    # Fallback: look for numeric columns
    if payoff_cols is None:
        # Try to find 8 consecutive numeric columns
        numeric_cols = []
        for i, row in enumerate(info['sample_rows']):
            if i == 0:
                for j, val in enumerate(row):
                    try:
                        float(val)
                        numeric_cols.append(j)
                    except (ValueError, TypeError):
                        pass
        if len(numeric_cols) >= 8:
            payoff_cols = numeric_cols[:8]
            print(f"  Using first 8 numeric columns as payoffs")
    
    # ID column
    id_col = None
    for candidate in ['game_id', 'gameid', 'id', 'game', 'index']:
        if candidate in headers:
            id_col = headers.index(candidate)
            break
    
    # Choice frequency column
    freq_col = None
    for candidate in ['choice_freq', 'prop_a', 'freq_a', 'pa', 'choice', 'prop_up']:
        if candidate in headers:
            freq_col = headers.index(candidate)
            break
    
    # Role column
    role_col = None
    for candidate in ['role', 'player_role', 'player']:
        if candidate in headers:
            role_col = headers.index(candidate)
            break
    
    # Now load the data
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        next(reader)  # Skip header
        
        for i, row in enumerate(reader):
            if max_games and i >= max_games:
                break
            
            try:
                # Extract payoffs
                if payoff_cols:
                    payoffs = [float(row[c]) for c in payoff_cols]
                else:
                    # Try to extract any 8 floats
                    floats = []
                    for val in row:
                        try:
                            floats.append(float(val))
                        except (ValueError, TypeError):
                            pass
                    if len(floats) >= 8:
                        payoffs = floats[:8]
                    else:
                        continue
                
                # Game ID
                game_id = row[id_col] if id_col is not None else str(i)
                
                # Role
                role = "row"
                if role_col is not None:
                    r = row[role_col].lower().strip()
                    if r in ['col', 'column', '1']:
                        role = "col"
                
                # Choice frequency
                freq = None
                if freq_col is not None:
                    try:
                        freq = float(row[freq_col])
                    except (ValueError, TypeError):
                        pass
                
                # Classify
                rg_row = classify_rg(payoffs, "row")
                rg_col = classify_rg(payoffs, "col")
                gti = classify_gti(payoffs, role)
                
                entry = GameEntry(
                    game_id=game_id,
                    payoffs=payoffs,
                    rg_type_row=rg_row,
                    rg_type_col=rg_col,
                    gti_type=gti,
                    player_role=role,
                    choice_freq=freq,
                )
                entries.append(entry)
                
            except Exception as e:
                if i < 5:  # Only warn for first few errors
                    print(f"  Warning: Error parsing row {i}: {e}")
                continue
    
    print(f"  Loaded {len(entries)} games")
    return entries


# =============================================================================
# CLAUDE CLASSIFIER
# =============================================================================

GTI_CLASSIFIER_PROMPT = """You are a game theory expert. Classify the following strategic interaction into one of these canonical game types:

CORE 2x2 TYPES (Robinson-Goforth topology):
- Prisoners_Dilemma: Dominant defection, mutual cooperation better but unstable
- Chicken: Anti-coordination, mutual aggression worst outcome  
- Stag_Hunt: Coordination with risky high-payoff option
- Battle_of_the_Sexes: Coordination with conflicting preferences
- Coordination_Game: Pure coordination, both equilibria acceptable
- Assurance_Game: Coordination requiring mutual trust
- Harmony: No conflict, cooperation dominates
- Deadlock: Defection dominates but is Pareto optimal
- Hero: One player sacrifices for mutual benefit
- Compromise: Mutual concession preferred
- Peace: Cooperation dominates, stable peace
- Concord: Mutual cooperation strongly preferred

EXTENDED TYPES:
- Matching_Pennies: Pure conflict, zero-sum
- Public_Goods_Game: N-player contribution dilemma
- Common_Pool_Resource: Shared resource extraction
- Ultimatum_Game: Proposer/responder bargaining
- Trust_Game: Sequential trust and reciprocity
- Bertrand_Competition: Price competition
- Cournot_Competition: Quantity competition

Analyze this game and respond with ONLY the canonical game type name (e.g., "Prisoners_Dilemma"):

{description}"""


class ClaudeClassifier:
    """Claude-based game type classifier"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package required: pip install anthropic")
        
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        self.calls = 0
        self.tokens = 0
    
    def classify(self, game: GameEntry) -> str:
        """Classify a game using Claude"""
        description = f"""Payoff matrix for a 2-player strategic game:

{game.to_matrix_str()}

The first number in each cell is Player 1's (Row's) payoff.
The second number is Player 2's (Column's) payoff.

What type of game is this?"""
        
        prompt = GTI_CLASSIFIER_PROMPT.format(description=description)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
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
    """Result of validating one game"""
    game_id: str
    expected: str
    predicted: str
    correct: bool
    rg_type: Optional[str] = None
    payoffs: List[float] = field(default_factory=list)


@dataclass 
class ValidationReport:
    """Summary of validation run"""
    timestamp: str
    n_games: int
    n_correct: int
    accuracy: float
    by_type: Dict[str, Dict[str, int]]
    confusion_matrix: Dict[str, Dict[str, int]]
    results: List[ValidationResult]
    api_calls: int = 0
    tokens_used: int = 0
    elapsed_seconds: float = 0


def run_validation(
    games: List[GameEntry],
    classifier: ClaudeClassifier,
    delay: float = 0.5
) -> ValidationReport:
    """Run validation on a set of games"""
    
    results = []
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(lambda: defaultdict(int))
    
    start_time = time.time()
    
    for i, game in enumerate(games):
        if game.gti_type is None:
            continue
        
        print(f"\r  Validating {i+1}/{len(games)}...", end="", flush=True)
        
        try:
            predicted = classifier.classify(game)
            expected = game.gti_type
            correct = (predicted == expected)
            
            result = ValidationResult(
                game_id=game.game_id,
                expected=expected,
                predicted=predicted,
                correct=correct,
                rg_type=game.rg_type_row,
                payoffs=game.payoffs,
            )
            results.append(result)
            
            by_type[expected]["total"] += 1
            if correct:
                by_type[expected]["correct"] += 1
            
            confusion[expected][predicted] += 1
            
            time.sleep(delay)
            
        except Exception as e:
            print(f"\n  Error on game {game.game_id}: {e}")
            continue
    
    print()  # New line after progress
    
    elapsed = time.time() - start_time
    n_correct = sum(1 for r in results if r.correct)
    accuracy = n_correct / len(results) if results else 0
    
    return ValidationReport(
        timestamp=datetime.now().isoformat(),
        n_games=len(results),
        n_correct=n_correct,
        accuracy=accuracy,
        by_type=dict(by_type),
        confusion_matrix={k: dict(v) for k, v in confusion.items()},
        results=results,
        api_calls=classifier.calls,
        tokens_used=classifier.tokens,
        elapsed_seconds=elapsed,
    )


def print_report(report: ValidationReport):
    """Print validation report to console"""
    print("\n" + "=" * 60)
    print("GTI VALIDATION REPORT")
    print("=" * 60)
    
    print(f"\nOverall: {report.n_correct}/{report.n_games} correct ({100*report.accuracy:.1f}%)")
    print(f"API calls: {report.api_calls}, Tokens: {report.tokens_used}")
    print(f"Time: {report.elapsed_seconds:.1f}s")
    
    print("\nBy Game Type:")
    print("-" * 50)
    for gti_type, stats in sorted(report.by_type.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {gti_type:25} : {stats['correct']:3}/{stats['total']:3} ({100*acc:5.1f}%)")
    
    # Show misclassifications
    print("\nMisclassifications:")
    print("-" * 50)
    misses = [r for r in report.results if not r.correct]
    for r in misses[:10]:  # Show first 10
        print(f"  {r.game_id}: Expected {r.expected}, Got {r.predicted}")
    if len(misses) > 10:
        print(f"  ... and {len(misses) - 10} more")


def save_report(report: ValidationReport, filepath: Path):
    """Save validation report to JSON"""
    data = {
        'timestamp': report.timestamp,
        'summary': {
            'n_games': report.n_games,
            'n_correct': report.n_correct,
            'accuracy': report.accuracy,
            'api_calls': report.api_calls,
            'tokens_used': report.tokens_used,
            'elapsed_seconds': report.elapsed_seconds,
        },
        'by_type': report.by_type,
        'confusion_matrix': report.confusion_matrix,
        'results': [asdict(r) for r in report.results],
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nReport saved to: {filepath}")


# =============================================================================
# CORPUS GENERATION
# =============================================================================

def generate_corpus(games: List[GameEntry], output_path: Path):
    """Generate GTI-format YAML corpus from games2p2k data"""
    
    cases = []
    for game in games:
        if game.gti_type is None:
            continue
        
        case = {
            'case_id': f"G2P2K-{game.game_id}",
            'expected': game.gti_type,
            'rg_type': game.rg_type_row,
            'difficulty': 'medium',  # Default
            'payoffs': game.payoffs,
            'matrix': game.to_matrix_str(),
        }
        
        if game.choice_freq is not None:
            case['human_choice_freq'] = game.choice_freq
        
        cases.append(case)
    
    corpus = {
        'metadata': {
            'source': 'games2p2k',
            'generated': datetime.now().isoformat(),
            'n_cases': len(cases),
        },
        'cases': cases,
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(corpus, f, default_flow_style=False)
    
    print(f"Generated corpus with {len(cases)} cases: {output_path}")
    
    # Show distribution
    type_counts = Counter(c['expected'] for c in cases)
    print("\nType distribution:")
    for gti_type, count in type_counts.most_common():
        print(f"  {gti_type:25} : {count:5}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GTI Validation with games2p2k Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gti_validate_games2p2k.py --discover --data games2p2k.csv
  python gti_validate_games2p2k.py --validate 100 --data games2p2k.csv
  python gti_validate_games2p2k.py --corpus --data games2p2k.csv
        """
    )
    
    parser.add_argument('--data', type=str, required=True, help='Path to games2p2k CSV file')
    parser.add_argument('--discover', action='store_true', help='Analyze data structure')
    parser.add_argument('--validate', type=str, metavar='N', help='Validate on N games (or "all")')
    parser.add_argument('--corpus', action='store_true', help='Generate GTI corpus')
    parser.add_argument('--stats', action='store_true', help='Show dataset statistics')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-20250514', help='Claude model')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between API calls')
    
    args = parser.parse_args()
    
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: File not found: {data_path}")
        sys.exit(1)
    
    # Discover mode
    if args.discover:
        info = discover_csv_structure(data_path)
        print("\n" + "=" * 60)
        print("DATA STRUCTURE DISCOVERY")
        print("=" * 60)
        print(f"\nFile: {info['filepath']}")
        print(f"Delimiter: {repr(info['delimiter'])}")
        print(f"Columns: {info['n_columns']}")
        print(f"Rows: {info['n_rows']}")
        print(f"\nHeaders:")
        for i, h in enumerate(info['headers']):
            print(f"  {i:2}: {h}")
        print(f"\nSample rows:")
        for row in info['sample_rows'][:3]:
            print(f"  {row[:8]}{'...' if len(row) > 8 else ''}")
        return
    
    # Load data
    games = load_games2p2k(data_path)
    
    # Stats mode
    if args.stats:
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        
        print(f"\nTotal games: {len(games)}")
        
        # R-G distribution
        rg_counts = Counter(g.rg_type_row for g in games if g.rg_type_row)
        print(f"\nRobinson-Goforth Types ({len(rg_counts)}):")
        for rg, count in rg_counts.most_common():
            gti = RG_TO_GTI.get(rg, "?")
            print(f"  {rg:15} -> {gti:25} : {count:5}")
        
        # GTI distribution
        gti_counts = Counter(g.gti_type for g in games if g.gti_type)
        print(f"\nGTI Types ({len(gti_counts)}):")
        for gti, count in gti_counts.most_common():
            print(f"  {gti:25} : {count:5}")
        
        # Unclassified
        unclass = sum(1 for g in games if g.gti_type is None)
        if unclass:
            print(f"\nUnclassified (ties): {unclass}")
        return
    
    # Corpus generation
    if args.corpus:
        output_path = Path(args.output or "gti_corpus_games2p2k.yaml")
        generate_corpus(games, output_path)
        return
    
    # Validation
    if args.validate:
        if not HAS_ANTHROPIC:
            print("Error: anthropic package required for validation")
            print("Install with: pip install anthropic")
            sys.exit(1)
        
        # Select games
        valid_games = [g for g in games if g.gti_type is not None]
        
        if args.validate.lower() == 'all':
            sample = valid_games
        else:
            n = int(args.validate)
            if n > len(valid_games):
                n = len(valid_games)
            sample = random.sample(valid_games, n)
        
        print(f"\nValidating on {len(sample)} games...")
        
        classifier = ClaudeClassifier(model=args.model)
        report = run_validation(sample, classifier, delay=args.delay)
        
        print_report(report)
        
        # Save report
        output_path = Path(args.output or f"gti_validation_{datetime.now():%Y%m%d_%H%M%S}.json")
        save_report(report, output_path)
        return
    
    # Default: show stats
    print("\nNo action specified. Use --help for options.")
    print("Quick stats:")
    gti_counts = Counter(g.gti_type for g in games if g.gti_type)
    for gti, count in gti_counts.most_common(5):
        print(f"  {gti}: {count}")


if __name__ == "__main__":
    main()
