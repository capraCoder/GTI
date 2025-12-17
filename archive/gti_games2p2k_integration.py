"""
Games2p2k Integration for GTI (Game Type Identification)

This script processes the games2p2k dataset from Zhu, Peterson, Enke & Griffiths (2024)
and creates a labeled corpus for validating the GTI classifier.

Dataset source: https://osf.io/xrvaw/
Paper: "Capturing the Complexity of Human Strategic Decision-Making with Machine Learning"
       Nature Human Behaviour (2024/2025)

Usage:
    python gti_games2p2k_integration.py --discover      # Discover data structure
    python gti_games2p2k_integration.py --convert       # Convert to GTI corpus
    python gti_games2p2k_integration.py --validate 100  # Validate on N games
    python gti_games2p2k_integration.py --stats         # Show dataset statistics
"""

import os
import sys
import json
import csv
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter
import random

# Import the R-G classifier (assumes it's in same directory)
try:
    from gti_robinson_goforth import (
        GameMatrix, classify_robinson_goforth, classify_to_gti,
        analyze_game_properties, RG_TO_GTI, ROBINSON_GOFORTH_TYPES
    )
except ImportError:
    print("Error: gti_robinson_goforth.py must be in the same directory")
    sys.exit(1)


# Configuration
DATA_DIR = Path(".")  # Current directory, adjust as needed
OUTPUT_DIR = Path("corpus")
GAMES2P2K_PATTERNS = [
    "games2p2k.csv",
    "games2p2k_data.csv", 
    "data/games2p2k.csv",
    "games2p2k/games2p2k.csv",
    "main_experiment_data.csv",
    "experiment_data.csv",
]


@dataclass
class Game2p2kEntry:
    """Represents one entry from the games2p2k dataset"""
    game_id: str
    payoffs: List[float]  # [a, b, c, d, x, y, z, w]
    player_role: str  # 'row' or 'col'
    rg_type: Optional[str] = None
    gti_type: Optional[str] = None
    choice_freq_A: Optional[float] = None  # Fraction choosing A
    n_observations: Optional[int] = None


def find_data_file() -> Optional[Path]:
    """Search for the games2p2k data file"""
    for pattern in GAMES2P2K_PATTERNS:
        path = DATA_DIR / pattern
        if path.exists():
            return path
    
    # Also search in current directory
    for f in Path(".").glob("*.csv"):
        if "game" in f.name.lower() or "2p2k" in f.name.lower():
            return f
    
    return None


def discover_csv_structure(filepath: Path, n_rows: int = 5) -> Dict:
    """Analyze CSV structure to understand column layout"""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        sample_rows = []
        for i, row in enumerate(reader):
            if i >= n_rows:
                break
            sample_rows.append(row)
    
    return {
        'filepath': str(filepath),
        'n_columns': len(headers),
        'headers': headers,
        'sample_rows': sample_rows,
    }


def discover_json_structure(filepath: Path, n_entries: int = 3) -> Dict:
    """Analyze JSON structure"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return {
            'type': 'array',
            'length': len(data),
            'sample': data[:n_entries] if len(data) > 0 else [],
            'first_entry_keys': list(data[0].keys()) if len(data) > 0 and isinstance(data[0], dict) else None
        }
    elif isinstance(data, dict):
        return {
            'type': 'object',
            'keys': list(data.keys()),
            'sample_values': {k: str(v)[:100] for k, v in list(data.items())[:5]}
        }
    return {'type': str(type(data))}


def discover_data_structure():
    """Discover and report the structure of available data files"""
    print("=" * 60)
    print("GAMES2P2K DATA DISCOVERY")
    print("=" * 60)
    
    # Find CSV files
    csv_files = list(Path(".").glob("**/*.csv"))[:10]  # Limit search
    json_files = list(Path(".").glob("**/*.json"))[:10]
    
    print(f"\nFound {len(csv_files)} CSV files, {len(json_files)} JSON files\n")
    
    for f in csv_files:
        print(f"\n--- {f} ---")
        try:
            info = discover_csv_structure(f)
            print(f"Columns ({info['n_columns']}): {info['headers']}")
            if info['sample_rows']:
                print(f"Sample row: {info['sample_rows'][0][:8]}...")
        except Exception as e:
            print(f"Error reading: {e}")
    
    for f in json_files:
        print(f"\n--- {f} ---")
        try:
            info = discover_json_structure(f)
            print(f"Type: {info['type']}")
            if info['type'] == 'array':
                print(f"Length: {info['length']}")
                if info['first_entry_keys']:
                    print(f"Entry keys: {info['first_entry_keys']}")
        except Exception as e:
            print(f"Error reading: {e}")


def parse_games2p2k_csv(filepath: Path) -> List[Game2p2kEntry]:
    """
    Parse the games2p2k CSV file.
    
    Expected columns (adjust based on actual structure):
    - game_id or index
    - payoffs (8 values, may be in separate columns or single column)
    - player_role
    - choice data
    """
    entries = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        print(f"Headers found: {headers}")
        
        # Try to identify payoff columns
        payoff_cols = []
        for h in headers:
            if h in ['a', 'b', 'c', 'd', 'x', 'y', 'z', 'w']:
                payoff_cols.append(h)
            elif h.startswith('payoff') or h.startswith('p'):
                payoff_cols.append(h)
        
        # Identify game ID column
        id_col = None
        for h in ['game_id', 'gameId', 'game', 'id', 'index']:
            if h in headers:
                id_col = h
                break
        
        # Identify role column
        role_col = None
        for h in ['role', 'player_role', 'playerRole']:
            if h in headers:
                role_col = h
                break
        
        print(f"Identified - ID: {id_col}, Payoffs: {payoff_cols}, Role: {role_col}")
        
        for i, row in enumerate(reader):
            try:
                # Extract game ID
                game_id = row.get(id_col, str(i)) if id_col else str(i)
                
                # Extract payoffs
                if len(payoff_cols) >= 8:
                    payoffs = [float(row[c]) for c in payoff_cols[:8]]
                elif 'payoffs' in headers:
                    # Payoffs might be in a single column as JSON or comma-separated
                    payoff_str = row['payoffs']
                    if payoff_str.startswith('['):
                        payoffs = json.loads(payoff_str)
                    else:
                        payoffs = [float(x) for x in payoff_str.split(',')]
                else:
                    # Try to find 8 numeric columns
                    numeric_vals = []
                    for h in headers:
                        try:
                            v = float(row[h])
                            numeric_vals.append(v)
                        except (ValueError, TypeError):
                            pass
                    if len(numeric_vals) >= 8:
                        payoffs = numeric_vals[:8]
                    else:
                        print(f"Warning: Could not extract payoffs for row {i}")
                        continue
                
                # Extract role
                role = row.get(role_col, 'row') if role_col else 'row'
                if role in ['0', '1']:
                    role = 'row' if role == '0' else 'col'
                
                # Create game matrix and classify
                game = GameMatrix.from_vector(payoffs)
                rg_type = classify_robinson_goforth(game, role)
                gti_type = classify_to_gti(game, role)
                
                # Extract choice frequency if available
                choice_freq = None
                for h in ['choice_freq', 'prop_A', 'freq_A', 'pA']:
                    if h in headers:
                        try:
                            choice_freq = float(row[h])
                        except (ValueError, TypeError):
                            pass
                        break
                
                entry = Game2p2kEntry(
                    game_id=game_id,
                    payoffs=payoffs,
                    player_role=role,
                    rg_type=rg_type,
                    gti_type=gti_type,
                    choice_freq_A=choice_freq,
                )
                entries.append(entry)
                
            except Exception as e:
                print(f"Error parsing row {i}: {e}")
                continue
    
    return entries


def create_gti_corpus(entries: List[Game2p2kEntry], output_path: Path):
    """
    Create a GTI-compatible YAML corpus from games2p2k entries.
    
    Format matches gti_corpus_55.yaml structure.
    """
    cases = []
    
    for entry in entries:
        if entry.gti_type is None:
            continue  # Skip unclassifiable games
        
        game = GameMatrix.from_vector(entry.payoffs)
        
        # Create natural language description
        description = f"""A 2x2 strategic game where:
- If both choose option 1: Player 1 gets {game.a}, Player 2 gets {game.x}
- If Player 1 chooses 1, Player 2 chooses 2: Player 1 gets {game.b}, Player 2 gets {game.y}  
- If Player 1 chooses 2, Player 2 chooses 1: Player 1 gets {game.c}, Player 2 gets {game.z}
- If both choose option 2: Player 1 gets {game.d}, Player 2 gets {game.w}

Payoff matrix (Row, Col):
       C          D
A  ({game.a},{game.x})   ({game.b},{game.y})
B  ({game.c},{game.z})   ({game.d},{game.w})
"""
        
        # Determine difficulty based on game properties
        props = analyze_game_properties(game)
        if props['dominant_row'] or props['dominant_col']:
            difficulty = 'easy'
        elif len(props['nash_equilibria']) == 1:
            difficulty = 'medium'
        else:
            difficulty = 'hard'
        
        case = {
            'case_id': f"G2P2K-{entry.game_id}",
            'difficulty': difficulty,
            'expected': entry.gti_type,
            'rg_type': entry.rg_type,
            'payoffs': entry.payoffs,
            'description': description,
        }
        
        if entry.choice_freq_A is not None:
            case['human_choice_freq_A'] = entry.choice_freq_A
        
        cases.append(case)
    
    corpus = {
        'metadata': {
            'source': 'games2p2k',
            'paper': 'Zhu, Peterson, Enke & Griffiths (2024)',
            'n_games': len(cases),
            'taxonomy': 'Robinson-Goforth mapped to GTI',
        },
        'cases': cases,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(corpus, f, default_flow_style=False, allow_unicode=True, width=120)
    
    print(f"Created corpus with {len(cases)} games at {output_path}")
    return corpus


def show_statistics(entries: List[Game2p2kEntry]):
    """Display statistics about the dataset"""
    print("\n" + "=" * 60)
    print("GAMES2P2K DATASET STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal entries: {len(entries)}")
    
    # R-G type distribution
    rg_counts = Counter(e.rg_type for e in entries if e.rg_type)
    print(f"\nRobinson-Goforth Types ({len(rg_counts)} unique):")
    for rg_type, count in rg_counts.most_common():
        gti = RG_TO_GTI.get(rg_type, "?")
        print(f"  {rg_type:15} -> {gti:25} : {count:5} ({100*count/len(entries):5.1f}%)")
    
    # GTI type distribution
    gti_counts = Counter(e.gti_type for e in entries if e.gti_type)
    print(f"\nGTI Types ({len(gti_counts)} unique):")
    for gti_type, count in gti_counts.most_common():
        print(f"  {gti_type:25} : {count:5} ({100*count/len(entries):5.1f}%)")
    
    # Unclassified
    unclassified = sum(1 for e in entries if e.rg_type is None)
    if unclassified > 0:
        print(f"\nUnclassified (ties in payoffs): {unclassified}")
    
    # Role distribution
    role_counts = Counter(e.player_role for e in entries)
    print(f"\nPlayer roles: {dict(role_counts)}")


def validate_with_claude(entries: List[Game2p2kEntry], n_samples: int = 100):
    """
    Validate Claude classifier against labeled games2p2k data.
    
    This creates a validation script that can be run separately.
    """
    # Sample games for validation
    valid_entries = [e for e in entries if e.gti_type is not None]
    if len(valid_entries) < n_samples:
        n_samples = len(valid_entries)
    
    sample = random.sample(valid_entries, n_samples)
    
    # Create validation cases file
    cases = []
    for e in sample:
        game = GameMatrix.from_vector(e.payoffs)
        cases.append({
            'case_id': f"G2P2K-{e.game_id}",
            'expected': e.gti_type,
            'rg_type': e.rg_type,
            'description': f"""Payoff matrix for a 2-player strategic game:
       C          D
A  ({game.a},{game.x})   ({game.b},{game.y})
B  ({game.c},{game.z})   ({game.d},{game.w})

Where the first number is Player 1's payoff and second is Player 2's.
What type of game is this?""",
        })
    
    output = {
        'metadata': {
            'source': 'games2p2k validation sample',
            'n_samples': n_samples,
        },
        'cases': cases,
    }
    
    val_path = Path("gti_validation_sample.yaml")
    with open(val_path, 'w') as f:
        yaml.dump(output, f, default_flow_style=False)
    
    print(f"\nCreated validation sample: {val_path}")
    print(f"Contains {n_samples} labeled games for classifier validation")
    print("\nType distribution in sample:")
    sample_counts = Counter(e.gti_type for e in sample)
    for gti_type, count in sample_counts.most_common():
        print(f"  {gti_type:25} : {count:3}")


def main():
    parser = argparse.ArgumentParser(description="Games2p2k Integration for GTI")
    parser.add_argument('--discover', action='store_true', help='Discover data structure')
    parser.add_argument('--convert', action='store_true', help='Convert to GTI corpus')
    parser.add_argument('--validate', type=int, metavar='N', help='Create validation sample of N games')
    parser.add_argument('--stats', action='store_true', help='Show dataset statistics')
    parser.add_argument('--data', type=str, default=None, help='Path to data file')
    parser.add_argument('--output', type=str, default='gti_corpus_games2p2k.yaml', help='Output corpus path')
    
    args = parser.parse_args()
    
    if args.discover:
        discover_data_structure()
        return
    
    # Find data file
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = find_data_file()
    
    if data_path is None or not data_path.exists():
        print("Error: Could not find games2p2k data file")
        print("Please specify with --data <path>")
        print("\nSearched for:")
        for p in GAMES2P2K_PATTERNS:
            print(f"  {p}")
        return
    
    print(f"Using data file: {data_path}")
    
    # Parse the data
    entries = parse_games2p2k_csv(data_path)
    print(f"Parsed {len(entries)} entries")
    
    if args.stats:
        show_statistics(entries)
    
    if args.convert:
        output_path = Path(args.output)
        create_gti_corpus(entries, output_path)
    
    if args.validate:
        validate_with_claude(entries, args.validate)
    
    # If no specific action, show stats
    if not any([args.discover, args.convert, args.validate, args.stats]):
        show_statistics(entries)


if __name__ == "__main__":
    main()
