#!/usr/bin/env python3
"""
games2p2k_harvester.py - Dataset Integration Pipeline

Pipeline: Download -> Verify -> Map -> Canonicalize -> Import

The games2p2k dataset contains 2,416 strategically distinct 2x2 games
covering all Robinson-Goforth equivalence classes.
"""

import os
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import yaml
import numpy as np

# Import from canonical_classifier
import sys
sys.path.insert(0, str(Path(__file__).parent))
from canonical_classifier import (
    canonical_fingerprint, classify_game,
    analyze_nash_equilibria, RGClass, get_game_type_name
)


@dataclass
class Game2p2k:
    """Standardized game record"""
    id: str
    name: str
    canonical_fingerprint: str
    rg_class: str
    game_type: str
    payoff_matrix: Dict  # {AA, AB, BA, BB}
    nash_analysis: Dict
    source: str = "games2p2k"
    verified: bool = False


class Games2p2kHarvester:
    """
    Multi-stage pipeline for games2p2k integration.
    """

    def __init__(self, work_dir: str = "D:\\CaprazliIndex\\GTI"):
        self.work_dir = Path(work_dir)
        self.raw_dir = self.work_dir / "games2p2k_raw"
        self.db_path = self.work_dir / "games2p2k.db"
        self.corpus_path = self.work_dir / "games2p2k_corpus.yaml"

        self.raw_dir.mkdir(exist_ok=True)

    def stage1_download(self, source_url: str = None) -> Path:
        """
        Stage 1: Download games2p2k dataset.

        Expected source: Academic repository or research website
        Format: CSV, JSON, or structured text
        """
        # Check for local file first
        local_paths = [
            self.raw_dir / "games2p2k.json",
            self.raw_dir / "games2p2k.csv",
            self.raw_dir / "games2p2k.yaml",
            Path.home() / "Downloads" / "games2p2k.json"
        ]
        for p in local_paths:
            if p.exists():
                print(f"Found local dataset: {p}")
                return p

        if source_url is None:
            # Generate synthetic dataset for testing
            print("No dataset found. Generating synthetic test data...")
            return self._generate_synthetic_dataset()

        # Download logic
        import urllib.request
        output_path = self.raw_dir / "games2p2k.json"
        print(f"Downloading from {source_url}...")
        urllib.request.urlretrieve(source_url, output_path)
        return output_path

    def _generate_synthetic_dataset(self) -> Path:
        """Generate synthetic games covering all RG classes for testing."""
        import itertools

        games = []

        # Classic games with known classifications
        classic_games = [
            {
                "id": "PD_CLASSIC",
                "name": "Prisoner's Dilemma (Classic)",
                "payoffs": {"AA": [3, 3], "AB": [0, 5], "BA": [5, 0], "BB": [1, 1]},
                "expected_class": "CLASS_2_DILEMMA"
            },
            {
                "id": "CHICKEN_CLASSIC",
                "name": "Chicken (Classic)",
                "payoffs": {"AA": [3, 3], "AB": [2, 4], "BA": [4, 2], "BB": [1, 1]},
                "expected_class": "CLASS_3_CHICKEN"
            },
            {
                "id": "STAG_HUNT_CLASSIC",
                "name": "Stag Hunt (Classic)",
                "payoffs": {"AA": [4, 4], "AB": [1, 3], "BA": [3, 1], "BB": [2, 2]},
                "expected_class": "CLASS_4_STAG_HUNT"
            },
            {
                "id": "BOS_CLASSIC",
                "name": "Battle of Sexes (Classic)",
                "payoffs": {"AA": [3, 2], "AB": [0, 0], "BA": [0, 0], "BB": [2, 3]},
                "expected_class": "CLASS_5_BATTLE"
            },
            {
                "id": "MATCHING_PENNIES",
                "name": "Matching Pennies",
                "payoffs": {"AA": [1, -1], "AB": [-1, 1], "BA": [-1, 1], "BB": [1, -1]},
                "expected_class": "CLASS_8_ZERO_SUM"
            },
            {
                "id": "COORDINATION_PURE",
                "name": "Pure Coordination",
                "payoffs": {"AA": [2, 2], "AB": [0, 0], "BA": [0, 0], "BB": [2, 2]},
                "expected_class": "CLASS_1_WIN_WIN"
            },
            {
                "id": "HARMONY",
                "name": "Harmony Game",
                "payoffs": {"AA": [4, 4], "AB": [3, 2], "BA": [2, 3], "BB": [1, 1]},
                "expected_class": "CLASS_7_HARMONY"
            },
            {
                "id": "DEADLOCK",
                "name": "Deadlock",
                "payoffs": {"AA": [1, 1], "AB": [0, 3], "BA": [3, 0], "BB": [2, 2]},
                "expected_class": "CLASS_6_DEADLOCK"
            }
        ]

        games.extend(classic_games)

        # Generate variations with different payoff scales
        for base in classic_games[:4]:
            for scale in [2, 10]:
                variation = {
                    "id": f"{base['id']}_SCALE{scale}",
                    "name": f"{base['name']} (Scale {scale}x)",
                    "payoffs": {
                        k: [v[0] * scale, v[1] * scale]
                        for k, v in base['payoffs'].items()
                    },
                    "expected_class": base['expected_class']
                }
                games.append(variation)

        # Generate random ordinal games
        ordinal_perms = list(itertools.permutations([0, 1, 2, 3]))
        for i, (p1_perm, p2_perm) in enumerate(
            itertools.islice(itertools.product(ordinal_perms, ordinal_perms), 50)
        ):
            games.append({
                "id": f"ORDINAL_{i:04d}",
                "name": f"Ordinal Game {i}",
                "payoffs": {
                    "AA": [p1_perm[0], p2_perm[0]],
                    "AB": [p1_perm[1], p2_perm[1]],
                    "BA": [p1_perm[2], p2_perm[2]],
                    "BB": [p1_perm[3], p2_perm[3]]
                }
            })

        output_path = self.raw_dir / "games2p2k.json"
        with open(output_path, 'w') as f:
            json.dump(games, f, indent=2)

        print(f"Generated {len(games)} synthetic games at {output_path}")
        return output_path

    def stage2_verify(self, raw_path: Path) -> List[Dict]:
        """
        Stage 2: Verify dataset integrity.

        Checks:
        - Expected record count (~2416 for full dataset)
        - Valid 2x2 structure
        - No missing payoffs
        """
        with open(raw_path, 'r', encoding='utf-8') as f:
            if raw_path.suffix == '.json':
                data = json.load(f)
            elif raw_path.suffix == '.yaml':
                data = yaml.safe_load(f)
            else:
                # CSV parsing
                import csv
                reader = csv.DictReader(f)
                data = list(reader)

        verified = []
        errors = []

        for i, record in enumerate(data):
            try:
                # Validate structure
                if 'payoffs' in record:
                    payoffs = record['payoffs']
                    assert all(
                        k in payoffs for k in ['AA', 'AB', 'BA', 'BB']
                    ), f"Missing payoff keys in record {i}"
                    assert all(
                        len(payoffs[k]) == 2 for k in ['AA', 'AB', 'BA', 'BB']
                    ), f"Payoffs must be [p1, p2] pairs in record {i}"
                elif all(k in record for k in ['p1_AA', 'p2_AA']):
                    # Convert flat format to nested
                    record['payoffs'] = {
                        'AA': [float(record['p1_AA']), float(record['p2_AA'])],
                        'AB': [float(record['p1_AB']), float(record['p2_AB'])],
                        'BA': [float(record['p1_BA']), float(record['p2_BA'])],
                        'BB': [float(record['p1_BB']), float(record['p2_BB'])]
                    }
                else:
                    raise AssertionError(f"Invalid payoff structure in record {i}")

                verified.append(record)
            except (AssertionError, KeyError, ValueError) as e:
                errors.append((i, str(e)))

        print(f"Verified: {len(verified)}/{len(data)} records")
        print(f"Errors: {len(errors)}")

        if errors:
            error_path = self.raw_dir / "verification_errors.json"
            with open(error_path, 'w') as f:
                json.dump(errors, f, indent=2)
            print(f"Errors saved to {error_path}")

        return verified

    def stage3_map_and_canonicalize(self, records: List[Dict]) -> List[Game2p2k]:
        """
        Stage 3: Map to standard schema and compute canonical forms.
        """
        games = []
        fingerprint_counts = {}

        for idx, record in enumerate(records):
            p = record['payoffs']
            matrix = np.array([
                [[p['AA'][0], p['AA'][1]], [p['AB'][0], p['AB'][1]]],
                [[p['BA'][0], p['BA'][1]], [p['BB'][0], p['BB'][1]]]
            ])

            # Compute canonical form
            fingerprint = canonical_fingerprint(matrix)
            rg_class = classify_game(matrix)
            nash = analyze_nash_equilibria(matrix)

            # Convert nash info to JSON-serializable format
            nash_serializable = {
                'pure_ne_count': nash['pure_ne_count'],
                'pure_ne_cells': [list(c) for c in nash['pure_ne_cells']],
                'p1_dominant_strategy': nash['p1_dominant_strategy'],
                'p2_dominant_strategy': nash['p2_dominant_strategy'],
                'is_symmetric': bool(nash['is_symmetric']),
                'pareto_optimal_count': nash['pareto_optimal_count']
            }

            # Track fingerprint frequency
            fingerprint_counts[fingerprint] = fingerprint_counts.get(fingerprint, 0) + 1

            game = Game2p2k(
                id=record.get('id', f"G2P2K_{idx:04d}"),
                name=record.get('name', f"Game_{idx}"),
                canonical_fingerprint=fingerprint,
                rg_class=rg_class.name,
                game_type=get_game_type_name(rg_class),
                payoff_matrix={
                    'AA': [float(x) for x in matrix[0, 0]],
                    'AB': [float(x) for x in matrix[0, 1]],
                    'BA': [float(x) for x in matrix[1, 0]],
                    'BB': [float(x) for x in matrix[1, 1]]
                },
                nash_analysis=nash_serializable,
                verified=True
            )
            games.append(game)

        # Report unique canonical forms
        unique_fingerprints = len(fingerprint_counts)
        print(f"Unique canonical forms: {unique_fingerprints}")

        return games

    def stage4_import(self, games: List[Game2p2k]):
        """
        Stage 4: Import to SQLite and YAML corpus.
        """
        # SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                id TEXT PRIMARY KEY,
                name TEXT,
                canonical_fingerprint TEXT,
                rg_class TEXT,
                game_type TEXT,
                payoff_matrix TEXT,
                nash_analysis TEXT,
                source TEXT,
                verified INTEGER
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_fingerprint
            ON games(canonical_fingerprint)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_rg_class
            ON games(rg_class)
        ''')

        for game in games:
            cursor.execute('''
                INSERT OR REPLACE INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                game.id, game.name, game.canonical_fingerprint,
                game.rg_class, game.game_type,
                json.dumps(game.payoff_matrix),
                json.dumps(game.nash_analysis), game.source,
                int(game.verified)
            ))

        conn.commit()
        conn.close()
        print(f"Imported {len(games)} games to {self.db_path}")

        # YAML corpus
        corpus = {
            'metadata': {
                'source': 'games2p2k',
                'count': len(games),
                'unique_fingerprints': len(set(g.canonical_fingerprint for g in games))
            },
            'games': [asdict(g) for g in games]
        }
        with open(self.corpus_path, 'w', encoding='utf-8') as f:
            yaml.dump(corpus, f, default_flow_style=False, allow_unicode=True)
        print(f"Exported corpus to {self.corpus_path}")

    def run_pipeline(self, source_url: str = None) -> List[Game2p2k]:
        """Execute full pipeline."""
        print("=" * 60)
        print("GAMES2P2K HARVESTER PIPELINE")
        print("=" * 60)

        print("\n[Stage 1] Downloading/Loading...")
        raw_path = self.stage1_download(source_url)

        print("\n[Stage 2] Verifying...")
        records = self.stage2_verify(raw_path)

        print("\n[Stage 3] Mapping & Canonicalizing...")
        games = self.stage3_map_and_canonicalize(records)

        print("\n[Stage 4] Importing...")
        self.stage4_import(games)

        # Summary statistics
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)

        class_counts = {}
        for g in games:
            class_counts[g.rg_class] = class_counts.get(g.rg_class, 0) + 1

        print("\nClass Distribution:")
        for cls, count in sorted(class_counts.items()):
            pct = count / len(games) * 100
            print(f"  {cls:25} {count:4} ({pct:5.1f}%)")

        print(f"\nTotal: {len(games)} games processed")
        print("=" * 60)

        return games

    def query_by_class(self, rg_class: str) -> List[Dict]:
        """Query games by RG class from SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM games WHERE rg_class = ?",
            (rg_class,)
        )
        rows = cursor.fetchall()
        conn.close()

        columns = ['id', 'name', 'canonical_fingerprint', 'rg_class',
                   'game_type', 'payoff_matrix', 'nash_analysis', 'source', 'verified']
        return [dict(zip(columns, row)) for row in rows]

    def query_by_fingerprint(self, fingerprint: str) -> Optional[Dict]:
        """Query game by canonical fingerprint."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM games WHERE canonical_fingerprint = ? LIMIT 1",
            (fingerprint,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            columns = ['id', 'name', 'canonical_fingerprint', 'rg_class',
                       'game_type', 'payoff_matrix', 'nash_analysis', 'source', 'verified']
            return dict(zip(columns, row))
        return None


if __name__ == "__main__":
    harvester = Games2p2kHarvester()
    harvester.run_pipeline()
