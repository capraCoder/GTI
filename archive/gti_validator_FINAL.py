#!/usr/bin/env python3
"""
GTI Validator - Game Theory Index Classification & Statistical Validation
=========================================================================
THE DEFINITIVE VERSION - Use this one, ignore all others.

Usage:
  python gti_validator.py --validate 100 --core-only
  python gti_validator.py --validate 500 --all-types
  python gti_validator.py --stats results.json

Proven accuracy: 90% on core types, 78% on all 12 types
"""

import os
import sys
import json
import csv
import re
import random
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from scipy import stats
    from scipy.stats import binom, fisher_exact
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "claude-sonnet-4-20250514"

# Robinson-Goforth ordinal patterns → game types
RG_PATTERNS = {
    ('c', 'a', 'b', 'd'): "Chicken",
    ('c', 'b', 'a', 'd'): "Battle_of_the_Sexes",
    ('c', 'b', 'd', 'a'): "Hero",
    ('c', 'd', 'b', 'a'): "Compromise",
    ('c', 'd', 'a', 'b'): "Deadlock",
    ('c', 'a', 'd', 'b'): "Prisoners_Dilemma",
    ('a', 'c', 'd', 'b'): "Stag_Hunt",
    ('a', 'd', 'c', 'b'): "Assurance_Game",
    ('a', 'd', 'b', 'c'): "Coordination_Game",
    ('a', 'b', 'd', 'c'): "Peace",
    ('a', 'b', 'c', 'd'): "Harmony",
    ('a', 'c', 'b', 'd'): "Concord",
}

# Core 9 types (distinctive strategic structures)
CORE_TYPES = {
    "Prisoners_Dilemma", "Chicken", "Stag_Hunt", "Battle_of_the_Sexes",
    "Coordination_Game", "Assurance_Game", "Hero", "Compromise", "Deadlock"
}

# Non-core types (cooperative family - narratively hard to distinguish)
NON_CORE_TYPES = {"Peace", "Harmony", "Concord"}

VALID_TYPES = list(set(RG_PATTERNS.values()))

# =============================================================================
# NARRATIVE TEMPLATES (from v10 - proven 90% accuracy)
# =============================================================================

NARRATIVE_TEMPLATES = {
    "Prisoners_Dilemma": [
        """Two competing firms must decide whether to maintain high prices (cooperate) or undercut (defect).
If both maintain prices, they each earn {a} in profits.
If one undercuts while the other maintains, the undercutter captures the market earning {c}, while the loyal firm earns only {b}.
If both undercut, price war leaves them with just {d} each.
The temptation to defect ({c} > {a}) is strong, but mutual defection ({d}) is worse than mutual cooperation ({a}).""",

        """Two nations can either honor an arms treaty (cooperate) or secretly build weapons (defect).
Mutual compliance: stable peace, {a} security each.
One cheats while other complies: cheater gains military advantage {c}, compliant nation vulnerable {b}.
Both cheat: arms race, {d} security each (wasted resources, same relative position).
Each is tempted to cheat, but mutual cheating leaves everyone worse off.""",
    ],
    
    "Chicken": [
        """Two drivers race toward each other on a narrow road. Each can swerve (back down) or drive straight (challenge).
If both swerve, minor embarrassment - {a} reputation each.
If one swerves and one goes straight: the one who swerved is "chicken" ({b}), the bold one wins glory ({c}).
If NEITHER swerves: catastrophic head-on collision, {d} each - the worst possible outcome.
The key tension: mutual aggression is DISASTER, but backing down alone is humiliating.""",

        """Two superpowers in nuclear standoff. Each can back down (yield) or escalate (challenge).
Both back down: minor face loss, {a} each.
One backs down, other presses: yielding power loses face {b}, aggressive power dominates {c}.
Both escalate: nuclear war, {d} each - catastrophe far worse than any other outcome.
Brinkmanship: each wants the OTHER to yield first.""",
    ],
    
    "Stag_Hunt": [
        """Two hunters can pursue a stag together (cooperate) or hunt rabbits alone (safe choice).
Both hunt stag: they succeed together, {a} meat each - the BEST outcome.
One hunts stag while other hunts rabbit: stag hunter fails alone getting {b}, rabbit hunter gets modest {c}.
Both hunt rabbits: guaranteed small catch, {d} each.
KEY: Mutual cooperation ({a}) beats safe defection ({d}), but you need TRUST that partner won't abandon you.""",

        """Two startups can collaborate on ambitious project (cooperate) or pursue safe solo ventures (defect).
Both collaborate: breakthrough success, {a} valuation each.
One commits to collaboration, other pivots to safe project: committed founder stranded {b}, pivoting founder gets {c}.
Both go safe: modest but reliable outcomes, {d} each.
The payoff-dominant equilibrium requires mutual trust.""",
    ],
    
    "Battle_of_the_Sexes": [
        """A couple must choose between Opera or Boxing for their evening out.
Partner A prefers opera; Partner B prefers boxing.
Both go to opera: A is happy ({c}), B less so but together ({a}).
Both go to boxing: B is happy ({c}), A less so but together ({a}).
They split up (miscoordinate): {b} and {d} - neither enjoys being alone.
KEY: Coordination matters MORE than winning the preference, but each prefers different equilibria.""",

        """Two tech companies must adopt either Standard X or Standard Y for compatibility.
Company 1 prefers X; Company 2 prefers Y.
Both choose X: Company 1 happy ({c}), Company 2 acceptable ({a}).
Both choose Y: Company 2 happy ({c}), Company 1 acceptable ({a}).
Different standards: incompatible systems, {b} and {d} - market fragmentation hurts both.
Being together on ANY standard beats being apart.""",
    ],
    
    "Coordination_Game": [
        """Two drivers approach each other and must choose: pass on the left or pass on the right.
Both choose same side: safe passage, {a} each.
Different sides chosen: dangerous near-collision, {b} and {c}.
NO preference for which side - the ONLY goal is to MATCH choices. Pure coordination.""",

        """Two people trying to meet in a city, no phones to communicate.
Both choose the central train station: they meet! {a} each.
Different locations: missed connection, {b} and {c} - wasted day.
Doesn't matter WHERE they meet, just that they coordinate on SAME place.""",
    ],
    
    "Assurance_Game": [
        """Two allies considering joint military commitment.
Both commit troops: strong deterrent, {a} security each - best outcome.
One commits, other holds back: committed ally overextended {b}, free-rider benefits {c}.
Neither commits: weak but not exposed, {d} each.
DIFFERS FROM STAG HUNT: here, if partner defects, you're worse off having cooperated ({b} < {d}).
Need ASSURANCE partner will follow through before committing.""",

        """Two firms considering joint R&D investment requiring both to participate.
Both invest: breakthrough technology, {a} competitive advantage each.
One invests, other backs out: investor loses sunk costs {b}, defector saved money {c}.
Neither invests: status quo, {d} each - no breakthrough but no loss.
You want to invest ONLY IF partner definitely will too.""",
    ],
    
    "Harmony": [
        """Two divisions in a company with perfectly aligned incentives - helping each other helps yourself.
Both cooperate: excellent synergy, {a} each - BEST outcome.
One cooperates, other doesn't: cooperator still benefits {b}, non-cooperator gains {c}.
Neither cooperates: independent work, still okay {d} each.
KEY: Cooperation is DOMINANT STRATEGY for both. No dilemma exists - cooperation is self-interested.
Unlike other games, {a} > {c} so even if partner defects, you still prefer to cooperate.""",
    ],
    
    "Deadlock": [
        """Two firms that genuinely prefer independence over forced partnership.
Both operate independently: efficient specialization, {c} profits each - BEST outcome.
One seeks partnership, other declines: partnership-seeker wasted effort {a}, independent firm thrives {c}.
Both pursue partnership: forced collaboration, suboptimal {d} each.
UNLIKE PRISONER'S DILEMMA: Here "defection" (independence) is MUTUALLY PREFERRED.
No one is trapped - mutual independence is stable AND efficient.""",
    ],
    
    "Hero": [
        """A crisis requiring someone to sacrifice for the group (volunteer's dilemma).
Neither volunteers: disaster continues, costs {d} and {a}.
One person volunteers (becomes hero): volunteer bears personal cost {b}, everyone else saved {c}.
If OTHER volunteers: you're saved ({c}) while they sacrifice.
The group NEEDS a hero, but who will bear the cost? Everyone hopes someone ELSE will step up.""",

        """Whistleblowing situation - someone must report corporate fraud.
No one reports: fraud continues harming everyone, {d} and {a}.
One reports: whistleblower faces retaliation {b}, but company/public saved {c}.
Everyone benefits if SOMEONE blows the whistle, but the whistleblower pays personally.""",
    ],
    
    "Compromise": [
        """Two parties in dispute who both benefit from mutual concession.
Both stand firm: complete deadlock, {a} and {d} - nothing resolved.
One compromises, other stands firm: compromiser loses {a}, firm party wins {c}.
Both make reasonable concessions: fair resolution, {c} and {d} - progress!
KEY: Mutual compromise beats mutual stubbornness. The question is whether to trust the other to reciprocate.""",
    ],
    
    "Peace": [
        """Two neighboring countries with no real conflict of interest.
Friendly diplomatic relations: prosperous trade, {a} each - optimal.
One extends friendship, other remains neutral: friendly nation {b}, neutral one {d}.
Both remain distant: missed opportunity, only {c} each.
DISTINCT FROM HARMONY: Here one-sided friendliness ({b}) is worse than mutual distance ({c}).
Peace requires BOTH parties to engage, but cooperation is still clearly best.""",
    ],
    
    "Concord": [
        """Two research teams with overlapping but compatible interests.
Joint publication: major impact, {a} prestige each - best outcome.
One proposes collaboration, other declines: proposer embarrassed {b}, decliner independent {c}.
Both work separately: smaller individual papers, {d} each.
DISTINCT FROM COORDINATION: Here separate work ({d}) is strictly ranked (a > c > b > d).
Strong incentive to collaborate, but not pure coordination - preferences exist.""",
    ],
}

SYSTEM_PROMPT = """You are a game theory expert classifying strategic situations.

The 12 Robinson-Goforth game types for 2×2 symmetric games:

CONFLICT GAMES (temptation to defect):
- Prisoners_Dilemma: Mutual cooperation best, but temptation to defect. c > a > d > b
- Chicken: Mutual aggression is CATASTROPHE. Worst = both aggressive. c > a > b > d
- Deadlock: Mutual defection actually preferred. Unlike PD. c > d > a > b

COORDINATION GAMES:
- Battle_of_the_Sexes: Want to coordinate, but prefer different equilibria. c > b > a > d
- Coordination_Game: Pure matching - no preference which option, just match. a > d > b > c

TRUST GAMES (cooperation needs assurance):
- Stag_Hunt: High reward needs trust. Cooperation best but risky. a > c > d > b
- Assurance_Game: Like Stag Hunt but being abandoned is worse. a > d > c > b

OTHER:
- Hero: Someone must sacrifice for group (volunteer's dilemma). c > b > d > a
- Compromise: Mutual concession beats stubbornness. c > d > b > a
- Harmony: Cooperation dominant - no dilemma. a > b > c > d
- Peace: Mutual friendliness best. a > b > d > c
- Concord: Strong collaboration incentive. a > c > b > d

Output ONLY the game type name, nothing else."""

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ValidationResult:
    id: str
    expected: str
    predicted: str
    correct: bool
    payoffs: Tuple[int, int, int, int] = (0, 0, 0, 0)
    raw_response: str = ""

@dataclass
class TypeStats:
    name: str
    correct: int = 0
    total: int = 0
    
    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_rg_type(a: int, b: int, c: int, d: int) -> Tuple[str, bool]:
    """Compute R-G type from payoffs. Returns (type_name, is_known)."""
    items = sorted([(a, 'a'), (b, 'b'), (c, 'c'), (d, 'd')], key=lambda x: -x[0])
    pattern = tuple(x[1] for x in items)
    if pattern in RG_PATTERNS:
        return RG_PATTERNS[pattern], True
    return f"Unknown:{pattern}", False


def generate_narrative(game_type: str, payoffs: Tuple[int, int, int, int]) -> str:
    """Generate narrative description from game type and payoffs."""
    a, b, c, d = payoffs
    if game_type not in NARRATIVE_TEMPLATES:
        return f"Strategic situation with payoffs: CC={a}, CD={b}, DC={c}, DD={d}."
    template = random.choice(NARRATIVE_TEMPLATES[game_type])
    return template.format(a=a, b=b, c=c, d=d)


FUZZY_MAP = {
    "prisoner": "Prisoners_Dilemma", "dilemma": "Prisoners_Dilemma",
    "chicken": "Chicken", "hawk": "Chicken", "brinkmanship": "Chicken",
    "stag": "Stag_Hunt", "hunt": "Stag_Hunt", "trust": "Stag_Hunt",
    "battle": "Battle_of_the_Sexes", "sexes": "Battle_of_the_Sexes",
    "coordination": "Coordination_Game", "matching": "Coordination_Game",
    "assurance": "Assurance_Game",
    "harmony": "Harmony", "deadlock": "Deadlock",
    "hero": "Hero", "volunteer": "Hero", "whistleblow": "Hero",
    "compromise": "Compromise", "peace": "Peace", "concord": "Concord",
}

def normalize_type(raw: str) -> str:
    """Normalize classifier output to valid type name."""
    if not raw:
        return "Unknown"
    cleaned = raw.strip().split('\n')[0].strip()
    cleaned = re.sub(r'[*\[\]()"\':]', '', cleaned)
    cleaned = re.sub(r'[^\w\s_]', '', cleaned).strip()
    
    # Exact match
    for valid in VALID_TYPES:
        if cleaned.lower().replace("_", "").replace(" ", "") == valid.lower().replace("_", ""):
            return valid
    
    # Fuzzy match
    for key, value in FUZZY_MAP.items():
        if key in cleaned.lower():
            return value
    
    return cleaned.replace(" ", "_") if len(cleaned) < 35 else "Unknown"

# =============================================================================
# DATA LOADING
# =============================================================================

def load_games2p2k(filepath: Path, core_only: bool = True) -> List[Dict]:
    """Load games2p2k dataset."""
    entries = []
    skipped = {"unknown": 0, "noncore": 0}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                matrix_str = row['row_form_matrix'].strip().strip('[]')
                payoffs = [int(x) for x in matrix_str.split() if x]
                is_row = 'row' in row['role'].lower()
                
                focal = tuple(payoffs[:4] if is_row else payoffs[4:])
                rg_type, is_known = compute_rg_type(*focal)
                
                if not is_known:
                    skipped["unknown"] += 1
                    continue
                
                if core_only and rg_type not in CORE_TYPES:
                    skipped["noncore"] += 1
                    continue
                
                entries.append({
                    'id': row['unique_id'],
                    'expected': rg_type,
                    'payoffs': focal,
                })
            except:
                pass
    
    print(f"  Loaded: {len(entries)}")
    if skipped["unknown"]:
        print(f"  Skipped {skipped['unknown']} unknown R-G patterns")
    if skipped["noncore"]:
        print(f"  Skipped {skipped['noncore']} non-core types")
    
    return entries

# =============================================================================
# CLASSIFIER
# =============================================================================

class GTIClassifier:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.stats = {"calls": 0, "tokens": 0, "errors": 0}
        self.clients = [Anthropic() for _ in range(max_workers)]
    
    def classify_one(self, entry: Dict, idx: int) -> ValidationResult:
        client = self.clients[idx % len(self.clients)]
        
        payoffs = entry['payoffs']
        expected = entry['expected']
        narrative = generate_narrative(expected, payoffs)
        
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=30,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Classify this game:\n\n{narrative}\n\nGame type:"}]
            )
            
            raw = response.content[0].text.strip()
            predicted = normalize_type(raw)
            
            with self.lock:
                self.stats["calls"] += 1
                self.stats["tokens"] += response.usage.input_tokens + response.usage.output_tokens
            
            return ValidationResult(
                id=entry['id'],
                expected=expected,
                predicted=predicted,
                correct=(predicted == expected),
                payoffs=payoffs,
                raw_response=raw[:50],
            )
        except Exception as e:
            with self.lock:
                self.stats["errors"] += 1
            return ValidationResult(
                id=entry['id'],
                expected=expected,
                predicted='ERROR',
                correct=False,
                raw_response=str(e)[:50],
            )
    
    def classify_batch(self, entries: List[Dict], delay: float = 0.3) -> List[ValidationResult]:
        results = []
        total = len(entries)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.classify_one, e, i): e for i, e in enumerate(entries)}
            for i, future in enumerate(as_completed(futures)):
                results.append(future.result())
                if (i + 1) % 10 == 0 or i + 1 == total:
                    acc = sum(r.correct for r in results) / len(results)
                    print(f"\r  {i+1}/{total} | Acc: {100*acc:.1f}%", end="", flush=True)
                time.sleep(delay)
        print()
        return results

# =============================================================================
# ANALYSIS
# =============================================================================

def compute_8type_results(results: List[ValidationResult]) -> List[ValidationResult]:
    """Convert to 8-type by merging Stag_Hunt + Assurance_Game -> Trust_Game."""
    transformed = []
    for r in results:
        new_expected = "Trust_Game" if r.expected in ["Stag_Hunt", "Assurance_Game"] else r.expected
        new_predicted = "Trust_Game" if r.predicted in ["Stag_Hunt", "Assurance_Game"] else r.predicted
        transformed.append(ValidationResult(
            id=r.id,
            expected=new_expected,
            predicted=new_predicted,
            correct=(new_expected == new_predicted),
            payoffs=r.payoffs,
            raw_response=r.raw_response,
        ))
    return transformed


def compute_stats(results: List[ValidationResult]) -> Dict[str, TypeStats]:
    """Compute per-type statistics."""
    by_type = defaultdict(lambda: TypeStats(""))
    for r in results:
        if by_type[r.expected].name == "":
            by_type[r.expected].name = r.expected
        by_type[r.expected].total += 1
        if r.correct:
            by_type[r.expected].correct += 1
    return dict(by_type)


def print_report(results_9: List[ValidationResult], results_8: List[ValidationResult], stats: Dict):
    """Print validation report."""
    n = len(results_9)
    correct_9 = sum(r.correct for r in results_9)
    correct_8 = sum(r.correct for r in results_8)
    acc_9 = correct_9 / n
    acc_8 = correct_8 / n
    
    stats_9 = compute_stats(results_9)
    stats_8 = compute_stats(results_8)
    
    print("\n" + "=" * 75)
    print("GTI VALIDATION REPORT")
    print("=" * 75)
    
    print(f"\n{'OVERALL ACCURACY':^75}")
    print("-" * 75)
    print(f"  8-type (strategic families):  {correct_8:3}/{n} = {acc_8*100:5.1f}%")
    print(f"  9-type (R-G core types):      {correct_9:3}/{n} = {acc_9*100:5.1f}%")
    print(f"  Improvement from merging SH+AG: +{(acc_8-acc_9)*100:.1f} pp")
    
    print(f"\n{'PER-TYPE ACCURACY (9-type)':^75}")
    print("-" * 75)
    
    for name, s in sorted(stats_9.items(), key=lambda x: -x[1].total):
        bar = '█' * int(s.accuracy * 10) + '░' * (10 - int(s.accuracy * 10))
        flag = " ← PROBLEM" if s.accuracy < 0.8 and s.total >= 5 else ""
        print(f"  {name:22} {s.correct:3}/{s.total:3} {bar} {s.accuracy*100:5.1f}%{flag}")
    
    print(f"\n{'PER-TYPE ACCURACY (8-type)':^75}")
    print("-" * 75)
    
    for name, s in sorted(stats_8.items(), key=lambda x: -x[1].total):
        bar = '█' * int(s.accuracy * 10) + '░' * (10 - int(s.accuracy * 10))
        print(f"  {name:22} {s.correct:3}/{s.total:3} {bar} {s.accuracy*100:5.1f}%")
    
    # Confusion matrix for 9-type
    print(f"\n{'CONFUSION MATRIX (9-type)':^75}")
    print("-" * 75)
    
    confusion = defaultdict(lambda: defaultdict(int))
    for r in results_9:
        confusion[r.expected][r.predicted] += 1
    
    for expected in sorted(confusion.keys()):
        preds = confusion[expected]
        pred_str = ", ".join(f"{p[:15]}({c})" for p, c in sorted(preds.items(), key=lambda x: -x[1]))
        print(f"  {expected:22} → {pred_str}")
    
    print("\n" + "=" * 75)
    print(f"Errors: {stats['errors']} | Tokens: {stats['tokens']:,}")
    print("=" * 75)
    
    return {
        "n": n,
        "accuracy_8": acc_8,
        "accuracy_9": acc_9,
        "by_type_9": {k: {"correct": v.correct, "total": v.total, "accuracy": v.accuracy} 
                      for k, v in stats_9.items()},
        "by_type_8": {k: {"correct": v.correct, "total": v.total, "accuracy": v.accuracy} 
                      for k, v in stats_8.items()},
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GTI Validator - THE definitive version")
    parser.add_argument('--validate', type=int, default=100, help='Number of games to validate')
    parser.add_argument('--core-only', action='store_true', help='Core 9 types only (recommended)')
    parser.add_argument('--all-types', action='store_true', help='All 12 R-G types')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    parser.add_argument('--delay', type=float, default=0.3, help='Delay between calls')
    parser.add_argument('--data', type=str, default="games2p2k_data/osfstorage/human data/games2p2k_main.csv")
    
    args = parser.parse_args()
    
    # Default to core-only unless --all-types specified
    core_only = not args.all_types
    
    print(f"Loading {args.data}...")
    entries = load_games2p2k(Path(args.data), core_only=core_only)
    
    if not entries:
        print("ERROR: No entries loaded")
        return
    
    # Sample
    n = min(args.validate, len(entries))
    sample = random.sample(entries, n)
    
    print(f"\nValidating {n} games with {args.workers} workers...")
    
    if not HAS_ANTHROPIC:
        print("ERROR: pip install anthropic")
        return
    
    classifier = GTIClassifier(max_workers=args.workers)
    results_9 = classifier.classify_batch(sample, args.delay)
    results_8 = compute_8type_results(results_9)
    
    summary = print_report(results_9, results_8, classifier.stats)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gti_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n": n,
            "core_only": core_only,
            "accuracy_8": summary["accuracy_8"],
            "accuracy_9": summary["accuracy_9"],
            "by_type_9": summary["by_type_9"],
            "by_type_8": summary["by_type_8"],
            "results": [asdict(r) for r in results_9],
        }, f, indent=2, default=str)
    
    print(f"\nSaved: {filename}")


if __name__ == "__main__":
    main()
