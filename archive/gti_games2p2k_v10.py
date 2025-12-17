"""
GTI games2p2k v10 - Fixed Narrative Classification

Fixes from v9:
1. Filter to KNOWN R-G patterns only (12 types, not 24)
2. Better differentiation between Peace/Concord/Harmony narratives
3. Report "core types" accuracy separately
"""

import os
import sys
import json
import csv
import time
import argparse
import random
import re
import ast
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# =============================================================================
# R-G CLASSIFICATION - COMPLETE 12-TYPE SYSTEM
# =============================================================================

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

# Core types with distinctive strategic structures (easier to classify)
CORE_TYPES = {
    "Prisoners_Dilemma", "Chicken", "Stag_Hunt", "Battle_of_the_Sexes",
    "Coordination_Game", "Assurance_Game", "Hero", "Compromise", "Deadlock"
}

# Similar types that get confused
COOPERATIVE_TYPES = {"Peace", "Harmony", "Concord"}


def compute_rg_type(a: int, b: int, c: int, d: int) -> Tuple[str, bool]:
    """Returns (type_name, is_known_pattern)"""
    items = sorted([(a, 'a'), (b, 'b'), (c, 'c'), (d, 'd')], key=lambda x: -x[0])
    pattern = tuple(x[1] for x in items)
    
    if pattern in RG_PATTERNS:
        return RG_PATTERNS[pattern], True
    else:
        return f"Unknown:{pattern}", False


# =============================================================================
# IMPROVED NARRATIVE TEMPLATES
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


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class GameEntry:
    unique_id: str
    game_id: str
    payoffs: List[int]
    role: str
    rg_type: str
    focal_payoffs: Tuple[int, int, int, int]
    is_known: bool
    is_core: bool
    narrative: str = ""


def load_games(filepath: Path, known_only: bool = True, core_only: bool = False) -> List[GameEntry]:
    entries = []
    skipped_unknown = 0
    skipped_noncore = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                matrix_str = row['row_form_matrix'].strip().strip('[]')
                payoffs = [int(x) for x in matrix_str.split() if x]
                is_row = 'row' in row['role'].lower()
                
                if is_row:
                    focal = (payoffs[0], payoffs[1], payoffs[2], payoffs[3])
                else:
                    focal = (payoffs[4], payoffs[5], payoffs[6], payoffs[7])
                
                rg_type, is_known = compute_rg_type(*focal)
                is_core = rg_type in CORE_TYPES
                
                if known_only and not is_known:
                    skipped_unknown += 1
                    continue
                
                if core_only and not is_core:
                    skipped_noncore += 1
                    continue
                
                entries.append(GameEntry(
                    unique_id=row['unique_id'],
                    game_id=row['game_id'],
                    payoffs=payoffs,
                    role='row' if is_row else 'col',
                    rg_type=rg_type,
                    focal_payoffs=focal,
                    is_known=is_known,
                    is_core=is_core,
                ))
            except:
                pass
    
    print(f"  Loaded: {len(entries)}")
    if skipped_unknown:
        print(f"  Skipped {skipped_unknown} unknown R-G patterns")
    if skipped_noncore:
        print(f"  Skipped {skipped_noncore} non-core types")
    
    return entries


def generate_narrative(game: GameEntry) -> str:
    a, b, c, d = game.focal_payoffs
    rg_type = game.rg_type
    
    if rg_type not in NARRATIVE_TEMPLATES:
        return f"A strategic situation with payoffs: cooperate-cooperate {a}, cooperate-defect {b}, defect-cooperate {c}, defect-defect {d}."
    
    template = random.choice(NARRATIVE_TEMPLATES[rg_type])
    return template.format(a=a, b=b, c=c, d=d)


# =============================================================================
# CLASSIFIER
# =============================================================================

VALID_TYPES = list(RG_PATTERNS.values())

FUZZY_MAP = {
    "prisoner": "Prisoners_Dilemma", "dilemma": "Prisoners_Dilemma",
    "chicken": "Chicken", "hawk": "Chicken", "brinkmanship": "Chicken",
    "stag": "Stag_Hunt", "hunt": "Stag_Hunt",
    "battle": "Battle_of_the_Sexes", "sexes": "Battle_of_the_Sexes",
    "coordination": "Coordination_Game",
    "assurance": "Assurance_Game",
    "harmony": "Harmony", "deadlock": "Deadlock",
    "hero": "Hero", "volunteer": "Hero", "whistleblow": "Hero",
    "compromise": "Compromise",
    "peace": "Peace", "concord": "Concord",
}

def normalize_type(raw: str) -> str:
    if not raw:
        return "Unknown"
    cleaned = raw.strip().split('\n')[0].strip()
    cleaned = re.sub(r'[*\[\]()"\']', '', cleaned)
    cleaned = re.sub(r'[^\w\s_]', '', cleaned).strip()
    
    for valid in VALID_TYPES:
        if cleaned.lower().replace("_", "").replace(" ", "") == valid.lower().replace("_", ""):
            return valid
    
    for key, value in FUZZY_MAP.items():
        if key in cleaned.lower():
            return value
    
    return cleaned.replace(" ", "_") if len(cleaned) < 35 else "Unknown"


SYSTEM_PROMPT = """You are a game theory expert classifying strategic situations.

The 12 Robinson-Goforth game types (2×2 symmetric games):

CONFLICT GAMES (temptation to defect):
- Prisoners_Dilemma: Tempted to defect, but mutual defection is worse than mutual cooperation
- Chicken: Mutual aggression is CATASTROPHIC (worst outcome) - someone must yield
- Deadlock: Mutual "defection" is actually PREFERRED - no dilemma, independence is good

COORDINATION GAMES:
- Stag_Hunt: High-risk cooperation beats safe defection, but need trust
- Assurance_Game: Like Stag Hunt but being abandoned after committing is worse than not committing
- Coordination_Game: Pure matching - no preference which option, just need to coordinate
- Battle_of_the_Sexes: Need to coordinate but prefer DIFFERENT equilibria

COOPERATIVE GAMES (cooperation dominates):
- Harmony: Cooperation is dominant strategy - no conflict whatsoever
- Peace: Both prefer mutual engagement, but one-sided effort is costly
- Concord: Strong cooperation preference with clear ranking

OTHER:
- Hero: Someone must sacrifice (volunteer's dilemma)
- Compromise: Mutual concession beats mutual stubbornness

Output ONLY the game type name."""


USER_TEMPLATE = """Classify this strategic situation:

{narrative}

Game type:"""


class ParallelClassifier:
    def __init__(self, model: str = "claude-sonnet-4-20250514", max_workers: int = 4):
        self.model = model
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.calls = 0
        self.tokens = 0
        self.errors = 0
        self.clients = [Anthropic() for _ in range(max_workers)]
    
    def _classify_single(self, game: GameEntry, idx: int) -> Dict:
        client = self.clients[idx % len(self.clients)]
        
        if not game.narrative:
            game.narrative = generate_narrative(game)
        
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=30,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": USER_TEMPLATE.format(narrative=game.narrative)}]
            )
            
            raw = response.content[0].text.strip()
            predicted = normalize_type(raw)
            
            with self.lock:
                self.calls += 1
                self.tokens += response.usage.input_tokens + response.usage.output_tokens
            
            return {
                'id': game.unique_id,
                'rg_type': game.rg_type,
                'predicted': predicted,
                'correct': predicted == game.rg_type,
                'raw': raw[:40],
                'is_core': game.is_core,
            }
        except Exception as e:
            with self.lock:
                self.errors += 1
            return {
                'id': game.unique_id,
                'rg_type': game.rg_type,
                'predicted': 'ERROR',
                'correct': False,
                'raw': str(e)[:30],
                'is_core': game.is_core,
            }
    
    def classify_batch(self, games: List[GameEntry], delay: float = 0.25) -> List[Dict]:
        results = []
        total = len(games)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._classify_single, g, i): g for i, g in enumerate(games)}
            for i, future in enumerate(as_completed(futures)):
                results.append(future.result())
                if (i + 1) % 5 == 0 or i + 1 == total:
                    acc = sum(r['correct'] for r in results) / len(results)
                    print(f"\r  {i+1}/{total} | Acc: {100*acc:.1f}%", end="", flush=True)
                time.sleep(delay)
        print()
        return results


def run_validation(games: List[GameEntry], workers: int = 4, delay: float = 0.25) -> Dict:
    print("  Generating narratives...")
    for g in games:
        g.narrative = generate_narrative(g)
    
    print("  Classifying...")
    classifier = ParallelClassifier(max_workers=workers)
    results = classifier.classify_batch(games, delay)
    
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(lambda: defaultdict(int))
    
    for r in results:
        by_type[r['rg_type']]["total"] += 1
        if r['correct']:
            by_type[r['rg_type']]["correct"] += 1
        confusion[r['rg_type']][r['predicted']] += 1
    
    # Core vs all accuracy
    core_results = [r for r in results if r['is_core']]
    core_correct = sum(r['correct'] for r in core_results)
    
    n_correct = sum(r['correct'] for r in results)
    n_errors = sum(1 for r in results if r['predicted'] == 'ERROR')
    
    return {
        'n': len(results),
        'correct': n_correct,
        'accuracy': n_correct / len(results) if results else 0,
        'core_n': len(core_results),
        'core_correct': core_correct,
        'core_accuracy': core_correct / len(core_results) if core_results else 0,
        'errors': n_errors,
        'by_type': dict(by_type),
        'confusion': {k: dict(v) for k, v in confusion.items()},
        'results': results,
        'tokens': classifier.tokens,
    }


def print_report(r: Dict):
    print(f"\n{'='*75}")
    print(f"OVERALL ACCURACY:    {r['correct']}/{r['n']} ({100*r['accuracy']:.1f}%)")
    print(f"CORE TYPES ACCURACY: {r['core_correct']}/{r['core_n']} ({100*r['core_accuracy']:.1f}%)")
    print(f"{'='*75}")
    print(f"Errors: {r['errors']} | Tokens: {r['tokens']}")
    
    print("\n" + "-"*75)
    print("Accuracy by Type:")
    print("-"*75)
    for t, s in sorted(r['by_type'].items(), key=lambda x: -x[1]['total']):
        if s['total'] == 0:
            continue
        acc = s['correct']/s['total']
        core_marker = "★" if t in CORE_TYPES else " "
        bar = '█' * int(acc * 10) + '░' * (10 - int(acc * 10))
        print(f"  {core_marker} {t:22} {s['correct']:3}/{s['total']:3} {bar} {100*acc:5.1f}%")
    
    print("\n  ★ = Core type (distinctive strategic structure)")
    
    print("\n" + "-"*75)
    print("Confusion Matrix:")
    print("-"*75)
    for rg_type, preds in sorted(r['confusion'].items()):
        top = sorted(preds.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{l[:15]}({c})" for l, c in top)
        print(f"  {rg_type:20} → {top_str}")


def main():
    parser = argparse.ArgumentParser(description="GTI v10 - Fixed Narrative Classification")
    parser.add_argument('--data', required=True)
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--validate', type=str, metavar='N')
    parser.add_argument('--core-only', action='store_true', help='Only core 9 types')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--delay', type=float, default=0.3)
    
    args = parser.parse_args()
    
    print(f"Loading {args.data}...")
    games = load_games(Path(args.data), known_only=True, core_only=args.core_only)
    
    if args.stats:
        counts = Counter(g.rg_type for g in games)
        print("\nR-G Type Distribution (known patterns only):")
        for t, c in counts.most_common():
            marker = "★" if t in CORE_TYPES else " "
            print(f"  {marker} {t:22}: {c:4}")
        return
    
    if args.validate:
        if not HAS_ANTHROPIC:
            print("Error: pip install anthropic")
            return
        
        n = len(games) if args.validate.lower() == 'all' else min(int(args.validate), len(games))
        sample = random.sample(games, n)
        
        print(f"\nValidating {n} games...")
        report = run_validation(sample, args.workers, args.delay)
        print_report(report)
        
        out_file = f"gti_v10_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(out_file, 'w') as f:
            json.dump({
                'accuracy': report['accuracy'],
                'core_accuracy': report['core_accuracy'],
                'by_type': report['by_type'],
            }, f, indent=2)
        print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
