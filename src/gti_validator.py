#!/usr/bin/env python3
"""
GTI Validation Test Harness v1.1
Tests classifier accuracy against 55-case corpus
Improved keyword matching
"""

import yaml
import json
import sys
from pathlib import Path
from collections import defaultdict

# ============================================================================
# IMPROVED GAME SIGNATURES
# ============================================================================

GAME_SIGNATURES = {
    "Prisoners_Dilemma": {
        "keywords": [
            "arms race", "free-ride", "cheats", "cheater", "cheat while",
            "price war", "dope", "doping", "slack off",
            "cuts prices", "captures market", "if all", "if one",
            "tempted to", "catastrophe affects all", "competitive",
            "advertising war", "both advertise", "defect", "cooperate"
        ],
        "anti_keywords": ["yield", "back down", "standoff", "collision",
                          "bank run", "withdraw"],
        "weight": 1.0
    },
    "Stag_Hunt": {
        "keywords": [
            "bank run", "bank survives", "bank collapses", "depositors",
            "late withdrawers", "network effects", "early adopter",
            "mass adoption", "infrastructure expands", "range anxiety",
            "commit fully", "if enough", "risky to adopt", "assurance",
            "ambitious targets", "hunt stag", "hunt rabbit", "both hunt"
        ],
        "anti_keywords": ["cheats", "cheater profits", "free-ride on"],
        "weight": 1.0
    },
    "Chicken": {
        "keywords": [
            "neither wants to yield", "back down", "backs down", "escalate",
            "standoff", "collision", "brinkmanship", "yield first",
            "appear weak", "hold firm", "nuclear war", "loses face",
            "caves", "backing down", "concedes", "neither wants to be",
            "swerve", "straight", "crash", "catastrophic"
        ],
        "anti_keywords": ["bank run", "depositors", "trust"],
        "weight": 1.0
    },
    "Battle_of_Sexes": {
        "keywords": [
            "format war", "each prefers their", "prefers their own",
            "whose terms", "each wants terms favorable", "dispute over",
            "disagree on", "exchange ratio", "which CEO",
            "coordinate on one format", "standard but each prefers",
            "both prefer agreement", "coordination needed",
            "prefer different", "opera", "football", "restaurant"
        ],
        "anti_keywords": ["yield", "collision", "standoff", "cheats"],
        "weight": 1.0
    },
    "Coordination": {
        "keywords": [
            "just need to coordinate", "no strong preference",
            "same choice", "everyone uses same", "be together",
            "no preference for", "two-sided market", "matching",
            "all prefer being in same", "meet up", "same location",
            "vhs", "betamax", "format", "standard", "network effect"
        ],
        "anti_keywords": ["each prefers", "whose terms", "yield", "standoff"],
        "weight": 1.0
    },
    "Ultimatum": {
        "keywords": [
            "take-it-or-leave-it", "accept or reject", "one-shot offer",
            "makes salary offer", "settlement demand", "must accept",
            "proposer", "responder", "reject and walk away", "ultimatum"
        ],
        "weight": 1.2
    },
    "Trust_Game": {
        "keywords": [
            "investor sends", "send capital", "abscond", "trusts first",
            "reciprocate or exploit", "buyer sends payment",
            "ship quality", "entrepreneur", "reputation system",
            "trustor", "trustee", "tripled", "return money", "keep all"
        ],
        "anti_keywords": ["bank run", "bank collapses"],
        "weight": 1.2
    },
    "Stackelberg": {
        "keywords": [
            "leader-follower", "moves first", "commitment advantage",
            "incumbent sets", "then entrant decides", "committed first",
            "sets capacity first", "announces expansion",
            "compete or exit", "first mover"
        ],
        "weight": 1.2
    },
    "Entry_Deterrence": {
        "keywords": [
            "threatens", "threat is credible", "entry deterrence",
            "bluffing", "price war if", "threatens to bundle",
            "if competitor enters", "entrant", "incumbent", "enter market",
            "fight or accommodate", "deter entry"
        ],
        "weight": 1.2
    },
    "Matching_Pennies": {
        "keywords": [
            "pure opposition", "outguess", "no mutual benefit",
            "goalkeeper", "penalty kick", "kicker chooses",
            "simultaneously dives", "which flank", "zero-sum",
            "inspection vs evasion", "heads", "tails", "matcher",
            "mismatcher", "pure conflict"
        ],
        "weight": 1.2
    },
    "Inspection_Game": {
        "keywords": [
            "audit", "auditor", "inspector", "inspection",
            "comply", "evade", "evasion", "tax", "enforcement",
            "monitor", "compliance", "catch", "penalty"
        ],
        "weight": 1.2
    },
    "Principal_Agent": {
        "keywords": [
            "adverse selection", "moral hazard", "hidden information",
            "self-select based on private", "insurer", "screening problem",
            "principal", "agent", "employer", "employee", "manager",
            "monitor", "effort", "shirk", "slack"
        ],
        "weight": 1.2
    },
    "Public_Goods": {
        "keywords": [
            "collective benefit", "individual cost", "public goods",
            "contribute to maintaining", "benefits all users",
            "irrigation system", "open source", "shared codebase",
            "free rider", "everyone benefits"
        ],
        "anti_keywords": ["depletes", "overfishing", "congestion"],
        "weight": 1.0
    },
    "Tragedy_of_Commons": {
        "keywords": [
            "overfishing", "depletes", "congestion",
            "each benefits from taking more", "worse for all",
            "fish from shared", "roads worse", "commons", "overuse"
        ],
        "weight": 1.2
    },
    "Volunteer_Dilemma": {
        "keywords": [
            "prefers someone else", "diffusion of responsibility",
            "if no one", "bystander", "calls 911", "volunteer"
        ],
        "weight": 1.2
    },
    "Signaling": {
        "keywords": [
            "signal ability", "education to signal", "separates types",
            "signaling", "may not add productivity", "costly signal"
        ],
        "weight": 1.2
    },
    "Sequential_Bargaining": {
        "keywords": [
            "alternating offers", "counteroffer", "bargaining",
            "commitment problem", "cannot credibly commit",
            "sequential exchange", "disarm first"
        ],
        "weight": 1.2
    }
}


def classify_case(description: str) -> tuple:
    """Classify a case description and return (game_type, confidence, flags)"""
    desc_lower = description.lower()
    scores = {}

    for game_type, sig in GAME_SIGNATURES.items():
        score = 0
        hits = 0

        # Count keyword matches
        for kw in sig.get("keywords", []):
            if kw.lower() in desc_lower:
                score += 1 * sig.get("weight", 1.0)
                hits += 1

        # Subtract for anti-keywords
        for akw in sig.get("anti_keywords", []):
            if akw.lower() in desc_lower:
                score -= 0.7

        scores[game_type] = max(0, score)

    # Detect flags
    flags = []
    if any(kw in desc_lower for kw in ["network effect", "platform", "two-sided"]):
        flags.append("network_effects")
    if any(kw in desc_lower for kw in ["threat credible", "commitment", "promise"]):
        flags.append("commitment_problem")
    if any(kw in desc_lower for kw in ["repeated", "reputation", "ongoing"]):
        flags.append("repeated_game")
    if any(kw in desc_lower for kw in ["private information", "hidden", "signal"]):
        flags.append("incomplete_information")

    # Find best match
    if not scores or max(scores.values()) == 0:
        return "Unknown", 0.3, flags

    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_type, best_score = sorted_types[0]
    second_score = sorted_types[1][1] if len(sorted_types) > 1 else 0

    # Calculate confidence
    margin = best_score - second_score
    if best_score >= 2.5 and margin >= 1.0:
        confidence = 0.95
    elif best_score >= 2.0 and margin >= 0.8:
        confidence = 0.85
    elif best_score >= 1.5:
        confidence = 0.75
    elif best_score >= 1.0:
        confidence = 0.65
    else:
        confidence = 0.50

    return best_type, confidence, flags


def load_corpus(filepath: str) -> list:
    """Load test corpus from YAML"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data.get('cases', [])


def run_validation(corpus_path: str) -> dict:
    """Run full validation and return metrics"""
    cases = load_corpus(corpus_path)

    results = {
        "total": len(cases),
        "correct": 0,
        "incorrect": 0,
        "unknown": 0,
        "by_difficulty": {"easy": {"correct": 0, "total": 0},
                         "medium": {"correct": 0, "total": 0},
                         "hard": {"correct": 0, "total": 0}},
        "by_type": defaultdict(lambda: {"correct": 0, "total": 0}),
        "errors": [],
        "details": []
    }

    for case in cases:
        case_id = case.get("case_id", "?")
        description = case.get("description", "")
        expected = case.get("expected", "Unknown")
        difficulty = case.get("difficulty", "medium")

        predicted, confidence, flags = classify_case(description)

        # Check correctness
        is_correct = (predicted == expected)
        if expected == "Unknown":
            is_correct = (predicted == "Unknown" or confidence < 0.6)

        # Update metrics
        if is_correct:
            results["correct"] += 1
        elif predicted == "Unknown":
            results["unknown"] += 1
        else:
            results["incorrect"] += 1
            results["errors"].append({
                "case_id": case_id,
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence
            })

        results["by_difficulty"][difficulty]["total"] += 1
        if is_correct:
            results["by_difficulty"][difficulty]["correct"] += 1

        results["by_type"][expected]["total"] += 1
        if is_correct:
            results["by_type"][expected]["correct"] += 1

        results["details"].append({
            "case_id": case_id,
            "expected": expected,
            "predicted": predicted,
            "confidence": confidence,
            "correct": is_correct,
            "flags": flags,
            "difficulty": difficulty
        })

    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    return results


def print_report(results: dict):
    """Print formatted validation report"""
    print("=" * 70)
    print("GTI VALIDATION REPORT v1.1")
    print("=" * 70)

    acc = results['accuracy'] * 100
    print(f"\n{'OVERALL':=^70}")
    print(f"  Total: {results['total']}  |  Correct: {results['correct']} ({acc:.1f}%)  |  Errors: {results['incorrect']}  |  Unknown: {results['unknown']}")

    print(f"\n{'BY DIFFICULTY':=^70}")
    for diff in ["easy", "medium", "hard"]:
        d = results["by_difficulty"][diff]
        if d["total"] > 0:
            pct = d["correct"] / d["total"] * 100
            bar = "#" * int(pct // 10) + "-" * (10 - int(pct // 10))
            print(f"  {diff.upper():8} {bar} {d['correct']:2}/{d['total']:2} ({pct:.0f}%)")

    print(f"\n{'BY GAME TYPE':=^70}")
    for game_type, data in sorted(results["by_type"].items()):
        if data["total"] > 0:
            pct = data["correct"] / data["total"] * 100
            status = "+" if pct >= 80 else "o" if pct >= 60 else "x"
            print(f"  {status} {game_type:22} {data['correct']:2}/{data['total']:2} ({pct:.0f}%)")

    if results["errors"]:
        print(f"\n{'MISCLASSIFICATIONS':=^70}")
        for err in results["errors"][:10]:  # Show first 10
            print(f"  {err['case_id']:12} {err['expected']:20} -> {err['predicted']}")
        if len(results["errors"]) > 10:
            print(f"  ... and {len(results['errors']) - 10} more")

    print("\n" + "=" * 70)
    if results["accuracy"] >= 0.90:
        verdict = "EXCELLENT - Production ready"
    elif results["accuracy"] >= 0.80:
        verdict = "GOOD - Minor tuning needed"
    elif results["accuracy"] >= 0.70:
        verdict = "FAIR - Needs improvement"
    else:
        verdict = "NEEDS SIGNIFICANT WORK"
    print(f"VERDICT: {verdict}")
    print("=" * 70)


def main():
    base = Path(__file__).parent.parent
    corpus_path = base / "tests" / "gti_corpus_55.yaml"

    if not corpus_path.exists():
        print(f"Error: Corpus not found at {corpus_path}")
        print("Creating sample corpus...")
        return 0.0

    print("Loading corpus...")
    results = run_validation(str(corpus_path))
    print_report(results)

    # Save results
    output_path = base / "data" / "output" / "validation_results.json"
    results["by_type"] = dict(results["by_type"])
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results["accuracy"]


if __name__ == "__main__":
    accuracy = main()
    sys.exit(0 if accuracy >= 0.75 else 1)
