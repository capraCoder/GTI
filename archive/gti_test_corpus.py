"""
GTI Claude Classifier - Test on 55 Corpus Cases
================================================
Reads gti_corpus_55.yaml and classifies each case using Claude API.
Compares results against expected labels to measure accuracy.

Usage:
    python gti_test_corpus.py --test       # Test API
    python gti_test_corpus.py --run 5      # Run on 5 cases
    python gti_test_corpus.py --run all    # Run on all 55 cases
"""

import os
import json
import time
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import httpx

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = "claude-sonnet-4-20250514"
    base_url: str = "https://api.anthropic.com/v1/messages"
    
    # Paths
    corpus_path: Path = Path("tests/gti_corpus_55.yaml")
    output_dir: Path = Path("results")
    
    # Rate limiting
    delay_between_calls: float = 1.5  # seconds
    backoff_seconds: list = field(default_factory=lambda: [60, 120, 300])

# ============================================================
# GAME TYPES
# ============================================================

GAME_TYPES = [
    "Prisoners_Dilemma", "Stag_Hunt", "Chicken", "Battle_of_Sexes",
    "Matching_Pennies", "Coordination", "Anti_Coordination", "Deadlock",
    "Harmony", "Hero", "Compromise", "Cyclic",
    "Assurance", "Trust_Game", "Ultimatum", "Dictator",
    "Public_Goods", "Common_Pool", "Auction", "Bargaining",
    "Signaling", "Screening"
]

SYSTEM_PROMPT = """You are GTI (Game Theory Identifier), an expert classifier for strategic interactions.

Analyze the scenario and classify it as ONE of these game types:
- Prisoners_Dilemma: Mutual defection dominant despite mutual cooperation being better
- Stag_Hunt: Coordination with risk; cooperation better but risky
- Chicken: Anti-coordination; mutual aggression is worst outcome
- Battle_of_Sexes: Want to coordinate but prefer different equilibria
- Matching_Pennies: Zero-sum; one wants to match, other to mismatch
- Coordination: Pure coordination; same action = good
- Anti_Coordination: Want different actions
- Deadlock: Both prefer mutual defection
- Harmony: Cooperation dominant for both; no dilemma
- Hero: One must sacrifice for collective good
- Compromise: Middle ground preferred
- Cyclic: Rock-paper-scissors; no pure equilibrium
- Assurance: Like Stag Hunt but stronger coordination pull
- Trust_Game: Sequential; first trusts, second can exploit or reward
- Ultimatum: Proposer offers, responder accepts or both get nothing
- Dictator: One decides, other has no choice
- Public_Goods: Contribute to shared resource; free-rider problem
- Common_Pool: Extract from shared resource; tragedy of commons
- Auction: Competitive bidding
- Bargaining: Negotiate division of surplus
- Signaling: Informed sends signal, uninformed responds
- Screening: Uninformed designs mechanism to reveal info

OUTPUT FORMAT (JSON only, no markdown):
{"game_type": "Type_Name", "confidence": 0.95, "reasoning": "Brief explanation"}"""

# ============================================================
# CLAUDE CLIENT
# ============================================================

class ClaudeClient:
    def __init__(self, config: Config):
        self.config = config
        self.total_calls = 0
        self.total_tokens = 0
    
    def classify(self, description: str) -> dict:
        """Classify a game description."""
        
        if not self.config.api_key:
            return {"success": False, "error": "ANTHROPIC_API_KEY not set"}
        
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "max_tokens": 300,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": f"Classify this scenario:\n\n{description}"}]
        }
        
        for attempt, backoff in enumerate(self.config.backoff_seconds + [None]):
            try:
                with httpx.Client(timeout=60) as client:
                    response = client.post(
                        self.config.base_url,
                        headers=headers,
                        json=payload
                    )
                
                if response.status_code == 200:
                    self.total_calls += 1
                    data = response.json()
                    
                    # Track tokens
                    usage = data.get("usage", {})
                    self.total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    
                    # Parse response
                    text = data["content"][0]["text"].strip()
                    
                    # Handle markdown code blocks
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0]
                    elif "```" in text:
                        text = text.split("```")[1].split("```")[0]
                    
                    result = json.loads(text)
                    result["success"] = True
                    return result
                
                elif response.status_code == 429 and backoff:
                    print(f"    ⏳ Rate limited, waiting {backoff}s...")
                    time.sleep(backoff)
                    continue
                
                else:
                    return {"success": False, "error": f"HTTP {response.status_code}"}
                    
            except json.JSONDecodeError as e:
                return {"success": False, "error": f"JSON parse: {e}", "raw": text}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Max retries exceeded"}

# ============================================================
# MAIN PIPELINE
# ============================================================

def load_corpus(path: Path) -> list:
    """Load test cases from YAML."""
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data.get("cases", [])

def normalize_type(game_type: str) -> str:
    """Normalize game type names for comparison."""
    # Remove common suffixes/variations
    normalized = game_type.strip()
    
    # Map variations to canonical names
    mappings = {
        "Prisoners_Dilemma": ["PD", "Prisoner's_Dilemma", "Prisoners_Dilemma"],
        "Chicken": ["Chicken_Game", "Hawk_Dove", "Chicken"],
        "Stag_Hunt": ["Stag_Hunt", "StagHunt"],
        "Battle_of_Sexes": ["Battle_of_Sexes", "BoS", "BattleOfSexes"],
        "Coordination": ["Coordination", "Coordination_Game", "Pure_Coordination"],
        "Trust_Game": ["Trust_Game", "Trust", "TrustGame"],
        "Public_Goods": ["Public_Goods", "PublicGoods", "Public_Goods_Game"],
        "Common_Pool": ["Common_Pool", "CommonPool", "Tragedy_of_Commons"],
    }
    
    for canonical, variants in mappings.items():
        if normalized in variants or normalized.replace("_", "") == canonical.replace("_", ""):
            return canonical
    
    return normalized

def run_test(config: Config, limit: int = None):
    """Run classification test on corpus."""
    
    print("\n" + "="*60)
    print("GTI CLAUDE CLASSIFIER - CORPUS VALIDATION")
    print("="*60)
    
    # Load corpus
    if not config.corpus_path.exists():
        print(f"\n✗ Corpus not found: {config.corpus_path}")
        return
    
    cases = load_corpus(config.corpus_path)
    if limit:
        cases = cases[:limit]
    
    print(f"\nCorpus: {config.corpus_path}")
    print(f"Cases to test: {len(cases)}")
    print(f"Model: {config.model}")
    
    # Initialize
    client = ClaudeClient(config)
    results = []
    correct = 0
    
    # Ensure output dir
    config.output_dir.mkdir(exist_ok=True)
    
    # Process each case
    start_time = time.time()
    
    for i, case in enumerate(cases, 1):
        case_id = case.get("case_id", f"case_{i}")
        expected = case.get("expected", "Unknown")
        difficulty = case.get("difficulty", "?")
        description = case.get("description", "").strip()
        
        print(f"\n[{i}/{len(cases)}] {case_id} ({difficulty})")
        print(f"  Expected: {expected}")
        
        # Classify
        result = client.classify(description)
        
        if result["success"]:
            predicted = result.get("game_type", "Unknown")
            confidence = result.get("confidence", 0)
            
            # Normalize for comparison
            expected_norm = normalize_type(expected)
            predicted_norm = normalize_type(predicted)
            
            is_correct = expected_norm == predicted_norm
            if is_correct:
                correct += 1
                print(f"  ✓ Predicted: {predicted} (conf: {confidence:.2f})")
            else:
                print(f"  ✗ Predicted: {predicted} (expected: {expected})")
            
            results.append({
                "case_id": case_id,
                "difficulty": difficulty,
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence,
                "correct": is_correct,
                "reasoning": result.get("reasoning", "")
            })
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown')[:80]}")
            results.append({
                "case_id": case_id,
                "expected": expected,
                "predicted": "ERROR",
                "correct": False,
                "error": result.get("error", "")
            })
        
        # Rate limit delay
        if i < len(cases):
            time.sleep(config.delay_between_calls)
    
    # Summary
    elapsed = time.time() - start_time
    accuracy = (correct / len(cases)) * 100 if cases else 0
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"  Total cases:  {len(cases)}")
    print(f"  Correct:      {correct}")
    print(f"  Accuracy:     {accuracy:.1f}%")
    print(f"  Time:         {elapsed:.1f}s")
    print(f"  API calls:    {client.total_calls}")
    print(f"  Tokens used:  {client.total_tokens:,}")
    
    # Breakdown by difficulty
    print("\n  By difficulty:")
    for diff in ["easy", "medium", "hard"]:
        diff_cases = [r for r in results if r.get("difficulty") == diff]
        if diff_cases:
            diff_correct = sum(1 for r in diff_cases if r.get("correct"))
            diff_acc = (diff_correct / len(diff_cases)) * 100
            print(f"    {diff:8s}: {diff_correct}/{len(diff_cases)} ({diff_acc:.0f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = config.output_dir / f"gti_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "model": config.model,
            "total_cases": len(cases),
            "correct": correct,
            "accuracy": accuracy,
            "elapsed_seconds": elapsed,
            "results": results
        }, f, indent=2)
    
    print(f"\n  Results saved: {output_file}")
    
    # Show errors if any
    errors = [r for r in results if not r.get("correct")]
    if errors and len(errors) <= 10:
        print("\n  Misclassifications:")
        for e in errors:
            print(f"    {e['case_id']}: {e.get('predicted', '?')} (expected {e['expected']})")

def test_api(config: Config):
    """Quick API test."""
    print("\nTesting Claude API...")
    client = ClaudeClient(config)
    
    result = client.classify("Two drivers on collision course. Each can swerve or not. Crash if both go straight.")
    
    if result["success"]:
        print(f"✓ API works!")
        print(f"  Classified as: {result.get('game_type')} (conf: {result.get('confidence', 0):.2f})")
    else:
        print(f"✗ Error: {result.get('error')}")

# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GTI Claude Classifier")
    parser.add_argument("--test", action="store_true", help="Test API")
    parser.add_argument("--run", type=str, help="Run on N cases or 'all'")
    parser.add_argument("--corpus", type=str, default="tests/gti_corpus_55.yaml", help="Corpus path")
    
    args = parser.parse_args()
    
    config = Config()
    config.corpus_path = Path(args.corpus)
    
    if args.test:
        test_api(config)
    elif args.run:
        limit = None if args.run.lower() == "all" else int(args.run)
        run_test(config, limit)
    else:
        parser.print_help()
        print("\n\nExamples:")
        print("  python gti_test_corpus.py --test")
        print("  python gti_test_corpus.py --run 5")
        print("  python gti_test_corpus.py --run all")

if __name__ == "__main__":
    main()
