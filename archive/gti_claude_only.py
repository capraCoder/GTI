"""
GTI Claude-Only Pipeline
========================
Uses Claude API for both harvesting AND classification.
Handles rate limits with automatic backoff.

Usage:
    python gti_claude_only.py --test      # Test API connectivity
    python gti_claude_only.py --run 10    # Process 10 games
    python gti_claude_only.py --run all   # Process all games
"""

import os
import json
import time
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import httpx

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = "claude-opus-4-20250514"
    base_url: str = "https://api.anthropic.com/v1/messages"
	
    # Rate limit handling
    requests_per_minute: int = 50  # Conservative estimate
    backoff_seconds: list = field(default_factory=lambda: [60, 120, 300, 600])
    
    # Paths
    db_path: Path = Path("games2p2k.db")
    output_dir: Path = Path("gti_results")
    
    # Processing
    batch_size: int = 10
    delay_between_calls: float = 1.2  # seconds

# ============================================================
# CLAUDE CLIENT
# ============================================================

class ClaudeClient:
    """Simple Claude API client with rate limit handling."""
    
    def __init__(self, config: Config):
        self.config = config
        self.consecutive_429s = 0
        self.total_calls = 0
        self.total_tokens = 0
        
    def call(self, system: str, user: str, max_tokens: int = 1000) -> dict:
        """Make API call with automatic retry on rate limit."""
        
        if not self.config.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}]
        }
        
        for attempt in range(len(self.config.backoff_seconds) + 1):
            try:
                with httpx.Client(timeout=60) as client:
                    response = client.post(
                        self.config.base_url,
                        headers=headers,
                        json=payload
                    )
                
                if response.status_code == 200:
                    self.consecutive_429s = 0
                    self.total_calls += 1
                    data = response.json()
                    
                    # Track tokens
                    usage = data.get("usage", {})
                    self.total_tokens += usage.get("input_tokens", 0)
                    self.total_tokens += usage.get("output_tokens", 0)
                    
                    return {
                        "success": True,
                        "text": data["content"][0]["text"],
                        "usage": usage
                    }
                
                elif response.status_code == 429:
                    self.consecutive_429s += 1
                    
                    if attempt < len(self.config.backoff_seconds):
                        wait = self.config.backoff_seconds[attempt]
                        print(f"  ⏳ Rate limited. Waiting {wait}s (attempt {attempt + 1})...")
                        time.sleep(wait)
                        continue
                    else:
                        return {"success": False, "error": "Rate limit exceeded after all retries"}
                
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text[:200]}"
                    }
                    
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Unknown error"}

# ============================================================
# GTI CLASSIFIER
# ============================================================

# The 22 game types from GTI v2.2
GAME_TYPES = [
    "Prisoners_Dilemma", "Stag_Hunt", "Chicken_Game", "Battle_of_Sexes",
    "Matching_Pennies", "Coordination_Game", "Anti_Coordination", "Deadlock",
    "Harmony_Game", "Hero_Game", "Compromise_Game", "Cyclic_Game",
    "Assurance_Game", "Trust_Game", "Ultimatum_Game", "Dictator_Game",
    "Public_Goods", "Common_Pool", "Auction", "Bargaining",
    "Signaling_Game", "Screening_Game"
]

SYSTEM_PROMPT = """You are GTI (Game Theory Identifier), an expert system for classifying strategic interactions.

Your task: Analyze the given game/scenario and classify it into ONE of these 22 canonical game types:

1. Prisoners_Dilemma - Mutual defection dominant despite mutual cooperation being better
2. Stag_Hunt - Coordination with risk; safer to defect but better to cooperate
3. Chicken_Game - Anti-coordination; worst outcome is mutual aggression
4. Battle_of_Sexes - Coordination preferred but different preferences on which equilibrium
5. Matching_Pennies - Zero-sum; one wants to match, other wants to mismatch
6. Coordination_Game - Pure coordination; same action = good, different = bad
7. Anti_Coordination - Opposite of coordination; want different actions
8. Deadlock - Both prefer mutual defection; no cooperation incentive
9. Harmony_Game - Cooperation is dominant for both; no dilemma
10. Hero_Game - One must sacrifice for collective good
11. Compromise_Game - Middle ground preferred to extremes
12. Cyclic_Game - Rock-paper-scissors dynamics; no pure equilibrium
13. Assurance_Game - Similar to Stag Hunt but with stronger coordination incentive
14. Trust_Game - Sequential; first mover trusts, second can exploit or reward
15. Ultimatum_Game - Proposer offers split, responder accepts or rejects (both get nothing)
16. Dictator_Game - One player decides allocation, other has no choice
17. Public_Goods - Contribute to shared resource; free-rider problem
18. Common_Pool - Extract from shared resource; tragedy of commons
19. Auction - Competitive bidding for scarce item
20. Bargaining - Negotiate division of surplus
21. Signaling_Game - Informed player sends signal, uninformed responds
22. Screening_Game - Uninformed player designs mechanism to reveal information

OUTPUT FORMAT (strict JSON):
{
    "game_type": "One_Of_The_22_Types",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of key features that determined classification",
    "alternative": "Second-best classification if confidence < 0.8, else null"
}"""

def classify_game(client: ClaudeClient, game_name: str, description: str) -> dict:
    """Classify a single game using Claude."""
    
    user_prompt = f"""Classify this strategic interaction:

GAME: {game_name}

DESCRIPTION:
{description}

Analyze the payoff structure, player incentives, and strategic dynamics. Output JSON only."""

    result = client.call(SYSTEM_PROMPT, user_prompt, max_tokens=500)
    
    if not result["success"]:
        return {"success": False, "error": result["error"]}
    
    # Parse JSON from response
    try:
        text = result["text"].strip()
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        classification = json.loads(text)
        classification["success"] = True
        classification["raw_response"] = result["text"]
        return classification
        
    except json.JSONDecodeError as e:
        return {
            "success": False, 
            "error": f"JSON parse error: {e}",
            "raw_response": result["text"]
        }

# ============================================================
# DATABASE HELPERS
# ============================================================

def get_unprocessed_games(db_path: Path, limit: Optional[int] = None) -> list:
    """Get games that haven't been classified yet."""
    
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        return []
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Check if gti_results table exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='gti_results'"
    )
    
    if not cursor.fetchone():
        # Create results table
        conn.execute("""
            CREATE TABLE gti_results (
                game_id TEXT PRIMARY KEY,
                game_name TEXT,
                game_type TEXT,
                confidence REAL,
                reasoning TEXT,
                alternative TEXT,
                processed_at TEXT,
                model TEXT,
                raw_response TEXT
            )
        """)
        conn.commit()
    
    # Get unprocessed games
    query = """
        SELECT g.id, g.name, g.description 
        FROM games g
        LEFT JOIN gti_results r ON g.id = r.game_id
        WHERE r.game_id IS NULL
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    games = [dict(row) for row in conn.execute(query).fetchall()]
    conn.close()
    
    return games

def save_result(db_path: Path, game_id: str, game_name: str, result: dict, model: str):
    """Save classification result to database."""
    
    conn = sqlite3.connect(db_path)
    
    conn.execute("""
        INSERT OR REPLACE INTO gti_results 
        (game_id, game_name, game_type, confidence, reasoning, alternative, processed_at, model, raw_response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        game_id,
        game_name,
        result.get("game_type", "ERROR"),
        result.get("confidence", 0.0),
        result.get("reasoning", ""),
        result.get("alternative"),
        datetime.now().isoformat(),
        model,
        result.get("raw_response", "")
    ))
    
    conn.commit()
    conn.close()

# ============================================================
# MAIN PIPELINE
# ============================================================

def test_api(config: Config):
    """Test Claude API connectivity."""
    
    print("\n" + "="*50)
    print("GTI CLAUDE API TEST")
    print("="*50)
    
    client = ClaudeClient(config)
    
    print(f"\nTesting {config.model}...")
    result = client.call(
        system="Reply with exactly: API_OK",
        user="Test connection",
        max_tokens=10
    )
    
    if result["success"]:
        print(f"  ✓ SUCCESS: {result['text'].strip()}")
        print(f"  Tokens used: {result.get('usage', {})}")
        return True
    else:
        print(f"  ✗ FAILED: {result['error']}")
        return False

def run_pipeline(config: Config, limit: Optional[int] = None):
    """Run the GTI classification pipeline."""
    
    print("\n" + "="*50)
    print("GTI CLAUDE-ONLY PIPELINE")
    print("="*50)
    
    # Get games to process
    games = get_unprocessed_games(config.db_path, limit)
    
    if not games:
        print("\n✓ No unprocessed games found!")
        return
    
    print(f"\nFound {len(games)} games to classify")
    print(f"Model: {config.model}")
    print(f"Delay between calls: {config.delay_between_calls}s")
    
    # Initialize client
    client = ClaudeClient(config)
    
    # Process games
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    for i, game in enumerate(games, 1):
        game_id = game["id"]
        game_name = game["name"]
        description = game.get("description", "No description available")
        
        print(f"\n[{i}/{len(games)}] {game_name[:50]}...")
        
        result = classify_game(client, game_name, description)
        
        if result["success"]:
            print(f"  ✓ {result['game_type']} (conf: {result.get('confidence', 0):.2f})")
            save_result(config.db_path, game_id, game_name, result, config.model)
            success_count += 1
        else:
            print(f"  ✗ Error: {result['error'][:100]}")
            error_count += 1
        
        # Rate limit delay
        if i < len(games):
            time.sleep(config.delay_between_calls)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
    print(f"  Processed: {len(games)}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Total API calls: {client.total_calls}")
    print(f"  Total tokens: {client.total_tokens:,}")

# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GTI Claude-Only Pipeline")
    parser.add_argument("--test", action="store_true", help="Test API connectivity")
    parser.add_argument("--run", type=str, help="Run pipeline: number or 'all'")
    parser.add_argument("--db", type=str, default="games2p2k.db", help="Database path")
    
    args = parser.parse_args()
    
    config = Config()
    config.db_path = Path(args.db)
    
    if args.test:
        test_api(config)
    elif args.run:
        limit = None if args.run.lower() == "all" else int(args.run)
        run_pipeline(config, limit)
    else:
        parser.print_help()
        print("\n\nQuick start:")
        print("  python gti_claude_only.py --test")
        print("  python gti_claude_only.py --run 5")

if __name__ == "__main__":
    main()