#!/usr/bin/env python3
"""
GTI Blind Test Runner
=====================
Run this from your GTI folder with your API key set.

Usage:
  python run_blind_test.py
"""

import time
import json
from datetime import datetime
from anthropic import Anthropic

SYSTEM_PROMPT = """You are a game theory expert classifying strategic situations.

Analyze the scenario and identify the game type based on the strategic structure:

CONFLICT GAMES (temptation to defect):
- Prisoners_Dilemma: Mutual cooperation is good, but each is tempted to defect. c > a > d > b
- Chicken: Mutual defection is CATASTROPHIC (worst outcome). c > a > b > d
- Deadlock: Mutual defection is actually preferred by both sides. c > d > a > b

COORDINATION GAMES:
- Battle_of_the_Sexes: Both want to coordinate, but prefer different equilibria. c > b > a > d
- Coordination_Game: Pure matching—no preference which option, just need to match. a > d > b > c

TRUST GAMES (cooperation needs assurance):
- Stag_Hunt: High reward from mutual cooperation, but risky—need trust. a > c > d > b
- Assurance_Game: Similar to Stag Hunt, but being abandoned is worse. a > d > c > b

OTHER:
- Hero: Someone must sacrifice for the group (volunteer's dilemma). c > b > d > a
- Compromise: Mutual concession beats mutual stubbornness. c > d > b > a

Output ONLY the game type name, nothing else."""

BLIND_CASES = [
    {
        "id": "BLIND-001",
        "title": "The Price War (Corporate Memo)",
        "description": """Listen, I know we're bleeding cash on this price war with Competitor X. 
Every time we drop our subscription fee by $5, they match it. We're both 
making 40% less revenue than we were a year ago. Ideally, we'd both just 
raise our prices back to the standard market rate tomorrow. But if we 
raise ours and they keep theirs low, we lose 90% of our customers overnight. 
I suspect they feel the exact same way. So, unless we can get a guarantee 
they will hike prices too (which is illegal collusion), we have to keep 
our prices at rock bottom."""
    },
    {
        "id": "BLIND-002",
        "title": "The Depleted Lake (Local News)",
        "description": """Tensions are rising in the coastal town of Oakhaven. The local lake has 
always supported the community, but yields are dropping fast. Marine 
biologists warn that the fish population is on the brink of collapse and 
needs a season to recover. However, the three major fishing families are 
intensifying their efforts. 'If I stop fishing to let the stock recover,' 
said one captain, 'my neighbor will just go out and catch the fish I left 
behind. I'd love to save the lake, but I have a mortgage to pay, and I 
can't let them get rich at my expense.' Everyone is now fishing double shifts."""
    },
    {
        "id": "BLIND-003",
        "title": "The Friday Night Plan (Text Messages)",
        "description": """So here's the issue. We both want to hang out tonight. I really want to 
go see that new sci-fi movie, but she hates sci-fi and really wants to 
go to the opera. I find the opera incredibly boring. However, the worst 
possible outcome is that we get into a fight and stay home alone in 
separate apartments. I'd rather suffer through the opera with her than 
be alone, and I know she'd rather sit through the sci-fi movie with me 
than be alone. We just have to pick one."""
    },
    {
        "id": "BLIND-004",
        "title": "The Interrogation Room (Crime Report)",
        "description": """Police have arrested two members of the 'Red Hand' gang. They are currently 
being interrogated in separate wings of the precinct. There isn't enough 
hard evidence to convict either of them for the major robbery—only a minor 
trespassing charge that carries 1 year in jail. The DA is offering a deal: 
'If you testify against your partner, we will let you walk free, and your 
partner gets 10 years.' The catch? If both of them testify against each 
other, the deal is off, and they both get 5 years. They have no way to 
communicate."""
    },
    {
        "id": "BLIND-005",
        "title": "The One-Lane Bridge (Traffic Incident)",
        "description": """Two cars are speeding toward a one-lane bridge from opposite directions. 
It is night. Neither driver is slowing down. If one swerves into the ditch, 
they damage their car and look like a coward, while the other driver 
crosses the bridge smoothly. If neither swerves, they will collide head-on, 
likely resulting in fatal injuries. Both drivers are flashing their high 
beams, signaling that they intend to stay the course."""
    },
    {
        "id": "BLIND-006",
        "title": "The July Crisis 1914 (Historical Document)",
        "description": """The Imperial Government of Austria-Hungary asserts that the murder of 
the Archduke was plotted in Belgrade. We can no longer tolerate the 
machinations of the Serbian government. We demand that Serbian officers 
allow Austrian officials into Serbia to suppress this movement. We give 
Serbia 48 hours to agree to all terms. If they refuse, we will break 
diplomatic relations and take necessary measures. We know that Russia 
may intervene to protect Serbia, which would force our ally Germany to 
mobilize, but we cannot back down and look weak in the eyes of our subjects. 
Our existence as a Great Power is at stake."""
    },
    {
        "id": "BLIND-007",
        "title": "The Eco-Partnership Deception (Press Release + Leaked Memo)",
        "description": """The CEO of Company A issued a press release today praising the new 
eco-friendly partnership with Company B: 'We are fully committed to this 
joint venture to reduce emissions. It is the right thing to do for the 
planet, regardless of the cost.' However, leaked internal memos reveal 
that Company A has secretly slashed the budget for this project to zero. 
They are waiting for Company B to invest the initial capital so they can 
use Company B's technology for free. If Company B finds out, the deal dies. 
If Company B invests blindly, Company A posts record profits this quarter."""
    },
]

def main():
    client = Anthropic()
    
    print("=" * 65)
    print("GTI BLIND TEST - 7 SCENARIOS (including deception test)")
    print("=" * 65)
    print()
    
    results = []
    
    for case in BLIND_CASES:
        print(f"Classifying: {case['id']} - {case['title']}")
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Classify this strategic situation:\n\n{case['description']}\n\nGame type:"
            }]
        )
        
        predicted = response.content[0].text.strip()
        results.append({
            "id": case['id'],
            "title": case['title'],
            "predicted": predicted
        })
        
        print(f"  → {predicted}")
        print()
        time.sleep(0.5)
    
    print("=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print()
    print(f"{'ID':<12} {'Title':<35} {'Predicted':<20}")
    print("-" * 65)
    for r in results:
        print(f"{r['id']:<12} {r['title']:<35} {r['predicted']:<20}")
    
    print()
    print("=" * 65)
    print("Awaiting answer key for grading...")
    print("=" * 65)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"blind_test_results_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)
    print(f"\nSaved: {filename}")


if __name__ == "__main__":
    main()
