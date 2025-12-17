#!/usr/bin/env python3
"""
GTI Blind Test - Production Engine v3.0
=======================================
Runs 7 blind test scenarios through the GTI Production Engine.
Includes deception detection test (BLIND-007).

Usage:
  python run_blind_dossier.py
  python run_blind_dossier.py --output results/blind_dossiers/ --visualize
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

try:
    from gti_engine import GTIEngine, format_dossier_text, render_strategic_matrix
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    print("WARNING: gti_engine.py not found")

try:
    from anthropic import Anthropic
except ImportError:
    print("ERROR: pip install anthropic")
    sys.exit(1)

# =============================================================================
# BLIND TEST CASES (7 scenarios)
# =============================================================================

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
        "title": "The July Crisis 1914 (Historical)",
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
        "title": "The Eco-Partnership Deception (DECEPTION TEST)",
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

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GTI Blind Test - Production v3.0")
    parser.add_argument('--output', type=str, default='results/blind_dossiers')
    parser.add_argument('--visualize', action='store_true', help='Generate matrix visualizations')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 85)
    print("GTI BLIND TEST - PRODUCTION ENGINE v3.0")
    print("=" * 85)
    print(f"\nOutput: {output_dir}")
    print(f"Scenarios: {len(BLIND_CASES)}")
    print(f"Visualize: {args.visualize}")
    print()
    
    if HAS_ENGINE:
        engine = GTIEngine()
    else:
        engine = None
        print("⚠️  Running in fallback mode (no full dossier)")
    
    results = []
    
    for i, case in enumerate(BLIND_CASES):
        print(f"[{i+1}/{len(BLIND_CASES)}] {case['id']}: {case['title']}")
        
        if engine:
            start = time.time()
            
            if args.visualize:
                dossier, images = engine.analyze_with_viz(
                    case['description'], 
                    str(output_dir), 
                    case['id']
                )
                for img in images:
                    print(f"       📊 {img}")
            else:
                dossier = engine.analyze(case['description'], case['id'])
            
            elapsed = time.time() - start
            
            # Save text dossier
            with open(output_dir / f"{case['id']}_dossier.txt", 'w', encoding='utf-8') as f:
                f.write(format_dossier_text(dossier))
            
            # Save JSON
            with open(output_dir / f"{case['id']}.json", 'w', encoding='utf-8') as f:
                try:
                    f.write(dossier.model_dump_json(indent=2))
                except:
                    f.write(json.dumps(dossier.__dict__, indent=2, default=str))
            
            # Extract values safely
            game_type = dossier.game_type.value if hasattr(dossier.game_type, 'value') else str(dossier.game_type)
            conf = int(dossier.confidence_score * 100)
            is_deceptive = dossier.deception.is_deceptive
            stated = dossier.deception.stated_game
            revealed = dossier.deception.revealed_game
            eq = dossier.equilibrium.primary_outcome
            risk = dossier.risk.level.value if hasattr(dossier.risk.level, 'value') else str(dossier.risk.level)
            
            # Print summary
            deception_flag = " 🚨 DECEPTION" if is_deceptive else ""
            print(f"       → Game: {game_type}{deception_flag}")
            print(f"       → Confidence: {conf}%")
            if is_deceptive:
                print(f"       → Words say: {stated}")
                print(f"       → Actions show: {revealed}")
            print(f"       → Equilibrium: {eq}")
            print(f"       → Risk: {risk}")
            print(f"       → Time: {elapsed:.1f}s")
            
            results.append({
                "id": case['id'],
                "title": case['title'],
                "game_type": game_type,
                "confidence": conf,
                "deception_detected": is_deceptive,
                "stated_game": stated,
                "revealed_game": revealed,
                "equilibrium": eq,
                "risk": risk,
            })
        else:
            # Fallback: simple classification
            client = Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=50,
                messages=[{
                    "role": "user",
                    "content": f"Classify this game (Prisoners_Dilemma, Chicken, Battle_of_the_Sexes, Stag_Hunt, etc.):\n\n{case['description']}\n\nGame type:"
                }]
            )
            game_type = response.content[0].text.strip()
            print(f"       → {game_type}")
            results.append({"id": case['id'], "title": case['title'], "game_type": game_type})
        
        print()
        time.sleep(1)
    
    # Save summary
    summary_file = output_dir / "blind_test_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "version": "3.0.0",
            "total_cases": len(BLIND_CASES),
            "results": results
        }, f, indent=2)
    
    # Print summary table
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print()
    print(f"{'ID':<12} {'Title':<35} {'Game Type':<20} {'Conf':<6} {'Deception':<12}")
    print("-" * 90)
    
    for r in results:
        conf = f"{r.get('confidence', '?')}%" if 'confidence' in r else "?"
        deception = "🚨 YES" if r.get('deception_detected', False) else ""
        game = r.get('game_type', 'Unknown')
        print(f"{r['id']:<12} {r['title'][:33]:<35} {game:<20} {conf:<6} {deception:<12}")
    
    # Deception analysis highlight
    deception_cases = [r for r in results if r.get('deception_detected', False)]
    if deception_cases:
        print()
        print("=" * 90)
        print("🚨 DECEPTION ANALYSIS")
        print("=" * 90)
        for r in deception_cases:
            print(f"\n{r['id']}: {r['title']}")
            print(f"  Words suggest: {r.get('stated_game', 'N/A')}")
            print(f"  Actions reveal: {r.get('revealed_game', 'N/A')}")
            print(f"  TRUE classification: {r.get('game_type', 'Unknown')}")
    
    print()
    print(f"💾 Summary: {summary_file}")
    print(f"📁 Dossiers: {output_dir}/BLIND-XXX_dossier.txt")
    if args.visualize:
        print(f"📊 Matrices: {output_dir}/BLIND-XXX_matrix.png")
    print()
    print("=" * 90)
    print("AWAITING ANSWER KEY FOR GRADING")
    print("=" * 90)


if __name__ == "__main__":
    main()
