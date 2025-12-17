#!/usr/bin/env python3
"""
GTI Strategic Dossier Generator v1.0
====================================
Transforms any strategic situation into a Universal Strategic Dossier.

Features:
  - Game Classification with confidence scoring
  - Abstract Payoff Matrix (T, R, P, S)
  - Preference Order extraction
  - Evidence mapping with quotes
  - Nash Equilibrium prediction
  - Stability assessment
  - Mechanism design recommendations
  - Actionable prescriptions

Usage:
  python gti_dossier.py --text "Your scenario here"
  python gti_dossier.py --file scenario.txt
  python gti_dossier.py --interactive

Author: GTI Project
"""

import os
import sys
import json
import re
import argparse
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "claude-sonnet-4-20250514"
VERSION = "1.0.0"

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class InteractionType(Enum):
    ZERO_SUM = "Zero-Sum"
    NON_ZERO_SUM = "Non-Zero-Sum"

class Symmetry(Enum):
    SYMMETRIC = "Symmetric"
    ASYMMETRIC = "Asymmetric"

class Timing(Enum):
    SIMULTANEOUS = "Simultaneous"
    SEQUENTIAL = "Sequential"

class Stability(Enum):
    STABLE = "Stable"
    UNSTABLE = "Unstable/Volatile"
    TRUST_DEPENDENT = "Trust-Dependent"

class RiskLevel(Enum):
    SAFE = "Safe"
    MODERATE = "Moderate"
    CRITICAL = "Critical/MAD"

@dataclass
class PayoffMatrix:
    """Abstract payoff variables"""
    T: str = ""  # Temptation: benefit of cheating while other cooperates
    R: str = ""  # Reward: benefit if both cooperate
    P: str = ""  # Punishment: cost if both defect
    S: str = ""  # Sucker: cost of cooperating while other cheats

@dataclass
class PlayerPreferences:
    """Player preference order"""
    player_id: str = ""
    preference_order: List[str] = field(default_factory=list)
    interpretation: str = ""

@dataclass
class Evidence:
    """Evidence from text"""
    defection_signals: List[str] = field(default_factory=list)
    cooperation_signals: List[str] = field(default_factory=list)
    payoff_signals: List[str] = field(default_factory=list)

@dataclass
class Confidence:
    """Confidence assessment"""
    score: int = 0  # 0-100
    completeness: int = 0  # Did text mention all outcomes?
    consistency: int = 0   # Are signals consistent?
    explicitness: int = 0  # Explicit vs implicit?
    ambiguity_flag: bool = False
    reasoning: str = ""

@dataclass
class Equilibrium:
    """Nash equilibrium prediction"""
    outcome: str = ""
    stability: str = ""
    stability_reasoning: str = ""

@dataclass
class RiskAssessment:
    """Trembling hand risk"""
    level: str = ""
    sensitivity: str = ""
    reasoning: str = ""

@dataclass
class Prescription:
    """Strategic prescription"""
    mechanism: str = ""
    lever_variable: str = ""
    player_a_advice: str = ""
    player_b_advice: str = ""
    mediator_advice: str = ""

@dataclass
class StrategicDossier:
    """Complete Strategic Dossier"""
    # Metadata
    id: str = ""
    timestamp: str = ""
    input_text: str = ""
    
    # Section 1: Diagnosis
    game_type: str = ""
    game_description: str = ""
    interaction_type: str = ""
    symmetry: str = ""
    timing: str = ""
    payoff_matrix: PayoffMatrix = field(default_factory=PayoffMatrix)
    player_a_preferences: PlayerPreferences = field(default_factory=PlayerPreferences)
    player_b_preferences: PlayerPreferences = field(default_factory=PlayerPreferences)
    
    # Section 2: Logic Trace
    evidence: Evidence = field(default_factory=Evidence)
    confidence: Confidence = field(default_factory=Confidence)
    
    # Section 3: Prediction
    equilibrium: Equilibrium = field(default_factory=Equilibrium)
    risk: RiskAssessment = field(default_factory=RiskAssessment)
    
    # Section 4: Prescription
    prescription: Prescription = field(default_factory=Prescription)

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

DOSSIER_SYSTEM_PROMPT = """You are an expert game theorist analyzing strategic situations. 

For any scenario, produce a comprehensive Strategic Dossier in the following JSON format:

{
  "game_type": "One of: Prisoners_Dilemma, Chicken, Battle_of_the_Sexes, Stag_Hunt, Coordination_Game, Hero, Deadlock, Compromise, Assurance_Game",
  "game_description": "One sentence explaining why this game type",
  
  "interaction_type": "Zero-Sum or Non-Zero-Sum",
  "symmetry": "Symmetric or Asymmetric", 
  "timing": "Simultaneous or Sequential",
  
  "payoff_matrix": {
    "T_temptation": "What does Player gain by defecting while other cooperates?",
    "R_reward": "What do both gain from mutual cooperation?",
    "P_punishment": "What do both suffer from mutual defection?",
    "S_sucker": "What does Player suffer by cooperating while other defects?"
  },
  
  "player_a": {
    "identity": "Who is Player A?",
    "preference_order": ["Best outcome", "Second best", "Third", "Worst"],
    "interpretation": "What does this preference order reveal about Player A's psychology?"
  },
  
  "player_b": {
    "identity": "Who is Player B?",
    "preference_order": ["Best outcome", "Second best", "Third", "Worst"],
    "interpretation": "What does this preference order reveal about Player B's psychology?"
  },
  
  "evidence": {
    "defection_signals": ["Quote 1 showing aggression/selfishness", "Quote 2..."],
    "cooperation_signals": ["Quote 1 showing willingness to cooperate", "Quote 2..."],
    "payoff_signals": ["Quote 1 indicating fear/desire of specific outcome", "Quote 2..."]
  },
  
  "confidence": {
    "score": 85,
    "completeness": "Did text mention all 4 outcomes? Full/Partial/Minimal",
    "consistency": "Are signals consistent or contradictory?",
    "explicitness": "Are payoffs explicit or implicit?",
    "ambiguity_flag": false,
    "reasoning": "Why this confidence level?"
  },
  
  "equilibrium": {
    "outcome": "The mathematically probable outcome (e.g., Defect/Defect)",
    "stability": "Stable, Unstable/Volatile, or Trust-Dependent",
    "reasoning": "Why is this the equilibrium and why this stability?"
  },
  
  "risk": {
    "level": "Safe, Moderate, or Critical/MAD",
    "sensitivity": "What happens if one player misreads the other?",
    "reasoning": "Does error self-correct or explode?"
  },
  
  "prescription": {
    "mechanism": "What mechanism would change the game?",
    "lever_variable": "Which variable (T, R, P, S) needs to change?",
    "player_a_advice": "Concrete advice for Player A",
    "player_b_advice": "Concrete advice for Player B", 
    "mediator_advice": "Advice for a neutral third party"
  }
}

GAME TYPE REFERENCE:
- Prisoners_Dilemma: T > R > P > S. Temptation to defect dominates, but mutual defection is worse than mutual cooperation.
- Chicken: T > R > S > P. Mutual defection is CATASTROPHIC (worst outcome). Brinkmanship.
- Battle_of_the_Sexes: Both want to coordinate but prefer different equilibria.
- Stag_Hunt: R > T > P > S. Mutual cooperation is best but requires trust.
- Coordination_Game: Just need to match, no preference which option.
- Hero: Someone must sacrifice for the group (volunteer's dilemma).
- Deadlock: Both actually prefer mutual defection.
- Compromise: Mutual concession beats mutual stubbornness.

OUTPUT ONLY VALID JSON. No markdown, no explanation outside the JSON."""

# =============================================================================
# ANALYSIS ENGINE
# =============================================================================

class DossierGenerator:
    """Generates Strategic Dossiers from text."""
    
    def __init__(self):
        self.client = Anthropic()
    
    def analyze(self, text: str, scenario_id: str = None) -> StrategicDossier:
        """Analyze text and generate Strategic Dossier."""
        
        if not scenario_id:
            scenario_id = f"GTI-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Call LLM
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=DOSSIER_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Analyze this strategic situation and produce a Strategic Dossier:\n\n{text}"
            }]
        )
        
        raw_response = response.content[0].text.strip()
        
        # Parse JSON
        try:
            # Clean up response (remove markdown if present)
            json_str = raw_response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Fallback: try to extract JSON from response
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"Could not parse JSON response: {e}")
        
        # Build dossier
        dossier = StrategicDossier(
            id=scenario_id,
            timestamp=datetime.now().isoformat(),
            input_text=text[:500] + "..." if len(text) > 500 else text,
            
            game_type=data.get("game_type", "Unknown"),
            game_description=data.get("game_description", ""),
            interaction_type=data.get("interaction_type", ""),
            symmetry=data.get("symmetry", ""),
            timing=data.get("timing", ""),
            
            payoff_matrix=PayoffMatrix(
                T=data.get("payoff_matrix", {}).get("T_temptation", ""),
                R=data.get("payoff_matrix", {}).get("R_reward", ""),
                P=data.get("payoff_matrix", {}).get("P_punishment", ""),
                S=data.get("payoff_matrix", {}).get("S_sucker", ""),
            ),
            
            player_a_preferences=PlayerPreferences(
                player_id=data.get("player_a", {}).get("identity", "Player A"),
                preference_order=data.get("player_a", {}).get("preference_order", []),
                interpretation=data.get("player_a", {}).get("interpretation", ""),
            ),
            
            player_b_preferences=PlayerPreferences(
                player_id=data.get("player_b", {}).get("identity", "Player B"),
                preference_order=data.get("player_b", {}).get("preference_order", []),
                interpretation=data.get("player_b", {}).get("interpretation", ""),
            ),
            
            evidence=Evidence(
                defection_signals=data.get("evidence", {}).get("defection_signals", []),
                cooperation_signals=data.get("evidence", {}).get("cooperation_signals", []),
                payoff_signals=data.get("evidence", {}).get("payoff_signals", []),
            ),
            
            confidence=Confidence(
                score=data.get("confidence", {}).get("score", 0),
                completeness=data.get("confidence", {}).get("completeness", ""),
                consistency=data.get("confidence", {}).get("consistency", ""),
                explicitness=data.get("confidence", {}).get("explicitness", ""),
                ambiguity_flag=data.get("confidence", {}).get("ambiguity_flag", False),
                reasoning=data.get("confidence", {}).get("reasoning", ""),
            ),
            
            equilibrium=Equilibrium(
                outcome=data.get("equilibrium", {}).get("outcome", ""),
                stability=data.get("equilibrium", {}).get("stability", ""),
                stability_reasoning=data.get("equilibrium", {}).get("reasoning", ""),
            ),
            
            risk=RiskAssessment(
                level=data.get("risk", {}).get("level", ""),
                sensitivity=data.get("risk", {}).get("sensitivity", ""),
                reasoning=data.get("risk", {}).get("reasoning", ""),
            ),
            
            prescription=Prescription(
                mechanism=data.get("prescription", {}).get("mechanism", ""),
                lever_variable=data.get("prescription", {}).get("lever_variable", ""),
                player_a_advice=data.get("prescription", {}).get("player_a_advice", ""),
                player_b_advice=data.get("prescription", {}).get("player_b_advice", ""),
                mediator_advice=data.get("prescription", {}).get("mediator_advice", ""),
            ),
        )
        
        return dossier

# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================

def format_dossier_text(d: StrategicDossier) -> str:
    """Format dossier as readable text report."""
    
    conf_bar = "█" * (d.confidence.score // 10) + "░" * (10 - d.confidence.score // 10)
    
    return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        UNIVERSAL STRATEGIC DOSSIER                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ID: {d.id:<71} ║
║  Generated: {d.timestamp:<64} ║
╚══════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
 SECTION 1: THE DIAGNOSIS (What is happening?)
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ GAME CLASSIFICATION                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Primary Model:     {d.game_type:<56} │
│ Description:       {d.game_description[:56]:<56} │
├─────────────────────────────────────────────────────────────────────────────┤
│ Interaction Type:  {d.interaction_type:<56} │
│ Symmetry:          {d.symmetry:<56} │
│ Timing:            {d.timing:<56} │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ ABSTRACT PAYOFF MATRIX                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ T (Temptation): {d.payoff_matrix.T[:58]:<58} │
│ R (Reward):     {d.payoff_matrix.R[:58]:<58} │
│ P (Punishment): {d.payoff_matrix.P[:58]:<58} │
│ S (Sucker):     {d.payoff_matrix.S[:58]:<58} │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PLAYER PREFERENCES (The "DNA")                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ {d.player_a_preferences.player_id}:
│   Order: {' > '.join(d.player_a_preferences.preference_order[:4])[:65]}
│   Logic: {d.player_a_preferences.interpretation[:65]}
├─────────────────────────────────────────────────────────────────────────────┤
│ {d.player_b_preferences.player_id}:
│   Order: {' > '.join(d.player_b_preferences.preference_order[:4])[:65]}
│   Logic: {d.player_b_preferences.interpretation[:65]}
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
 SECTION 2: THE LOGIC TRACE (Why do I think this?)
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ EVIDENCE MAPPING                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Defection Signals:                                                          │
{chr(10).join(f'│   • "{s[:68]}"' for s in d.evidence.defection_signals[:3])}
├─────────────────────────────────────────────────────────────────────────────┤
│ Cooperation Signals:                                                        │
{chr(10).join(f'│   • "{s[:68]}"' for s in d.evidence.cooperation_signals[:3])}
├─────────────────────────────────────────────────────────────────────────────┤
│ Payoff Signals:                                                             │
{chr(10).join(f'│   • "{s[:68]}"' for s in d.evidence.payoff_signals[:3])}
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CONFIDENCE METER                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Score: [{conf_bar}] {d.confidence.score}%
│                                                                             │
│   Completeness:  {d.confidence.completeness:<57} │
│   Consistency:   {d.confidence.consistency:<57} │
│   Explicitness:  {d.confidence.explicitness:<57} │
│   Ambiguity:     {"⚠️ YES - Multiple interpretations possible" if d.confidence.ambiguity_flag else "✓ NO - Clear signals":<57} │
│                                                                             │
│   Reasoning: {d.confidence.reasoning[:62]}
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
 SECTION 3: THE PREDICTION (What happens next?)
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ NASH EQUILIBRIUM                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Probable Outcome:  {d.equilibrium.outcome:<55} │
│ Stability:         {d.equilibrium.stability:<55} │
│                                                                             │
│ Reasoning: {d.equilibrium.stability_reasoning[:63]}
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ "TREMBLING HAND" RISK ASSESSMENT                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Risk Level:  {d.risk.level:<61} │
│                                                                             │
│ Sensitivity: {d.risk.sensitivity[:61]}
│                                                                             │
│ Analysis: {d.risk.reasoning[:64]}
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
 SECTION 4: THE STRATEGIC PRESCRIPTION (How to fix it?)
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ MECHANISM DESIGN                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ The Lever:  Change variable [{d.prescription.lever_variable}]
│                                                                             │
│ Mechanism: {d.prescription.mechanism[:63]}
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ ACTIONABLE ADVICE                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ For {d.player_a_preferences.player_id}:
│   → {d.prescription.player_a_advice[:69]}
│                                                                             │
│ For {d.player_b_preferences.player_id}:
│   → {d.prescription.player_b_advice[:69]}
│                                                                             │
│ For a Mediator:
│   → {d.prescription.mediator_advice[:69]}
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                              END OF DOSSIER
═══════════════════════════════════════════════════════════════════════════════
"""


def format_dossier_json(d: StrategicDossier) -> str:
    """Format dossier as JSON."""
    return json.dumps(asdict(d), indent=2, default=str)


def format_dossier_markdown(d: StrategicDossier) -> str:
    """Format dossier as Markdown."""
    
    conf_emoji = "🟢" if d.confidence.score >= 80 else "🟡" if d.confidence.score >= 50 else "🔴"
    
    return f"""# Universal Strategic Dossier

**ID:** `{d.id}`  
**Generated:** {d.timestamp}

---

## Section 1: The Diagnosis

### Game Classification

| Property | Value |
|----------|-------|
| **Primary Model** | {d.game_type} |
| **Description** | {d.game_description} |
| **Interaction Type** | {d.interaction_type} |
| **Symmetry** | {d.symmetry} |
| **Timing** | {d.timing} |

### Abstract Payoff Matrix

| Variable | Meaning |
|----------|---------|
| **T (Temptation)** | {d.payoff_matrix.T} |
| **R (Reward)** | {d.payoff_matrix.R} |
| **P (Punishment)** | {d.payoff_matrix.P} |
| **S (Sucker)** | {d.payoff_matrix.S} |

### Player Preferences

**{d.player_a_preferences.player_id}:**
- Order: {' > '.join(d.player_a_preferences.preference_order)}
- Interpretation: {d.player_a_preferences.interpretation}

**{d.player_b_preferences.player_id}:**
- Order: {' > '.join(d.player_b_preferences.preference_order)}
- Interpretation: {d.player_b_preferences.interpretation}

---

## Section 2: The Logic Trace

### Evidence Mapping

**Defection Signals:**
{chr(10).join(f'- "{s}"' for s in d.evidence.defection_signals)}

**Cooperation Signals:**
{chr(10).join(f'- "{s}"' for s in d.evidence.cooperation_signals)}

**Payoff Signals:**
{chr(10).join(f'- "{s}"' for s in d.evidence.payoff_signals)}

### Confidence Meter

{conf_emoji} **Score: {d.confidence.score}%**

| Check | Result |
|-------|--------|
| Completeness | {d.confidence.completeness} |
| Consistency | {d.confidence.consistency} |
| Explicitness | {d.confidence.explicitness} |
| Ambiguity | {"⚠️ Yes" if d.confidence.ambiguity_flag else "✓ No"} |

**Reasoning:** {d.confidence.reasoning}

---

## Section 3: The Prediction

### Nash Equilibrium

| Property | Value |
|----------|-------|
| **Probable Outcome** | {d.equilibrium.outcome} |
| **Stability** | {d.equilibrium.stability} |

**Reasoning:** {d.equilibrium.stability_reasoning}

### Trembling Hand Risk

| Property | Value |
|----------|-------|
| **Risk Level** | {d.risk.level} |
| **Sensitivity** | {d.risk.sensitivity} |

**Analysis:** {d.risk.reasoning}

---

## Section 4: The Strategic Prescription

### Mechanism Design

**The Lever:** Change variable **[{d.prescription.lever_variable}]**

**Mechanism:** {d.prescription.mechanism}

### Actionable Advice

| For | Advice |
|-----|--------|
| **{d.player_a_preferences.player_id}** | {d.prescription.player_a_advice} |
| **{d.player_b_preferences.player_id}** | {d.prescription.player_b_advice} |
| **Mediator** | {d.prescription.mediator_advice} |

---

*Generated by GTI Strategic Dossier Generator v{VERSION}*
"""

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GTI Strategic Dossier Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gti_dossier.py --text "Two firms are in a price war..."
  python gti_dossier.py --file scenario.txt
  python gti_dossier.py --interactive
  python gti_dossier.py --file scenario.txt --format markdown --output report.md
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Scenario text to analyze')
    input_group.add_argument('--file', type=str, help='File containing scenario')
    input_group.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    parser.add_argument('--format', choices=['text', 'json', 'markdown', 'all'], 
                        default='text', help='Output format')
    parser.add_argument('--output', type=str, help='Output file (default: stdout)')
    parser.add_argument('--id', type=str, help='Custom scenario ID')
    
    args = parser.parse_args()
    
    if not HAS_ANTHROPIC:
        print("ERROR: pip install anthropic")
        sys.exit(1)
    
    # Get input text
    if args.interactive:
        print("=" * 70)
        print("GTI STRATEGIC DOSSIER GENERATOR - Interactive Mode")
        print("=" * 70)
        print("\nEnter your scenario (press Enter twice to finish):\n")
        
        lines = []
        while True:
            line = input()
            if line == "":
                if lines and lines[-1] == "":
                    break
            lines.append(line)
        
        text = "\n".join(lines[:-1])  # Remove trailing empty line
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = args.text
    
    if not text.strip():
        print("ERROR: No input text provided")
        sys.exit(1)
    
    # Generate dossier
    print("\nAnalyzing scenario...")
    generator = DossierGenerator()
    
    start = time.time()
    dossier = generator.analyze(text, args.id)
    elapsed = time.time() - start
    
    print(f"Analysis complete ({elapsed:.1f}s)\n")
    
    # Format output
    if args.format == 'text':
        output = format_dossier_text(dossier)
    elif args.format == 'json':
        output = format_dossier_json(dossier)
    elif args.format == 'markdown':
        output = format_dossier_markdown(dossier)
    elif args.format == 'all':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = args.output or f"dossier_{timestamp}"
        
        with open(f"{base}.txt", 'w', encoding='utf-8') as f:
            f.write(format_dossier_text(dossier))
        with open(f"{base}.json", 'w', encoding='utf-8') as f:
            f.write(format_dossier_json(dossier))
        with open(f"{base}.md", 'w', encoding='utf-8') as f:
            f.write(format_dossier_markdown(dossier))
        
        print(f"Saved: {base}.txt, {base}.json, {base}.md")
        output = format_dossier_text(dossier)
    
    # Output
    if args.output and args.format != 'all':
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Saved: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
