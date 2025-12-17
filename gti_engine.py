#!/usr/bin/env python3
"""
GTI Production Engine v3.0
==========================
Production-grade Game Theory Intelligence with:
- Pydantic type-safe schemas
- Deception detection
- Scientific visualization
- Machine-readable output

Usage:
  python gti_engine.py --text "Your scenario"
  python gti_engine.py --file scenario.txt --visualize
  python gti_engine.py --interactive

Requirements:
  pip install anthropic pydantic matplotlib

Author: GTI Project
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple
from uuid import uuid4

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    print("WARNING: pip install pydantic for type safety")

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# =============================================================================
# PYDANTIC SCHEMA DEFINITIONS (Type-Safe Architecture)
# =============================================================================

class GameType(str, Enum):
    """12 canonical Robinson-Goforth game types (orthogonal classification)"""
    PRISONERS_DILEMMA = "Prisoners_Dilemma"
    CHICKEN = "Chicken"
    STAG_HUNT = "Stag_Hunt"
    BATTLE_OF_SEXES = "Battle_of_the_Sexes"
    COORDINATION = "Coordination_Game"
    HERO = "Hero"
    DEADLOCK = "Deadlock"
    COMPROMISE = "Compromise"
    ASSURANCE = "Assurance_Game"
    HARMONY = "Harmony"
    ASYMMETRIC_DILEMMA = "Asymmetric_Dilemma"
    CYCLIC = "Cyclic_Game"
    # Out of scope markers
    OUT_OF_SCOPE = "Out_of_Scope"  # Zero-sum, sequential, n-player, etc.
    UNKNOWN = "Unknown"


class RiskLevel(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical_MAD"


class Stability(str, Enum):
    STABLE = "Stable"
    UNSTABLE = "Unstable"
    VOLATILE = "Volatile"
    TRUST_DEPENDENT = "Trust_Dependent"


if HAS_PYDANTIC:
    class Player(BaseModel):
        name: str
        motive: str
        stated_preference: List[str] = []
        revealed_preference: List[str] = []
        irrationality_index: float = 0.0

    class PayoffMatrix(BaseModel):
        T_temptation: str = ""
        R_reward: str = ""
        P_punishment: str = ""
        S_sucker: str = ""
        player_a_order: List[str] = ["T", "R", "P", "S"]
        player_b_order: List[str] = ["T", "R", "P", "S"]
        payoff_CC: Tuple[float, float] = (3.0, 3.0)
        payoff_CD: Tuple[float, float] = (0.0, 5.0)
        payoff_DC: Tuple[float, float] = (5.0, 0.0)
        payoff_DD: Tuple[float, float] = (1.0, 1.0)

    class DeceptionMetrics(BaseModel):
        is_deceptive: bool = False
        deception_score: float = 0.0
        cheap_talk_detected: bool = False
        stated_game: Optional[str] = None
        revealed_game: Optional[str] = None
        credibility_score: float = 1.0
        contradiction_evidence: List[str] = []
        cheap_talk_evidence: List[str] = []
        credible_signals: List[str] = []

    class Equilibrium(BaseModel):
        primary_outcome: str = ""
        secondary_outcome: Optional[str] = None
        stability: Stability = Stability.STABLE
        stability_score: float = 0.5
        reasoning: str = ""

    class RiskAssessment(BaseModel):
        level: RiskLevel = RiskLevel.MODERATE
        sensitivity: str = ""
        mad_potential: bool = False
        reasoning: str = ""

    class Prescription(BaseModel):
        pivot_variable: str = ""
        mechanism_design: str = ""
        player_a_advice: str = ""
        player_b_advice: str = ""
        mediator_advice: str = ""
        victim_warning: Optional[str] = None

    class StrategicDossier(BaseModel):
        scan_id: str = ""
        timestamp: str = ""
        version: str = "3.0.0"
        game_type: GameType = GameType.UNKNOWN
        game_description: str = ""
        confidence_score: float = 0.0
        players: List[Player] = []
        matrix: PayoffMatrix = PayoffMatrix()
        deception: DeceptionMetrics = DeceptionMetrics()
        equilibrium: Equilibrium = Equilibrium()
        risk: RiskAssessment = RiskAssessment()
        prescription: Prescription = Prescription()
        input_summary: str = ""

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are the Game Theory Engine v3.0. You decode strategic interactions from text into formal mathematical models.

### CORE DIRECTIVE
You extract INCENTIVES, not summaries. Words lie. Actions reveal truth.

### CLASSIFICATION FRAMEWORK: ROBINSON-GOFORTH ORTHOGONAL TYPES
You MUST classify into ONE of these 12 canonical 2x2 ordinal game types (based on T/R/P/S preference ordering):

| Type | Player A Order | Key Feature |
|------|---------------|-------------|
| Prisoners_Dilemma | T>R>P>S | Defection dominant, mutual coop preferred to mutual defect |
| Chicken | T>R>S>P | Mutual defection is WORST (catastrophe) |
| Stag_Hunt | R>T>P>S | Cooperation best but risky |
| Deadlock | T>P>R>S | Both prefer mutual defection |
| Harmony | R>T>S>P | Cooperation dominant |
| Battle_of_the_Sexes | Asymmetric | Coordinate but different preferences |
| Hero | Asymmetric | One must sacrifice |
| Compromise | Mixed | Middle-ground stable |
| Assurance_Game | R>T>P>S | Same as Stag Hunt (trust-based) |
| Coordination_Game | Match | Pure coordination, no conflict |
| Asymmetric_Dilemma | Unequal | Different orderings per player |
| Cyclic_Game | Circular | No pure Nash equilibrium |

### OUT OF SCOPE DETECTION
If the scenario is NOT a 2x2 simultaneous ordinal game, classify as "Out_of_Scope":
- Zero-sum games (Matching Pennies, Rock-Paper-Scissors) → Out_of_Scope
- Sequential games (Trust Game, Ultimatum) → Out_of_Scope  
- N-player games (>2 players without reduction) → Out_of_Scope
- Cardinal payoff games (exact numbers matter, not ordering) → Out_of_Scope

For Out_of_Scope, still provide analysis but note WHY it doesn't fit the framework.

### ANALYSIS PROTOCOL

**Phase 1: Player Extraction**
- Identify agents and their PRIMARY motive (Survival, Profit, Honor, Fear, Greed)
- Irrationality Index: 0.0 = pure logic, 1.0 = pure emotion/honor
- Separate STATED preferences (quotes) from REVEALED preferences (actions)

**Phase 2: Payoff Mapping**
- T (Temptation): Benefit of defecting while other cooperates
- R (Reward): Benefit of mutual cooperation  
- P (Punishment): Cost of mutual defection
- S (Sucker): Cost of being exploited

Determine the ORDINAL preference order for each player, then match to canonical type.

**Phase 3: Deception Check**
Compare STATED vs REVEALED preferences:
- If different → is_deceptive = true
- Identify cheap talk (costless) vs credible signals (costly)
- Output both stated_game and revealed_game

**Phase 4: Equilibrium & Risk**
- If Chicken + irrationality > 0.5 → risk = Critical_MAD
- If PD + no communication → equilibrium = Defect/Defect

### OUTPUT FORMAT
Return STRICTLY VALID JSON:

{
  "scan_id": "GTI-XXXXXXXX",
  "timestamp": "ISO-8601",
  "version": "3.0.0",
  "game_type": "Prisoners_Dilemma|Chicken|Stag_Hunt|Battle_of_the_Sexes|Coordination_Game|Hero|Deadlock|Compromise|Assurance_Game|Harmony|Asymmetric_Dilemma|Cyclic_Game|Out_of_Scope|Unknown",
  "game_description": "One sentence based on ACTIONS",
  "confidence_score": 0.0-1.0,
  
  "players": [
    {
      "name": "Player A",
      "motive": "Primary driver",
      "stated_preference": ["claimed preferences"],
      "revealed_preference": ["actual preferences from actions"],
      "irrationality_index": 0.0-1.0
    },
    {"name": "Player B", "motive": "", "stated_preference": [], "revealed_preference": [], "irrationality_index": 0.0}
  ],
  
  "matrix": {
    "T_temptation": "What defector gains",
    "R_reward": "Mutual cooperation benefit",
    "P_punishment": "Mutual defection cost",
    "S_sucker": "Cooperator's loss",
    "player_a_order": ["T", "R", "P", "S"],
    "player_b_order": ["T", "R", "P", "S"],
    "payoff_CC": [3.0, 3.0],
    "payoff_CD": [0.0, 5.0],
    "payoff_DC": [5.0, 0.0],
    "payoff_DD": [1.0, 1.0]
  },
  
  "deception": {
    "is_deceptive": false,
    "deception_score": 0.0-1.0,
    "cheap_talk_detected": false,
    "stated_game": "Game words suggest or null",
    "revealed_game": "Game actions reveal or null",
    "credibility_score": 0.0-1.0,
    "contradiction_evidence": ["quotes showing mismatch"],
    "cheap_talk_evidence": ["costless statements"],
    "credible_signals": ["costly commitments"]
  },
  
  "equilibrium": {
    "primary_outcome": "e.g. Defect/Defect",
    "secondary_outcome": null,
    "stability": "Stable|Unstable|Volatile|Trust_Dependent",
    "stability_score": 0.0-1.0,
    "reasoning": "Why?"
  },
  
  "risk": {
    "level": "Low|Moderate|High|Critical_MAD",
    "sensitivity": "What if misread?",
    "mad_potential": false,
    "reasoning": ""
  },
  
  "prescription": {
    "pivot_variable": "T|R|P|S",
    "mechanism_design": "How to fix",
    "player_a_advice": "",
    "player_b_advice": "",
    "mediator_advice": "",
    "victim_warning": "Warning if deception or null"
  },
  
  "input_summary": "First 200 chars"
}

RETURN ONLY VALID JSON."""


# =============================================================================
# VISUALIZATION ENGINE
# =============================================================================

def render_strategic_matrix(dossier, filename="strategic_matrix.png", show_shadow=False):
    """Render publication-quality Game Theory Matrix."""
    if not HAS_MATPLOTLIB:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    COLORS = {
        'bg': '#fafafa', 'line': '#2c3e50', 'cooperate': '#27ae60',
        'defect': '#c0392b', 'equilibrium': '#3498db', 'risk': '#e74c3c',
        'text': '#2c3e50', 'light': '#7f8c8d'
    }
    
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    # Get data
    p1 = dossier.players[0].name if dossier.players else "Player A"
    p2 = dossier.players[1].name if len(dossier.players) > 1 else "Player B"
    matrix = dossier.matrix
    eq = dossier.equilibrium.primary_outcome
    
    # Draw grid
    ax.add_patch(patches.Rectangle((0, 0), 2, 2, fill=False, edgecolor=COLORS['line'], linewidth=3))
    ax.plot([1, 1], [0, 2], color=COLORS['line'], linewidth=2)
    ax.plot([0, 2], [1, 1], color=COLORS['line'], linewidth=2)
    
    # Highlight equilibrium
    eq_pos = {'Cooperate/Cooperate': (0, 1), 'Cooperate/Defect': (1, 1),
              'Defect/Cooperate': (0, 0), 'Defect/Defect': (1, 0)}
    
    if eq in eq_pos:
        x, y = eq_pos[eq]
        danger = 'Defect' in eq and dossier.risk.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        color = COLORS['risk'] if danger else COLORS['equilibrium']
        ax.add_patch(patches.Rectangle((x, y), 1, 1, color=color, alpha=0.25))
        ax.text(x + 0.5, y + 0.15, "★ NASH EQUILIBRIUM", ha='center', fontsize=8,
                fontweight='bold', color=color)
    
    # Labels
    ax.text(-0.3, 1.5, p1[:15], ha='right', fontsize=11, fontweight='bold', color=COLORS['text'])
    ax.text(-0.3, 1.3, "Cooperate", ha='right', fontsize=9, color=COLORS['cooperate'], style='italic')
    ax.text(-0.3, 0.5, p1[:15], ha='right', fontsize=11, fontweight='bold', color=COLORS['text'])
    ax.text(-0.3, 0.3, "Defect", ha='right', fontsize=9, color=COLORS['defect'], style='italic')
    
    ax.text(0.5, 2.25, p2[:15], ha='center', fontsize=11, fontweight='bold', color=COLORS['text'])
    ax.text(0.5, 2.05, "Cooperate", ha='center', fontsize=9, color=COLORS['cooperate'], style='italic')
    ax.text(1.5, 2.25, p2[:15], ha='center', fontsize=11, fontweight='bold', color=COLORS['text'])
    ax.text(1.5, 2.05, "Defect", ha='center', fontsize=9, color=COLORS['defect'], style='italic')
    
    # Payoffs
    def fmt(p): return f"({p[0]:.0f}, {p[1]:.0f})"
    
    cells = {
        (0, 1): (matrix.payoff_CC, "R,R", "Mutual Cooperation"),
        (1, 1): (matrix.payoff_CD, "S,T", f"{p1[:8]} Exploited"),
        (0, 0): (matrix.payoff_DC, "T,S", f"{p2[:8]} Exploited"),
        (1, 0): (matrix.payoff_DD, "P,P", "Mutual Defection"),
    }
    
    for (x, y), (payoff, label, desc) in cells.items():
        ax.text(x + 0.5, y + 0.55, fmt(payoff), ha='center', fontsize=16,
                fontweight='bold', fontfamily='serif', color=COLORS['text'])
        ax.text(x + 0.5, y + 0.8, f"({label})", ha='center', fontsize=9, color=COLORS['light'])
        ax.text(x + 0.5, y + 0.35, desc, ha='center', fontsize=8, style='italic', color=COLORS['light'])
    
    # Title
    game_name = dossier.game_type.value.replace('_', ' ')
    prefix = "⚠️ REVEALED: " if show_shadow and dossier.deception.is_deceptive else ""
    plt.suptitle(f"{prefix}{game_name}", fontsize=16, fontweight='bold', y=0.98)
    
    conf = int(dossier.confidence_score * 100)
    risk = dossier.risk.level.value.replace('_', ' ')
    ax.set_title(f"Confidence: {conf}% | Risk: {risk} | {dossier.equilibrium.stability.value}",
                 fontsize=10, color=COLORS['light'], y=1.02)
    
    # Deception warning
    if dossier.deception.is_deceptive and not show_shadow:
        warning = f"⚠️ DECEPTION: Words={dossier.deception.stated_game}, Actions={dossier.deception.revealed_game}"
        ax.text(1.0, -0.35, warning, ha='center', fontsize=9, color=COLORS['risk'],
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='#ffe6e6', edgecolor=COLORS['risk']))
    
    ax.text(1.0, -0.6, f"ID: {dossier.scan_id}", ha='center', fontsize=8, color=COLORS['light'])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    return filename


# =============================================================================
# GTI ENGINE
# =============================================================================

class GTIEngine:
    """The Magical Funnel: Text → Strategic Insight"""
    
    def __init__(self, model="claude-sonnet-4-20250514"):
        if not HAS_ANTHROPIC:
            raise ImportError("pip install anthropic")
        self.client = Anthropic()
        self.model = model
    
    def analyze(self, text: str, scan_id: str = None):
        """Analyze strategic situation and return typed dossier."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"ANALYZE THIS SCENARIO:\n\n{text}"}]
        )
        
        raw = response.content[0].text.strip()
        
        # Parse JSON
        try:
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            data = json.loads(raw)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError("Invalid JSON from AI")
        
        if scan_id:
            data['scan_id'] = scan_id
        else:
            data['scan_id'] = data.get('scan_id', f"GTI-{uuid4().hex[:8].upper()}")
        
        data['input_summary'] = text[:200] + "..." if len(text) > 200 else text
        data['timestamp'] = data.get('timestamp', datetime.now().isoformat())
        
        # Fix tuples
        if 'matrix' in data:
            for k in ['payoff_CC', 'payoff_CD', 'payoff_DC', 'payoff_DD']:
                if k in data['matrix'] and isinstance(data['matrix'][k], list):
                    data['matrix'][k] = tuple(data['matrix'][k])
        
        # Normalize game_type to valid enum value
        if 'game_type' in data:
            raw_game = data['game_type']
            valid_types = [e.value for e in GameType]
            
            if raw_game not in valid_types:
                # Map to canonical types or Out_of_Scope
                canonical_mappings = {
                    # Variations of canonical types
                    'Prisoners_Dilemma_Game': 'Prisoners_Dilemma',
                    'Game_of_Chicken': 'Chicken',
                    'Hawk_Dove': 'Chicken',
                    'Bach_or_Stravinsky': 'Battle_of_the_Sexes',
                    'Coordination': 'Coordination_Game',
                    'Pure_Coordination': 'Coordination_Game',
                    'Assurance': 'Assurance_Game',
                    'Trust_Game': 'Stag_Hunt',  # Trust games map to Stag Hunt structure
                }
                
                # Out of scope types (zero-sum, sequential, etc.)
                out_of_scope_keywords = [
                    'zero', 'sum', 'matching', 'pennies', 'rock', 'paper', 'scissors',
                    'ultimatum', 'dictator', 'sequential', 'centipede', 'signaling',
                    'auction', 'bargaining', 'repeated', 'evolutionary'
                ]
                
                raw_lower = raw_game.lower()
                
                # Check if it's an out-of-scope type
                if any(kw in raw_lower for kw in out_of_scope_keywords):
                    data['game_type'] = 'Out_of_Scope'
                    print(f"INFO: '{raw_game}' is outside 2x2 ordinal framework → Out_of_Scope")
                # Check direct mapping
                elif raw_game in canonical_mappings:
                    data['game_type'] = canonical_mappings[raw_game]
                # Fuzzy match to canonical types
                elif 'chicken' in raw_lower or 'hawk' in raw_lower:
                    data['game_type'] = 'Chicken'
                elif 'prisoner' in raw_lower or 'dilemma' in raw_lower:
                    data['game_type'] = 'Prisoners_Dilemma'
                elif 'stag' in raw_lower or 'hunt' in raw_lower:
                    data['game_type'] = 'Stag_Hunt'
                elif 'battle' in raw_lower or 'sexes' in raw_lower:
                    data['game_type'] = 'Battle_of_the_Sexes'
                elif 'deadlock' in raw_lower:
                    data['game_type'] = 'Deadlock'
                elif 'harmony' in raw_lower:
                    data['game_type'] = 'Harmony'
                elif 'hero' in raw_lower:
                    data['game_type'] = 'Hero'
                elif 'coord' in raw_lower:
                    data['game_type'] = 'Coordination_Game'
                else:
                    # Default to Unknown for truly unknown types
                    print(f"WARNING: Cannot map '{raw_game}' to canonical type → Unknown")
                    data['game_type'] = 'Unknown'
        
        if HAS_PYDANTIC:
            return StrategicDossier(**data)
        return type('Dossier', (), data)()
    
    def analyze_with_viz(self, text: str, output_dir: str = ".", scan_id: str = None):
        """Analyze and generate visualizations."""
        dossier = self.analyze(text, scan_id)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        images = []
        
        img = render_strategic_matrix(dossier, f"{output_dir}/{dossier.scan_id}_matrix.png")
        if img:
            images.append(img)
        
        if dossier.deception.is_deceptive:
            img2 = render_strategic_matrix(dossier, f"{output_dir}/{dossier.scan_id}_revealed.png", True)
            if img2:
                images.append(img2)
        
        return dossier, images


# =============================================================================
# OUTPUT FORMATTER
# =============================================================================

def format_dossier_text(d) -> str:
    """Format dossier as text report."""
    conf_bar = "█" * int(d.confidence_score * 10) + "░" * (10 - int(d.confidence_score * 10))
    
    deception_banner = ""
    if d.deception.is_deceptive:
        dec_bar = "█" * int(d.deception.deception_score * 10) + "░" * (10 - int(d.deception.deception_score * 10))
        deception_banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🚨 DECEPTION DETECTED                                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Words suggest:   {str(d.deception.stated_game or 'Unknown'):<57} ║
║  Actions reveal:  {str(d.deception.revealed_game or 'Unknown'):<57} ║
║  Credibility:     [{dec_bar}] {int(d.deception.credibility_score * 100):>3}%{' ' * 35} ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    
    p1 = d.players[0] if d.players else type('P', (), {'name': 'Player A', 'motive': 'Unknown', 'stated_preference': [], 'revealed_preference': [], 'irrationality_index': 0})()
    p2 = d.players[1] if len(d.players) > 1 else type('P', (), {'name': 'Player B', 'motive': 'Unknown', 'stated_preference': [], 'revealed_preference': [], 'irrationality_index': 0})()
    
    return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     GTI STRATEGIC DOSSIER v3.0                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Scan ID:    {d.scan_id:<63} ║
║  Timestamp:  {d.timestamp:<63} ║
╚══════════════════════════════════════════════════════════════════════════════╝
{deception_banner}
═══════════════════════════════════════════════════════════════════════════════
 CLASSIFICATION
═══════════════════════════════════════════════════════════════════════════════

  Game Type:   {d.game_type.value if hasattr(d.game_type, 'value') else d.game_type:<60}
  Confidence:  [{conf_bar}] {int(d.confidence_score * 100)}%
  Description: {d.game_description[:70]}

═══════════════════════════════════════════════════════════════════════════════
 PLAYERS
═══════════════════════════════════════════════════════════════════════════════

  {p1.name}:
    Motive:      {p1.motive}
    Rationality: {int((1 - p1.irrationality_index) * 100)}%
    SAYS:        {' > '.join(p1.stated_preference[:3]) if p1.stated_preference else 'N/A'}
    DOES:        {' > '.join(p1.revealed_preference[:3]) if p1.revealed_preference else 'N/A'}

  {p2.name}:
    Motive:      {p2.motive}
    Rationality: {int((1 - p2.irrationality_index) * 100)}%
    SAYS:        {' > '.join(p2.stated_preference[:3]) if p2.stated_preference else 'N/A'}
    DOES:        {' > '.join(p2.revealed_preference[:3]) if p2.revealed_preference else 'N/A'}

═══════════════════════════════════════════════════════════════════════════════
 PAYOFF MATRIX
═══════════════════════════════════════════════════════════════════════════════

  T (Temptation): {d.matrix.T_temptation[:58]}
  R (Reward):     {d.matrix.R_reward[:58]}
  P (Punishment): {d.matrix.P_punishment[:58]}
  S (Sucker):     {d.matrix.S_sucker[:58]}

                        {p2.name}
                    Cooperate      Defect
  {p1.name[:10]:<10}  Coop   {str(d.matrix.payoff_CC):^12}  {str(d.matrix.payoff_CD):^12}
              Defect {str(d.matrix.payoff_DC):^12}  {str(d.matrix.payoff_DD):^12}

═══════════════════════════════════════════════════════════════════════════════
 EQUILIBRIUM & RISK
═══════════════════════════════════════════════════════════════════════════════

  Primary Outcome:  {d.equilibrium.primary_outcome}
  Stability:        {d.equilibrium.stability.value if hasattr(d.equilibrium.stability, 'value') else d.equilibrium.stability}
  Risk Level:       {d.risk.level.value if hasattr(d.risk.level, 'value') else d.risk.level}
  MAD Potential:    {"🚨 YES" if d.risk.mad_potential else "No"}
  
  Reasoning: {d.equilibrium.reasoning[:65]}

═══════════════════════════════════════════════════════════════════════════════
 PRESCRIPTION
═══════════════════════════════════════════════════════════════════════════════

  Pivot Variable: [{d.prescription.pivot_variable}]
  Mechanism:      {d.prescription.mechanism_design[:60]}
  
  For {p1.name}: {d.prescription.player_a_advice[:60]}
  For {p2.name}: {d.prescription.player_b_advice[:60]}
  For Mediator:  {d.prescription.mediator_advice[:60]}
{f'''
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🚨 VICTIM WARNING: {d.prescription.victim_warning[:55] if d.prescription.victim_warning else ""}
└─────────────────────────────────────────────────────────────────────────────┘
''' if d.prescription.victim_warning else ''}
═══════════════════════════════════════════════════════════════════════════════
"""


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GTI Production Engine v3.0")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Scenario text')
    input_group.add_argument('--file', type=str, help='File containing scenario')
    input_group.add_argument('--interactive', action='store_true')
    
    parser.add_argument('--format', choices=['text', 'json', 'both'], default='text')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--output', type=str, default='.')
    parser.add_argument('--id', type=str)
    
    args = parser.parse_args()
    
    # Get input
    if args.interactive:
        print("=" * 70)
        print("GTI PRODUCTION ENGINE v3.0")
        print("=" * 70)
        print("\nEnter scenario (Enter twice to finish):\n")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        text = "\n".join(lines[:-1])
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = args.text
    
    if not text.strip():
        print("ERROR: No input")
        sys.exit(1)
    
    print("\n⚙️  Analyzing...")
    engine = GTIEngine()
    
    start = time.time()
    if args.visualize:
        dossier, images = engine.analyze_with_viz(text, args.output, args.id)
        print(f"📊 Visualizations: {', '.join(images)}")
    else:
        dossier = engine.analyze(text, args.id)
    
    elapsed = time.time() - start
    
    if dossier.deception.is_deceptive:
        print(f"🚨 DECEPTION DETECTED ({elapsed:.1f}s)")
        print(f"   Words: {dossier.deception.stated_game}")
        print(f"   Actions: {dossier.deception.revealed_game}")
    else:
        print(f"✅ Complete ({elapsed:.1f}s)")
    
    print(f"   Game: {dossier.game_type.value if hasattr(dossier.game_type, 'value') else dossier.game_type}")
    print(f"   Confidence: {int(dossier.confidence_score * 100)}%")
    
    if args.format in ['text', 'both']:
        print(format_dossier_text(dossier))
    
    if args.format in ['json', 'both']:
        if HAS_PYDANTIC:
            print(dossier.model_dump_json(indent=2))
        else:
            print(json.dumps(dossier.__dict__, indent=2, default=str))
    
    # Save
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(f"{args.output}/{dossier.scan_id}.json", 'w') as f:
        if HAS_PYDANTIC:
            f.write(dossier.model_dump_json(indent=2))
        else:
            f.write(json.dumps(dossier.__dict__, indent=2, default=str))
    print(f"\n💾 Saved: {args.output}/{dossier.scan_id}.json")


if __name__ == "__main__":
    main()
