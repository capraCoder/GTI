#!/usr/bin/env python3
"""
GTI Strategic Dossier Generator v2.0
====================================
Enhanced with DECEPTION DETECTION.

Key Innovation: Separates STATED preferences (words) from REVEALED preferences (actions).
A naive AI looks at words. This tool looks at incentive structures.

Usage:
  python gti_dossier_v2.py --text "Your scenario here"
  python gti_dossier_v2.py --file scenario.txt
  python gti_dossier_v2.py --interactive

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

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "claude-sonnet-4-20250514"
VERSION = "2.0.0"

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PayoffMatrix:
    """Abstract payoff variables"""
    T: str = ""  # Temptation
    R: str = ""  # Reward
    P: str = ""  # Punishment
    S: str = ""  # Sucker

@dataclass
class PlayerPreferences:
    """Player preference order"""
    player_id: str = ""
    stated_preferences: List[str] = field(default_factory=list)  # What they SAY
    revealed_preferences: List[str] = field(default_factory=list)  # What they DO
    interpretation: str = ""

@dataclass 
class DeceptionAnalysis:
    """Deception detection results"""
    deception_detected: bool = False
    deception_score: int = 0  # 0-100
    stated_game: str = ""      # What the words suggest
    revealed_game: str = ""    # What the actions suggest
    contradiction_signals: List[str] = field(default_factory=list)
    cheap_talk_indicators: List[str] = field(default_factory=list)
    credible_signals: List[str] = field(default_factory=list)
    analysis: str = ""

@dataclass
class Evidence:
    """Evidence from text - separated by type"""
    # Verbal signals (what they SAY)
    verbal_cooperation: List[str] = field(default_factory=list)
    verbal_defection: List[str] = field(default_factory=list)
    # Action signals (what they DO)
    action_cooperation: List[str] = field(default_factory=list)
    action_defection: List[str] = field(default_factory=list)
    # Payoff signals
    payoff_signals: List[str] = field(default_factory=list)

@dataclass
class Confidence:
    """Confidence assessment"""
    score: int = 0
    naive_classification: str = ""  # What a naive word-matcher would say
    structural_classification: str = ""  # What incentive analysis reveals
    classification_divergence: bool = False  # Do they disagree?
    reasoning: str = ""

@dataclass
class Equilibrium:
    """Nash equilibrium prediction"""
    outcome: str = ""
    stability: str = ""
    reasoning: str = ""

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
    victim_warning: str = ""  # NEW: Warning for potential victims of deception

@dataclass
class StrategicDossier:
    """Complete Strategic Dossier v2.0"""
    # Metadata
    id: str = ""
    timestamp: str = ""
    input_text: str = ""
    
    # Section 1: Diagnosis
    game_type: str = ""  # FINAL classification (based on actions, not words)
    game_description: str = ""
    interaction_type: str = ""
    symmetry: str = ""
    timing: str = ""
    payoff_matrix: PayoffMatrix = field(default_factory=PayoffMatrix)
    player_a: PlayerPreferences = field(default_factory=PlayerPreferences)
    player_b: PlayerPreferences = field(default_factory=PlayerPreferences)
    
    # Section 2: Deception Analysis (NEW)
    deception: DeceptionAnalysis = field(default_factory=DeceptionAnalysis)
    
    # Section 3: Logic Trace
    evidence: Evidence = field(default_factory=Evidence)
    confidence: Confidence = field(default_factory=Confidence)
    
    # Section 4: Prediction
    equilibrium: Equilibrium = field(default_factory=Equilibrium)
    risk: RiskAssessment = field(default_factory=RiskAssessment)
    
    # Section 5: Prescription
    prescription: Prescription = field(default_factory=Prescription)

# =============================================================================
# ENHANCED SYSTEM PROMPT WITH DECEPTION DETECTION
# =============================================================================

DOSSIER_SYSTEM_PROMPT = """You are an expert game theorist with special training in DECEPTION DETECTION.

CRITICAL PRINCIPLE: Words lie. Actions reveal truth. Incentives don't lie.

When analyzing a scenario:
1. FIRST identify what players SAY (verbal signals, press releases, stated intentions)
2. THEN identify what players DO (actual actions, budget decisions, revealed behavior)
3. If words and actions CONTRADICT, the ACTIONS determine the true game type
4. "Cheap talk" (costless statements) should be heavily discounted
5. "Credible signals" (costly commitments) should be trusted

DECEPTION PATTERNS TO DETECT:
- Player says "cooperation" but takes defection actions
- Public statements contradict private/leaked information
- Stated preferences don't match revealed incentive structure
- One player trying to induce cooperation while planning defection

For any scenario, produce a Strategic Dossier in this JSON format:

{
  "game_type": "The TRUE game based on ACTIONS and INCENTIVES (not words)",
  "game_description": "Why this classification based on structural analysis",
  
  "interaction_type": "Zero-Sum or Non-Zero-Sum",
  "symmetry": "Symmetric or Asymmetric", 
  "timing": "Simultaneous or Sequential",
  
  "payoff_matrix": {
    "T_temptation": "What does defector gain while other cooperates?",
    "R_reward": "What do both gain from mutual cooperation?",
    "P_punishment": "What do both suffer from mutual defection?",
    "S_sucker": "What does cooperator suffer when other defects?"
  },
  
  "player_a": {
    "identity": "Who is Player A?",
    "stated_preferences": ["What they CLAIM to want, in order"],
    "revealed_preferences": ["What their ACTIONS show they want, in order"],
    "interpretation": "Do stated and revealed match? What does this reveal?"
  },
  
  "player_b": {
    "identity": "Who is Player B?",
    "stated_preferences": ["What they CLAIM to want"],
    "revealed_preferences": ["What their ACTIONS show"],
    "interpretation": "Analysis of B's position"
  },
  
  "deception_analysis": {
    "deception_detected": true/false,
    "deception_score": 0-100,
    "stated_game": "What a NAIVE classifier looking at words would say",
    "revealed_game": "What STRUCTURAL analysis of actions reveals",
    "contradiction_signals": ["Quote showing words contradict actions"],
    "cheap_talk_indicators": ["Costless statements that should be discounted"],
    "credible_signals": ["Costly commitments that can be trusted"],
    "analysis": "Detailed explanation of deception pattern if detected"
  },
  
  "evidence": {
    "verbal_cooperation": ["Quotes where player SAYS cooperative things"],
    "verbal_defection": ["Quotes where player SAYS aggressive things"],
    "action_cooperation": ["Evidence of actual cooperative BEHAVIOR"],
    "action_defection": ["Evidence of actual defection BEHAVIOR"],
    "payoff_signals": ["Quotes indicating payoff structure"]
  },
  
  "confidence": {
    "score": 85,
    "naive_classification": "What surface-level word analysis suggests",
    "structural_classification": "What incentive structure analysis reveals",
    "classification_divergence": true/false,
    "reasoning": "Why final classification, especially if naive differs from structural"
  },
  
  "equilibrium": {
    "outcome": "Predicted outcome given TRUE incentives",
    "stability": "Stable, Unstable/Volatile, or Trust-Dependent",
    "reasoning": "Why this equilibrium?"
  },
  
  "risk": {
    "level": "Safe, Moderate, or Critical",
    "sensitivity": "What happens if deception succeeds/fails?",
    "reasoning": "Risk analysis"
  },
  
  "prescription": {
    "mechanism": "What would fix the underlying incentive problem?",
    "lever_variable": "T, R, P, or S - which needs to change?",
    "player_a_advice": "Advice for Player A",
    "player_b_advice": "Advice for Player B",
    "mediator_advice": "Advice for neutral third party",
    "victim_warning": "WARNING for potential victim of deception (if detected)"
  }
}

GAME TYPE REFERENCE:
- Prisoners_Dilemma: T > R > P > S. Temptation to defect dominates. "I want to cooperate" but secretly defecting = classic PD deception.
- Chicken: T > R > S > P. Mutual defection is CATASTROPHIC.
- Battle_of_the_Sexes: Both want to coordinate but prefer different equilibria.
- Stag_Hunt: R > T > P > S. Mutual cooperation is best but requires trust.
- Coordination_Game: Just need to match.
- Hero: Someone must sacrifice.
- Deadlock: Both prefer mutual defection.

REMEMBER: If someone SAYS "partnership" but ACTS to exploit → it's NOT a partnership game, it's Prisoner's Dilemma with deception.

OUTPUT ONLY VALID JSON."""

# =============================================================================
# ANALYSIS ENGINE
# =============================================================================

class DossierGenerator:
    """Generates Strategic Dossiers with Deception Detection."""
    
    def __init__(self):
        self.client = Anthropic()
    
    def analyze(self, text: str, scenario_id: str = None) -> StrategicDossier:
        """Analyze text and generate Strategic Dossier."""
        
        if not scenario_id:
            scenario_id = f"GTI-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=3000,
            system=DOSSIER_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Analyze this strategic situation. Remember: ACTIONS reveal truth, not WORDS.\n\n{text}"
            }]
        )
        
        raw_response = response.content[0].text.strip()
        
        # Parse JSON
        try:
            json_str = raw_response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"Could not parse JSON: {e}")
        
        # Build dossier
        deception_data = data.get("deception_analysis", {})
        evidence_data = data.get("evidence", {})
        
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
            
            player_a=PlayerPreferences(
                player_id=data.get("player_a", {}).get("identity", "Player A"),
                stated_preferences=data.get("player_a", {}).get("stated_preferences", []),
                revealed_preferences=data.get("player_a", {}).get("revealed_preferences", []),
                interpretation=data.get("player_a", {}).get("interpretation", ""),
            ),
            
            player_b=PlayerPreferences(
                player_id=data.get("player_b", {}).get("identity", "Player B"),
                stated_preferences=data.get("player_b", {}).get("stated_preferences", []),
                revealed_preferences=data.get("player_b", {}).get("revealed_preferences", []),
                interpretation=data.get("player_b", {}).get("interpretation", ""),
            ),
            
            deception=DeceptionAnalysis(
                deception_detected=deception_data.get("deception_detected", False),
                deception_score=deception_data.get("deception_score", 0),
                stated_game=deception_data.get("stated_game", ""),
                revealed_game=deception_data.get("revealed_game", ""),
                contradiction_signals=deception_data.get("contradiction_signals", []),
                cheap_talk_indicators=deception_data.get("cheap_talk_indicators", []),
                credible_signals=deception_data.get("credible_signals", []),
                analysis=deception_data.get("analysis", ""),
            ),
            
            evidence=Evidence(
                verbal_cooperation=evidence_data.get("verbal_cooperation", []),
                verbal_defection=evidence_data.get("verbal_defection", []),
                action_cooperation=evidence_data.get("action_cooperation", []),
                action_defection=evidence_data.get("action_defection", []),
                payoff_signals=evidence_data.get("payoff_signals", []),
            ),
            
            confidence=Confidence(
                score=data.get("confidence", {}).get("score", 0),
                naive_classification=data.get("confidence", {}).get("naive_classification", ""),
                structural_classification=data.get("confidence", {}).get("structural_classification", ""),
                classification_divergence=data.get("confidence", {}).get("classification_divergence", False),
                reasoning=data.get("confidence", {}).get("reasoning", ""),
            ),
            
            equilibrium=Equilibrium(
                outcome=data.get("equilibrium", {}).get("outcome", ""),
                stability=data.get("equilibrium", {}).get("stability", ""),
                reasoning=data.get("equilibrium", {}).get("reasoning", ""),
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
                victim_warning=data.get("prescription", {}).get("victim_warning", ""),
            ),
        )
        
        return dossier

# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================

def format_dossier_text(d: StrategicDossier) -> str:
    """Format dossier as readable text report with deception analysis."""
    
    conf_bar = "█" * (d.confidence.score // 10) + "░" * (10 - d.confidence.score // 10)
    deception_bar = "█" * (d.deception.deception_score // 10) + "░" * (10 - d.deception.deception_score // 10)
    
    # Deception alert banner
    deception_banner = ""
    if d.deception.deception_detected:
        deception_banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ⚠️  DECEPTION DETECTED  ⚠️                                                   ║
║                                                                              ║
║  Words say: {d.deception.stated_game:<63}║
║  Actions reveal: {d.deception.revealed_game:<58}║
║                                                                              ║
║  CLASSIFICATION BASED ON ACTIONS, NOT WORDS                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    
    return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    UNIVERSAL STRATEGIC DOSSIER v2.0                          ║
║                      [ WITH DECEPTION DETECTION ]                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ID: {d.id:<71} ║
║  Generated: {d.timestamp:<64} ║
╚══════════════════════════════════════════════════════════════════════════════╝
{deception_banner}
═══════════════════════════════════════════════════════════════════════════════
 SECTION 1: THE DIAGNOSIS (What is REALLY happening?)
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ GAME CLASSIFICATION (Based on ACTIONS, not words)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ TRUE Game Type:    {d.game_type:<56} │
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
│ PLAYER ANALYSIS: STATED vs REVEALED PREFERENCES                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ {d.player_a.player_id}:
│   SAYS:  {' > '.join(d.player_a.stated_preferences[:3])[:65] if d.player_a.stated_preferences else 'N/A'}
│   DOES:  {' > '.join(d.player_a.revealed_preferences[:3])[:65] if d.player_a.revealed_preferences else 'N/A'}
│   Analysis: {d.player_a.interpretation[:60]}
│                                                                             │
│ {d.player_b.player_id}:
│   SAYS:  {' > '.join(d.player_b.stated_preferences[:3])[:65] if d.player_b.stated_preferences else 'N/A'}
│   DOES:  {' > '.join(d.player_b.revealed_preferences[:3])[:65] if d.player_b.revealed_preferences else 'N/A'}
│   Analysis: {d.player_b.interpretation[:60]}
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
 SECTION 2: DECEPTION ANALYSIS (Words vs Actions)
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ DECEPTION METER                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Deception Score: [{deception_bar}] {d.deception.deception_score}%
│   Status: {"🚨 DECEPTION DETECTED" if d.deception.deception_detected else "✓ No deception detected"}
│                                                                             │
│   Naive (word-based) classification:     {d.deception.stated_game:<30} │
│   Structural (action-based) classification: {d.deception.revealed_game:<27} │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CONTRADICTION SIGNALS (Where words ≠ actions)                               │
├─────────────────────────────────────────────────────────────────────────────┤
{chr(10).join(f'│   ⚠️ "{s[:66]}"' for s in d.deception.contradiction_signals[:3]) if d.deception.contradiction_signals else '│   (None detected)'}
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CHEAP TALK (Costless statements - discount heavily)                         │
├─────────────────────────────────────────────────────────────────────────────┤
{chr(10).join(f'│   💨 "{s[:66]}"' for s in d.deception.cheap_talk_indicators[:3]) if d.deception.cheap_talk_indicators else '│   (None detected)'}
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CREDIBLE SIGNALS (Costly commitments - can be trusted)                      │
├─────────────────────────────────────────────────────────────────────────────┤
{chr(10).join(f'│   ✓ "{s[:66]}"' for s in d.deception.credible_signals[:3]) if d.deception.credible_signals else '│   (None detected)'}
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ DECEPTION ANALYSIS                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ {d.deception.analysis[:73]}
│ {d.deception.analysis[73:146] if len(d.deception.analysis) > 73 else ''}
│ {d.deception.analysis[146:219] if len(d.deception.analysis) > 146 else ''}
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
 SECTION 3: EVIDENCE MAPPING (Separated by Type)
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ VERBAL SIGNALS (What they SAY)                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Cooperation rhetoric:                                                       │
{chr(10).join(f'│   🗣️ "{s[:66]}"' for s in d.evidence.verbal_cooperation[:2]) if d.evidence.verbal_cooperation else '│   (None)'}
│ Defection rhetoric:                                                         │
{chr(10).join(f'│   🗣️ "{s[:66]}"' for s in d.evidence.verbal_defection[:2]) if d.evidence.verbal_defection else '│   (None)'}
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ ACTION SIGNALS (What they DO) ← THESE DETERMINE CLASSIFICATION              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Cooperative actions:                                                        │
{chr(10).join(f'│   ✅ "{s[:66]}"' for s in d.evidence.action_cooperation[:2]) if d.evidence.action_cooperation else '│   (None)'}
│ Defection actions:                                                          │
{chr(10).join(f'│   ❌ "{s[:66]}"' for s in d.evidence.action_defection[:2]) if d.evidence.action_defection else '│   (None)'}
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CONFIDENCE ASSESSMENT                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Overall Confidence: [{conf_bar}] {d.confidence.score}%
│                                                                             │
│   Naive classification:      {d.confidence.naive_classification:<42} │
│   Structural classification: {d.confidence.structural_classification:<42} │
│   Classifications diverge?   {"⚠️ YES" if d.confidence.classification_divergence else "No":<42} │
│                                                                             │
│   Reasoning: {d.confidence.reasoning[:62]}
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
 SECTION 4: PREDICTION
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ NASH EQUILIBRIUM (Based on TRUE incentives)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Probable Outcome:  {d.equilibrium.outcome:<55} │
│ Stability:         {d.equilibrium.stability:<55} │
│                                                                             │
│ Reasoning: {d.equilibrium.reasoning[:63]}
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ RISK ASSESSMENT                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Risk Level:  {d.risk.level:<61} │
│                                                                             │
│ Sensitivity: {d.risk.sensitivity[:61]}
│                                                                             │
│ Analysis: {d.risk.reasoning[:64]}
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
 SECTION 5: STRATEGIC PRESCRIPTION
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
│ For {d.player_a.player_id}:
│   → {d.prescription.player_a_advice[:69]}
│                                                                             │
│ For {d.player_b.player_id}:
│   → {d.prescription.player_b_advice[:69]}
│                                                                             │
│ For a Mediator:
│   → {d.prescription.mediator_advice[:69]}
└─────────────────────────────────────────────────────────────────────────────┘
{"" if not d.prescription.victim_warning else f'''
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🚨 WARNING FOR POTENTIAL VICTIM                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ {d.prescription.victim_warning[:73]}
│ {d.prescription.victim_warning[73:146] if len(d.prescription.victim_warning) > 73 else ""}
└─────────────────────────────────────────────────────────────────────────────┘
'''}
═══════════════════════════════════════════════════════════════════════════════
                              END OF DOSSIER
═══════════════════════════════════════════════════════════════════════════════

KEY INSIGHT: {"⚠️ WORDS AND ACTIONS DIVERGE - Classification based on ACTIONS" if d.deception.deception_detected else "Words and actions are consistent"}
"""


def format_dossier_markdown(d: StrategicDossier) -> str:
    """Format dossier as Markdown with deception analysis."""
    
    deception_alert = ""
    if d.deception.deception_detected:
        deception_alert = f"""
> ⚠️ **DECEPTION DETECTED**
> 
> - Words suggest: **{d.deception.stated_game}**
> - Actions reveal: **{d.deception.revealed_game}**
> 
> Classification based on ACTIONS, not words.

"""
    
    return f"""# Universal Strategic Dossier v2.0

**ID:** `{d.id}`  
**Generated:** {d.timestamp}

{deception_alert}---

## Section 1: The Diagnosis

### Game Classification (Based on ACTIONS)

| Property | Value |
|----------|-------|
| **TRUE Game Type** | **{d.game_type}** |
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

### Player Analysis: Stated vs Revealed

**{d.player_a.player_id}:**
- **SAYS:** {' > '.join(d.player_a.stated_preferences) if d.player_a.stated_preferences else 'N/A'}
- **DOES:** {' > '.join(d.player_a.revealed_preferences) if d.player_a.revealed_preferences else 'N/A'}
- **Analysis:** {d.player_a.interpretation}

**{d.player_b.player_id}:**
- **SAYS:** {' > '.join(d.player_b.stated_preferences) if d.player_b.stated_preferences else 'N/A'}
- **DOES:** {' > '.join(d.player_b.revealed_preferences) if d.player_b.revealed_preferences else 'N/A'}
- **Analysis:** {d.player_b.interpretation}

---

## Section 2: Deception Analysis

### Deception Meter

🎯 **Deception Score: {d.deception.deception_score}%** {"🚨 DETECTED" if d.deception.deception_detected else "✓ Clear"}

| Classification Method | Result |
|----------------------|--------|
| Naive (word-based) | {d.deception.stated_game} |
| Structural (action-based) | {d.deception.revealed_game} |

### Contradiction Signals
{chr(10).join(f'- ⚠️ "{s}"' for s in d.deception.contradiction_signals) if d.deception.contradiction_signals else '- (None detected)'}

### Cheap Talk (Discount These)
{chr(10).join(f'- 💨 "{s}"' for s in d.deception.cheap_talk_indicators) if d.deception.cheap_talk_indicators else '- (None detected)'}

### Credible Signals (Trust These)
{chr(10).join(f'- ✓ "{s}"' for s in d.deception.credible_signals) if d.deception.credible_signals else '- (None detected)'}

### Analysis
{d.deception.analysis}

---

## Section 3: Evidence

### Verbal Signals (What They SAY)
**Cooperation:** {', '.join(d.evidence.verbal_cooperation) if d.evidence.verbal_cooperation else 'None'}
**Defection:** {', '.join(d.evidence.verbal_defection) if d.evidence.verbal_defection else 'None'}

### Action Signals (What They DO) ← Determines Classification
**Cooperation:** {', '.join(d.evidence.action_cooperation) if d.evidence.action_cooperation else 'None'}
**Defection:** {', '.join(d.evidence.action_defection) if d.evidence.action_defection else 'None'}

### Confidence
- **Score:** {d.confidence.score}%
- **Naive classification:** {d.confidence.naive_classification}
- **Structural classification:** {d.confidence.structural_classification}
- **Divergence:** {"⚠️ Yes" if d.confidence.classification_divergence else "No"}

---

## Section 4: Prediction

### Nash Equilibrium
| Property | Value |
|----------|-------|
| **Outcome** | {d.equilibrium.outcome} |
| **Stability** | {d.equilibrium.stability} |

**Reasoning:** {d.equilibrium.reasoning}

### Risk Assessment
- **Level:** {d.risk.level}
- **Sensitivity:** {d.risk.sensitivity}

---

## Section 5: Prescription

### Mechanism Design
**Lever:** Change **[{d.prescription.lever_variable}]**

{d.prescription.mechanism}

### Advice

| For | Recommendation |
|-----|----------------|
| **{d.player_a.player_id}** | {d.prescription.player_a_advice} |
| **{d.player_b.player_id}** | {d.prescription.player_b_advice} |
| **Mediator** | {d.prescription.mediator_advice} |

{f'### 🚨 Victim Warning{chr(10)}{chr(10)}**{d.prescription.victim_warning}**' if d.prescription.victim_warning else ''}

---

*Generated by GTI Strategic Dossier Generator v{VERSION}*
"""


def format_dossier_json(d: StrategicDossier) -> str:
    """Format dossier as JSON."""
    return json.dumps(asdict(d), indent=2, default=str)

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GTI Strategic Dossier Generator v2.0 (with Deception Detection)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gti_dossier_v2.py --text "The CEO said partnership but slashed the budget..."
  python gti_dossier_v2.py --file scenario.txt
  python gti_dossier_v2.py --interactive
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Scenario text')
    input_group.add_argument('--file', type=str, help='File containing scenario')
    input_group.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    parser.add_argument('--format', choices=['text', 'json', 'markdown', 'all'], 
                        default='text', help='Output format')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--id', type=str, help='Custom scenario ID')
    
    args = parser.parse_args()
    
    if not HAS_ANTHROPIC:
        print("ERROR: pip install anthropic")
        sys.exit(1)
    
    # Get input
    if args.interactive:
        print("=" * 70)
        print("GTI STRATEGIC DOSSIER v2.0 - With Deception Detection")
        print("=" * 70)
        print("\nEnter your scenario (press Enter twice to finish):\n")
        
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
    
    # Generate
    print("\nAnalyzing (checking for deception)...")
    generator = DossierGenerator()
    
    start = time.time()
    dossier = generator.analyze(text, args.id)
    elapsed = time.time() - start
    
    if dossier.deception.deception_detected:
        print(f"⚠️  DECEPTION DETECTED ({elapsed:.1f}s)")
        print(f"   Words say: {dossier.deception.stated_game}")
        print(f"   Actions reveal: {dossier.deception.revealed_game}")
    else:
        print(f"✓ Analysis complete ({elapsed:.1f}s)")
    
    # Output
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
        
        print(f"\nSaved: {base}.txt, {base}.json, {base}.md")
        output = format_dossier_text(dossier)
    
    if args.output and args.format != 'all':
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"\nSaved: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
