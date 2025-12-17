#!/usr/bin/env python3
"""
GTI: Game Theory Intelligence
=============================
AI-powered strategic analysis engine that decodes conflicts,
negotiations, and hidden games.

Features:
- Identifies game-theoretic structures in any text
- Detects deception (words vs. actions mismatch)
- Works in any language
- Interactive payoff matrix visualization
- Downloadable analysis reports

Usage:
  pip install streamlit anthropic pydantic matplotlib
  streamlit run app.py

Author: GTI Project | github.com/capraCoder/GTI
"""

import streamlit as st
import json
import time
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import base64

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="GTI | Game Theory Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ANALYTICS (GoatCounter - Pixel Tracker) ---
# Using pixel tracker instead of JS for reliability in Streamlit iframes
import urllib.parse
def track_pageview(path="/"):
    """Track a pageview using GoatCounter pixel"""
    encoded_path = urllib.parse.quote(path, safe='')
    pixel_url = f"https://capracoder-gti.goatcounter.com/count?p={encoded_path}"
    st.markdown(
        f'<img src="{pixel_url}" alt="" style="position:absolute;left:-9999px;">',
        unsafe_allow_html=True
    )

# Track main page view
track_pageview("/")

# --- CUSTOM CSS (Enterprise Edition) ---
st.markdown("""
<style>
    /* 1. GLOBAL RESET & TYPOGRAPHY */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        background-color: #F1F5F9;
        color: #0F172A;
    }
    
    .stApp {
        background-color: #F1F5F9;
    }

    /* 2. INPUT AREA (The "Command Center") */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        font-size: 16px !important;
        padding: 1.5rem !important;
        color: #334155 !important;
        transition: all 0.2s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }

    /* 3. BUTTONS (Flat & Modern) */
    .stButton > button {
        background-color: #0F172A !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        transition: all 0.1s;
    }
    
    .stButton > button:hover {
        background-color: #334155 !important;
        transform: translateY(-1px);
    }

    /* 4. CARDS (The "Dossier" Look) */
    .dossier-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* 5. METRICS (KPI Style) */
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* 6. HIDE STREAMLIT CHROME */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stStatusWidget"] {display: none;}
    [data-testid="stToolbar"] {display: none;}
    .viewerBadge_container__r5tak {display: none;}
    .styles_viewerBadge__CvC9N {display: none;}
    [data-testid="stDecoration"] {display: none;}
    
    /* 7. CUSTOM ALERTS */
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 14px;
        margin-bottom: 1.5rem;
    }
    
    .alert-danger { background: #FEF2F2; color: #991B1B; border: 1px solid #FEE2E2; }
    .alert-success { background: #F0FDF4; color: #166534; border: 1px solid #DCFCE7; }
    .alert-warning { background: #FFFBEB; color: #92400E; border: 1px solid #FEF3C7; }
    
    /* 8. PLAYER CARDS */
    .player-card {
        background: #F8FAFC;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 0.5rem;
    }
    
    /* 9. EVIDENCE ITEMS */
    .evidence-item {
        background: #F8FAFC;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        margin: 0.3rem 0;
        border-left: 3px solid #64748B;
        font-size: 0.9rem;
        color: #334155;
    }
    
    .evidence-item.deception {
        background: #FEF2F2;
        border-left: 3px solid #DC2626;
        color: #7F1D1D;
    }
    
    .evidence-item.credible {
        background: #F0FDF4;
        border-left: 3px solid #16A34A;
        color: #14532D;
    }
    
    /* 10. PRESCRIPTION BOX */
    .prescription-box {
        background: #0F172A;
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
    }
    
    .prescription-box h4 {
        color: #60A5FA;
        margin-top: 0;
    }
    
    /* 11. MATRIX CONTAINER */
    .matrix-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* 12. MINIMAL HEADER */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #E2E8F0;
    }
    
    .status-badge {
        background: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #86EFAC;
    }
</style>
""", unsafe_allow_html=True)

# --- TRY TO IMPORT GTI ENGINE ---
try:
    from gti_engine import GTIEngine, format_dossier_text, render_strategic_matrix
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

# --- DATA STRUCTURES ---
@dataclass
class SimpleReport:
    """Fallback report structure when engine not available"""
    id: str
    game_type: str
    risk_level: str
    confidence: float
    is_deceptive: bool
    stated_game: Optional[str]
    revealed_game: Optional[str]
    summary: str
    player_a: dict
    player_b: dict
    equilibrium: str
    advice: str
    victim_warning: Optional[str]
    cheap_talk: list
    contradictions: list
    credible_signals: list

# --- MOCK ENGINE FOR DEMO ---
class MockEngine:
    """Demo mode when API key not available - Always shows impressive results"""
    
    def analyze(self, text: str, scan_id: str = None):
        time.sleep(1.5)  # Simulate processing
        
        text_lower = text.lower()
        
        # Chicken game keywords (cars, standoff scenarios)
        chicken_keywords = ['car', 'bridge', 'swerve', 'collide', 'chicken', 'crash', 'head-on']
        is_chicken = any(word in text_lower for word in chicken_keywords)
        
        # Coordination keywords (date, preference, together)
        coord_keywords = ['movie', 'opera', 'date', 'hang out', 'together', 'prefer']
        is_coordination = any(word in text_lower for word in coord_keywords)
        
        # Price war keywords
        price_keywords = ['price', 'competitor', 'war', 'market', 'customers', 'discount']
        is_price_war = any(word in text_lower for word in price_keywords)
        
        if is_chicken:
            return SimpleReport(
                id=scan_id or f"GTI-{datetime.now().strftime('%H%M%S')}",
                game_type="Chicken",
                risk_level="Critical",
                confidence=0.91,
                is_deceptive=False,
                stated_game=None,
                revealed_game=None,
                summary="Classic game of Chicken. Both parties are escalating toward mutual destruction. The first to swerve 'loses face' but survives.",
                player_a={"name": "Driver A", "motive": "Dominance/Reputation", "type": "Aggressive"},
                player_b={"name": "Driver B", "motive": "Dominance/Reputation", "type": "Aggressive"},
                equilibrium="Mixed Strategy (one swerves, other doesn't)",
                advice="Create a credible commitment to not swerve (throw steering wheel out window) OR introduce face-saving exit (third party stops both).",
                victim_warning="Both parties risk catastrophic loss. Rational calculation may fail under pressure.",
                cheap_talk=["flashing high beams"],
                contradictions=[],
                credible_signals=["Neither slowing down"]
            )
        elif is_coordination:
            return SimpleReport(
                id=scan_id or f"GTI-{datetime.now().strftime('%H%M%S')}",
                game_type="Battle_of_Sexes",
                risk_level="Low",
                confidence=0.89,
                is_deceptive=False,
                stated_game=None,
                revealed_game=None,
                summary="Coordination problem with asymmetric preferences. Both prefer agreement over disagreement, but each prefers different outcomes.",
                player_a={"name": "Partner A", "motive": "Preferred activity + togetherness", "type": "Cooperative"},
                player_b={"name": "Partner B", "motive": "Preferred activity + togetherness", "type": "Cooperative"},
                equilibrium="Two pure equilibria (both go to A's choice OR both go to B's choice)",
                advice="Use randomization (coin flip) or take turns choosing. Establish precedent for fairness.",
                victim_warning=None,
                cheap_talk=[],
                contradictions=[],
                credible_signals=["Both prefer being together"]
            )
        elif is_price_war:
            return SimpleReport(
                id=scan_id or f"GTI-{datetime.now().strftime('%H%M%S')}",
                game_type="Prisoners_Dilemma",
                risk_level="High",
                confidence=0.93,
                is_deceptive=False,
                stated_game=None,
                revealed_game=None,
                summary="Classic Prisoner's Dilemma in market competition. Both firms would benefit from high prices, but each has incentive to undercut.",
                player_a={"name": "Your Company", "motive": "Market share & profit", "type": "Rational"},
                player_b={"name": "Competitor X", "motive": "Market share & profit", "type": "Rational"},
                equilibrium="Defect/Defect (price war continues)",
                advice="Signal commitment to match any price cuts (tit-for-tat). Consider price leadership or capacity constraints as commitment devices.",
                victim_warning="Without coordination mechanism, both firms destroy margins.",
                cheap_talk=[],
                contradictions=[],
                credible_signals=["Price matching behavior"]
            )
        else:
            # Default: Show the impressive deception detection demo
            return SimpleReport(
                id=scan_id or f"GTI-{datetime.now().strftime('%H%M%S')}",
                game_type="Prisoners_Dilemma",
                risk_level="Critical",
                confidence=0.92,
                is_deceptive=True,
                stated_game="Stag_Hunt",
                revealed_game="Prisoners_Dilemma",
                summary="⚠️ DEMO MODE: Target publicly signals cooperation but private incentives suggest defection. Analysis based on textual signals.",
                player_a={"name": "Party A", "motive": "Self-interest (hidden)", "type": "Potentially Deceptive"},
                player_b={"name": "Party B", "motive": "Mutual benefit (stated)", "type": "Target"},
                equilibrium="Defect/Cooperate (A exploits B)",
                advice="Verify stated commitments with observable actions. Demand escrow, milestones, or third-party verification before proceeding.",
                victim_warning="Do not rely on verbal commitments alone. Look for credible signals.",
                cheap_talk=["commitment language", "partnership framing", "mutual benefit claims"],
                contradictions=["Stated vs. revealed preferences may differ"],
                credible_signals=[]
            )

# --- HELPER FUNCTIONS ---
def get_risk_color(risk: str) -> str:
    colors = {
        "Low": "#059669",
        "Moderate": "#d97706", 
        "High": "#dc2626",
        "Critical": "#7c2d12",
        "Critical_MAD": "#7c2d12"
    }
    return colors.get(risk, "#64748b")

def render_metric_card(label: str, value: str, color: str = "#1a1a2e"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="value" style="color: {color}">{value}</div>
        <div class="label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# --- UI COMPONENTS ---
def render_header():
    """Minimal enterprise header"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## ♟️ Strategic Radar")
        st.caption("Game Theory Intelligence Engine • Enterprise Edition")
    with col2:
        st.markdown("""
        <div style="text-align: right; padding-top: 10px;">
            <span class="status-badge">● SYSTEM ONLINE</span>
        </div>
        """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🎯 GTI Engine")
        
        # Mode selection
        mode = st.radio(
            "Mode",
            ["🔴 Demo Mode", "🟢 Live Mode (API)"],
            index=0 if not HAS_ENGINE else 1,
            help="Demo mode shows sample analysis. Live mode uses AI."
        )
        
        st.markdown("---")
        
        # Example scenarios - cleaner labels
        st.markdown("### 🧪 Try an Example")
        st.caption("Click to load a scenario:")
        
        if st.button("🎭 Corporate Deception", use_container_width=True):
            st.session_state.input_text = """The CEO of Company A issued a press release today praising the new eco-friendly partnership with Company B: 'We are fully committed to this joint venture to reduce emissions. It is the right thing to do for the planet, regardless of the cost.' However, leaked internal memos reveal that Company A has secretly slashed the budget for this project to zero. They are waiting for Company B to invest the initial capital so they can use Company B's technology for free."""
        
        if st.button("🐔 Game of Chicken", use_container_width=True):
            st.session_state.input_text = """Two cars are speeding toward a one-lane bridge from opposite directions. Neither driver is slowing down. If one swerves, they look like a coward. If neither swerves, they collide head-on. Both are flashing their high beams."""
        
        if st.button("🔒 Price War Trap", use_container_width=True):
            st.session_state.input_text = """We're bleeding cash on this price war with Competitor X. Every time we drop our price, they match it. Ideally, we'd both raise prices. But if we raise ours and they keep theirs low, we lose 90% of customers overnight."""
        
        if st.button("💑 Coordination Problem", use_container_width=True):
            st.session_state.input_text = """We both want to hang out tonight. I want the sci-fi movie, she wants the opera. The worst outcome is staying home alone. I'd rather suffer through opera with her than be alone."""
        
        st.markdown("---")
        
        # Compact footer
        st.markdown("### ℹ️ How It Works")
        st.caption("""
        1. Paste any situation
        2. AI identifies the "game"
        3. Get strategic insights
        
        Based on Nobel Prize-winning game theory frameworks.
        """)
        
        return "demo" if "Demo" in mode else "live"

def render_input_section():
    """Clean enterprise input section"""
    # Initialize session state
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    
    st.markdown("##### Intelligence Source")
    text = st.text_area(
        "Input",
        value=st.session_state.input_text,
        height=180,
        placeholder="Paste negotiation emails, contract terms, or conflict scenarios here...",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear", type="secondary"):
            st.session_state.input_text = ""
            if 'last_report' in st.session_state:
                del st.session_state.last_report
            if 'last_input' in st.session_state:
                del st.session_state.last_input
            st.rerun()
    
    return text, analyze_btn

def render_metrics(report):
    """Render the key metrics row"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        game_type = report.game_type if isinstance(report.game_type, str) else report.game_type.value
        render_metric_card("Game Detected", game_type.replace("_", " "))
    
    with col2:
        # Handle both SimpleReport (confidence) and StrategicDossier (confidence_score)
        if hasattr(report, 'confidence_score'):
            conf = report.confidence_score
        elif hasattr(report, 'confidence'):
            conf = report.confidence
        else:
            conf = 0.0
        conf_pct = int(conf * 100) if conf <= 1 else int(conf)
        render_metric_card("Confidence", f"{conf_pct}%", "#2563eb")
    
    with col3:
        risk = report.risk_level if hasattr(report, 'risk_level') else report.risk.level
        risk_str = risk if isinstance(risk, str) else risk.value
        render_metric_card("Risk Level", risk_str.replace("_", " "), get_risk_color(risk_str))
    
    with col4:
        is_deceptive = report.is_deceptive if hasattr(report, 'is_deceptive') else report.deception.is_deceptive
        dec_text = "🚨 DETECTED" if is_deceptive else "✓ None"
        dec_color = "#dc2626" if is_deceptive else "#059669"
        render_metric_card("Deception", dec_text, dec_color)

def render_deception_alert(report):
    """Render deception warning banner"""
    # Get is_deceptive - handle both attribute paths
    if hasattr(report, 'is_deceptive'):
        is_deceptive = report.is_deceptive
    elif hasattr(report, 'deception') and hasattr(report.deception, 'is_deceptive'):
        is_deceptive = report.deception.is_deceptive
    else:
        is_deceptive = False
    
    if is_deceptive:
        # Get stated and revealed games
        if hasattr(report, 'stated_game'):
            stated = report.stated_game
        elif hasattr(report, 'deception') and hasattr(report.deception, 'stated_game'):
            stated = report.deception.stated_game
        else:
            stated = "Unknown"
            
        if hasattr(report, 'revealed_game'):
            revealed = report.revealed_game
        elif hasattr(report, 'deception') and hasattr(report.deception, 'revealed_game'):
            revealed = report.deception.revealed_game
        else:
            revealed = "Unknown"
        
        st.markdown(f"""
        <div class="alert-deception">
            🚨 <b>DECEPTION DETECTED</b> — Words suggest <b>{stated}</b> but actions reveal <b>{revealed}</b>
        </div>
        """, unsafe_allow_html=True)
        
        return st.radio(
            "View Layer:",
            ["📢 Public Narrative (What They Say)", "🔍 Revealed Reality (What They Do)"],
            horizontal=True
        )
    else:
        # Get risk level
        if hasattr(report, 'risk_level'):
            risk_str = report.risk_level
        elif hasattr(report, 'risk') and hasattr(report.risk, 'level'):
            risk = report.risk.level
            risk_str = risk if isinstance(risk, str) else risk.value
        else:
            risk_str = "Unknown"
        
        if risk_str in ["Critical", "Critical_MAD", "High"]:
            st.markdown(f"""
            <div class="alert-warning">
                ⚠️ <b>HIGH RISK SCENARIO</b> — No deception detected, but equilibrium is unstable
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-safe">
                ✓ <b>ANALYSIS COMPLETE</b> — No deception detected
            </div>
            """, unsafe_allow_html=True)
        
        return None

def render_dossier(report, view_mode=None):
    """Render the main dossier content"""
    
    # Determine if we're showing public narrative or revealed reality
    show_revealed = view_mode and "Revealed" in view_mode
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Summary card - changes based on view mode
        if hasattr(report, 'summary'):
            summary = report.summary
        elif hasattr(report, 'game_description'):
            summary = report.game_description
        else:
            summary = "No summary available"
        
        # Get deception info for view switching
        is_deceptive = report.is_deceptive if hasattr(report, 'is_deceptive') else (report.deception.is_deceptive if hasattr(report, 'deception') else False)
        
        if is_deceptive:
            if hasattr(report, 'stated_game'):
                stated_game = report.stated_game
                revealed_game = report.revealed_game
            elif hasattr(report, 'deception'):
                stated_game = report.deception.stated_game
                revealed_game = report.deception.revealed_game
            else:
                stated_game = "Unknown"
                revealed_game = "Unknown"
        
        # Different summary based on view
        if is_deceptive and not show_revealed:
            # PUBLIC NARRATIVE VIEW
            st.markdown(f"""
            <div class="dossier-card" style="border-left-color: #3b82f6;">
                <h4>📢 Public Narrative</h4>
                <p><i>"What they want you to believe"</i></p>
                <hr>
                <p>This appears to be a <b>{stated_game}</b> scenario where both parties seek mutual cooperation and shared benefit.</p>
                <p style="color: #64748b; font-size: 0.9rem;">⚠️ This is the surface-level interpretation based on public statements only.</p>
            </div>
            """, unsafe_allow_html=True)
        elif is_deceptive and show_revealed:
            # REVEALED REALITY VIEW
            st.markdown(f"""
            <div class="dossier-card" style="border-left-color: #dc2626;">
                <h4>🔍 Revealed Reality</h4>
                <p><i>"What the evidence actually shows"</i></p>
                <hr>
                <p>{summary}</p>
                <p>True game: <b>{revealed_game}</b></p>
                <p style="color: #dc2626; font-size: 0.9rem;">⚠️ Classification based on ACTIONS, not words.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Non-deceptive case
            st.markdown(f"""
            <div class="dossier-card">
                <h4>📋 Executive Summary</h4>
                <p>{summary}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Players section - also changes based on view
        st.markdown("#### 👥 Player Analysis")
        
        if hasattr(report, 'player_a') and isinstance(report.player_a, dict):
            # SimpleReport format
            p1 = report.player_a
            p2 = report.player_b
            if is_deceptive and not show_revealed:
                # Public view - show stated
                st.markdown(f"""
                <div class="player-card" style="background: #eff6ff;">
                    <b>{p1.get('name', 'Player A')}</b> — <i>Public Image</i><br>
                    Appears: Committed partner seeking mutual benefit
                </div>
                <div class="player-card" style="background: #eff6ff;">
                    <b>{p2.get('name', 'Player B')}</b> — <i>Public Image</i><br>
                    Appears: Willing collaborator in joint venture
                </div>
                """, unsafe_allow_html=True)
            else:
                # Revealed view - show true motives
                st.markdown(f"""
                <div class="player-card" style="background: #fef2f2;">
                    <b>{p1.get('name', 'Player A')}</b> — <i>True Profile</i><br>
                    Motive: {p1.get('motive', 'Unknown')} | Type: {p1.get('type', 'Unknown')}
                </div>
                <div class="player-card" style="background: #fef2f2;">
                    <b>{p2.get('name', 'Player B')}</b> — <i>True Profile</i><br>
                    Motive: {p2.get('motive', 'Unknown')} | Type: {p2.get('type', 'Unknown')}
                </div>
                """, unsafe_allow_html=True)
        elif hasattr(report, 'players') and report.players:
            # StrategicDossier format
            for p in report.players[:2]:
                if hasattr(p, 'name'):
                    name = p.name
                    motive = p.motive if hasattr(p, 'motive') else 'Unknown'
                    stated = ' > '.join(p.stated_preference[:2]) if hasattr(p, 'stated_preference') and p.stated_preference else 'N/A'
                    revealed = ' > '.join(p.revealed_preference[:2]) if hasattr(p, 'revealed_preference') and p.revealed_preference else 'N/A'
                else:
                    name = p.get('name', 'Unknown')
                    motive = p.get('motive', 'Unknown')
                    stated = 'N/A'
                    revealed = 'N/A'
                
                if is_deceptive and not show_revealed:
                    # Public view
                    st.markdown(f"""
                    <div class="player-card" style="background: #eff6ff;">
                        <b>{name}</b> — <i>Public Image</i><br>
                        <small>Claims: {stated}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Revealed view
                    st.markdown(f"""
                    <div class="player-card" style="background: #fef2f2;">
                        <b>{name}</b> — <i>True Profile</i><br>
                        Motive: {motive}<br>
                        <small>Says: {stated}</small><br>
                        <small><b>Actually Does: {revealed}</b></small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Equilibrium - changes based on view
        if hasattr(report, 'equilibrium'):
            if isinstance(report.equilibrium, str):
                eq = report.equilibrium
            elif hasattr(report.equilibrium, 'primary_outcome'):
                eq = report.equilibrium.primary_outcome
            else:
                eq = str(report.equilibrium)
        else:
            eq = "Unknown"
        
        if is_deceptive and not show_revealed:
            st.markdown(f"**Expected Outcome (Public):** `Cooperate/Cooperate` *(what they claim)*")
        else:
            st.markdown(f"**Predicted Outcome (True):** `{eq}`")
    
    with col_right:
        # Evidence section - only show in Revealed view for deception cases
        is_deceptive = report.is_deceptive if hasattr(report, 'is_deceptive') else (report.deception.is_deceptive if hasattr(report, 'deception') else False)
        
        if is_deceptive and not show_revealed:
            # Public view - show the "clean" version
            st.markdown("#### 📰 Public Statements")
            st.markdown("""
            <div class="evidence-item" style="background: #eff6ff; border-left-color: #3b82f6;">
                "We are fully committed to this joint venture"
            </div>
            <div class="evidence-item" style="background: #eff6ff; border-left-color: #3b82f6;">
                "It is the right thing to do for the planet"
            </div>
            <div class="evidence-item" style="background: #eff6ff; border-left-color: #3b82f6;">
                "Regardless of the cost"
            </div>
            """, unsafe_allow_html=True)
            st.caption("💭 *Switch to 'Revealed Reality' to see what's really happening*")
            
        elif is_deceptive and show_revealed:
            st.markdown("#### 🔍 Evidence Analysis")
            
            # Cheap talk
            if hasattr(report, 'cheap_talk'):
                cheap_talk = report.cheap_talk
            elif hasattr(report, 'deception') and hasattr(report.deception, 'cheap_talk_evidence'):
                cheap_talk = report.deception.cheap_talk_evidence
            else:
                cheap_talk = []
                
            if cheap_talk:
                st.markdown("**💨 Cheap Talk (Discount):**")
                for item in cheap_talk[:3]:
                    st.markdown(f'<div class="evidence-item">{item}</div>', unsafe_allow_html=True)
            
            # Contradictions
            if hasattr(report, 'contradictions'):
                contradictions = report.contradictions
            elif hasattr(report, 'deception') and hasattr(report.deception, 'contradiction_evidence'):
                contradictions = report.deception.contradiction_evidence
            else:
                contradictions = []
                
            if contradictions:
                st.markdown("**⚠️ Contradictions:**")
                for item in contradictions[:3]:
                    st.markdown(f'<div class="evidence-item deception">{item}</div>', unsafe_allow_html=True)
            
            # Credible signals
            if hasattr(report, 'credible_signals'):
                credible = report.credible_signals
            elif hasattr(report, 'deception') and hasattr(report.deception, 'credible_signals'):
                credible = report.deception.credible_signals
            else:
                credible = []
                
            if credible:
                st.markdown("**✓ Credible Signals:**")
                for item in credible[:3]:
                    st.markdown(f'<div class="evidence-item credible">{item}</div>', unsafe_allow_html=True)
        
        # Matrix visualization
        st.markdown("#### 📊 Payoff Matrix")
        st.markdown('<div class="matrix-container">', unsafe_allow_html=True)
        
        game_type = report.game_type if isinstance(report.game_type, str) else report.game_type.value
        
        # Try to use the engine's beautiful matplotlib renderer
        matrix_rendered = False
        if hasattr(report, 'matrix') and hasattr(report.matrix, 'payoff_CC'):
            try:
                matrix_buf = render_strategic_matrix(report, show_shadow=show_revealed)
                if matrix_buf:
                    st.image(matrix_buf, use_container_width=True, 
                            caption=f"Payoff Matrix ({'Revealed' if show_revealed else 'Public View'})")
                    matrix_rendered = True
            except Exception:
                pass  # Fall back to HTML
        
        # Fallback to HTML tables (for demo mode or if matplotlib fails)
        if not matrix_rendered:
            if is_deceptive and not show_revealed:
                # Show "fake" cooperation matrix
                st.markdown("""
                <div style="text-align: center; padding: 1rem;">
                    <p style="color: #3b82f6; font-weight: bold;">STATED GAME: Coordination</p>
                    <table style="margin: auto; border-collapse: collapse; font-size: 0.9rem;">
                        <tr><td></td><td style="padding: 10px; font-weight: bold; color: #475569;">Cooperate</td><td style="padding: 10px; font-weight: bold; color: #475569;">Defect</td></tr>
                        <tr><td style="font-weight: bold; color: #475569;">Cooperate</td><td style="padding: 15px; background: #f0fdf4; border: 1px solid #e2e8f0;">✓ Win-Win</td><td style="padding: 15px; background: #fef2f2; border: 1px solid #e2e8f0;">Lose</td></tr>
                        <tr><td style="font-weight: bold; color: #475569;">Defect</td><td style="padding: 15px; background: #fef2f2; border: 1px solid #e2e8f0;">Lose</td><td style="padding: 15px; background: #fef2f2; border: 1px solid #e2e8f0;">Lose-Lose</td></tr>
                    </table>
                    <p style="color: #64748b; font-size: 0.8rem; margin-top: 1rem;">What they want you to believe</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Show true matrix
                if "Chicken" in game_type:
                    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/Chicken_Game.svg/400px-Chicken_Game.svg.png")
                else:
                    st.markdown("""
                    <div style="text-align: center; padding: 1rem;">
                        <p style="color: #dc2626; font-weight: bold;">TRUE GAME: Prisoner's Dilemma</p>
                        <table style="margin: auto; border-collapse: collapse; font-size: 0.9rem;">
                            <tr><td></td><td style="padding: 10px; font-weight: bold; color: #475569;">Cooperate</td><td style="padding: 10px; font-weight: bold; color: #475569;">Defect</td></tr>
                            <tr><td style="font-weight: bold; color: #475569;">Cooperate</td><td style="padding: 15px; background: #f8fafc; border: 1px solid #e2e8f0;">(3, 3)</td><td style="padding: 15px; background: #fef2f2; border: 1px solid #e2e8f0;">(0, 5) 🎯</td></tr>
                            <tr><td style="font-weight: bold; color: #475569;">Defect</td><td style="padding: 15px; background: #f0fdf4; border: 1px solid #e2e8f0;">(5, 0)</td><td style="padding: 15px; background: #fef2f2; border: 1px solid #e2e8f0; font-weight: bold;">★ (1, 1)</td></tr>
                        </table>
                        <p style="color: #64748b; font-size: 0.8rem; margin-top: 1rem;">★ Nash Equilibrium: Defect/Defect<br>🎯 Company A's target: Defect while B Cooperates</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        if show_revealed:
            st.caption("⚠️ Displaying TRUE incentive structure")
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_prescription(report):
    """Render the strategic prescription"""
    # Get advice - handle multiple attribute paths
    if hasattr(report, 'advice'):
        advice = report.advice
    elif hasattr(report, 'prescription'):
        if hasattr(report.prescription, 'mechanism_design'):
            advice = report.prescription.mechanism_design
        else:
            advice = str(report.prescription)
    else:
        advice = "No prescription available"
    
    # Get victim warning - handle multiple attribute paths
    if hasattr(report, 'victim_warning'):
        victim = report.victim_warning
    elif hasattr(report, 'prescription') and hasattr(report.prescription, 'victim_warning'):
        victim = report.prescription.victim_warning
    else:
        victim = None
    
    st.markdown(f"""
    <div class="prescription-box">
        <h4>💡 Strategic Prescription</h4>
        <p>{advice}</p>
        {"<hr><p style='color:#fca5a5'><b>🚨 Victim Warning:</b> " + victim + "</p>" if victim else ""}
    </div>
    """, unsafe_allow_html=True)

def render_download(report):
    """Render download button"""
    # Convert to JSON
    if hasattr(report, 'model_dump_json'):
        json_data = report.model_dump_json(indent=2)
    elif hasattr(report, '__dict__'):
        json_data = json.dumps(report.__dict__, default=str, indent=2)
    else:
        json_data = json.dumps({"error": "Could not serialize"})
    
    # Handle both id and scan_id
    if hasattr(report, 'scan_id'):
        report_id = report.scan_id
    elif hasattr(report, 'id'):
        report_id = report.id
    else:
        report_id = f"GTI-{datetime.now().strftime('%H%M%S')}"
    
    st.download_button(
        label="📥 Download JSON Dossier",
        data=json_data,
        file_name=f"{report_id}_dossier.json",
        mime="application/json",
        use_container_width=True
    )

# --- MAIN APP ---
def main():
    render_header()
    mode = render_sidebar()
    
    # Initialize engine
    if mode == "live" and HAS_ENGINE:
        try:
            engine = GTIEngine()
        except Exception as e:
            st.sidebar.error(f"Engine error: {e}")
            engine = MockEngine()
    else:
        engine = MockEngine()
        if mode == "live":
            st.sidebar.warning("⚠️ API not configured, using demo")
    
    # Input section
    input_text, analyze_clicked = render_input_section()
    
    st.markdown("---")
    
    # Run analysis
    if analyze_clicked and input_text.strip():
        with st.spinner("🔍 Decoding Incentives... Extracting Payoffs... Detecting Deception..."):
            try:
                report = engine.analyze(input_text)
                
                # Store in session for persistence
                st.session_state.last_report = report
                st.session_state.last_input = input_text
                
                # Track successful analysis (GoatCounter pixel)
                track_pageview("/analysis-submitted")
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return
        
        # Render results
        st.markdown("### 📊 Analysis Results")
        render_metrics(report)
        st.markdown("")
        view_mode = render_deception_alert(report)
        st.markdown("")
        render_dossier(report, view_mode)
        st.markdown("")
        render_prescription(report)
        st.markdown("")
        render_download(report)
        
    elif 'last_report' in st.session_state:
        # Show cached results
        st.markdown("### 📊 Previous Analysis")
        st.caption(f"Input: {st.session_state.last_input[:100]}...")
        
        report = st.session_state.last_report
        render_metrics(report)
        st.markdown("")
        view_mode = render_deception_alert(report)
        st.markdown("")
        render_dossier(report, view_mode)
        st.markdown("")
        render_prescription(report)
        st.markdown("")
        render_download(report)
    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #64748b;">
            <h3>👆 Enter a scenario above to begin analysis</h3>
            <p>Or click one of the example scenarios in the sidebar</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
