import streamlit as st
import pandas as pd
import altair as alt

# --- 1. ENTERPRISE CONFIG & CSS ---
st.set_page_config(layout="wide", page_title="GTI Terminal", page_icon="♟️")

st.markdown("""
<style>
    /* RESET & TYPOGRAPHY */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-color: #F8FAFC;
        --card-bg: #FFFFFF;
        --text-primary: #0F172A;
        --text-secondary: #64748B;
        --border-color: #E2E8F0;
        --accent-blue: #2563EB;
        --accent-red: #DC2626;
        --success-green: #059669;
    }

    .stApp {
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    /* CARD SYSTEM */
    .gti-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 4px; /* Sharper corners for enterprise feel */
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* TYPOGRAPHY OVERRIDES */
    h1, h2, h3 { font-weight: 700; letter-spacing: -0.02em; }
    
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 0.25rem;
    }
    
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    /* STATUS BADGES */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-critical { background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5; }
    .badge-safe { background: #F0FDF4; color: #166534; border: 1px solid #86EFAC; }

    /* HIDE STREAMLIT CHROME */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* INPUT STYLING */
    .stTextArea textarea {
        border-radius: 4px;
        border: 1px solid #CBD5E1;
        font-family: 'Inter', monospace; /* Monospace for code/input feel */
        font-size: 0.9rem;
    }
    
    /* TABLE STYLING */
    .topology-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .topology-table td {
        padding: 8px;
        border-bottom: 1px solid #eee;
    }
    .topology-table .var-name { font-weight: 600; color: #334155; }
    
</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR (Command & Control) ---
with st.sidebar:
    st.markdown("### ♟️ GTI // TERMINAL")
    st.markdown('<div style="font-size: 0.7rem; color: #64748B; margin-bottom: 2rem;">V3.1 | ENTERPRISE EDITION</div>', unsafe_allow_html=True)
    
    st.markdown("**SESSION CONTROLS**")
    mode = st.selectbox("Analysis Engine", ["GTI-Pro (Live)", "GTI-Lite (Demo)"])
    st.checkbox("Enforce Strict Topology", value=True)
    st.checkbox("Show Hidden Payoffs", value=True)
    
    st.markdown("---")
    st.markdown("**RECENT SCANS**")
    st.markdown("""
    <div style="font-size:0.8rem; color:#64748B;">
    • BLIND-007 (Deception)<br>
    • NEG-292 (Contract)<br>
    • DIP-118 (Border)
    </div>
    """, unsafe_allow_html=True)

# --- 3. MAIN DASHBOARD ---

# HEADER ROW
c1, c2 = st.columns([3, 1])
with c1:
    st.title("Strategic Radar")
with c2:
    st.markdown("""
    <div style="text-align:right; padding-top:10px;">
        <span class="badge badge-safe">● SYSTEM ONLINE</span>
    </div>
    """, unsafe_allow_html=True)

# INPUT CARD
st.markdown('<div class="gti-card">', unsafe_allow_html=True)
input_text = st.text_area("INTELLIGENCE SOURCE", height=100, placeholder="Paste raw negotiation text, contract clauses, or diplomatic cables...")
if st.button("RUN DEEP SCAN", type="primary"):
    st.session_state['analyzed'] = True
st.markdown('</div>', unsafe_allow_html=True)

# RESULTS (Conditional)
if st.session_state.get('analyzed'):
    
    # KPI ROW
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="gti-card"><div class="kpi-label">Detected Game</div><div class="kpi-value">Prisoner\'s Dilemma</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="gti-card"><div class="kpi-label">Risk Level</div><div class="kpi-value" style="color:#DC2626">CRITICAL</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="gti-card"><div class="kpi-label">Deception</div><div class="kpi-value" style="color:#DC2626">DETECTED</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="gti-card"><div class="kpi-label">Confidence</div><div class="kpi-value">95.2%</div></div>', unsafe_allow_html=True)

    # MAIN CONTENT GRID
    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        # EXEC SUMMARY
        st.markdown(f"""
        <div class="gti-card" style="border-left: 4px solid #DC2626;">
            <div class="kpi-label">EXECUTIVE SUMMARY</div>
            <p style="font-size: 1rem; margin-top: 0.5rem;">
                <b>Deception Pattern Identified:</b> The counterparty is publicly signaling a <i>Coordination Game</i> ("We win together") but their financial structure (Zero Budget) reveals a <i>Defection Strategy</i> ($T > R$).
            </p>
            <p style="font-size: 0.9rem; color: #64748B;">
                The divergence between 'Stated' and 'Revealed' preferences creates a 95% probability of exploitation if you proceed without escrow.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # TOPOLOGY ANALYSIS (The New Feature)
        st.markdown('<div class="gti-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-label">STRUCTURAL TOPOLOGY (12-GAME ANALYSIS)</div>', unsafe_allow_html=True)
        
        # Topology Visualization
        st.markdown("##### Current DNA: `T > R > P > S`")
        
        # A simple visual bar chart for variables
        chart_data = pd.DataFrame({
            'Variable': ['T (Temptation)', 'R (Reward)', 'P (Punishment)', 'S (Sucker)'],
            'Value': [90, 60, 20, 5],
            'Color': ['#DC2626', '#2563EB', '#64748B', '#64748B']
        })
        
        c = alt.Chart(chart_data).mark_bar().encode(
            x='Value',
            y=alt.Y('Variable', sort=None),
            color=alt.Color('Color', scale=None),
            tooltip=['Variable', 'Value']
        ).properties(height=150)
        
        st.altair_chart(c, use_container_width=True)
        
        st.markdown("""
        <div style="background: #F1F5F9; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
            <div class="kpi-label">PHASE TRANSITION DETECTED</div>
            <p style="font-size: 0.9rem; margin-top: 0.5rem;">
                <b>Nearest Cooperative Exit:</b> <span style="color:#2563EB">Stag Hunt</span>
                <br>
                <b>Required Shift:</b> Decrease <b>T</b> (Temptation) by 35%.
                <br>
                <b>Strategic Lever:</b> If you remove the ability to profit from defection (e.g., via IP clawback clauses), the game structurally transforms into a cooperation game.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        # MATRIX CARD
        st.markdown('<div class="gti-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-label">REVEALED INCENTIVE MATRIX</div>', unsafe_allow_html=True)
        # Placeholder for your engine's image
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/df/Prisoners_dilemma_2.svg/1200px-Prisoners_dilemma_2.svg.png", caption="Nash Equilibrium: Defect/Defect")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # PRESCRIPTION CARD
        st.markdown('<div class="gti-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-label">PRESCRIPTION PROTOCOL</div>', unsafe_allow_html=True)
        st.info("1. Demand Pre-Commitment (Escrow)")
        st.info("2. Signal 'Grim Trigger' Strategy")
        st.markdown('</div>', unsafe_allow_html=True)