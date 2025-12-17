# 🎯 GTI - Game Theory Intelligence Engine v3.0

> **Strategic Radar for Real-World Interactions**  
> Decode incentives. Detect deception. Predict outcomes.

![GTI Screenshot](https://img.shields.io/badge/version-3.0-blue) ![Python](https://img.shields.io/badge/python-3.10+-green) ![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red)

## What is GTI?

GTI transforms natural language scenarios into formal game-theoretic analysis:

- **Input**: News articles, contracts, negotiations, social situations
- **Output**: Game classification, payoff matrix, equilibrium prediction, deception detection

### Key Features

| Feature | Description |
|---------|-------------|
| 🎲 **12 Orthogonal Types** | Robinson-Goforth canonical 2×2 game classification |
| 🚨 **Deception Detection** | Compares STATED vs REVEALED preferences |
| 📊 **Dual View Mode** | Toggle between Public Narrative and Revealed Reality |
| ⚖️ **Scope Awareness** | Gracefully handles out-of-scope games (zero-sum, sequential, etc.) |
| 🔬 **Evidence Analysis** | Categorizes cheap talk vs credible signals |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (optional - demo mode works without it)
set ANTHROPIC_API_KEY=sk-ant-...

# Run the app
streamlit run app.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   2×2 ORDINAL GAME SPACE                    │
│                                                             │
│   ┌─────────┬─────────┬─────────┬─────────┐                │
│   │   PD    │ Chicken │  Stag   │Deadlock │                │
│   ├─────────┼─────────┼─────────┼─────────┤                │
│   │ Harmony │  Hero   │  BoS    │  Coord  │                │
│   ├─────────┼─────────┼─────────┼─────────┤                │
│   │Assurance│Compromise│ Asymm  │ Cyclic  │                │
│   └─────────┴─────────┴─────────┴─────────┘                │
│                                                             │
│   ✓ Complete coverage of all 2×2 ordinal games             │
│   ✓ Mutually exclusive (orthogonal)                        │
│   ✓ Based on T/R/P/S preference orderings                  │
└─────────────────────────────────────────────────────────────┘
                            │
                  ┌─────────┴─────────┐
                  │   OUT OF SCOPE    │
                  ├───────────────────┤
                  │ • Zero-sum games  │
                  │ • Sequential      │
                  │ • N-player        │
                  │ • Cardinal        │
                  └───────────────────┘
```

## Game Type Reference

| Type | Preference Order | Key Insight |
|------|-----------------|-------------|
| Prisoners_Dilemma | T > R > P > S | Defection dominant |
| Chicken | T > R > S > P | Mutual defection catastrophic |
| Stag_Hunt | R > T > P > S | Cooperation best but risky |
| Deadlock | T > P > R > S | Both prefer mutual defection |
| Harmony | R > T > S > P | Cooperation dominant |
| Battle_of_Sexes | Asymmetric | Coordinate, different preferences |
| Hero | Asymmetric | Someone must sacrifice |

## Deception Detection

GTI separates **what players SAY** from **what players DO**:

```
┌─────────────────────────────────────────────────┐
│ 🚨 DECEPTION DETECTED                           │
│                                                 │
│ Words suggest: Coordination_Game                │
│ Actions reveal: Prisoners_Dilemma               │
│                                                 │
│ 💨 Cheap Talk: "fully committed", "partnership" │
│ ⚠️ Contradiction: "regardless of cost" vs $0    │
│ ✓ Credible Signal: leaked internal memos        │
└─────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web dashboard |
| `gti_engine.py` | Core analysis engine with Pydantic models |
| `canonical_cases.yaml` | Test scenarios (blind test suite) |
| `requirements.txt` | Python dependencies |

## API Usage

```python
from gti_engine import GTIEngine

engine = GTIEngine()
dossier = engine.analyze("""
    Company A issued a press release praising their 
    partnership with Company B...
""")

print(dossier.game_type)        # Prisoners_Dilemma
print(dossier.deception.is_deceptive)  # True
print(dossier.equilibrium.primary_outcome)  # Defect/Defect
```

## License

MIT

---

*Built with Claude + Game Theory + Healthy Paranoia* 🎯
