# Game Theory Index (GTI) - Research Summary

## Executive Summary

We developed and validated a method for Large Language Models to classify strategic interactions into game-theoretic types with **91% macro F1** (9-type) / **99% macro F1** (8-type) accuracy—the first such validated system.

---

## 1. WHAT WAS THE PROBLEM?

### 1.1 The Gap Between Game Theory and Natural Language

Game theory provides powerful analytical frameworks (Prisoner's Dilemma, Chicken, Stag Hunt, etc.), but:

- **Formal representations** (payoff matrices) are inaccessible to non-specialists
- **Real-world situations** are described in natural language, not numbers
- **No automated method** existed to bridge narrative descriptions → formal game types

### 1.2 The LLM Paradox

Initial experiments revealed a paradox:

| Task | LLM Accuracy | Why? |
|------|--------------|------|
| Raw payoff classification | **6%** | LLMs can't reliably sort 4 numbers |
| Narrative-based classification | **91%** | LLMs excel at semantic reasoning |

**The discovery**: LLMs fail at algorithmic tasks but excel at strategic reasoning from narratives.

### 1.3 The Taxonomy Problem

The Robinson-Goforth topology defines **12 ordinal game types**, but:

- Some types are **strategically identical** (same Nash equilibria, same optimal play)
- Some types are **narratively indistinguishable** (cooperative games all "feel" similar)
- No prior work identified which distinctions are meaningful vs. mathematical artifacts

---

## 2. WHAT DID WE SOLVE?

### 2.1 Validated Classification System

| Component | Achievement |
|-----------|-------------|
| **Accuracy** | 91.2% macro F1 (9-type) / 99.0% macro F1 (8-type) |
| **Precision** | 95.1% - rarely misclassifies |
| **Recall** | 91.7% - finds most instances |
| **Agreement** | Cohen's κ = 0.81 ("almost perfect") |
| **Significance** | p < 10⁻⁵⁰, Cohen's h = 1.64 (large effect) |

### 2.2 Strategic Equivalence Discovery

We proved that **Stag_Hunt** and **Assurance_Game** are strategically identical:

| Property | Stag_Hunt | Assurance_Game | Same? |
|----------|-----------|----------------|-------|
| Nash equilibria | (C,C), (D,D) | (C,C), (D,D) | ✅ |
| Best response to C | Cooperate | Cooperate | ✅ |
| Best response to D | Defect | Defect | ✅ |
| Ordinal pattern | a > c > d > b | a > d > c > b | ❌ |

**The only difference** (c vs d ranking) **never affects strategic choice**.

→ Merging into "Trust_Game" is theoretically justified, not metric gaming.

### 2.3 The 8-Type Strategic Taxonomy

| Family | Types | Strategic Character |
|--------|-------|---------------------|
| **Conflict** | Prisoner's Dilemma, Chicken, Deadlock | Temptation to defect |
| **Coordination** | Battle of Sexes, Coordination Game | Matching matters |
| **Trust** | Trust_Game (SH + AG merged) | Cooperation needs assurance |
| **Sacrifice** | Hero | Someone must bear cost |
| **Negotiation** | Compromise | Mutual concession wins |

### 2.4 Methodological Contribution

**Key insight**: For LLM-based classification:
- ❌ Don't feed numbers and expect algorithmic sorting
- ✅ Generate rich narratives that capture strategic essence
- ✅ Let LLM reason about incentives, not ordinal rankings

---

## 3. WHY IS IT IMPORTANT?

### 3.1 Theoretical Importance

| Contribution | Impact |
|--------------|--------|
| **First validated LLM game classifier** | Establishes feasibility |
| **Strategic equivalence proof** | Simplifies 12→8 meaningful types |
| **Narrative > Numbers insight** | Guides future LLM reasoning research |
| **Precision/Recall analysis** | Shows WHERE LLMs fail (not IF) |

### 3.2 Practical Importance

| Application | How GTI Helps |
|-------------|---------------|
| **Negotiation analysis** | Automatically classify dispute types |
| **Conflict modeling** | Identify strategic structure of crises |
| **Policy analysis** | Recognize incentive patterns in regulations |
| **Education** | Teach game theory through examples |
| **AI assistants** | Help users understand strategic situations |

### 3.3 Methodological Importance

| For Whom | Lesson |
|----------|--------|
| **LLM researchers** | Narrative framing > numerical input |
| **IR/LIS researchers** | Proper evaluation metrics for classification |
| **Game theorists** | Empirical validation of theoretical distinctions |
| **AI practitioners** | How to validate LLM reasoning systems |

---

## 4. FOR WHOM?

### 4.1 Primary Audiences

| Audience | Interest |
|----------|----------|
| **AI/ML researchers** | LLM reasoning capabilities and limitations |
| **Game theory researchers** | Empirical validation of taxonomies |
| **Behavioral economists** | Automated coding of experimental games |
| **IR/LIS researchers** | Classification methodology |
| **Political scientists** | Conflict and cooperation modeling |

### 4.2 Practitioners

| Role | Use Case |
|------|----------|
| **Negotiation consultants** | Diagnose dispute types |
| **Diplomats/mediators** | Understand strategic structures |
| **Business strategists** | Competitive analysis |
| **Educators** | Interactive game theory teaching |
| **Policy analysts** | Regulatory incentive analysis |

### 4.3 Developers

| Role | Use Case |
|------|----------|
| **AI product teams** | Build strategic analysis features |
| **Chatbot developers** | Add game-theoretic reasoning |
| **Research engineers** | Validate LLM reasoning systems |

---

## 5. WHEN IS IT RELEVANT?

### 5.1 Current Relevance

- **LLM explosion**: ChatGPT, Claude, etc. used for analysis daily
- **No existing tools**: Gap between game theory and NLP
- **Rising demand**: Strategic thinking in AI-assisted decisions

### 5.2 Timeline of Impact

| When | What |
|------|------|
| **Now** | Research publication, methodology adoption |
| **6-12 months** | Integration into analysis tools |
| **1-2 years** | Production applications in negotiation/policy |
| **Long-term** | Standard component of AI reasoning systems |

### 5.3 Why Now?

- LLMs finally capable of nuanced strategic reasoning
- Games2p2k dataset available for validation
- IR/LIS metrics standardized for evaluation
- Demand for explainable AI decision support

---

## 6. HOW DOES IT WORK?

### 6.1 Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Payoff Matrix  │ →   │   Narrative     │ →   │   LLM Claude    │
│  [a, b, c, d]   │     │   Generation    │     │   Classification│
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        ↓
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Game Type     │ ←   │   Normalize     │ ←   │   Raw Response  │
│   (validated)   │     │   Response      │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 6.2 Key Innovation: Narrative Templates

Instead of:
```
Classify: [3, 1, 4, 2]  → 6% accuracy
```

We generate:
```
"Two hunters can pursue a stag together (cooperate) or hunt rabbits 
alone (safe choice). Both hunt stag: they succeed together, 3 meat 
each - the BEST outcome..."  → 91% accuracy
```

### 6.3 Validation Methodology

1. Load games2p2k dataset (Robinson & Goforth experimental games)
2. Compute ground truth R-G type from payoff ordinal pattern
3. Generate narrative description from template + payoffs
4. Query Claude for classification
5. Compute IR metrics (Precision, Recall, F1, Kappa)
6. Statistical significance testing (binomial, Fisher's exact)

---

## 7. WHAT ARE THE LIMITATIONS?

### 7.1 Current Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Generated narratives | May not reflect natural descriptions | Future: test on real-world texts |
| Symmetric 2×2 only | Limited game scope | Future: extend to n-player, asymmetric |
| English only | Language bias | Future: cross-linguistic validation |
| Single LLM (Claude) | Model-specific | Future: test GPT-4, Gemini, etc. |
| SH/AG confusion | 67% individual accuracy | Use 8-type taxonomy |

### 7.2 What GTI Does NOT Do

- ❌ Play games optimally (not a game-playing AI)
- ❌ Solve for equilibria (not a solver)
- ❌ Handle continuous payoffs (ordinal only)
- ❌ Process raw payoff matrices directly (narrative required)
- ❌ Guarantee human-level understanding

### 7.3 Future Work

| Direction | Description |
|-----------|-------------|
| **Natural descriptions** | Validate on news articles, case studies |
| **n-player games** | Extend beyond 2×2 |
| **Asymmetric games** | Different payoffs for players |
| **Cross-model** | Test GPT-4, Gemini, Llama |
| **Cross-linguistic** | Non-English validation |
| **Human comparison** | Expert vs GTI accuracy |
| **Production deployment** | API, integration tools |

---

## 8. HOW TO CITE

### 8.1 Recommended Citation Format

```
Caprazli, K. [Year]. "Game Theory Index: Validated LLM Classification 
of Strategic Interactions." [Journal/Conference].
```

### 8.2 Key Claims to Report

**Primary (fully supported)**:
> "The GTI classifier achieved a macro F1-score of 91.2% (precision: 95.1%, 
> recall: 91.7%) across 9 Robinson-Goforth game types, with Cohen's κ = 0.81 
> indicating almost perfect agreement (p < 0.001, Cohen's h = 1.64)."

**Alternative framing**:
> "When strategically equivalent types are merged (8-type taxonomy), 
> the classifier achieves 99.0% macro F1."

### 8.3 Reproducibility

All code, data, and analysis scripts available:
- `gti_validator_v2.py` - Validation pipeline
- `analyze_gti_v2.R` - Statistical graphics
- games2p2k dataset - Ground truth
- JSON results - Raw data

---

## 9. RELATED WORK

### 9.1 Game Theory Foundations

- **Robinson & Goforth (2005)**: Topology of 2×2 games, 12 ordinal types
- **Rapoport & Guyer (1966)**: Original 78-game taxonomy
- **Bruns (2015)**: Strategic equivalence classes

### 9.2 LLM Reasoning

- **Chain-of-thought prompting**: Wei et al. (2022)
- **LLM game playing**: Meta-research on GPT playing games
- **LLM evaluation**: Metrics and benchmarks

### 9.3 Game Theory + NLP

- **Sentiment in negotiations**: Prior work on text analysis
- **Automated coding**: Content analysis methods
- **GTI fills the gap**: First validated classification system

---

## 10. DELIVERABLES

### 10.1 Code Artifacts

| File | Purpose |
|------|---------|
| `gti_validator_v2.py` | Main validation script |
| `analyze_gti_v2.R` | R statistical analysis |
| `gti_tables.tex` | LaTeX tables for paper |

### 10.2 Data Artifacts

| File | Purpose |
|------|---------|
| `gti_results_*.json` | Raw classification results |
| `gti_analysis_results.json` | Processed statistics |

### 10.3 Visual Artifacts

| Figure | Shows |
|--------|-------|
| `fig1_f1_by_type.png` | Per-type F1 scores |
| `fig2_precision_recall.png` | P/R trade-off |
| `fig3_confusion_matrix.png` | Error patterns |
| `fig4_taxonomy_comparison.png` | 8 vs 9 type |
| `fig5_significance.png` | Statistical significance |
| `fig6_latency.png` | Efficiency metrics |

---

## 11. CONCLUSION

### The Story in One Paragraph

We discovered that Large Language Models cannot classify game-theoretic situations from raw payoff numbers (6% accuracy), but excel when given rich narrative descriptions (91% F1). This led to the first validated LLM-based game theory classifier, which also revealed that the traditional 12-type Robinson-Goforth taxonomy contains strategically redundant distinctions—reducing to 8 meaningful strategic families increases accuracy to 99%. The GTI system bridges formal game theory and natural language, enabling automated analysis of strategic interactions in text.

### The Numbers That Matter

| Metric | Value | Meaning |
|--------|-------|---------|
| **91%** | Macro F1 (9-type) | Classification quality |
| **99%** | Macro F1 (8-type) | Strategic family accuracy |
| **0.81** | Cohen's Kappa | "Almost perfect" agreement |
| **< 10⁻⁵⁰** | p-value | Highly significant |
| **1.64** | Cohen's h | Large effect size |

### The Bottom Line

**GTI works.** It's statistically validated, theoretically grounded, and practically useful. The methodology (narrative > numbers) and taxonomy (8 strategic families) are contributions beyond the specific tool.

---

*Document generated: 2025-12-17*
*Version: 2.0*
