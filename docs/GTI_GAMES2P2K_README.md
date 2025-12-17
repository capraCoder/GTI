# GTI (Game Type Identification) - games2p2k Integration

## Overview

This package integrates the **games2p2k dataset** from Zhu, Peterson, Enke & Griffiths (2024) with the GTI classifier system. The games2p2k dataset provides **2,416 procedurally generated 2×2 matrix games** with ground-truth labels based on the **Robinson-Goforth topology**.

### Key Features

- **Ground Truth Labels**: Each game is classified using the Robinson-Goforth ordinal topology (12 types)
- **Large Scale**: 2,416 labeled games vs. the original 55-case corpus
- **Human Behavioral Data**: 93,460 strategic decisions from 4,900 participants
- **Validated Mapping**: R-G types mapped to GTI's expanded 26-type taxonomy

## Files

| File | Description |
|------|-------------|
| `gti_validate_games2p2k.py` | Main validation script |
| `gti_robinson_goforth.py` | Robinson-Goforth classifier module |
| `gti_taxonomy_v3.yaml` | Expanded GTI taxonomy (26 types) |
| `gti_games2p2k_integration.py` | Data processing utilities |

## Quick Start

### 1. Setup

```powershell
# Install dependencies
pip install anthropic pyyaml pandas

# Set API key
$env:ANTHROPIC_API_KEY = "your-key-here"
```

### 2. Discover Data Structure

```powershell
python gti_validate_games2p2k.py --discover --data path/to/games2p2k.csv
```

### 3. View Statistics

```powershell
python gti_validate_games2p2k.py --stats --data path/to/games2p2k.csv
```

### 4. Run Validation

```powershell
# Validate on 100 random games
python gti_validate_games2p2k.py --validate 100 --data path/to/games2p2k.csv

# Validate on all games (takes ~20 minutes)
python gti_validate_games2p2k.py --validate all --data path/to/games2p2k.csv
```

### 5. Generate Corpus

```powershell
python gti_validate_games2p2k.py --corpus --data path/to/games2p2k.csv --output gti_corpus_2416.yaml
```

## Robinson-Goforth to GTI Mapping

The 12 Robinson-Goforth ordinal types map to GTI as follows:

| R-G Type | Ordering | GTI Type |
|----------|----------|----------|
| Dilemma | c > a > d > b | Prisoners_Dilemma |
| Chicken | c > a > b > d | Chicken |
| Staghunt | a > c > d > b | Stag_Hunt |
| Leader | c > b > a > d | Battle_of_the_Sexes |
| Safecoord | a > d > b > c | Coordination_Game |
| Assurance | a > d > c > b | Assurance_Game |
| Harmony | a > b > c > d | Harmony |
| Deadlock | c > d > a > b | Deadlock |
| Hero | c > b > d > a | Hero |
| Compromise | c > d > b > a | Compromise |
| Peace | a > b > d > c | Peace |
| Concord | a > c > b > d | Concord |

### New Types Added to GTI v3.0

The games2p2k integration adds 4 types not in the original GTI taxonomy:
- **Hero**: One player sacrifices for mutual benefit
- **Compromise**: Mutual concession preferred over mutual first choice
- **Peace**: Cooperation dominates, stable equilibrium
- **Concord**: Strong preference for mutual cooperation

## Expected Data Format

The games2p2k CSV should contain 8 payoff values per game:

```
a, b, c, d, x, y, z, w

Where:
  Row player payoffs: a=(A,C), b=(A,D), c=(B,C), d=(B,D)
  Col player payoffs: x=(A,C), y=(A,D), z=(B,C), w=(B,D)

Matrix form:
       C           D
A   (a, x)      (b, y)
B   (c, z)      (d, w)
```

The script auto-detects column names including:
- Individual: `a`, `b`, `c`, `d`, `x`, `y`, `z`, `w`
- Numbered: `payoff1` through `payoff8`
- Any 8 numeric columns

## Validation Output

Running validation produces a JSON report with:

```json
{
  "summary": {
    "n_games": 100,
    "n_correct": 87,
    "accuracy": 0.87,
    "api_calls": 100,
    "tokens_used": 15000
  },
  "by_type": {
    "Prisoners_Dilemma": {"correct": 15, "total": 18},
    "Chicken": {"correct": 12, "total": 14},
    ...
  },
  "confusion_matrix": {
    "Prisoners_Dilemma": {"Prisoners_Dilemma": 15, "Common_Pool_Resource": 2, ...}
  },
  "results": [...]
}
```

## Performance Expectations

Based on preliminary testing:

| Difficulty | Expected Accuracy |
|------------|-------------------|
| Easy (dominant strategies) | 95%+ |
| Medium (single NE) | 85-90% |
| Hard (multiple NE, mixed) | 75-85% |

### Known Classification Challenges

1. **PD vs Common Pool Resource**: Fishery/extraction scenarios may be classified as CPR
2. **Stag Hunt vs Assurance**: Similar payoff structures can cause confusion
3. **Coordination variants**: Safecoord, Assurance, and BoS have overlapping features

## References

- Robinson, D., & Goforth, D. (2005). *The Topology of 2×2 Games*. Routledge.
- Zhu, J.-Q., Peterson, J. C., Enke, B., & Griffiths, T. L. (2024). Capturing the Complexity of Human Strategic Decision-Making with Machine Learning. *Nature Human Behaviour*.
- Dataset: https://osf.io/xrvaw/

## Integration with Existing GTI

To use with existing GTI infrastructure:

```python
from gti_robinson_goforth import classify_to_gti, GameMatrix

# From payoff vector
payoffs = [3, 0, 5, 1, 3, 5, 0, 1]  # Classic PD
game = GameMatrix.from_vector(payoffs)
gti_type = classify_to_gti(game, player="row")
print(gti_type)  # "Prisoners_Dilemma"
```

## Cost Estimation

Validating against full dataset:
- **2,416 games × ~30 tokens/call ≈ 72,480 tokens**
- **Estimated cost: ~$0.02-0.05** (Claude Sonnet)
- **Time: ~20 minutes** (with 0.5s delay between calls)

## Troubleshooting

### "Could not extract payoffs"
The CSV format wasn't recognized. Use `--discover` to see column structure, then check if payoffs are in unexpected columns.

### "Unclassified (ties)"
Games with tied payoffs (e.g., a=b) don't fit the R-G ordinal topology. These are skipped in validation.

### API Rate Limits
Increase `--delay` if hitting rate limits:
```powershell
python gti_validate_games2p2k.py --validate 100 --data games2p2k.csv --delay 1.0
```
