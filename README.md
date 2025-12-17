# Game Theory Index (GTI)

Validated LLM Classification of Strategic Interactions

## Results
- Macro F1: 91.2% (9-type) / 99.0% (8-type)
- Cohen's Kappa: 0.81 (almost perfect agreement)

## Usage
```
python gti_validator.py --validate 100 --data "data/games2p2k/games2p2k_main.csv"
Rscript analyze_gti.R results/json/gti_results_XXXXXX.json
```

## Structure
- gti_validator.py - Main validation script
- analyze_gti.R - R statistical analysis
- data/ - Datasets
- results/ - Outputs (figures, json, tables)
- docs/ - Documentation
- archive/ - Old versions
