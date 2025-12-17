#!/usr/bin/env python3
"""
gti_statistical_validation.py - Paper-Ready Statistics

Parallel to OI validation methodology:
- Chi-square vs random chance
- Cohen's Kappa inter-rater reliability
- Effect size (Cohen's h)
- Wilson confidence intervals
- K-fold cross-validation
- Confusion matrix analysis
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional, Callable
from collections import Counter, defaultdict
import json
from pathlib import Path


def chi_square_vs_chance(correct: int, total: int, num_classes: int) -> Dict:
    """
    Chi-square test: classifier vs random baseline.

    H0: Classifier performs at chance level (1/num_classes)
    H1: Classifier performs better than chance
    """
    chance = 1 / num_classes
    expected_correct = total * chance
    expected_wrong = total * (1 - chance)

    observed = [correct, total - correct]
    expected = [expected_correct, expected_wrong]

    chi2, p = stats.chisquare(observed, expected)

    return {
        'chi_square': float(chi2),
        'p_value': float(p),
        'df': 1,
        'reject_null': p < 0.001,
        'significance': 'p < 0.001' if p < 0.001 else f'p = {p:.4f}'
    }


def cohens_kappa(correct: int, total: int, num_classes: int) -> Dict:
    """
    Cohen's Kappa for inter-rater reliability.

    Measures agreement beyond chance.
    """
    p_observed = correct / total
    p_expected = 1 / num_classes

    if p_expected == 1:
        kappa = 1.0
    else:
        kappa = (p_observed - p_expected) / (1 - p_expected)

    # Interpretation (Landis & Koch, 1977)
    if kappa >= 0.81:
        interpretation = "Almost Perfect"
    elif kappa >= 0.61:
        interpretation = "Substantial"
    elif kappa >= 0.41:
        interpretation = "Moderate"
    elif kappa >= 0.21:
        interpretation = "Fair"
    elif kappa >= 0.0:
        interpretation = "Slight"
    else:
        interpretation = "Poor"

    return {
        'kappa': float(kappa),
        'p_observed': float(p_observed),
        'p_expected': float(p_expected),
        'interpretation': interpretation
    }


def cohens_h(p1: float, p2: float) -> Dict:
    """
    Cohen's h effect size for proportions.

    Compares two proportions (e.g., accuracy vs chance).
    """
    # Handle edge cases
    p1 = max(0.001, min(0.999, p1))
    p2 = max(0.001, min(0.999, p2))

    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    h = abs(phi1 - phi2)

    # Cohen's conventions
    if h >= 0.8:
        magnitude = "LARGE"
    elif h >= 0.5:
        magnitude = "MEDIUM"
    elif h >= 0.2:
        magnitude = "SMALL"
    else:
        magnitude = "NEGLIGIBLE"

    return {
        'h': float(h),
        'magnitude': magnitude,
        'p1': float(p1),
        'p2': float(p2)
    }


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Wilson score confidence interval.

    More accurate than normal approximation, especially for small samples
    or proportions near 0 or 1.
    """
    if total == 0:
        return (0.0, 0.0)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / total

    denominator = 1 + z**2 / total
    center = p_hat + z**2 / (2 * total)
    spread = z * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2))

    lower = max(0, (center - spread) / denominator)
    upper = min(1, (center + spread) / denominator)

    return (float(lower), float(upper))


def confusion_matrix(predictions: List[str], ground_truth: List[str]) -> Dict:
    """
    Generate confusion matrix and per-class metrics.
    """
    classes = sorted(set(ground_truth) | set(predictions))
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Build matrix
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for pred, true in zip(predictions, ground_truth):
        matrix[class_to_idx[true], class_to_idx[pred]] += 1

    # Per-class metrics
    class_metrics = {}
    for cls in classes:
        idx = class_to_idx[cls]
        tp = matrix[idx, idx]
        fp = matrix[:, idx].sum() - tp
        fn = matrix[idx, :].sum() - tp
        tn = matrix.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[cls] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(matrix[idx, :].sum())
        }

    # Macro averages
    macro_precision = np.mean([m['precision'] for m in class_metrics.values()])
    macro_recall = np.mean([m['recall'] for m in class_metrics.values()])
    macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])

    return {
        'matrix': matrix.tolist(),
        'classes': classes,
        'class_metrics': class_metrics,
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1)
    }


def cross_validate(
    games: List[Dict],
    classifier_fn: Callable,
    k: int = 5,
    stratified: bool = True
) -> Dict:
    """
    K-fold cross-validation.

    Args:
        games: List of game dicts with 'payoffs' and 'ground_truth'
        classifier_fn: Function(payoffs) -> prediction
        k: Number of folds
        stratified: Whether to maintain class distribution in folds
    """
    games = list(games)  # Copy to avoid mutation
    np.random.shuffle(games)

    if stratified:
        # Group by class
        by_class = defaultdict(list)
        for g in games:
            by_class[g['ground_truth']].append(g)

        # Distribute across folds
        folds = [[] for _ in range(k)]
        for cls, cls_games in by_class.items():
            for i, g in enumerate(cls_games):
                folds[i % k].append(g)
    else:
        fold_size = len(games) // k
        folds = [games[i*fold_size:(i+1)*fold_size] for i in range(k)]
        # Handle remainder
        for i, g in enumerate(games[k*fold_size:]):
            folds[i].append(g)

    fold_accuracies = []
    fold_details = []

    for i in range(k):
        test_set = folds[i]
        train_set = [g for j in range(k) if j != i for g in folds[j]]

        # Evaluate on test set
        correct = 0
        predictions = []
        for g in test_set:
            pred = classifier_fn(g['payoffs'])
            predictions.append(pred)
            if pred == g['ground_truth']:
                correct += 1

        accuracy = correct / len(test_set) if test_set else 0
        fold_accuracies.append(accuracy)
        fold_details.append({
            'fold': i + 1,
            'test_size': len(test_set),
            'correct': correct,
            'accuracy': accuracy
        })

    return {
        'mean_accuracy': float(np.mean(fold_accuracies)),
        'std_accuracy': float(np.std(fold_accuracies)),
        'min_accuracy': float(np.min(fold_accuracies)),
        'max_accuracy': float(np.max(fold_accuracies)),
        'fold_accuracies': [float(a) for a in fold_accuracies],
        'fold_details': fold_details,
        'k': k
    }


def generate_report(
    predictions: List[str],
    ground_truth: List[str],
    title: str = "GTI STATISTICAL VALIDATION"
) -> str:
    """
    Generate paper-ready validation report.
    """
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    total = len(predictions)
    num_classes = len(set(ground_truth))
    accuracy = correct / total if total > 0 else 0

    chi2 = chi_square_vs_chance(correct, total, num_classes)
    kappa = cohens_kappa(correct, total, num_classes)
    effect = cohens_h(accuracy, 1/num_classes)
    ci_low, ci_high = wilson_ci(correct, total)
    cm = confusion_matrix(predictions, ground_truth)

    report = f"""
{'=' * 70}
{title:^70}
{'=' * 70}

{'-' * 25} SUMMARY {'-' * 36}
  Total Cases (N):     {total}
  Correct:             {correct}
  Accuracy:            {accuracy*100:.1f}%
  Game Types (k):      {num_classes}
  Chance Baseline:     {100/num_classes:.1f}% (1/{num_classes})

{'-' * 25} STATISTICAL TESTS {'-' * 26}

1. CHI-SQUARE TEST vs CHANCE
   H0: Classifier = random (1/{num_classes} = {100/num_classes:.1f}%)
   X2(1) = {chi2['chi_square']:.2f}
   {chi2['significance']}
   Result: {'REJECT H0 - Significantly better than chance' if chi2['reject_null'] else 'FAIL TO REJECT H0'}

2. COHEN'S KAPPA (Inter-rater Reliability)
   kappa = {kappa['kappa']:.3f}
   Observed agreement (Po) = {kappa['p_observed']*100:.1f}%
   Expected agreement (Pe) = {kappa['p_expected']*100:.1f}%
   Interpretation: {kappa['interpretation']}

3. EFFECT SIZE (Cohen's h)
   h = {effect['h']:.3f}
   Magnitude: {effect['magnitude']}
   (Accuracy {accuracy*100:.1f}% vs Chance {100/num_classes:.1f}%)

4. CONFIDENCE INTERVAL (95% Wilson)
   Accuracy: {accuracy*100:.1f}%
   95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]

{'-' * 25} PER-CLASS METRICS {'-' * 26}
"""

    # Add per-class metrics
    for cls in sorted(cm['class_metrics'].keys()):
        m = cm['class_metrics'][cls]
        report += f"  {cls:25} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (n={m['support']})\n"

    report += f"""
{'-' * 25} MACRO AVERAGES {'-' * 29}
  Precision: {cm['macro_precision']:.3f}
  Recall:    {cm['macro_recall']:.3f}
  F1 Score:  {cm['macro_f1']:.3f}

{'=' * 70}
PAPER-READY STATISTICS:
  * Accuracy: {accuracy*100:.1f}% (95% CI: {ci_low*100:.1f}%-{ci_high*100:.1f}%)
  * vs Chance: X2(1) = {chi2['chi_square']:.1f}, {chi2['significance']}
  * Effect Size: h = {effect['h']:.2f} ({effect['magnitude']})
  * Inter-rater: kappa = {kappa['kappa']:.2f} ({kappa['interpretation']})
  * Macro F1: {cm['macro_f1']:.3f}
{'=' * 70}
"""
    return report


def run_validation_suite(
    corpus_path: str,
    classifier_fn: Callable,
    output_path: Optional[str] = None
) -> Dict:
    """
    Run complete validation suite on a corpus.

    Args:
        corpus_path: Path to YAML corpus with cases
        classifier_fn: Classification function
        output_path: Optional path to save results JSON
    """
    import yaml

    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = yaml.safe_load(f)

    cases = corpus.get('cases', corpus.get('games', []))

    predictions = []
    ground_truth = []

    for case in cases:
        # Handle different corpus formats
        if 'payoffs' in case:
            payoffs = case['payoffs']
        elif 'payoff_matrix' in case:
            payoffs = case['payoff_matrix']
        else:
            continue

        expected = case.get('expected', case.get('ground_truth', case.get('rg_class', 'Unknown')))

        pred = classifier_fn(payoffs)
        predictions.append(pred)
        ground_truth.append(expected)

    # Generate all statistics
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    total = len(predictions)
    num_classes = len(set(ground_truth))
    accuracy = correct / total if total > 0 else 0

    results = {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'num_classes': num_classes,
        'chi_square': chi_square_vs_chance(correct, total, num_classes),
        'cohens_kappa': cohens_kappa(correct, total, num_classes),
        'effect_size': cohens_h(accuracy, 1/num_classes),
        'confidence_interval': {
            '95%': wilson_ci(correct, total, 0.95),
            '99%': wilson_ci(correct, total, 0.99)
        },
        'confusion_matrix': confusion_matrix(predictions, ground_truth)
    }

    # Print report
    print(generate_report(predictions, ground_truth))

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys

    # Demo with synthetic data
    print("=" * 70)
    print("GTI STATISTICAL VALIDATION - DEMO")
    print("=" * 70)

    # Simulate classifier results
    np.random.seed(42)
    classes = ['PD', 'Chicken', 'Stag_Hunt', 'Coordination', 'BoS']
    n_samples = 100

    # Generate ground truth
    ground_truth = np.random.choice(classes, n_samples).tolist()

    # Simulate predictions (90% accuracy with some noise)
    predictions = []
    for gt in ground_truth:
        if np.random.random() < 0.90:
            predictions.append(gt)
        else:
            predictions.append(np.random.choice(classes))

    report = generate_report(predictions, ground_truth, "DEMO VALIDATION REPORT")
    print(report)
