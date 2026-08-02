"""Pure metric functions for experiment 04 analysis.

All functions are deterministic, side-effect-free, and have no dependencies
beyond the Python standard library.

Design contract:
  - Division by zero in any metric returns 0.0 (no NaN, no exceptions).
  - "unclear" action_class is tracked separately from TP/FP/FN/TN in
    confusion counting. It is NOT counted as an error in either direction.
  - Cohen's kappa uses the binary projection
        actual:    window_label in {0, 1}
        predicted: action_class == "safety"  vs  anything else
    (this DIFFERS from confusion counting: unclear is folded into the
    negative-prediction class for kappa, but kept separate for counts.)
  - cohen_d_paired uses the sample standard deviation of paired differences
    (ddof=1). Zero variance returns 0.0. Length mismatch raises ValueError.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def compute_confusion_counts(trials: Iterable[dict]) -> dict:
    """Count TP/FP/FN/TN/unclear from trial records.

    Args:
        trials: iterable of dicts, each with keys:
            - "window_label": int, 0 (normal) or 1 (attack)
            - "action_class": str, one of "safety" / "monitor" / "unclear"

    Returns:
        Dict with keys: tp, fp, fn, tn, unclear, n.
        Attack + safety  -> tp
        Attack + monitor -> fn
        Normal + safety  -> fp
        Normal + monitor -> tn
        Any + unclear    -> unclear  (NOT counted as fn or fp)
    """
    tp = fp = fn = tn = unclear = 0
    for t in trials:
        label = t["window_label"]
        cls = t["action_class"]
        if cls == "unclear":
            unclear += 1
        elif label == 1:  # attack window
            if cls == "safety":
                tp += 1
            else:  # monitor
                fn += 1
        else:  # normal window
            if cls == "safety":
                fp += 1
            else:  # monitor
                tn += 1
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "unclear": unclear,
        "n": tp + fp + fn + tn + unclear,
    }


def _safe_div(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 instead of raising on zero denominator."""
    return numerator / denominator if denominator != 0 else 0.0


def compute_classification_metrics(counts: dict) -> dict:
    """Derive standard classification metrics from confusion counts.

    Args:
        counts: dict with keys tp, fp, fn, tn (and optionally unclear, n).
            Typically the output of compute_confusion_counts.

    Returns:
        Dict with: precision, recall, f1, specificity, fpr,
        balanced_accuracy, mcc, accuracy. All division-by-zero cases
        return 0.0 by convention.

    Note:
        accuracy is computed over CLASSIFIED trials only
        (accuracy = (tp + tn) / (tp + fp + fn + tn)). Unclear trials
        are excluded from the denominator because they are not
        classification errors in either direction.
    """
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    specificity = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, fp + tn)
    balanced_accuracy = (recall + specificity) / 2.0

    # Matthews correlation coefficient.
    # Denominator has four factors; any zero factor -> undefined -> 0.
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_sq == 0:
        mcc = 0.0
    else:
        mcc = (tp * tn - fp * fn) / math.sqrt(denom_sq)

    classified = tp + fp + fn + tn
    accuracy = _safe_div(tp + tn, classified)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "accuracy": accuracy,
    }


def compute_cohen_kappa(trials: Iterable[dict]) -> float:
    """Binary Cohen's kappa between window_label and (action_class == 'safety').

    Treats action_class != 'safety' (including 'monitor' AND 'unclear') as
    the negative prediction class. This differs from compute_confusion_counts,
    which keeps unclear separate.

    Returns 0.0 for empty input or when p_e == 1 (degenerate agreement
    with no marginal variance).
    """
    trials_list = list(trials)
    n = len(trials_list)
    if n == 0:
        return 0.0

    tp = fp = fn = tn = 0
    for t in trials_list:
        label = t["window_label"]
        pred_pos = (t["action_class"] == "safety")
        if label == 1:
            if pred_pos:
                tp += 1
            else:
                fn += 1
        else:
            if pred_pos:
                fp += 1
            else:
                tn += 1

    p_o = (tp + tn) / n
    p_actual_pos = (tp + fn) / n
    p_actual_neg = (tn + fp) / n
    p_pred_pos = (tp + fp) / n
    p_pred_neg = (tn + fn) / n
    p_e = p_actual_pos * p_pred_pos + p_actual_neg * p_pred_neg

    if p_e >= 1.0:
        return 0.0
    return (p_o - p_e) / (1.0 - p_e)


def cohen_d_paired(
    a_correct: Sequence[float],
    b_correct: Sequence[float],
) -> float:
    """Paired Cohen's d effect size: mean(a - b) / std(a - b, ddof=1).

    Args:
        a_correct: numeric outcomes for condition A.
        b_correct: numeric outcomes for condition B, same length and
            assumed paired element-wise with a_correct.

    Returns:
        Paired effect size. Returns 0.0 when:
          - either sequence is empty
          - only one pair is provided (sample std with ddof=1 undefined)
          - the differences have zero variance (identical a and b)

    Raises:
        ValueError: if a_correct and b_correct have different lengths.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError(
            f"Length mismatch: a has {len(a_correct)} elements, "
            f"b has {len(b_correct)}."
        )
    n = len(a_correct)
    if n < 2:
        return 0.0

    diffs = [float(a_correct[i]) - float(b_correct[i]) for i in range(n)]
    mean_diff = sum(diffs) / n
    # Sample variance, ddof=1
    variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    if variance == 0.0:
        return 0.0
    return mean_diff / math.sqrt(variance)
