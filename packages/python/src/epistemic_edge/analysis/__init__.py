"""Analysis utilities for experiment outputs.

Pure, side-effect-free metric computation functions used by the
experiments/04_batadal_llm_analysis.py orchestration script.
"""
from epistemic_edge.analysis.metrics import (
    cohen_d_paired,
    compute_classification_metrics,
    compute_cohen_kappa,
    compute_confusion_counts,
)

__all__ = [
    "compute_confusion_counts",
    "compute_classification_metrics",
    "compute_cohen_kappa",
    "cohen_d_paired",
]
