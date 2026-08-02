"""Unit tests for epistemic_edge.analysis.metrics.

Strict TDD: these tests are written BEFORE the implementation.
First run should fail with ImportError (module does not exist yet).
Second run (after implementation) should pass all 23 tests.

Covers the four metric-computation functions used by
experiments/04_batadal_llm_analysis.py:

  1. compute_confusion_counts(trials)          -> dict
  2. compute_classification_metrics(counts)    -> dict
  3. compute_cohen_kappa(trials)               -> float
  4. cohen_d_paired(a_correct, b_correct)      -> float

Design contract (decided before implementation, documented here so the
implementer is not tempted to change it):

- "unclear" action_class is tracked separately. It is NOT counted as an
  FN on attack windows, nor FP on normal windows. This lets the paper
  honestly report "model produced non-vocabulary output" vs "model chose
  the wrong class".
- Division-by-zero in any metric returns 0.0 by convention (precision,
  recall, specificity, F1, MCC, kappa).
- Cohen's kappa is computed on the binary projection
      label in {attack, normal}  vs  pred in {safety, not-safety}
  where "not-safety" includes monitor AND unclear. This matches the
  confusion-count treatment where an unclear action on an attack is not
  a true positive.
- cohen_d_paired uses sample std (ddof=1). Zero variance -> 0.0.
  Length mismatch -> ValueError.
"""
from __future__ import annotations

import pytest

from epistemic_edge.analysis.metrics import (
    cohen_d_paired,
    compute_classification_metrics,
    compute_cohen_kappa,
    compute_confusion_counts,
)


# --- Helpers ---------------------------------------------------------------

def _trial(label, action_class, window_idx=0, event_id=None,
           condition="C", rep=0):
    """Minimal dict representing one trial CSV row."""
    return {
        "window_idx": window_idx,
        "window_label": label,           # 0=normal, 1=attack
        "event_id": event_id,
        "condition": condition,
        "rep": rep,
        "action_class": action_class,    # "safety" | "monitor" | "unclear"
    }


# --- Hand-crafted fixtures (expected values pre-computed) ------------------

@pytest.fixture
def fixture_C():
    """Clean: TP=3, FP=1, FN=1, TN=3, unclear=0 (n=8).
    precision=recall=f1=specificity=0.75, FPR=0.25, MCC=0.5, kappa=0.5.
    """
    trials = []
    # 3 TP: attack + safety
    for i in range(3):
        trials.append(_trial(1, "safety", window_idx=i,
                             event_id=1, rep=i))
    # 1 FN: attack + monitor
    trials.append(_trial(1, "monitor", window_idx=3,
                         event_id=1, rep=0))
    # 3 TN: normal + monitor
    for i in range(3):
        trials.append(_trial(0, "monitor", window_idx=100 + i,
                             event_id=None, rep=i))
    # 1 FP: normal + safety
    trials.append(_trial(0, "safety", window_idx=103,
                         event_id=None, rep=0))
    return trials


@pytest.fixture
def fixture_G_all_unclear():
    """All 8 trials produce unclear actions. Everything-zero expected."""
    trials = []
    for i in range(4):
        trials.append(_trial(1, "unclear", window_idx=i,
                             event_id=1, rep=i))
    for i in range(4):
        trials.append(_trial(0, "unclear", window_idx=100 + i,
                             event_id=None, rep=i))
    return trials


@pytest.fixture
def fixture_perfect():
    """TP=5, TN=5, no errors. All metrics = 1.0."""
    trials = []
    for i in range(5):
        trials.append(_trial(1, "safety", window_idx=i,
                             event_id=1, rep=i))
    for i in range(5):
        trials.append(_trial(0, "monitor", window_idx=100 + i,
                             event_id=None, rep=i))
    return trials


@pytest.fixture
def fixture_always_safety():
    """Model outputs safety on every trial. TP=4, FP=4.
    precision=0.5, recall=1.0, f1=2/3, specificity=0.0, MCC=0, kappa=0.
    """
    trials = []
    for i in range(4):
        trials.append(_trial(1, "safety", window_idx=i,
                             event_id=1, rep=i))
    for i in range(4):
        trials.append(_trial(0, "safety", window_idx=100 + i,
                             event_id=None, rep=i))
    return trials


# --- 1. compute_confusion_counts ------------------------------------------

class TestConfusionCounts:

    def test_clean_fixture(self, fixture_C):
        c = compute_confusion_counts(fixture_C)
        assert c["tp"] == 3
        assert c["fp"] == 1
        assert c["fn"] == 1
        assert c["tn"] == 3
        assert c["unclear"] == 0
        assert c["n"] == 8

    def test_all_unclear(self, fixture_G_all_unclear):
        c = compute_confusion_counts(fixture_G_all_unclear)
        assert c["tp"] == 0
        assert c["fp"] == 0
        assert c["fn"] == 0
        assert c["tn"] == 0
        assert c["unclear"] == 8
        assert c["n"] == 8

    def test_perfect_classifier(self, fixture_perfect):
        c = compute_confusion_counts(fixture_perfect)
        assert c["tp"] == 5
        assert c["tn"] == 5
        assert c["fp"] == 0
        assert c["fn"] == 0
        assert c["unclear"] == 0
        assert c["n"] == 10

    def test_always_safety(self, fixture_always_safety):
        c = compute_confusion_counts(fixture_always_safety)
        assert c["tp"] == 4
        assert c["fp"] == 4
        assert c["fn"] == 0
        assert c["tn"] == 0

    def test_empty_list(self):
        c = compute_confusion_counts([])
        assert c == {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
                     "unclear": 0, "n": 0}

    def test_unclear_not_counted_as_error(self):
        """Contract check: unclear on attack is NOT FN; unclear on normal
        is NOT FP. Tracked separately via the 'unclear' key.
        """
        trials = [
            _trial(1, "unclear", window_idx=0, event_id=1),
            _trial(0, "unclear", window_idx=1, event_id=None),
        ]
        c = compute_confusion_counts(trials)
        assert c["tp"] == 0
        assert c["fn"] == 0
        assert c["fp"] == 0
        assert c["tn"] == 0
        assert c["unclear"] == 2
        assert c["n"] == 2


# --- 2. compute_classification_metrics ------------------------------------

class TestClassificationMetrics:

    def test_clean_fixture_metrics(self, fixture_C):
        c = compute_confusion_counts(fixture_C)
        m = compute_classification_metrics(c)
        # TP=3, FP=1, FN=1, TN=3
        assert m["precision"] == pytest.approx(0.75)
        assert m["recall"] == pytest.approx(0.75)
        assert m["f1"] == pytest.approx(0.75)
        assert m["specificity"] == pytest.approx(0.75)
        assert m["fpr"] == pytest.approx(0.25)
        assert m["balanced_accuracy"] == pytest.approx(0.75)
        # MCC = (3*3 - 1*1) / sqrt(4*4*4*4) = 8/16 = 0.5
        assert m["mcc"] == pytest.approx(0.5)
        # accuracy = (TP+TN)/n = 6/8 = 0.75
        assert m["accuracy"] == pytest.approx(0.75)

    def test_perfect_classifier_metrics(self, fixture_perfect):
        c = compute_confusion_counts(fixture_perfect)
        m = compute_classification_metrics(c)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)
        assert m["fpr"] == pytest.approx(0.0)
        assert m["balanced_accuracy"] == pytest.approx(1.0)
        assert m["mcc"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_always_safety_metrics(self, fixture_always_safety):
        c = compute_confusion_counts(fixture_always_safety)
        m = compute_classification_metrics(c)
        # TP=4, FP=4, FN=0, TN=0
        assert m["precision"] == pytest.approx(0.5)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(2.0 / 3.0)
        assert m["specificity"] == pytest.approx(0.0)
        assert m["fpr"] == pytest.approx(1.0)
        assert m["balanced_accuracy"] == pytest.approx(0.5)
        # MCC denominator has a zero factor (TN+FN=0) -> 0 by convention
        assert m["mcc"] == pytest.approx(0.0)

    def test_zero_positives(self):
        """No ground-truth positives, no positive predictions."""
        c = {"tp": 0, "fp": 0, "fn": 0, "tn": 10, "unclear": 0, "n": 10}
        m = compute_classification_metrics(c)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0
        assert m["specificity"] == pytest.approx(1.0)
        assert m["fpr"] == pytest.approx(0.0)
        assert m["mcc"] == 0.0

    def test_zero_trials(self):
        c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "unclear": 0, "n": 0}
        m = compute_classification_metrics(c)
        for key in ("precision", "recall", "f1", "specificity", "fpr",
                    "balanced_accuracy", "mcc", "accuracy"):
            assert m[key] == 0.0, f"{key} should be 0.0 for empty counts"

    def test_mcc_negative_correlation(self):
        """Fully-inverted classifier -> MCC = -1."""
        c = {"tp": 0, "fp": 5, "fn": 5, "tn": 0, "unclear": 0, "n": 10}
        m = compute_classification_metrics(c)
        # (0*0 - 5*5) / sqrt(5*5*5*5) = -25/25 = -1
        assert m["mcc"] == pytest.approx(-1.0)


# --- 3. compute_cohen_kappa ------------------------------------------------

class TestCohenKappa:

    def test_clean_fixture_kappa(self, fixture_C):
        """Binary kappa over label in {attack,normal} vs pred in {safety,not-safety}.
        p_o = (TP+TN)/n = 6/8 = 0.75
        p_e = ((TP+FN)(TP+FP) + (TN+FN)(TN+FP)) / n^2 = (4*4 + 4*4)/64 = 0.5
        kappa = (0.75 - 0.5) / (1 - 0.5) = 0.5
        """
        kappa = compute_cohen_kappa(fixture_C)
        assert kappa == pytest.approx(0.5)

    def test_perfect_classifier_kappa(self, fixture_perfect):
        kappa = compute_cohen_kappa(fixture_perfect)
        assert kappa == pytest.approx(1.0)

    def test_always_safety_kappa(self, fixture_always_safety):
        """Classifier always picks safety -> chance-level agreement, kappa=0.
        p_o = 4/8 = 0.5
        p_e = ((4+0)(4+4) + (0+0)(0+4))/64 = 32/64 = 0.5
        kappa = (0.5-0.5)/(1-0.5) = 0.0
        """
        kappa = compute_cohen_kappa(fixture_always_safety)
        assert kappa == pytest.approx(0.0)

    def test_all_unclear_kappa(self, fixture_G_all_unclear):
        """All predictions non-positive -> degenerate -> 0 by convention."""
        kappa = compute_cohen_kappa(fixture_G_all_unclear)
        assert kappa == pytest.approx(0.0)

    def test_empty_trials_kappa(self):
        assert compute_cohen_kappa([]) == 0.0


# --- 4. cohen_d_paired -----------------------------------------------------

class TestCohenDPaired:

    def test_known_effect(self):
        """a=[1,1,1,0], b=[0,0,0,1]
        diffs=[1,1,1,-1], mean=0.5, std(ddof=1)=sqrt(((0.5)^2*3 + (-1.5)^2)/3)
                                   = sqrt((0.75 + 2.25)/3) = sqrt(1) = 1.0
        d = 0.5 / 1.0 = 0.5
        """
        a = [1, 1, 1, 0]
        b = [0, 0, 0, 1]
        assert cohen_d_paired(a, b) == pytest.approx(0.5)

    def test_zero_effect_identical(self):
        """Identical arrays -> zero variance -> 0 by convention."""
        a = [1, 0, 1, 0]
        b = [1, 0, 1, 0]
        assert cohen_d_paired(a, b) == 0.0

    def test_sign_symmetry(self):
        d_pos = cohen_d_paired([1, 1, 1, 1], [0, 0, 0, 1])
        d_neg = cohen_d_paired([0, 0, 0, 1], [1, 1, 1, 1])
        assert d_pos > 0
        assert d_neg < 0
        assert d_pos == pytest.approx(-d_neg)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cohen_d_paired([1, 0, 1], [0, 1])

    def test_empty_inputs(self):
        assert cohen_d_paired([], []) == 0.0

    def test_single_element(self):
        """Single pair -> sample std undefined (ddof=1 div by zero) -> 0."""
        assert cohen_d_paired([1], [0]) == 0.0
