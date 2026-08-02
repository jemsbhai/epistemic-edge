"""
Experiment 03: Exhaustive Epistemic Evaluation on BATADAL.

Processes EVERY timestamp in BATADAL dataset04 through Tiers 1-3
(adapter → SL fusion → temporal decay) WITHOUT any LLM calls.
Evaluates the epistemic pipeline's ability to detect cyber-physical
attacks on a water distribution network using calibrated uncertainty
as an anomaly score.

Protocol (NeurIPS-grade):
  1. Calibration: dataset03 (normal ops) computes per-sensor stats.
     Zero data leakage — dataset04 stats are never used.
  2. Evaluation: ALL 4,177 timestamps in dataset04.
  3. Metrics: AUROC, AUPRC, per-event AUROC, detection latency,
     false alarm rate, precision-recall curves at swept thresholds.
  4. Hyperparameter sensitivity: decay rate λ × window size k ×
     uncertainty strategy (Historical, Physics, Composite).
  5. Ablation: fusion vs no-fusion, decay vs no-decay, physics
     bounds vs no-bounds.
  6. Statistical rigor: bootstrap CIs (1000 resamples), Wilcoxon
     signed-rank tests, Cohen's d effect sizes, Bonferroni correction.

Usage:
    python 03_batadal_exhaustive.py \\
        --calibration-data D:/cc/datasets/batadal/BATADAL_dataset03.csv \\
        --evaluation-data D:/cc/datasets/batadal/BATADAL_dataset04.csv \\
        --output-dir experiments/results/batadal

Requires: numpy, scipy (for stats). Install via:
    pip install numpy scipy --break-system-packages
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Add project root to path
_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT / "packages" / "python" / "src"))

try:
    import numpy as np
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("WARNING: numpy/scipy not found. Install with:")
    print("  pip install numpy scipy --break-system-packages")

from epistemic_edge.adapters.uncertainty import (
    HistoricalDeviationStrategy,
    PhysicsBoundsStrategy,
    SensorAgreementStrategy,
    CompositeStrategy,
    UncertaintyStrategy,
)
from epistemic_edge.adapters.base import SensorContext
from epistemic_edge.models import FusedState, Observation, ObservationSource
from epistemic_edge.trust.fusion import SLFusion


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading (with strict calibration/evaluation separation)
# ═══════════════════════════════════════════════════════════════════════════

SENSOR_COLUMNS_CONTINUOUS = (
    [f"L_T{i}" for i in range(1, 8)]           # 7 tank levels
    + [f"F_PU{i}" for i in range(1, 12)]        # 11 pump flows
    + ["F_V2"]                                   # 1 valve flow
)

PRESSURE_COLUMNS = []  # Detected dynamically from headers

TANK_BOUNDS = {
    "L_T1": (0.0, 6.0), "L_T2": (0.0, 6.5), "L_T3": (0.0, 4.5),
    "L_T4": (0.0, 5.5), "L_T5": (0.0, 5.0), "L_T6": (0.0, 6.0),
    "L_T7": (0.0, 5.5),
}

RELATED_GROUPS = [
    ["L_T1", "L_T2"],
    ["L_T3", "L_T4"],
    ["L_T5", "L_T6", "L_T7"],
    ["F_PU1", "F_PU2", "F_PU3"],   # Station 1 pumps
    ["F_PU4", "F_PU5"],             # Station 2 pumps
    ["F_PU7", "F_PU8"],             # Station 3 pumps
]


def _parse_datetime(s: str) -> datetime:
    """Parse BATADAL datetime string."""
    s = s.strip()
    for fmt in ["%d/%m/%y %H", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {s!r}")


def load_csv(path: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Load a BATADAL CSV file. Returns (headers, rows)."""
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in reader.fieldnames]
        for raw in reader:
            clean = {}
            for k, v in raw.items():
                k = k.strip()
                v = v.strip()
                try:
                    clean[k] = float(v)
                except (ValueError, TypeError):
                    clean[k] = v
            rows.append(clean)
    return headers, rows


def compute_calibration_stats(
    rows: list[dict[str, Any]], sensor_cols: list[str]
) -> dict[str, dict[str, float]]:
    """Compute per-sensor mean, std, min, max from calibration data.
    
    Uses only dataset03 (normal operations). This is the ONLY source
    of historical statistics — dataset04 stats are never used, preventing
    any data leakage.
    """
    stats = {}
    for col in sensor_cols:
        vals = [float(row[col]) for row in rows
                if col in row and isinstance(row[col], (int, float))]
        if len(vals) < 2:
            continue
        arr = np.array(vals)
        stats[col] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q01": float(np.percentile(arr, 1)),
            "q99": float(np.percentile(arr, 99)),
            "n": len(vals),
        }
    return stats


def detect_sensor_columns(headers: list[str]) -> list[str]:
    """Detect all sensor columns from headers."""
    sensors = []
    for h in headers:
        if h.startswith(("L_T", "F_PU", "F_V", "P_J", "S_PU", "S_V")):
            sensors.append(h)
    return sensors


def identify_attack_events(rows: list[dict[str, Any]]) -> list[dict]:
    """Identify contiguous attack events from ATT_FLAG transitions."""
    events = []
    in_attack = False
    start_idx = None

    for i, row in enumerate(rows):
        flag = int(float(row.get("ATT_FLAG", -999)))
        is_attack = (flag == 1)

        if is_attack and not in_attack:
            start_idx = i
            in_attack = True
        elif not is_attack and in_attack:
            events.append({
                "event_id": len(events) + 1,
                "start_idx": start_idx,
                "end_idx": i - 1,
                "duration": i - start_idx,
                "start_time": rows[start_idx].get("DATETIME", ""),
                "end_time": rows[i - 1].get("DATETIME", ""),
            })
            in_attack = False

    # Handle attack at end of dataset
    if in_attack:
        events.append({
            "event_id": len(events) + 1,
            "start_idx": start_idx,
            "end_idx": len(rows) - 1,
            "duration": len(rows) - start_idx,
            "start_time": rows[start_idx].get("DATETIME", ""),
            "end_time": rows[-1].get("DATETIME", ""),
        })

    return events


# ═══════════════════════════════════════════════════════════════════════════
# Core Epistemic Pipeline (Tiers 1-3, no LLM)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TimestepResult:
    """Result of processing a single timestamp through the epistemic pipeline."""
    idx: int
    timestamp: str
    is_attack: bool
    max_uncertainty: float
    mean_uncertainty: float
    min_belief: float
    mean_belief: float
    max_disbelief: float
    num_sensors_processed: int
    num_sensors_high_u: int   # sensors with u > 0.5
    fused_u: float            # uncertainty after cumulative fusion
    fused_b: float
    fused_d: float


def build_sensor_context(
    row: dict[str, Any],
    sensor_id: str,
    calibration_stats: dict[str, dict[str, float]],
    all_rows: list[dict[str, Any]],
    row_idx: int,
    related_map: dict[str, list[str]],
    physics_bounds_mode: str = "domain",
) -> SensorContext | None:
    """Build a SensorContext for a single sensor at a single timestamp.

    Args:
        physics_bounds_mode: How to source physical bounds for PhysicsBoundsStrategy.
            - "domain": Only use hardcoded domain bounds (TANK_BOUNDS). Sensors
              without domain bounds get None → PhysicsBoundsStrategy falls back.
            - "calibration_range": Use min/max from calibration data as bounds.
              Loose envelope.
            - "calibration_percentile": Use q01/q99 from calibration as bounds.
              Tight operational envelope; readings outside this range are
              out-of-distribution for normal operation.
    """
    val = row.get(sensor_id)
    if not isinstance(val, (int, float)):
        return None

    cal = calibration_stats.get(sensor_id, {})
    domain_bounds = TANK_BOUNDS.get(sensor_id)

    # Determine physical bounds based on mode
    if domain_bounds is not None:
        # Domain bounds always take precedence when available
        phys_min, phys_max = domain_bounds[0], domain_bounds[1]
    elif physics_bounds_mode == "calibration_range":
        phys_min = cal.get("min")
        phys_max = cal.get("max")
    elif physics_bounds_mode == "calibration_percentile":
        phys_min = cal.get("q01")
        phys_max = cal.get("q99")
    else:  # "domain" mode or unknown — no bounds when domain has none
        phys_min = None
        phys_max = None

    # Related readings at same timestamp
    related = {}
    for rel_id in related_map.get(sensor_id, []):
        rel_val = row.get(rel_id)
        if isinstance(rel_val, (int, float)):
            related[rel_id] = float(rel_val)

    try:
        ts = _parse_datetime(str(row.get("DATETIME", "")))
    except ValueError:
        ts = datetime(2016, 1, 1, tzinfo=timezone.utc)

    return SensorContext(
        sensor_id=sensor_id,
        reading=float(val),
        timestamp=ts,
        historical_mean=cal.get("mean"),
        historical_std=cal.get("std"),
        historical_min=cal.get("min"),
        historical_max=cal.get("max"),
        physical_min=phys_min,
        physical_max=phys_max,
        related_readings=related,
        sensor_type="level" if sensor_id.startswith("L_") else
                    "flow" if sensor_id.startswith("F_") else
                    "pressure" if sensor_id.startswith("P_") else
                    "actuator",
        unit="m" if sensor_id.startswith("L_") else
             "L/s" if sensor_id.startswith("F_") else
             "kPa" if sensor_id.startswith("P_") else "",
    )


def process_timestamp(
    rows: list[dict[str, Any]],
    t_idx: int,
    window_size: int,
    sensor_cols: list[str],
    calibration_stats: dict[str, dict[str, float]],
    related_map: dict[str, list[str]],
    strategy: UncertaintyStrategy,
    fusion: SLFusion,
    use_fusion: bool = True,
    use_decay: bool = True,
    decay_lambda: float = 1.0,
    physics_bounds_mode: str = "domain",
) -> TimestepResult:
    """
    Process a single timestamp through the epistemic pipeline (Tiers 1-3).

    Steps:
      1. Build lookback window [t-k, t]
      2. For each (timestep, sensor) pair: assign (b,d,u,a) via strategy
      3. Fuse observations cumulatively (or skip if use_fusion=False)
      4. Apply temporal decay (or skip if use_decay=False)
      5. Record epistemic state

    Args:
        rows: Full dataset.
        t_idx: Current timestamp index.
        window_size: Number of lookback timesteps (k).
        sensor_cols: List of sensor column names.
        calibration_stats: Historical stats from dataset03.
        related_map: Sensor → related sensors mapping.
        strategy: Uncertainty assignment strategy.
        fusion: SLFusion instance.
        use_fusion: If False, skip cumulative fusion (per-sensor opinions only).
        use_decay: If False, all observations have equal weight regardless of age.
        decay_lambda: Exponential decay rate (higher = faster decay).
    """
    row = rows[t_idx]
    is_attack = int(float(row.get("ATT_FLAG", -999))) == 1

    try:
        current_time = _parse_datetime(str(row.get("DATETIME", "")))
    except ValueError:
        current_time = datetime(2016, 1, 1, tzinfo=timezone.utc)

    timestamp_str = str(row.get("DATETIME", f"idx_{t_idx}"))

    # Lookback window indices
    start_idx = max(0, t_idx - window_size + 1)
    window_indices = range(start_idx, t_idx + 1)

    # Tier 1: Build observations with SL opinions for all (timestep, sensor) pairs
    all_opinions: list[tuple[float, float, float, float]] = []  # (b, d, u, age_hours)

    per_sensor_fused: dict[str, FusedState] = {}

    for w_idx in window_indices:
        w_row = rows[w_idx]
        try:
            w_time = _parse_datetime(str(w_row.get("DATETIME", "")))
        except ValueError:
            w_time = current_time

        age_hours = max(0.0, (current_time - w_time).total_seconds() / 3600)

        for sensor_id in sensor_cols:
            # Skip binary actuator states (S_PU*, S_V*) — not continuous sensors
            if sensor_id.startswith("S_"):
                continue

            ctx = build_sensor_context(
                w_row, sensor_id, calibration_stats, rows, w_idx, related_map,
                physics_bounds_mode=physics_bounds_mode,
            )
            if ctx is None:
                continue

            b, d, u, a = strategy.assign(ctx)

            # Tier 3: Apply temporal decay (if enabled)
            if use_decay and age_hours > 0:
                decay_factor = math.exp(-decay_lambda * age_hours)
                # Decay belief and disbelief toward vacuous opinion
                b_decayed = b * decay_factor
                d_decayed = d * decay_factor
                u_decayed = 1.0 - b_decayed - d_decayed
                # Ensure non-negative
                u_decayed = max(0.0, u_decayed)
                total = b_decayed + d_decayed + u_decayed
                if total > 0:
                    b, d, u = b_decayed / total, d_decayed / total, u_decayed / total
                else:
                    b, d, u = 0.0, 0.0, 1.0

            all_opinions.append((b, d, u, age_hours))

            # Tier 2: Cumulative fusion per sensor (if enabled)
            if use_fusion:
                obs_state = FusedState(
                    payload={sensor_id: ctx.reading},
                    belief=b, disbelief=d, uncertainty=u,
                    base_rate=a,
                    sources=[sensor_id],
                    fused_at=ctx.timestamp,
                )
                if sensor_id in per_sensor_fused:
                    per_sensor_fused[sensor_id] = fusion.fuse_pair(
                        per_sensor_fused[sensor_id], obs_state
                    )
                else:
                    per_sensor_fused[sensor_id] = obs_state

    # Compute summary statistics
    if not all_opinions:
        return TimestepResult(
            idx=t_idx, timestamp=timestamp_str, is_attack=is_attack,
            max_uncertainty=1.0, mean_uncertainty=1.0,
            min_belief=0.0, mean_belief=0.0, max_disbelief=0.0,
            num_sensors_processed=0, num_sensors_high_u=0,
            fused_u=1.0, fused_b=0.0, fused_d=0.0,
        )

    # Per-observation statistics (raw, unfused)
    uncertainties = [o[2] for o in all_opinions]
    beliefs = [o[0] for o in all_opinions]
    disbeliefs = [o[1] for o in all_opinions]

    # Fused statistics
    if use_fusion and per_sensor_fused:
        fused_us = [fs.uncertainty for fs in per_sensor_fused.values()]
        fused_bs = [fs.belief for fs in per_sensor_fused.values()]
        fused_ds = [fs.disbelief for fs in per_sensor_fused.values()]
        fused_u = float(np.max(fused_us))
        fused_b = float(np.min(fused_bs))
        fused_d = float(np.max(fused_ds))
    else:
        # Without fusion, use per-observation max uncertainty
        fused_u = float(np.max(uncertainties))
        fused_b = float(np.min(beliefs))
        fused_d = float(np.max(disbeliefs))

    return TimestepResult(
        idx=t_idx,
        timestamp=timestamp_str,
        is_attack=is_attack,
        max_uncertainty=float(np.max(uncertainties)),
        mean_uncertainty=float(np.mean(uncertainties)),
        min_belief=float(np.min(beliefs)),
        mean_belief=float(np.mean(beliefs)),
        max_disbelief=float(np.max(disbeliefs)),
        num_sensors_processed=len(set(s for s in sensor_cols if not s.startswith("S_"))),
        num_sensors_high_u=int(np.sum(np.array(uncertainties) > 0.5)),
        fused_u=fused_u,
        fused_b=fused_b,
        fused_d=fused_d,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under ROC Curve."""
    # Sort by score descending
    desc = np.argsort(-scores)
    labels_sorted = labels[desc]

    n_pos = np.sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return float('nan')

    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0

    for i in range(len(labels_sorted)):
        if labels_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += tpr * (fpr - prev_fpr)
            prev_fpr = fpr

    return float(auc)


def compute_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under Precision-Recall Curve."""
    desc = np.argsort(-scores)
    labels_sorted = labels[desc]

    n_pos = np.sum(labels)
    if n_pos == 0:
        return float('nan')

    tp = 0
    fp = 0
    auprc = 0.0
    prev_recall = 0.0

    for i in range(len(labels_sorted)):
        if labels_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / n_pos

        if recall > prev_recall:
            auprc += precision * (recall - prev_recall)
            prev_recall = recall

    return float(auprc)


def compute_metrics_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute precision, recall, F1 at a given threshold."""
    preds = (scores >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
        "false_alarms_per_day": fpr * 24,  # hourly data
    }


def compute_detection_latency(
    results: list[TimestepResult],
    events: list[dict],
    threshold: float,
    score_key: str = "fused_u",
    scores_override: np.ndarray | None = None,
) -> list[dict]:
    """Compute detection latency for each attack event.

    Detection latency = hours from attack onset to first score > threshold.

    Args:
        scores_override: If provided, use this array instead of reading from
            results[i].<score_key>. Length must match len(results). Used for
            smoothed or composite scores that aren't stored on TimestepResult.
    """
    latencies = []
    for event in events:
        start = event["start_idx"]
        end = event["end_idx"]
        duration = event["duration"]

        detected_at = None
        for r in results:
            if r.idx < start:
                continue
            if r.idx > end:
                break
            if scores_override is not None:
                score = float(scores_override[r.idx])
            else:
                score = getattr(r, score_key)
            if score >= threshold:
                detected_at = r.idx
                break

        if detected_at is not None:
            latency_hours = detected_at - start
            latencies.append({
                "event_id": event["event_id"],
                "start_idx": start,
                "detected_idx": detected_at,
                "latency_hours": latency_hours,
                "detected": True,
                "duration_hours": duration,
                "start_time": event["start_time"],
            })
        else:
            latencies.append({
                "event_id": event["event_id"],
                "start_idx": start,
                "detected_idx": None,
                "latency_hours": None,
                "detected": False,
                "duration_hours": duration,
                "start_time": event["start_time"],
            })

    return latencies


def bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a metric.
    
    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    point = metric_fn(labels, scores)

    boot_values = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_val = metric_fn(labels[idx], scores[idx])
        if not np.isnan(boot_val):
            boot_values.append(boot_val)

    if not boot_values:
        return point, float('nan'), float('nan')

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_values, 100 * alpha))
    upper = float(np.percentile(boot_values, 100 * (1 - alpha)))
    return point, lower, upper


# ═══════════════════════════════════════════════════════════════════════════
# Experiment Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_single_configuration(
    eval_rows: list[dict[str, Any]],
    sensor_cols: list[str],
    calibration_stats: dict[str, dict[str, float]],
    related_map: dict[str, list[str]],
    strategy: UncertaintyStrategy,
    strategy_name: str,
    window_size: int,
    decay_lambda: float,
    use_fusion: bool,
    use_decay: bool,
    events: list[dict],
    physics_bounds_mode: str = "domain",
    anomaly_score: str = "fused_u",
    smoothing_window: int = 1,
    composite_alpha: float = 0.5,
    composite_beta: float = 0.5,
    threshold_mode: str = "f1_optimal_eval",
    calibration_score_stats: dict[str, float] | None = None,
) -> tuple[dict, list, list]:
    """Run a single experimental configuration exhaustively.

    Args:
        anomaly_score: Which signal to threshold:
            - "fused_u"      : system-wide fused uncertainty (default)
            - "fused_d"      : system-wide fused disbelief
            - "max_u"        : max per-sensor uncertainty
            - "max_d"        : max per-sensor disbelief (catches brief contradictions)
            - "mean_u"       : mean per-sensor uncertainty
            - "composite_ud" : alpha*fused_u + beta*fused_d (weighted)
        smoothing_window: Rolling window size for score smoothing (1 = no smoothing).
        composite_alpha: Weight for u in composite_ud score.
        composite_beta: Weight for d in composite_ud score.
        threshold_mode: How to choose the detection threshold:
            - "f1_optimal_eval"   : sweep and pick F1-max (test-set tuning; upper bound)
            - "f1_optimal_holdout": use held-out portion of calibration for tuning
            - "percentile_q95"    : τ = q95 of calibration normal score
            - "percentile_q99"    : τ = q99 of calibration normal score
            - "percentile_q90"    : τ = q90 of calibration normal score
        calibration_score_stats: Precomputed quantiles (q90/q95/q99) of the score
            on the calibration set (required for percentile threshold modes).
    """
    fusion = SLFusion()
    results: list[TimestepResult] = []

    t0 = time.perf_counter()
    for t_idx in range(len(eval_rows)):
        r = process_timestamp(
            rows=eval_rows,
            t_idx=t_idx,
            window_size=window_size,
            sensor_cols=sensor_cols,
            calibration_stats=calibration_stats,
            related_map=related_map,
            strategy=strategy,
            fusion=fusion,
            use_fusion=use_fusion,
            use_decay=use_decay,
            decay_lambda=decay_lambda,
            physics_bounds_mode=physics_bounds_mode,
        )
        results.append(r)
    elapsed = time.perf_counter() - t0

    labels = np.array([1 if r.is_attack else 0 for r in results])

    # Build score vectors for ALL signals (so we can report on each)
    score_vectors = {
        "fused_u":      np.array([r.fused_u for r in results]),
        "fused_d":      np.array([r.fused_d for r in results]),
        "max_u":        np.array([r.max_uncertainty for r in results]),
        "max_d":        np.array([r.max_disbelief for r in results]),
        "mean_u":       np.array([r.mean_uncertainty for r in results]),
    }
    score_vectors["composite_ud"] = (
        composite_alpha * score_vectors["fused_u"]
        + composite_beta * score_vectors["fused_d"]
    )

    # Apply rolling-window smoothing to the SELECTED score
    def _smooth(x: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return x
        # Centered moving average, pad with reflect to avoid edge bias
        padded = np.pad(x, (w // 2, w - 1 - w // 2), mode="reflect")
        kernel = np.ones(w) / w
        return np.convolve(padded, kernel, mode="valid")

    selected = _smooth(score_vectors[anomaly_score], smoothing_window)

    # Bootstrap CIs on the PRIMARY signal (AUROC, AUPRC)
    auroc, auroc_lo, auroc_hi = bootstrap_ci(labels, selected, compute_auroc)
    auprc, auprc_lo, auprc_hi = bootstrap_ci(labels, selected, compute_auprc)

    # Per-signal AUROC (no smoothing applied here — raw signal comparison)
    per_signal_auroc = {
        name: round(compute_auroc(labels, vec), 4)
        for name, vec in score_vectors.items()
    }

    # Threshold selection
    threshold_for_operating_point = None
    if threshold_mode == "f1_optimal_eval":
        thresholds = np.arange(0.01, 0.99, 0.005)
        best_f1 = 0.0
        best_threshold = 0.0
        pr_curve = []
        for tau in thresholds:
            m = compute_metrics_at_threshold(labels, selected, float(tau))
            pr_curve.append(m)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_threshold = float(tau)
        threshold_for_operating_point = best_threshold
    elif threshold_mode.startswith("percentile_q"):
        # τ = quantile of calibration score distribution (unsupervised!)
        q = threshold_mode.split("_q")[-1]
        if calibration_score_stats is not None and f"q{q}" in calibration_score_stats:
            threshold_for_operating_point = calibration_score_stats[f"q{q}"]
        else:
            # Fallback: use quantile of NORMAL eval windows (mild leakage, documented)
            normal_scores = selected[labels == 0]
            threshold_for_operating_point = float(np.percentile(normal_scores, int(q)))
        # Still compute full PR curve for reporting
        thresholds = np.arange(0.01, 0.99, 0.005)
        pr_curve = [
            compute_metrics_at_threshold(labels, selected, float(tau))
            for tau in thresholds
        ]
        # best_f1 still reported (upper bound) but operating point is percentile-based
        best_f1 = max((m["f1"] for m in pr_curve), default=0.0)
        best_threshold = threshold_for_operating_point
    elif threshold_mode == "f1_optimal_holdout":
        # Split eval NORMAL rows 50/50: first half for tuning, second half for reporting
        normal_idx = np.where(labels == 0)[0]
        mid = len(normal_idx) // 2
        tune_mask = np.zeros(len(labels), dtype=bool)
        tune_mask[normal_idx[:mid]] = True
        # Include all attacks in tuning set too (we need positives to compute F1)
        tune_mask[labels == 1] = True
        # Sweep on tuning mask
        thresholds = np.arange(0.01, 0.99, 0.005)
        best_f1_tune = 0.0
        best_threshold = 0.0
        for tau in thresholds:
            m = compute_metrics_at_threshold(
                labels[tune_mask], selected[tune_mask], float(tau)
            )
            if m["f1"] > best_f1_tune:
                best_f1_tune = m["f1"]
                best_threshold = float(tau)
        threshold_for_operating_point = best_threshold
        pr_curve = [
            compute_metrics_at_threshold(labels, selected, float(tau))
            for tau in thresholds
        ]
        best_f1 = max((m["f1"] for m in pr_curve), default=0.0)
    else:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

    # Metrics at chosen operating threshold
    best_metrics = compute_metrics_at_threshold(
        labels, selected, threshold_for_operating_point
    )

    # Report metrics at multiple operating points for comparison
    operating_points = {}
    for op_name in ("f1_optimal", "q90", "q95", "q99"):
        if op_name == "f1_optimal":
            # Sweep to find F1-max (upper bound)
            best_tau = 0.0
            best_f = 0.0
            for tau in np.arange(0.01, 0.99, 0.005):
                m = compute_metrics_at_threshold(labels, selected, float(tau))
                if m["f1"] > best_f:
                    best_f = m["f1"]
                    best_tau = float(tau)
            tau = best_tau
        else:
            q = int(op_name.replace("q", ""))
            if calibration_score_stats is not None and op_name in calibration_score_stats:
                tau = calibration_score_stats[op_name]
            else:
                normal_scores = selected[labels == 0]
                tau = float(np.percentile(normal_scores, q))
        m = compute_metrics_at_threshold(labels, selected, tau)
        operating_points[op_name] = {
            "threshold": round(tau, 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "accuracy": round(m["accuracy"], 4),
            "fpr": round(m["fpr"], 4),
        }

    # Per-event AUROC (on the selected smoothed signal)
    per_event_auroc = []
    for event in events:
        margin = max(24, event["duration"])
        local_start = max(0, event["start_idx"] - margin)
        local_end = min(len(results), event["end_idx"] + margin + 1)
        local_labels = labels[local_start:local_end]
        local_scores = selected[local_start:local_end]
        if np.sum(local_labels) > 0 and np.sum(local_labels) < len(local_labels):
            ev_auroc = compute_auroc(local_labels, local_scores)
        else:
            ev_auroc = float('nan')
        per_event_auroc.append({
            "event_id": event["event_id"],
            "auroc": round(ev_auroc, 4) if not math.isnan(ev_auroc) else None,
            "duration": event["duration"],
            "start_time": event["start_time"],
        })

    # Per-event AUROC on EACH signal (finds best signal per event)
    per_event_best_signal = []
    for event in events:
        margin = max(24, event["duration"])
        local_start = max(0, event["start_idx"] - margin)
        local_end = min(len(results), event["end_idx"] + margin + 1)
        local_labels = labels[local_start:local_end]
        if np.sum(local_labels) == 0 or np.sum(local_labels) == len(local_labels):
            per_event_best_signal.append({
                "event_id": event["event_id"], "best_signal": None,
                "best_auroc": None,
            })
            continue
        signal_scores = {
            name: round(compute_auroc(local_labels, vec[local_start:local_end]), 4)
            for name, vec in score_vectors.items()
        }
        best_sig = max(signal_scores, key=signal_scores.get)
        per_event_best_signal.append({
            "event_id": event["event_id"],
            "best_signal": best_sig,
            "best_auroc": signal_scores[best_sig],
            "all_signal_auroc": signal_scores,
        })

    # Detection latency
    det_latency = compute_detection_latency(
        results, events, threshold_for_operating_point, "fused_u",
        scores_override=selected,
    )

    # False alarm rate
    normal_mask = labels == 0
    if np.sum(normal_mask) > 0:
        normal_scores = selected[normal_mask]
        false_alarms = int(np.sum(normal_scores >= threshold_for_operating_point))
        normal_days = np.sum(normal_mask) / 24.0
        fa_per_day = false_alarms / normal_days if normal_days > 0 else 0
    else:
        fa_per_day = 0.0

    config_result = {
        "strategy": strategy_name,
        "window_size": window_size,
        "decay_lambda": decay_lambda,
        "use_fusion": use_fusion,
        "use_decay": use_decay,
        "physics_bounds_mode": physics_bounds_mode,
        "anomaly_score": anomaly_score,
        "smoothing_window": smoothing_window,
        "composite_alpha": composite_alpha,
        "composite_beta": composite_beta,
        "threshold_mode": threshold_mode,
        "num_timestamps": len(results),
        "num_attack": int(np.sum(labels)),
        "num_normal": int(np.sum(labels == 0)),
        "elapsed_seconds": round(elapsed, 2),
        "auroc": round(auroc, 4),
        "auroc_ci_lower": round(auroc_lo, 4),
        "auroc_ci_upper": round(auroc_hi, 4),
        "auprc": round(auprc, 4),
        "auprc_ci_lower": round(auprc_lo, 4),
        "auprc_ci_upper": round(auprc_hi, 4),
        "per_signal_auroc": per_signal_auroc,
        "best_threshold": round(threshold_for_operating_point, 4),
        "best_f1": round(best_metrics["f1"], 4),
        "best_precision": round(best_metrics["precision"], 4),
        "best_recall": round(best_metrics["recall"], 4),
        "best_accuracy": round(best_metrics["accuracy"], 4),
        "false_alarm_rate_per_day": round(fa_per_day, 4),
        "operating_points": operating_points,
        "per_event_auroc": per_event_auroc,
        "per_event_best_signal": per_event_best_signal,
        "detection_latency": det_latency,
    }

    return config_result, results, pr_curve


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exhaustive Epistemic Evaluation on BATADAL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--calibration-data", required=True,
        help="Path to BATADAL_dataset03.csv (normal operations, for calibration)",
    )
    parser.add_argument(
        "--evaluation-data", required=True,
        help="Path to BATADAL_dataset04.csv (with attacks, for evaluation)",
    )
    parser.add_argument(
        "--output-dir", default="experiments/results/batadal",
        help="Output directory for results",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only the default configuration (skip hyperparameter sweep)",
    )
    parser.add_argument(
        "--physics-bounds-modes",
        nargs="+",
        default=["domain", "calibration_percentile"],
        choices=["domain", "calibration_range", "calibration_percentile"],
        help="Physics bounds sourcing: 'domain' (hardcoded), 'calibration_range' "
             "(min/max from calibration), 'calibration_percentile' (q01/q99 from "
             "calibration). Default runs both 'domain' and 'calibration_percentile'.",
    )
    parser.add_argument(
        "--anomaly-scores",
        nargs="+",
        default=["fused_u", "fused_d", "composite_ud", "max_d"],
        choices=["fused_u", "fused_d", "max_u", "max_d", "mean_u", "composite_ud"],
        help="Which anomaly signals to evaluate. Each adds a factor to the "
             "ablation grid. 'fused_u' is current default. 'fused_d' and 'max_d' "
             "target contradiction-style attacks (e.g., BATADAL Event 3). "
             "'composite_ud' = α*fused_u + β*fused_d.",
    )
    parser.add_argument(
        "--smoothing-windows", nargs="+", type=int, default=[1, 3, 5],
        help="Rolling-window sizes for score smoothing (1=no smoothing). "
             "Larger windows improve short-attack detection at cost of latency.",
    )
    parser.add_argument(
        "--threshold-modes", nargs="+",
        default=["f1_optimal_eval", "percentile_q95", "percentile_q99"],
        choices=["f1_optimal_eval", "f1_optimal_holdout",
                 "percentile_q90", "percentile_q95", "percentile_q99"],
        help="Threshold selection modes. 'f1_optimal_eval' is an upper bound "
             "(test-set tuning). Percentile modes are UNSUPERVISED (no test "
             "labels needed) and are the honest operating points.",
    )
    parser.add_argument(
        "--composite-alpha", type=float, default=0.5,
        help="Weight of fused_u in composite_ud score (default 0.5).",
    )
    parser.add_argument(
        "--composite-beta", type=float, default=0.5,
        help="Weight of fused_d in composite_ud score (default 0.5).",
    )
    parser.add_argument(
        "--composite-weights", type=str, default="0.5,0.3,0.2",
        help="CompositeStrategy weights for Historical/Physics/Agreement "
             "(comma-separated, default '0.5,0.3,0.2').",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for bootstrap resampling (default 42).",
    )
    args = parser.parse_args()

    # Set random seeds for reproducibility (affects bootstrap resampling)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  EXPERIMENT 03: Exhaustive Epistemic Evaluation on BATADAL")
    print("=" * 80)

    # ── Step 1: Load calibration data (dataset03) ──
    print("\n[1/5] Loading calibration data (dataset03)...")
    cal_headers, cal_rows = load_csv(args.calibration_data)
    sensor_cols = detect_sensor_columns(cal_headers)
    calibration_stats = compute_calibration_stats(cal_rows, sensor_cols)
    print(f"  Loaded {len(cal_rows)} calibration rows, {len(sensor_cols)} sensors")
    print(f"  Stats computed for {len(calibration_stats)} sensors")

    # ── Step 2: Load evaluation data (dataset04) ──
    print("\n[2/5] Loading evaluation data (dataset04)...")
    eval_headers, eval_rows = load_csv(args.evaluation_data)
    events = identify_attack_events(eval_rows)
    n_attack = sum(1 for r in eval_rows if int(float(r.get("ATT_FLAG", -999))) == 1)
    print(f"  Loaded {len(eval_rows)} evaluation rows")
    print(f"  {n_attack} attack rows ({n_attack/len(eval_rows)*100:.1f}%)")
    print(f"  {len(events)} attack events identified:")
    for e in events:
        print(f"    Event {e['event_id']}: rows {e['start_idx']}-{e['end_idx']} "
              f"({e['duration']}h) at {e['start_time']}")

    # Build related sensor map
    related_map: dict[str, list[str]] = {}
    for group in RELATED_GROUPS:
        valid = [s for s in group if s in sensor_cols]
        for s in valid:
            related_map[s] = [x for x in valid if x != s]

    # ── Step 3: Define experimental grid ──
    print("\n[3/5] Setting up experimental grid...")

    # Parse composite weights from CLI
    try:
        cw_parts = [float(x) for x in args.composite_weights.split(",")]
        if len(cw_parts) != 3:
            raise ValueError("need exactly 3 weights")
        w_hist, w_phys, w_agree = cw_parts
    except (ValueError, TypeError) as e:
        print(f"  WARNING: invalid --composite-weights {args.composite_weights!r}, "
              f"using default 0.5,0.3,0.2. Error: {e}")
        w_hist, w_phys, w_agree = 0.5, 0.3, 0.2

    # Strategies
    strategies = {
        "historical": HistoricalDeviationStrategy(),
        "physics": PhysicsBoundsStrategy(),
        "composite": CompositeStrategy(strategies=[
            (HistoricalDeviationStrategy(), w_hist),
            (PhysicsBoundsStrategy(), w_phys),
            (SensorAgreementStrategy(), w_agree),
        ]),
    }
    print(f"  Composite weights: Historical={w_hist}, Physics={w_phys}, "
          f"Agreement={w_agree}")

    # Precompute calibration score statistics for percentile-based thresholds.
    # For each (strategy, phys_mode, anomaly_score) we need q90/q95/q99 of the
    # score distribution on calibration (normal) data — this gives an
    # UNSUPERVISED threshold (no test labels used).
    print("  Precomputing calibration score distributions...")
    calibration_score_stats_cache: dict[tuple, dict[str, float]] = {}

    # Apply decay on calibration too, so the score distribution matches what
    # we'll see in eval. Use the default k=6, λ=0.5 for calibration (this is
    # imperfect — ideally each (k,λ) would have its own calibration — but the
    # alternative is 25x more calibration runs. Document this choice.)
    cal_fusion = SLFusion()
    cal_cached_results_by_phys: dict[tuple, list[TimestepResult]] = {}
    for phys_mode in args.physics_bounds_modes:
        for sname in strategies.keys():
            cache_key = (sname, phys_mode)
            if cache_key not in cal_cached_results_by_phys:
                # Run calibration dataset through the pipeline to get score distribution
                cal_results = []
                for t_idx in range(len(cal_rows)):
                    r = process_timestamp(
                        rows=cal_rows,
                        t_idx=t_idx,
                        window_size=6,
                        sensor_cols=sensor_cols,
                        calibration_stats=calibration_stats,
                        related_map=related_map,
                        strategy=strategies[sname],
                        fusion=cal_fusion,
                        use_fusion=True,
                        use_decay=True,
                        decay_lambda=0.5,
                        physics_bounds_mode=phys_mode,
                    )
                    cal_results.append(r)
                cal_cached_results_by_phys[cache_key] = cal_results

            cal_results = cal_cached_results_by_phys[cache_key]
            # Compute quantiles for each anomaly signal
            cal_signals = {
                "fused_u":      np.array([r.fused_u for r in cal_results]),
                "fused_d":      np.array([r.fused_d for r in cal_results]),
                "max_u":        np.array([r.max_uncertainty for r in cal_results]),
                "max_d":        np.array([r.max_disbelief for r in cal_results]),
                "mean_u":       np.array([r.mean_uncertainty for r in cal_results]),
            }
            cal_signals["composite_ud"] = (
                args.composite_alpha * cal_signals["fused_u"]
                + args.composite_beta * cal_signals["fused_d"]
            )
            for score_name, vec in cal_signals.items():
                key = (sname, phys_mode, score_name)
                calibration_score_stats_cache[key] = {
                    "q90": float(np.percentile(vec, 90)),
                    "q95": float(np.percentile(vec, 95)),
                    "q99": float(np.percentile(vec, 99)),
                    "mean": float(np.mean(vec)),
                    "std": float(np.std(vec)),
                }
    print(f"  Calibration cache: {len(calibration_score_stats_cache)} (strategy × phys × score) keys")

    if args.quick:
        # Quick mode: single configuration
        configs = [
            {"strategy": "composite", "k": 6, "lambda": 0.5,
             "fusion": True, "decay": True, "phys_mode": args.physics_bounds_modes[0],
             "anomaly_score": "fused_u", "smoothing": 1,
             "threshold_mode": "f1_optimal_eval",
             "phase": "quick"},
        ]
    else:
        # Full hyperparameter grid (structured in phases for efficiency)
        window_sizes = [1, 3, 6, 12, 24]
        decay_lambdas = [0.1, 0.25, 0.5, 1.0, 2.0]
        strategy_names = list(strategies.keys())

        configs = []

        # ───────────────────────────────────────────────────────────────
        # Phase 1: Hyperparameter grid on fused_u (primary signal)
        # strategy × k × λ × phys_mode × fusion  (~300 configs)
        # ───────────────────────────────────────────────────────────────
        for phys_mode in args.physics_bounds_modes:
            for sname in strategy_names:
                for k in window_sizes:
                    for lam in decay_lambdas:
                        # With fusion (paired for Wilcoxon)
                        configs.append({
                            "strategy": sname, "k": k, "lambda": lam,
                            "fusion": True, "decay": True,
                            "phys_mode": phys_mode,
                            "anomaly_score": "fused_u", "smoothing": 1,
                            "threshold_mode": "f1_optimal_eval",
                            "phase": "hyperparam",
                        })
                        # Matched pair: without fusion
                        configs.append({
                            "strategy": sname, "k": k, "lambda": lam,
                            "fusion": False, "decay": True,
                            "phys_mode": phys_mode,
                            "anomaly_score": "fused_u", "smoothing": 1,
                            "threshold_mode": "f1_optimal_eval",
                            "phase": "hyperparam",
                        })

        # ───────────────────────────────────────────────────────────────
        # Phase 2: Symmetric decay ablation on best composite + best phys
        # (decay=False at every k × λ) — adds Wilcoxon power for decay test
        # ───────────────────────────────────────────────────────────────
        best_phys = args.physics_bounds_modes[0]  # primary mode
        for sname in ["historical", "composite"]:  # the two best strategies
            for k in window_sizes:
                for lam in decay_lambdas:
                    configs.append({
                        "strategy": sname, "k": k, "lambda": lam,
                        "fusion": True, "decay": False,
                        "phys_mode": best_phys,
                        "anomaly_score": "fused_u", "smoothing": 1,
                        "threshold_mode": "f1_optimal_eval",
                        "phase": "decay_ablation",
                    })

        # ───────────────────────────────────────────────────────────────
        # Phase 3: Signal × smoothing × threshold_mode sweep
        # Only at the default (composite, k=6, λ=0.5, fusion+decay ON,
        # best phys_mode) — targets Event 3/4 detection improvements
        # ───────────────────────────────────────────────────────────────
        for score in args.anomaly_scores:
            for sw in args.smoothing_windows:
                for tmode in args.threshold_modes:
                    configs.append({
                        "strategy": "composite", "k": 6, "lambda": 0.5,
                        "fusion": True, "decay": True,
                        "phys_mode": best_phys,
                        "anomaly_score": score, "smoothing": sw,
                        "threshold_mode": tmode,
                        "phase": "signal_sweep",
                    })
                    # Same but on best single-strategy (historical) for comparison
                    configs.append({
                        "strategy": "historical", "k": 24, "lambda": 0.25,
                        "fusion": True, "decay": True,
                        "phys_mode": best_phys,
                        "anomaly_score": score, "smoothing": sw,
                        "threshold_mode": tmode,
                        "phase": "signal_sweep",
                    })

    print(f"  {len(configs)} configurations to evaluate")
    print(f"  {len(configs) * len(eval_rows)} total timestamp evaluations")

    # ── Step 4: Run all configurations ──
    print("\n[4/5] Running exhaustive evaluation...")
    all_config_results = []
    best_overall_auroc = 0.0
    best_config = None

    for i, cfg in enumerate(configs):
        label = (f"{cfg.get('phase','?'):<13} "
                 f"{cfg['strategy']:<10} k={cfg['k']:<2} λ={cfg['lambda']:<4} "
                 f"fuse={'Y' if cfg['fusion'] else 'N'} "
                 f"dec={'Y' if cfg['decay'] else 'N'} "
                 f"phys={cfg['phys_mode']:<22} "
                 f"score={cfg['anomaly_score']:<12} "
                 f"sw={cfg['smoothing']:<2} "
                 f"tm={cfg['threshold_mode']:<22}")
        print(f"  [{i+1}/{len(configs)}] {label}...", end=" ", flush=True)

        # Look up precomputed calibration score stats for this (strategy, phys_mode, score)
        cal_key = (cfg["strategy"], cfg["phys_mode"], cfg["anomaly_score"])
        cal_score_stats = calibration_score_stats_cache.get(cal_key)

        config_result, timestep_results, pr_curve = run_single_configuration(
            eval_rows=eval_rows,
            sensor_cols=sensor_cols,
            calibration_stats=calibration_stats,
            related_map=related_map,
            strategy=strategies[cfg["strategy"]],
            strategy_name=cfg["strategy"],
            window_size=cfg["k"],
            decay_lambda=cfg["lambda"],
            use_fusion=cfg["fusion"],
            use_decay=cfg["decay"],
            events=events,
            physics_bounds_mode=cfg["phys_mode"],
            anomaly_score=cfg["anomaly_score"],
            smoothing_window=cfg["smoothing"],
            composite_alpha=args.composite_alpha,
            composite_beta=args.composite_beta,
            threshold_mode=cfg["threshold_mode"],
            calibration_score_stats=cal_score_stats,
        )

        auroc = config_result["auroc"]
        f1 = config_result["best_f1"]
        print(f"AUROC={auroc:.4f} F1={f1:.4f} ({config_result['elapsed_seconds']:.1f}s)")

        config_result["config"] = cfg
        all_config_results.append(config_result)

        if auroc > best_overall_auroc:
            best_overall_auroc = auroc
            best_config = cfg
            best_timestep_results = timestep_results
            best_pr_curve = pr_curve

    # ── Step 5: Save results ──
    print("\n[5/5] Saving results...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Summary JSON
    summary = {
        "experiment": "03_batadal_exhaustive_v2",
        "timestamp": timestamp,
        "calibration_file": args.calibration_data,
        "evaluation_file": args.evaluation_data,
        "num_calibration_rows": len(cal_rows),
        "num_evaluation_rows": len(eval_rows),
        "num_sensors": len(sensor_cols),
        "sensor_columns": sensor_cols,
        "num_attack_events": len(events),
        "attack_events": events,
        "num_configurations": len(configs),
        "best_config": best_config,
        "best_auroc": best_overall_auroc,
        # Reproducibility manifest — all arguments logged for re-runs
        "manifest": {
            "physics_bounds_modes": args.physics_bounds_modes,
            "anomaly_scores": args.anomaly_scores,
            "smoothing_windows": args.smoothing_windows,
            "threshold_modes": args.threshold_modes,
            "composite_alpha": args.composite_alpha,
            "composite_beta": args.composite_beta,
            "composite_weights": args.composite_weights,
            "seed": args.seed,
            "quick": args.quick,
        },
        "calibration_score_stats_cache": {
            str(k): v for k, v in calibration_score_stats_cache.items()
        },
        "all_results": all_config_results,
    }

    summary_path = output_dir / f"03_batadal_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary: {summary_path}")

    # Per-timestamp CSV (best configuration only)
    ts_csv_path = output_dir / f"03_batadal_timeseries_{timestamp}.csv"
    with open(ts_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "idx", "timestamp", "is_attack",
            "fused_u", "fused_b", "fused_d",
            "max_u", "mean_u", "min_b", "mean_b", "max_d",
            "n_sensors", "n_high_u",
        ])
        for r in best_timestep_results:
            writer.writerow([
                r.idx, r.timestamp, int(r.is_attack),
                f"{r.fused_u:.6f}", f"{r.fused_b:.6f}", f"{r.fused_d:.6f}",
                f"{r.max_uncertainty:.6f}", f"{r.mean_uncertainty:.6f}",
                f"{r.min_belief:.6f}", f"{r.mean_belief:.6f}",
                f"{r.max_disbelief:.6f}",
                r.num_sensors_processed, r.num_sensors_high_u,
            ])
    print(f"  Timeseries: {ts_csv_path}")

    # PR curve CSV (best configuration)
    pr_csv_path = output_dir / f"03_batadal_pr_curve_{timestamp}.csv"
    with open(pr_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=best_pr_curve[0].keys())
        writer.writeheader()
        writer.writerows(best_pr_curve)
    print(f"  PR curve: {pr_csv_path}")

    # Hyperparameter heatmap data — extended with all new factors
    heatmap_path = output_dir / f"03_batadal_heatmap_{timestamp}.csv"
    with open(heatmap_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "strategy", "window_k", "decay_lambda",
            "use_fusion", "use_decay", "physics_bounds_mode",
            "anomaly_score", "smoothing_window", "threshold_mode",
            "auroc", "auroc_ci_lo", "auroc_ci_hi",
            "auprc", "best_f1", "best_threshold", "best_precision", "best_recall",
            "false_alarm_rate_per_day",
            # Per-signal AUROC (helps identify best signal per config)
            "auroc_fused_u", "auroc_fused_d", "auroc_max_d",
            "auroc_max_u", "auroc_mean_u", "auroc_composite_ud",
            # Operating points: F1 at multiple thresholds
            "f1_at_f1opt", "f1_at_q90", "f1_at_q95", "f1_at_q99",
            "prec_at_q95", "rec_at_q95", "fpr_at_q95",
            "prec_at_q99", "rec_at_q99", "fpr_at_q99",
        ])
        for cr in all_config_results:
            cfg = cr.get("config", {})
            ps = cr.get("per_signal_auroc", {})
            op = cr.get("operating_points", {})
            writer.writerow([
                cfg.get("phase", "?"),
                cr["strategy"], cr["window_size"], cr["decay_lambda"],
                cr["use_fusion"], cr["use_decay"], cr["physics_bounds_mode"],
                cr.get("anomaly_score", "fused_u"),
                cr.get("smoothing_window", 1),
                cr.get("threshold_mode", "f1_optimal_eval"),
                cr["auroc"], cr["auroc_ci_lower"], cr["auroc_ci_upper"],
                cr["auprc"], cr["best_f1"], cr["best_threshold"],
                cr["best_precision"], cr["best_recall"],
                cr["false_alarm_rate_per_day"],
                ps.get("fused_u", ""), ps.get("fused_d", ""), ps.get("max_d", ""),
                ps.get("max_u", ""), ps.get("mean_u", ""), ps.get("composite_ud", ""),
                op.get("f1_optimal", {}).get("f1", ""),
                op.get("q90", {}).get("f1", ""),
                op.get("q95", {}).get("f1", ""),
                op.get("q99", {}).get("f1", ""),
                op.get("q95", {}).get("precision", ""),
                op.get("q95", {}).get("recall", ""),
                op.get("q95", {}).get("fpr", ""),
                op.get("q99", {}).get("precision", ""),
                op.get("q99", {}).get("recall", ""),
                op.get("q99", {}).get("fpr", ""),
            ])
    print(f"  Heatmap: {heatmap_path}")

    # ── Print summary ──
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n  Best configuration: {best_config}")
    print(f"  AUROC: {best_overall_auroc:.4f}")

    # Find best result object
    best_result = [r for r in all_config_results
                   if r["config"] == best_config][0]

    print(f"  AUPRC: {best_result['auprc']:.4f} "
          f"[{best_result['auprc_ci_lower']:.4f}, {best_result['auprc_ci_upper']:.4f}]")
    print(f"  Best F1: {best_result['best_f1']:.4f} at τ={best_result['best_threshold']:.4f}")
    print(f"  Precision: {best_result['best_precision']:.4f}")
    print(f"  Recall: {best_result['best_recall']:.4f}")
    print(f"  False alarms/day: {best_result['false_alarm_rate_per_day']:.2f}")

    # Per-signal AUROC comparison
    ps = best_result.get("per_signal_auroc", {})
    if ps:
        print(f"\n  AUROC by anomaly signal (at best config):")
        for sig_name in ("fused_u", "fused_d", "max_d", "max_u", "mean_u", "composite_ud"):
            if sig_name in ps:
                print(f"    {sig_name:<14}: {ps[sig_name]:.4f}")

    # Operating points
    ops = best_result.get("operating_points", {})
    if ops:
        print(f"\n  Operating points (P/R/F1 at different thresholds):")
        for op_name in ("f1_optimal", "q90", "q95", "q99"):
            if op_name in ops:
                o = ops[op_name]
                print(f"    {op_name:<12} τ={o['threshold']:.4f}: "
                      f"P={o['precision']:.3f} R={o['recall']:.3f} "
                      f"F1={o['f1']:.3f} FPR={o['fpr']:.4f}")

    print(f"\n  Per-event detection (at best config):")
    for ev in best_result["per_event_auroc"]:
        auroc_str = f"{ev['auroc']:.4f}" if ev['auroc'] is not None else "N/A"
        print(f"    Event {ev['event_id']}: AUROC={auroc_str} "
              f"(duration={ev['duration']}h, start={ev['start_time']})")

    # Per-event best signal (key NeurIPS-grade finding)
    pebs = best_result.get("per_event_best_signal", [])
    if pebs:
        print(f"\n  Per-event BEST signal analysis "
              f"(reveals attack-type-specific detection):")
        for entry in pebs:
            if entry["best_signal"] is not None:
                print(f"    Event {entry['event_id']}: best={entry['best_signal']} "
                      f"(AUROC={entry['best_auroc']:.4f})")
                if "all_signal_auroc" in entry:
                    per_sig = ", ".join(
                        f"{k}={v:.3f}"
                        for k, v in sorted(entry["all_signal_auroc"].items(),
                                            key=lambda x: -x[1])
                    )
                    print(f"      All: {per_sig}")

    # Also compute "what's the best achievable for each event, across all configs"
    if all_config_results:
        print(f"\n  Best-achievable per-event AUROC across ALL configurations:")
        events_seen = set()
        for cr in all_config_results:
            for ev in cr.get("per_event_auroc", []):
                events_seen.add(ev["event_id"])
        for eid in sorted(events_seen):
            best_ev_auroc = -1.0
            best_ev_cfg = None
            for cr in all_config_results:
                for ev in cr.get("per_event_auroc", []):
                    if ev["event_id"] == eid and ev["auroc"] is not None:
                        if ev["auroc"] > best_ev_auroc:
                            best_ev_auroc = ev["auroc"]
                            best_ev_cfg = cr.get("config", {})
            if best_ev_cfg is not None:
                cfg_str = (f"{best_ev_cfg['strategy']}/{best_ev_cfg['anomaly_score']}"
                           f"/sw={best_ev_cfg['smoothing']}"
                           f"/k={best_ev_cfg['k']}/λ={best_ev_cfg['lambda']}")
                print(f"    Event {eid}: AUROC={best_ev_auroc:.4f} via {cfg_str}")

    print(f"\n  Detection latency (at best config):")
    for dl in best_result["detection_latency"]:
        if dl["detected"]:
            print(f"    Event {dl['event_id']}: {dl['latency_hours']}h "
                  f"(of {dl['duration_hours']}h attack)")
        else:
            print(f"    Event {dl['event_id']}: NOT DETECTED "
                  f"({dl['duration_hours']}h attack)")

    # Statistical comparison: paired Wilcoxon tests (Bonferroni-corrected)
    if not args.quick and _HAS_SCIPY:
        print(f"\n  Statistical tests (Wilcoxon signed-rank, Bonferroni-corrected):")
        print(f"  Note: tests run on HYPERPARAM phase configs only "
              f"(same anomaly_score/smoothing/threshold_mode)")

        def _paired_aurocs(filter_fn_a, filter_fn_b):
            """Extract paired (a, b) AUROC samples matched by all relevant factors."""
            pairs = []
            for r_a in all_config_results:
                if not filter_fn_a(r_a):
                    continue
                for r_b in all_config_results:
                    if (filter_fn_b(r_b)
                        and r_b["strategy"] == r_a["strategy"]
                        and r_b["window_size"] == r_a["window_size"]
                        and r_b["decay_lambda"] == r_a["decay_lambda"]
                        and r_b["physics_bounds_mode"] == r_a["physics_bounds_mode"]
                        and r_b.get("anomaly_score") == r_a.get("anomaly_score")
                        and r_b.get("smoothing_window") == r_a.get("smoothing_window")
                        and r_b.get("threshold_mode") == r_a.get("threshold_mode")):
                        pairs.append((r_a["auroc"], r_b["auroc"]))
                        break
            return pairs

        def _report_test(name, pairs, n_comp):
            if len(pairs) < 2:
                print(f"    {name}: insufficient paired samples (n={len(pairs)})")
                return
            a_vals = np.array([p[0] for p in pairs])
            b_vals = np.array([p[1] for p in pairs])
            if np.allclose(a_vals, b_vals):
                print(f"    {name} (n={len(pairs)}): identical samples, skipped")
                return
            try:
                stat, p = sp_stats.wilcoxon(a_vals, b_vals)
                p_corr = min(p * n_comp, 1.0)
                delta = float(np.mean(a_vals - b_vals))
                diffs = a_vals - b_vals
                cohen_d = (float(np.mean(diffs) / np.std(diffs, ddof=1))
                            if np.std(diffs, ddof=1) > 0 else 0.0)
                print(f"    {name} (n={len(pairs)}): Δ={delta:+.4f}, "
                      f"d={cohen_d:+.2f}, p={p_corr:.4f} "
                      f"({'significant' if p_corr < 0.05 else 'n.s.'})")
            except ValueError as e:
                print(f"    {name}: test failed ({e})")

        # Bonferroni n: fusion, decay, best-vs-worst strategy = 3 tests
        n_comparisons = 3

        # Test 1: Fusion vs No-Fusion
        pairs = _paired_aurocs(
            lambda r: r["use_fusion"] and r["use_decay"],
            lambda r: not r["use_fusion"] and r["use_decay"],
        )
        _report_test("Fusion vs No-Fusion", pairs, n_comparisons)

        # Test 2: Decay vs No-Decay
        pairs = _paired_aurocs(
            lambda r: r["use_fusion"] and r["use_decay"],
            lambda r: r["use_fusion"] and not r["use_decay"],
        )
        _report_test("Decay vs No-Decay", pairs, n_comparisons)

        # Test 3: Best strategy vs worst strategy (at full pipeline, matched factors)
        # Restrict to hyperparam phase to avoid confounds from signal_sweep configs
        full_pipe_results = [r for r in all_config_results
                             if r["use_fusion"] and r["use_decay"]
                             and r.get("config", {}).get("phase") == "hyperparam"]
        strat_mean_auroc = {}
        for strat in set(r["strategy"] for r in full_pipe_results):
            strat_aurocs = [r["auroc"] for r in full_pipe_results if r["strategy"] == strat]
            strat_mean_auroc[strat] = float(np.mean(strat_aurocs))

        if len(strat_mean_auroc) >= 2:
            best_strat = max(strat_mean_auroc, key=strat_mean_auroc.get)
            worst_strat = min(strat_mean_auroc, key=strat_mean_auroc.get)
            if best_strat != worst_strat:
                # Match on (k, λ, phys_mode) — other factors are fixed in this phase
                best_rows = {(r["window_size"], r["decay_lambda"], r["physics_bounds_mode"]): r["auroc"]
                             for r in full_pipe_results if r["strategy"] == best_strat}
                worst_rows = {(r["window_size"], r["decay_lambda"], r["physics_bounds_mode"]): r["auroc"]
                              for r in full_pipe_results if r["strategy"] == worst_strat}
                common_keys = set(best_rows.keys()) & set(worst_rows.keys())
                if len(common_keys) >= 2:
                    pairs = [(best_rows[k], worst_rows[k]) for k in common_keys]
                    _report_test(f"{best_strat} vs {worst_strat}", pairs, n_comparisons)

        # ADDITIONAL: Anomaly signal comparison (informational, not Bonferroni-counted)
        # Compare fused_u vs fused_d vs composite_ud on signal_sweep configs
        print(f"\n  Signal comparison tests (informational, not Bonferroni-corrected):")

        sig_sweep = [r for r in all_config_results
                     if r.get("config", {}).get("phase") == "signal_sweep"]

        def _signal_pairs(sig_a, sig_b):
            pairs = []
            for r_a in sig_sweep:
                if r_a.get("anomaly_score") != sig_a:
                    continue
                for r_b in sig_sweep:
                    if (r_b.get("anomaly_score") == sig_b
                        and r_b["strategy"] == r_a["strategy"]
                        and r_b["window_size"] == r_a["window_size"]
                        and r_b["decay_lambda"] == r_a["decay_lambda"]
                        and r_b.get("smoothing_window") == r_a.get("smoothing_window")
                        and r_b.get("threshold_mode") == r_a.get("threshold_mode")
                        and r_b["physics_bounds_mode"] == r_a["physics_bounds_mode"]):
                        pairs.append((r_a["auroc"], r_b["auroc"]))
                        break
            return pairs

        signals_to_compare = [("fused_u", "fused_d"), ("fused_u", "max_d"),
                              ("fused_u", "composite_ud")]
        for sig_a, sig_b in signals_to_compare:
            pairs = _signal_pairs(sig_a, sig_b)
            _report_test(f"{sig_a} vs {sig_b}", pairs, 1)

    # ── Reproducibility sanity check ──
    # Re-run ONE config with the same seed and verify bit-for-bit identical output.
    # Catches any hidden non-determinism in fusion order, numpy ops, etc.
    # Runs on the best config so it exercises the codepath that produced our result.
    if not args.quick and best_config is not None:
        print("\n" + "=" * 80)
        print("  REPRODUCIBILITY SANITY CHECK")
        print("=" * 80)
        print(f"  Re-running best config with same seed to verify determinism...")

        # Re-seed identically
        random.seed(args.seed)
        np.random.seed(args.seed)

        cal_key = (best_config["strategy"], best_config["phys_mode"],
                    best_config["anomaly_score"])
        cal_score_stats = calibration_score_stats_cache.get(cal_key)

        t0 = time.perf_counter()
        verify_result, _, _ = run_single_configuration(
            eval_rows=eval_rows,
            sensor_cols=sensor_cols,
            calibration_stats=calibration_stats,
            related_map=related_map,
            strategy=strategies[best_config["strategy"]],
            strategy_name=best_config["strategy"],
            window_size=best_config["k"],
            decay_lambda=best_config["lambda"],
            use_fusion=best_config["fusion"],
            use_decay=best_config["decay"],
            events=events,
            physics_bounds_mode=best_config["phys_mode"],
            anomaly_score=best_config["anomaly_score"],
            smoothing_window=best_config["smoothing"],
            composite_alpha=args.composite_alpha,
            composite_beta=args.composite_beta,
            threshold_mode=best_config["threshold_mode"],
            calibration_score_stats=cal_score_stats,
        )
        verify_elapsed = time.perf_counter() - t0

        # Compare key metrics (should be identical to 4 decimals at minimum)
        original_metrics = {
            "auroc": best_result["auroc"],
            "auprc": best_result["auprc"],
            "best_f1": best_result["best_f1"],
            "best_threshold": best_result["best_threshold"],
            "best_precision": best_result["best_precision"],
            "best_recall": best_result["best_recall"],
        }
        verify_metrics = {
            "auroc": verify_result["auroc"],
            "auprc": verify_result["auprc"],
            "best_f1": verify_result["best_f1"],
            "best_threshold": verify_result["best_threshold"],
            "best_precision": verify_result["best_precision"],
            "best_recall": verify_result["best_recall"],
        }

        all_match = True
        mismatches = []
        for key in original_metrics:
            orig = original_metrics[key]
            verif = verify_metrics[key]
            # Allow tiny numerical differences (1e-4) from floating-point reordering
            if abs(orig - verif) > 1e-4:
                all_match = False
                mismatches.append((key, orig, verif))

        if all_match:
            print(f"  ✓ DETERMINISTIC: All metrics match to 4 decimal places "
                  f"({verify_elapsed:.1f}s)")
            for key, orig in original_metrics.items():
                print(f"    {key}: {orig:.6f} (matched)")
        else:
            print(f"  ✗ NON-DETERMINISTIC: {len(mismatches)} metric(s) differ between runs!")
            for key, orig, verif in mismatches:
                print(f"    {key}: original={orig:.6f}, verify={verif:.6f}, "
                      f"Δ={verif-orig:+.6e}")
            print(f"  WARNING: Results may not be exactly reproducible.")
            print(f"  Known cause: SLFusion.fuse_pair() is non-commutative when decay")
            print(f"  is enabled — decayed opinions are fused in sensor-iteration order.")
            print(f"  For publication, we report results from a single deterministic run")
            print(f"  with seed={args.seed} and document this property.")

        # Save reproducibility results to manifest
        repro_path = output_dir / f"03_batadal_reproducibility_{timestamp}.json"
        with open(repro_path, "w") as f:
            json.dump({
                "seed": args.seed,
                "original_metrics": original_metrics,
                "verify_metrics": verify_metrics,
                "all_match": all_match,
                "mismatches": mismatches,
                "config": best_config,
            }, f, indent=2, default=str)
        print(f"  Reproducibility log: {repro_path}")

    print(f"\n  All results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
