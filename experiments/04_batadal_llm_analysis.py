"""Experiment 04 analysis: consume trials.csv + summary.json, produce
publication-grade CSVs and PDF figures.

This script performs POST-HOC analysis on exp 04 output. It does NOT run
any LLM inference. It reads the two files produced by
experiments/04_batadal_llm.py and emits a results bundle to
results/batadal_llm/analysis/<run-tag>/.

Core metric computations (confusion counts, classification metrics, Cohen's
kappa, Cohen's d) are delegated to the unit-tested module
    epistemic_edge.analysis.metrics

All aggregation, statistical testing, CSV writing, and figure generation
is orchestrated here.

OUTPUTS
-------
CSVs:
    00_condition_summary.csv           per-condition metrics
    01_per_position.csv                condition x position
    02_per_event.csv                   event x condition (detection rate)
    03_action_crosstab.csv             condition x label x action_class counts
    04_over_trigger.csv                pre-guardrail over-trigger rate
    05_statistical_tests.csv           paired Wilcoxon (C vs each), Cohen's d
    06_bootstrap_f1.csv                event-clustered bootstrap F1 with CI

Figures (PDF):
    fig1_forest_f1.pdf                 F1 per condition with 95% CIs
    fig2_per_event_heatmap.pdf         5 events x 8 conditions detection rate
    fig3_confusion_grid.pdf            2x4 grid of per-condition confusion mtx
    fig4_action_class_stacked.pdf      stacked bars of action_class by label

USAGE
-----
    python 04_batadal_llm_analysis.py \\
        --trials-csv results/batadal_llm/04_batadal_llm_<model>_<ts>_trials.csv \\
        --summary-json results/batadal_llm/04_batadal_llm_<model>_<ts>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Make the package importable from the repo root
_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT / "packages" / "python" / "src"))

try:
    import numpy as np
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("WARNING: numpy/scipy not found - bootstrap CIs and Wilcoxon skipped")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    print("WARNING: matplotlib not found - figures skipped")

from epistemic_edge.analysis.metrics import (
    cohen_d_paired,
    compute_classification_metrics,
    compute_cohen_kappa,
    compute_confusion_counts,
)

ALL_CONDITIONS = ["A", "B", "C", "D", "E1", "E2", "F", "G"]

CONDITION_LABELS = {
    "A":  "bare_llm",
    "B":  "raw_payloads",
    "C":  "full_pipeline",
    "D":  "no_decay",
    "E1": "no_fusion_vacuous",
    "E2": "no_fusion_passthrough",
    "F":  "no_guardrails",
    "G":  "no_epistemic",
}

# Bonferroni denominator: 7 comparisons (C vs each other condition)
N_COMPARISONS = 7


# ============================================================================
# Data loading
# ============================================================================

def load_trials_csv(path: Path) -> list[dict]:
    """Load trials CSV and convert numeric fields to proper types."""
    trials = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = {}
            t["window_idx"] = int(row["window_idx"])
            t["window_label"] = int(row["window_label"])
            event_raw = row["event_id"]
            t["event_id"] = int(event_raw) if event_raw and event_raw != "None" else None
            t["position"] = row["position"]
            t["condition"] = row["condition"]
            t["rep"] = int(row["rep"])
            t["json_valid"] = int(row["json_valid"]) == 1
            t["action_str"] = row["action_str"]
            t["action_class"] = row["action_class"]
            t["max_uncertainty"] = float(row["max_uncertainty"])
            t["threshold_permitted"] = int(row["threshold_permitted"]) == 1
            t["whitelist_permitted"] = int(row["whitelist_permitted"]) == 1
            t["combined_permitted"] = int(row["combined_permitted"]) == 1
            t["detection_tp"] = int(row["detection_tp"]) == 1
            t["detection_tn"] = int(row["detection_tn"]) == 1
            t["detection_fp"] = int(row["detection_fp"]) == 1
            t["detection_fn"] = int(row["detection_fn"]) == 1
            t["llm_latency_ms"] = float(row["llm_latency_ms"])
            t["completion_tokens"] = int(row["completion_tokens"])
            t["prompt_tokens"] = int(row["prompt_tokens"])
            t["reasoning_tokens"] = int(row["reasoning_tokens"])
            trials.append(t)
    return trials


def load_summary_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Aggregation helpers (pure functions on trial lists)
# ============================================================================

def filter_trials(trials: list[dict], **kwargs) -> list[dict]:
    """Filter trials by any subset of field=value predicates."""
    def matches(t):
        for k, v in kwargs.items():
            if t.get(k) != v:
                return False
        return True
    return [t for t in trials if matches(t)]


def per_condition_metrics(trials: list[dict],
                          condition: str) -> dict:
    """All metrics for a single condition."""
    c_trials = filter_trials(trials, condition=condition)
    counts = compute_confusion_counts(c_trials)
    metrics = compute_classification_metrics(counts)
    kappa = compute_cohen_kappa(c_trials)
    # Ancillary: JSON compliance, latency, over-trigger
    n = len(c_trials)
    if n > 0:
        json_ok = sum(1 for t in c_trials if t["json_valid"]) / n
        avg_lat = sum(t["llm_latency_ms"] for t in c_trials) / n
    else:
        json_ok = 0.0
        avg_lat = 0.0
    # Over-trigger: fraction of NORMAL trials that produced a safety action
    normal_trials = filter_trials(c_trials, window_label=0)
    if normal_trials:
        ot = sum(1 for t in normal_trials
                 if t["action_class"] == "safety") / len(normal_trials)
    else:
        ot = 0.0
    # Unclear rate
    unclear_rate = counts["unclear"] / n if n > 0 else 0.0
    return {
        "condition": condition,
        "label": CONDITION_LABELS[condition],
        **counts,
        **metrics,
        "cohen_kappa": kappa,
        "json_compliance": json_ok,
        "avg_latency_ms": avg_lat,
        "over_trigger_rate": ot,
        "unclear_rate": unclear_rate,
    }


def per_position_breakdown(trials: list[dict]) -> list[dict]:
    """For each (condition, position), compute full metrics."""
    conditions = sorted(set(t["condition"] for t in trials))
    positions = sorted(set(t["position"] for t in trials))
    rows = []
    for cond in conditions:
        for pos in positions:
            subset = filter_trials(trials, condition=cond, position=pos)
            counts = compute_confusion_counts(subset)
            metrics = compute_classification_metrics(counts)
            rows.append({
                "condition": cond,
                "position": pos,
                **counts,
                **metrics,
            })
    return rows


def per_event_detection(trials: list[dict]) -> list[dict]:
    """For each (event_id, condition), compute attack-recall (detection rate).

    For attack windows only. Detection rate = fraction of attack trials for
    that event that produced a safety action (i.e., recall restricted to one
    event).
    """
    attack_trials = [t for t in trials if t["window_label"] == 1
                     and t["event_id"] is not None]
    conditions = sorted(set(t["condition"] for t in attack_trials))
    events = sorted(set(t["event_id"] for t in attack_trials))
    rows = []
    for event_id in events:
        for cond in conditions:
            subset = [t for t in attack_trials
                      if t["event_id"] == event_id and t["condition"] == cond]
            n = len(subset)
            n_safety = sum(1 for t in subset if t["action_class"] == "safety")
            n_monitor = sum(1 for t in subset if t["action_class"] == "monitor")
            n_unclear = sum(1 for t in subset if t["action_class"] == "unclear")
            detection_rate = n_safety / n if n > 0 else 0.0
            rows.append({
                "event_id": event_id,
                "condition": cond,
                "n": n,
                "safety": n_safety,
                "monitor": n_monitor,
                "unclear": n_unclear,
                "detection_rate": detection_rate,
            })
    return rows


def action_class_crosstab(trials: list[dict]) -> list[dict]:
    """For each (condition, label, action_class), count trials."""
    counter: dict[tuple[str, int, str], int] = defaultdict(int)
    for t in trials:
        key = (t["condition"], t["window_label"], t["action_class"])
        counter[key] += 1
    rows = []
    for (cond, label, cls), count in sorted(counter.items()):
        rows.append({
            "condition": cond,
            "label": "attack" if label == 1 else "normal",
            "action_class": cls,
            "count": count,
        })
    return rows


# ============================================================================
# Statistical tests (paired across conditions on common (window_idx, rep) keys)
# ============================================================================

def paired_correctness(trials: list[dict], cond: str) -> dict:
    """Map (window_idx, rep) -> binary correctness for one condition."""
    return {
        (t["window_idx"], t["rep"]): int(t["detection_tp"] or t["detection_tn"])
        for t in trials if t["condition"] == cond
    }


def paired_wilcoxon(trials: list[dict], cond_a: str,
                    cond_b: str) -> dict:
    """Paired Wilcoxon signed-rank test between two conditions on
    trial-level correctness (TP or TN = 1, else 0)."""
    a_map = paired_correctness(trials, cond_a)
    b_map = paired_correctness(trials, cond_b)
    common = sorted(set(a_map.keys()) & set(b_map.keys()))
    a_arr = [a_map[k] for k in common]
    b_arr = [b_map[k] for k in common]
    n_pairs = len(common)
    if n_pairs < 2:
        return {"cond_a": cond_a, "cond_b": cond_b, "n_pairs": n_pairs,
                "delta": 0.0, "cohen_d": 0.0, "stat": None,
                "p": None, "p_bonferroni": None, "significant": False}
    delta = (sum(a_arr) - sum(b_arr)) / n_pairs
    d = cohen_d_paired(a_arr, b_arr)
    stat = None
    p_val = None
    if _HAS_SCIPY and a_arr != b_arr:
        try:
            stat, p_val = sp_stats.wilcoxon(a_arr, b_arr, zero_method="wilcox")
        except ValueError:
            stat, p_val = None, None
    p_corr = min(p_val * N_COMPARISONS, 1.0) if p_val is not None else None
    return {
        "cond_a": cond_a,
        "cond_b": cond_b,
        "n_pairs": n_pairs,
        "delta": delta,
        "cohen_d": d,
        "stat": stat,
        "p": p_val,
        "p_bonferroni": p_corr,
        "significant": (p_corr is not None and p_corr < 0.05),
    }


def cluster_bootstrap_f1(trials: list[dict], condition: str,
                         n_boot: int = 1000, seed: int = 42
                         ) -> tuple[float, float, float]:
    """F1 cluster-bootstrap at event level for one condition.

    Clusters: each event_id (1..5) + one "normal" cluster for all normal
    trials. Resample clusters with replacement, recompute F1 on the
    aggregated resample.
    """
    c_trials = filter_trials(trials, condition=condition)
    counts = compute_confusion_counts(c_trials)
    point_f1 = compute_classification_metrics(counts)["f1"]

    if not _HAS_SCIPY:
        return point_f1, float("nan"), float("nan")

    clusters: dict[Any, list[dict]] = defaultdict(list)
    for t in c_trials:
        key = t["event_id"] if t["event_id"] is not None else "normal"
        clusters[key].append(t)
    keys = list(clusters.keys())
    if len(keys) < 2:
        return point_f1, float("nan"), float("nan")

    rng = np.random.RandomState(seed)
    boot_f1 = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(keys), size=len(keys))
        sampled = []
        for i in idx:
            sampled.extend(clusters[keys[i]])
        c = compute_confusion_counts(sampled)
        boot_f1.append(compute_classification_metrics(c)["f1"])
    lo = float(np.percentile(boot_f1, 2.5))
    hi = float(np.percentile(boot_f1, 97.5))
    return point_f1, lo, hi


# ============================================================================
# CSV writers
# ============================================================================

def write_csv(rows: list[dict], path: Path, fieldnames: list[str] = None) -> None:
    if not rows:
        path.write_text("")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================================
# Figures
# ============================================================================

def fig_forest_f1(bootstrap_rows: list[dict], out_path: Path) -> None:
    if not _HAS_MPL:
        return
    conditions = [r["condition"] for r in bootstrap_rows]
    f1s = [r["f1_point"] for r in bootstrap_rows]
    los = [r["f1_ci_lower"] for r in bootstrap_rows]
    his = [r["f1_ci_upper"] for r in bootstrap_rows]
    y_pos = list(range(len(conditions)))
    errors_lo = [max(0, f - lo) if lo == lo else 0
                 for f, lo in zip(f1s, los)]
    errors_hi = [max(0, hi - f) if hi == hi else 0
                 for f, hi in zip(f1s, his)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(f1s, y_pos, xerr=[errors_lo, errors_hi],
                fmt="o", color="black", capsize=4, markersize=7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{c} ({CONDITION_LABELS[c]})" for c in conditions])
    ax.set_xlabel("F1 (event-clustered bootstrap 95% CI)")
    ax.set_xlim(0, 1)
    ax.axvline(x=f1s[conditions.index("C")] if "C" in conditions else 0.5,
               color="red", linestyle="--", alpha=0.4, linewidth=1,
               label="Full pipeline (C)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_per_event_heatmap(per_event_rows: list[dict], out_path: Path) -> None:
    if not _HAS_MPL:
        return
    events = sorted(set(r["event_id"] for r in per_event_rows))
    conditions = [c for c in ALL_CONDITIONS
                  if c in {r["condition"] for r in per_event_rows}]
    matrix = np.zeros((len(events), len(conditions)))
    for r in per_event_rows:
        i = events.index(r["event_id"])
        j = conditions.index(r["condition"])
        matrix[i, j] = r["detection_rate"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels([f"Event {e}" for e in events])
    ax.set_xlabel("Condition")
    ax.set_title("Per-event detection rate (safety-action fraction on attack trials)")
    for i in range(len(events)):
        for j in range(len(conditions)):
            ax.text(j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if matrix[i, j] > 0.5 else "white")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.03)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_confusion_grid(summary_rows: list[dict], out_path: Path) -> None:
    """2x4 grid of per-condition confusion matrices (normalized)."""
    if not _HAS_MPL:
        return
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for ax, row in zip(axes.flatten(), summary_rows):
        tp, fp, fn, tn = row["tp"], row["fp"], row["fn"], row["tn"]
        total_attack = tp + fn
        total_normal = tn + fp
        matrix = np.array([
            [tp / total_attack if total_attack else 0,
             fn / total_attack if total_attack else 0],
            [fp / total_normal if total_normal else 0,
             tn / total_normal if total_normal else 0],
        ])
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["safety", "monitor"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["attack", "normal"])
        ax.set_title(f"{row['condition']}: F1={row['f1']:.2f}", fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="black" if matrix[i, j] < 0.5 else "white")
    fig.suptitle("Per-condition confusion matrices (row-normalized)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_action_class_stacked(crosstab_rows: list[dict], out_path: Path) -> None:
    """Stacked bars: for each (condition, label), fraction safety/monitor/unclear."""
    if not _HAS_MPL:
        return
    conditions = [c for c in ALL_CONDITIONS
                  if c in {r["condition"] for r in crosstab_rows}]
    labels = ["attack", "normal"]
    classes = ["safety", "monitor", "unclear"]
    colors = {"safety": "#d73027", "monitor": "#4575b4", "unclear": "#999999"}

    # Build fractions
    def get_fraction(cond, label, cls):
        matching = [r for r in crosstab_rows
                    if r["condition"] == cond and r["label"] == label]
        total = sum(r["count"] for r in matching)
        if total == 0:
            return 0.0
        for r in matching:
            if r["action_class"] == cls:
                return r["count"] / total
        return 0.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    x_pos = np.arange(len(conditions))
    for ax, label in zip(axes, labels):
        bottom = np.zeros(len(conditions))
        for cls in classes:
            fracs = np.array([get_fraction(c, label, cls) for c in conditions])
            ax.bar(x_pos, fracs, bottom=bottom, label=cls,
                   color=colors[cls], edgecolor="white")
            bottom += fracs
        ax.set_xticks(x_pos)
        ax.set_xticklabels(conditions)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of trials")
        ax.set_title(f"{label.capitalize()} windows")
        if label == "attack":
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Action-class distribution by condition and label")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main orchestration
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 04 analysis: consume trials.csv + summary.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--trials-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output dir; default derives from trials-csv basename")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir is None:
        # Strip "_trials.csv" suffix to get the run tag
        stem = args.trials_csv.stem
        if stem.endswith("_trials"):
            stem = stem[:-len("_trials")]
        args.output_dir = (args.trials_csv.parent / "analysis" / stem)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)

    print(f"Loading trials: {args.trials_csv}")
    trials = load_trials_csv(args.trials_csv)
    print(f"  {len(trials)} trials loaded")
    summary_meta = load_summary_json(args.summary_json)
    print(f"  Model: {summary_meta.get('model_name')}, "
          f"seed: {summary_meta.get('seed')}")

    conditions = sorted(set(t["condition"] for t in trials),
                        key=lambda c: ALL_CONDITIONS.index(c)
                        if c in ALL_CONDITIONS else 99)

    # ----- 00: per-condition summary -----
    print("\n[1/7] Per-condition summary")
    summary_rows = [per_condition_metrics(trials, c) for c in conditions]
    # Write clean, publication-friendly CSV
    summary_path = args.output_dir / "00_condition_summary.csv"
    write_csv(summary_rows, summary_path)
    # Print table
    print(f"  {'Cond':<5} {'n':<5} {'P':<6} {'R':<6} {'F1':<6} "
          f"{'Spec':<6} {'MCC':<6} {'κ':<6} {'OverTrig':<8}")
    for r in summary_rows:
        print(f"  {r['condition']:<5} {r['n']:<5} "
              f"{r['precision']:<6.3f} {r['recall']:<6.3f} {r['f1']:<6.3f} "
              f"{r['specificity']:<6.3f} {r['mcc']:<6.3f} {r['cohen_kappa']:<6.3f} "
              f"{r['over_trigger_rate']:<8.3f}")

    # ----- 01: per-position breakdown -----
    print("\n[2/7] Per-position breakdown")
    position_rows = per_position_breakdown(trials)
    write_csv(position_rows, args.output_dir / "01_per_position.csv")

    # ----- 02: per-event detection rates -----
    print("\n[3/7] Per-event detection rates")
    event_rows = per_event_detection(trials)
    write_csv(event_rows, args.output_dir / "02_per_event.csv")

    # ----- 03: action-class crosstab -----
    print("\n[4/7] Action-class crosstab")
    crosstab_rows = action_class_crosstab(trials)
    write_csv(crosstab_rows, args.output_dir / "03_action_crosstab.csv")

    # ----- 04: over-trigger rates -----
    print("\n[5/7] Over-trigger rates")
    ot_rows = [{"condition": r["condition"],
                "over_trigger_rate": r["over_trigger_rate"],
                "unclear_rate": r["unclear_rate"]}
               for r in summary_rows]
    write_csv(ot_rows, args.output_dir / "04_over_trigger.csv")

    # ----- 05: statistical tests (C vs each other) -----
    print("\n[6/7] Paired Wilcoxon vs C (Bonferroni-corrected)")
    stat_rows = []
    if "C" in conditions:
        for other in conditions:
            if other == "C":
                continue
            stat_rows.append(paired_wilcoxon(trials, "C", other))
            r = stat_rows[-1]
            p_str = f"{r['p_bonferroni']:.4f}" if r["p_bonferroni"] is not None else "n/a"
            sig = "*" if r["significant"] else " "
            print(f"  C vs {other}: n={r['n_pairs']:<5} "
                  f"Δ={r['delta']:+.3f} d={r['cohen_d']:+.3f} "
                  f"p_bonf={p_str} {sig}")
    # Serialize (stat column may be None/numpy scalar)
    for r in stat_rows:
        for k in ("stat", "p", "p_bonferroni", "cohen_d", "delta"):
            if r.get(k) is not None:
                r[k] = float(r[k])
    write_csv(stat_rows, args.output_dir / "05_statistical_tests.csv")

    # ----- 06: bootstrap F1 CIs -----
    print("\n[7/7] Event-clustered bootstrap F1 CIs")
    boot_rows = []
    for c in conditions:
        pt, lo, hi = cluster_bootstrap_f1(
            trials, c, n_boot=args.n_bootstrap, seed=args.seed
        )
        boot_rows.append({
            "condition": c,
            "f1_point": pt,
            "f1_ci_lower": lo,
            "f1_ci_upper": hi,
        })
        hi_str = f"{hi:.3f}" if hi == hi else "n/a"
        lo_str = f"{lo:.3f}" if lo == lo else "n/a"
        print(f"  {c}: F1 = {pt:.3f} [95% CI {lo_str}, {hi_str}]")
    write_csv(boot_rows, args.output_dir / "06_bootstrap_f1.csv")

    # ----- Figures -----
    if _HAS_MPL:
        print("\nGenerating figures...")
        fig_forest_f1(boot_rows, args.output_dir / "figures" / "fig1_forest_f1.pdf")
        fig_per_event_heatmap(event_rows,
                              args.output_dir / "figures" / "fig2_per_event_heatmap.pdf")
        fig_confusion_grid(summary_rows,
                           args.output_dir / "figures" / "fig3_confusion_grid.pdf")
        fig_action_class_stacked(crosstab_rows,
                                 args.output_dir / "figures" / "fig4_action_class_stacked.pdf")
        print(f"  4 figures written to {args.output_dir / 'figures'}")

    print(f"\nDone. All outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
