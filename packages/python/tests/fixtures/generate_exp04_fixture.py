"""Generate a realistic synthetic trials.csv + summary.json matching the
schema produced by experiments/04_batadal_llm.py.

The synthetic data is seeded (deterministic) and shaped with realistic
condition × label distributions so that:
  - Condition C (full pipeline) beats G (no epistemic) on attack detection.
  - Condition F (no guardrails) over-triggers on normal windows.
  - Condition E1 (vacuous u) is degenerate.
  - Overall patterns align with the exp 02/03 findings so the analysis
    pipeline exercises its full range of statistical tests.

USAGE
-----
    python generate_exp04_fixture.py --output-dir ./out
produces:
    out/04_batadal_llm_synthetic_19700101_000000_trials.csv
    out/04_batadal_llm_synthetic_19700101_000000.json

These are structurally identical to real exp 04 outputs and can be fed
directly to experiments/04_batadal_llm_analysis.py for verification.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ALL_CONDITIONS = ["A", "B", "C", "D", "E1", "E2", "F", "G"]

# Realistic action-class distributions per condition, per label.
# Each tuple: (p_safety, p_monitor, p_unclear). Must sum to 1.
# Designed so the analysis-script outputs are nontrivial but plausible.
DIST_ATTACK = {
    "A":  (0.50, 0.35, 0.15),  # bare LLM - context-free, medium
    "B":  (0.65, 0.25, 0.10),  # raw payloads - reasonable
    "C":  (0.75, 0.15, 0.10),  # full pipeline - best
    "D":  (0.70, 0.20, 0.10),  # no decay - nearly full
    "E1": (0.25, 0.60, 0.15),  # vacuous u - degenerate
    "E2": (0.70, 0.20, 0.10),  # passthrough - ~full
    "F":  (0.80, 0.10, 0.10),  # no guardrails - aggressive
    "G":  (0.40, 0.50, 0.10),  # no epistemic - worse than C
}
DIST_NORMAL = {
    "A":  (0.20, 0.70, 0.10),
    "B":  (0.15, 0.75, 0.10),
    "C":  (0.10, 0.80, 0.10),  # few false alarms
    "D":  (0.12, 0.78, 0.10),
    "E1": (0.15, 0.75, 0.10),
    "E2": (0.12, 0.78, 0.10),
    "F":  (0.25, 0.65, 0.10),  # over-triggers without guardrails
    "G":  (0.35, 0.55, 0.10),  # over-triggers worst
}

# Canned action strings per class, for realism.
ACTION_STRINGS = {
    "safety": [
        "Shutdown pipeline immediately",
        "Close distribution valves",
        "Trigger emergency protocol",
        "Isolate affected tanks",
        "Activate alarm",
    ],
    "monitor": [
        "Monitor and adjust the water flow rate based on sensor data",
        "Continue monitoring the network",
        "Observe sensor readings",
        "Check tank levels",
        "Report current status",
    ],
    "unclear": [
        "",
        "The system indicates potential anomalies in the water network",
        "Consider reviewing the data",
    ],
}


def _sample_action(rng: random.Random, label: int,
                   condition: str) -> tuple[str, str, bool]:
    """Return (action_class, action_str, json_valid)."""
    dist = DIST_ATTACK[condition] if label == 1 else DIST_NORMAL[condition]
    r = rng.random()
    if r < dist[0]:
        cls = "safety"
    elif r < dist[0] + dist[1]:
        cls = "monitor"
    else:
        cls = "unclear"
    action_str = rng.choice(ACTION_STRINGS[cls])
    # JSON compliance: high for most conditions, a bit lower for E1 (vacuous),
    # and lowest for unclear (often truncation).
    if cls == "unclear":
        json_valid = rng.random() < 0.3
    else:
        json_valid = rng.random() < 0.95
    return cls, action_str, json_valid


def _sample_uncertainty(rng: random.Random, label: int,
                        condition: str) -> float:
    """Simulate max_uncertainty. Vacuous conditions = 1.0, others realistic."""
    if condition in ("E1", "G"):
        return 1.0
    if condition in ("A", "B"):
        # These don't use epistemic - threshold check doesn't apply, but the
        # pipeline still records max_u from the state. Fake it as moderate.
        return round(rng.uniform(0.3, 0.7), 6)
    # Epistemic-enabled conditions: attack windows show higher u than normal
    if label == 1:
        return round(rng.uniform(0.08, 0.45), 6)
    return round(rng.uniform(0.03, 0.20), 6)


def _sample_latency(rng: random.Random, condition: str) -> float:
    """Bonsai-1.7B-like latencies ~2500 ms, with some spread."""
    base = 2500.0
    return round(rng.gauss(base, 400), 1)


def generate(seed: int = 42, n_attack_per_event: int = 10,
             n_normal: int = 50, n_events: int = 5,
             conditions: list[str] = None, reps: int = 10,
             model_name: str = "synthetic",
             timestamp: str = "19700101_000000") -> tuple[list[dict], dict]:
    """Build trials list and summary dict.

    Returns:
        (trials, summary) where trials is a list of dicts matching the CSV
        schema and summary is a dict matching the exp 04 JSON structure.
    """
    if conditions is None:
        conditions = ALL_CONDITIONS

    rng = random.Random(seed)

    # Build window list: attack-adjacent (5 events × 10 each) + 50 normal baselines
    windows = []
    idx_counter = 100
    for event_id in range(1, n_events + 1):
        half = n_attack_per_event // 2
        # Onset-centered
        for _ in range(half):
            windows.append({
                "idx": idx_counter,
                "label": 1,
                "event_id": event_id,
                "position": "onset",
            })
            idx_counter += 1
        # Recovery-centered
        for _ in range(half):
            windows.append({
                "idx": idx_counter,
                "label": 1,
                "event_id": event_id,
                "position": "recovery",
            })
            idx_counter += 1

    # Normal baselines
    for _ in range(n_normal):
        windows.append({
            "idx": idx_counter,
            "label": 0,
            "event_id": None,
            "position": "normal_baseline",
        })
        idx_counter += 1

    trials = []
    for condition in conditions:
        for window in windows:
            for rep in range(reps):
                cls, action_str, json_valid = _sample_action(
                    rng, window["label"], condition
                )
                max_u = _sample_uncertainty(rng, window["label"], condition)
                latency = _sample_latency(rng, condition)

                # Guardrail decisions
                threshold_perm = max_u < 0.15 if condition not in ("A", "B", "F") else True
                if condition in ("E1", "G"):
                    threshold_perm = False
                whitelist_perm = cls in ("safety", "monitor")
                guardrails_active = condition in ("C", "D", "E1", "E2", "G")
                combined_perm = (threshold_perm and whitelist_perm) if guardrails_active else True

                # Detection outcomes (mirror the exp 04 logic)
                is_attack = window["label"] == 1
                det_tp = is_attack and cls == "safety"
                det_tn = (not is_attack) and cls == "monitor"
                det_fp = (not is_attack) and cls == "safety"
                det_fn = is_attack and cls == "monitor"

                trials.append({
                    "window_idx": window["idx"],
                    "window_label": window["label"],
                    "event_id": window["event_id"] if window["event_id"] else "",
                    "position": window["position"],
                    "condition": condition,
                    "rep": rep,
                    "json_valid": int(json_valid),
                    "action_str": action_str,
                    "action_class": cls,
                    "max_uncertainty": f"{max_u:.6f}",
                    "threshold_permitted": int(threshold_perm),
                    "whitelist_permitted": int(whitelist_perm),
                    "combined_permitted": int(combined_perm),
                    "detection_tp": int(det_tp),
                    "detection_tn": int(det_tn),
                    "detection_fp": int(det_fp),
                    "detection_fn": int(det_fn),
                    "llm_latency_ms": f"{latency:.1f}",
                    "completion_tokens": rng.randint(5, 30),
                    "prompt_tokens": rng.randint(300, 800),
                    "reasoning_tokens": 0,
                })

    summary = {
        "experiment": "04_batadal_llm",
        "timestamp": timestamp,
        "model_name": model_name,
        "model_id": None,
        "strategy": "historical",
        "seed": seed,
        "n_windows": len(windows),
        "n_attack_windows": sum(1 for w in windows if w["label"] == 1),
        "n_normal_windows": sum(1 for w in windows if w["label"] == 0),
        "n_reps": reps,
        "n_conditions": len(conditions),
        "total_trials": len(trials),
        "pre_registered_config": {
            "WINDOW_LOOKBACK_HOURS": 6,
            "BOUNDARY_MARGIN_HOURS": 5,
            "NORMAL_BUFFER_HOURS": 48,
            "TOP_K_PER_CATEGORY": 2,
            "UNCERTAINTY_THRESHOLD": 0.15,
            "DECAY_LAMBDA": 0.25,
            "PHYSICS_BOUNDS_MODE": "calibration_percentile",
        },
        "windows": windows,
        "_note": "SYNTHETIC FIXTURE - generated by generate_exp04_fixture.py",
    }
    return trials, summary


def write_outputs(output_dir: Path, model_name: str, timestamp: str,
                  trials: list[dict], summary: dict) -> tuple[Path, Path]:
    """Write trials CSV and summary JSON; return the paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / f"04_batadal_llm_{model_name}_{timestamp}_trials.csv"
    summary_path = output_dir / f"04_batadal_llm_{model_name}_{timestamp}.json"

    with open(trials_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window_idx", "window_label", "event_id", "position",
            "condition", "rep",
            "json_valid", "action_str", "action_class",
            "max_uncertainty", "threshold_permitted", "whitelist_permitted",
            "combined_permitted",
            "detection_tp", "detection_tn", "detection_fp", "detection_fn",
            "llm_latency_ms", "completion_tokens", "prompt_tokens",
            "reasoning_tokens",
        ])
        for t in trials:
            writer.writerow([
                t["window_idx"], t["window_label"], t["event_id"], t["position"],
                t["condition"], t["rep"],
                t["json_valid"], t["action_str"], t["action_class"],
                t["max_uncertainty"], t["threshold_permitted"],
                t["whitelist_permitted"], t["combined_permitted"],
                t["detection_tp"], t["detection_tn"],
                t["detection_fp"], t["detection_fn"],
                t["llm_latency_ms"], t["completion_tokens"],
                t["prompt_tokens"], t["reasoning_tokens"],
            ])

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return trials_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic exp 04 fixture")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="synthetic")
    parser.add_argument("--timestamp", default="19700101_000000")
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()

    trials, summary = generate(
        seed=args.seed, reps=args.reps,
        model_name=args.model_name, timestamp=args.timestamp,
    )
    trials_path, summary_path = write_outputs(
        args.output_dir, args.model_name, args.timestamp, trials, summary
    )
    print(f"Wrote {len(trials)} trials to {trials_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
