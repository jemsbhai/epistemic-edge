"""
Experiment 04: BATADAL × LLM × Conditions — Full Pipeline on Real Data.

Runs the complete Tier 1-4 Epistemic Edge pipeline (ingest → SL fusion →
temporal decay → LLM inference → dual guardrails) on real BATADAL attack
data across 8 experimental conditions.

Pre-registered design (NeurIPS-grade rigor):
  - Stratified sampling: 50 attack-adjacent + 50 normal baselines = 100 windows
  - Fixed seed (42) for reproducibility; robustness checks with seeds 43/44/45
  - Top-K=8 sensors per prompt, stratified across {tank, pump, pressure, valve}
  - Identical prompt format for attack and normal classes (no leakage)
  - Action classification declared a priori: SAFETY_ACTIONS vs MONITOR_ACTIONS
  - Cluster-robust bootstrap CIs at attack-event level
  - Paired Wilcoxon tests with Bonferroni correction across conditions

Ground truth mapping:
  Attack windows  → LLM should propose a SAFETY_ACTION (detection = TP)
  Normal windows  → LLM should propose a MONITOR_ACTION (correct = TN)

Usage:
    python 04_batadal_llm.py \\
        --calibration-data D:/cc/datasets/batadal/BATADAL_dataset03.csv \\
        --evaluation-data D:/cc/datasets/batadal/BATADAL_dataset04.csv \\
        --model-name bonsai-1.7B \\
        --port 8080 \\
        --reps 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
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
    print("WARNING: numpy/scipy not found.")

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from epistemic_edge.adapters.uncertainty import (
    CompositeStrategy,
    HistoricalDeviationStrategy,
    PhysicsBoundsStrategy,
    SensorAgreementStrategy,
    UncertaintyStrategy,
)
from epistemic_edge.adapters.base import SensorContext
from epistemic_edge.models import FusedState
from epistemic_edge.trust.fusion import SLFusion


# ═══════════════════════════════════════════════════════════════════════════
# Pre-registered configuration (DO NOT CHANGE after results are seen)
# ═══════════════════════════════════════════════════════════════════════════

# Window sampling
WINDOW_LOOKBACK_HOURS = 6                  # Fixed lookback window for decay
BOUNDARY_MARGIN_HOURS = 5                  # ±5h around attack transitions
NORMAL_BUFFER_HOURS = 48                   # Normal samples ≥48h from any attack
N_ATTACK_WINDOWS = 50                      # Split across 5 events (10 each)
N_NORMAL_WINDOWS = 50                      # Uniformly sampled from safe periods

# Prompt construction
TOP_K_SENSORS = 8                          # Total sensors in prompt
TOP_K_PER_CATEGORY = 2                     # From each of {tank, pump, pressure, valve}

# Uncertainty threshold for guardrail
UNCERTAINTY_THRESHOLD = 0.15
DECAY_LAMBDA = 0.25                        # Best from exp 03
PHYSICS_BOUNDS_MODE = "calibration_percentile"  # Best performer from exp 03
UNCERTAINTY_STRATEGY = "historical"        # Best performer from exp 03 (0.8447 AUROC)

# Action vocabulary (declared a priori)
SAFETY_ACTIONS = {
    "shutdown", "shut_down", "shut", "stop", "halt",
    "isolate", "emergency_stop", "emergency_shutdown",
    "close_valve", "close", "disable",
    "alert", "alarm", "trigger_alarm", "activate_alarm", "raise_alarm",
    "evacuate", "escalate", "intervene",
    "initiate_shutdown", "activate_emergency", "emergency_response",
}

MONITOR_ACTIONS = {
    "monitor", "observe", "watch", "continue", "nominal",
    "check", "verify", "inspect", "query", "status",
    "report_status", "log", "log_status", "record",
    "noop", "no_action", "none", "ok", "healthy",
    "report", "assess",
}

# Bonferroni correction denominator (number of planned comparisons)
N_COMPARISONS = 7  # 7 paired condition comparisons against full pipeline

# Sensor category prefixes (for top-K stratification)
SENSOR_CATEGORIES = {
    "tank":     lambda s: s.startswith("L_T"),
    "pump":     lambda s: s.startswith("F_PU"),
    "pressure": lambda s: s.startswith("P_J"),
    "other":    lambda s: s.startswith(("F_V", "S_")),
}

# ═══════════════════════════════════════════════════════════════════════════
# Experimental conditions (mirror of exp 02, adapted for BATADAL)
# ═══════════════════════════════════════════════════════════════════════════

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

CONDITION_DESCRIPTIONS = {
    "A":  "Query only — no sensor context",
    "B":  "Query + raw sensor values — no SL annotations",
    "C":  "Full pipeline — SL fusion + decay + guardrails",
    "D":  "No decay — SL fusion + guardrails, stale data retained",
    "E1": "No fusion (vacuous) — decay + guardrails, u=1.0",
    "E2": "No fusion (passthrough) — decay + guardrails, per-sensor unfused",
    "F":  "No guardrails — full fusion + decay, all intents permitted",
    "G":  "No epistemic — raw payloads + guardrails (vacuous u)",
}


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading (shared helpers with exp 03)
# ═══════════════════════════════════════════════════════════════════════════

TANK_BOUNDS = {
    "L_T1": (0.0, 6.0), "L_T2": (0.0, 6.5), "L_T3": (0.0, 4.5),
    "L_T4": (0.0, 5.5), "L_T5": (0.0, 5.0), "L_T6": (0.0, 6.0),
    "L_T7": (0.0, 5.5),
}

RELATED_GROUPS = [
    ["L_T1", "L_T2"], ["L_T3", "L_T4"], ["L_T5", "L_T6", "L_T7"],
    ["F_PU1", "F_PU2", "F_PU3"], ["F_PU4", "F_PU5"], ["F_PU7", "F_PU8"],
]


def _parse_datetime(s: str) -> datetime:
    s = s.strip()
    for fmt in ["%d/%m/%y %H", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {s!r}")


def load_csv(path: str) -> tuple[list[str], list[dict[str, Any]]]:
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
    sensors = []
    for h in headers:
        if h.startswith(("L_T", "F_PU", "F_V", "P_J", "S_PU", "S_V")):
            sensors.append(h)
    return sensors


def identify_attack_events(rows: list[dict[str, Any]]) -> list[dict]:
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
            })
            in_attack = False
    if in_attack:
        events.append({
            "event_id": len(events) + 1,
            "start_idx": start_idx,
            "end_idx": len(rows) - 1,
            "duration": len(rows) - start_idx,
            "start_time": rows[start_idx].get("DATETIME", ""),
        })
    return events


# ═══════════════════════════════════════════════════════════════════════════
# Window Sampling (pre-registered, seeded)
# ═══════════════════════════════════════════════════════════════════════════

def sample_windows(
    rows: list[dict[str, Any]],
    events: list[dict],
    seed: int = 42,
    n_attack_per_event: int = 10,
    n_normal: int = 50,
    boundary_margin: int = 5,
    normal_buffer: int = 48,
) -> list[dict[str, Any]]:
    """Pre-registered stratified window sampling.

    For each attack event, sample n_attack_per_event/2 windows near onset
    and n_attack_per_event/2 near recovery (±boundary_margin hours).

    For normal baselines, sample n_normal windows from indices at least
    normal_buffer hours from any attack, using fixed seed.

    Returns list of dicts: {idx, label (0=normal, 1=attack), event_id, position_in_event}
    """
    rng = random.Random(seed)
    windows = []

    # Attack-adjacent windows (boundary-focused)
    for event in events:
        half = n_attack_per_event // 2
        # Onset-centered
        onset = event["start_idx"]
        for offset in range(-boundary_margin, boundary_margin + 1):
            idx = onset + offset
            if 0 <= idx < len(rows):
                windows.append({
                    "idx": idx,
                    "label": 1 if int(float(rows[idx].get("ATT_FLAG", -999))) == 1 else 0,
                    "event_id": event["event_id"],
                    "position": "onset",
                    "offset_from_event": offset,
                })
            if len([w for w in windows if w["event_id"] == event["event_id"]
                    and w["position"] == "onset"]) >= half:
                break

        # Recovery-centered
        recovery = event["end_idx"]
        for offset in range(-boundary_margin, boundary_margin + 1):
            idx = recovery + offset
            if 0 <= idx < len(rows):
                windows.append({
                    "idx": idx,
                    "label": 1 if int(float(rows[idx].get("ATT_FLAG", -999))) == 1 else 0,
                    "event_id": event["event_id"],
                    "position": "recovery",
                    "offset_from_event": offset,
                })
            if len([w for w in windows if w["event_id"] == event["event_id"]
                    and w["position"] == "recovery"]) >= half:
                break

    # Normal baselines (seeded uniform from safe periods)
    attack_mask = [False] * len(rows)
    for event in events:
        start = max(0, event["start_idx"] - normal_buffer)
        end = min(len(rows), event["end_idx"] + normal_buffer + 1)
        for i in range(start, end):
            attack_mask[i] = True

    safe_indices = [i for i, blocked in enumerate(attack_mask) if not blocked
                    and int(float(rows[i].get("ATT_FLAG", -999))) != 1]

    rng.shuffle(safe_indices)
    for idx in safe_indices[:n_normal]:
        windows.append({
            "idx": idx,
            "label": 0,
            "event_id": None,
            "position": "normal_baseline",
            "offset_from_event": None,
        })

    return windows


# ═══════════════════════════════════════════════════════════════════════════
# Sensor Context and Fused State Building
# ═══════════════════════════════════════════════════════════════════════════

def build_sensor_context(
    row: dict[str, Any],
    sensor_id: str,
    calibration_stats: dict[str, dict[str, float]],
    related_map: dict[str, list[str]],
) -> SensorContext | None:
    val = row.get(sensor_id)
    if not isinstance(val, (int, float)):
        return None

    cal = calibration_stats.get(sensor_id, {})
    domain_bounds = TANK_BOUNDS.get(sensor_id)

    if domain_bounds is not None:
        phys_min, phys_max = domain_bounds[0], domain_bounds[1]
    else:
        phys_min = cal.get("q01")
        phys_max = cal.get("q99")

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


def build_fused_state_at_window(
    rows: list[dict[str, Any]],
    t_idx: int,
    window_size: int,
    sensor_cols: list[str],
    calibration_stats: dict[str, dict[str, float]],
    related_map: dict[str, list[str]],
    strategy: UncertaintyStrategy,
    use_fusion: bool = True,
    use_decay: bool = True,
    decay_lambda: float = DECAY_LAMBDA,
) -> dict[str, dict[str, Any]]:
    """Build per-sensor fused state over a lookback window.

    Returns dict: sensor_id -> {reading, b, d, u, a, unit}
    """
    row = rows[t_idx]
    try:
        current_time = _parse_datetime(str(row.get("DATETIME", "")))
    except ValueError:
        current_time = datetime(2016, 1, 1, tzinfo=timezone.utc)

    start_idx = max(0, t_idx - window_size + 1)
    window_indices = range(start_idx, t_idx + 1)

    per_sensor_states: dict[str, FusedState] = {}
    per_sensor_readings: dict[str, float] = {}
    per_sensor_units: dict[str, str] = {}

    fusion = SLFusion()

    for w_idx in window_indices:
        w_row = rows[w_idx]
        try:
            w_time = _parse_datetime(str(w_row.get("DATETIME", "")))
        except ValueError:
            w_time = current_time
        age_hours = max(0.0, (current_time - w_time).total_seconds() / 3600)

        for sensor_id in sensor_cols:
            if sensor_id.startswith("S_"):
                continue  # skip binary actuator states

            ctx = build_sensor_context(w_row, sensor_id, calibration_stats, related_map)
            if ctx is None:
                continue

            b, d, u, a = strategy.assign(ctx)

            if use_decay and age_hours > 0:
                decay_factor = math.exp(-decay_lambda * age_hours)
                b = b * decay_factor
                d = d * decay_factor
                u_new = 1.0 - b - d
                u = max(0.0, u_new)
                total = b + d + u
                if total > 0:
                    b, d, u = b / total, d / total, u / total
                else:
                    b, d, u = 0.0, 0.0, 1.0

            # Track most recent reading/unit for the sensor
            if w_idx == t_idx:
                per_sensor_readings[sensor_id] = ctx.reading
                per_sensor_units[sensor_id] = ctx.unit

            if use_fusion:
                state = FusedState(
                    payload={sensor_id: ctx.reading},
                    belief=b, disbelief=d, uncertainty=u,
                    base_rate=a,
                    sources=[sensor_id],
                    fused_at=ctx.timestamp,
                )
                if sensor_id in per_sensor_states:
                    per_sensor_states[sensor_id] = fusion.fuse_pair(
                        per_sensor_states[sensor_id], state
                    )
                else:
                    per_sensor_states[sensor_id] = state
            else:
                # No fusion — use only the current timestep's opinion
                if w_idx == t_idx:
                    per_sensor_states[sensor_id] = FusedState(
                        payload={sensor_id: ctx.reading},
                        belief=b, disbelief=d, uncertainty=u,
                        base_rate=a, sources=[sensor_id],
                        fused_at=ctx.timestamp,
                    )

    # Build output
    result = {}
    for sid, state in per_sensor_states.items():
        result[sid] = {
            "reading": per_sensor_readings.get(sid, 0.0),
            "belief": state.belief,
            "disbelief": state.disbelief,
            "uncertainty": state.uncertainty,
            "base_rate": state.base_rate,
            "unit": per_sensor_units.get(sid, ""),
        }
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Top-K Sensor Selection (pre-registered: stratified by category)
# ═══════════════════════════════════════════════════════════════════════════

def select_top_k_sensors(
    fused_state: dict[str, dict[str, Any]],
    k_per_category: int = TOP_K_PER_CATEGORY,
) -> list[str]:
    """Select top sensors by uncertainty+disbelief, stratified by category.

    Pre-registered rule: 2 most anomalous per category (tank, pump, pressure, other).
    Anomaly score = disbelief + uncertainty (higher = more alarming).
    """
    selected = []
    for cat_name, predicate in SENSOR_CATEGORIES.items():
        cat_sensors = [
            (sid, state["disbelief"] + state["uncertainty"])
            for sid, state in fused_state.items()
            if predicate(sid)
        ]
        cat_sensors.sort(key=lambda x: -x[1])  # highest first
        selected.extend([sid for sid, _ in cat_sensors[:k_per_category]])
    return selected


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Construction (identical format for attack and normal)
# ═══════════════════════════════════════════════════════════════════════════

_CONTROLLER_PREAMBLE = (
    "You are an AIoT edge controller for a water distribution network. "
    "Your role is to decide appropriate actions based on current sensor state."
)

_JSON_INSTRUCTION = (
    'Respond ONLY with a JSON object: '
    '{"action": "...", "target": "...", "parameters": {}}. '
    'No additional text.'
)


def build_prompt_A_bare(window_info: dict) -> str:
    """Condition A: query only, no sensor context."""
    return (
        f"{_CONTROLLER_PREAMBLE}\n\n"
        f"Based on the current system state, what action should be taken?\n\n"
        f"{_JSON_INSTRUCTION}"
    )


def build_prompt_B_raw(
    fused_state: dict[str, dict[str, Any]], top_k: list[str],
) -> str:
    """Condition B: raw sensor values, no SL annotations."""
    lines = []
    for sid in top_k:
        state = fused_state[sid]
        lines.append(f"- {sid}: {state['reading']:.2f} {state['unit']}")
    payload_block = "\n".join(lines) if lines else "  (no sensor data)"

    return (
        f"{_CONTROLLER_PREAMBLE}\n"
        f"Sensor readings ({len(top_k)} sensors):\n"
        f"{payload_block}\n\n"
        f"Based on the current sensor readings, what action should be taken?\n\n"
        f"{_JSON_INSTRUCTION}"
    )


def build_prompt_fused(
    fused_state: dict[str, dict[str, Any]], top_k: list[str],
) -> str:
    """Conditions C/D/F: query + fused state with b/d/u annotations."""
    lines = []
    for sid in top_k:
        state = fused_state[sid]
        lines.append(
            f"- {sid}: {state['reading']:.2f} {state['unit']} "
            f"| b={state['belief']:.2f} d={state['disbelief']:.2f} u={state['uncertainty']:.2f}"
        )
    facts_block = "\n".join(lines) if lines else "  (no active facts)"

    return (
        f"{_CONTROLLER_PREAMBLE}\n"
        f"Current state ({len(top_k)} most relevant sensors; "
        f"b=belief, d=disbelief, u=uncertainty):\n"
        f"{facts_block}\n\n"
        f"Based on this fused state, what action should be taken?\n\n"
        f"{_JSON_INSTRUCTION}"
    )


def build_prompt_vacuous(
    fused_state: dict[str, dict[str, Any]], top_k: list[str],
) -> str:
    """Condition E1: all uncertainty=1.0."""
    lines = []
    for sid in top_k:
        state = fused_state[sid]
        lines.append(
            f"- {sid}: {state['reading']:.2f} {state['unit']} "
            f"| b=0.00 d=0.00 u=1.00"
        )
    facts_block = "\n".join(lines) if lines else "  (no active facts)"

    return (
        f"{_CONTROLLER_PREAMBLE}\n"
        f"Current state ({len(top_k)} sensors, all with vacuous uncertainty):\n"
        f"{facts_block}\n\n"
        f"Based on this state, what action should be taken?\n\n"
        f"{_JSON_INSTRUCTION}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# HTTP Inference (reuses format from exp 02)
# ═══════════════════════════════════════════════════════════════════════════

_MODEL_ID: str | None = None
_HTTP_TIMEOUT: int = 120
_MAX_TOKENS: int = 4096


def _count_reasoning_tokens(content: str, reasoning_field: str) -> int:
    total_chars = 0
    if reasoning_field:
        total_chars += len(reasoning_field)
    for pattern in (r"<think>(.*?)</think>", r"<reasoning>(.*?)</reasoning>",
                    r"<thought>(.*?)</thought>"):
        for match in re.finditer(pattern, content, flags=re.DOTALL):
            total_chars += len(match.group(1))
    if total_chars == 0:
        return 0
    return int(total_chars / 4)


def _extract_json_response(raw: str) -> str:
    if not raw:
        return raw
    text = raw
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()

    fence_match = re.search(r"```(?:json|JSON)?\s*\n?(.*?)```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    brace_match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
        if '"' in candidate and ":" in candidate:
            return candidate

    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                if '"' in candidate and ":" in candidate:
                    return candidate
    return text


def infer_http(
    base_url: str, prompt: str, max_tokens: int | None = None,
) -> tuple[str, float, int, int, int]:
    if max_tokens is None:
        max_tokens = _MAX_TOKENS
    payload_dict: dict = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "top_p": 0.85,
        "top_k": 20,
    }
    if _MODEL_ID is not None:
        payload_dict["model"] = _MODEL_ID
    payload = json.dumps(payload_dict).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = (time.perf_counter() - t0) * 1000

    message = data["choices"][0]["message"]
    raw_content = message.get("content", "").strip() if message.get("content") else ""
    reasoning_field = message.get("reasoning", "") or ""

    reasoning_tokens = _count_reasoning_tokens(raw_content, reasoning_field)
    completion_tokens = data.get("usage", {}).get(
        "completion_tokens", len(raw_content.split()) if raw_content else 0
    )
    prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
    text = _extract_json_response(raw_content)
    return text, elapsed, completion_tokens, prompt_tokens, reasoning_tokens


# ═══════════════════════════════════════════════════════════════════════════
# Action Classification
# ═══════════════════════════════════════════════════════════════════════════

def classify_action(action_str: str | None) -> str:
    """Classify an LLM-proposed action as 'safety', 'monitor', or 'unclear'."""
    if action_str is None:
        return "unclear"
    a = action_str.strip().lower().replace(" ", "_").replace("-", "_")
    if a in SAFETY_ACTIONS:
        return "safety"
    if a in MONITOR_ACTIONS:
        return "monitor"
    # Try partial matching for compound actions
    for safety in SAFETY_ACTIONS:
        if safety in a:
            return "safety"
    for monitor in MONITOR_ACTIONS:
        if monitor in a:
            return "monitor"
    return "unclear"


def parse_llm_output(raw: str) -> dict | None:
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "action" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Trial Runner
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrialResult:
    window_idx: int
    window_label: int       # 0=normal, 1=attack
    event_id: int | None
    position: str           # onset/recovery/normal_baseline
    condition: str
    rep: int
    prompt: str
    raw_output: str
    json_valid: bool
    action_str: str | None
    action_class: str       # safety/monitor/unclear
    max_uncertainty: float
    threshold_permitted: bool    # epistemic check
    whitelist_permitted: bool    # behavioral check (action in known vocab)
    combined_permitted: bool
    detection_tp: bool      # attack window + safety action
    detection_tn: bool      # normal window + monitor action
    detection_fp: bool      # normal window + safety action (false alarm)
    detection_fn: bool      # attack window + monitor action (missed attack)
    llm_latency_ms: float
    completion_tokens: int
    prompt_tokens: int
    reasoning_tokens: int


def evaluate_trial(
    window: dict[str, Any],
    condition: str,
    rep: int,
    fused_state_full: dict[str, dict[str, Any]],
    fused_state_nodecay: dict[str, dict[str, Any]],
    fused_state_nofusion: dict[str, dict[str, Any]],
    top_k: list[str],
    base_url: str,
) -> TrialResult:
    """Run a single (window, condition, rep) trial."""
    # Select appropriate fused state per condition
    if condition == "D":  # no decay
        state = fused_state_nodecay
    elif condition in ("E2",):  # no fusion (passthrough)
        state = fused_state_nofusion
    else:
        state = fused_state_full

    # Build prompt per condition
    if condition == "A":
        prompt = build_prompt_A_bare(window)
    elif condition in ("B", "G"):
        prompt = build_prompt_B_raw(state, top_k)
    elif condition == "E1":
        prompt = build_prompt_vacuous(state, top_k)
    else:
        prompt = build_prompt_fused(state, top_k)

    # Inference
    raw, latency, tokens, ptokens, rtokens = infer_http(base_url, prompt)
    parsed = parse_llm_output(raw)
    json_valid = parsed is not None
    action_str = parsed.get("action") if json_valid else None
    action_class = classify_action(action_str)

    # Guardrail evaluation
    if state:
        max_u = max(s["uncertainty"] for s in state.values())
    else:
        max_u = 1.0

    # Conditions E1, G use vacuous uncertainty — threshold cannot discriminate
    if condition in ("E1", "G"):
        threshold_permitted = False  # max_u = 1.0 always > threshold
        max_u_for_guard = 1.0
    else:
        threshold_permitted = max_u < UNCERTAINTY_THRESHOLD
        max_u_for_guard = max_u

    # Whitelist = action must be in known vocabulary
    whitelist_permitted = action_class in ("safety", "monitor")

    # Conditions without guardrails (A, B, F) always permit
    guardrails_active = condition in ("C", "D", "E1", "E2", "G")
    if guardrails_active:
        combined_permitted = threshold_permitted and whitelist_permitted
    else:
        combined_permitted = True
        threshold_permitted = True
        whitelist_permitted = True

    # Detection metrics
    is_attack = window["label"] == 1

    # Detection = combined_permitted decision × action_class
    # Attack should be BLOCKED (combined_permitted=False) AND/OR action=safety
    # For simplicity: count safety-action on attack as TP, monitor-action on normal as TN
    detection_tp = is_attack and action_class == "safety"
    detection_tn = (not is_attack) and action_class == "monitor"
    detection_fp = (not is_attack) and action_class == "safety"
    detection_fn = is_attack and action_class == "monitor"

    return TrialResult(
        window_idx=window["idx"],
        window_label=window["label"],
        event_id=window.get("event_id"),
        position=window["position"],
        condition=condition,
        rep=rep,
        prompt=prompt,
        raw_output=raw,
        json_valid=json_valid,
        action_str=action_str,
        action_class=action_class,
        max_uncertainty=max_u_for_guard,
        threshold_permitted=threshold_permitted,
        whitelist_permitted=whitelist_permitted,
        combined_permitted=combined_permitted,
        detection_tp=detection_tp,
        detection_tn=detection_tn,
        detection_fp=detection_fp,
        detection_fn=detection_fn,
        llm_latency_ms=latency,
        completion_tokens=tokens,
        prompt_tokens=ptokens,
        reasoning_tokens=rtokens,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Metrics (cluster-robust bootstrap at event level)
# ═══════════════════════════════════════════════════════════════════════════

def compute_detection_metrics(trials: list[TrialResult]) -> dict:
    """Compute precision, recall, F1 from trial outcomes."""
    tp = sum(1 for t in trials if t.detection_tp)
    fp = sum(1 for t in trials if t.detection_fp)
    fn = sum(1 for t in trials if t.detection_fn)
    tn = sum(1 for t in trials if t.detection_tn)
    unclear = sum(1 for t in trials
                  if t.action_class == "unclear")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(trials) if trials else 0.0

    return {
        "n": len(trials),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "unclear": unclear,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "json_compliance": round(
            sum(1 for t in trials if t.json_valid) / len(trials), 4
        ) if trials else 0.0,
    }


def cluster_bootstrap_f1(
    trials: list[TrialResult],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Cluster-robust bootstrap CI for F1, clustered by event_id.

    Normal baselines form one cluster; each attack event is its own cluster.
    """
    rng = np.random.RandomState(seed)

    # Group by cluster
    clusters: dict[Any, list[TrialResult]] = {}
    for t in trials:
        key = t.event_id if t.event_id is not None else "normal"
        clusters.setdefault(key, []).append(t)
    cluster_keys = list(clusters.keys())

    if len(cluster_keys) < 2:
        m = compute_detection_metrics(trials)
        return m["f1"], float("nan"), float("nan")

    point_f1 = compute_detection_metrics(trials)["f1"]
    boot_f1s = []
    for _ in range(n_bootstrap):
        sampled_keys = [cluster_keys[i] for i in rng.randint(0, len(cluster_keys),
                                                              size=len(cluster_keys))]
        sampled_trials = []
        for k in sampled_keys:
            sampled_trials.extend(clusters[k])
        m = compute_detection_metrics(sampled_trials)
        boot_f1s.append(m["f1"])

    lower = float(np.percentile(boot_f1s, 2.5))
    upper = float(np.percentile(boot_f1s, 97.5))
    return point_f1, lower, upper


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 04: BATADAL × LLM × Conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--calibration-data", required=True)
    parser.add_argument("--evaluation-data", required=True)
    parser.add_argument("--output-dir", default="experiments/results/batadal_llm")
    parser.add_argument("--model-name", required=True,
                        help="Display name for results labeling")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--model-id", type=str, default=None,
                        help="Ollama model id; triggers Ollama mode")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS,
                        choices=ALL_CONDITIONS)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for window sampling")
    parser.add_argument("--strategy", default=UNCERTAINTY_STRATEGY,
                        choices=["historical", "physics", "composite"])
    args = parser.parse_args()

    # Server setup
    is_ollama = args.model_id is not None
    if args.base_url:
        base_url = args.base_url
    elif is_ollama:
        port = args.port or 11434
        base_url = f"http://localhost:{port}"
    else:
        port = args.port or 8080
        base_url = f"http://localhost:{port}"

    global _MODEL_ID, _HTTP_TIMEOUT, _MAX_TOKENS
    if is_ollama:
        _MODEL_ID = args.model_id
    _HTTP_TIMEOUT = args.timeout
    _MAX_TOKENS = args.max_tokens

    # Verify server
    try:
        if is_ollama:
            urllib.request.urlopen(base_url.rstrip("/"), timeout=5)
        else:
            urllib.request.urlopen(f"{base_url}/health", timeout=5)
    except Exception:
        print(f"ERROR: Cannot reach LLM server at {base_url}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"  EXPERIMENT 04: BATADAL × LLM — {args.model_name}")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading data...")
    cal_headers, cal_rows = load_csv(args.calibration_data)
    sensor_cols = detect_sensor_columns(cal_headers)
    calibration_stats = compute_calibration_stats(cal_rows, sensor_cols)

    eval_headers, eval_rows = load_csv(args.evaluation_data)
    events = identify_attack_events(eval_rows)
    print(f"  Calibration: {len(cal_rows)} rows, {len(sensor_cols)} sensors")
    print(f"  Evaluation: {len(eval_rows)} rows, {len(events)} attack events")

    # Build related map
    related_map: dict[str, list[str]] = {}
    for group in RELATED_GROUPS:
        valid = [s for s in group if s in sensor_cols]
        for s in valid:
            related_map[s] = [x for x in valid if x != s]

    # Sample windows
    print("\n[2/5] Sampling windows...")
    windows = sample_windows(
        eval_rows, events,
        seed=args.seed,
        n_attack_per_event=N_ATTACK_WINDOWS // len(events) if events else 10,
        n_normal=N_NORMAL_WINDOWS,
        boundary_margin=BOUNDARY_MARGIN_HOURS,
        normal_buffer=NORMAL_BUFFER_HOURS,
    )
    n_attack = sum(1 for w in windows if w["label"] == 1)
    n_normal = sum(1 for w in windows if w["label"] == 0)
    print(f"  Sampled {len(windows)} windows: {n_attack} attack, {n_normal} normal (seed={args.seed})")

    # Select strategy
    strategies = {
        "historical": HistoricalDeviationStrategy(),
        "physics": PhysicsBoundsStrategy(),
        "composite": CompositeStrategy(strategies=[
            (HistoricalDeviationStrategy(), 0.5),
            (PhysicsBoundsStrategy(), 0.3),
            (SensorAgreementStrategy(), 0.2),
        ]),
    }
    strategy = strategies[args.strategy]

    # Precompute fused states per window (3 variants for conditions)
    print("\n[3/5] Precomputing fused states...")
    window_states_full = {}
    window_states_nodecay = {}
    window_states_nofusion = {}
    window_top_k = {}

    for i, window in enumerate(windows):
        t_idx = window["idx"]
        # Full: fusion + decay
        full = build_fused_state_at_window(
            eval_rows, t_idx, WINDOW_LOOKBACK_HOURS, sensor_cols,
            calibration_stats, related_map, strategy,
            use_fusion=True, use_decay=True, decay_lambda=DECAY_LAMBDA,
        )
        # No decay: fusion only
        nodecay = build_fused_state_at_window(
            eval_rows, t_idx, WINDOW_LOOKBACK_HOURS, sensor_cols,
            calibration_stats, related_map, strategy,
            use_fusion=True, use_decay=False, decay_lambda=DECAY_LAMBDA,
        )
        # No fusion: decay only, per-sensor opinions
        nofusion = build_fused_state_at_window(
            eval_rows, t_idx, WINDOW_LOOKBACK_HOURS, sensor_cols,
            calibration_stats, related_map, strategy,
            use_fusion=False, use_decay=True, decay_lambda=DECAY_LAMBDA,
        )
        window_states_full[t_idx] = full
        window_states_nodecay[t_idx] = nodecay
        window_states_nofusion[t_idx] = nofusion
        window_top_k[t_idx] = select_top_k_sensors(full, TOP_K_PER_CATEGORY)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(windows)}]")

    # Run trials
    print("\n[4/5] Running LLM trials...")
    all_trials: list[TrialResult] = []
    total_trials = len(windows) * len(args.conditions) * args.reps
    trial_num = 0
    t_start = time.perf_counter()

    for condition in args.conditions:
        print(f"\n~~ Condition {condition} ({CONDITION_LABELS[condition]}) ~~")
        for window in windows:
            for rep in range(args.reps):
                trial_num += 1
                try:
                    result = evaluate_trial(
                        window, condition, rep,
                        window_states_full[window["idx"]],
                        window_states_nodecay[window["idx"]],
                        window_states_nofusion[window["idx"]],
                        window_top_k[window["idx"]],
                        base_url,
                    )
                    all_trials.append(result)
                    elapsed = time.perf_counter() - t_start
                    rate = trial_num / elapsed if elapsed > 0 else 0
                    eta_sec = (total_trials - trial_num) / rate if rate > 0 else 0
                    print(f"  [{trial_num}/{total_trials}] "
                          f"{condition}/w{window['idx']}/r{rep+1} "
                          f"action={result.action_str!r} class={result.action_class} "
                          f"label={'A' if window['label']==1 else 'N'} "
                          f"({result.llm_latency_ms:.0f}ms, ETA {eta_sec/60:.1f}min)")
                except Exception as e:
                    print(f"  [{trial_num}/{total_trials}] FAILED: {e}")

    # Aggregate metrics
    print("\n[5/5] Computing metrics...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-condition summaries
    condition_summaries = {}
    for condition in args.conditions:
        cond_trials = [t for t in all_trials if t.condition == condition]
        metrics = compute_detection_metrics(cond_trials)
        f1, f1_lo, f1_hi = cluster_bootstrap_f1(cond_trials, n_bootstrap=1000,
                                                  seed=args.seed)
        metrics["f1_ci_lower"] = round(f1_lo, 4)
        metrics["f1_ci_upper"] = round(f1_hi, 4)
        metrics["avg_llm_latency_ms"] = round(
            np.mean([t.llm_latency_ms for t in cond_trials]), 1
        ) if cond_trials else 0
        metrics["avg_completion_tokens"] = round(
            np.mean([t.completion_tokens for t in cond_trials]), 1
        ) if cond_trials else 0
        metrics["avg_reasoning_tokens"] = round(
            np.mean([t.reasoning_tokens for t in cond_trials]), 1
        ) if cond_trials else 0
        metrics["reasoning_ratio"] = round(
            sum(t.reasoning_tokens for t in cond_trials) /
            max(1, sum(t.completion_tokens for t in cond_trials)), 4
        ) if cond_trials else 0
        condition_summaries[condition] = metrics

    # Save
    summary = {
        "experiment": "04_batadal_llm",
        "timestamp": timestamp,
        "model_name": args.model_name,
        "model_id": args.model_id,
        "strategy": args.strategy,
        "seed": args.seed,
        "n_windows": len(windows),
        "n_attack_windows": n_attack,
        "n_normal_windows": n_normal,
        "n_reps": args.reps,
        "n_conditions": len(args.conditions),
        "total_trials": len(all_trials),
        "pre_registered_config": {
            "WINDOW_LOOKBACK_HOURS": WINDOW_LOOKBACK_HOURS,
            "BOUNDARY_MARGIN_HOURS": BOUNDARY_MARGIN_HOURS,
            "NORMAL_BUFFER_HOURS": NORMAL_BUFFER_HOURS,
            "TOP_K_PER_CATEGORY": TOP_K_PER_CATEGORY,
            "UNCERTAINTY_THRESHOLD": UNCERTAINTY_THRESHOLD,
            "DECAY_LAMBDA": DECAY_LAMBDA,
            "PHYSICS_BOUNDS_MODE": PHYSICS_BOUNDS_MODE,
        },
        "windows": windows,
        "condition_summaries": condition_summaries,
    }

    summary_path = output_dir / f"04_batadal_llm_{args.model_name}_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary: {summary_path}")

    # Trial-level CSV
    trials_csv = output_dir / f"04_batadal_llm_{args.model_name}_{timestamp}_trials.csv"
    with open(trials_csv, "w", newline="", encoding="utf-8") as f:
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
        for t in all_trials:
            writer.writerow([
                t.window_idx, t.window_label, t.event_id, t.position,
                t.condition, t.rep,
                int(t.json_valid), t.action_str or "", t.action_class,
                f"{t.max_uncertainty:.6f}",
                int(t.threshold_permitted), int(t.whitelist_permitted),
                int(t.combined_permitted),
                int(t.detection_tp), int(t.detection_tn),
                int(t.detection_fp), int(t.detection_fn),
                f"{t.llm_latency_ms:.1f}", t.completion_tokens,
                t.prompt_tokens, t.reasoning_tokens,
            ])
    print(f"  Trials: {trials_csv}")

    # Print summary
    print("\n" + "=" * 80)
    print(f"  RESULTS SUMMARY — {args.model_name}")
    print("=" * 80)
    print(f"{'Cond':<5} {'Label':<22} {'JSON':<6} {'Prec':<6} {'Rec':<6} "
          f"{'F1':<6} {'F1 95% CI':<18} {'Lat(ms)':<8} {'R/Tok':<8}")
    print("-" * 95)
    for condition in args.conditions:
        m = condition_summaries[condition]
        ci_str = f"[{m['f1_ci_lower']:.3f},{m['f1_ci_upper']:.3f}]"
        print(f"{condition:<5} {CONDITION_LABELS[condition]:<22} "
              f"{m['json_compliance']:<6.2f} {m['precision']:<6.2f} "
              f"{m['recall']:<6.2f} {m['f1']:<6.2f} {ci_str:<18} "
              f"{m['avg_llm_latency_ms']:<8.0f} {m['reasoning_ratio']:<8.3f}")

    # Paired Wilcoxon: full pipeline (C) vs others
    if _HAS_SCIPY and "C" in args.conditions and len(args.conditions) > 1:
        print("\n  Paired Wilcoxon tests vs Condition C (Bonferroni-corrected):")
        # Group trials by window_idx for pairing
        c_trials = {(t.window_idx, t.rep): t for t in all_trials if t.condition == "C"}

        for other_cond in args.conditions:
            if other_cond == "C":
                continue
            o_trials = {(t.window_idx, t.rep): t for t in all_trials if t.condition == other_cond}
            common_keys = set(c_trials.keys()) & set(o_trials.keys())
            if len(common_keys) < 10:
                continue
            # Binary outcome: correctly classified (TP or TN)
            c_correct = np.array([
                int(c_trials[k].detection_tp or c_trials[k].detection_tn)
                for k in common_keys
            ])
            o_correct = np.array([
                int(o_trials[k].detection_tp or o_trials[k].detection_tn)
                for k in common_keys
            ])
            if np.array_equal(c_correct, o_correct):
                print(f"    C vs {other_cond}: identical, test skipped")
                continue
            try:
                stat, p = sp_stats.wilcoxon(c_correct, o_correct, zero_method="wilcox")
                p_corr = min(p * N_COMPARISONS, 1.0)
                delta = float(np.mean(c_correct - o_correct))
                print(f"    C vs {other_cond} (n={len(common_keys)}): "
                      f"Δaccuracy={delta:+.4f} p={p_corr:.4f} "
                      f"({'significant' if p_corr < 0.05 else 'n.s.'})")
            except ValueError as e:
                print(f"    C vs {other_cond}: test failed ({e})")

    print(f"\n  All results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
