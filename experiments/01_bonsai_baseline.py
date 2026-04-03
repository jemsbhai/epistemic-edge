"""
Experiment 01: Bonsai Baseline — raw inference quality and speed.

Tests Bonsai model sizes on structured JSON output tasks to establish
baselines for JSON compliance, instruction following, and latency.

This experiment does NOT use the epistemic-edge pipeline. It measures
the raw LLM as a standalone component before we add trust/memory layers.

Requires:
  - PrismML llama-server running: llama-server.exe -m <model>.gguf -ngl 99 --port 8080

Usage:
    python 01_bonsai_baseline.py --size 1.7B --reps 10
    python 01_bonsai_baseline.py --size 8B --port 8080 --reps 10
    python 01_bonsai_baseline.py --size 4B --reps 20
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── Test prompts ────────────────────────────────────────────────────────────

STRUCTURED_PROMPTS = [
    {
        "id": "simple_action",
        "prompt": (
            "You are an AIoT edge controller. A temperature sensor reads 85°C, "
            "which is above the 80°C safety threshold.\n"
            'Respond ONLY with a JSON object: {"action": "...", "target": "..."}.'
        ),
        "expected_action_contains": ["shut", "cool", "stop", "alert", "close"],
    },
    {
        "id": "multi_sensor_conflict",
        "prompt": (
            "You are an AIoT controller. Sensor A reports pressure=NORMAL (belief=0.9). "
            "Sensor B reports pressure=CRITICAL (belief=0.7). "
            "Given this conflict, what action should be taken?\n"
            'Respond ONLY with a JSON object: {"action": "...", "target": "...", '
            '"parameters": {}}.'
        ),
        "expected_action_contains": ["inspect", "check", "alert", "shut", "verify", "close"],
    },
    {
        "id": "stale_data_awareness",
        "prompt": (
            "You are an AIoT controller for node 'pipeline_west'.\n"
            "Active state (2 facts):\n"
            "- [b=0.90 d=0.05 u=0.05] {'valve_status': 'open'}\n"
            "- [b=0.30 d=0.10 u=0.60] {'leak_detected': True}\n"
            "The leak reading has HIGH uncertainty (u=0.60). What should you do?\n"
            'Respond ONLY with a JSON object: {"action": "...", "target": "..."}.'
        ),
        "expected_action_contains": ["inspect", "verify", "check", "recheck", "alert", "close"],
    },
    {
        "id": "free_form_intent",
        "prompt": (
            "You are an industrial robot controller. An operator says: "
            "'Stop the conveyor belt on line 3 immediately.'\n"
            "Translate this into a structured command.\n"
            'Respond ONLY with a JSON object: {"action": "...", "target": "..."}.'
        ),
        "expected_action_contains": ["stop", "halt", "pause", "shutdown"],
    },
    {
        "id": "no_action_needed",
        "prompt": (
            "You are an AIoT controller. All sensors report nominal.\n"
            "Temperature: 25°C (normal). Pressure: 101kPa (normal). Vibration: 0.1g (normal).\n"
            'Respond ONLY with a JSON object: {"action": "...", "target": "..."}.'
        ),
        "expected_action_contains": ["noop", "none", "monitor", "log", "no_action", "continue"],
    },
    {
        "id": "multi_step_reasoning",
        "prompt": (
            "You are an AIoT controller. Valve A is OPEN. Pump B is OFF. "
            "Pressure is rising at 5kPa/min. The safety limit is 200kPa. "
            "Current pressure is 185kPa.\n"
            'Respond ONLY with a JSON object: {"action": "...", "target": "...", '
            '"parameters": {}}.'
        ),
        "expected_action_contains": ["close", "shut", "stop", "reduce", "open", "activate"],
    },
    {
        "id": "ambiguous_reading",
        "prompt": (
            "You are an AIoT controller. A sensor reading says 'ERR_OVERFLOW'. "
            "You don't know if this is a real overflow or a sensor malfunction.\n"
            'Respond ONLY with a JSON object: {"action": "...", "target": "..."}.'
        ),
        "expected_action_contains": ["check", "inspect", "verify", "diagnose", "recheck", "alert"],
    },
    {
        "id": "json_with_nested_params",
        "prompt": (
            "You are an AIoT controller. Set motor_1 speed to 1500 RPM "
            "with a ramp time of 10 seconds.\n"
            'Respond ONLY with a JSON object: {"action": "...", "target": "...", '
            '"parameters": {"rpm": ..., "ramp_time_s": ...}}.'
        ),
        "expected_action_contains": ["set", "adjust", "configure", "change", "update"],
    },
]


@dataclass
class TrialResult:
    prompt_id: str
    model_size: str
    raw_output: str
    parsed_json: dict[str, Any] | None
    json_valid: bool
    has_action: bool
    has_target: bool
    has_parameters: bool
    action_relevant: bool
    latency_ms: float
    tokens_generated: int
    completion_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "model_size": self.model_size,
            "raw_output": self.raw_output,
            "json_valid": self.json_valid,
            "has_action": self.has_action,
            "has_target": self.has_target,
            "has_parameters": self.has_parameters,
            "action_relevant": self.action_relevant,
            "latency_ms": round(self.latency_ms, 1),
            "tokens_generated": self.tokens_generated,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class ModelResults:
    model_size: str
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def json_compliance_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.json_valid) / len(self.trials)

    @property
    def field_compliance_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.has_action and t.has_target) / len(self.trials)

    @property
    def relevance_rate(self) -> float:
        valid = [t for t in self.trials if t.json_valid]
        if not valid:
            return 0.0
        return sum(1 for t in valid if t.action_relevant) / len(valid)

    @property
    def avg_latency_ms(self) -> float:
        if not self.trials:
            return 0.0
        return sum(t.latency_ms for t in self.trials) / len(self.trials)

    @property
    def avg_tok_s(self) -> float:
        """Average tokens per second across trials."""
        rates = []
        for t in self.trials:
            if t.latency_ms > 0 and t.completion_tokens > 0:
                rates.append(t.completion_tokens / (t.latency_ms / 1000))
        return sum(rates) / len(rates) if rates else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "model_size": self.model_size,
            "num_trials": len(self.trials),
            "json_compliance_rate": round(self.json_compliance_rate, 3),
            "field_compliance_rate": round(self.field_compliance_rate, 3),
            "relevance_rate": round(self.relevance_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_tok_s": round(self.avg_tok_s, 1),
        }


# ── Inference via llama-server HTTP ─────────────────────────────────────────


def infer_http(
    base_url: str,
    prompt: str,
    max_tokens: int = 128,
) -> tuple[str, float, int]:
    """
    Send a chat completion request to llama-server.

    Returns (response_text, latency_ms, completion_tokens).
    """
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "top_p": 0.85,
        "top_k": 20,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = (time.perf_counter() - t0) * 1000

    text = data["choices"][0]["message"]["content"].strip()
    completion_tokens = data.get("usage", {}).get("completion_tokens", len(text.split()))

    return text, elapsed, completion_tokens


# ── Evaluation ──────────────────────────────────────────────────────────────


def evaluate_output(
    prompt_id: str,
    model_size: str,
    raw: str,
    expected_actions: list[str],
    latency_ms: float,
    completion_tokens: int,
) -> TrialResult:
    """Evaluate a single LLM output against expectations."""
    parsed = None
    json_valid = False
    try:
        parsed = json.loads(raw)
        json_valid = True
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end])
                json_valid = True
            except json.JSONDecodeError:
                pass

    has_action = parsed is not None and "action" in parsed
    has_target = parsed is not None and "target" in parsed
    has_parameters = parsed is not None and "parameters" in parsed

    action_relevant = False
    if has_action and parsed is not None:
        action_lower = str(parsed["action"]).lower()
        action_relevant = any(kw in action_lower for kw in expected_actions)

    return TrialResult(
        prompt_id=prompt_id,
        model_size=model_size,
        raw_output=raw,
        parsed_json=parsed,
        json_valid=json_valid,
        has_action=has_action,
        has_target=has_target,
        has_parameters=has_parameters,
        action_relevant=action_relevant,
        latency_ms=latency_ms,
        tokens_generated=len(raw.split()),
        completion_tokens=completion_tokens,
    )


# ── Main ────────────────────────────────────────────────────────────────────


def run_experiment(
    model_size: str,
    base_url: str,
    repetitions: int = 10,
) -> dict[str, Any]:
    """Run the full baseline experiment for a single model size."""
    model_results = ModelResults(model_size=model_size)

    print(f"\n{'='*60}")
    print(f"  Bonsai {model_size} Baseline")
    print(f"  Server: {base_url}")
    print(f"  Prompts: {len(STRUCTURED_PROMPTS)} × {repetitions} reps = "
          f"{len(STRUCTURED_PROMPTS) * repetitions} trials")
    print(f"{'='*60}")

    for prompt_def in STRUCTURED_PROMPTS:
        for rep in range(repetitions):
            print(f"  [{prompt_def['id']}] rep {rep+1}/{repetitions}...", end=" ", flush=True)
            try:
                raw, latency, comp_tokens = infer_http(base_url, prompt_def["prompt"])
                trial = evaluate_output(
                    prompt_id=prompt_def["id"],
                    model_size=model_size,
                    raw=raw,
                    expected_actions=prompt_def["expected_action_contains"],
                    latency_ms=latency,
                    completion_tokens=comp_tokens,
                )
                model_results.trials.append(trial)
                status = "✓ JSON" if trial.json_valid else "✗ FAIL"
                tok_s = comp_tokens / (latency / 1000) if latency > 0 else 0
                print(f"{status} ({latency:.0f}ms, {tok_s:.0f} tok/s)")
            except Exception as e:
                print(f"ERROR: {e}")

    return {
        "experiment": "01_bonsai_baseline",
        "model_size": model_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "server": base_url,
        "repetitions": repetitions,
        "num_prompts": len(STRUCTURED_PROMPTS),
        "total_trials": len(model_results.trials),
        "summary": model_results.summary(),
        "trials": [t.to_dict() for t in model_results.trials],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bonsai Baseline Experiment")
    parser.add_argument("--size", required=True, choices=["1.7B", "4B", "8B"],
                        help="Model size (must match the model loaded in llama-server)")
    parser.add_argument("--port", type=int, default=8080, help="llama-server port")
    parser.add_argument("--reps", type=int, default=10, help="Repetitions per prompt")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"

    # Verify server is reachable
    try:
        urllib.request.urlopen(f"{base_url}/health", timeout=5)
    except Exception:
        print(f"ERROR: Cannot reach llama-server at {base_url}")
        print(f"Start it first:")
        print(f"  llama-server.exe -m <model>.gguf --port {args.port} -ngl 99")
        return

    results = run_experiment(args.size, base_url, repetitions=args.reps)
    summary = results["summary"]

    print(f"\n{'='*60}")
    print(f"  RESULTS: Bonsai {args.size}")
    print(f"{'='*60}")
    print(f"    Trials:           {summary['num_trials']}")
    print(f"    JSON compliance:  {summary['json_compliance_rate']*100:.1f}%")
    print(f"    Field compliance: {summary['field_compliance_rate']*100:.1f}%")
    print(f"    Action relevance: {summary['relevance_rate']*100:.1f}%")
    print(f"    Avg latency:      {summary['avg_latency_ms']:.0f}ms")
    print(f"    Avg throughput:   {summary['avg_tok_s']:.1f} tok/s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"01_bonsai_{args.size}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
