"""
Experiment 01: Bonsai Baseline — raw inference quality and speed.

Tests all three Bonsai model sizes (1.7B, 4B, 8B) on structured JSON
output tasks to establish baselines for:
  - Token generation speed (tok/s)
  - JSON compliance rate (does the output parse as valid JSON?)
  - Instruction following accuracy (are action/target fields present?)
  - Hallucination tendency (does it invent fields not requested?)

This experiment does NOT use the epistemic-edge pipeline. It measures
the raw LLM as a standalone component before we add trust/memory layers.

Requires:
  - Bonsai GGUF models (run 00_download_models.py first)
  - PrismML's llama.cpp fork built with CUDA (see README)
    OR llama-cpp-python built from PrismML's fork

Usage:
    python 01_bonsai_baseline.py                    # Test all sizes
    python 01_bonsai_baseline.py --size 1.7B        # Specific size
    python 01_bonsai_baseline.py --use-cli           # Use llama.cpp CLI
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

BONSAI_FILES = {
    "1.7B": "Bonsai-1.7B-Q1_0_g128.gguf",
    "4B": "Bonsai-4B-Q1_0_g128.gguf",
    "8B": "Bonsai-8B-Q1_0_g128.gguf",
}

# ── Test prompts ────────────────────────────────────────────────────────────

STRUCTURED_PROMPTS = [
    {
        "id": "simple_action",
        "prompt": (
            "You are an AIoT edge controller. A temperature sensor reads 85°C, "
            "which is above the 80°C safety threshold.\n"
            "Respond ONLY with a JSON object: {\"action\": \"...\", \"target\": \"...\"}."
        ),
        "expected_action_contains": ["shut", "cool", "stop", "alert", "close"],
    },
    {
        "id": "multi_sensor_conflict",
        "prompt": (
            "You are an AIoT controller. Sensor A reports pressure=NORMAL (belief=0.9). "
            "Sensor B reports pressure=CRITICAL (belief=0.7). "
            "Given this conflict, what action should be taken?\n"
            "Respond ONLY with a JSON object: {\"action\": \"...\", \"target\": \"...\", "
            "\"parameters\": {}}."
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
            "Respond ONLY with a JSON object: {\"action\": \"...\", \"target\": \"...\"}."
        ),
        "expected_action_contains": ["inspect", "verify", "check", "recheck", "alert", "close"],
    },
    {
        "id": "free_form_intent",
        "prompt": (
            "You are an industrial robot controller. An operator says: "
            "'Stop the conveyor belt on line 3 immediately.'\n"
            "Translate this into a structured command.\n"
            "Respond ONLY with a JSON object: {\"action\": \"...\", \"target\": \"...\"}."
        ),
        "expected_action_contains": ["stop", "halt", "pause", "shutdown"],
    },
    {
        "id": "no_action_needed",
        "prompt": (
            "You are an AIoT controller. All sensors report nominal.\n"
            "Temperature: 25°C (normal). Pressure: 101kPa (normal). Vibration: 0.1g (normal).\n"
            "Respond ONLY with a JSON object: {\"action\": \"...\", \"target\": \"...\"}."
        ),
        "expected_action_contains": ["noop", "none", "monitor", "log", "no_action", "continue"],
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
    action_relevant: bool
    latency_ms: float
    tokens_generated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "model_size": self.model_size,
            "raw_output": self.raw_output,
            "json_valid": self.json_valid,
            "has_action": self.has_action,
            "has_target": self.has_target,
            "action_relevant": self.action_relevant,
            "latency_ms": round(self.latency_ms, 1),
            "tokens_generated": self.tokens_generated,
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

    def summary(self) -> dict[str, Any]:
        return {
            "model_size": self.model_size,
            "num_trials": len(self.trials),
            "json_compliance_rate": round(self.json_compliance_rate, 3),
            "field_compliance_rate": round(self.field_compliance_rate, 3),
            "relevance_rate": round(self.relevance_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


# ── Inference backends ──────────────────────────────────────────────────────


def infer_python(model_path: str, prompt: str, max_tokens: int = 128) -> tuple[str, float]:
    """Run inference via llama-cpp-python."""
    from llama_cpp import Llama

    llm = Llama(model_path=model_path, n_ctx=2048, verbose=False, n_gpu_layers=99)
    t0 = time.perf_counter()
    result = llm(prompt, max_tokens=max_tokens, temperature=0.3, top_p=0.85)
    elapsed = (time.perf_counter() - t0) * 1000
    text = result["choices"][0]["text"].strip()
    return text, elapsed


def infer_cli(model_path: str, prompt: str, max_tokens: int = 128) -> tuple[str, float]:
    """Run inference via PrismML's llama.cpp CLI (subprocess)."""
    t0 = time.perf_counter()
    result = subprocess.run(
        [
            "llama-cli",
            "-m", model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", "0.3",
            "--top-p", "0.85",
            "--top-k", "20",
            "-ngl", "99",
            "--no-display-prompt",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return result.stdout.strip(), elapsed


# ── Evaluation ──────────────────────────────────────────────────────────────


def evaluate_output(
    prompt_id: str,
    model_size: str,
    raw: str,
    expected_actions: list[str],
    latency_ms: float,
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

    action_relevant = False
    if has_action and parsed is not None:
        action_lower = str(parsed["action"]).lower()
        action_relevant = any(kw in action_lower for kw in expected_actions)

    tokens_generated = len(raw.split())

    return TrialResult(
        prompt_id=prompt_id,
        model_size=model_size,
        raw_output=raw,
        parsed_json=parsed,
        json_valid=json_valid,
        has_action=has_action,
        has_target=has_target,
        action_relevant=action_relevant,
        latency_ms=latency_ms,
        tokens_generated=tokens_generated,
    )


# ── Main ────────────────────────────────────────────────────────────────────


def find_model_path(size: str) -> str | None:
    """Locate the GGUF file for a given model size."""
    filename = BONSAI_FILES.get(size)
    if not filename:
        return None
    path = MODELS_DIR / f"bonsai-{size.lower()}" / filename
    if path.exists():
        return str(path)
    # Check if downloaded via huggingface cache (may nest differently)
    alt = MODELS_DIR / f"bonsai-{size.lower()}"
    if alt.exists():
        for f in alt.rglob("*.gguf"):
            return str(f)
    return None


def run_experiment(
    sizes: list[str],
    use_cli: bool = False,
    repetitions: int = 3,
) -> dict[str, Any]:
    """Run the full baseline experiment across model sizes."""
    all_results: dict[str, ModelResults] = {}

    for size in sizes:
        model_path = find_model_path(size)
        if model_path is None:
            print(f"[SKIP] Bonsai {size}: model not found. Run 00_download_models.py first.")
            continue

        print(f"\n{'='*60}")
        print(f"  Testing Bonsai {size}: {model_path}")
        print(f"{'='*60}")

        model_results = ModelResults(model_size=size)
        infer_fn = infer_cli if use_cli else infer_python

        for prompt_def in STRUCTURED_PROMPTS:
            for rep in range(repetitions):
                print(f"  [{prompt_def['id']}] rep {rep+1}/{repetitions}...", end=" ")
                try:
                    raw, latency = infer_fn(model_path, prompt_def["prompt"])
                    trial = evaluate_output(
                        prompt_id=prompt_def["id"],
                        model_size=size,
                        raw=raw,
                        expected_actions=prompt_def["expected_action_contains"],
                        latency_ms=latency,
                    )
                    model_results.trials.append(trial)
                    status = "✓ JSON" if trial.json_valid else "✗ FAIL"
                    print(f"{status} ({latency:.0f}ms)")
                except Exception as e:
                    print(f"ERROR: {e}")

        all_results[size] = model_results

    return {
        "experiment": "01_bonsai_baseline",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "backend": "cli" if use_cli else "python",
        "repetitions": repetitions,
        "summaries": {k: v.summary() for k, v in all_results.items()},
        "trials": {
            k: [t.to_dict() for t in v.trials] for k, v in all_results.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bonsai Baseline Experiment")
    parser.add_argument("--size", choices=["1.7B", "4B", "8B"], default=None)
    parser.add_argument("--use-cli", action="store_true", help="Use llama.cpp CLI")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per prompt")
    args = parser.parse_args()

    sizes = [args.size] if args.size else ["1.7B", "4B", "8B"]

    results = run_experiment(sizes, use_cli=args.use_cli, repetitions=args.reps)

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for size, summary in results["summaries"].items():
        print(f"\n  Bonsai {size}:")
        print(f"    JSON compliance:  {summary['json_compliance_rate']*100:.1f}%")
        print(f"    Field compliance: {summary['field_compliance_rate']*100:.1f}%")
        print(f"    Action relevance: {summary['relevance_rate']*100:.1f}%")
        print(f"    Avg latency:      {summary['avg_latency_ms']:.0f}ms")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"01_bonsai_baseline_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
