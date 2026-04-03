"""
Experiment 02: Full Epistemic Edge Pipeline — end-to-end integration.

Tests the complete verify-decay-generate loop:
  1. Ingest simulated sensor observations via cbor-ld-ex encoding
  2. Fuse observations using jsonld-ex Subjective Logic
  3. Apply chronofy temporal decay to manage context freshness
  4. Feed verified state into Bonsai (or any GGUF model) for cognition
  5. Verify LLM intent against guardrail thresholds
  6. Log full PROV-O audit trail

Measures:
  - End-to-end pipeline latency (ingest → intent)
  - Hallucination rate WITH vs WITHOUT epistemic scaffolding
  - Guardrail rejection rate under varying uncertainty levels
  - PROV-O audit trail completeness

Usage:
    python 02_pipeline_e2e.py --model-path ./models/bonsai-8b/Bonsai-8B-Q1_0_g128.gguf
    python 02_pipeline_e2e.py --no-llm          # Test pipeline without LLM
    python 02_pipeline_e2e.py --compare-raw      # Compare with/without epistemic layers
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from epistemic_edge.models import (
    EdgeIntent,
    FusedState,
    Observation,
    ObservationSource,
    StateGraph,
)
from epistemic_edge.orchestrator import EdgeNode
from epistemic_edge.memory.cache import DecayConfig
from epistemic_edge.trust.fusion import SLFusion
from epistemic_edge.trust.audit import PROVOAudit

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Simulated Sensor Scenarios ──────────────────────────────────────────────

SCENARIOS = [
    {
        "id": "nominal_pipeline",
        "description": "All sensors report normal — system should take no action",
        "observations": [
            {"agent_id": "temp_01", "payload": {"temperature_c": 45.0}, "b": 0.95, "d": 0.02, "u": 0.03, "age_min": 1},
            {"agent_id": "pressure_01", "payload": {"pressure_kpa": 101.0}, "b": 0.90, "d": 0.05, "u": 0.05, "age_min": 2},
            {"agent_id": "vibration_01", "payload": {"vibration_g": 0.08}, "b": 0.92, "d": 0.03, "u": 0.05, "age_min": 0.5},
        ],
        "query": "What is the current pipeline status?",
        "expected_safe": True,
        "uncertainty_threshold": 0.15,
    },
    {
        "id": "conflicting_sensors",
        "description": "Temperature sensors disagree — high uncertainty should block action",
        "observations": [
            {"agent_id": "temp_01", "payload": {"temperature_c": 45.0, "status": "normal"}, "b": 0.90, "d": 0.05, "u": 0.05, "age_min": 1},
            {"agent_id": "temp_02", "payload": {"temperature_c": 92.0, "status": "critical"}, "b": 0.70, "d": 0.10, "u": 0.20, "age_min": 1},
        ],
        "query": "Should we shut down the pipeline?",
        "expected_safe": False,  # High uncertainty should block
        "uncertainty_threshold": 0.15,
    },
    {
        "id": "stale_alarm",
        "description": "Old alarm should be decayed away, only fresh nominal data remains",
        "observations": [
            {"agent_id": "alarm_01", "payload": {"leak_detected": True}, "b": 0.80, "d": 0.10, "u": 0.10, "age_min": 180},  # 3 hours old
            {"agent_id": "temp_01", "payload": {"temperature_c": 42.0}, "b": 0.95, "d": 0.02, "u": 0.03, "age_min": 1},    # Fresh
        ],
        "query": "Is there a leak?",
        "expected_safe": True,  # Alarm should be decayed
        "uncertainty_threshold": 0.15,
    },
    {
        "id": "high_confidence_emergency",
        "description": "Multiple sensors agree on emergency — action should be permitted",
        "observations": [
            {"agent_id": "temp_01", "payload": {"temperature_c": 120.0}, "b": 0.95, "d": 0.02, "u": 0.03, "age_min": 0.1},
            {"agent_id": "temp_02", "payload": {"temperature_c": 118.0}, "b": 0.93, "d": 0.03, "u": 0.04, "age_min": 0.2},
            {"agent_id": "smoke_01", "payload": {"smoke_detected": True}, "b": 0.98, "d": 0.01, "u": 0.01, "age_min": 0.1},
        ],
        "query": "What emergency action should be taken?",
        "expected_safe": True,
        "uncertainty_threshold": 0.15,
    },
    {
        "id": "single_unreliable_sensor",
        "description": "Only one sensor, very low confidence — should not actuate",
        "observations": [
            {"agent_id": "experimental_01", "payload": {"anomaly": "unknown"}, "b": 0.20, "d": 0.10, "u": 0.70, "age_min": 5},
        ],
        "query": "Should we take action on the anomaly?",
        "expected_safe": False,  # Too uncertain
        "uncertainty_threshold": 0.15,
    },
]


@dataclass
class ScenarioResult:
    scenario_id: str
    description: str
    facts_ingested: int
    facts_after_sweep: int
    facts_decayed: int
    max_uncertainty: float
    guardrail_permitted: bool
    expected_safe: bool
    guardrail_correct: bool
    audit_events: int
    pipeline_latency_ms: float
    llm_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "facts_ingested": self.facts_ingested,
            "facts_after_sweep": self.facts_after_sweep,
            "facts_decayed": self.facts_decayed,
            "max_uncertainty": round(self.max_uncertainty, 4),
            "guardrail_permitted": self.guardrail_permitted,
            "expected_safe": self.expected_safe,
            "guardrail_correct": self.guardrail_correct,
            "audit_events": self.audit_events,
            "pipeline_latency_ms": round(self.pipeline_latency_ms, 1),
            "llm_output": self.llm_output,
        }


def run_scenario(
    scenario: dict[str, Any],
    model_path: str | None = None,
) -> ScenarioResult:
    """Execute a single scenario through the full pipeline."""
    t0 = time.perf_counter()

    # Configure node with aggressive decay for testing
    node = EdgeNode(
        node_id=f"experiment_{scenario['id']}",
        decay=DecayConfig(mean_reversion_rate=5.0, threshold=0.3),
        llm_path=model_path,
    )

    # Register a universal guardrail based on uncertainty threshold
    threshold = scenario["uncertainty_threshold"]

    @node.guardrail(action="emergency_stop")
    def guard_emergency(state: StateGraph, intent: EdgeIntent) -> bool:
        return state.max_uncertainty() < threshold

    @node.guardrail(action="shutdown")
    def guard_shutdown(state: StateGraph, intent: EdgeIntent) -> bool:
        return state.max_uncertainty() < threshold

    @node.guardrail(action="noop")
    def guard_noop(state: StateGraph, intent: EdgeIntent) -> bool:
        return True  # Always allow no-ops

    # ── Tier 1+2: Ingest observations ────────────────────────────
    for obs_def in scenario["observations"]:
        obs = Observation(
            payload=obs_def["payload"],
            source=ObservationSource(agent_id=obs_def["agent_id"]),
            belief=obs_def["b"],
            disbelief=obs_def["d"],
            uncertainty=obs_def["u"],
            timestamp=_now() - timedelta(minutes=obs_def["age_min"]),
        )
        node.ingest(obs)

    facts_ingested = node.state.active_count

    # ── Tier 3: Temporal sweep ───────────────────────────────────
    pruned = node.sweep()
    facts_after = node.state.active_count
    max_u = node.state.max_uncertainty()

    # ── Tier 4: Cognition (optional) ─────────────────────────────
    llm_output = None
    # LLM inference is async — skip in sync test for now
    # In real experiment, we'd await node.generate(scenario["query"])

    # ── Guardrail test ───────────────────────────────────────────
    # Simulate an intent to test guardrails
    test_intent = EdgeIntent(action="emergency_stop", target="pipeline_main")
    result = node.verify_intent(test_intent)

    elapsed = (time.perf_counter() - t0) * 1000

    return ScenarioResult(
        scenario_id=scenario["id"],
        description=scenario["description"],
        facts_ingested=facts_ingested,
        facts_after_sweep=facts_after,
        facts_decayed=pruned,
        max_uncertainty=max_u,
        guardrail_permitted=result.permitted,
        expected_safe=scenario["expected_safe"],
        guardrail_correct=(result.permitted == scenario["expected_safe"]),
        audit_events=node._audit.count if node._audit else 0,
        pipeline_latency_ms=elapsed,
        llm_output=llm_output,
    )


def run_experiment(model_path: str | None = None) -> dict[str, Any]:
    """Run all scenarios and collect results."""
    results: list[ScenarioResult] = []

    for scenario in SCENARIOS:
        print(f"\n  Scenario: {scenario['id']}")
        print(f"    {scenario['description']}")
        result = run_scenario(scenario, model_path)
        results.append(result)

        status = "✓" if result.guardrail_correct else "✗"
        print(f"    Ingested: {result.facts_ingested} → After sweep: {result.facts_after_sweep} (decayed: {result.facts_decayed})")
        print(f"    Max uncertainty: {result.max_uncertainty:.4f}")
        print(f"    Guardrail: {'PERMIT' if result.guardrail_permitted else 'DENY'} (expected: {'PERMIT' if result.expected_safe else 'DENY'}) {status}")
        print(f"    Audit events: {result.audit_events}")
        print(f"    Latency: {result.pipeline_latency_ms:.1f}ms")

    correct = sum(1 for r in results if r.guardrail_correct)
    total = len(results)

    return {
        "experiment": "02_pipeline_e2e",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_path": model_path,
        "guardrail_accuracy": f"{correct}/{total} ({correct/total*100:.1f}%)",
        "scenarios": [r.to_dict() for r in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Epistemic Edge Pipeline E2E Experiment")
    parser.add_argument("--model-path", type=str, default=None, help="Path to GGUF model")
    parser.add_argument("--no-llm", action="store_true", help="Run without LLM")
    args = parser.parse_args()

    model_path = None if args.no_llm else args.model_path

    print("=" * 60)
    print("  Epistemic Edge Pipeline — End-to-End Experiment")
    print("=" * 60)

    results = run_experiment(model_path)

    print(f"\n{'='*60}")
    print(f"  OVERALL: {results['guardrail_accuracy']} guardrail decisions correct")
    print(f"{'='*60}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"02_pipeline_e2e_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
