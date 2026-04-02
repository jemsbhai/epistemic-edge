"""Tests for core models and orchestrator logic."""

from datetime import datetime, timedelta

import pytest

from epistemic_edge.models import (
    ActuationResult,
    EdgeIntent,
    FusedState,
    Observation,
    ObservationSource,
    StateGraph,
)
from epistemic_edge.orchestrator import EdgeNode
from epistemic_edge.memory.cache import DecayConfig


# -- Model tests --------------------------------------------------------------


class TestObservation:
    def test_defaults(self) -> None:
        obs = Observation(
            payload={"temp": 42.0},
            source=ObservationSource(agent_id="sensor_1"),
        )
        assert obs.payload["temp"] == 42.0
        assert obs.source.agent_id == "sensor_1"
        assert obs.belief is None
        assert obs.uncertainty is None

    def test_with_sl_bounds(self) -> None:
        obs = Observation(
            payload={"pressure": 101.3},
            source=ObservationSource(agent_id="sensor_2", calibration_confidence=0.95),
            belief=0.8,
            disbelief=0.05,
            uncertainty=0.15,
        )
        assert obs.belief == 0.8
        assert obs.disbelief == 0.05
        assert obs.uncertainty == 0.15


class TestFusedState:
    def test_expected_value(self) -> None:
        fs = FusedState(
            payload={},
            belief=0.7,
            disbelief=0.1,
            uncertainty=0.2,
        )
        # expected = b + 0.5 * u = 0.7 + 0.1 = 0.8
        assert abs(fs.expected_value - 0.8) < 1e-9

    def test_vacuous_opinion(self) -> None:
        fs = FusedState(payload={}, belief=0.0, disbelief=0.0, uncertainty=1.0)
        assert abs(fs.expected_value - 0.5) < 1e-9


class TestStateGraph:
    def test_empty_max_uncertainty(self) -> None:
        sg = StateGraph(node_id="test")
        assert sg.max_uncertainty() == 1.0

    def test_active_count(self) -> None:
        sg = StateGraph(
            node_id="test",
            facts=[
                FusedState(payload={}, belief=0.9, disbelief=0.05, uncertainty=0.05),
                FusedState(payload={}, belief=0.5, disbelief=0.2, uncertainty=0.3),
            ],
        )
        assert sg.active_count == 2
        assert abs(sg.max_uncertainty() - 0.3) < 1e-9


class TestEdgeIntent:
    def test_basic(self) -> None:
        intent = EdgeIntent(action="close_valve", target="valve_3")
        assert intent.action == "close_valve"
        assert intent.grammar_constrained is False


# -- Orchestrator tests --------------------------------------------------------


class TestEdgeNode:
    def test_ingest_creates_fused_state(self) -> None:
        node = EdgeNode(node_id="test_node")
        obs = Observation(
            payload={"temp": 55.0},
            source=ObservationSource(agent_id="thermo_1"),
            belief=0.9,
            disbelief=0.05,
            uncertainty=0.05,
        )
        fused = node.ingest(obs)
        assert fused.belief == 0.9
        assert node.state.active_count == 1

    def test_guardrail_fail_closed(self) -> None:
        node = EdgeNode(node_id="test_node")
        intent = EdgeIntent(action="unknown_action", target="x")
        result = node.verify_intent(intent)
        assert result.permitted is False
        assert "No guardrail registered" in result.reason

    def test_guardrail_registered(self) -> None:
        node = EdgeNode(node_id="test_node")

        @node.guardrail(action="close_valve")
        def check(state: StateGraph, intent: EdgeIntent) -> bool:
            return state.max_uncertainty() < 0.5

        # With empty state, uncertainty = 1.0 > 0.5, should be denied
        intent = EdgeIntent(action="close_valve", target="valve_1")
        result = node.verify_intent(intent)
        assert result.permitted is False

        # Add a high-confidence fact
        node.state.facts.append(
            FusedState(payload={}, belief=0.9, disbelief=0.05, uncertainty=0.05)
        )
        result = node.verify_intent(intent)
        assert result.permitted is True

    def test_sweep_removes_old_facts(self) -> None:
        node = EdgeNode(
            node_id="test_node",
            decay=DecayConfig(mean_reversion_rate=10.0, threshold=0.5),
        )
        old_time = datetime.utcnow() - timedelta(hours=2)
        node.state.facts.append(
            FusedState(
                payload={"old": True},
                belief=0.8,
                disbelief=0.1,
                uncertainty=0.1,
                fused_at=old_time,
            )
        )
        now_fact = FusedState(
            payload={"fresh": True},
            belief=0.9,
            disbelief=0.05,
            uncertainty=0.05,
        )
        node.state.facts.append(now_fact)

        pruned = node.sweep()
        assert pruned >= 1
        assert node.state.active_count >= 1
