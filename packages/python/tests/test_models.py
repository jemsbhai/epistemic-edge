"""Comprehensive tests for models.py and orchestrator.py."""

from datetime import datetime, timedelta, timezone

import pytest

from epistemic_edge.models import (
    ActuationResult,
    EdgeIntent,
    FusedState,
    Observation,
    ObservationSource,
    StateGraph,
    TierLabel,
    _utcnow,
)
from epistemic_edge.orchestrator import EdgeNode
from epistemic_edge.memory.cache import DecayConfig


def _now() -> datetime:
    return datetime.now(timezone.utc)


# =============================================================================
# Model tests
# =============================================================================


class TestUtcnow:
    def test_returns_timezone_aware(self) -> None:
        ts = _utcnow()
        assert ts.tzinfo is not None

    def test_is_utc(self) -> None:
        ts = _utcnow()
        assert ts.tzinfo == timezone.utc

    def test_is_recent(self) -> None:
        before = _now()
        ts = _utcnow()
        after = _now()
        assert before <= ts <= after


class TestTierLabel:
    def test_values(self) -> None:
        assert TierLabel.TRANSPORT == "transport"
        assert TierLabel.TRUST == "trust"
        assert TierLabel.MEMORY == "memory"
        assert TierLabel.COGNITION == "cognition"

    def test_is_string(self) -> None:
        assert isinstance(TierLabel.TRANSPORT, str)


class TestObservationSource:
    def test_defaults(self) -> None:
        src = ObservationSource(agent_id="s1")
        assert src.agent_type == "sensor"
        assert src.calibration_confidence is None
        assert src.prov_o_entity_id is None

    def test_custom(self) -> None:
        src = ObservationSource(
            agent_id="cam_01",
            agent_type="camera",
            calibration_confidence=0.98,
            prov_o_entity_id="urn:device:cam_01",
        )
        assert src.agent_id == "cam_01"
        assert src.agent_type == "camera"


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
        assert obs.timestamp.tzinfo is not None

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

    def test_custom_timestamp(self) -> None:
        ts = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        obs = Observation(
            payload={},
            source=ObservationSource(agent_id="s"),
            timestamp=ts,
        )
        assert obs.timestamp == ts


class TestFusedState:
    def test_expected_value(self) -> None:
        fs = FusedState(payload={}, belief=0.7, disbelief=0.1, uncertainty=0.2)
        assert fs.expected_value == pytest.approx(0.8)

    def test_vacuous_opinion(self) -> None:
        fs = FusedState(payload={}, belief=0.0, disbelief=0.0, uncertainty=1.0)
        assert fs.expected_value == pytest.approx(0.5)

    def test_dogmatic_belief(self) -> None:
        fs = FusedState(payload={}, belief=1.0, disbelief=0.0, uncertainty=0.0)
        assert fs.expected_value == pytest.approx(1.0)

    def test_dogmatic_disbelief(self) -> None:
        fs = FusedState(payload={}, belief=0.0, disbelief=1.0, uncertainty=0.0)
        assert fs.expected_value == pytest.approx(0.0)

    def test_fused_at_is_timezone_aware(self) -> None:
        fs = FusedState(payload={}, belief=0.5, disbelief=0.2, uncertainty=0.3)
        assert fs.fused_at.tzinfo is not None

    def test_sources_default_empty(self) -> None:
        fs = FusedState(payload={}, belief=0.5, disbelief=0.2, uncertainty=0.3)
        assert fs.sources == []


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

    def test_max_uncertainty_picks_highest(self) -> None:
        sg = StateGraph(
            node_id="test",
            facts=[
                FusedState(payload={}, belief=0.9, disbelief=0.05, uncertainty=0.05),
                FusedState(payload={}, belief=0.3, disbelief=0.1, uncertainty=0.6),
                FusedState(payload={}, belief=0.5, disbelief=0.2, uncertainty=0.3),
            ],
        )
        assert sg.max_uncertainty() == pytest.approx(0.6)

    def test_stale_list_independent(self) -> None:
        sg = StateGraph(node_id="test")
        assert sg.stale == []
        sg.stale.append(FusedState(payload={}, belief=0.1, disbelief=0.8, uncertainty=0.1))
        assert len(sg.stale) == 1
        assert sg.active_count == 0


class TestEdgeIntent:
    def test_basic(self) -> None:
        intent = EdgeIntent(action="close_valve", target="valve_3")
        assert intent.action == "close_valve"
        assert intent.grammar_constrained is False
        assert intent.parameters == {}
        assert intent.raw_llm_output is None

    def test_with_parameters(self) -> None:
        intent = EdgeIntent(
            action="set_speed",
            target="motor_1",
            parameters={"rpm": 1500},
        )
        assert intent.parameters["rpm"] == 1500

    def test_generated_at_is_timezone_aware(self) -> None:
        intent = EdgeIntent(action="x", target="y")
        assert intent.generated_at.tzinfo is not None


class TestActuationResult:
    def test_permitted(self) -> None:
        intent = EdgeIntent(action="x", target="y")
        result = ActuationResult(intent=intent, permitted=True)
        assert result.permitted is True
        assert result.reason == ""

    def test_denied_with_reason(self) -> None:
        intent = EdgeIntent(action="x", target="y")
        result = ActuationResult(
            intent=intent, permitted=False, reason="Uncertainty too high"
        )
        assert result.permitted is False
        assert "Uncertainty" in result.reason


# =============================================================================
# Orchestrator tests
# =============================================================================


class TestEdgeNodeInit:
    def test_defaults(self) -> None:
        node = EdgeNode(node_id="n1")
        assert node.node_id == "n1"
        assert node.state.node_id == "n1"
        assert node.llm_path is None
        assert node.audit_mode == "prov-o"
        assert node._audit is not None

    def test_no_audit(self) -> None:
        node = EdgeNode(node_id="n1", audit_mode="none")
        assert node._audit is None

    def test_custom_decay(self) -> None:
        cfg = DecayConfig(mean_reversion_rate=5.0, threshold=0.3)
        node = EdgeNode(node_id="n1", decay=cfg)
        assert node.decay_config.mean_reversion_rate == 5.0


class TestEdgeNodeIngest:
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
        assert fused.belief == pytest.approx(0.9)
        assert node.state.active_count == 1

    def test_multiple_ingests(self) -> None:
        node = EdgeNode(node_id="test_node")
        for i in range(10):
            obs = Observation(
                payload={"i": i},
                source=ObservationSource(agent_id=f"s_{i}"),
                belief=0.8,
                disbelief=0.1,
                uncertainty=0.1,
            )
            node.ingest(obs)
        assert node.state.active_count == 10

    def test_ingest_logs_audit_trail(self) -> None:
        node = EdgeNode(node_id="test_node")
        obs = Observation(
            payload={},
            source=ObservationSource(agent_id="s1"),
            belief=0.9,
            disbelief=0.05,
            uncertainty=0.05,
        )
        node.ingest(obs)
        assert node._audit is not None
        assert node._audit.count == 1

    def test_ingest_without_audit(self) -> None:
        node = EdgeNode(node_id="test_node", audit_mode="none")
        obs = Observation(
            payload={},
            source=ObservationSource(agent_id="s1"),
            belief=0.5,
            disbelief=0.2,
            uncertainty=0.3,
        )
        fused = node.ingest(obs)
        assert fused.prov_o_activity_id is None


class TestEdgeNodeGuardrails:
    def test_fail_closed(self) -> None:
        node = EdgeNode(node_id="test_node")
        intent = EdgeIntent(action="unknown_action", target="x")
        result = node.verify_intent(intent)
        assert result.permitted is False
        assert "No guardrail registered" in result.reason

    def test_guardrail_permits(self) -> None:
        node = EdgeNode(node_id="test_node")

        @node.guardrail(action="close_valve")
        def check(state: StateGraph, intent: EdgeIntent) -> bool:
            return state.max_uncertainty() < 0.5

        node.state.facts.append(
            FusedState(payload={}, belief=0.9, disbelief=0.05, uncertainty=0.05)
        )
        intent = EdgeIntent(action="close_valve", target="valve_1")
        result = node.verify_intent(intent)
        assert result.permitted is True
        assert result.executed_at is not None

    def test_guardrail_denies(self) -> None:
        node = EdgeNode(node_id="test_node")

        @node.guardrail(action="close_valve")
        def check(state: StateGraph, intent: EdgeIntent) -> bool:
            return state.max_uncertainty() < 0.5

        # Empty state → uncertainty=1.0 > 0.5
        intent = EdgeIntent(action="close_valve", target="valve_1")
        result = node.verify_intent(intent)
        assert result.permitted is False

    def test_guardrail_exception_denies(self) -> None:
        node = EdgeNode(node_id="test_node")

        @node.guardrail(action="explode")
        def check(state: StateGraph, intent: EdgeIntent) -> bool:
            raise ValueError("Sensor offline")

        intent = EdgeIntent(action="explode", target="x")
        result = node.verify_intent(intent)
        assert result.permitted is False
        assert "Guardrail raised" in result.reason

    def test_multiple_guardrails(self) -> None:
        node = EdgeNode(node_id="test_node")

        @node.guardrail(action="open")
        def g1(state: StateGraph, intent: EdgeIntent) -> bool:
            return True

        @node.guardrail(action="close")
        def g2(state: StateGraph, intent: EdgeIntent) -> bool:
            return False

        assert node.verify_intent(EdgeIntent(action="open", target="x")).permitted is True
        assert node.verify_intent(EdgeIntent(action="close", target="x")).permitted is False

    def test_permitted_intent_gets_prov_o_receipt(self) -> None:
        node = EdgeNode(node_id="test_node")

        @node.guardrail(action="go")
        def check(state: StateGraph, intent: EdgeIntent) -> bool:
            return True

        result = node.verify_intent(EdgeIntent(action="go", target="y"))
        assert result.permitted is True
        assert result.prov_o_receipt_id is not None
        assert result.prov_o_receipt_id.startswith("urn:epistemic-edge:activity:")


class TestEdgeNodeSweep:
    def test_sweep_removes_old_facts(self) -> None:
        node = EdgeNode(
            node_id="test_node",
            decay=DecayConfig(mean_reversion_rate=10.0, threshold=0.5),
        )
        old_time = _now() - timedelta(hours=2)
        node.state.facts.append(
            FusedState(
                payload={"old": True},
                belief=0.8,
                disbelief=0.1,
                uncertainty=0.1,
                fused_at=old_time,
            )
        )
        node.state.facts.append(
            FusedState(payload={"fresh": True}, belief=0.9, disbelief=0.05, uncertainty=0.05)
        )
        pruned = node.sweep()
        assert pruned >= 1
        assert node.state.active_count >= 1

    def test_sweep_moves_to_stale(self) -> None:
        node = EdgeNode(
            node_id="test_node",
            decay=DecayConfig(mean_reversion_rate=10.0, threshold=0.5),
        )
        old_time = _now() - timedelta(hours=5)
        node.state.facts.append(
            FusedState(payload={"old": True}, belief=0.8, disbelief=0.1, uncertainty=0.1,
                       fused_at=old_time)
        )
        node.sweep()
        assert len(node.state.stale) >= 1

    def test_sweep_empty_state(self) -> None:
        node = EdgeNode(node_id="test_node")
        pruned = node.sweep()
        assert pruned == 0
        assert node.state.active_count == 0

    def test_sweep_updates_last_sweep(self) -> None:
        node = EdgeNode(node_id="test_node")
        assert node.state.last_sweep is None
        node.sweep()
        assert node.state.last_sweep is not None

    def test_sweep_with_custom_query_time(self) -> None:
        node = EdgeNode(
            node_id="test_node",
            decay=DecayConfig(mean_reversion_rate=10.0, threshold=0.5),
        )
        fact_time = _now()
        node.state.facts.append(
            FusedState(payload={}, belief=0.9, disbelief=0.05, uncertainty=0.05,
                       fused_at=fact_time)
        )
        # Sweep at a time far in the future — should prune
        future = fact_time + timedelta(days=30)
        pruned = node.sweep(query_time=future)
        assert pruned == 1


class TestEdgeNodeGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_none_without_llm(self) -> None:
        node = EdgeNode(node_id="test_node")
        result = await node.generate("What is happening?")
        assert result is None


class TestEdgeNodeOnActuate:
    def test_on_actuate_registers(self) -> None:
        node = EdgeNode(node_id="test_node")

        @node.on_actuate
        async def handler(intent, receipt):
            pass

        assert node._actuate_handler is not None
