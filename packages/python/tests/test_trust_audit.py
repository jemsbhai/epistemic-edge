"""Comprehensive tests for trust/audit.py — PROVOAudit."""

from datetime import datetime, timezone

import pytest

from epistemic_edge.models import (
    EdgeIntent,
    FusedState,
    Observation,
    ObservationSource,
)
from epistemic_edge.trust.audit import PROVOActivity, PROVOAudit


class TestPROVOActivity:
    """Tests for individual PROV-O activity records."""

    def test_auto_generates_id(self) -> None:
        act = PROVOActivity(
            activity_type="test",
            entity_ids=["e1"],
            agent_ids=["a1"],
        )
        assert act.id.startswith("urn:epistemic-edge:activity:")
        assert len(act.id) > 30

    def test_unique_ids(self) -> None:
        ids = {
            PROVOActivity("test", ["e1"], ["a1"]).id
            for _ in range(100)
        }
        assert len(ids) == 100

    def test_to_jsonld_structure(self) -> None:
        act = PROVOActivity(
            activity_type="sensor_fusion",
            entity_ids=["urn:sensor:s1", "urn:sensor:s2"],
            agent_ids=["agent_alpha"],
        )
        doc = act.to_jsonld()
        assert doc["@type"] == "prov:Activity"
        assert doc["@id"] == act.id
        assert doc["ee:activityType"] == "sensor_fusion"
        assert len(doc["prov:used"]) == 2
        assert doc["prov:wasAssociatedWith"][0]["@id"] == "agent_alpha"

    def test_custom_timestamp(self) -> None:
        ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        act = PROVOActivity("test", ["e1"], ["a1"], timestamp=ts)
        assert act.timestamp == ts
        doc = act.to_jsonld()
        assert "2026-01-15" in doc["prov:startedAtTime"]

    def test_default_timestamp_is_recent(self) -> None:
        act = PROVOActivity("test", ["e1"], ["a1"])
        now = datetime.now(timezone.utc)
        delta = abs((now - act.timestamp).total_seconds())
        assert delta < 2.0


class TestPROVOAudit:
    """Tests for the audit trail manager."""

    @pytest.fixture
    def audit(self) -> PROVOAudit:
        return PROVOAudit()

    def test_starts_empty(self, audit: PROVOAudit) -> None:
        assert audit.count == 0
        assert audit.activities == []

    def test_log_fusion(self, audit: PROVOAudit) -> None:
        obs = Observation(
            payload={"temp": 42},
            source=ObservationSource(agent_id="thermo_1"),
        )
        fused = FusedState(
            payload={"temp": 42},
            belief=0.9,
            disbelief=0.05,
            uncertainty=0.05,
        )
        activity_id = audit.log_fusion(obs, fused)
        assert audit.count == 1
        assert activity_id.startswith("urn:epistemic-edge:activity:")
        # Should stamp the fused state with the activity ID
        assert fused.prov_o_activity_id == activity_id

    def test_log_actuation(self, audit: PROVOAudit) -> None:
        intent = EdgeIntent(action="close_valve", target="valve_3")
        activity_id = audit.log_actuation(intent)
        assert audit.count == 1
        assert activity_id.startswith("urn:epistemic-edge:activity:")

    def test_multiple_events_tracked(self, audit: PROVOAudit) -> None:
        for i in range(5):
            obs = Observation(
                payload={"i": i},
                source=ObservationSource(agent_id=f"sensor_{i}"),
            )
            fused = FusedState(payload={}, belief=0.5, disbelief=0.2, uncertainty=0.3)
            audit.log_fusion(obs, fused)

        audit.log_actuation(EdgeIntent(action="open_valve", target="v1"))
        assert audit.count == 6

    def test_export_graph_structure(self, audit: PROVOAudit) -> None:
        obs = Observation(
            payload={},
            source=ObservationSource(agent_id="s1"),
        )
        fused = FusedState(payload={}, belief=0.8, disbelief=0.1, uncertainty=0.1)
        audit.log_fusion(obs, fused)
        audit.log_actuation(EdgeIntent(action="stop", target="motor_1"))

        graph = audit.export_graph()
        assert "@context" in graph
        assert "prov" in graph["@context"]
        assert "ee" in graph["@context"]
        assert "@graph" in graph
        assert len(graph["@graph"]) == 2

    def test_export_graph_empty(self, audit: PROVOAudit) -> None:
        graph = audit.export_graph()
        assert graph["@graph"] == []

    def test_fusion_activity_references_sensor(self, audit: PROVOAudit) -> None:
        obs = Observation(
            payload={},
            source=ObservationSource(agent_id="lidar_01"),
        )
        fused = FusedState(payload={}, belief=0.7, disbelief=0.1, uncertainty=0.2)
        audit.log_fusion(obs, fused)

        activity = audit.activities[0]
        assert "urn:sensor-reading:lidar_01" in activity.entity_ids
        assert "lidar_01" in activity.agent_ids

    def test_actuation_activity_references_engine(self, audit: PROVOAudit) -> None:
        intent = EdgeIntent(action="deploy", target="drone_1")
        audit.log_actuation(intent)

        activity = audit.activities[0]
        assert "epistemic-edge:cognition-engine" in activity.agent_ids
        assert "urn:intent:deploy:drone_1" in activity.entity_ids
