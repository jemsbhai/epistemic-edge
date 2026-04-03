"""Comprehensive tests for trust/fusion.py — SLFusion."""

from datetime import datetime, timezone

import pytest

from epistemic_edge.models import FusedState, Observation, ObservationSource, StateGraph
from epistemic_edge.trust.fusion import SLFusion


@pytest.fixture
def fusion() -> SLFusion:
    return SLFusion()


@pytest.fixture
def empty_state() -> StateGraph:
    return StateGraph(node_id="test")


class TestFuseObservation:
    """Tests for SLFusion.fuse_observation."""

    def test_with_explicit_sl_bounds(self, fusion: SLFusion, empty_state: StateGraph) -> None:
        obs = Observation(
            payload={"temp": 42.0},
            source=ObservationSource(agent_id="s1"),
            belief=0.8,
            disbelief=0.1,
            uncertainty=0.1,
        )
        result = fusion.fuse_observation(obs, empty_state)
        assert result.belief == pytest.approx(0.8)
        assert result.disbelief == pytest.approx(0.1)
        assert result.uncertainty == pytest.approx(0.1)
        assert result.payload == {"temp": 42.0}
        assert result.sources == ["s1"]

    def test_vacuous_opinion_when_no_bounds(
        self, fusion: SLFusion, empty_state: StateGraph
    ) -> None:
        """Observation without SL bounds gets vacuous (0, 0, 1)."""
        obs = Observation(
            payload={"x": 1},
            source=ObservationSource(agent_id="s2"),
        )
        result = fusion.fuse_observation(obs, empty_state)
        assert result.belief == 0.0
        assert result.disbelief == 0.0
        assert result.uncertainty == 1.0

    def test_normalization_when_sum_not_one(
        self, fusion: SLFusion, empty_state: StateGraph
    ) -> None:
        """If sensor provides raw (b,d,u) that don't sum to 1, normalize them."""
        obs = Observation(
            payload={},
            source=ObservationSource(agent_id="s3"),
            belief=4.0,
            disbelief=2.0,
            uncertainty=4.0,
        )
        result = fusion.fuse_observation(obs, empty_state)
        assert result.belief == pytest.approx(0.4)
        assert result.disbelief == pytest.approx(0.2)
        assert result.uncertainty == pytest.approx(0.4)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)

    def test_already_normalized_stays_same(
        self, fusion: SLFusion, empty_state: StateGraph
    ) -> None:
        obs = Observation(
            payload={},
            source=ObservationSource(agent_id="s4"),
            belief=0.5,
            disbelief=0.3,
            uncertainty=0.2,
        )
        result = fusion.fuse_observation(obs, empty_state)
        assert result.belief == pytest.approx(0.5)
        assert result.disbelief == pytest.approx(0.3)
        assert result.uncertainty == pytest.approx(0.2)

    def test_partial_sl_belief_only_gets_vacuous_uncertainty(
        self, fusion: SLFusion, empty_state: StateGraph
    ) -> None:
        """Only belief provided — disbelief defaults to 0, uncertainty to 1 (vacuous)."""
        obs = Observation(
            payload={},
            source=ObservationSource(agent_id="s5"),
            belief=0.9,
        )
        result = fusion.fuse_observation(obs, empty_state)
        # b=0.9, d=0, u=1.0 → sum=1.9, gets normalized
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)
        assert result.belief == pytest.approx(0.9 / 1.9)
        assert result.disbelief == pytest.approx(0.0)
        assert result.uncertainty == pytest.approx(1.0 / 1.9)

    def test_all_zeros_stays_vacuous(
        self, fusion: SLFusion, empty_state: StateGraph
    ) -> None:
        """Edge case: all SL bounds explicitly set to 0."""
        obs = Observation(
            payload={},
            source=ObservationSource(agent_id="s6"),
            belief=0.0,
            disbelief=0.0,
            uncertainty=0.0,
        )
        result = fusion.fuse_observation(obs, empty_state)
        # total=0 → normalization guard prevents div-by-zero
        assert result.belief == 0.0
        assert result.disbelief == 0.0
        assert result.uncertainty == 0.0

    def test_payload_passes_through(
        self, fusion: SLFusion, empty_state: StateGraph
    ) -> None:
        payload = {"sensor": "lidar", "range_m": 15.3, "confidence": 0.92}
        obs = Observation(
            payload=payload,
            source=ObservationSource(agent_id="lidar_01"),
            belief=0.9,
            disbelief=0.05,
            uncertainty=0.05,
        )
        result = fusion.fuse_observation(obs, empty_state)
        assert result.payload == payload


class TestFusePair:
    """Tests for SLFusion.fuse_pair — cumulative fusion."""

    def test_fuse_pair_merges_sources(self, fusion: SLFusion) -> None:
        a = FusedState(
            payload={"a": 1}, belief=0.8, disbelief=0.1, uncertainty=0.1, sources=["s1"]
        )
        b = FusedState(
            payload={"b": 2}, belief=0.6, disbelief=0.2, uncertainty=0.2, sources=["s2"]
        )
        result = fusion.fuse_pair(a, b)
        assert "s1" in result.sources
        assert "s2" in result.sources
        assert len(result.sources) == 2

    def test_fuse_pair_merges_payloads(self, fusion: SLFusion) -> None:
        a = FusedState(
            payload={"temp": 42}, belief=0.8, disbelief=0.1, uncertainty=0.1, sources=["s1"]
        )
        b = FusedState(
            payload={"pressure": 101}, belief=0.7, disbelief=0.1, uncertainty=0.2, sources=["s2"]
        )
        result = fusion.fuse_pair(a, b)
        assert result.payload["temp"] == 42
        assert result.payload["pressure"] == 101

    def test_fuse_pair_produces_valid_opinion(self, fusion: SLFusion) -> None:
        a = FusedState(
            payload={}, belief=0.9, disbelief=0.05, uncertainty=0.05, sources=["s1"]
        )
        b = FusedState(
            payload={}, belief=0.7, disbelief=0.1, uncertainty=0.2, sources=["s2"]
        )
        result = fusion.fuse_pair(a, b)
        total = result.belief + result.disbelief + result.uncertainty
        assert total == pytest.approx(1.0, abs=0.01)

    def test_fuse_pair_symmetric_inputs(self, fusion: SLFusion) -> None:
        """Fusing identical opinions should produce similar confidence."""
        a = FusedState(
            payload={}, belief=0.6, disbelief=0.2, uncertainty=0.2, sources=["s1"]
        )
        b = FusedState(
            payload={}, belief=0.6, disbelief=0.2, uncertainty=0.2, sources=["s2"]
        )
        result = fusion.fuse_pair(a, b)
        # Two agreeing sources → uncertainty should decrease
        assert result.uncertainty <= a.uncertainty
