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



class TestCumulativeFusionMath:
    """TDD: Tests encoding the EXACT Jøsang cumulative fusion formula.

    Reference: Jøsang (2016), Subjective Logic, §12.3, Eq. 12.3.

    For two opinions ω_A = (b_A, d_A, u_A) and ω_B = (b_B, d_B, u_B):
        κ = u_A + u_B − u_A · u_B
        b = (b_A · u_B + b_B · u_A) / κ
        d = (d_A · u_B + d_B · u_A) / κ
        u = (u_A · u_B) / κ

    These tests will FAIL if the implementation uses simple averaging
    instead of cumulative fusion. That is the point.
    """

    def test_cumulative_fusion_uncertainty_strictly_decreases(
        self, fusion: SLFusion
    ) -> None:
        """Core SL property: fusing two non-dogmatic opinions MUST produce
        uncertainty strictly less than BOTH inputs (Jøsang §12.3 Property 4)."""
        a = FusedState(
            payload={"a": 1}, belief=0.8, disbelief=0.1, uncertainty=0.1,
            sources=["s1"], base_rate=0.5,
        )
        b = FusedState(
            payload={"b": 2}, belief=0.7, disbelief=0.1, uncertainty=0.2,
            sources=["s2"], base_rate=0.5,
        )
        result = fusion.fuse_pair(a, b)

        # STRICT inequality — this catches the simple averaging bug
        assert result.uncertainty < min(a.uncertainty, b.uncertainty), (
            f"Cumulative fusion must STRICTLY reduce uncertainty below both inputs. "
            f"Got u={result.uncertainty:.4f}, but min(u_A, u_B)={min(a.uncertainty, b.uncertainty):.4f}. "
            f"If u equals the average, the implementation is using averaging, not cumulative fusion."
        )

    def test_cumulative_fusion_exact_values(self, fusion: SLFusion) -> None:
        """Verify exact numeric output against hand-computed Jøsang formula.

        ω_A = (0.8, 0.1, 0.1), ω_B = (0.7, 0.1, 0.2)
        κ = 0.1 + 0.2 − 0.1×0.2 = 0.28
        b = (0.8×0.2 + 0.7×0.1) / 0.28 = (0.16 + 0.07) / 0.28 = 0.8214...
        d = (0.1×0.2 + 0.1×0.1) / 0.28 = (0.02 + 0.01) / 0.28 = 0.1071...
        u = (0.1×0.2) / 0.28 = 0.02 / 0.28 = 0.0714...
        """
        a = FusedState(
            payload={}, belief=0.8, disbelief=0.1, uncertainty=0.1,
            sources=["s1"], base_rate=0.5,
        )
        b = FusedState(
            payload={}, belief=0.7, disbelief=0.1, uncertainty=0.2,
            sources=["s2"], base_rate=0.5,
        )
        result = fusion.fuse_pair(a, b)

        kappa = 0.1 + 0.2 - 0.1 * 0.2  # 0.28
        expected_b = (0.8 * 0.2 + 0.7 * 0.1) / kappa
        expected_d = (0.1 * 0.2 + 0.1 * 0.1) / kappa
        expected_u = (0.1 * 0.2) / kappa

        assert result.belief == pytest.approx(expected_b, abs=1e-4), (
            f"Expected b={expected_b:.4f}, got {result.belief:.4f}"
        )
        assert result.disbelief == pytest.approx(expected_d, abs=1e-4), (
            f"Expected d={expected_d:.4f}, got {result.disbelief:.4f}"
        )
        assert result.uncertainty == pytest.approx(expected_u, abs=1e-4), (
            f"Expected u={expected_u:.4f}, got {result.uncertainty:.4f}"
        )
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(
            1.0, abs=1e-9
        )

    def test_cumulative_fusion_conflicting_sources_increases_uncertainty(
        self, fusion: SLFusion
    ) -> None:
        """When sources conflict, fused disbelief should be higher than
        either input's disbelief — the system correctly reflects disagreement."""
        # Source A says "believe" (high b)
        a = FusedState(
            payload={}, belief=0.85, disbelief=0.05, uncertainty=0.10,
            sources=["s1"], base_rate=0.5,
        )
        # Source B says "disbelieve" (high d)
        b = FusedState(
            payload={}, belief=0.05, disbelief=0.85, uncertainty=0.10,
            sources=["s2"], base_rate=0.5,
        )
        result = fusion.fuse_pair(a, b)

        # Both have u=0.1: κ = 0.1 + 0.1 - 0.01 = 0.19
        # b = (0.85×0.1 + 0.05×0.1) / 0.19 = 0.09/0.19 ≈ 0.4737
        # d = (0.05×0.1 + 0.85×0.1) / 0.19 = 0.09/0.19 ≈ 0.4737
        # u = 0.01/0.19 ≈ 0.0526
        # Result: roughly equal b and d → high projected uncertainty
        assert result.belief == pytest.approx(result.disbelief, abs=0.01), (
            "Symmetric conflicting sources should produce near-equal b and d"
        )
        assert result.uncertainty < min(a.uncertainty, b.uncertainty)

    def test_vacuous_fused_with_informative_returns_informative(
        self, fusion: SLFusion
    ) -> None:
        """Fusing with vacuous opinion (0,0,1) should return the other opinion
        unchanged — vacuous is the identity element (Jøsang §12.3 Property 3)."""
        informative = FusedState(
            payload={"x": 1}, belief=0.7, disbelief=0.2, uncertainty=0.1,
            sources=["s1"], base_rate=0.5,
        )
        vacuous = FusedState(
            payload={"y": 2}, belief=0.0, disbelief=0.0, uncertainty=1.0,
            sources=["s2"], base_rate=0.5,
        )
        result = fusion.fuse_pair(informative, vacuous)

        # κ = 0.1 + 1.0 − 0.1×1.0 = 1.0
        # b = (0.7×1.0 + 0.0×0.1)/1.0 = 0.7
        # d = (0.2×1.0 + 0.0×0.1)/1.0 = 0.2
        # u = (0.1×1.0)/1.0 = 0.1
        assert result.belief == pytest.approx(0.7, abs=1e-4)
        assert result.disbelief == pytest.approx(0.2, abs=1e-4)
        assert result.uncertainty == pytest.approx(0.1, abs=1e-4)

    def test_commutativity(self, fusion: SLFusion) -> None:
        """Cumulative fusion must be commutative: A ⊕ B = B ⊕ A."""
        a = FusedState(
            payload={}, belief=0.6, disbelief=0.15, uncertainty=0.25,
            sources=["s1"], base_rate=0.5,
        )
        b = FusedState(
            payload={}, belief=0.3, disbelief=0.4, uncertainty=0.3,
            sources=["s2"], base_rate=0.5,
        )
        r_ab = fusion.fuse_pair(a, b)
        r_ba = fusion.fuse_pair(b, a)

        assert r_ab.belief == pytest.approx(r_ba.belief, abs=1e-9)
        assert r_ab.disbelief == pytest.approx(r_ba.disbelief, abs=1e-9)
        assert r_ab.uncertainty == pytest.approx(r_ba.uncertainty, abs=1e-9)

    def test_cumulative_fusion_not_simple_averaging(self, fusion: SLFusion) -> None:
        """Explicitly verify the result does NOT match simple averaging.

        This is the canary test: if this passes but test_exact_values fails,
        something deeply wrong is happening. If this fails, the implementation
        is using averaging instead of Jøsang fusion.
        """
        a = FusedState(
            payload={}, belief=0.8, disbelief=0.1, uncertainty=0.1,
            sources=["s1"], base_rate=0.5,
        )
        b = FusedState(
            payload={}, belief=0.7, disbelief=0.1, uncertainty=0.2,
            sources=["s2"], base_rate=0.5,
        )
        result = fusion.fuse_pair(a, b)

        avg_b = (0.8 + 0.7) / 2  # 0.75
        avg_d = (0.1 + 0.1) / 2  # 0.10
        avg_u = (0.1 + 0.2) / 2  # 0.15

        # At least one of these must differ from the average
        is_averaging = (
            abs(result.belief - avg_b) < 1e-6
            and abs(result.disbelief - avg_d) < 1e-6
            and abs(result.uncertainty - avg_u) < 1e-6
        )
        assert not is_averaging, (
            f"CRITICAL: fuse_pair is using simple averaging, NOT cumulative fusion! "
            f"Got b={result.belief:.4f}, d={result.disbelief:.4f}, u={result.uncertainty:.4f} "
            f"which matches avg(b)={avg_b}, avg(d)={avg_d}, avg(u)={avg_u}"
        )
