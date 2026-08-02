"""Tests for dataset adapter base classes and uncertainty strategies.

TDD: These tests define the expected 4-tuple (b, d, u, a) interface
for all uncertainty strategies and the adapter base class helpers.

Every strategy.assign() must return (belief, disbelief, uncertainty, base_rate)
where b+d+u=1.0, all >= 0, and a in [0,1].
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from epistemic_edge.adapters.base import (
    DatasetAdapter,
    DatasetMetadata,
    GroundTruth,
    SensorContext,
    TimeWindow,
)
from epistemic_edge.adapters.uncertainty import (
    CompositeStrategy,
    HistoricalDeviationStrategy,
    PhysicsBoundsStrategy,
    SensorAgreementStrategy,
    UncertaintyStrategy,
)
from epistemic_edge.models import Observation, ObservationSource


# ── Fixtures ─────────────────────────────────────────────────────────────────

NOW = datetime(2026, 4, 3, 12, 0, 0, tzinfo=timezone.utc)


def _ctx(
    reading: float = 25.0,
    sensor_id: str = "T_001",
    historical_mean: float | None = 25.0,
    historical_std: float | None = 2.0,
    physical_min: float | None = 0.0,
    physical_max: float | None = 100.0,
    related_readings: dict[str, float] | None = None,
) -> SensorContext:
    """Helper to build a SensorContext with sensible defaults."""
    return SensorContext(
        sensor_id=sensor_id,
        reading=reading,
        timestamp=NOW,
        historical_mean=historical_mean,
        historical_std=historical_std,
        physical_min=physical_min,
        physical_max=physical_max,
        related_readings=related_readings or {},
        sensor_type="analog",
        unit="°C",
    )


def _assert_valid_opinion(result: tuple, msg: str = "") -> None:
    """Assert that a result is a valid 4-tuple SL opinion with base_rate."""
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}. {msg}"
    assert len(result) == 4, f"Expected 4-tuple (b,d,u,a), got {len(result)}-tuple. {msg}"

    b, d, u, a = result

    # All must be floats (or at least numeric)
    for name, val in [("b", b), ("d", d), ("u", u), ("a", a)]:
        assert isinstance(val, (int, float)), f"{name} must be numeric, got {type(val)}. {msg}"

    # b, d, u >= 0
    assert b >= 0, f"b must be >= 0, got {b}. {msg}"
    assert d >= 0, f"d must be >= 0, got {d}. {msg}"
    assert u >= 0, f"u must be >= 0, got {u}. {msg}"

    # b + d + u = 1.0 (within floating point tolerance)
    total = b + d + u
    assert abs(total - 1.0) < 1e-9, f"b+d+u must = 1.0, got {total}. {msg}"

    # a in [0, 1]
    assert 0.0 <= a <= 1.0, f"base_rate must be in [0,1], got {a}. {msg}"


# ── _clamp_and_normalize tests ───────────────────────────────────────────────


class TestClampAndNormalize:
    """Tests for the shared _clamp_and_normalize helper."""

    def test_returns_4_tuple(self):
        strategy = SensorAgreementStrategy()
        result = strategy._clamp_and_normalize(0.5, 0.2, 0.3)
        _assert_valid_opinion(result, "_clamp_and_normalize default base_rate")

    def test_default_base_rate_is_0_5(self):
        strategy = SensorAgreementStrategy()
        _b, _d, _u, a = strategy._clamp_and_normalize(0.5, 0.2, 0.3)
        assert a == 0.5

    def test_custom_base_rate_propagated(self):
        strategy = SensorAgreementStrategy()
        _b, _d, _u, a = strategy._clamp_and_normalize(0.5, 0.2, 0.3, a=0.8)
        assert a == 0.8

    def test_base_rate_clamped_to_unit_interval(self):
        strategy = SensorAgreementStrategy()
        _, _, _, a_high = strategy._clamp_and_normalize(0.5, 0.2, 0.3, a=1.5)
        assert a_high == 1.0
        _, _, _, a_low = strategy._clamp_and_normalize(0.5, 0.2, 0.3, a=-0.3)
        assert a_low == 0.0

    def test_all_zero_returns_vacuous(self):
        strategy = SensorAgreementStrategy()
        b, d, u, a = strategy._clamp_and_normalize(0.0, 0.0, 0.0, a=0.7)
        assert b == 0.0
        assert d == 0.0
        assert u == 1.0
        assert a == 0.7

    def test_negative_values_clamped(self):
        strategy = SensorAgreementStrategy()
        result = strategy._clamp_and_normalize(-0.1, 0.5, 0.5, a=0.5)
        _assert_valid_opinion(result, "negative b clamped")
        assert result[0] == 0.0  # b should be clamped to 0

    def test_normalization(self):
        strategy = SensorAgreementStrategy()
        b, d, u, _ = strategy._clamp_and_normalize(2.0, 1.0, 1.0)
        assert abs(b - 0.5) < 1e-9
        assert abs(d - 0.25) < 1e-9
        assert abs(u - 0.25) < 1e-9


# ── SensorAgreementStrategy tests ───────────────────────────────────────────


class TestSensorAgreementStrategy:
    """Tests for SensorAgreementStrategy.assign() returning 4-tuples."""

    def setup_method(self):
        self.strategy = SensorAgreementStrategy()

    def test_returns_4_tuple_no_related(self):
        """No related sensors: moderate fallback, still 4-tuple."""
        ctx = _ctx(related_readings={})
        result = self.strategy.assign(ctx)
        _assert_valid_opinion(result, "no related sensors")

    def test_returns_4_tuple_with_agreement(self):
        """Strong agreement across related sensors."""
        ctx = _ctx(reading=25.0, related_readings={"T_002": 25.1, "T_003": 24.9})
        result = self.strategy.assign(ctx)
        _assert_valid_opinion(result, "strong agreement")
        b, d, u, a = result
        assert b > 0.7, "High agreement should yield high belief"

    def test_returns_4_tuple_with_disagreement(self):
        """Sensors disagree significantly."""
        ctx = _ctx(reading=25.0, related_readings={"T_002": 50.0, "T_003": 80.0})
        result = self.strategy.assign(ctx)
        _assert_valid_opinion(result, "disagreement")
        b, d, u, a = result
        assert u > 0.4, "Disagreement should yield high uncertainty"

    def test_default_base_rate(self):
        """Base rate should default to 0.5."""
        ctx = _ctx(related_readings={})
        _, _, _, a = self.strategy.assign(ctx)
        assert a == 0.5


# ── HistoricalDeviationStrategy tests ────────────────────────────────────────


class TestHistoricalDeviationStrategy:
    """Tests for HistoricalDeviationStrategy.assign() returning 4-tuples."""

    def setup_method(self):
        self.strategy = HistoricalDeviationStrategy()

    def test_returns_4_tuple_no_history(self):
        """No historical data: vacuous opinion, still 4-tuple."""
        ctx = _ctx(historical_mean=None, historical_std=None)
        result = self.strategy.assign(ctx)
        _assert_valid_opinion(result, "no history")
        b, d, u, a = result
        assert u == 1.0, "No history should be vacuous"

    def test_returns_4_tuple_normal_reading(self):
        """Reading within 1 sigma of mean."""
        ctx = _ctx(reading=25.5, historical_mean=25.0, historical_std=2.0)
        result = self.strategy.assign(ctx)
        _assert_valid_opinion(result, "normal reading")
        b, d, u, a = result
        assert b > 0.7, "Normal reading should have high belief"

    def test_returns_4_tuple_anomalous_reading(self):
        """Reading beyond 3 sigma."""
        ctx = _ctx(reading=35.0, historical_mean=25.0, historical_std=2.0)
        result = self.strategy.assign(ctx)
        _assert_valid_opinion(result, "anomalous reading")
        b, d, u, a = result
        assert d > 0.3, "Anomalous reading should have elevated disbelief"

    def test_returns_4_tuple_constant_sensor_exact(self):
        """Constant sensor (std=0), reading matches mean."""
        ctx = _ctx(reading=25.0, historical_mean=25.0, historical_std=0.0)
        result = self.strategy.assign(ctx)
        _assert_valid_opinion(result, "constant sensor exact match")

    def test_returns_4_tuple_constant_sensor_deviation(self):
        """Constant sensor (std=0), reading deviates from mean."""
        ctx = _ctx(reading=26.0, historical_mean=25.0, historical_std=0.0)
        result = self.strategy.assign(ctx)
        _assert_valid_opinion(result, "constant sensor deviation")

    def test_default_base_rate(self):
        ctx = _ctx(reading=25.0, historical_mean=25.0, historical_std=2.0)
        _, _, _, a = self.strategy.assign(ctx)
        assert a == 0.5


# ── PhysicsBoundsStrategy tests ──────────────────────────────────────────────


class TestPhysicsBoundsStrategy:
    """Tests for PhysicsBoundsStrategy.assign() returning 4-tuples."""

    def setup_method(self):
        self.strategy = PhysicsBoundsStrategy()

    def test_returns_4_tuple_within_bounds(self):
        """Reading comfortably within physical bounds."""
        ctx = _ctx(reading=50.0, physical_min=0.0, physical_max=100.0)
        result = self.strategy.assign(ctx)
        _assert_valid_opinion(result, "within bounds")
        b, d, u, a = result
        assert b > 0.7, "Within bounds should have high belief"

    def test_returns_4_tuple_below_min(self):
        """Reading below physical minimum: impossible."""
        ctx = _ctx(reading=-5.0, physical_min=0.0, physical_max=100.0)
        result = self.strategy.assign(ctx)
        _assert_valid_opinion(result, "below physical min")
        b, d, u, a = result
        assert d > 0.5, "Below physical min should have high disbelief"

    def test_returns_4_tuple_above_max(self):
        """Reading above physical maximum: impossible."""
        ctx = _ctx(reading=110.0, physical_min=0.0, physical_max=100.0)
        result = self.strategy.assign(ctx)
        _assert_valid_opinion(result, "above physical max")
        b, d, u, a = result
        assert d > 0.5, "Above physical max should have high disbelief"

    def test_returns_4_tuple_no_bounds_fallback(self):
        """No physical bounds: falls back to HistoricalDeviation, still 4-tuple."""
        ctx = _ctx(
            reading=25.0,
            physical_min=None,
            physical_max=None,
            historical_mean=25.0,
            historical_std=2.0,
        )
        result = self.strategy.assign(ctx)
        _assert_valid_opinion(result, "no bounds fallback to historical")

    def test_returns_4_tuple_near_boundary(self):
        """Reading near a physical boundary."""
        ctx = _ctx(reading=2.0, physical_min=0.0, physical_max=100.0)
        result = self.strategy.assign(ctx)
        _assert_valid_opinion(result, "near boundary")

    def test_returns_4_tuple_degenerate_bounds(self):
        """Degenerate bounds (min == max)."""
        ctx = _ctx(reading=50.0, physical_min=50.0, physical_max=50.0)
        result = self.strategy.assign(ctx)
        _assert_valid_opinion(result, "degenerate bounds")

    def test_default_base_rate(self):
        ctx = _ctx(reading=50.0, physical_min=0.0, physical_max=100.0)
        _, _, _, a = self.strategy.assign(ctx)
        assert a == 0.5


# ── CompositeStrategy tests ──────────────────────────────────────────────────


class TestCompositeStrategy:
    """Tests for CompositeStrategy.assign() returning 4-tuples."""

    def setup_method(self):
        self.strategy = CompositeStrategy()

    def test_returns_4_tuple(self):
        """Composite of all three strategies returns valid 4-tuple."""
        ctx = _ctx(
            reading=25.0,
            related_readings={"T_002": 25.1},
            historical_mean=25.0,
            historical_std=2.0,
            physical_min=0.0,
            physical_max=100.0,
        )
        result = self.strategy.assign(ctx)
        _assert_valid_opinion(result, "composite strategy")

    def test_returns_4_tuple_minimal_context(self):
        """Composite with minimal sensor context (no related, no history, no bounds)."""
        ctx = _ctx(
            related_readings={},
            historical_mean=None,
            historical_std=None,
            physical_min=None,
            physical_max=None,
        )
        result = self.strategy.assign(ctx)
        _assert_valid_opinion(result, "composite minimal context")

    def test_default_base_rate(self):
        ctx = _ctx(
            reading=25.0,
            related_readings={"T_002": 25.1},
            historical_mean=25.0,
            historical_std=2.0,
            physical_min=0.0,
            physical_max=100.0,
        )
        _, _, _, a = self.strategy.assign(ctx)
        assert a == 0.5

    def test_custom_weights(self):
        """Custom-weighted composite still returns valid 4-tuple."""
        strategy = CompositeStrategy(
            strategies=[
                (SensorAgreementStrategy(), 3.0),
                (HistoricalDeviationStrategy(), 1.0),
            ]
        )
        ctx = _ctx(
            reading=25.0,
            related_readings={"T_002": 25.1},
            historical_mean=25.0,
            historical_std=2.0,
        )
        result = strategy.assign(ctx)
        _assert_valid_opinion(result, "custom weighted composite")


# ── DatasetAdapter base class helper tests ───────────────────────────────────


class TestAdapterBaseHelpers:
    """Tests for DatasetAdapter._assign_bdu and _make_observation."""

    def _make_adapter(self, strategy=None):
        """Create a minimal concrete adapter for testing base class methods."""

        class _StubAdapter(DatasetAdapter):
            def load(self, path, split="test"):
                pass

            def get_windows(self, window_size=60, stride=30, max_windows=None):
                yield from []

            def get_sensor_contexts(self, row_idx):
                return []

        return _StubAdapter(strategy=strategy)

    def test_assign_bdu_returns_4_tuple(self):
        """_assign_bdu must return (b, d, u, a)."""
        adapter = self._make_adapter(strategy=HistoricalDeviationStrategy())
        ctx = _ctx(reading=25.0, historical_mean=25.0, historical_std=2.0)
        result = adapter._assign_bdu(ctx)
        _assert_valid_opinion(result, "_assign_bdu")

    def test_assign_bdu_default_strategy(self):
        """Default strategy (HistoricalDeviation) also returns 4-tuple."""
        adapter = self._make_adapter()
        ctx = _ctx(reading=25.0, historical_mean=25.0, historical_std=2.0)
        result = adapter._assign_bdu(ctx)
        _assert_valid_opinion(result, "_assign_bdu default strategy")

    def test_make_observation_includes_base_rate(self):
        """_make_observation must pass base_rate through to Observation."""
        adapter = self._make_adapter()
        ctx = _ctx(reading=25.0)
        obs = adapter._make_observation(ctx, b=0.8, d=0.1, u=0.1, a=0.7)
        assert isinstance(obs, Observation)
        assert obs.belief == 0.8
        assert obs.disbelief == 0.1
        assert obs.uncertainty == 0.1
        assert obs.base_rate == 0.7

    def test_make_observation_default_base_rate(self):
        """_make_observation with default base_rate=0.5."""
        adapter = self._make_adapter()
        ctx = _ctx(reading=25.0)
        obs = adapter._make_observation(ctx, b=0.8, d=0.1, u=0.1)
        assert obs.base_rate == 0.5


# ── GroundTruth tests ────────────────────────────────────────────────────────


class TestGroundTruth:
    """Tests for GroundTruth.expected_safe property."""

    def test_attack_is_not_safe(self):
        gt = GroundTruth(is_attack=True, attack_type="injection")
        assert gt.expected_safe is False

    def test_normal_is_safe(self):
        gt = GroundTruth(is_attack=False)
        assert gt.expected_safe is True


# ── TimeWindow tests ─────────────────────────────────────────────────────────


class TestTimeWindow:
    """Tests for TimeWindow dataclass."""

    def test_to_dict(self):
        gt = GroundTruth(is_attack=True, attack_type="replay")
        tw = TimeWindow(
            observations=[],
            ground_truth=gt,
            start_time=NOW,
            end_time=NOW,
            window_id=1,
            dataset_name="test",
        )
        d = tw.to_dict()
        assert d["window_id"] == 1
        assert d["ground_truth_is_attack"] is True
        assert d["ground_truth_expected_safe"] is False

    def test_num_observations(self):
        gt = GroundTruth(is_attack=False)
        obs = Observation(
            payload={"val": 1.0},
            source=ObservationSource(agent_id="s1"),
            timestamp=NOW,
        )
        tw = TimeWindow(
            observations=[obs, obs],
            ground_truth=gt,
            start_time=NOW,
            end_time=NOW,
        )
        assert tw.num_observations == 2



# ═══════════════════════════════════════════════════════════════════════════
# TDD: Strategy applicability signaling (fix for composite weighting bug)
# ═══════════════════════════════════════════════════════════════════════════


class TestStrategyApplicability:
    """Tests for the is_applicable() method that strategies must provide.

    A strategy is 'applicable' when it has sufficient context to produce
    an informative opinion. When not applicable, it should signal this so
    CompositeStrategy can skip it and renormalize weights over the applicable
    strategies only.

    This prevents constant-noise fallbacks from dragging down the composite
    average when most sensors lack the required context (e.g., SensorAgreement
    returning (0.6, 0.1, 0.3) for sensors without related_readings).
    """

    def test_sensor_agreement_not_applicable_without_related(self):
        """SensorAgreementStrategy should report not-applicable when no
        related_readings are available (only returns constant fallback)."""
        strat = SensorAgreementStrategy()
        ctx = _ctx(related_readings=None)
        assert strat.is_applicable(ctx) is False, (
            "SensorAgreementStrategy must signal not-applicable without related "
            "readings — otherwise its constant (0.6, 0.1, 0.3) fallback drags down "
            "composite averages."
        )

    def test_sensor_agreement_applicable_with_related(self):
        """SensorAgreementStrategy IS applicable when related_readings exist."""
        strat = SensorAgreementStrategy()
        ctx = _ctx(related_readings={"other_sensor": 25.1})
        assert strat.is_applicable(ctx) is True

    def test_historical_not_applicable_without_stats(self):
        """HistoricalDeviationStrategy needs historical_mean and historical_std."""
        strat = HistoricalDeviationStrategy()
        ctx = _ctx(historical_mean=None, historical_std=None)
        assert strat.is_applicable(ctx) is False

    def test_historical_applicable_with_stats(self):
        strat = HistoricalDeviationStrategy()
        ctx = _ctx(historical_mean=25.0, historical_std=2.0)
        assert strat.is_applicable(ctx) is True

    def test_historical_not_applicable_with_zero_std(self):
        """Zero std means no variance in calibration data — strategy can still
        produce a meaningful opinion (exact match vs deviation), so IS applicable."""
        strat = HistoricalDeviationStrategy()
        ctx = _ctx(historical_mean=25.0, historical_std=0.0)
        # Zero std is a valid-but-edge case: still produces meaningful output
        assert strat.is_applicable(ctx) is True

    def test_physics_not_applicable_without_bounds(self):
        """PhysicsBoundsStrategy needs physical_min and physical_max.

        NOTE: Current implementation falls back to HistoricalDeviation when
        bounds are missing. For applicability purposes, physics is NOT applicable
        if the physics-specific logic isn't used.
        """
        strat = PhysicsBoundsStrategy()
        ctx = _ctx(physical_min=None, physical_max=None)
        assert strat.is_applicable(ctx) is False

    def test_physics_applicable_with_bounds(self):
        strat = PhysicsBoundsStrategy()
        ctx = _ctx(physical_min=0.0, physical_max=100.0)
        assert strat.is_applicable(ctx) is True

    def test_physics_not_applicable_degenerate_bounds(self):
        """Degenerate bounds (min >= max) are not applicable."""
        strat = PhysicsBoundsStrategy()
        ctx = _ctx(physical_min=10.0, physical_max=10.0)
        assert strat.is_applicable(ctx) is False


class TestCompositeSkipsNonApplicable:
    """Tests verifying CompositeStrategy skips non-applicable strategies and
    renormalizes weights over those that apply."""

    def test_composite_uses_only_applicable_strategies(self):
        """When only historical is applicable, composite output must equal
        historical output (not an average with fallback noise)."""
        composite = CompositeStrategy(strategies=[
            (HistoricalDeviationStrategy(), 0.5),
            (PhysicsBoundsStrategy(), 0.3),
            (SensorAgreementStrategy(), 0.2),
        ])

        # Historical can work (has stats); others cannot (no bounds, no related)
        ctx = _ctx(
            reading=30.0,
            historical_mean=25.0,
            historical_std=2.0,
            physical_min=None,
            physical_max=None,
            related_readings=None,
        )

        # What historical alone would produce
        hist_only = HistoricalDeviationStrategy().assign(ctx)
        # What composite produces
        comp_result = composite.assign(ctx)

        # Should be essentially identical (composite = historical when others skip)
        assert comp_result[0] == pytest.approx(hist_only[0], abs=1e-4), (
            f"Composite should produce historical's output when only historical "
            f"is applicable. Got composite b={comp_result[0]:.4f}, "
            f"historical b={hist_only[0]:.4f}"
        )
        assert comp_result[1] == pytest.approx(hist_only[1], abs=1e-4)
        assert comp_result[2] == pytest.approx(hist_only[2], abs=1e-4)

    def test_composite_fallback_to_vacuous_when_nothing_applicable(self):
        """When NO strategies are applicable, composite returns vacuous (0,0,1)."""
        composite = CompositeStrategy(strategies=[
            (HistoricalDeviationStrategy(), 0.5),
            (PhysicsBoundsStrategy(), 0.3),
            (SensorAgreementStrategy(), 0.2),
        ])

        # Nothing applicable: no history, no bounds, no related
        ctx = _ctx(
            historical_mean=None,
            historical_std=None,
            physical_min=None,
            physical_max=None,
            related_readings=None,
        )
        b, d, u, a = composite.assign(ctx)

        assert b == pytest.approx(0.0, abs=1e-6)
        assert d == pytest.approx(0.0, abs=1e-6)
        assert u == pytest.approx(1.0, abs=1e-6)

    def test_composite_weights_renormalize_over_applicable(self):
        """When only 2 of 3 strategies are applicable, their weights must
        renormalize to sum to 1.0 (not stay at original fraction)."""
        composite = CompositeStrategy(strategies=[
            (HistoricalDeviationStrategy(), 0.5),
            (PhysicsBoundsStrategy(), 0.3),
            (SensorAgreementStrategy(), 0.2),
        ])

        # Historical and Physics applicable; Agreement is not
        ctx = _ctx(
            reading=50.0,
            historical_mean=50.0,
            historical_std=5.0,
            physical_min=0.0,
            physical_max=100.0,
            related_readings=None,
        )

        hist = HistoricalDeviationStrategy().assign(ctx)
        phys = PhysicsBoundsStrategy().assign(ctx)
        comp = composite.assign(ctx)

        # Composite should weight historical by 0.5/(0.5+0.3)=0.625 and physics by 0.375
        expected_b = hist[0] * (0.5 / 0.8) + phys[0] * (0.3 / 0.8)
        assert comp[0] == pytest.approx(expected_b, abs=1e-4), (
            f"Weights should renormalize over applicable strategies. "
            f"Expected b={expected_b:.4f}, got {comp[0]:.4f}"
        )

    def test_composite_backward_compat_all_applicable(self):
        """When ALL strategies are applicable, composite behaves as before
        (weighted average over all three)."""
        composite = CompositeStrategy(strategies=[
            (HistoricalDeviationStrategy(), 1.0),
            (PhysicsBoundsStrategy(), 1.0),
            (SensorAgreementStrategy(), 1.0),
        ])

        ctx = _ctx(
            reading=25.0,
            historical_mean=25.0,
            historical_std=2.0,
            physical_min=0.0,
            physical_max=100.0,
            related_readings={"other": 25.2},
        )

        hist = HistoricalDeviationStrategy().assign(ctx)
        phys = PhysicsBoundsStrategy().assign(ctx)
        agr = SensorAgreementStrategy().assign(ctx)
        comp = composite.assign(ctx)

        expected_b = (hist[0] + phys[0] + agr[0]) / 3
        assert comp[0] == pytest.approx(expected_b, abs=1e-4)

    def test_composite_is_applicable_if_any_substrategy_is(self):
        """Composite itself is applicable if at least one sub-strategy is."""
        composite = CompositeStrategy(strategies=[
            (HistoricalDeviationStrategy(), 0.5),
            (PhysicsBoundsStrategy(), 0.3),
            (SensorAgreementStrategy(), 0.2),
        ])

        # Only historical applicable
        ctx_hist = _ctx(historical_mean=25.0, historical_std=2.0,
                        physical_min=None, physical_max=None,
                        related_readings=None)
        assert composite.is_applicable(ctx_hist) is True

        # Nothing applicable
        ctx_none = _ctx(historical_mean=None, historical_std=None,
                        physical_min=None, physical_max=None,
                        related_readings=None)
        assert composite.is_applicable(ctx_none) is False
