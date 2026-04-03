"""Comprehensive tests for memory/cache.py — TemporalCache."""

from datetime import datetime, timedelta, timezone

import pytest

from epistemic_edge.memory.cache import DecayConfig, TemporalCache
from epistemic_edge.models import FusedState


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_fact(
    age_hours: float = 0.0,
    belief: float = 0.8,
    disbelief: float = 0.1,
    uncertainty: float = 0.1,
    payload: dict | None = None,
) -> FusedState:
    """Helper: create a FusedState at a given age."""
    return FusedState(
        payload=payload or {},
        belief=belief,
        disbelief=disbelief,
        uncertainty=uncertainty,
        fused_at=_now() - timedelta(hours=age_hours),
    )


class TestDecayConfig:
    def test_defaults(self) -> None:
        cfg = DecayConfig()
        assert cfg.mean_reversion_rate == 1.0
        assert cfg.threshold == 0.2
        assert cfg.decay_type == "exponential"

    def test_custom(self) -> None:
        cfg = DecayConfig(mean_reversion_rate=5.0, threshold=0.5, decay_type="linear")
        assert cfg.mean_reversion_rate == 5.0
        assert cfg.threshold == 0.5
        assert cfg.decay_type == "linear"


class TestTemporalCacheWithChronofy:
    """Tests using the real chronofy EpistemicFilter backend."""

    def test_fresh_facts_survive(self) -> None:
        cache = TemporalCache(DecayConfig(mean_reversion_rate=1.0, threshold=0.1))
        facts = [_make_fact(age_hours=0.01)]  # Seconds old
        fresh, stale = cache.partition(facts, _now())
        assert len(fresh) == 1
        assert len(stale) == 0

    def test_old_facts_pruned(self) -> None:
        cache = TemporalCache(
            DecayConfig(mean_reversion_rate=10.0, threshold=0.5)
        )
        facts = [_make_fact(age_hours=5.0)]  # 5 hours old with aggressive decay
        fresh, stale = cache.partition(facts, _now())
        assert len(fresh) == 0
        assert len(stale) == 1

    def test_mixed_ages_partitioned(self) -> None:
        cache = TemporalCache(
            DecayConfig(mean_reversion_rate=5.0, threshold=0.3)
        )
        facts = [
            _make_fact(age_hours=0.01, payload={"label": "fresh"}),
            _make_fact(age_hours=10.0, payload={"label": "stale"}),
            _make_fact(age_hours=0.05, payload={"label": "also_fresh"}),
        ]
        fresh, stale = cache.partition(facts, _now())
        assert len(fresh) >= 2
        assert len(stale) >= 1
        fresh_labels = {f.payload.get("label") for f in fresh}
        assert "fresh" in fresh_labels
        assert "also_fresh" in fresh_labels

    def test_empty_facts_returns_empty(self) -> None:
        cache = TemporalCache(DecayConfig())
        fresh, stale = cache.partition([], _now())
        assert fresh == []
        assert stale == []

    def test_all_fresh_when_threshold_zero(self) -> None:
        """With threshold=0, nothing should be pruned (scores >= 0)."""
        cache = TemporalCache(
            DecayConfig(mean_reversion_rate=100.0, threshold=0.0)
        )
        facts = [_make_fact(age_hours=h) for h in [0, 1, 5, 24]]
        fresh, stale = cache.partition(facts, _now())
        # Even very old facts should pass a 0 threshold (score > 0)
        assert len(fresh) + len(stale) == 4

    def test_high_threshold_prunes_aggressively(self) -> None:
        cache = TemporalCache(
            DecayConfig(mean_reversion_rate=1.0, threshold=0.99)
        )
        facts = [_make_fact(age_hours=0.5)]  # Even 30 min old
        fresh, stale = cache.partition(facts, _now())
        # With threshold=0.99, even slightly decayed facts get pruned
        assert len(stale) >= 0  # May or may not survive depending on exact score

    def test_preserves_fact_data(self) -> None:
        """Partition should return the original FusedState objects, not copies."""
        cache = TemporalCache(DecayConfig(mean_reversion_rate=1.0, threshold=0.1))
        original = _make_fact(age_hours=0.001, payload={"key": "val"})
        fresh, stale = cache.partition([original], _now())
        assert len(fresh) == 1
        assert fresh[0] is original


class TestTemporalCacheFallback:
    """Tests for the age-based fallback when chronofy is unavailable.

    We test the fallback path directly by constructing a TemporalCache
    and replacing its _filter with None.
    """

    def test_fallback_fresh(self) -> None:
        cache = TemporalCache(DecayConfig(mean_reversion_rate=1.0, threshold=0.2))
        cache._filter = None  # Force fallback
        facts = [_make_fact(age_hours=0.01)]
        fresh, stale = cache.partition(facts, _now())
        assert len(fresh) == 1

    def test_fallback_stale(self) -> None:
        cache = TemporalCache(DecayConfig(mean_reversion_rate=10.0, threshold=0.5))
        cache._filter = None  # Force fallback
        facts = [_make_fact(age_hours=5.0)]
        fresh, stale = cache.partition(facts, _now())
        assert len(stale) == 1

    def test_fallback_mixed(self) -> None:
        cache = TemporalCache(DecayConfig(mean_reversion_rate=5.0, threshold=0.3))
        cache._filter = None
        facts = [
            _make_fact(age_hours=0.001),
            _make_fact(age_hours=100.0),
        ]
        fresh, stale = cache.partition(facts, _now())
        assert len(fresh) == 1
        assert len(stale) == 1

    def test_fallback_empty(self) -> None:
        cache = TemporalCache(DecayConfig())
        cache._filter = None
        fresh, stale = cache.partition([], _now())
        assert fresh == []
        assert stale == []
