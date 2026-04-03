"""
Temporal cache: wraps chronofy's TLDA to manage EdgeNode state validity.

Acts as an epistemic garbage collector, decaying stale facts out of the
active context window before the 1-bit LLM ever sees them.
"""

from __future__ import annotations

import logging
from datetime import datetime

from pydantic import BaseModel

from epistemic_edge.models import FusedState

logger = logging.getLogger(__name__)


class DecayConfig(BaseModel):
    """Configuration for temporal decay behavior."""

    mean_reversion_rate: float = 1.0
    threshold: float = 0.2
    decay_type: str = "exponential"


class TemporalCache:
    """
    Manages temporal validity of FusedState facts via chronofy.

    Wraps chronofy.EpistemicFilter to partition facts into fresh (above
    threshold) and stale (below threshold) sets based on their age and
    the configured decay function.
    """

    def __init__(self, config: DecayConfig | None = None) -> None:
        self.config = config or DecayConfig()
        self._filter = self._build_filter()

    def _build_filter(self) -> object:
        """Construct the chronofy EpistemicFilter from config."""
        try:
            from chronofy import ExponentialDecay, HalfLifeDecay, LinearDecay, EpistemicFilter

            decay_map = {
                "exponential": ExponentialDecay,
                "linear": LinearDecay,
                "half_life": HalfLifeDecay,
            }
            decay_cls = decay_map.get(self.config.decay_type, ExponentialDecay)

            if self.config.decay_type == "exponential":
                decay_fn = decay_cls.from_mean_reversion_rate(
                    kappa={"default": self.config.mean_reversion_rate}
                )
            else:
                decay_fn = decay_cls()

            return EpistemicFilter(decay_fn=decay_fn, threshold=self.config.threshold)

        except ImportError:
            logger.warning(
                "chronofy not available; TemporalCache will use age-based fallback."
            )
            return None

    def partition(
        self,
        facts: list[FusedState],
        query_time: datetime,
    ) -> tuple[list[FusedState], list[FusedState]]:
        """
        Split facts into (fresh, stale) based on temporal validity.

        If chronofy is available, delegates to EpistemicFilter.
        Otherwise, uses a simple age-based cutoff as fallback.
        """
        if self._filter is not None:
            return self._partition_chronofy(facts, query_time)
        return self._partition_fallback(facts, query_time)

    def _partition_chronofy(
        self,
        facts: list[FusedState],
        query_time: datetime,
    ) -> tuple[list[FusedState], list[FusedState]]:
        """Partition using chronofy EpistemicFilter.partition() directly."""
        from chronofy import TemporalFact

        if not facts:
            return [], []

        # Build TemporalFacts and maintain index mapping back to FusedState
        temporal_facts: list[TemporalFact] = []
        fact_map: dict[int, FusedState] = {}

        for i, fact in enumerate(facts):
            tf = TemporalFact(
                content=str(fact.payload),
                timestamp=fact.fused_at,
                fact_type="default",
                source_quality=fact.expected_value,
            )
            temporal_facts.append(tf)
            fact_map[id(tf)] = fact

        # Use the filter's own partition method (handles _decay_fn internally)
        valid_tfs, expired_tfs = self._filter.partition(temporal_facts, query_time)

        valid_ids = {id(tf) for tf in valid_tfs}
        expired_ids = {id(tf) for tf in expired_tfs}

        fresh = [fact_map[tid] for tid in valid_ids if tid in fact_map]
        stale = [fact_map[tid] for tid in expired_ids if tid in fact_map]

        return fresh, stale

    def _partition_fallback(
        self,
        facts: list[FusedState],
        query_time: datetime,
    ) -> tuple[list[FusedState], list[FusedState]]:
        """Simple age-based fallback when chronofy is not installed."""
        import math

        fresh: list[FusedState] = []
        stale: list[FusedState] = []

        for fact in facts:
            age_seconds = (query_time - fact.fused_at).total_seconds()
            score = math.exp(-self.config.mean_reversion_rate * age_seconds / 3600)
            if score >= self.config.threshold:
                fresh.append(fact)
            else:
                stale.append(fact)

        return fresh, stale
