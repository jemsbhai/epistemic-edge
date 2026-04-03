"""
Subjective Logic fusion via jsonld-ex compliance algebra.

Wraps jsonld_ex.confidence_algebra to resolve conflicting sensor readings
into a single (b, d, u) opinion with deterministic conflict handling.
"""

from __future__ import annotations

import logging
from typing import Any

from epistemic_edge.models import FusedState, Observation, StateGraph

logger = logging.getLogger(__name__)


class SLFusion:
    """Fuses incoming observations using jsonld-ex Subjective Logic operations."""

    def fuse_observation(self, obs: Observation, state: StateGraph) -> FusedState:
        """
        Fuse a single observation into the state graph.

        If the observation carries its own SL bounds, those are used directly.
        Otherwise, a vacuous opinion (b=0, d=0, u=1) is assigned.

        Design decision — vacuous default for missing uncertainty:
            When a sensor provides belief but NOT uncertainty, we default u=1.0
            (maximum uncertainty) rather than computing u = 1 - b - d (which
            would preserve the sensor's stated confidence). This is deliberate:
            in a safety-critical actuation pipeline, a sensor that cannot report
            its own uncertainty should not be treated as confident. The vacuous
            default ensures the guardrail layer can block actuation until a
            properly calibrated sensor provides a complete (b, d, u) triple.
            See Jøsang (2016), Subjective Logic, §3.2: "The vacuous opinion
            represents total uncertainty."
        """
        b = obs.belief if obs.belief is not None else 0.0
        d = obs.disbelief if obs.disbelief is not None else 0.0
        u = obs.uncertainty if obs.uncertainty is not None else 1.0

        # Normalize if the sensor provided raw values that don't sum to 1
        total = b + d + u
        if total > 0 and abs(total - 1.0) > 1e-9:
            b, d, u = b / total, d / total, u / total

        return FusedState(
            payload=obs.payload,
            belief=b,
            disbelief=d,
            uncertainty=u,
            sources=[obs.source.agent_id],
        )

    def fuse_pair(self, a: FusedState, b: FusedState) -> FusedState:
        """
        Combine two fused states using cumulative fusion (Josang).

        Delegates to jsonld-ex when available, falls back to direct calculation.
        """
        try:
            from jsonld_ex.confidence_algebra import cumulative_fusion

            result = cumulative_fusion(
                {"belief": a.belief, "disbelief": a.disbelief, "uncertainty": a.uncertainty},
                {"belief": b.belief, "disbelief": b.disbelief, "uncertainty": b.uncertainty},
            )
            return FusedState(
                payload={**a.payload, **b.payload},
                belief=result["belief"],
                disbelief=result["disbelief"],
                uncertainty=result["uncertainty"],
                sources=a.sources + b.sources,
            )
        except ImportError:
            logger.warning("jsonld-ex cumulative_fusion not available; using simple average.")
            return FusedState(
                payload={**a.payload, **b.payload},
                belief=(a.belief + b.belief) / 2,
                disbelief=(a.disbelief + b.disbelief) / 2,
                uncertainty=(a.uncertainty + b.uncertainty) / 2,
                sources=a.sources + b.sources,
            )
