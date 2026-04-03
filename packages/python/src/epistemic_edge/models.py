"""Core data models for the Epistemic Edge framework."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Timezone-aware UTC now, avoiding deprecated datetime.utcnow()."""
    return datetime.now(timezone.utc)


class TierLabel(str, Enum):
    """Identifies which architectural tier produced or owns a piece of state."""

    TRANSPORT = "transport"
    TRUST = "trust"
    MEMORY = "memory"
    COGNITION = "cognition"


class ObservationSource(BaseModel):
    """Metadata about the hardware or software agent that produced an observation."""

    agent_id: str
    agent_type: str = "sensor"
    calibration_confidence: float | None = None
    prov_o_entity_id: str | None = None


class Observation(BaseModel):
    """
    A single sensor or telemetry reading entering the pipeline.

    Carries optional Subjective Logic bounds (belief, disbelief, uncertainty)
    as produced by cbor-ld-ex deserialization or sensor firmware.
    """

    payload: dict[str, Any]
    source: ObservationSource
    timestamp: datetime = Field(default_factory=_utcnow)
    belief: float | None = None
    disbelief: float | None = None
    uncertainty: float | None = None


class FusedState(BaseModel):
    """
    The result of jsonld-ex compliance algebra fusion across one or more observations.

    The (b, d, u) triple here is the *fused* opinion after conflict resolution.
    """

    payload: dict[str, Any]
    belief: float
    disbelief: float
    uncertainty: float
    sources: list[str] = Field(default_factory=list)
    prov_o_activity_id: str | None = None
    fused_at: datetime = Field(default_factory=_utcnow)

    @property
    def expected_value(self) -> float:
        """Projected probability (b + a*u) assuming default base rate a=0.5."""
        return self.belief + 0.5 * self.uncertainty


class StateGraph(BaseModel):
    """
    The aggregate world-state maintained by an EdgeNode.

    Contains temporally valid, trust-verified facts ready for LLM consumption.
    """

    node_id: str
    facts: list[FusedState] = Field(default_factory=list)
    stale: list[FusedState] = Field(default_factory=list)
    last_sweep: datetime | None = None

    @property
    def active_count(self) -> int:
        return len(self.facts)

    def max_uncertainty(self) -> float:
        """Return the highest uncertainty across all active facts."""
        if not self.facts:
            return 1.0
        return max(f.uncertainty for f in self.facts)


class EdgeIntent(BaseModel):
    """
    An actuation intent produced by the cognition layer.

    Must pass through the guardrail check before physical execution.
    """

    action: str
    target: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=_utcnow)
    raw_llm_output: str | None = None
    grammar_constrained: bool = False


class ActuationResult(BaseModel):
    """Outcome of an actuation attempt after guardrail verification."""

    intent: EdgeIntent
    permitted: bool
    reason: str = ""
    executed_at: datetime | None = None
    prov_o_receipt_id: str | None = None
