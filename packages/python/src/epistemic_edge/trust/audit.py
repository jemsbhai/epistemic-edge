"""
PROV-O audit trail via jsonld-ex owl_interop.

Logs every fusion event and actuation as a W3C PROV-O Activity,
creating a deterministic, queryable lineage graph for XAI compliance.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from epistemic_edge.models import EdgeIntent, FusedState, Observation

logger = logging.getLogger(__name__)


class PROVOActivity:
    """A single logged PROV-O activity in the audit trail."""

    def __init__(
        self,
        activity_type: str,
        entity_ids: list[str],
        agent_ids: list[str],
        timestamp: datetime | None = None,
    ) -> None:
        self.id = f"urn:epistemic-edge:activity:{uuid.uuid4().hex[:12]}"
        self.activity_type = activity_type
        self.entity_ids = entity_ids
        self.agent_ids = agent_ids
        self.timestamp = timestamp or datetime.utcnow()

    def to_jsonld(self) -> dict[str, Any]:
        """Serialize as a JSON-LD fragment compatible with PROV-O."""
        return {
            "@type": "prov:Activity",
            "@id": self.id,
            "prov:startedAtTime": self.timestamp.isoformat(),
            "prov:used": [{"@id": eid} for eid in self.entity_ids],
            "prov:wasAssociatedWith": [{"@id": aid} for aid in self.agent_ids],
            "ee:activityType": self.activity_type,
        }


class PROVOAudit:
    """
    Maintains an in-memory PROV-O audit trail for the EdgeNode.

    Every sensor fusion and actuation event is logged as a PROV-O Activity
    with references to the Entities (data) and Agents (sensors/LLM) involved.
    """

    def __init__(self) -> None:
        self.activities: list[PROVOActivity] = []

    def log_fusion(self, observation: Observation, fused: FusedState) -> str:
        """Log a sensor fusion event. Returns the activity ID."""
        activity = PROVOActivity(
            activity_type="sensor_fusion",
            entity_ids=[f"urn:sensor-reading:{observation.source.agent_id}"],
            agent_ids=[observation.source.agent_id],
        )
        self.activities.append(activity)
        fused.prov_o_activity_id = activity.id
        logger.debug("PROV-O logged fusion: %s", activity.id)
        return activity.id

    def log_actuation(self, intent: EdgeIntent) -> str:
        """Log an actuation event. Returns the activity ID."""
        activity = PROVOActivity(
            activity_type="actuation",
            entity_ids=[f"urn:intent:{intent.action}:{intent.target}"],
            agent_ids=["epistemic-edge:cognition-engine"],
        )
        self.activities.append(activity)
        logger.debug("PROV-O logged actuation: %s", activity.id)
        return activity.id

    def export_graph(self) -> dict[str, Any]:
        """Export the full audit trail as a JSON-LD document."""
        return {
            "@context": {
                "prov": "http://www.w3.org/ns/prov#",
                "ee": "https://epistemic-edge.io/ns#",
            },
            "@graph": [a.to_jsonld() for a in self.activities],
        }

    @property
    def count(self) -> int:
        return len(self.activities)
