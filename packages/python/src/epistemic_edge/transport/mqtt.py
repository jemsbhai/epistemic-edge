"""
MQTT transport: async listener that deserializes cbor-ld-ex payloads.

Wraps aiomqtt and pipes incoming binary messages through
cbor_ld_ex.decode() to produce Observation objects.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from epistemic_edge.models import Observation
from epistemic_edge.transport.config import MQTTConfig

logger = logging.getLogger(__name__)


class MQTTListener:
    """Async MQTT subscriber that yields decoded Observations."""

    def __init__(self, config: MQTTConfig) -> None:
        self.config = config

    async def listen(self) -> AsyncIterator[Observation]:
        """
        Subscribe to the configured topic and yield Observations.

        Requires: pip install epistemic-edge[transport]
        """
        try:
            import aiomqtt  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "MQTT transport requires the 'transport' extra: "
                "pip install epistemic-edge[transport]"
            ) from exc

        try:
            from cbor_ld_ex import decode as cborld_decode
        except ImportError as exc:
            raise ImportError(
                "cbor-ld-ex is required for MQTT payload deserialization."
            ) from exc

        # TODO: parse broker URI properly
        async with aiomqtt.Client(self.config.broker) as client:
            await client.subscribe(self.config.topic)
            logger.info("MQTT subscribed to %s on %s", self.config.topic, self.config.broker)
            async for message in client.messages:
                try:
                    payload = cborld_decode(message.payload)
                    obs = Observation(
                        payload=payload,
                        source=payload.get("_source", {}),
                        belief=payload.get("sl:belief"),
                        disbelief=payload.get("sl:disbelief"),
                        uncertainty=payload.get("sl:uncertainty"),
                    )
                    yield obs
                except Exception:
                    logger.exception("Failed to decode MQTT message on %s", message.topic)
