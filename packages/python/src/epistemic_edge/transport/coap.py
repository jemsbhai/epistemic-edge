"""
CoAP transport: async listener for constrained UDP-based networks.

Wraps aiocoap and pipes incoming payloads through cbor_ld_ex.decode()
to produce Observation objects. Ideal for LoRa/NB-IoT deep-edge sensors.
"""

from __future__ import annotations

import logging

from epistemic_edge.models import Observation
from epistemic_edge.transport.config import CoAPConfig

logger = logging.getLogger(__name__)


class CoAPListener:
    """Async CoAP resource that accepts cbor-ld-ex encoded observations."""

    def __init__(self, config: CoAPConfig) -> None:
        self.config = config

    async def start(self) -> None:
        """
        Bind the CoAP server and begin accepting requests.

        Requires: pip install epistemic-edge[transport]
        """
        try:
            import aiocoap  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "CoAP transport requires the 'transport' extra: "
                "pip install epistemic-edge[transport]"
            ) from exc

        # TODO: implement CoAP resource site with aiocoap.resource
        raise NotImplementedError("CoAP listener is not yet implemented.")
