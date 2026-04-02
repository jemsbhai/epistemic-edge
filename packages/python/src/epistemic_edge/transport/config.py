"""Transport configuration models."""

from __future__ import annotations

from pydantic import BaseModel


class MQTTConfig(BaseModel):
    """Configuration for the MQTT transport listener."""

    broker: str = "mqtt://localhost:1883"
    topic: str = "sensors/+/cborld"
    client_id: str | None = None
    keepalive: int = 60


class CoAPConfig(BaseModel):
    """Configuration for the CoAP transport listener."""

    bind: str = "0.0.0.0"
    port: int = 5683
    resource_path: str = "sensors"
