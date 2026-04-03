"""Comprehensive tests for transport/config.py — MQTTConfig, CoAPConfig."""

import pytest

from epistemic_edge.transport.config import CoAPConfig, MQTTConfig


class TestMQTTConfig:
    def test_defaults(self) -> None:
        cfg = MQTTConfig()
        assert cfg.broker == "mqtt://localhost:1883"
        assert cfg.topic == "sensors/+/cborld"
        assert cfg.client_id is None
        assert cfg.keepalive == 60

    def test_custom(self) -> None:
        cfg = MQTTConfig(
            broker="mqtt://edge-hub.local:1884",
            topic="factory/line_3/#",
            client_id="gateway_alpha",
            keepalive=30,
        )
        assert cfg.broker == "mqtt://edge-hub.local:1884"
        assert cfg.topic == "factory/line_3/#"
        assert cfg.client_id == "gateway_alpha"
        assert cfg.keepalive == 30

    def test_serialization_roundtrip(self) -> None:
        cfg = MQTTConfig(broker="mqtt://test", topic="t", keepalive=10)
        data = cfg.model_dump()
        restored = MQTTConfig(**data)
        assert restored == cfg


class TestCoAPConfig:
    def test_defaults(self) -> None:
        cfg = CoAPConfig()
        assert cfg.bind == "0.0.0.0"
        assert cfg.port == 5683
        assert cfg.resource_path == "sensors"

    def test_custom(self) -> None:
        cfg = CoAPConfig(bind="192.168.1.100", port=5684, resource_path="telemetry/v2")
        assert cfg.bind == "192.168.1.100"
        assert cfg.port == 5684
        assert cfg.resource_path == "telemetry/v2"

    def test_serialization_roundtrip(self) -> None:
        cfg = CoAPConfig(port=9999)
        data = cfg.model_dump()
        restored = CoAPConfig(**data)
        assert restored == cfg
