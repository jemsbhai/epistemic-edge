"""MQTT-IoT-IDS2020 dataset adapter.

Maps the MQTT-based IoT Intrusion Detection dataset to
epistemic-edge Observations.

Dataset structure:
  - MQTT protocol traffic from a real IoT network
  - Normal traffic + 5 attack types (brute force, flood, malformed,
    slowITE, DoS)
  - Features extracted from network packets: packet length, inter-arrival
    time, TCP flags, MQTT message types, etc.
  - Labeled per-packet: normal or attack type

This dataset tests the framework on IoT-native protocol data, which is
the closest to real edge deployment. The sensor readings here are network
telemetry metrics rather than physical process variables.

Reference:
  Vaccari et al., "MQTTset, a New Dataset for Machine Learning Techniques
  on MQTT." Sensors, 20(22), 2020.

Download: IEEE DataPort or GitHub
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

from epistemic_edge.adapters.base import (
    DatasetAdapter,
    DatasetMetadata,
    GroundTruth,
    SensorContext,
    TimeWindow,
)

logger = logging.getLogger(__name__)

# Network telemetry "sensors" extracted from MQTT traffic
NETWORK_SENSORS = {
    "pkt_length": {"type": "network", "unit": "bytes", "bounds": (0.0, 65535.0)},
    "inter_arrival_time": {"type": "network", "unit": "ms", "bounds": (0.0, 60000.0)},
    "tcp_flags": {"type": "network", "unit": "flags", "bounds": (0.0, 255.0)},
    "mqtt_msg_type": {"type": "protocol", "unit": "type", "bounds": (1.0, 14.0)},
    "mqtt_qos": {"type": "protocol", "unit": "level", "bounds": (0.0, 2.0)},
    "mqtt_retain": {"type": "protocol", "unit": "flag", "bounds": (0.0, 1.0)},
    "payload_size": {"type": "network", "unit": "bytes", "bounds": (0.0, 65535.0)},
    "conn_duration": {"type": "network", "unit": "s", "bounds": (0.0, 3600.0)},
}

# Attack types in the dataset
ATTACK_TYPES = [
    "bruteforce",
    "flood",
    "malformed",
    "slowITE",
    "DoS",
]

# Related sensor groups (network metrics that correlate)
RELATED_GROUPS = [
    ["pkt_length", "payload_size"],           # Size metrics
    ["inter_arrival_time", "conn_duration"],   # Timing metrics
    ["mqtt_msg_type", "mqtt_qos"],            # Protocol metrics
]


class MQTTIoTAdapter(DatasetAdapter):
    """Adapter for the MQTT-IoT-IDS2020 intrusion detection dataset.

    Maps network traffic features to sensor-like observations. This tests
    the framework on a fundamentally different data modality (network
    telemetry vs physical process variables) to demonstrate generalization.
    """

    def __init__(self, strategy: Any = None) -> None:
        super().__init__(strategy=strategy)
        self._data: list[dict[str, Any]] = []
        self._sensor_columns: list[str] = []
        self._historical: dict[str, dict[str, float]] = {}

    def load(self, path: Path, split: str = "test") -> None:
        """Load MQTT-IoT-IDS2020 dataset.

        Args:
            path: Path to directory containing CSV files or a single CSV.
                Expected: train70_reduced.csv, test30_reduced.csv,
                or the full pcap-derived CSVs.
            split: "train" loads normal-only data for calibration.
                   "test" loads mixed normal+attack data.
        """
        # TODO: Implement
        # Key implementation notes:
        # - CSV with ~33 features per packet/flow
        # - Label column: "target" with values: legitimate, bruteforce,
        #   flood, malformed, slowITE, DoS
        # - Some versions have pcap files that need tshark extraction
        # - We window by time (e.g., 1-second windows) and aggregate
        #   packet features within each window
        # - A window is "attack" if any packet in it is labeled attack
        # - For SensorContext: network metrics serve as "sensors"
        raise NotImplementedError(
            "MQTT-IoT-IDS2020 adapter — download from IEEE DataPort "
            "or GitHub: https://github.com/Avcu/MQTTset"
        )

    def get_sensor_contexts(self, row_idx: int) -> list[SensorContext]:
        raise NotImplementedError("Load data first")

    def get_windows(
        self,
        window_size: int = 60,
        stride: int = 30,
        max_windows: int | None = None,
    ) -> Iterator[TimeWindow]:
        raise NotImplementedError("Load data first")
