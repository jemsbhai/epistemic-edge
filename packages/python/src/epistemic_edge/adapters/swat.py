"""SWaT (Secure Water Treatment) dataset adapter.

Maps the iTrust SWaT testbed data to epistemic-edge Observations.

Dataset structure:
  - 51 sensor/actuator variables across 6 water treatment stages
  - 7 days normal operation + 4 days with 36 labeled physical attacks
  - 1-second sampling interval
  - Includes: flow meters, level sensors, pressure sensors, pH, conductivity
  - Attack types: single-point, multi-point, coordinated cyber-physical

Reference:
  Mathur & Tippenhauer, "SWaT: a Water Treatment Testbed for Research
  and Training on ICS Security." CySWater 2016.

Access: Request from https://itrust.sutd.edu.sg/itrust-labs_datasets/
  (up to 3 business days for approval)
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
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

# SWaT sensor columns by process stage
# P1: Raw water intake
P1_SENSORS = {
    "FIT101": {"type": "flow", "unit": "m3/h", "bounds": (0.0, 5.0)},
    "LIT101": {"type": "level", "unit": "mm", "bounds": (0.0, 1200.0)},
    "MV101": {"type": "valve", "unit": "state", "bounds": (0.0, 2.0)},
    "P101": {"type": "pump", "unit": "state", "bounds": (0.0, 2.0)},
    "P102": {"type": "pump", "unit": "state", "bounds": (0.0, 2.0)},
}

# P2: Chemical dosing
P2_SENSORS = {
    "AIT201": {"type": "conductivity", "unit": "uS/cm", "bounds": (100.0, 400.0)},
    "AIT202": {"type": "pH", "unit": "pH", "bounds": (6.0, 9.0)},
    "AIT203": {"type": "ORP", "unit": "mV", "bounds": (0.0, 800.0)},
    "FIT201": {"type": "flow", "unit": "m3/h", "bounds": (0.0, 5.0)},
}

# P3: Ultrafiltration
P3_SENSORS = {
    "DPIT301": {"type": "pressure_diff", "unit": "kPa", "bounds": (0.0, 100.0)},
    "FIT301": {"type": "flow", "unit": "m3/h", "bounds": (0.0, 5.0)},
    "LIT301": {"type": "level", "unit": "mm", "bounds": (0.0, 1200.0)},
}

# P4: Dechlorination
P4_SENSORS = {
    "AIT401": {"type": "hardness", "unit": "mg/L", "bounds": (0.0, 500.0)},
    "AIT402": {"type": "ORP", "unit": "mV", "bounds": (0.0, 800.0)},
    "FIT401": {"type": "flow", "unit": "m3/h", "bounds": (0.0, 5.0)},
    "LIT401": {"type": "level", "unit": "mm", "bounds": (0.0, 1200.0)},
}

# P5: Reverse osmosis
P5_SENSORS = {
    "AIT501": {"type": "conductivity", "unit": "uS/cm", "bounds": (100.0, 400.0)},
    "AIT502": {"type": "conductivity", "unit": "uS/cm", "bounds": (0.0, 50.0)},
    "AIT503": {"type": "ORP", "unit": "mV", "bounds": (0.0, 800.0)},
    "AIT504": {"type": "ORP", "unit": "mV", "bounds": (0.0, 800.0)},
    "FIT501": {"type": "flow", "unit": "m3/h", "bounds": (0.0, 5.0)},
    "FIT502": {"type": "flow", "unit": "m3/h", "bounds": (0.0, 5.0)},
    "FIT503": {"type": "flow", "unit": "m3/h", "bounds": (0.0, 5.0)},
    "FIT504": {"type": "flow", "unit": "m3/h", "bounds": (0.0, 5.0)},
    "PIT501": {"type": "pressure", "unit": "kPa", "bounds": (0.0, 1000.0)},
    "PIT502": {"type": "pressure", "unit": "kPa", "bounds": (0.0, 1000.0)},
    "PIT503": {"type": "pressure", "unit": "kPa", "bounds": (0.0, 500.0)},
}

# P6: Backwash/Cleaning
P6_SENSORS = {
    "FIT601": {"type": "flow", "unit": "m3/h", "bounds": (0.0, 5.0)},
}

# Merge all sensor definitions
ALL_SENSORS = {**P1_SENSORS, **P2_SENSORS, **P3_SENSORS,
               **P4_SENSORS, **P5_SENSORS, **P6_SENSORS}

# Related sensor groups for agreement strategy
RELATED_GROUPS = [
    ["FIT101", "FIT201", "FIT301"],  # Flow through P1->P2->P3
    ["FIT401", "FIT501"],             # Flow through P4->P5
    ["LIT101", "LIT301", "LIT401"],   # Tank levels across stages
    ["AIT201", "AIT501"],             # Conductivity sensors
    ["AIT203", "AIT402"],             # ORP sensors
    ["PIT501", "PIT502", "PIT503"],   # RO pressures
]


class SWaTAdapter(DatasetAdapter):
    """Adapter for the SWaT water treatment dataset.

    Loads the SWaT A1&A2 Dec 2015 dataset (or later versions).
    Expects CSV files with columns: Timestamp, sensor columns, Normal/Attack label.
    """

    def __init__(self, strategy: Any = None) -> None:
        super().__init__(strategy=strategy)
        self._data: list[dict[str, Any]] = []
        self._sensor_columns: list[str] = []
        self._historical: dict[str, dict[str, float]] = {}
        self._related_map: dict[str, list[str]] = {}

    def load(self, path: Path, split: str = "test") -> None:
        """Load SWaT dataset.

        Args:
            path: Path to directory containing SWaT CSVs or a single CSV.
                Expected: SWaT_Dataset_Normal_v1.csv and SWaT_Dataset_Attack_v0.csv
            split: "train" loads normal data, "test" loads attack data.
        """
        # TODO: Implement when SWaT data access is approved
        # Key implementation notes:
        # - Normal data: SWaT_Dataset_Normal_v1.csv (7 days, ~496,800 rows at 1Hz)
        # - Attack data: SWaT_Dataset_Attack_v0.csv (4 days, ~449,919 rows)
        # - Label column: "Normal/Attack" with values "Normal" or "Attack"
        # - Some versions have "A ttack" (space) due to CSV formatting
        # - Timestamp format: varies by version
        # - 51 columns total; filter to sensor columns (exclude actuator commands
        #   unless we want to track actuator state too)
        raise NotImplementedError(
            "SWaT adapter pending data access approval from iTrust SUTD. "
            "Submit request at: https://itrust.sutd.edu.sg/itrust-labs_datasets/"
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
