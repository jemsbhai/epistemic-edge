"""BATADAL dataset adapter.

Maps the BATtle of the Attack Detection ALgorithms dataset
(water distribution network, C-Town) to epistemic-edge Observations.

Dataset structure:
  - 43 sensor/actuator variables (tank levels, flows, pressures, valve states)
  - Training set 1: ~1 year of normal operations
  - Training set 2: ~6 months with labeled attacks (ATT_FLAG column)
  - Test set: unlabeled (used in competition, labels available separately)

Reference:
  Taormina et al., "The Battle of the Attack Detection Algorithms:
  Disclosing Cyber Attacks on Water Distribution Networks."
  Journal of Water Resources Planning and Management, 144(8), 2018.

Download: https://www.batadal.net/ or Kaggle
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
from epistemic_edge.models import Observation

logger = logging.getLogger(__name__)

# Known sensor columns in BATADAL (C-Town network)
# Tank levels (continuous, have physical bounds)
TANK_LEVEL_COLUMNS = [f"L_T{i}" for i in range(1, 8)]  # L_T1 .. L_T7

# Pump flow rates (continuous)
PUMP_FLOW_COLUMNS = [f"F_PU{i}" for i in range(1, 12)]  # F_PU1 .. F_PU11

# Junction pressures (continuous)
# The exact column names vary by dataset version; we detect them dynamically
PRESSURE_PREFIX = "P_J"

# Valve status columns (binary/categorical)
VALVE_PREFIX = "S_"

# Attack flag column
ATTACK_COLUMN = "ATT_FLAG"

# Known physical bounds for C-Town tanks (meters, approximate)
TANK_BOUNDS = {
    "L_T1": (0.0, 6.0),
    "L_T2": (0.0, 6.5),
    "L_T3": (0.0, 4.5),
    "L_T4": (0.0, 5.5),
    "L_T5": (0.0, 5.0),
    "L_T6": (0.0, 6.0),
    "L_T7": (0.0, 5.5),
}

# Related sensor groups (sensors that should agree or correlate)
RELATED_GROUPS = [
    # Tank levels that are hydraulically connected
    ["L_T1", "L_T2"],  # Upstream tanks
    ["L_T3", "L_T4"],  # Midstream
    ["L_T5", "L_T6", "L_T7"],  # Downstream
]


class BATADALAdapter(DatasetAdapter):
    """Adapter for the BATADAL water distribution dataset."""

    def __init__(self, strategy: Any = None) -> None:
        super().__init__(strategy=strategy)
        self._data: list[dict[str, Any]] = []
        self._sensor_columns: list[str] = []
        self._historical: dict[str, dict[str, float]] = {}
        self._related_map: dict[str, list[str]] = {}

    def load(self, path: Path, split: str = "test") -> None:
        """Load BATADAL CSV data.

        Args:
            path: Path to directory containing BATADAL CSVs, or direct path
                to a single CSV file. Expected files:
                - BATADAL_dataset03.csv (training, no attacks)
                - BATADAL_dataset04.csv (training, with attacks)
                - Or a single CSV with ATT_FLAG column.
            split: "train" loads attack-free data for calibration.
                   "test" loads attack data for evaluation.
        """
        path = Path(path)

        if path.is_file():
            self._load_csv(path)
        elif path.is_dir():
            # Look for standard BATADAL filenames
            if split == "train":
                candidates = ["BATADAL_dataset03.csv", "BATADAL_training1.csv"]
            else:
                candidates = [
                    "BATADAL_dataset04.csv",
                    "BATADAL_training2.csv",
                    "BATADAL_test.csv",
                ]
            for name in candidates:
                csv_path = path / name
                if csv_path.exists():
                    self._load_csv(csv_path)
                    break
            else:
                # Try any CSV in the directory
                csvs = list(path.glob("*.csv"))
                if csvs:
                    self._load_csv(csvs[0])
                else:
                    raise FileNotFoundError(
                        f"No BATADAL CSV found in {path}. "
                        f"Expected one of: {candidates}"
                    )
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")

        # Detect sensor columns
        if self._data:
            all_cols = list(self._data[0].keys())
            self._sensor_columns = [
                c for c in all_cols
                if c.startswith(("L_T", "F_PU", "P_J", "S_PU", "S_V"))
                and c != ATTACK_COLUMN
            ]

        # Build related sensor map
        self._related_map = {}
        for group in RELATED_GROUPS:
            valid = [s for s in group if s in self._sensor_columns]
            for sensor in valid:
                self._related_map[sensor] = [s for s in valid if s != sensor]

        # Compute historical statistics from the loaded data
        self._compute_historical()

        # Count attacks
        num_attack = sum(
            1 for row in self._data if self._is_attack(row)
        )

        self._metadata = DatasetMetadata(
            name="BATADAL",
            description="BATtle of the Attack Detection ALgorithms — C-Town water distribution",
            num_sensors=len(self._sensor_columns),
            num_samples=len(self._data),
            num_attack_windows=num_attack,
            num_normal_windows=len(self._data) - num_attack,
            sampling_rate_hz=1 / 3600,  # Hourly in some versions, 15-min in others
            duration_hours=len(self._data),  # Approximate
            sensor_ids=self._sensor_columns,
            source_url="https://www.batadal.net/",
            citation=(
                "Taormina et al., JWRPM, 144(8), 2018. "
                "DOI: 10.1061/(ASCE)WR.1943-5452.0000969"
            ),
        )

        self._loaded = True
        logger.info(
            "Loaded BATADAL: %d rows, %d sensors, %d attack rows",
            len(self._data),
            len(self._sensor_columns),
            num_attack,
        )

    def _load_csv(self, path: Path) -> None:
        """Load a single CSV file."""
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            self._data = []
            for row in reader:
                # Normalize column names (strip whitespace)
                clean = {k.strip(): v.strip() for k, v in row.items()}
                # Convert numeric columns
                parsed: dict[str, Any] = {}
                for k, v in clean.items():
                    try:
                        parsed[k] = float(v)
                    except (ValueError, TypeError):
                        parsed[k] = v
                self._data.append(parsed)

    def _compute_historical(self) -> None:
        """Compute per-sensor mean and std from loaded data."""
        if not self._data or not self._sensor_columns:
            return

        for col in self._sensor_columns:
            values = []
            for row in self._data:
                val = row.get(col)
                if isinstance(val, (int, float)):
                    values.append(float(val))

            if len(values) > 1:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                std = variance ** 0.5
                self._historical[col] = {
                    "mean": mean,
                    "std": std,
                    "min": min(values),
                    "max": max(values),
                }

    def _is_attack(self, row: dict[str, Any]) -> bool:
        """Check if a row is labeled as an attack."""
        flag = row.get(ATTACK_COLUMN, row.get("ATT_FLAG", 0))
        try:
            return int(float(flag)) == 1
        except (ValueError, TypeError):
            return False

    def _parse_timestamp(self, row: dict[str, Any]) -> datetime:
        """Extract timestamp from a row."""
        for col in ["DATETIME", "datetime", "Timestamp", "timestamp", "DATE_TIME"]:
            if col in row:
                val = row[col]
                if isinstance(val, str):
                    # Try common formats
                    for fmt in [
                        "%d/%m/%y %H",
                        "%d/%m/%Y %H:%M",
                        "%Y-%m-%d %H:%M:%S",
                        "%m/%d/%Y %H:%M",
                    ]:
                        try:
                            return datetime.strptime(val, fmt).replace(
                                tzinfo=timezone.utc
                            )
                        except ValueError:
                            continue
        # Fallback: use row index as seconds from epoch
        return datetime(2015, 12, 22, tzinfo=timezone.utc)

    def get_sensor_contexts(self, row_idx: int) -> list[SensorContext]:
        """Build SensorContext objects for a single timestep."""
        if row_idx >= len(self._data):
            return []

        row = self._data[row_idx]
        timestamp = self._parse_timestamp(row)
        contexts = []

        for col in self._sensor_columns:
            val = row.get(col)
            if not isinstance(val, (int, float)):
                continue

            reading = float(val)
            hist = self._historical.get(col, {})
            bounds = TANK_BOUNDS.get(col)

            # Build related readings
            related = {}
            for related_id in self._related_map.get(col, []):
                rel_val = row.get(related_id)
                if isinstance(rel_val, (int, float)):
                    related[related_id] = float(rel_val)

            ctx = SensorContext(
                sensor_id=col,
                reading=reading,
                timestamp=timestamp,
                historical_mean=hist.get("mean"),
                historical_std=hist.get("std"),
                historical_min=hist.get("min"),
                historical_max=hist.get("max"),
                physical_min=bounds[0] if bounds else None,
                physical_max=bounds[1] if bounds else None,
                related_readings=related,
                sensor_type="level" if col.startswith("L_") else
                            "flow" if col.startswith("F_") else
                            "pressure" if col.startswith("P_") else
                            "actuator",
                unit="m" if col.startswith("L_") else
                     "L/s" if col.startswith("F_") else
                     "m" if col.startswith("P_") else "",
            )
            contexts.append(ctx)

        return contexts

    def get_windows(
        self,
        window_size: int = 60,
        stride: int = 30,
        max_windows: int | None = None,
    ) -> Iterator[TimeWindow]:
        """Yield time windows of observations with ground truth."""
        if not self._loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        window_id = 0
        i = 0

        while i + window_size <= len(self._data):
            if max_windows is not None and window_id >= max_windows:
                break

            window_rows = self._data[i : i + window_size]

            # Build observations for this window
            observations = []
            for j in range(i, i + window_size):
                contexts = self.get_sensor_contexts(j)
                for ctx in contexts:
                    b, d, u = self._assign_bdu(ctx)
                    obs = self._make_observation(ctx, b, d, u)
                    observations.append(obs)

            # Ground truth: attack if ANY row in the window is attack
            any_attack = any(self._is_attack(row) for row in window_rows)
            attack_rows = [
                row for row in window_rows if self._is_attack(row)
            ]

            ground_truth = GroundTruth(
                is_attack=any_attack,
                attack_type="cyber_physical" if any_attack else None,
                attack_description=(
                    f"{len(attack_rows)}/{window_size} rows flagged"
                    if any_attack
                    else None
                ),
            )

            # Timestamps
            start_time = self._parse_timestamp(self._data[i])
            end_time = self._parse_timestamp(self._data[i + window_size - 1])

            # Build query based on sensor state
            query = self._build_query(window_rows)

            yield TimeWindow(
                observations=observations,
                ground_truth=ground_truth,
                start_time=start_time,
                end_time=end_time,
                window_id=window_id,
                query=query,
                dataset_name="BATADAL",
                dataset_split="test" if any_attack else "normal",
            )

            window_id += 1
            i += stride

    def _build_query(self, window_rows: list[dict[str, Any]]) -> str:
        """Build an LLM query from the last row in the window."""
        last_row = window_rows[-1]

        # Summarize key readings
        parts = []
        for col in TANK_LEVEL_COLUMNS:
            val = last_row.get(col)
            if isinstance(val, (int, float)):
                parts.append(f"{col}={val:.2f}m")

        sensor_summary = ", ".join(parts[:5])  # Keep prompt concise
        return (
            f"Water distribution network status: {sensor_summary}. "
            f"Based on the current sensor readings and their uncertainty, "
            f"what action should be taken?"
        )
