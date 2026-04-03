"""NASA Bearing dataset adapter.

Maps the NASA IMS (Intelligent Maintenance Systems) bearing dataset
to epistemic-edge Observations.

Dataset structure:
  - 4 bearings on a loaded shaft, run to failure
  - Each bearing has 2 accelerometers (x, y axis)
  - 20 kHz sampling in 1-second snapshots every 10 minutes
  - 3 test sets with different failure modes
  - Failure progression: normal -> degradation -> failure

This dataset tests the temporal decay tier specifically: as bearings
degrade, older "normal" readings become stale and should be decayed
so the LLM sees the degradation trajectory, not a mix of old-normal
and new-degraded readings.

Reference:
  Lee et al., "Intelligent Prognostics Tools and E-Maintenance."
  Computers in Industry, 57(6), 2006.

Download: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
  (publicly available, no registration required)
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

# Bearing sensor configuration
# 4 bearings, 2 channels each (accelerometer x and y)
BEARING_SENSORS = {
    "bearing1_ch1": {"type": "vibration", "unit": "g", "bounds": (-20.0, 20.0)},
    "bearing1_ch2": {"type": "vibration", "unit": "g", "bounds": (-20.0, 20.0)},
    "bearing2_ch1": {"type": "vibration", "unit": "g", "bounds": (-20.0, 20.0)},
    "bearing2_ch2": {"type": "vibration", "unit": "g", "bounds": (-20.0, 20.0)},
    "bearing3_ch1": {"type": "vibration", "unit": "g", "bounds": (-20.0, 20.0)},
    "bearing3_ch2": {"type": "vibration", "unit": "g", "bounds": (-20.0, 20.0)},
    "bearing4_ch1": {"type": "vibration", "unit": "g", "bounds": (-20.0, 20.0)},
    "bearing4_ch2": {"type": "vibration", "unit": "g", "bounds": (-20.0, 20.0)},
}

# Related sensor groups (channels on the same bearing)
RELATED_GROUPS = [
    ["bearing1_ch1", "bearing1_ch2"],
    ["bearing2_ch1", "bearing2_ch2"],
    ["bearing3_ch1", "bearing3_ch2"],
    ["bearing4_ch1", "bearing4_ch2"],
]

# Test sets and their known failure modes
TEST_SETS = {
    "1st_test": {
        "description": "Inner race failure on bearing 3, roller element failure on bearing 4",
        "failed_bearings": ["bearing3", "bearing4"],
        "duration_files": 2156,  # Number of snapshot files
    },
    "2nd_test": {
        "description": "Outer race failure on bearing 1",
        "failed_bearings": ["bearing1"],
        "duration_files": 984,
    },
    "3rd_test": {
        "description": "Outer race failure on bearing 3",
        "failed_bearings": ["bearing3"],
        "duration_files": 6324,
    },
}


class NASABearingAdapter(DatasetAdapter):
    """Adapter for the NASA IMS bearing degradation dataset.

    Loads vibration data from accelerometers and maps RMS amplitude
    to epistemic-edge Observations. The key feature tested is temporal
    decay: early normal readings should be decayed as the bearing degrades.
    """

    def __init__(self, strategy: Any = None) -> None:
        super().__init__(strategy=strategy)
        self._data: list[dict[str, Any]] = []
        self._historical: dict[str, dict[str, float]] = {}

    def load(self, path: Path, split: str = "test") -> None:
        """Load NASA Bearing dataset.

        Args:
            path: Path to a test set directory (e.g., "1st_test/").
                Each file is a snapshot with 8 columns (4 bearings x 2 channels),
                20,480 samples at 20 kHz.
            split: "train" uses first 20% of files (assumed normal).
                   "test" uses all files for full degradation evaluation.
        """
        # TODO: Implement
        # Key implementation notes:
        # - Each file is tab-separated, no header, 20480 rows x 8 columns
        # - Filenames are timestamps (e.g., "2003.10.22.12.06.24")
        # - We compute RMS amplitude per channel per snapshot as the reading
        # - RMS increasing over time = degradation
        # - Ground truth: last N% of files are "failure" (no per-file labels,
        #   but failure mode is known and RMS threshold can be set empirically)
        raise NotImplementedError(
            "NASA Bearing adapter — download from: "
            "https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/"
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
