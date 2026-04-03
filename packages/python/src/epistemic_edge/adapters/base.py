"""Base classes for dataset adapters.

Defines the abstract interface that all dataset adapters must implement,
plus shared data structures for time windows and ground truth.

Design principles:
  - Adapters produce Observation objects with b/d/u assigned by a
    pluggable UncertaintyStrategy — the strategy is NOT baked into
    the adapter, allowing systematic comparison across strategies.
  - Ground truth is always preserved alongside observations, enabling
    measurement of guardrail correctness against real labeled data.
  - Time windows support configurable size and stride for sensitivity
    analysis on temporal resolution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

from epistemic_edge.models import Observation, ObservationSource, _utcnow


# ── Shared Data Structures ──────────────────────────────────────────────────


class SensorContext(BaseModel):
    """Context for a single sensor, used by UncertaintyStrategy.

    Carries the information an uncertainty strategy needs to assign b/d/u
    to a raw reading: the reading itself, historical statistics, physical
    bounds, and cross-sensor readings for agreement checks.
    """

    sensor_id: str
    reading: float
    timestamp: datetime

    # Historical statistics (populated during adapter.load from training data)
    historical_mean: float | None = None
    historical_std: float | None = None
    historical_min: float | None = None
    historical_max: float | None = None

    # Physical bounds (from domain knowledge or dataset documentation)
    physical_min: float | None = None
    physical_max: float | None = None

    # Cross-sensor readings (for agreement strategy)
    # Maps related_sensor_id -> their reading at the same timestamp
    related_readings: dict[str, float] = Field(default_factory=dict)

    # Sensor metadata
    sensor_type: str = "analog"
    unit: str = ""


class GroundTruth(BaseModel):
    """Ground truth label for a time window.

    Maps dataset-specific labels to our framework's expected_safe concept.
    """

    is_attack: bool
    attack_type: str | None = None
    attack_description: str | None = None
    affected_sensors: list[str] = Field(default_factory=list)

    # The key mapping: does the ground truth say actuation is safe?
    # attack=True -> expected_safe=False (should deny)
    # attack=False -> expected_safe=True (should permit)
    @property
    def expected_safe(self) -> bool:
        return not self.is_attack


@dataclass
class TimeWindow:
    """A window of sensor observations with ground truth.

    This is the unit of evaluation: the pipeline processes one TimeWindow
    at a time, and the guardrail decision is compared against ground truth.
    """

    observations: list[Observation]
    ground_truth: GroundTruth
    start_time: datetime
    end_time: datetime
    window_id: int = 0

    # Query to send to the LLM (adapter-specific)
    query: str = "What action should be taken given the current sensor state?"

    # Dataset metadata
    dataset_name: str = ""
    dataset_split: str = ""  # "train" or "test"

    @property
    def num_observations(self) -> int:
        return len(self.observations)

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "dataset_name": self.dataset_name,
            "num_observations": self.num_observations,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "ground_truth_is_attack": self.ground_truth.is_attack,
            "ground_truth_attack_type": self.ground_truth.attack_type,
            "ground_truth_expected_safe": self.ground_truth.expected_safe,
            "query": self.query,
        }


# ── Dataset Metadata ────────────────────────────────────────────────────────


@dataclass
class DatasetMetadata:
    """Metadata about a loaded dataset."""

    name: str
    description: str
    num_sensors: int
    num_samples: int
    num_attack_windows: int
    num_normal_windows: int
    sampling_rate_hz: float
    duration_hours: float
    sensor_ids: list[str] = field(default_factory=list)
    source_url: str = ""
    citation: str = ""


# ── Abstract Base Class ─────────────────────────────────────────────────────


class DatasetAdapter(ABC):
    """Abstract base for mapping IoT datasets to epistemic-edge Observations.

    Subclasses implement dataset-specific loading and parsing. The b/d/u
    assignment is delegated to a pluggable UncertaintyStrategy, not
    hardcoded in the adapter — this separation is critical for the ablation
    study comparing uncertainty assignment methods.

    Usage::

        from epistemic_edge.adapters.uncertainty import HistoricalDeviationStrategy

        adapter = SWaTAdapter(strategy=HistoricalDeviationStrategy())
        adapter.load(Path("./data/SWaT_A2_Dec2015"))

        for window in adapter.get_windows(window_size=60, stride=30):
            # window.observations: list[Observation] with b/d/u assigned
            # window.ground_truth: GroundTruth with is_attack label
            node.ingest_batch(window.observations)
            ...
    """

    def __init__(self, strategy: Any = None) -> None:
        """Initialize with an uncertainty assignment strategy.

        Args:
            strategy: An UncertaintyStrategy instance. If None, a default
                HistoricalDeviationStrategy is used.
        """
        # Import here to avoid circular dependency
        from epistemic_edge.adapters.uncertainty import HistoricalDeviationStrategy

        self.strategy = strategy or HistoricalDeviationStrategy()
        self._loaded = False
        self._metadata: DatasetMetadata | None = None

    @property
    def metadata(self) -> DatasetMetadata:
        if self._metadata is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self._metadata

    @abstractmethod
    def load(self, path: Path, split: str = "test") -> None:
        """Load the raw dataset from disk.

        Args:
            path: Path to the dataset directory or file.
            split: Which split to load ("train" for calibration, "test" for evaluation).

        After loading, the adapter computes historical statistics from the
        training split (for HistoricalDeviationStrategy) and prepares the
        data for windowing.
        """
        ...

    @abstractmethod
    def get_windows(
        self,
        window_size: int = 60,
        stride: int = 30,
        max_windows: int | None = None,
    ) -> Iterator[TimeWindow]:
        """Yield time windows of observations with ground truth.

        Args:
            window_size: Number of timesteps per window.
            stride: Step between consecutive windows.
            max_windows: If set, stop after this many windows.

        Yields:
            TimeWindow objects with observations (b/d/u assigned via strategy)
            and ground truth labels from the dataset.
        """
        ...

    @abstractmethod
    def get_sensor_contexts(
        self,
        row_idx: int,
    ) -> list[SensorContext]:
        """Build SensorContext objects for a single timestep.

        This is where dataset-specific column parsing happens. The adapter
        knows which columns are sensors, what their physical bounds are,
        and which sensors are related (for agreement checks).

        Args:
            row_idx: Index into the loaded data.

        Returns:
            List of SensorContext objects, one per sensor.
        """
        ...

    def _assign_bdu(
        self, context: SensorContext
    ) -> tuple[float, float, float, float]:
        """Delegate b/d/u/a assignment to the configured strategy.

        Returns:
            Tuple of (belief, disbelief, uncertainty, base_rate).
        """
        return self.strategy.assign(context)

    def _make_observation(
        self,
        context: SensorContext,
        b: float,
        d: float,
        u: float,
        a: float = 0.5,
    ) -> Observation:
        """Create an Observation from a SensorContext and assigned b/d/u/a.

        Args:
            context: The sensor context.
            b: Belief.
            d: Disbelief.
            u: Uncertainty.
            a: Base rate (prior probability). Default 0.5.
        """
        return Observation(
            payload={context.sensor_id: context.reading, "unit": context.unit},
            source=ObservationSource(
                agent_id=context.sensor_id,
                agent_type=context.sensor_type,
            ),
            timestamp=context.timestamp,
            belief=b,
            disbelief=d,
            uncertainty=u,
            base_rate=a,
        )
