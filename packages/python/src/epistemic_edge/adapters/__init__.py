"""Dataset adapters for mapping real IoT datasets to epistemic-edge Observations.

Provides:
  - UncertaintyStrategy: pluggable b/d/u assignment from raw sensor values
  - DatasetAdapter: abstract base for dataset-specific loaders
  - Concrete adapters for SWaT, BATADAL, NASA Bearing, MQTT-IoT-IDS2020
"""

from epistemic_edge.adapters.base import (
    DatasetAdapter,
    DatasetMetadata,
    GroundTruth,
    SensorContext,
    TimeWindow,
)
from epistemic_edge.adapters.uncertainty import (
    CompositeStrategy,
    HistoricalDeviationStrategy,
    PhysicsBoundsStrategy,
    SensorAgreementStrategy,
    UncertaintyStrategy,
)
from epistemic_edge.adapters.batadal import BATADALAdapter
from epistemic_edge.adapters.swat import SWaTAdapter
from epistemic_edge.adapters.nasa_bearing import NASABearingAdapter
from epistemic_edge.adapters.mqtt_iot import MQTTIoTAdapter

__all__ = [
    # Base
    "DatasetAdapter",
    "DatasetMetadata",
    "GroundTruth",
    "SensorContext",
    "TimeWindow",
    # Strategies
    "UncertaintyStrategy",
    "SensorAgreementStrategy",
    "HistoricalDeviationStrategy",
    "PhysicsBoundsStrategy",
    "CompositeStrategy",
    # Concrete adapters
    "BATADALAdapter",
    "SWaTAdapter",
    "NASABearingAdapter",
    "MQTTIoTAdapter",
]
