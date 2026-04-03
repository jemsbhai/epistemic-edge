"""Uncertainty assignment strategies for mapping raw sensor readings to b/d/u.

Three scientifically distinct approaches, each independently defensible:

1. SensorAgreementStrategy: Cross-sensor disagreement -> high uncertainty.
   Grounded in Josang (2016) multi-source fusion theory. When sensors that
   should agree disagree, the fused opinion's uncertainty rises.

2. HistoricalDeviationStrategy: Distance from historical mean -> belief/uncertainty.
   Grounded in statistical process control (Shewhart). Readings within 2 sigma
   get high belief; beyond 3 sigma get high disbelief.

3. PhysicsBoundsStrategy: Physical impossibility -> disbelief.
   Grounded in domain engineering. A water level below 0 or above tank capacity
   is physically impossible regardless of what the sensor reports.

The experiment runner can swap strategies to show the safety finding holds
across all three — making the result robust to the specific b/d/u assignment
method chosen.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from epistemic_edge.adapters.base import SensorContext


class UncertaintyStrategy(ABC):
    """Abstract base for b/d/u assignment from raw sensor readings.

    All strategies must satisfy the Subjective Logic constraint:
        b + d + u = 1.0, where b,d,u >= 0
    """

    @abstractmethod
    def assign(self, context: SensorContext) -> tuple[float, float, float, float]:
        """Assign (belief, disbelief, uncertainty, base_rate) to a sensor reading.

        Args:
            context: Full sensor context including the reading, historical
                statistics, physical bounds, and cross-sensor data.

        Returns:
            Tuple of (belief, disbelief, uncertainty, base_rate).
            b+d+u must sum to 1.0. base_rate in [0,1].
        """
        ...

    def _clamp_and_normalize(
        self, b: float, d: float, u: float, a: float = 0.5
    ) -> tuple[float, float, float, float]:
        """Ensure b+d+u=1.0, all values in [0,1], and pass through base_rate."""
        b = max(0.0, b)
        d = max(0.0, d)
        u = max(0.0, u)
        total = b + d + u
        if total == 0:
            return (0.0, 0.0, 1.0, a)  # Vacuous
        return (b / total, d / total, u / total, max(0.0, min(1.0, a)))


class SensorAgreementStrategy(UncertaintyStrategy):
    """Assign b/d/u based on cross-sensor agreement.

    When multiple sensors measure related quantities (e.g., two temperature
    sensors on the same pipe), agreement increases belief and disagreement
    increases uncertainty.

    If no related sensors are available, falls back to a moderate-confidence
    default (b=0.6, d=0.1, u=0.3) rather than vacuous, reflecting that a
    single sensor reading provides some evidence.

    Parameters:
        agreement_threshold: Maximum relative difference (fraction) for sensors
            to be considered "in agreement". Default 0.1 (10%).
        high_belief: Belief assigned when sensors agree. Default 0.9.
        disagreement_uncertainty: Uncertainty assigned when sensors disagree.
            Default 0.7.
    """

    def __init__(
        self,
        agreement_threshold: float = 0.1,
        high_belief: float = 0.9,
        disagreement_uncertainty: float = 0.7,
    ) -> None:
        self.agreement_threshold = agreement_threshold
        self.high_belief = high_belief
        self.disagreement_uncertainty = disagreement_uncertainty

    def assign(self, context: SensorContext) -> tuple[float, float, float, float]:
        if not context.related_readings:
            # No related sensors — moderate confidence fallback
            return self._clamp_and_normalize(0.6, 0.1, 0.3)

        # Compute agreement ratio across related sensors
        agreements = 0
        disagreements = 0
        for _sensor_id, related_reading in context.related_readings.items():
            if context.reading == 0 and related_reading == 0:
                agreements += 1
                continue
            denominator = max(abs(context.reading), abs(related_reading), 1e-9)
            relative_diff = abs(context.reading - related_reading) / denominator
            if relative_diff <= self.agreement_threshold:
                agreements += 1
            else:
                disagreements += 1

        total_comparisons = agreements + disagreements
        if total_comparisons == 0:
            return self._clamp_and_normalize(0.6, 0.1, 0.3)

        agreement_ratio = agreements / total_comparisons

        if agreement_ratio >= 0.8:
            # Strong agreement
            b = self.high_belief
            d = 0.02
            u = 1.0 - b - d
        elif agreement_ratio >= 0.5:
            # Partial agreement
            b = 0.5
            d = 0.1
            u = 0.4
        else:
            # Disagreement — high uncertainty
            b = 0.15
            d = 0.15
            u = self.disagreement_uncertainty

        return self._clamp_and_normalize(b, d, u)


class HistoricalDeviationStrategy(UncertaintyStrategy):
    """Assign b/d/u based on deviation from historical normal behavior.

    Uses statistical process control principles (Shewhart charts):
      - Within 1 sigma: high belief, low uncertainty
      - 1-2 sigma: moderate belief, moderate uncertainty
      - 2-3 sigma: low belief, high uncertainty (warning zone)
      - Beyond 3 sigma: high disbelief (anomalous)

    If no historical statistics are available, falls back to vacuous
    opinion (b=0, d=0, u=1) per Josang's principle that absence of
    evidence should not be treated as evidence of absence.

    Parameters:
        sigma_thresholds: Tuple of (warn, critical) sigma multiples.
            Default (2.0, 3.0).
    """

    def __init__(
        self,
        sigma_thresholds: tuple[float, float] = (2.0, 3.0),
    ) -> None:
        self.sigma_warn, self.sigma_critical = sigma_thresholds

    def assign(self, context: SensorContext) -> tuple[float, float, float, float]:
        if context.historical_mean is None or context.historical_std is None:
            # No history — vacuous opinion
            return self._clamp_and_normalize(0.0, 0.0, 1.0)

        if context.historical_std == 0:
            # Constant sensor — any deviation is suspicious
            if context.reading == context.historical_mean:
                return self._clamp_and_normalize(0.95, 0.02, 0.03)
            else:
                return self._clamp_and_normalize(0.1, 0.6, 0.3)

        # Compute z-score
        z = abs(context.reading - context.historical_mean) / context.historical_std

        if z <= 1.0:
            # Normal operating range
            b = 0.90 - 0.05 * z  # 0.90 -> 0.85
            d = 0.02 + 0.03 * z  # 0.02 -> 0.05
            u = 1.0 - b - d
        elif z <= self.sigma_warn:
            # Elevated — increasing uncertainty
            t = (z - 1.0) / (self.sigma_warn - 1.0)  # 0->1 over this range
            b = 0.85 - 0.45 * t  # 0.85 -> 0.40
            d = 0.05 + 0.10 * t  # 0.05 -> 0.15
            u = 1.0 - b - d
        elif z <= self.sigma_critical:
            # Warning zone — high uncertainty
            t = (z - self.sigma_warn) / (self.sigma_critical - self.sigma_warn)
            b = 0.40 - 0.30 * t  # 0.40 -> 0.10
            d = 0.15 + 0.25 * t  # 0.15 -> 0.40
            u = 1.0 - b - d
        else:
            # Anomalous — high disbelief
            # Exponential decay of belief beyond critical threshold
            excess = z - self.sigma_critical
            b = 0.10 * math.exp(-0.5 * excess)
            d = min(0.85, 0.40 + 0.10 * excess)
            u = 1.0 - b - d

        return self._clamp_and_normalize(b, d, u)


class PhysicsBoundsStrategy(UncertaintyStrategy):
    """Assign b/d/u based on physical plausibility of the reading.

    Uses domain knowledge about sensor physical bounds:
      - Reading within bounds: belief proportional to distance from bounds
      - Reading at boundary: elevated uncertainty
      - Reading outside bounds: high disbelief (physically impossible)

    Falls back to HistoricalDeviationStrategy if no physical bounds are
    configured for the sensor.

    Parameters:
        boundary_margin: Fraction of range near boundaries where uncertainty
            increases. Default 0.1 (10% of range at each end).
    """

    def __init__(self, boundary_margin: float = 0.1) -> None:
        self.boundary_margin = boundary_margin
        self._fallback = HistoricalDeviationStrategy()

    def assign(self, context: SensorContext) -> tuple[float, float, float, float]:
        if context.physical_min is None or context.physical_max is None:
            # No physical bounds — delegate to historical deviation
            return self._fallback.assign(context)

        pmin = context.physical_min
        pmax = context.physical_max
        prange = pmax - pmin

        if prange <= 0:
            return self._clamp_and_normalize(0.0, 0.0, 1.0)  # Degenerate bounds

        reading = context.reading

        if reading < pmin:
            # Below physical minimum — impossible
            overshoot = (pmin - reading) / prange
            d = min(0.95, 0.7 + 0.25 * overshoot)
            b = 0.02
            u = 1.0 - b - d
            return self._clamp_and_normalize(b, d, u)

        if reading > pmax:
            # Above physical maximum — impossible
            overshoot = (reading - pmax) / prange
            d = min(0.95, 0.7 + 0.25 * overshoot)
            b = 0.02
            u = 1.0 - b - d
            return self._clamp_and_normalize(b, d, u)

        # Within bounds — compute normalized position
        normalized = (reading - pmin) / prange  # 0.0 to 1.0

        # Distance from nearest boundary (0.0 at boundary, 0.5 at center)
        dist_from_boundary = min(normalized, 1.0 - normalized)

        if dist_from_boundary < self.boundary_margin:
            # Near boundary — elevated uncertainty
            t = dist_from_boundary / self.boundary_margin  # 0->1
            b = 0.50 + 0.35 * t  # 0.50 -> 0.85
            d = 0.10 - 0.05 * t  # 0.10 -> 0.05
            u = 1.0 - b - d
        else:
            # Comfortably within bounds
            b = 0.85
            d = 0.03
            u = 0.12

        return self._clamp_and_normalize(b, d, u)


class CompositeStrategy(UncertaintyStrategy):
    """Combines multiple strategies via weighted averaging.

    This allows using all three strategies simultaneously, which is the
    most robust approach for the paper. The final b/d/u is a weighted
    average of each strategy's assignment.

    Parameters:
        strategies: List of (strategy, weight) tuples. Weights are normalized.
    """

    def __init__(
        self,
        strategies: list[tuple[UncertaintyStrategy, float]] | None = None,
    ) -> None:
        if strategies is None:
            # Default: equal weight to all three core strategies
            strategies = [
                (SensorAgreementStrategy(), 1.0),
                (HistoricalDeviationStrategy(), 1.0),
                (PhysicsBoundsStrategy(), 1.0),
            ]
        self.strategies = strategies
        total_weight = sum(w for _, w in strategies)
        self.normalized_weights = [w / total_weight for _, w in strategies]

    def assign(self, context: SensorContext) -> tuple[float, float, float, float]:
        b_total, d_total, u_total, a_total = 0.0, 0.0, 0.0, 0.0
        for (strategy, _), weight in zip(self.strategies, self.normalized_weights):
            b, d, u, a = strategy.assign(context)
            b_total += b * weight
            d_total += d * weight
            u_total += u * weight
            a_total += a * weight
        return self._clamp_and_normalize(b_total, d_total, u_total, a_total)
