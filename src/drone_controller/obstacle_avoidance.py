# -*- coding: utf-8 -*-
"""
MediReach — Obstacle Avoidance (LiDAR/Ultrasonic).

Simulated obstacle detection using distance sensors
and avoidance manoeuvre computation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SensorReading:
    """Distance sensor reading."""
    direction: str  # front, back, left, right, up, down
    distance_m: float
    is_obstacle: bool
    timestamp: float


class ObstacleAvoidance:
    """Obstacle detection and avoidance for drone flight.

    Uses simulated ultrasonic/LiDAR sensors in 6 directions.
    Computes avoidance vectors when obstacles are within
    the safety margin.
    """

    DIRECTIONS = ["front", "back", "left", "right", "up", "down"]
    SAFETY_MARGIN_M: float = 5.0
    CRITICAL_DISTANCE_M: float = 2.0

    def __init__(self, simulation_mode: bool = True) -> None:
        self.simulation_mode = simulation_mode
        self._readings: Dict[str, float] = {d: 100.0 for d in self.DIRECTIONS}
        logger.info("ObstacleAvoidance init, sim=%s", simulation_mode)

    def get_readings(self) -> Dict[str, SensorReading]:
        if self.simulation_mode:
            self._simulate_readings()
        import time
        return {
            d: SensorReading(
                direction=d, distance_m=self._readings[d],
                is_obstacle=self._readings[d] < self.SAFETY_MARGIN_M,
                timestamp=time.time(),
            )
            for d in self.DIRECTIONS
        }

    def is_path_clear(self, direction: str = "front") -> bool:
        return self._readings.get(direction, 100.0) >= self.SAFETY_MARGIN_M

    def is_critical(self) -> bool:
        return any(d < self.CRITICAL_DISTANCE_M for d in self._readings.values())

    def compute_avoidance_vector(self) -> Tuple[float, float, float]:
        """Compute avoidance direction (dx, dy, dz).

        Returns:
            Tuple of normalised direction components to move toward safety.
        """
        dx, dy, dz = 0.0, 0.0, 0.0

        mapping = {
            "front": (0, 1, 0), "back": (0, -1, 0),
            "left": (-1, 0, 0), "right": (1, 0, 0),
            "up": (0, 0, 1), "down": (0, 0, -1),
        }

        for direction, (mx, my, mz) in mapping.items():
            dist = self._readings[direction]
            if dist < self.SAFETY_MARGIN_M:
                repulsion = (self.SAFETY_MARGIN_M - dist) / self.SAFETY_MARGIN_M
                dx -= mx * repulsion
                dy -= my * repulsion
                dz -= mz * repulsion

        magnitude = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude
            dz /= magnitude

        return dx, dy, dz

    def _simulate_readings(self) -> None:
        for d in self.DIRECTIONS:
            base = self._readings[d]
            noise = random.gauss(0, 0.5)
            self._readings[d] = max(0.5, base + noise)

    def set_reading(self, direction: str, distance_m: float) -> None:
        if direction in self.DIRECTIONS:
            self._readings[direction] = distance_m

    def reset(self) -> None:
        self._readings = {d: 100.0 for d in self.DIRECTIONS}
