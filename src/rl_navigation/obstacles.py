# -*- coding: utf-8 -*-
"""
MediReach — Obstacle & No-Fly Zone Manager.

Manages static obstacles and no-fly zones within the RL grid
environment.  Provides collision detection, sensor readings,
and zone checks used by the navigation environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NoFlyZone:
    """Circular no-fly zone definition."""
    zone_id: str
    center_x: float
    center_y: float
    radius: float
    active: bool = True


class ObstacleManager:
    """Manages obstacles and no-fly zones in the grid world.

    Obstacles are point objects that cause collisions.
    No-fly zones are circular regions that incur large penalties.
    """

    # Directions for 8-directional obstacle sensors (dx, dy)
    SENSOR_DIRS: List[Tuple[float, float]] = [
        (0, 1),    # North
        (1, 1),    # NE
        (1, 0),    # East
        (1, -1),   # SE
        (0, -1),   # South
        (-1, -1),  # SW
        (-1, 0),   # West
        (-1, 1),   # NW
    ]

    SENSOR_RANGE: float = 10.0  # Maximum sensor detection range

    def __init__(self, grid_size: int = 100) -> None:
        """Initialise the obstacle manager.

        Args:
            grid_size: Size of the square grid world.
        """
        self.grid_size = grid_size
        self.obstacles: List[np.ndarray] = []
        self.no_fly_zones: List[NoFlyZone] = []

    def reset(
        self,
        num_obstacles: int = 15,
        num_no_fly_zones: int = 5,
        rng: Optional[np.random.Generator] = None,
        drone_pos: Optional[np.ndarray] = None,
        target_pos: Optional[np.ndarray] = None,
    ) -> None:
        """Randomise obstacles and no-fly zones for a new episode.

        Avoids placing obstacles on or very near the drone start
        or target positions.

        Args:
            num_obstacles: Number of point obstacles.
            num_no_fly_zones: Number of circular no-fly zones.
            rng: Numpy random generator for reproducibility.
            drone_pos: Drone start position [x, y] to keep clear.
            target_pos: Target position [x, y] to keep clear.
        """
        if rng is None:
            rng = np.random.default_rng()

        safe_dist = 5.0  # minimum distance from drone/target

        # Generate obstacles
        self.obstacles = []
        attempts = 0
        while len(self.obstacles) < num_obstacles and attempts < num_obstacles * 10:
            pos = rng.uniform(0, self.grid_size, size=2).astype(np.float32)
            too_close = False

            if drone_pos is not None:
                if np.linalg.norm(pos - drone_pos[:2]) < safe_dist:
                    too_close = True
            if target_pos is not None:
                if np.linalg.norm(pos - target_pos) < safe_dist:
                    too_close = True

            if not too_close:
                self.obstacles.append(pos)
            attempts += 1

        # Generate no-fly zones
        self.no_fly_zones = []
        attempts = 0
        while len(self.no_fly_zones) < num_no_fly_zones and attempts < num_no_fly_zones * 10:
            cx = float(rng.uniform(10, self.grid_size - 10))
            cy = float(rng.uniform(10, self.grid_size - 10))
            radius = float(rng.uniform(3, 8))

            # Ensure no-fly zone doesn't cover drone or target
            too_close = False
            if drone_pos is not None:
                if math.hypot(cx - drone_pos[0], cy - drone_pos[1]) < radius + safe_dist:
                    too_close = True
            if target_pos is not None:
                if math.hypot(cx - target_pos[0], cy - target_pos[1]) < radius + safe_dist:
                    too_close = True

            if not too_close:
                nfz = NoFlyZone(
                    zone_id=f"NFZ-{len(self.no_fly_zones):02d}",
                    center_x=cx,
                    center_y=cy,
                    radius=radius,
                    active=True,
                )
                self.no_fly_zones.append(nfz)
            attempts += 1

        logger.debug(
            "ObstacleManager reset: %d obstacles, %d no-fly zones",
            len(self.obstacles), len(self.no_fly_zones),
        )

    def check_collision(self, x: float, y: float, tolerance: float = 1.0) -> bool:
        """Check if position collides with any obstacle.

        Args:
            x: X coordinate.
            y: Y coordinate.
            tolerance: Collision radius in grid units.

        Returns:
            True if collision detected.
        """
        pos = np.array([x, y])
        for obs in self.obstacles:
            if np.linalg.norm(pos - obs) < tolerance:
                return True
        return False

    def check_no_fly_zone(self, x: float, y: float) -> bool:
        """Check if position is inside any active no-fly zone.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if inside a no-fly zone.
        """
        for nfz in self.no_fly_zones:
            if not nfz.active:
                continue
            dist = math.hypot(x - nfz.center_x, y - nfz.center_y)
            if dist < nfz.radius:
                return True
        return False

    def is_obstacle_nearby(
        self, x: float, y: float, radius: float = 3.0
    ) -> bool:
        """Check if any obstacle is within radius of position.

        Args:
            x: X coordinate.
            y: Y coordinate.
            radius: Detection radius.

        Returns:
            True if obstacle within radius.
        """
        pos = np.array([x, y])
        for obs in self.obstacles:
            if np.linalg.norm(pos - obs) < radius:
                return True
        return False

    def get_sensor_readings(
        self,
        x: float,
        y: float,
        num_directions: int = 8,
    ) -> np.ndarray:
        """Get normalised obstacle sensor readings in 8 directions.

        Each sensor returns a value between 0 (obstacle at position)
        and 1 (no obstacle within sensor range).

        Args:
            x: Current X position.
            y: Current Y position.
            num_directions: Number of sensor directions.

        Returns:
            Array of shape (num_directions,) with values in [0, 1].
        """
        readings = np.ones(num_directions, dtype=np.float32)
        pos = np.array([x, y])

        for i, (dx, dy) in enumerate(self.SENSOR_DIRS[:num_directions]):
            min_dist = self.SENSOR_RANGE
            dir_vec = np.array([dx, dy], dtype=np.float32)
            dir_norm = dir_vec / max(np.linalg.norm(dir_vec), 1e-8)

            for obs in self.obstacles:
                # Project obstacle onto sensor ray
                to_obs = obs - pos
                proj_len = float(np.dot(to_obs, dir_norm))

                if 0 < proj_len < self.SENSOR_RANGE:
                    perp_dist = float(np.linalg.norm(
                        to_obs - proj_len * dir_norm
                    ))
                    if perp_dist < 1.5:  # obstacle width
                        min_dist = min(min_dist, proj_len)

            # Also check no-fly zone boundaries
            for nfz in self.no_fly_zones:
                if not nfz.active:
                    continue
                nfz_pos = np.array([nfz.center_x, nfz.center_y])
                to_nfz = nfz_pos - pos
                proj_len = float(np.dot(to_nfz, dir_norm))

                if 0 < proj_len < self.SENSOR_RANGE:
                    perp_dist = float(np.linalg.norm(
                        to_nfz - proj_len * dir_norm
                    ))
                    if perp_dist < nfz.radius:
                        edge_dist = max(0.0, proj_len - nfz.radius + perp_dist)
                        min_dist = min(min_dist, edge_dist)

            readings[i] = min_dist / self.SENSOR_RANGE

        return readings

    def get_no_fly_flags(
        self, x: float, y: float, max_zones: int = 5
    ) -> np.ndarray:
        """Get binary proximity flags for nearest no-fly zones.

        Returns 1.0 if the drone is within 2× the zone radius, else 0.0.

        Args:
            x: Current X position.
            y: Current Y position.
            max_zones: Number of flags to return.

        Returns:
            Array of shape (max_zones,) with binary flags.
        """
        flags = np.zeros(max_zones, dtype=np.float32)

        for i, nfz in enumerate(self.no_fly_zones[:max_zones]):
            if not nfz.active:
                continue
            dist = math.hypot(x - nfz.center_x, y - nfz.center_y)
            if dist < nfz.radius * 2.0:
                flags[i] = 1.0

        return flags

    def add_obstacle(self, x: float, y: float) -> None:
        """Manually add a point obstacle.

        Args:
            x: X coordinate.
            y: Y coordinate.
        """
        self.obstacles.append(np.array([x, y], dtype=np.float32))

    def add_no_fly_zone(
        self,
        zone_id: str,
        cx: float,
        cy: float,
        radius: float,
    ) -> None:
        """Manually add a no-fly zone.

        Args:
            zone_id: Unique zone identifier.
            cx: Center X coordinate.
            cy: Center Y coordinate.
            radius: Zone radius in grid units.
        """
        nfz = NoFlyZone(
            zone_id=zone_id,
            center_x=cx,
            center_y=cy,
            radius=radius,
            active=True,
        )
        self.no_fly_zones.append(nfz)
        logger.info("Added no-fly zone %s at (%.1f, %.1f) r=%.1f",
                     zone_id, cx, cy, radius)

    def get_obstacle_count(self) -> int:
        """Return total number of obstacles."""
        return len(self.obstacles)

    def get_no_fly_zone_count(self) -> int:
        """Return total number of no-fly zones."""
        return len(self.no_fly_zones)
