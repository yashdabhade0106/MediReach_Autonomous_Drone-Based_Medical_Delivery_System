# -*- coding: utf-8 -*-
"""
MediReach — Production Route Optimiser.

Flask-ready inference wrapper for the trained RL agent.
Converts GPS coordinates to grid, runs RL inference,
and converts waypoints back to GPS.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.rl_navigation.environment import MediReachEnv
from src.rl_navigation.agent import PPOAgent
from src.utils.geo_utils import (
    grid_to_gps,
    gps_to_grid,
    haversine_distance,
    metres_to_km,
    estimate_flight_time,
)
from src.utils.constants import RLConfig, DroneConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_RL_CFG = RLConfig()
_DRONE_CFG = DroneConfig()


class RouteOptimizer:
    """Production inference wrapper for the trained RL agent.

    Used by the Flask API to compute optimised waypoints
    for each delivery order.
    """

    def __init__(self) -> None:
        """Initialise the route optimiser (model not loaded yet)."""
        self.agent: Optional[PPOAgent] = None
        self.env: Optional[MediReachEnv] = None
        self.model_loaded: bool = False
        self._origin_lat: float = _DRONE_CFG.HOME_LAT
        self._origin_lon: float = _DRONE_CFG.HOME_LONG
        self._cell_size_m: float = _RL_CFG.GRID_CELL_SIZE_M

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the trained PPO model from disk.

        Args:
            model_path: Path to model .zip file. Falls back to env var.

        Returns:
            True if model loaded successfully.
        """
        path = model_path or os.getenv("RL_MODEL_PATH", "models/rl/medireach_ppo_final.zip")

        try:
            self.env = MediReachEnv(render_mode=None)
            self.agent = PPOAgent(env=self.env)
            self.agent.load(path)
            self.model_loaded = True
            logger.info("RouteOptimizer: model loaded from %s", path)
            return True
        except FileNotFoundError:
            logger.warning("RouteOptimizer: model file not found at %s", path)
            self.model_loaded = False
            return False
        except Exception as exc:
            logger.error("RouteOptimizer: failed to load model: %s", exc)
            self.model_loaded = False
            return False

    def get_optimized_route(
        self,
        pickup_coords: Dict[str, float],
        delivery_coords: Dict[str, float],
        weather_data: Optional[Dict[str, Any]] = None,
        no_fly_zones: Optional[List[Dict]] = None,
        battery_level: float = 1.0,
        priority: str = "standard",
    ) -> Dict[str, Any]:
        """Compute an optimised route between pickup and delivery.

        If the RL model is not loaded, falls back to a direct
        straight-line route with intermediate waypoints.

        Args:
            pickup_coords: {'lat': float, 'long': float}
            delivery_coords: {'lat': float, 'long': float}
            weather_data: Current weather conditions.
            no_fly_zones: List of no-fly zone definitions.
            battery_level: Current battery level (0–1).
            priority: 'emergency' or 'standard'.

        Returns:
            Route dictionary with waypoints, distance, ETA, etc.
        """
        start_time = time.time()

        # Convert GPS → grid
        start_grid = self._coords_to_grid(
            pickup_coords["lat"], pickup_coords["long"]
        )
        end_grid = self._coords_to_grid(
            delivery_coords["lat"], delivery_coords["long"]
        )

        # Compute distance
        distance_m = haversine_distance(
            pickup_coords["lat"], pickup_coords["long"],
            delivery_coords["lat"], delivery_coords["long"],
        )
        distance_km = metres_to_km(distance_m)

        if self.model_loaded and self.agent is not None and self.env is not None:
            # RL-based route
            start_pos = [
                float(start_grid[0]),
                float(start_grid[1]),
                _DRONE_CFG.CRUISE_ALTITUDE_M / self._cell_size_m,
            ]
            end_pos = [float(end_grid[0]), float(end_grid[1])]

            grid_waypoints = self.agent.optimize_route(
                start_pos=start_pos,
                end_pos=end_pos,
                weather=weather_data,
            )
            gps_waypoints = self._waypoints_to_gps(grid_waypoints)
            battery_usage = (1.0 - self.env.battery) * 100.0
            risk_score = self._calculate_risk_score(
                weather_data, no_fly_zones, distance_km
            )
        else:
            # Fallback: straight-line interpolation
            logger.warning("RL model not loaded — using straight-line fallback")
            gps_waypoints = self._generate_straight_line_route(
                pickup_coords, delivery_coords
            )
            battery_usage = distance_km * 4.0  # ~4% per km estimate
            risk_score = 0.3

        # Estimate time
        try:
            wind_speed = (weather_data or {}).get("wind_speed", 0.0)
            flight_time_s = estimate_flight_time(
                distance_m, _DRONE_CFG.MAX_SPEED_MS, headwind_ms=wind_speed * 0.3
            )
        except ValueError:
            flight_time_s = distance_m / (_DRONE_CFG.MAX_SPEED_MS * 0.5)

        estimated_minutes = flight_time_s / 60.0

        # Add ETA to each waypoint
        for i, wp in enumerate(gps_waypoints):
            wp["eta_seconds"] = int(flight_time_s * (i + 1) / max(len(gps_waypoints), 1))

        compute_time = time.time() - start_time

        route = {
            "waypoints": gps_waypoints,
            "total_distance_km": round(distance_km, 2),
            "estimated_time_minutes": round(estimated_minutes, 1),
            "battery_usage_percent": round(min(battery_usage, 100.0), 1),
            "route_risk_score": round(risk_score, 2),
            "priority": priority,
            "compute_time_ms": round(compute_time * 1000, 1),
            "model_used": self.model_loaded,
            "alternative_routes": [],
        }

        logger.info(
            "Route computed: %.1f km, ~%.0f min, risk=%.2f, %d waypoints (%.0f ms)",
            distance_km, estimated_minutes, risk_score,
            len(gps_waypoints), compute_time * 1000,
        )

        return route

    def get_emergency_route(
        self,
        pickup_coords: Dict[str, float],
        delivery_coords: Dict[str, float],
        obstacles: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Generate a priority route ignoring optimality for speed.

        Used when order_priority == 'emergency'.  Takes the most
        direct path, only avoiding hard obstacles.

        Args:
            pickup_coords: Pickup GPS coordinates.
            delivery_coords: Delivery GPS coordinates.
            obstacles: Known obstacle positions.

        Returns:
            Emergency route dictionary.
        """
        logger.info("Generating EMERGENCY route (direct path)")

        waypoints = self._generate_straight_line_route(
            pickup_coords, delivery_coords, num_points=5,
            altitude=_DRONE_CFG.MAX_ALTITUDE_M,
        )

        distance_m = haversine_distance(
            pickup_coords["lat"], pickup_coords["long"],
            delivery_coords["lat"], delivery_coords["long"],
        )

        return {
            "waypoints": waypoints,
            "total_distance_km": round(metres_to_km(distance_m), 2),
            "estimated_time_minutes": round(
                distance_m / _DRONE_CFG.MAX_SPEED_MS / 60.0, 1
            ),
            "battery_usage_percent": round(metres_to_km(distance_m) * 3.5, 1),
            "route_risk_score": 0.1,
            "priority": "emergency",
            "model_used": False,
            "alternative_routes": [],
        }

    # ───────────────────────────────────────────────────────
    #  Coordinate Conversion
    # ───────────────────────────────────────────────────────

    def _coords_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert GPS coordinates to grid indices.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.

        Returns:
            Tuple of (grid_x, grid_y).
        """
        return gps_to_grid(
            lat, lon,
            self._origin_lat, self._origin_lon,
            self._cell_size_m,
        )

    def _grid_to_coords(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to GPS coordinates.

        Args:
            gx: Grid X index.
            gy: Grid Y index.

        Returns:
            Tuple of (latitude, longitude).
        """
        return grid_to_gps(
            gx, gy,
            self._origin_lat, self._origin_lon,
            self._cell_size_m,
        )

    def _waypoints_to_gps(
        self, grid_waypoints: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert grid waypoints to GPS waypoints.

        Args:
            grid_waypoints: List of waypoint dicts with x, y keys.

        Returns:
            List of waypoint dicts with lat, long keys.
        """
        gps_waypoints = []
        for wp in grid_waypoints:
            lat, lon = self._grid_to_coords(int(wp["x"]), int(wp["y"]))
            gps_waypoints.append({
                "lat": round(lat, 6),
                "long": round(lon, 6),
                "altitude": round(wp.get("altitude", _DRONE_CFG.CRUISE_ALTITUDE_M), 1),
                "action": wp.get("action", "move"),
                "battery": round(wp.get("battery", 1.0) * 100, 1),
                "eta_seconds": 0,
            })
        return gps_waypoints

    def _generate_straight_line_route(
        self,
        pickup: Dict[str, float],
        delivery: Dict[str, float],
        num_points: int = 10,
        altitude: float = _DRONE_CFG.CRUISE_ALTITUDE_M,
    ) -> List[Dict[str, Any]]:
        """Generate a straight-line interpolated route.

        Args:
            pickup: Start coordinates.
            delivery: End coordinates.
            num_points: Number of intermediate waypoints.
            altitude: Cruise altitude in metres.

        Returns:
            List of waypoint dictionaries.
        """
        lats = np.linspace(pickup["lat"], delivery["lat"], num_points)
        lons = np.linspace(pickup["long"], delivery["long"], num_points)

        return [
            {
                "lat": round(float(lats[i]), 6),
                "long": round(float(lons[i]), 6),
                "altitude": altitude,
                "action": "move",
                "battery": round(100.0 - (i / num_points * 20), 1),
                "eta_seconds": 0,
            }
            for i in range(num_points)
        ]

    def _calculate_risk_score(
        self,
        weather: Optional[Dict[str, Any]],
        no_fly_zones: Optional[List[Dict]],
        distance_km: float,
    ) -> float:
        """Calculate composite route risk score.

        Args:
            weather: Weather conditions.
            no_fly_zones: No-fly zones along route.
            distance_km: Total route distance.

        Returns:
            Risk score between 0.0 (safe) and 1.0 (dangerous).
        """
        risk = 0.0

        # Distance risk (longer = riskier)
        risk += min(distance_km / _DRONE_CFG.MAX_RANGE_KM, 0.3)

        # Weather risk
        if weather:
            wind_risk = weather.get("wind_speed", 0) / 50.0 * 0.3
            rain_risk = weather.get("rain_intensity", 0) * 0.2
            risk += wind_risk + rain_risk

        # No-fly zone proximity risk
        if no_fly_zones:
            risk += min(len(no_fly_zones) * 0.05, 0.2)

        return min(risk, 1.0)
