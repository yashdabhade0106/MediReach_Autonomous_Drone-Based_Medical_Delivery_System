# -*- coding: utf-8 -*-
"""
MediReach — Weather Simulation for RL Training & Inference.

Provides stochastic weather simulation with five profiles
(clear, windy, rainy, storm, foggy) and a random-walk model
for realistic temporal variation during training episodes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.utils.constants import WeatherProfile, WeatherThresholds
from src.utils.logger import get_logger

logger = get_logger(__name__)

_THRESHOLDS = WeatherThresholds()


@dataclass
class WeatherState:
    """Current weather conditions."""
    wind_speed: float = 5.0       # m/s
    wind_direction: float = 0.0   # degrees (0=North)
    rain_intensity: float = 0.0   # 0.0–1.0
    visibility: float = 1.0       # 0.0–1.0
    profile: str = "clear"


class WeatherSimulator:
    """Realistic weather simulation for RL training and real-time inference.

    During training, weather evolves via a random-walk process each timestep.
    During inference, weather can be set from real API data.
    """

    WEATHER_PROFILES: Dict[str, Dict[str, Any]] = {
        "clear": {
            "wind": (0.0, 5.0),
            "rain": 0.0,
            "visibility": 1.0,
        },
        "windy": {
            "wind": (15.0, 30.0),
            "rain": 0.0,
            "visibility": 0.9,
        },
        "rainy": {
            "wind": (5.0, 15.0),
            "rain": 0.7,
            "visibility": 0.6,
        },
        "storm": {
            "wind": (30.0, 50.0),
            "rain": 1.0,
            "visibility": 0.2,
        },
        "foggy": {
            "wind": (0.0, 5.0),
            "rain": 0.0,
            "visibility": 0.3,
        },
    }

    # Random-walk step sizes
    WIND_SPEED_STEP: float = 1.5
    WIND_DIR_STEP: float = 15.0
    RAIN_STEP: float = 0.05
    VISIBILITY_STEP: float = 0.03

    def __init__(self) -> None:
        """Initialise with clear weather."""
        self._state = WeatherState()
        self._rng: Optional[np.random.Generator] = None

    def set_profile(self, profile: str) -> None:
        """Set weather to a named profile.

        Args:
            profile: One of 'clear', 'windy', 'rainy', 'storm', 'foggy'.

        Raises:
            ValueError: If profile name is unknown.
        """
        if profile not in self.WEATHER_PROFILES:
            raise ValueError(
                f"Unknown weather profile '{profile}'. "
                f"Valid: {list(self.WEATHER_PROFILES.keys())}"
            )

        params = self.WEATHER_PROFILES[profile]
        wind_min, wind_max = params["wind"]
        self._state = WeatherState(
            wind_speed=random.uniform(wind_min, wind_max),
            wind_direction=random.uniform(0.0, 360.0),
            rain_intensity=params["rain"],
            visibility=params["visibility"],
            profile=profile,
        )
        logger.debug("Weather set to profile '%s': %s", profile, self._state)

    def randomise(self, rng: Optional[np.random.Generator] = None) -> None:
        """Randomise weather by selecting a random profile.

        Args:
            rng: Optional numpy random generator for reproducibility.
        """
        self._rng = rng
        profiles = list(self.WEATHER_PROFILES.keys())
        # Weight toward calmer weather for training stability
        weights = [0.35, 0.25, 0.20, 0.10, 0.10]
        if rng is not None:
            idx = rng.choice(len(profiles), p=weights)
            chosen = profiles[idx]
        else:
            chosen = random.choices(profiles, weights=weights, k=1)[0]
        self.set_profile(chosen)

    def get_current_conditions(self) -> Dict[str, Any]:
        """Return current weather state as a dictionary.

        Returns:
            Dictionary with wind_speed, wind_direction, rain_intensity,
            visibility, and profile keys.
        """
        return {
            "wind_speed": self._state.wind_speed,
            "wind_direction": self._state.wind_direction,
            "rain_intensity": self._state.rain_intensity,
            "visibility": self._state.visibility,
            "profile": self._state.profile,
        }

    def step_weather(self) -> Dict[str, Any]:
        """Advance weather by one timestep using random walk.

        Each parameter is perturbed by a small random amount,
        then clipped to valid ranges.

        Returns:
            Updated weather conditions dictionary.
        """
        rng = self._rng

        # Wind speed random walk
        if rng is not None:
            ws_delta = rng.normal(0, self.WIND_SPEED_STEP)
            wd_delta = rng.normal(0, self.WIND_DIR_STEP)
            rain_delta = rng.normal(0, self.RAIN_STEP)
            vis_delta = rng.normal(0, self.VISIBILITY_STEP)
        else:
            ws_delta = random.gauss(0, self.WIND_SPEED_STEP)
            wd_delta = random.gauss(0, self.WIND_DIR_STEP)
            rain_delta = random.gauss(0, self.RAIN_STEP)
            vis_delta = random.gauss(0, self.VISIBILITY_STEP)

        self._state.wind_speed = float(np.clip(
            self._state.wind_speed + ws_delta, 0.0, 50.0
        ))
        self._state.wind_direction = (self._state.wind_direction + wd_delta) % 360.0
        self._state.rain_intensity = float(np.clip(
            self._state.rain_intensity + rain_delta, 0.0, 1.0
        ))
        self._state.visibility = float(np.clip(
            self._state.visibility + vis_delta, 0.0, 1.0
        ))

        return self.get_current_conditions()

    def is_flyable(self, conditions: Optional[Dict[str, Any]] = None) -> bool:
        """Determine if conditions are safe for flight.

        Args:
            conditions: Weather dict, or None to use current state.

        Returns:
            True if wind and rain are within safe thresholds.
        """
        if conditions is None:
            conditions = self.get_current_conditions()

        wind_ok = conditions["wind_speed"] <= _THRESHOLDS.MAX_WIND_SPEED_MS
        rain_ok = conditions["rain_intensity"] <= _THRESHOLDS.MAX_RAIN_INTENSITY
        vis_ok = conditions["visibility"] >= _THRESHOLDS.MIN_VISIBILITY

        return wind_ok and rain_ok and vis_ok

    def get_wind_resistance_factor(
        self,
        wind_speed: float,
        wind_direction: float,
        drone_heading: float,
    ) -> float:
        """Calculate battery drain multiplier based on wind resistance.

        Headwind increases drain, tailwind decreases it.

        Args:
            wind_speed: Current wind speed in m/s.
            wind_direction: Wind compass direction in degrees.
            drone_heading: Drone heading in degrees.

        Returns:
            Multiplier ≥ 0.0 (0 = pure tailwind, 2 = pure headwind).
        """
        # Angle between wind and drone heading
        relative_angle = abs(wind_direction - drone_heading)
        if relative_angle > 180.0:
            relative_angle = 360.0 - relative_angle

        # cos(0°)=1 (headwind), cos(180°)=-1 (tailwind)
        headwind_component = math.cos(math.radians(relative_angle))

        # Normalise wind effect: factor = 1 + (headwind_component * wind_speed / 50)
        factor = 1.0 + headwind_component * (wind_speed / 50.0)
        return max(0.0, factor)

    def set_from_api_data(self, api_data: Dict[str, Any]) -> None:
        """Set weather from real-time weather API response.

        Args:
            api_data: Dictionary with 'wind_speed', 'wind_direction',
                'rain_intensity', 'visibility' keys.
        """
        self._state = WeatherState(
            wind_speed=float(api_data.get("wind_speed", 5.0)),
            wind_direction=float(api_data.get("wind_direction", 0.0)),
            rain_intensity=float(api_data.get("rain_intensity", 0.0)),
            visibility=float(api_data.get("visibility", 1.0)),
            profile="api_live",
        )
        logger.info("Weather set from API data: wind=%.1f m/s", self._state.wind_speed)
