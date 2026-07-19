# -*- coding: utf-8 -*-
"""
MediReach — AirSim / Internal Simulation Bridge.

Unified interface for AirSim and internal drone simulation.
All flight controller code uses this bridge.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import numpy as np

from src.utils.constants import DroneConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)
_CFG = DroneConfig()

try:
    import airsim  # type: ignore[import-not-found]
    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False


class SimulationBridge:
    """Unified interface for AirSim and internal simulation."""

    def __init__(self, use_airsim: bool = False) -> None:
        self.use_airsim = use_airsim and AIRSIM_AVAILABLE
        self._state: Dict[str, Any] = {}

        if self.use_airsim:
            try:
                self.client = airsim.MultirotorClient()
                self.client.confirmConnection()
                self.client.enableApiControl(True)
                self.client.armDisarm(True)
                logger.info("AirSim connection established")
            except Exception as exc:
                logger.error("AirSim connection failed: %s", exc)
                self.use_airsim = False
                self._state = self._init_internal_state()
        else:
            self._state = self._init_internal_state()
            logger.info("Using internal simulation")

    def takeoff(self, altitude: float = 50.0) -> bool:
        if self.use_airsim:
            self.client.takeoffAsync().join()
            self.client.moveToZAsync(-altitude, 3).join()
        else:
            self._state["altitude"] = altitude
            self._state["status"] = "IN_FLIGHT"
        logger.info("Takeoff to %.0fm", altitude)
        return True

    def fly_to_waypoint(
        self, lat: float, lon: float, alt: float, speed: float = 15.0
    ) -> bool:
        if self.use_airsim:
            self.client.moveToPositionAsync(lon * 100000, lat * 100000, -alt, speed).join()
        else:
            self._state["lat"] = lat
            self._state["long"] = lon
            self._state["altitude"] = alt
            self._state["speed"] = speed
            self._state["battery"] -= 0.5
        logger.debug("Flying to (%.4f, %.4f) alt=%.0f", lat, lon, alt)
        return True

    def land(self) -> bool:
        if self.use_airsim:
            self.client.landAsync().join()
        else:
            self._state["altitude"] = 0.0
            self._state["speed"] = 0.0
            self._state["status"] = "LANDED"
        logger.info("Landing sequence complete")
        return True

    def hover(self) -> bool:
        if self.use_airsim:
            self.client.hoverAsync().join()
        else:
            self._state["speed"] = 0.0
            self._state["status"] = "HOVERING"
        return True

    def return_to_home(self) -> bool:
        logger.info("Returning to home base")
        return self.fly_to_waypoint(_CFG.HOME_LAT, _CFG.HOME_LONG, _CFG.CRUISE_ALTITUDE_M)

    def get_state(self) -> Dict[str, Any]:
        if self.use_airsim:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            return {
                "lat": pos.y_val / 100000, "long": pos.x_val / 100000,
                "altitude": -pos.z_val, "speed": state.kinematics_estimated.linear_velocity.get_length(),
                "heading": 0.0, "battery": 100.0, "status": "IN_FLIGHT",
            }
        return self._state.copy()

    def get_camera_frame(self) -> np.ndarray:
        if self.use_airsim:
            responses = self.client.simGetImages([
                airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)
            ])
            return self._decode_airsim_image(responses[0])
        return self._generate_synthetic_frame()

    def _generate_synthetic_frame(self) -> np.ndarray:
        frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        # Draw a synthetic landing pad
        center = (320, 240)
        import cv2
        cv2.circle(frame, center, 60, (0, 200, 0), -1)
        cv2.circle(frame, center, 40, (255, 255, 255), -1)
        cv2.drawMarker(frame, center, (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        return frame

    def _decode_airsim_image(self, response: Any) -> np.ndarray:
        img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        return img_1d.reshape(response.height, response.width, 3)

    def _init_internal_state(self) -> Dict[str, Any]:
        return {
            "lat": _CFG.HOME_LAT, "long": _CFG.HOME_LONG,
            "altitude": 0.0, "speed": 0.0, "heading": 0.0,
            "battery": 100.0, "status": "GROUNDED",
        }
