# -*- coding: utf-8 -*-
"""
MediReach — Flight Controller Interface.

High-level flight controller orchestrating GPS, battery,
obstacle avoidance, telemetry, and simulation bridge
for complete mission execution.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from src.drone_controller.gps_handler import GPSHandler
from src.drone_controller.battery_monitor import BatteryMonitor
from src.drone_controller.obstacle_avoidance import ObstacleAvoidance
from src.drone_controller.telemetry import TelemetryPublisher, TelemetryPacket
from src.drone_controller.simulation_bridge import SimulationBridge
from src.utils.constants import DroneConfig, MissionStatus
from src.utils.geo_utils import haversine_distance
from src.utils.logger import get_logger

logger = get_logger(__name__)
_CFG = DroneConfig()


class FlightController:
    """High-level flight controller for MediReach drone.

    Orchestrates all drone sub-systems for a complete
    pickup-to-delivery mission.
    """

    def __init__(
        self,
        drone_id: str = "DRN-001",
        simulation_mode: bool = True,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
    ) -> None:
        self.drone_id = drone_id
        self.simulation_mode = simulation_mode
        self.status = MissionStatus.PENDING
        self.current_mission_id: Optional[str] = None

        # Sub-systems
        self.gps = GPSHandler(simulation_mode=simulation_mode)
        self.battery = BatteryMonitor(simulation_mode=simulation_mode)
        self.obstacle_avoidance = ObstacleAvoidance(simulation_mode=simulation_mode)
        self.sim_bridge = SimulationBridge(use_airsim=not simulation_mode)
        self.telemetry = TelemetryPublisher(
            broker_host=mqtt_host, broker_port=mqtt_port, drone_id=drone_id,
        )

        # Mission state
        self._abort_requested = False
        self._on_status_change: Optional[Callable] = None

        # Subscribe to commands
        self.telemetry.subscribe_to_commands(self._handle_command)

        logger.info("FlightController init: drone=%s, sim=%s", drone_id, simulation_mode)

    def execute_mission(
        self,
        mission_id: str,
        waypoints: List[Dict[str, Any]],
        on_arrival: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Execute a complete delivery mission.

        Args:
            mission_id: Mission identifier.
            waypoints: Optimised route waypoints.
            on_arrival: Callback when drone arrives at destination.

        Returns:
            Mission result dict with status, metrics.
        """
        self.current_mission_id = mission_id
        self._abort_requested = False
        self._set_status(MissionStatus.DISPATCHED)

        result = {
            "mission_id": mission_id,
            "success": False,
            "status": "FAILED",
            "flight_time_s": 0.0,
            "distance_km": 0.0,
            "battery_used": 0.0,
        }

        start_time = time.time()
        start_battery = self.battery.percent

        try:
            # Pre-flight checks
            if not self._preflight_check():
                result["failure_reason"] = "Pre-flight check failed"
                return result

            # Takeoff
            self._set_status(MissionStatus.IN_FLIGHT)
            self.sim_bridge.takeoff(altitude=_CFG.CRUISE_ALTITUDE_M)

            # Start telemetry publishing
            self.telemetry.start_continuous_publish(
                lambda: self._build_telemetry_packet()
            )

            # Navigate through waypoints
            for i, wp in enumerate(waypoints):
                if self._abort_requested:
                    logger.warning("Mission ABORTED at waypoint %d", i)
                    self.sim_bridge.return_to_home()
                    self._set_status(MissionStatus.ABORTED)
                    result["status"] = "ABORTED"
                    break

                logger.info("Flying to waypoint %d/%d", i + 1, len(waypoints))
                self.sim_bridge.fly_to_waypoint(
                    wp["lat"], wp["long"],
                    wp.get("altitude", _CFG.CRUISE_ALTITUDE_M),
                )
                self.battery.update(0.5)

                # Check battery
                if self.battery.percent < _CFG.CRITICAL_BATTERY_THRESHOLD * 100:
                    logger.critical("Battery critical — returning home")
                    self.sim_bridge.return_to_home()
                    result["failure_reason"] = "Battery critical"
                    self._set_status(MissionStatus.FAILED)
                    break

                if i == len(waypoints) - 1:
                    self._set_status(MissionStatus.APPROACHING)

            else:
                # Successfully navigated all waypoints
                self._set_status(MissionStatus.LANDING)
                self.sim_bridge.hover()

                if on_arrival:
                    on_arrival()

                self._set_status(MissionStatus.QR_PENDING)
                result["success"] = True
                result["status"] = "QR_PENDING"

        except Exception as exc:
            logger.error("Mission execution error: %s", exc)
            result["failure_reason"] = str(exc)
            self._set_status(MissionStatus.FAILED)

        finally:
            result["flight_time_s"] = time.time() - start_time
            result["battery_used"] = start_battery - self.battery.percent
            self.telemetry.stop()

        return result

    def abort_mission(self) -> None:
        self._abort_requested = True
        logger.warning("Mission abort requested")

    def return_home(self) -> None:
        self.sim_bridge.return_to_home()
        self.sim_bridge.land()
        self._set_status(MissionStatus.RETURNED)

    def _preflight_check(self) -> bool:
        checks = {
            "battery": self.battery.percent >= _CFG.LOW_BATTERY_THRESHOLD * 100,
            "gps": self.gps.get_current_position() is not None,
            "obstacles_clear": not self.obstacle_avoidance.is_critical(),
        }
        all_pass = all(checks.values())
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            logger.info("Preflight %s: %s", check, status)
        return all_pass

    def _build_telemetry_packet(self) -> TelemetryPacket:
        state = self.sim_bridge.get_state()
        return TelemetryPacket(
            drone_id=self.drone_id,
            mission_id=self.current_mission_id or "",
            timestamp=time.time(),
            latitude=state["lat"],
            longitude=state["long"],
            altitude=state["altitude"],
            speed_ms=state["speed"],
            heading_degrees=state.get("heading", 0.0),
            battery_percent=self.battery.percent,
            signal_strength=-45,
            status=self.status.value,
            obstacle_near=self.obstacle_avoidance.is_critical(),
            eta_seconds=0,
        )

    def _handle_command(self, command: Dict) -> None:
        cmd = command.get("command", "")
        if cmd == "abort":
            self.abort_mission()
        elif cmd == "return_home":
            self.return_home()
        elif cmd == "hover":
            self.sim_bridge.hover()
        logger.info("Command executed: %s", cmd)

    def _set_status(self, status: MissionStatus) -> None:
        self.status = status
        logger.info("Mission status → %s", status.value)
        if self._on_status_change:
            self._on_status_change(status)

    def on_status_change(self, callback: Callable) -> None:
        self._on_status_change = callback
