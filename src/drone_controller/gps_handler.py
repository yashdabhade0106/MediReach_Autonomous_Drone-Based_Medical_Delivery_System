# -*- coding: utf-8 -*-
"""
MediReach — GPS Module Interface.

Interface for NEO-6M GPS via UART on Raspberry Pi.
Includes simulation mode for development without hardware.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.utils.geo_utils import haversine_distance, calculate_bearing
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GPSReading:
    """A single GPS fix reading."""
    latitude: float
    longitude: float
    altitude: float
    speed_knots: float
    heading: float
    satellites: int
    fix_quality: int
    timestamp: float
    is_valid: bool


class GPSHandler:
    """GPS module interface with simulation fallback.

    Reads NMEA sentences from serial UART on Pi, or simulates
    GPS movement through waypoints for development.
    """

    def __init__(
        self,
        port: str = "/dev/ttyAMA0",
        baud: int = 9600,
        simulation_mode: bool = False,
    ) -> None:
        self.simulation_mode = simulation_mode
        self._last_reading: Optional[GPSReading] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        if not simulation_mode:
            try:
                import serial
                self.serial = serial.Serial(port, baud, timeout=1)
                self._start_reading_thread()
                logger.info("GPS handler started on %s @ %d baud", port, baud)
            except (ImportError, Exception) as exc:
                logger.warning("GPS hardware unavailable (%s), switching to simulation", exc)
                self.simulation_mode = True
                self._init_sim()
        else:
            self._init_sim()

    def _init_sim(self) -> None:
        self._sim_lat = 18.5204
        self._sim_lon = 73.8567
        self._sim_alt = 0.0
        self._last_reading = GPSReading(
            latitude=self._sim_lat, longitude=self._sim_lon,
            altitude=self._sim_alt, speed_knots=0.0, heading=0.0,
            satellites=8, fix_quality=1, timestamp=time.time(), is_valid=True,
        )
        logger.info("GPS handler in SIMULATION mode")

    def get_current_position(self) -> Optional[GPSReading]:
        with self._lock:
            return self._last_reading

    def wait_for_fix(self, timeout_seconds: int = 60) -> bool:
        start = time.time()
        while time.time() - start < timeout_seconds:
            reading = self.get_current_position()
            if reading and reading.is_valid and reading.fix_quality >= 1:
                logger.info("GPS fix acquired: %d satellites", reading.satellites)
                return True
            time.sleep(1.0)
        logger.warning("GPS fix timeout after %ds", timeout_seconds)
        return False

    def simulate_movement(
        self, waypoints: List[Dict[str, float]], speed_ms: float = 15.0
    ) -> None:
        if not self.simulation_mode:
            logger.warning("simulate_movement called in hardware mode")
            return

        def _move() -> None:
            for i, wp in enumerate(waypoints):
                target_lat = wp["lat"]
                target_lon = wp["long"]
                target_alt = wp.get("altitude", 50.0)

                while True:
                    dist = haversine_distance(self._sim_lat, self._sim_lon, target_lat, target_lon)
                    if dist < 5.0:
                        break

                    bearing = calculate_bearing(self._sim_lat, self._sim_lon, target_lat, target_lon)
                    step_m = speed_ms * 0.5
                    dlat = (step_m / 111320.0) * math.cos(math.radians(bearing))
                    dlon = (step_m / (111320.0 * math.cos(math.radians(self._sim_lat)))) * math.sin(math.radians(bearing))

                    self._sim_lat += dlat
                    self._sim_lon += dlon
                    self._sim_alt += (target_alt - self._sim_alt) * 0.1

                    with self._lock:
                        self._last_reading = GPSReading(
                            latitude=self._sim_lat, longitude=self._sim_lon,
                            altitude=self._sim_alt, speed_knots=speed_ms * 1.944,
                            heading=bearing, satellites=10, fix_quality=1,
                            timestamp=time.time(), is_valid=True,
                        )
                    time.sleep(0.5)

                logger.debug("Sim waypoint %d reached: (%.4f, %.4f)", i, target_lat, target_lon)

        thread = threading.Thread(target=_move, daemon=True)
        thread.start()

    def _parse_nmea(self, sentence: str) -> Optional[GPSReading]:
        try:
            import pynmea2
            msg = pynmea2.parse(sentence)
            if isinstance(msg, pynmea2.types.talker.GGA):
                return GPSReading(
                    latitude=msg.latitude, longitude=msg.longitude,
                    altitude=float(msg.altitude or 0), speed_knots=0.0,
                    heading=0.0, satellites=int(msg.num_sats or 0),
                    fix_quality=int(msg.gps_qual or 0),
                    timestamp=time.time(), is_valid=int(msg.gps_qual or 0) >= 1,
                )
        except (ImportError, Exception) as exc:
            logger.debug("NMEA parse error: %s", exc)
        return None

    def _reading_thread(self) -> None:
        while self._running:
            try:
                line = self.serial.readline().decode("ascii", errors="replace").strip()
                if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                    reading = self._parse_nmea(line)
                    if reading:
                        with self._lock:
                            self._last_reading = reading
            except Exception as exc:
                logger.debug("GPS read error: %s", exc)
            time.sleep(0.1)

    def _start_reading_thread(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._reading_thread, daemon=True)
        self._thread.start()

    def calculate_bearing_to(self, target_lat: float, target_long: float) -> float:
        pos = self.get_current_position()
        if pos is None:
            return 0.0
        return calculate_bearing(pos.latitude, pos.longitude, target_lat, target_long)

    def is_at_destination(
        self, target_lat: float, target_long: float, tolerance_meters: float = 10.0
    ) -> bool:
        pos = self.get_current_position()
        if pos is None:
            return False
        return haversine_distance(pos.latitude, pos.longitude, target_lat, target_long) <= tolerance_meters

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if not self.simulation_mode and hasattr(self, "serial"):
            self.serial.close()
        logger.info("GPS handler stopped")
