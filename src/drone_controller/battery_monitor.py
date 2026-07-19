# -*- coding: utf-8 -*-
"""
MediReach — Battery Monitor.

Tracks battery state, estimates remaining flight time,
and raises alerts at configurable thresholds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from src.utils.constants import DroneConfig, AlertType
from src.utils.logger import get_logger

logger = get_logger(__name__)
_CFG = DroneConfig()


@dataclass
class BatteryState:
    """Current battery state."""
    percent: float
    voltage: float
    current_draw_ma: float
    temperature_c: float
    estimated_remaining_s: float
    is_charging: bool
    cycles: int


class BatteryMonitor:
    """Monitors drone battery state and raises alerts."""

    def __init__(
        self, capacity_mah: int = _CFG.BATTERY_CAPACITY_MAH,
        simulation_mode: bool = True,
    ) -> None:
        self.capacity_mah = capacity_mah
        self.simulation_mode = simulation_mode
        self._percent = 100.0
        self._voltage = 16.8  # 4S LiPo fully charged
        self._drain_rate = 0.0
        self._alert_callbacks: List[Callable] = []
        self._low_warned = False
        self._critical_warned = False
        logger.info("BatteryMonitor init: %d mAh, sim=%s", capacity_mah, simulation_mode)

    @property
    def percent(self) -> float:
        return self._percent

    def get_state(self) -> BatteryState:
        voltage = self._percent / 100.0 * 4.2 * 4  # 4S LiPo
        return BatteryState(
            percent=self._percent, voltage=round(voltage, 2),
            current_draw_ma=self._drain_rate * 1000,
            temperature_c=35.0 + (100 - self._percent) * 0.1,
            estimated_remaining_s=self._estimate_remaining(),
            is_charging=False, cycles=0,
        )

    def update(self, drain_percent: float) -> None:
        self._percent = max(0.0, self._percent - drain_percent)
        self._drain_rate = drain_percent
        self._check_thresholds()

    def set_percent(self, percent: float) -> None:
        self._percent = max(0.0, min(100.0, percent))
        self._check_thresholds()

    def can_complete_mission(self, estimated_usage_percent: float) -> bool:
        reserve = _CFG.RETURN_BATTERY_RESERVE * 100
        return self._percent >= estimated_usage_percent + reserve

    def register_alert_callback(self, callback: Callable) -> None:
        self._alert_callbacks.append(callback)

    def _check_thresholds(self) -> None:
        if self._percent <= _CFG.CRITICAL_BATTERY_THRESHOLD * 100 and not self._critical_warned:
            self._critical_warned = True
            logger.critical("BATTERY CRITICAL: %.1f%%", self._percent)
            self._fire_alert(AlertType.BATTERY_CRITICAL)
        elif self._percent <= _CFG.LOW_BATTERY_THRESHOLD * 100 and not self._low_warned:
            self._low_warned = True
            logger.warning("Battery LOW: %.1f%%", self._percent)
            self._fire_alert(AlertType.BATTERY_LOW)

    def _fire_alert(self, alert_type: AlertType) -> None:
        for cb in self._alert_callbacks:
            try:
                cb(alert_type, self.get_state())
            except Exception as exc:
                logger.error("Alert callback error: %s", exc)

    def _estimate_remaining(self) -> float:
        if self._drain_rate <= 0:
            return float("inf")
        return (self._percent / self._drain_rate) * 2.0  # seconds at current rate

    def reset(self) -> None:
        self._percent = 100.0
        self._low_warned = False
        self._critical_warned = False
        logger.info("Battery reset to 100%%")
