# -*- coding: utf-8 -*-
"""
MediReach — LoRa Communication (Simulated).

Simulated LoRa radio communication for long-range
telemetry and command relay in areas without WiFi/4G.
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LoRaPacket:
    """LoRa radio packet."""
    sender_id: str
    receiver_id: str
    payload: Dict[str, Any]
    rssi: int  # dBm
    snr: float
    timestamp: float
    sequence: int


class LoRaComms:
    """Simulated LoRa communication for drone telemetry.

    In production, this would interface with an SX1278/SX1276
    LoRa module via SPI on the Raspberry Pi.
    """

    def __init__(
        self,
        device_id: str = "DRN-001",
        frequency_mhz: float = 433.0,
        spreading_factor: int = 7,
        bandwidth_khz: float = 125.0,
        simulation_mode: bool = True,
    ) -> None:
        self.device_id = device_id
        self.frequency = frequency_mhz
        self.sf = spreading_factor
        self.bw = bandwidth_khz
        self.simulation_mode = simulation_mode
        self._sequence = 0
        self._rx_callbacks: List[Callable] = []
        self._rx_buffer: List[LoRaPacket] = []
        self._running = False
        logger.info("LoRaComms init: id=%s, freq=%.1f MHz, SF=%d", device_id, frequency_mhz, spreading_factor)

    def send(self, receiver_id: str, payload: Dict[str, Any]) -> bool:
        self._sequence += 1
        packet = LoRaPacket(
            sender_id=self.device_id, receiver_id=receiver_id,
            payload=payload, rssi=-45, snr=9.5,
            timestamp=time.time(), sequence=self._sequence,
        )
        if self.simulation_mode:
            logger.debug("LoRa TX → %s: %s", receiver_id, json.dumps(payload)[:100])
            self._rx_buffer.append(packet)
            return True
        return False

    def receive(self, timeout_s: float = 5.0) -> Optional[LoRaPacket]:
        start = time.time()
        while time.time() - start < timeout_s:
            if self._rx_buffer:
                return self._rx_buffer.pop(0)
            time.sleep(0.1)
        return None

    def register_callback(self, callback: Callable[[LoRaPacket], None]) -> None:
        self._rx_callbacks.append(callback)

    def start_listening(self) -> None:
        self._running = True
        thread = threading.Thread(target=self._listen_loop, daemon=True)
        thread.start()

    def _listen_loop(self) -> None:
        while self._running:
            if self._rx_buffer:
                pkt = self._rx_buffer.pop(0)
                for cb in self._rx_callbacks:
                    try:
                        cb(pkt)
                    except Exception as exc:
                        logger.error("LoRa RX callback error: %s", exc)
            time.sleep(0.1)

    def stop(self) -> None:
        self._running = False
        logger.info("LoRa comms stopped")

    def get_signal_strength(self) -> int:
        return -45 if self.simulation_mode else -80
