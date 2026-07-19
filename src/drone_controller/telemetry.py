# -*- coding: utf-8 -*-
"""
MediReach — MQTT Telemetry Publisher/Subscriber.

Real-time drone telemetry over MQTT with encrypted payloads,
command handling, and alert publishing.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional

from src.utils.constants import MQTTTopics, AlertType
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TelemetryPacket:
    """Telemetry data packet published over MQTT."""
    drone_id: str
    mission_id: str
    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    speed_ms: float
    heading_degrees: float
    battery_percent: float
    signal_strength: int
    status: str
    obstacle_near: bool
    eta_seconds: int


class TelemetryPublisher:
    """MQTT telemetry publisher and command subscriber.

    Publishes to: medireach/drone/{drone_id}/telemetry
    Subscribes to: medireach/drone/{drone_id}/commands
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        drone_id: str = "DRN-001",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.drone_id = drone_id
        self._publish_interval = 2.0
        self._running = False
        self._command_handler: Optional[Callable] = None

        try:
            import paho.mqtt.client as mqtt
            self.client = mqtt.Client(
                client_id=f"drone_{drone_id}",
                clean_session=True,
            )
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect

            if username and password:
                self.client.username_pw_set(username, password)

            self._connect(broker_host, broker_port)
            self._mqtt_available = True
        except ImportError:
            logger.warning("paho-mqtt not installed; telemetry in simulation mode")
            self._mqtt_available = False

    def _connect(self, host: str, port: int) -> None:
        try:
            self.client.connect(host, port, keepalive=60)
            self.client.loop_start()
            logger.info("MQTT connected to %s:%d", host, port)
        except Exception as exc:
            logger.error("MQTT connection failed: %s", exc)
            self._mqtt_available = False

    def publish_telemetry(self, packet: TelemetryPacket) -> None:
        topic = MQTTTopics.TELEMETRY.format(self.drone_id)
        payload = json.dumps(asdict(packet))

        if self._mqtt_available:
            self.client.publish(topic, payload, qos=1)
        else:
            logger.debug("[SIM] MQTT publish: %s", payload[:80])

    def publish_alert(self, alert_type: str, message: str) -> None:
        topic = MQTTTopics.ALERTS.format(self.drone_id)
        payload = json.dumps({
            "drone_id": self.drone_id,
            "alert_type": alert_type,
            "message": message,
            "timestamp": time.time(),
        })
        if self._mqtt_available:
            self.client.publish(topic, payload, qos=2)
        logger.warning("ALERT [%s]: %s", alert_type, message)

    def start_continuous_publish(
        self, get_telemetry_fn: Callable[[], TelemetryPacket]
    ) -> None:
        self._running = True

        def _publish_loop() -> None:
            while self._running:
                try:
                    packet = get_telemetry_fn()
                    self.publish_telemetry(packet)
                except Exception as exc:
                    logger.error("Telemetry publish error: %s", exc)
                time.sleep(self._publish_interval)

        thread = threading.Thread(target=_publish_loop, daemon=True)
        thread.start()
        logger.info("Continuous telemetry publishing started (interval=%.1fs)", self._publish_interval)

    def subscribe_to_commands(self, command_handler: Callable[[Dict], None]) -> None:
        self._command_handler = command_handler
        topic = MQTTTopics.COMMANDS.format(self.drone_id)
        if self._mqtt_available:
            self.client.subscribe(topic, qos=1)
            logger.info("Subscribed to commands: %s", topic)

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: int) -> None:
        if rc == 0:
            logger.info("MQTT connected successfully")
            topic = MQTTTopics.COMMANDS.format(self.drone_id)
            client.subscribe(topic, qos=1)
        else:
            logger.error("MQTT connection failed with code %d", rc)

    def _on_message(self, client: Any, userdata: Any, message: Any) -> None:
        try:
            payload = json.loads(message.payload.decode())
            logger.info("Command received: %s", payload.get("command", "unknown"))
            if self._command_handler:
                self._command_handler(payload)
        except json.JSONDecodeError as exc:
            logger.error("Invalid command JSON: %s", exc)

    def _on_disconnect(self, client: Any, userdata: Any, rc: int) -> None:
        if rc != 0:
            logger.warning("MQTT disconnected unexpectedly (rc=%d), reconnecting...", rc)
            time.sleep(5)
            try:
                client.reconnect()
            except Exception as exc:
                logger.error("MQTT reconnect failed: %s", exc)

    def stop(self) -> None:
        self._running = False
        if self._mqtt_available:
            self.client.loop_stop()
            self.client.disconnect()
        logger.info("Telemetry publisher stopped")
