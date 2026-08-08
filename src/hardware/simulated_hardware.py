from typing import Tuple, Optional
import numpy as np
import time
from src.interfaces.hardware_interfaces import IGPS, IUltrasonicSensor, IServoBox, ICamera, ITelemetryRadio

class SimulatedGPS(IGPS):
    def __init__(self, start_lat: float, start_lng: float, start_alt: float = 0.0):
        self.lat = start_lat
        self.lng = start_lng
        self.alt = start_alt

    def get_location(self) -> Tuple[float, float, float]:
        # Simulate slight movement
        self.lat += 0.00001
        self.lng += 0.00001
        return (self.lat, self.lng, self.alt)

class SimulatedUltrasonic(IUltrasonicSensor):
    def __init__(self, default_distance: float = 10.0):
        self.distance = default_distance

    def get_distance_meters(self) -> float:
        # Simulate distance decreasing as drone lands
        if self.distance > 0:
            self.distance -= 0.5
        return max(0.0, self.distance)

class SimulatedServoBox(IServoBox):
    def __init__(self):
        self.is_locked = True
        self.led_color = (0, 0, 0)

    def unlock(self) -> bool:
        print("SimulatedServoBox: Unlocked")
        self.is_locked = False
        return True

    def lock(self) -> bool:
        print("SimulatedServoBox: Locked")
        self.is_locked = True
        return True

    def set_led_color(self, r: int, g: int, b: int) -> None:
        self.led_color = (r, g, b)
        print(f"SimulatedServoBox: LED set to RGB({r}, {g}, {b})")

class SimulatedCamera(ICamera):
    def get_frame(self) -> np.ndarray:
        # Return a dummy black frame
        print("SimulatedCamera: Captured frame")
        return np.zeros((480, 640, 3), dtype=np.uint8)

class SimulatedTelemetryRadio(ITelemetryRadio):
    def send_message(self, topic: str, payload: dict) -> bool:
        print(f"SimulatedTelemetryRadio: Sent message on '{topic}': {payload}")
        return True
