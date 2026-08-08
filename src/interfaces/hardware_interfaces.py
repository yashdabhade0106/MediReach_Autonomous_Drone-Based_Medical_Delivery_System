from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class IGPS(ABC):
    @abstractmethod
    def get_location(self) -> Tuple[float, float, float]:
        """Returns (lat, lng, alt)"""
        pass

class IUltrasonicSensor(ABC):
    @abstractmethod
    def get_distance_meters(self) -> float:
        """Returns distance in meters"""
        pass

class IServoBox(ABC):
    @abstractmethod
    def unlock(self) -> bool:
        pass

    @abstractmethod
    def lock(self) -> bool:
        pass

    @abstractmethod
    def set_led_color(self, r: int, g: int, b: int) -> None:
        pass

class ICamera(ABC):
    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """Returns the current camera frame as a numpy array"""
        pass

class ITelemetryRadio(ABC):
    @abstractmethod
    def send_message(self, topic: str, payload: dict) -> bool:
        """Sends a message via telemetry radio"""
        pass
