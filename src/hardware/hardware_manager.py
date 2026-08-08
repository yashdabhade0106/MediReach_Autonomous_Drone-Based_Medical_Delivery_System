import os
from typing import Dict, Any

from src.interfaces.hardware_interfaces import IGPS, IUltrasonicSensor, IServoBox, ICamera, ITelemetryRadio
from src.hardware.simulated_hardware import (
    SimulatedGPS, SimulatedUltrasonic, SimulatedServoBox, 
    SimulatedCamera, SimulatedTelemetryRadio
)
from src.hardware.real_hardware import (
    RealGPS, RealUltrasonic, RealServoBox, 
    RealCamera, RealTelemetryRadio
)

class HardwareManager:
    """
    Factory class to instantiate either Real or Simulated hardware components
    based on the 'HARDWARE_ENV' environment variable.
    """
    def __init__(self):
        self.env = os.getenv("HARDWARE_ENV", "virtual").lower()
        self.components: Dict[str, Any] = {}
        self._initialize_components()
        
    def _initialize_components(self):
        if self.env == "physical":
            print("Initializing PHYSICAL hardware components...")
            self.components['gps'] = RealGPS()
            self.components['ultrasonic'] = RealUltrasonic()
            self.components['servo_box'] = RealServoBox()
            self.components['camera'] = RealCamera()
            self.components['telemetry'] = RealTelemetryRadio()
        else:
            print("Initializing VIRTUAL/SIMULATED hardware components...")
            self.components['gps'] = SimulatedGPS(37.7749, -122.4194)
            self.components['ultrasonic'] = SimulatedUltrasonic()
            self.components['servo_box'] = SimulatedServoBox()
            self.components['camera'] = SimulatedCamera()
            self.components['telemetry'] = SimulatedTelemetryRadio()
            
    def get_gps(self) -> IGPS:
        return self.components['gps']
        
    def get_ultrasonic(self) -> IUltrasonicSensor:
        return self.components['ultrasonic']
        
    def get_servo_box(self) -> IServoBox:
        return self.components['servo_box']
        
    def get_camera(self) -> ICamera:
        return self.components['camera']
        
    def get_telemetry(self) -> ITelemetryRadio:
        return self.components['telemetry']

# Global singleton instance for easy access
hw_manager = HardwareManager()
