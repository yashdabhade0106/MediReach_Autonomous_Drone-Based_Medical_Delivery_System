# Hardware Integration Guide

This document maps the simulated hardware layer to the physical components that will be installed on the Raspberry Pi Zero 2W / 4B edge device. The system is designed such that swapping out simulated components for real ones only requires providing a new class that implements the interface.

## Abstraction Interfaces

All hardware interacts with the software through interfaces defined in `src/interfaces/hardware_interfaces.py`.

### 1. GPS Module
- **Simulated Class:** `SimulatedGPS`
- **Real Component:** Neo-6M GPS Module (via UART)
- **Interface to Implement:** `IGPS.get_location() -> Tuple[float, float, float]`
- **Wiring Expectation:** Connect to Raspberry Pi UART pins (TX/RX) and read NMEA sentences using `pynmea2`.

### 2. Ultrasonic / Obstacle Sensor
- **Simulated Class:** `SimulatedUltrasonic`
- **Real Component:** HC-SR04 Ultrasonic Distance Sensor
- **Interface to Implement:** `IUltrasonicSensor.get_distance_meters() -> float`
- **Wiring Expectation:** Connect Trigger to GPIO and Echo to GPIO (with voltage divider to protect the Pi's 3.3V pins). 

### 3. Payload Servo Box & LED
- **Simulated Class:** `SimulatedServoBox`
- **Real Component:** SG90 Micro Servo + RGB LED
- **Interface to Implement:** `IServoBox.unlock() -> bool`, `IServoBox.lock() -> bool`, `IServoBox.set_led_color(r, g, b) -> None`
- **Wiring Expectation:** Servo PWM control via GPIO. LED channels to PWM-capable GPIO pins.

### 4. Camera (CV Landing & QR Scanning)
- **Simulated Class:** `SimulatedCamera`
- **Real Component:** Raspberry Pi Camera Module 3 or USB Webcam
- **Interface to Implement:** `ICamera.get_frame() -> np.ndarray`
- **Wiring Expectation:** Connect via CSI port or USB. Use OpenCV `cv2.VideoCapture(0)` to feed frames into the CV model and QR verifier.

### 5. Telemetry Radio
- **Simulated Class:** `SimulatedTelemetryRadio`
- **Real Component:** LoRa SX1278 (SPI) or 4G LTE Hat (MQTT)
- **Interface to Implement:** `ITelemetryRadio.send_message(topic, payload) -> bool`
- **Wiring Expectation:** If LoRa, connect via SPI pins. If LTE, connect via USB/UART and use the MQTT client.

## How to Switch to Real Hardware

The software logic (e.g. `QRVerifier`, `RouteOptimizer`) does not know whether the hardware is real or simulated.
When physical components are ready:
1. Write `RealGPS`, `RealServoBox`, etc. in `src/hardware/` inheriting from the base interfaces.
2. Update the main application initialization (e.g. in `app.py` or a dependency injection container) to instantiate `RealX` instead of `SimulatedX`. No business logic needs to be rewritten!
