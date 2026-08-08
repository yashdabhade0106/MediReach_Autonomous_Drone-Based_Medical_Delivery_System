import time
import numpy as np
import cv2
from typing import Tuple
from src.interfaces.hardware_interfaces import IGPS, IUltrasonicSensor, IServoBox, ICamera, ITelemetryRadio

try:
    import RPi.GPIO as GPIO
    import serial
    import pynmea2
    import paho.mqtt.client as mqtt
except ImportError:
    GPIO = None
    serial = None
    pynmea2 = None
    mqtt = None

class RealGPS(IGPS):
    def __init__(self, port: str = '/dev/serial0', baudrate: int = 9600):
        if serial is None:
            raise ImportError("serial module is not installed or available.")
        self.ser = serial.Serial(port, baudrate, timeout=1)

    def get_location(self) -> Tuple[float, float, float]:
        """Reads NMEA sentences from the UART port."""
        try:
            line = self.ser.readline().decode('ascii', errors='replace')
            if line.startswith('$GPGGA') or line.startswith('$GNRMC'):
                msg = pynmea2.parse(line)
                return (msg.latitude, msg.longitude, getattr(msg, 'altitude', 0.0))
        except Exception as e:
            print(f"GPS Error: {e}")
        return (0.0, 0.0, 0.0)

class RealUltrasonic(IUltrasonicSensor):
    def __init__(self, trigger_pin: int = 23, echo_pin: int = 24):
        if GPIO is None:
            raise ImportError("RPi.GPIO is not installed or available.")
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trigger_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)

    def get_distance_meters(self) -> float:
        """Ping the ultrasonic sensor and return distance in meters."""
        GPIO.output(self.trigger_pin, True)
        time.sleep(0.00001)
        GPIO.output(self.trigger_pin, False)
        
        start_time = time.time()
        stop_time = time.time()
        
        while GPIO.input(self.echo_pin) == 0:
            start_time = time.time()
            
        while GPIO.input(self.echo_pin) == 1:
            stop_time = time.time()
            
        time_elapsed = stop_time - start_time
        distance_cm = (time_elapsed * 34300) / 2
        return distance_cm / 100.0

class RealServoBox(IServoBox):
    def __init__(self, servo_pin: int = 18, led_r: int = 17, led_g: int = 27, led_b: int = 22):
        if GPIO is None:
            raise ImportError("RPi.GPIO is not installed or available.")
        self.servo_pin = servo_pin
        self.led_r = led_r
        self.led_g = led_g
        self.led_b = led_b
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.servo_pin, GPIO.OUT)
        GPIO.setup(self.led_r, GPIO.OUT)
        GPIO.setup(self.led_g, GPIO.OUT)
        GPIO.setup(self.led_b, GPIO.OUT)
        
        self.pwm = GPIO.PWM(self.servo_pin, 50) # 50Hz
        self.pwm.start(0)
        self.is_locked = True

    def unlock(self) -> bool:
        """Rotates servo to unlock position."""
        self.pwm.ChangeDutyCycle(12) # ~180 degrees
        time.sleep(0.5)
        self.pwm.ChangeDutyCycle(0)
        self.is_locked = False
        return True

    def lock(self) -> bool:
        """Rotates servo to locked position."""
        self.pwm.ChangeDutyCycle(2) # ~0 degrees
        time.sleep(0.5)
        self.pwm.ChangeDutyCycle(0)
        self.is_locked = True
        return True

    def set_led_color(self, r: int, g: int, b: int) -> None:
        """Set RGB LED color using digital output (simplistic mode)."""
        GPIO.output(self.led_r, GPIO.HIGH if r > 127 else GPIO.LOW)
        GPIO.output(self.led_g, GPIO.HIGH if g > 127 else GPIO.LOW)
        GPIO.output(self.led_b, GPIO.HIGH if b > 127 else GPIO.LOW)

class RealCamera(ICamera):
    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        
    def get_frame(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            print("Camera error: failed to capture frame.")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return frame
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

class RealTelemetryRadio(ITelemetryRadio):
    def __init__(self, broker: str = "broker.emqx.io", port: int = 1883):
        if mqtt is None:
            raise ImportError("paho-mqtt is not installed.")
        self.client = mqtt.Client()
        self.client.connect(broker, port, 60)
        self.client.loop_start()

    def send_message(self, topic: str, payload: dict) -> bool:
        import json
        self.client.publish(topic, json.dumps(payload))
        return True
    
    def __del__(self):
        if hasattr(self, 'client'):
            self.client.loop_stop()
            self.client.disconnect()
