# -*- coding: utf-8 -*-
"""
MediReach — Payload Box Controller (GPIO Servo).

Controls the tamper-proof medication payload box via
Raspberry Pi GPIO, servo motor, LEDs, and buzzer.
"""

from __future__ import annotations

import time
from typing import Optional

from src.utils.constants import GPIOPins
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import RPi.GPIO as GPIO  # type: ignore[import-not-found]
    PI_AVAILABLE = True
except ImportError:
    PI_AVAILABLE = False


class PayloadBoxController:
    """Controls the tamper-proof medication payload box."""

    LOCKED_ANGLE = 0
    UNLOCKED_ANGLE = 90

    def __init__(self, simulation_mode: bool = False) -> None:
        self.simulation_mode = simulation_mode or not PI_AVAILABLE
        self.is_locked = True
        self.servo_pwm: Optional[object] = None

        if not self.simulation_mode:
            self._setup_gpio()
        else:
            logger.info("PayloadBox running in SIMULATION MODE")

    def _setup_gpio(self) -> None:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for pin in [GPIOPins.LED_LOCKED, GPIOPins.LED_UNLOCKED,
                     GPIOPins.LED_SCANNING, GPIOPins.BUZZER]:
            GPIO.setup(pin, GPIO.OUT)

        GPIO.setup(GPIOPins.SERVO_PWM, GPIO.OUT)
        self.servo_pwm = GPIO.PWM(GPIOPins.SERVO_PWM, 50)
        self.servo_pwm.start(0)

        GPIO.setup(GPIOPins.TAMPER_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(
            GPIOPins.TAMPER_SENSOR, GPIO.FALLING,
            callback=self._tamper_detected, bouncetime=300,
        )
        self._set_leds(locked=True)
        logger.info("GPIO setup complete for payload box")

    def lock(self) -> bool:
        success = self._set_servo_angle(self.LOCKED_ANGLE)
        if success:
            self.is_locked = True
            self._set_leds(locked=True)
            logger.info("Payload box LOCKED")
        return success

    def unlock(self, verified_order_id: str) -> bool:
        success = self._set_servo_angle(self.UNLOCKED_ANGLE)
        if success:
            self.is_locked = False
            self._set_leds(locked=False)
            self._sound_success_beep()
            logger.info("Payload box UNLOCKED for order: %s", verified_order_id)
        return success

    def scanning_mode(self) -> None:
        self._set_leds(scanning=True)

    def alert_wrong_qr(self) -> None:
        self._sound_error_beep()
        logger.warning("WRONG QR attempted")

    def _set_servo_angle(self, angle: int) -> bool:
        if not self.simulation_mode and self.servo_pwm:
            duty = angle / 18.0 + 2.0
            self.servo_pwm.ChangeDutyCycle(duty)
            time.sleep(0.5)
            self.servo_pwm.ChangeDutyCycle(0)
        else:
            logger.info("[SIM] Servo set to %d degrees", angle)
        return True

    def _set_leds(
        self, locked: Optional[bool] = None, scanning: bool = False
    ) -> None:
        if self.simulation_mode:
            state = "LOCKED" if locked else ("SCANNING" if scanning else "UNLOCKED")
            logger.debug("[SIM] LED state: %s", state)
            return

        if scanning:
            GPIO.output(GPIOPins.LED_LOCKED, GPIO.LOW)
            GPIO.output(GPIOPins.LED_UNLOCKED, GPIO.LOW)
            GPIO.output(GPIOPins.LED_SCANNING, GPIO.HIGH)
        elif locked:
            GPIO.output(GPIOPins.LED_LOCKED, GPIO.HIGH)
            GPIO.output(GPIOPins.LED_UNLOCKED, GPIO.LOW)
            GPIO.output(GPIOPins.LED_SCANNING, GPIO.LOW)
        else:
            GPIO.output(GPIOPins.LED_LOCKED, GPIO.LOW)
            GPIO.output(GPIOPins.LED_UNLOCKED, GPIO.HIGH)
            GPIO.output(GPIOPins.LED_SCANNING, GPIO.LOW)

    def _sound_success_beep(self) -> None:
        if self.simulation_mode:
            logger.debug("[SIM] Beep: success (3 short)")
            return
        for _ in range(3):
            GPIO.output(GPIOPins.BUZZER, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(GPIOPins.BUZZER, GPIO.LOW)
            time.sleep(0.1)

    def _sound_error_beep(self) -> None:
        if self.simulation_mode:
            logger.debug("[SIM] Beep: error (2 long)")
            return
        for _ in range(2):
            GPIO.output(GPIOPins.BUZZER, GPIO.HIGH)
            time.sleep(0.4)
            GPIO.output(GPIOPins.BUZZER, GPIO.LOW)
            time.sleep(0.2)

    def _tamper_detected(self, channel: int) -> None:
        logger.critical("TAMPER DETECTED on payload box!")
        self.lock()

    def cleanup(self) -> None:
        if not self.simulation_mode and PI_AVAILABLE:
            if self.servo_pwm:
                self.servo_pwm.stop()
            GPIO.cleanup()
            logger.info("Payload box GPIO cleanup complete")
