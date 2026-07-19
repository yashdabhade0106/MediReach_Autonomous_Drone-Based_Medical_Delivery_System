# -*- coding: utf-8 -*-
"""
MediReach — Edge Inference for Raspberry Pi.

Optimised ONNX Runtime inference targeting ≥5 FPS on Pi 4B.
Includes GPIO LED signalling and MQTT result publishing.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from src.utils.constants import CVConfig, GPIOPins
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CFG = CVConfig()

# Conditional GPIO import
try:
    import RPi.GPIO as GPIO  # type: ignore[import-not-found]
    PI_AVAILABLE = True
except ImportError:
    PI_AVAILABLE = False


class EdgeInference:
    """Pi-optimised landing zone detection using ONNX Runtime.

    Provides low-latency inference with GPIO LED feedback
    and MQTT result publishing for integration with the
    drone controller.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = _CFG.CONFIDENCE_THRESHOLD,
        use_gpio: bool = True,
        input_size: int = _CFG.IMAGE_SIZE,
    ) -> None:
        """Initialise ONNX inference engine.

        Args:
            model_path: Path to .onnx model file.
            confidence_threshold: Minimum detection confidence.
            use_gpio: Whether to use GPIO LEDs for signalling.
            input_size: Model input resolution.

        Raises:
            ImportError: If onnxruntime is not installed.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime required: pip install onnxruntime"
            ) from exc

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.conf_threshold = confidence_threshold
        self.input_size = input_size
        self.use_gpio = use_gpio and PI_AVAILABLE
        self._input_name = self.session.get_inputs()[0].name

        if self.use_gpio:
            self._setup_gpio()

        logger.info(
            "EdgeInference ready: model=%s, size=%d, gpio=%s",
            model_path, input_size, self.use_gpio,
        )

    def _setup_gpio(self) -> None:
        """Set up GPIO pins for LED indicators."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        GPIO.setup(GPIOPins.LED_SAFE_ZONE, GPIO.OUT)
        GPIO.setup(GPIOPins.LED_UNSAFE_ZONE, GPIO.OUT)
        GPIO.setup(GPIOPins.LED_SCANNING, GPIO.OUT)

        # Turn on scanning LED
        GPIO.output(GPIOPins.LED_SCANNING, GPIO.HIGH)
        logger.info("GPIO pins configured for edge inference")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalise frame for ONNX input.

        Args:
            frame: Input BGR image.

        Returns:
            Preprocessed array of shape (1, 3, H, W) float32.
        """
        # Resize with aspect-preserving letterbox
        h, w = frame.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas
        canvas = np.full(
            (self.input_size, self.input_size, 3), 114, dtype=np.uint8
        )
        pad_y = (self.input_size - new_h) // 2
        pad_x = (self.input_size - new_w) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Normalise and transpose to (1, C, H, W)
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC → CHW
        blob = np.expand_dims(blob, axis=0)  # Add batch dim

        return blob

    def infer(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run ONNX inference and parse detections.

        Args:
            frame: Input BGR image.

        Returns:
            List of detection dictionaries with keys:
            class_id, class_name, confidence, bbox (xyxy).
        """
        blob = self.preprocess(frame)
        outputs = self.session.run(None, {self._input_name: blob})

        detections = self._parse_outputs(outputs, frame.shape[:2])
        return detections

    def _parse_outputs(
        self,
        outputs: List[np.ndarray],
        original_shape: tuple,
    ) -> List[Dict[str, Any]]:
        """Parse ONNX output tensor into detection list.

        Args:
            outputs: Raw ONNX model outputs.
            original_shape: (height, width) of original frame.

        Returns:
            List of parsed detection dictionaries.
        """
        detections: List[Dict[str, Any]] = []

        if not outputs or outputs[0] is None:
            return detections

        # YOLOv8 output: (1, num_classes+4, num_detections)
        output = outputs[0]
        if len(output.shape) == 3:
            output = output[0].T  # Transpose to (num_det, num_classes+4)

        class_names = list(_CFG.ALL_CLASSES)
        safe_classes = set(_CFG.SAFE_CLASSES)

        for row in output:
            if len(row) < 5:
                continue

            # First 4 values: cx, cy, w, h
            cx, cy, w, h = row[:4]
            class_scores = row[4:]

            if len(class_scores) == 0:
                continue

            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < self.conf_threshold:
                continue

            # Convert to xyxy
            x1 = int((cx - w / 2) * original_shape[1] / self.input_size)
            y1 = int((cy - h / 2) * original_shape[0] / self.input_size)
            x2 = int((cx + w / 2) * original_shape[1] / self.input_size)
            y2 = int((cy + h / 2) * original_shape[0] / self.input_size)

            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "is_safe": class_name in safe_classes,
            })

        return detections

    def signal_landing_zone(self, is_safe: bool) -> None:
        """Flash appropriate LED based on detection result.

        Args:
            is_safe: Whether the detected zone is safe.
        """
        if not self.use_gpio:
            logger.debug("[SIM] LED signal: safe=%s", is_safe)
            return

        GPIO.output(GPIOPins.LED_SCANNING, GPIO.LOW)

        if is_safe:
            GPIO.output(GPIOPins.LED_SAFE_ZONE, GPIO.HIGH)
            GPIO.output(GPIOPins.LED_UNSAFE_ZONE, GPIO.LOW)
        else:
            GPIO.output(GPIOPins.LED_SAFE_ZONE, GPIO.LOW)
            GPIO.output(GPIOPins.LED_UNSAFE_ZONE, GPIO.HIGH)

    def run_continuous(
        self,
        camera_index: int = 0,
        on_detection: Optional[Callable[[List[Dict]], None]] = None,
        max_frames: int = 0,
    ) -> None:
        """Main loop for continuous detection on Raspberry Pi.

        Args:
            camera_index: Camera device index.
            on_detection: Optional callback for each detection result.
            max_frames: Maximum frames to process (0 = unlimited).
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("Cannot open camera %d", camera_index)
            return

        logger.info("Starting continuous edge inference...")
        frame_count = 0
        fps_start = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Camera read failed")
                    break

                detections = self.infer(frame)
                safe_detections = [d for d in detections if d["is_safe"]]

                if safe_detections:
                    self.signal_landing_zone(True)
                elif detections:
                    self.signal_landing_zone(False)

                if on_detection:
                    on_detection(detections)

                frame_count += 1

                # FPS logging every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_start
                    fps = frame_count / max(elapsed, 0.001)
                    logger.info("Edge inference FPS: %.1f", fps)

                if 0 < max_frames <= frame_count:
                    break

        except KeyboardInterrupt:
            logger.info("Edge inference stopped by user")
        finally:
            cap.release()
            self.cleanup()

    def cleanup(self) -> None:
        """Release camera and GPIO resources."""
        if self.use_gpio:
            GPIO.output(GPIOPins.LED_SAFE_ZONE, GPIO.LOW)
            GPIO.output(GPIOPins.LED_UNSAFE_ZONE, GPIO.LOW)
            GPIO.output(GPIOPins.LED_SCANNING, GPIO.LOW)
            GPIO.cleanup()
            logger.info("GPIO cleanup complete")
