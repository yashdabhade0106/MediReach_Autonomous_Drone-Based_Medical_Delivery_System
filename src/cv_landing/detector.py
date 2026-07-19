# -*- coding: utf-8 -*-
"""
MediReach — YOLOv8-Based Landing Zone Detection.

Detects safe landing surfaces in real-time from the drone's
downward-facing camera.  Classes include both safe (flat ground,
driveway, rooftop) and unsafe (water, crowd, slope) surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.constants import CVConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CFG = CVConfig()


@dataclass
class LandingZone:
    """Detection result for a single landing zone."""
    zone_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]          # (cx, cy)
    area_pixels: int
    is_safe: bool
    safety_score: float
    recommendation: str              # "land" | "hover" | "abort"


class LandingZoneDetector:
    """YOLOv8-based landing zone detection for MediReach drones.

    Detects and classifies landing surfaces, computes safety
    scores, and provides landing recommendations.
    """

    SAFE_CLASSES = list(_CFG.SAFE_CLASSES)
    UNSAFE_CLASSES = list(_CFG.UNSAFE_CLASSES)

    # Priority order for safe zone selection (lower index = higher priority)
    ZONE_PRIORITY = {
        "safe_flat_ground": 0,
        "driveway": 1,
        "rooftop": 2,
        "balcony": 3,
    }

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = _CFG.CONFIDENCE_THRESHOLD,
    ) -> None:
        """Initialise the detector with a YOLOv8 model.

        Args:
            model_path: Path to YOLO .pt or .onnx model file.
            confidence_threshold: Minimum confidence for detections.

        Raises:
            ImportError: If ultralytics is not installed.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics package required: pip install ultralytics"
            ) from exc

        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        logger.info(
            "LandingZoneDetector loaded: model=%s, conf=%.2f",
            model_path, confidence_threshold,
        )

    def detect(self, frame: np.ndarray) -> List[LandingZone]:
        """Run detection on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, C).

        Returns:
            List of LandingZone objects sorted by confidence (desc).
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        zones: List[LandingZone] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                zone = self._parse_detection(box, frame)
                if zone is not None:
                    zones.append(zone)

        zones.sort(key=lambda z: z.confidence, reverse=True)
        return zones

    def get_best_landing_zone(
        self, frame: np.ndarray
    ) -> Optional[LandingZone]:
        """Return the single best landing zone or None.

        Priority: safe_flat_ground > driveway > rooftop > balcony.
        Among same type, highest confidence wins.

        Args:
            frame: BGR image.

        Returns:
            Best LandingZone or None if no safe zone detected.
        """
        zones = self.detect(frame)
        safe_zones = [z for z in zones if z.is_safe]

        if not safe_zones:
            logger.debug("No safe landing zone detected")
            return None

        # Sort by priority, then confidence
        safe_zones.sort(
            key=lambda z: (
                self.ZONE_PRIORITY.get(z.zone_type, 99),
                -z.confidence,
            )
        )

        best = safe_zones[0]
        logger.debug(
            "Best zone: %s (conf=%.2f, score=%.2f)",
            best.zone_type, best.confidence, best.safety_score,
        )
        return best

    def annotate_frame(
        self,
        frame: np.ndarray,
        zones: List[LandingZone],
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and safety indicators on frame.

        Green = safe, Red = unsafe, Yellow = low confidence.

        Args:
            frame: Original BGR image.
            zones: List of detected zones.

        Returns:
            Annotated image copy.
        """
        annotated = frame.copy()

        for zone in zones:
            x1, y1, x2, y2 = zone.bbox

            # Colour based on safety
            if zone.is_safe:
                colour = (0, 255, 0)  # Green
            elif zone.confidence < 0.5:
                colour = (0, 255, 255)  # Yellow
            else:
                colour = (0, 0, 255)  # Red

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

            # Label
            label = (
                f"{zone.zone_type} {zone.confidence:.0%} "
                f"[{zone.recommendation.upper()}]"
            )
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 8),
                (x1 + label_size[0], y1),
                colour, -1,
            )
            cv2.putText(
                annotated, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

            # Center crosshair
            cx, cy = zone.center
            cv2.drawMarker(
                annotated, (cx, cy), colour,
                cv2.MARKER_CROSS, 15, 2,
            )

        return annotated

    def process_video_stream(
        self,
        source: int = 0,
    ) -> Generator[Tuple[np.ndarray, Optional[LandingZone]], None, None]:
        """Generator for real-time camera feed processing.

        Yields (annotated_frame, best_zone) tuples.

        Args:
            source: Camera device index.

        Yields:
            Tuple of (annotated frame, best LandingZone or None).
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error("Failed to open camera source %s", source)
            return

        logger.info("Starting video stream from source %s", source)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break

                zones = self.detect(frame)
                best = None
                safe_zones = [z for z in zones if z.is_safe]
                if safe_zones:
                    safe_zones.sort(
                        key=lambda z: (
                            self.ZONE_PRIORITY.get(z.zone_type, 99),
                            -z.confidence,
                        )
                    )
                    best = safe_zones[0]

                annotated = self.annotate_frame(frame, zones)
                yield annotated, best

        finally:
            cap.release()
            logger.info("Video stream stopped")

    def _parse_detection(
        self, box: object, frame: np.ndarray
    ) -> Optional[LandingZone]:
        """Parse a single YOLO detection box into a LandingZone.

        Args:
            box: Ultralytics Box object.
            frame: Original frame for context.

        Returns:
            LandingZone object or None if parsing fails.
        """
        try:
            # Extract bbox coordinates
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            confidence = float(box.conf[0].cpu().numpy())
            class_idx = int(box.cls[0].cpu().numpy())

            # Get class name from model
            class_names = self.model.names
            zone_type = class_names.get(class_idx, f"class_{class_idx}")

            # Compute centre and area
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)

            # Determine safety
            is_safe = zone_type in self.SAFE_CLASSES

            # Calculate safety score
            safety_score = self._calculate_safety_score(
                zone_type, confidence, area, frame
            )

            # Get recommendation
            recommendation = self._get_recommendation(
                zone_type, is_safe, safety_score, area
            )

            return LandingZone(
                zone_type=zone_type,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                area_pixels=area,
                is_safe=is_safe,
                safety_score=safety_score,
                recommendation=recommendation,
            )

        except (IndexError, AttributeError, ValueError) as exc:
            logger.warning("Failed to parse detection: %s", exc)
            return None

    def _calculate_safety_score(
        self,
        zone_type: str,
        confidence: float,
        area: int,
        frame: np.ndarray,
    ) -> float:
        """Calculate composite safety score for a detected zone.

        Combines class priority, detection confidence, and
        landing area size.

        Args:
            zone_type: Detected class name.
            confidence: Model confidence score.
            area: Bounding box area in pixels.
            frame: Original frame for resolution context.

        Returns:
            Safety score between 0.0 (unsafe) and 1.0 (very safe).
        """
        if zone_type in self.UNSAFE_CLASSES:
            return 0.0

        # Type score based on priority
        type_scores = {
            "safe_flat_ground": 1.0,
            "driveway": 0.85,
            "rooftop": 0.7,
            "balcony": 0.55,
        }
        type_score = type_scores.get(zone_type, 0.3)

        # Area score (larger = safer)
        frame_area = frame.shape[0] * frame.shape[1]
        area_ratio = area / max(frame_area, 1)
        area_score = min(area_ratio * 10, 1.0)  # Saturate at 10% of frame

        # Minimum area check
        if area < _CFG.MIN_LANDING_AREA_PIXELS:
            area_score *= 0.5

        # Composite: 40% type + 40% confidence + 20% area
        return 0.4 * type_score + 0.4 * confidence + 0.2 * area_score

    @staticmethod
    def _get_recommendation(
        zone_type: str,
        is_safe: bool,
        safety_score: float,
        area: int,
    ) -> str:
        """Determine landing recommendation.

        Args:
            zone_type: Detected zone class.
            is_safe: Whether zone is in safe classes.
            safety_score: Composite safety score.
            area: Zone area in pixels.

        Returns:
            "land", "hover", or "abort".
        """
        if not is_safe:
            return "abort"

        if zone_type == "balcony":
            return "hover"  # Too small for full landing

        if safety_score >= 0.6 and area >= _CFG.MIN_LANDING_AREA_PIXELS:
            return "land"

        if safety_score >= 0.4:
            return "hover"

        return "abort"
