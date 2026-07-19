# -*- coding: utf-8 -*-
"""
MediReach — Landing Zone Safety Classifier.

Secondary classifier for additional safety verification
of detected landing zones using texture, colour, and
edge analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ZoneClassification:
    """Result of zone safety classification."""
    zone_type: str
    is_safe: bool
    flatness_score: float
    texture_score: float
    colour_score: float
    overall_score: float
    details: Dict[str, float]


class ZoneClassifier:
    """Secondary safety classifier using image analysis.

    Analyses the cropped landing zone region for:
    - Surface flatness (edge density analysis)
    - Texture uniformity (standard deviation of pixel values)
    - Colour safety (avoids blue=water, green=vegetation)
    - Size adequacy (minimum area check)
    """

    # Thresholds
    FLATNESS_THRESHOLD: float = 0.6
    TEXTURE_THRESHOLD: float = 0.5
    SAFE_SCORE_THRESHOLD: float = 0.55
    MIN_ZONE_SIZE_PX: int = 2500

    def __init__(self) -> None:
        """Initialise the zone classifier."""
        logger.info("ZoneClassifier initialised")

    def classify(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        detected_class: str = "unknown",
    ) -> ZoneClassification:
        """Classify a detected zone region for safety.

        Args:
            frame: Full BGR image.
            bbox: Bounding box (x1, y1, x2, y2).
            detected_class: YOLO-detected class name.

        Returns:
            ZoneClassification result.
        """
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return ZoneClassification(
                zone_type=detected_class,
                is_safe=False,
                flatness_score=0.0,
                texture_score=0.0,
                colour_score=0.0,
                overall_score=0.0,
                details={"error": "Empty ROI"},
            )

        flatness = self._analyse_flatness(roi)
        texture = self._analyse_texture(roi)
        colour = self._analyse_colour_safety(roi)
        size_ok = (x2 - x1) * (y2 - y1) >= self.MIN_ZONE_SIZE_PX

        # Composite score: 35% flatness + 30% texture + 25% colour + 10% size
        size_score = 1.0 if size_ok else 0.3
        overall = (
            0.35 * flatness
            + 0.30 * texture
            + 0.25 * colour
            + 0.10 * size_score
        )

        is_safe = overall >= self.SAFE_SCORE_THRESHOLD and size_ok

        return ZoneClassification(
            zone_type=detected_class,
            is_safe=is_safe,
            flatness_score=flatness,
            texture_score=texture,
            colour_score=colour,
            overall_score=overall,
            details={
                "flatness": flatness,
                "texture": texture,
                "colour_safety": colour,
                "size_adequate": float(size_ok),
                "area_pixels": float((x2 - x1) * (y2 - y1)),
            },
        )

    def _analyse_flatness(self, roi: np.ndarray) -> float:
        """Analyse surface flatness via edge density.

        Flat surfaces have fewer edges; rough/sloped surfaces
        have dense edge patterns.

        Args:
            roi: Cropped region of interest (BGR).

        Returns:
            Flatness score 0.0–1.0 (1.0 = very flat).
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / max(edges.size, 1)

        # Invert: fewer edges = flatter surface
        flatness = max(0.0, 1.0 - edge_density * 5.0)
        return min(flatness, 1.0)

    def _analyse_texture(self, roi: np.ndarray) -> float:
        """Analyse texture uniformity via pixel variance.

        Uniform surfaces (concrete, asphalt) have low variance.

        Args:
            roi: Cropped region (BGR).

        Returns:
            Texture score 0.0–1.0 (1.0 = very uniform).
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Local standard deviation using box filter
        mean = cv2.blur(gray.astype(np.float32), (15, 15))
        mean_sq = cv2.blur((gray.astype(np.float32)) ** 2, (15, 15))
        local_std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

        avg_std = np.mean(local_std)

        # Normalise: lower std = more uniform = higher score
        texture_score = max(0.0, 1.0 - avg_std / 60.0)
        return min(texture_score, 1.0)

    def _analyse_colour_safety(self, roi: np.ndarray) -> float:
        """Analyse colour to detect unsafe surfaces.

        Blue-dominant regions suggest water.
        Excessive green may indicate vegetation.

        Args:
            roi: Cropped region (BGR).

        Returns:
            Colour safety score 0.0–1.0 (1.0 = safe colours).
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_pixels = max(roi.shape[0] * roi.shape[1], 1)

        # Water detection (blue hue range)
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blue_ratio = np.sum(blue_mask > 0) / total_pixels

        # Dense vegetation (green hue range)
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_ratio = np.sum(green_mask > 0) / total_pixels

        # Penalise blue (water) heavily, green moderately
        penalty = blue_ratio * 2.0 + green_ratio * 0.5
        safety = max(0.0, 1.0 - penalty)
        return min(safety, 1.0)

    def classify_batch(
        self,
        frame: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        classes: Optional[List[str]] = None,
    ) -> List[ZoneClassification]:
        """Classify multiple zones from a single frame.

        Args:
            frame: Full image.
            bboxes: List of bounding boxes.
            classes: Optional class names per bbox.

        Returns:
            List of ZoneClassification results.
        """
        results = []
        for i, bbox in enumerate(bboxes):
            cls = classes[i] if classes and i < len(classes) else "unknown"
            results.append(self.classify(frame, bbox, cls))
        return results
