# -*- coding: utf-8 -*-
"""
MediReach — Image Preprocessing Pipeline for CV Landing Detection.

Handles resizing, normalisation, denoising, and colour space
conversion for both training and inference.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.constants import CVConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CFG = CVConfig()


class ImagePreprocessor:
    """Image preprocessing for landing zone detection.

    Provides standardised preprocessing for both training
    data preparation and real-time inference.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (_CFG.IMAGE_SIZE, _CFG.IMAGE_SIZE),
        normalize: bool = True,
        denoise: bool = False,
    ) -> None:
        """Initialise the preprocessor.

        Args:
            target_size: Output (width, height) in pixels.
            normalize: Whether to normalise pixel values to [0, 1].
            denoise: Whether to apply denoising filter.
        """
        self.target_size = target_size
        self.normalize = normalize
        self.denoise = denoise

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply full preprocessing pipeline to a single image.

        Args:
            image: Input BGR image as numpy array.

        Returns:
            Preprocessed image.

        Raises:
            ValueError: If image is empty or has wrong dimensions.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        result = image.copy()

        # Denoise if enabled
        if self.denoise:
            result = self.apply_denoise(result)

        # Resize
        result = self.resize(result)

        # Normalise
        if self.normalize:
            result = self.apply_normalize(result)

        return result

    def resize(
        self,
        image: np.ndarray,
        size: Optional[Tuple[int, int]] = None,
        keep_aspect: bool = True,
    ) -> np.ndarray:
        """Resize image with optional aspect ratio preservation.

        Args:
            image: Input image.
            size: Target (width, height) or None for default.
            keep_aspect: If True, pad to maintain aspect ratio.

        Returns:
            Resized image.
        """
        target = size or self.target_size
        h, w = image.shape[:2]
        tw, th = target

        if keep_aspect:
            scale = min(tw / w, th / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Pad to target size
            canvas = np.zeros((th, tw, 3), dtype=image.dtype)
            if self.normalize and image.dtype == np.float32:
                canvas[:] = 0.5  # grey padding for normalised images
            else:
                canvas[:] = 114  # grey padding (YOLO convention)

            pad_y = (th - new_h) // 2
            pad_x = (tw - new_w) // 2
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
            return canvas
        else:
            return cv2.resize(image, target, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def apply_normalize(image: np.ndarray) -> np.ndarray:
        """Normalise pixel values from [0, 255] to [0.0, 1.0].

        Args:
            image: Input uint8 image.

        Returns:
            Float32 normalised image.
        """
        return image.astype(np.float32) / 255.0

    @staticmethod
    def apply_denoise(
        image: np.ndarray,
        strength: int = 10,
    ) -> np.ndarray:
        """Apply fast non-local means denoising.

        Args:
            image: Input BGR image.
            strength: Filter strength (higher = more denoising).

        Returns:
            Denoised image.
        """
        return cv2.fastNlMeansDenoisingColored(
            image, None, strength, strength, 7, 21
        )

    @staticmethod
    def to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert BGR to RGB colour space.

        Args:
            image: BGR image.

        Returns:
            RGB image.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale.

        Args:
            image: BGR or RGB image.

        Returns:
            Grayscale image.
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def enhance_contrast(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation).

        Args:
            image: Input BGR image.
            clip_limit: Threshold for contrast limiting.
            tile_size: Size of grid for histogram equalisation.

        Returns:
            Contrast-enhanced image.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_enhanced = clahe.apply(l_channel)

        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """Apply unsharp masking to sharpen image.

        Args:
            image: Input image.

        Returns:
            Sharpened image.
        """
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

    def preprocess_batch(
        self, images: list[np.ndarray]
    ) -> np.ndarray:
        """Preprocess a batch of images.

        Args:
            images: List of input images.

        Returns:
            Stacked numpy array of shape (N, H, W, C).
        """
        processed = [self.preprocess(img) for img in images]
        return np.stack(processed, axis=0)
