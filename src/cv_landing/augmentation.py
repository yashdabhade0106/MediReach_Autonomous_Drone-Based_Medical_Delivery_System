# -*- coding: utf-8 -*-
"""
MediReach — Data Augmentation Pipeline for CV Training.

Albumentations-based augmentation with drone-specific transforms
including aerial perspective, weather overlays, and lighting changes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AugmentationPipeline:
    """Drone-specific data augmentation for landing zone detection.

    Uses Albumentations for composable augmentation transforms
    optimised for aerial/downward-looking camera images.
    """

    def __init__(
        self,
        image_size: int = 640,
        augment_level: str = "medium",
    ) -> None:
        """Initialise augmentation pipeline.

        Args:
            image_size: Target image size for resizing.
            augment_level: 'light', 'medium', or 'heavy'.
        """
        self.image_size = image_size
        self.augment_level = augment_level
        self._transform = self._build_pipeline(augment_level)

    def _build_pipeline(self, level: str):  # type: ignore[no-untyped-def]
        """Build the Albumentations compose pipeline.

        Args:
            level: Augmentation intensity level.

        Returns:
            Albumentations Compose object.
        """
        try:
            import albumentations as A
        except ImportError:
            logger.warning("albumentations not installed; augmentation disabled")
            return None

        if level == "light":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                A.Resize(self.image_size, self.image_size),
            ], bbox_params=A.BboxParams(
                format="yolo", label_fields=["class_labels"]
            ))

        elif level == "medium":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2,
                    rotate_limit=15, p=0.5,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10, sat_shift_limit=20,
                        val_shift_limit=20, p=1.0
                    ),
                ], p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.2),
                A.GaussNoise(var_limit=(10, 30), p=0.2),
                A.Resize(self.image_size, self.image_size),
            ], bbox_params=A.BboxParams(
                format="yolo", label_fields=["class_labels"],
                min_visibility=0.3,
            ))

        else:  # heavy
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15, scale_limit=0.3,
                    rotate_limit=30, p=0.6,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=15, sat_shift_limit=30,
                        val_shift_limit=30, p=1.0
                    ),
                    A.RandomGamma(gamma_limit=(60, 140), p=1.0),
                ], p=0.6),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.RandomRain(
                        slant_lower=-10, slant_upper=10,
                        drop_length=10, drop_width=1,
                        drop_color=(200, 200, 200), p=1.0,
                    ),
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
                    A.RandomShadow(p=1.0),
                ], p=0.3),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.CLAHE(clip_limit=3.0, p=0.2),
                A.Resize(self.image_size, self.image_size),
            ], bbox_params=A.BboxParams(
                format="yolo", label_fields=["class_labels"],
                min_visibility=0.2,
            ))

    def augment(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int],
    ) -> Dict[str, object]:
        """Apply augmentation to a single image with bounding boxes.

        Args:
            image: Input BGR image (H, W, C).
            bboxes: List of YOLO-format bboxes [[cx, cy, w, h], ...].
            class_labels: List of class indices matching bboxes.

        Returns:
            Dictionary with 'image', 'bboxes', 'class_labels' keys.
        """
        if self._transform is None:
            return {
                "image": image,
                "bboxes": bboxes,
                "class_labels": class_labels,
            }

        result = self._transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels,
        )

        return {
            "image": result["image"],
            "bboxes": result["bboxes"],
            "class_labels": result["class_labels"],
        }

    def augment_no_bbox(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to image without bounding boxes.

        Useful for classification tasks or preview generation.

        Args:
            image: Input image.

        Returns:
            Augmented image.
        """
        if self._transform is None:
            return image

        # Build a simpler pipeline without bbox params
        try:
            import albumentations as A
        except ImportError:
            return image

        simple = A.Compose([
            t for t in self._transform.transforms
        ])

        return simple(image=image)["image"]

    def generate_augmented_dataset(
        self,
        images: List[np.ndarray],
        bboxes_list: List[List[List[float]]],
        labels_list: List[List[int]],
        multiplier: int = 5,
    ) -> Tuple[List[np.ndarray], List[List[List[float]]], List[List[int]]]:
        """Generate augmented copies of an entire dataset.

        Args:
            images: List of original images.
            bboxes_list: List of bbox lists per image.
            labels_list: List of label lists per image.
            multiplier: Number of augmented copies per original.

        Returns:
            Tuple of (augmented_images, augmented_bboxes, augmented_labels).
        """
        aug_images: List[np.ndarray] = []
        aug_bboxes: List[List[List[float]]] = []
        aug_labels: List[List[int]] = []

        for idx, (img, bboxes, labels) in enumerate(
            zip(images, bboxes_list, labels_list)
        ):
            # Include original
            aug_images.append(img)
            aug_bboxes.append(bboxes)
            aug_labels.append(labels)

            # Generate augmented copies
            for _ in range(multiplier):
                result = self.augment(img, bboxes, labels)
                aug_images.append(result["image"])
                aug_bboxes.append(list(result["bboxes"]))
                aug_labels.append(list(result["class_labels"]))

            if (idx + 1) % 100 == 0:
                logger.info(
                    "Augmented %d/%d images (×%d)",
                    idx + 1, len(images), multiplier,
                )

        logger.info(
            "Augmentation complete: %d → %d images",
            len(images), len(aug_images),
        )
        return aug_images, aug_bboxes, aug_labels
