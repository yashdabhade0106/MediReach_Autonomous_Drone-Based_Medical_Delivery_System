# -*- coding: utf-8 -*-
"""
MediReach — Dataset Builder for YOLO Training.

Creates and validates the YOLO dataset directory structure,
generates dataset.yaml configuration, and provides utilities
for train/val/test splitting.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.utils.constants import CVConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CFG = CVConfig()


class DatasetBuilder:
    """YOLO dataset structure builder and validator.

    Creates the directory layout and configuration files
    required by Ultralytics YOLOv8 training.
    """

    def __init__(
        self,
        base_path: str = "data/landing_zones",
        classes: Optional[List[str]] = None,
    ) -> None:
        """Initialise dataset builder.

        Args:
            base_path: Root directory for the dataset.
            classes: List of class names. Defaults to CVConfig.ALL_CLASSES.
        """
        self.base_path = Path(base_path)
        self.classes = list(classes or _CFG.ALL_CLASSES)

    def create_directory_structure(self) -> None:
        """Create the YOLO dataset directory structure.

        Creates::
            base_path/
            ├── train/
            │   ├── images/
            │   └── labels/
            ├── val/
            │   ├── images/
            │   └── labels/
            └── test/
                ├── images/
                └── labels/
        """
        for split in ("train", "val", "test"):
            for subdir in ("images", "labels"):
                path = self.base_path / split / subdir
                path.mkdir(parents=True, exist_ok=True)

        logger.info("Dataset structure created at %s", self.base_path)

    def generate_dataset_yaml(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate the YOLO dataset.yaml configuration file.

        Args:
            output_path: Custom output path. Defaults to base_path/dataset.yaml.

        Returns:
            Absolute path to the generated YAML file.
        """
        yaml_path = output_path or str(self.base_path / "dataset.yaml")

        config = {
            "path": str(self.base_path.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(self.classes),
            "names": self.classes,
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info("Dataset YAML generated at %s", yaml_path)
        return yaml_path

    def validate_dataset(self, dataset_path: Optional[str] = None) -> Dict:
        """Validate dataset structure, class balance, and image integrity.

        Args:
            dataset_path: Path to validate. Defaults to self.base_path.

        Returns:
            Validation report dictionary.
        """
        path = Path(dataset_path) if dataset_path else self.base_path

        report: Dict = {
            "valid": True,
            "total_images": 0,
            "total_labels": 0,
            "splits": {},
            "class_distribution": {cls: 0 for cls in self.classes},
            "missing_labels": [],
            "orphan_labels": [],
            "corrupted_images": [],
            "recommendations": [],
        }

        for split in ("train", "val", "test"):
            images_dir = path / split / "images"
            labels_dir = path / split / "labels"

            if not images_dir.exists():
                report["valid"] = False
                report["recommendations"].append(
                    f"Missing directory: {images_dir}"
                )
                continue

            image_files = set()
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                image_files.update(images_dir.glob(ext))

            label_files = set(labels_dir.glob("*.txt")) if labels_dir.exists() else set()

            image_stems = {f.stem for f in image_files}
            label_stems = {f.stem for f in label_files}

            # Check for missing labels
            missing = image_stems - label_stems
            for m in missing:
                report["missing_labels"].append(f"{split}/labels/{m}.txt")

            # Check for orphan labels
            orphans = label_stems - image_stems
            for o in orphans:
                report["orphan_labels"].append(f"{split}/labels/{o}.txt")

            # Count class distribution from labels
            for label_file in label_files:
                try:
                    with open(label_file, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_idx = int(parts[0])
                                if 0 <= class_idx < len(self.classes):
                                    report["class_distribution"][
                                        self.classes[class_idx]
                                    ] += 1
                except (ValueError, IndexError) as exc:
                    logger.warning("Malformed label file %s: %s", label_file, exc)

            split_info = {
                "images": len(image_files),
                "labels": len(label_files),
                "missing_labels": len(missing),
                "orphan_labels": len(orphans),
            }
            report["splits"][split] = split_info
            report["total_images"] += len(image_files)
            report["total_labels"] += len(label_files)

        # Generate recommendations
        if report["total_images"] == 0:
            report["valid"] = False
            report["recommendations"].append("No images found in dataset.")

        total_labels = sum(report["class_distribution"].values())
        if total_labels > 0:
            for cls, count in report["class_distribution"].items():
                ratio = count / total_labels
                if ratio < 0.05:
                    report["recommendations"].append(
                        f"Class '{cls}' is underrepresented ({ratio:.1%}). "
                        "Consider collecting more samples or augmentation."
                    )

        if report["missing_labels"]:
            report["recommendations"].append(
                f"{len(report['missing_labels'])} images have no labels."
            )

        logger.info(
            "Dataset validation: %d images, %d labels, valid=%s",
            report["total_images"], report["total_labels"], report["valid"],
        )
        return report

    def split_dataset(
        self,
        source_images_dir: str,
        source_labels_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, int]:
        """Split a flat directory of images/labels into train/val/test.

        Args:
            source_images_dir: Directory containing all images.
            source_labels_dir: Directory containing all labels.
            train_ratio: Fraction for training set.
            val_ratio: Fraction for validation set.
            test_ratio: Fraction for test set.
            seed: Random seed for reproducibility.

        Returns:
            Dictionary with split counts.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        self.create_directory_structure()

        images_path = Path(source_images_dir)
        labels_path = Path(source_labels_dir)

        image_files = sorted(
            [f for f in images_path.iterdir()
             if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        )

        random.seed(seed)
        random.shuffle(image_files)

        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits = {
            "train": image_files[:n_train],
            "val": image_files[n_train:n_train + n_val],
            "test": image_files[n_train + n_val:],
        }

        counts = {}
        for split_name, files in splits.items():
            for img_file in files:
                # Copy image
                dst_img = self.base_path / split_name / "images" / img_file.name
                shutil.copy2(img_file, dst_img)

                # Copy label if exists
                label_file = labels_path / f"{img_file.stem}.txt"
                if label_file.exists():
                    dst_lbl = self.base_path / split_name / "labels" / label_file.name
                    shutil.copy2(label_file, dst_lbl)

            counts[split_name] = len(files)
            logger.info("Split '%s': %d images", split_name, len(files))

        return counts
