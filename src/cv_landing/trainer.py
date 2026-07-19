# -*- coding: utf-8 -*-
"""
MediReach — YOLOv8 Training Pipeline for Landing Zone Detection.

Complete training pipeline: dataset validation → training →
evaluation → model export for edge deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.cv_landing.dataset_builder import DatasetBuilder
from src.utils.constants import CVConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CFG = CVConfig()


@dataclass
class TrainingConfig:
    """YOLOv8 training configuration."""
    dataset_yaml: str = "data/landing_zones/dataset.yaml"
    base_model: str = "yolov8n.pt"
    epochs: int = _CFG.EPOCHS
    image_size: int = _CFG.IMAGE_SIZE
    batch_size: int = _CFG.BATCH_SIZE
    patience: int = _CFG.PATIENCE
    project_name: str = "medireach_landing"
    save_dir: str = "models/cv"
    device: str = "auto"
    augment: bool = True
    classes: List[str] = field(default_factory=lambda: list(_CFG.ALL_CLASSES))


class YOLOv8Trainer:
    """Complete YOLOv8 training pipeline for landing zone detection.

    Handles dataset validation, training, evaluation, and export.
    """

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        """Initialise trainer with configuration.

        Args:
            config: Training config. Defaults to TrainingConfig().
        """
        self.config = config or TrainingConfig()
        self.model: Any = None
        self._training_results: Any = None

    def validate_dataset(
        self, dataset_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate the training dataset before starting.

        Args:
            dataset_path: Path to dataset root.

        Returns:
            Validation report dictionary.
        """
        path = dataset_path or str(Path(self.config.dataset_yaml).parent)
        builder = DatasetBuilder(
            base_path=path, classes=self.config.classes
        )
        report = builder.validate_dataset()

        if not report["valid"]:
            logger.error("Dataset validation FAILED")
            for rec in report["recommendations"]:
                logger.warning("  → %s", rec)
        else:
            logger.info(
                "Dataset validation PASSED: %d images, %d labels",
                report["total_images"], report["total_labels"],
            )

        return report

    def train(self) -> Dict[str, Any]:
        """Run full YOLOv8 training with logging and checkpointing.

        Returns:
            Training metrics dictionary.

        Raises:
            RuntimeError: If training fails.
        """
        try:
            from ultralytics import YOLO
            import torch
        except ImportError as exc:
            logger.error("Required packages not installed: %s", exc)
            raise

        # Resolve device
        if self.config.device == "auto":
            device = "0" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device

        logger.info("=" * 50)
        logger.info("YOLOv8 Training — MediReach Landing Detection")
        logger.info("  Dataset : %s", self.config.dataset_yaml)
        logger.info("  Base    : %s", self.config.base_model)
        logger.info("  Epochs  : %d", self.config.epochs)
        logger.info("  ImgSize : %d", self.config.image_size)
        logger.info("  Batch   : %d", self.config.batch_size)
        logger.info("  Device  : %s", device)
        logger.info("=" * 50)

        # Load pre-trained model
        self.model = YOLO(self.config.base_model)

        # Train
        self._training_results = self.model.train(
            data=self.config.dataset_yaml,
            epochs=self.config.epochs,
            imgsz=self.config.image_size,
            batch=self.config.batch_size,
            name=self.config.project_name,
            patience=self.config.patience,
            save=True,
            plots=True,
            augment=self.config.augment,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            device=device,
        )

        metrics = self._extract_metrics(self._training_results)
        logger.info("Training complete. Metrics: %s", metrics)
        return metrics

    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation on the validation set.

        Returns:
            Evaluation metrics (mAP50, mAP50-95, precision, recall).

        Raises:
            RuntimeError: If model not trained/loaded.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Train or load a model first.")

        logger.info("Running model evaluation...")
        results = self.model.val()

        metrics = {
            "mAP50": float(getattr(results, "map50", 0.0)),
            "mAP50_95": float(getattr(results, "map", 0.0)),
            "precision": float(getattr(results, "mp", 0.0)),
            "recall": float(getattr(results, "mr", 0.0)),
        }

        logger.info("Evaluation: mAP50=%.3f, mAP50-95=%.3f, P=%.3f, R=%.3f",
                     metrics["mAP50"], metrics["mAP50_95"],
                     metrics["precision"], metrics["recall"])
        return metrics

    def export_for_edge(
        self,
        format: str = "onnx",
        output_dir: Optional[str] = None,
    ) -> str:
        """Export trained model for Raspberry Pi deployment.

        Args:
            format: Export format ('onnx', 'tflite', 'edgetpu').
            output_dir: Output directory for exported model.

        Returns:
            Path to the exported model file.

        Raises:
            RuntimeError: If model not trained/loaded.
        """
        if self.model is None:
            raise RuntimeError("No model loaded.")

        logger.info("Exporting model to %s format...", format)
        export_path = self.model.export(format=format)

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            import shutil
            dest = os.path.join(output_dir, os.path.basename(str(export_path)))
            shutil.move(str(export_path), dest)
            export_path = dest

        logger.info("Model exported to %s", export_path)
        return str(export_path)

    def load_model(self, model_path: str) -> None:
        """Load a previously trained model.

        Args:
            model_path: Path to .pt model file.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics not installed") from exc

        self.model = YOLO(model_path)
        logger.info("Model loaded from %s", model_path)

    def _extract_metrics(self, results: Any) -> Dict[str, Any]:
        """Extract training metrics from YOLO results.

        Args:
            results: Training results object from ultralytics.

        Returns:
            Dictionary of metrics.
        """
        try:
            return {
                "epochs_completed": self.config.epochs,
                "best_mAP50": float(getattr(results, "map50", 0.0) if results else 0.0),
                "best_mAP50_95": float(getattr(results, "map", 0.0) if results else 0.0),
                "model_path": str(
                    getattr(results, "save_dir", self.config.save_dir)
                ),
            }
        except (AttributeError, TypeError):
            return {
                "epochs_completed": self.config.epochs,
                "note": "Metrics extraction failed — check training logs.",
            }
