#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threading utilities for background processing
"""

import logging
from pathlib import Path
from typing import Callable
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np

from app.gui.utils.image_io import load_image_array, save_mask, save_metrics

logger = logging.getLogger(__name__)


class PredictionWorker(QThread):
    """Worker thread for running segmentation predictions"""
    
    progress = pyqtSignal(str)  # Progress message
    completed = pyqtSignal(dict)  # Result dict with mask, metrics, etc.
    error = pyqtSignal(str)  # Error message
    
    def __init__(
        self,
        predict_func: Callable,
        image: np.ndarray,
        **kwargs
    ):
        super().__init__()
        self.predict_func = predict_func
        self.image = image
        self.kwargs = kwargs
        self._is_running = True
    
    def run(self):
        """Run prediction in thread"""
        try:
            self.progress.emit("Starting prediction...")
            logger.debug(f"Prediction worker starting with image shape {self.image.shape}")
            
            result = self.predict_func(self.image, **self.kwargs)
            
            if result is None:
                error_msg = (
                    "Prediction returned None. This usually means:\n"
                    "• Model failed to load properly\n"
                    "• Image format is incompatible with the model\n"
                    "• An error occurred during inference\n\n"
                    "Check the application logs for more details."
                )
                logger.error(f"Prediction worker: {error_msg}")
                self.error.emit(error_msg)
                return
            
            self.progress.emit("Prediction complete")
            logger.debug(f"Prediction successful, mask shape: {result.segmentation_mask.shape}")
            
            # Prepare result dict from SegmentationResult object
            result_dict = {
                'mask': result.segmentation_mask,
                'metadata': result.metadata,
                'model_type': result.model_type.value
            }
            
            self.completed.emit(result_dict)
            
        except Exception as e:
            logger.error(f"Prediction error in worker thread: {e}", exc_info=True)
            error_msg = f"Prediction failed: {str(e)}"
            self.error.emit(error_msg)
    
    def stop(self):
        """Stop the worker thread"""
        self._is_running = False
        self.wait()


class BatchPredictionWorker(QThread):
    """Worker thread for sequential batch segmentation."""

    progress = pyqtSignal(str)
    item_completed = pyqtSignal(str, dict)
    item_error = pyqtSignal(str, str)
    completed = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        predict_func: Callable,
        image_paths: list,
        output_dir: str,
        output_suffix: str,
        export_metrics: bool = True,
        **kwargs
    ):
        super().__init__()
        self.predict_func = predict_func
        self.image_paths = image_paths
        self.output_dir = Path(output_dir)
        self.output_suffix = output_suffix
        self.export_metrics = export_metrics
        self.kwargs = kwargs
        self._is_running = True

    def run(self):
        """Run batch prediction in the thread."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            total = len(self.image_paths)
            succeeded = 0
            failed = 0

            for index, image_path in enumerate(self.image_paths, start=1):
                if not self._is_running:
                    break

                image_name = Path(image_path).name
                self.progress.emit(f"Processing {index}/{total}: {image_name}")

                try:
                    image = load_image_array(image_path)
                    result = self.predict_func(image, **self.kwargs)
                    if result is None:
                        raise RuntimeError("Prediction returned None")

                    output_path = self.output_dir / f"{Path(image_path).stem}_{self.output_suffix}.png"
                    save_mask(str(output_path), result.segmentation_mask)

                    metadata = dict(result.metadata)
                    metadata["source_image"] = image_path
                    metadata["mask_path"] = str(output_path)
                    if self.export_metrics:
                        save_metrics(str(output_path.with_suffix(".csv")), metadata)

                    succeeded += 1
                    self.item_completed.emit(image_path, {
                        "mask_path": str(output_path),
                        "metadata": metadata,
                    })

                except Exception as exc:
                    failed += 1
                    logger.error("Batch prediction failed for %s: %s", image_path, exc)
                    self.item_error.emit(image_path, str(exc))

            self.completed.emit({
                "total": total,
                "succeeded": succeeded,
                "failed": failed,
                "cancelled": not self._is_running,
            })

        except Exception as exc:
            logger.error("Batch prediction error: %s", exc)
            self.error.emit(str(exc))

    def stop(self):
        """Stop the worker thread."""
        self._is_running = False
        self.wait()


class TrainingWorker(QThread):
    """Worker thread for model training"""
    
    progress = pyqtSignal(dict)  # {'epoch': int, 'loss': float, 'val_loss': float}
    completed = pyqtSignal(str)  # Path to saved model
    error = pyqtSignal(str)  # Error message
    
    def __init__(
        self,
        train_func: Callable,
        **kwargs
    ):
        super().__init__()
        self.train_func = train_func
        self.kwargs = kwargs
        self._is_running = True
    
    def run(self):
        """Run training in thread"""
        try:
            # Call training function with progress callback
            def progress_callback(epoch, loss, val_loss=None):
                if self._is_running:
                    self.progress.emit({
                        'epoch': epoch,
                        'loss': loss,
                        'val_loss': val_loss
                    })
            
            model_path = self.train_func(
                progress_callback=progress_callback,
                **self.kwargs
            )
            
            if model_path:
                self.completed.emit(str(model_path))
            else:
                self.error.emit("Training completed but no model path returned")
                
        except Exception as e:
            logger.error(f"Training error: {e}")
            self.error.emit(str(e))
    
    def stop(self):
        """Stop training"""
        self._is_running = False
        self.wait()
