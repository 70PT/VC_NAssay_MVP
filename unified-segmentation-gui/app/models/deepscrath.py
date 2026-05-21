#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepScratch model wrapper for cell detection segmentation
"""

import sys
from pathlib import Path
import numpy as np
import torch
import logging
import cv2

from .base import BasePredictor, ModelType, SegmentationResult
from app.utils.devices import resolve_device

logger = logging.getLogger(__name__)


class DeepScratchPredictor(BasePredictor):
    """Wrapper for DeepScratch cell detection model"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__(ModelType.DEEPSCRATH, device)
        self.device_torch = self._resolve_device(device)
        self._last_image_shape = None

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve configured device names to torch.device instances."""
        return torch.device(resolve_device(device))

    def load_model(
        self,
        model_path: str,
        model_name: str = "unet-flat-48",
        n_ch_in: int = 1,
        n_ch_out: int = 1,
        loss_type: str = "l2-G1.5",
        task: str = "detection",
        nms_threshold_abs: float = None,
        nms_threshold_rel: float = None,
        nms_min_distance: int = 3,
        **kwargs
    ) -> bool:
        """Load DeepScratch model
        
        Args:
            model_path: Path to model checkpoint or directory
            model_name: Model architecture name (e.g., 'unet-flat-48')
            n_ch_in: Number of input channels expected by the model
            n_ch_out: Number of output channels/classes
            loss_type: DeepScratch loss identifier used when training the checkpoint
            task: 'detection' for coordinate output or 'segmentation' for masks
            **kwargs: Additional model arguments
            
        Returns:
            Success flag
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists() or not model_path.is_file():
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            
            # Import DeepScratch modules dynamically
            deepscratch_root = Path(__file__).resolve().parents[3] / "DeepScratch"
            if str(deepscratch_root) not in sys.path:
                sys.path.insert(0, str(deepscratch_root))
            from cell_localization.models import get_model
            
            if task == "segmentation" and not model_name.startswith("seg+"):
                model_name = f"seg+{model_name}"

            logger.info("Loading DeepScratch model: %s", model_name)
            
            # Create and load model
            self.model = get_model(
                model_name,
                n_ch_in=n_ch_in,
                n_ch_out=n_ch_out,
                loss_type=loss_type,
                nms_threshold_abs=nms_threshold_abs,
                nms_threshold_rel=nms_threshold_rel,
                nms_min_distance=nms_min_distance,
            )
            
            checkpoint = torch.load(str(model_path), map_location=self.device_torch)
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            state_dict = {
                key.replace("module.", "", 1): value
                for key, value in state_dict.items()
            }
            self.model.load_state_dict(state_dict)
            logger.info("Loaded weights from %s", model_path)
            
            self.model = self.model.to(self.device_torch)
            self.model.eval()
            
            self.model_config = {
                "model_path": str(model_path),
                "model_name": model_name,
                "n_ch_in": n_ch_in,
                "n_ch_out": n_ch_out,
                "loss_type": loss_type,
                "task": task,
                "device": str(self.device_torch),
                **kwargs,
            }
            return True
            
        except Exception as e:
            logger.error(f"Failed to load DeepScratch model: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> SegmentationResult:
        """Run cell detection on image
        
        Args:
            image: Input microscopy image (HxW or HxWxC)
            
        Returns:
            SegmentationResult with cell detection mask
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model first.")
            
            # Preprocess
            self._last_image_shape = image.shape[:2]
            image_tensor = self.preprocess_image(image)
            
            # Predict
            with torch.no_grad():
                output = self.model(image_tensor)
            
            # Postprocess
            mask = self.postprocess_output(output)
            
            # Calculate metrics
            metadata = self._metadata_from_mask(mask)
            
            return SegmentationResult(
                model_type=ModelType.DEEPSCRATH,
                segmentation_mask=mask,
                raw_output={"output": output},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Normalize and prepare image for DeepScratch
        
        Args:
            image: Input image (HxW or HxWxC)
            
        Returns:
            Normalized torch tensor
        """
        image = np.asarray(image)

        # Match the channel count the checkpoint was configured with.
        n_ch_in = int(self.model_config.get("n_ch_in", 1))
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        if n_ch_in == 1 and image.ndim == 3:
            image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif n_ch_in == 3 and image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        # Convert to float and normalize to [0, 1]
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        max_value = float(np.max(image)) if image.size else 0.0
        if max_value > 1.0:
            image = image / max_value
        
        # Convert to torch tensor
        if len(image.shape) == 2:
            # Grayscale: HxW -> 1xHxW
            image_tensor = torch.from_numpy(image[np.newaxis, :, :])
        else:
            # Color: HxWxC -> CxHxW
            image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        
        # Add batch dimension: CxHxW -> 1xCxHxW
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device_torch)
        
        return image_tensor
    
    def postprocess_output(self, raw_output) -> np.ndarray:
        """Convert model output to segmentation mask
        
        Args:
            raw_output: Raw model output tensor
            
        Returns:
            Segmentation mask as numpy array
        """
        if isinstance(raw_output, list):
            raw_output = raw_output[0] if raw_output else {}

        if isinstance(raw_output, dict) and "coordinates" in raw_output:
            return self._coordinates_to_mask(raw_output)

        if isinstance(raw_output, tuple):
            raw_output = raw_output[0]

        # Move to CPU and convert to numpy
        mask = raw_output.squeeze().detach().cpu().numpy()
        
        # Threshold to binary mask
        if mask.ndim == 3:
            # Multi-channel, take argmax
            mask = np.argmax(mask, axis=0)
        else:
            # Single channel, threshold at 0.5
            mask = (mask > 0.5).astype(np.uint8)
        
        return mask.astype(np.uint16 if np.max(mask) > 255 else np.uint8)

    def _coordinates_to_mask(self, output: dict) -> np.ndarray:
        """Convert DeepScratch coordinate detections into an instance mask."""
        if self._last_image_shape is None:
            raise RuntimeError("Cannot convert detections without input image shape")

        mask = np.zeros(self._last_image_shape, dtype=np.uint16)
        coordinates = output.get("coordinates")
        if coordinates is None:
            return mask

        coordinates = coordinates.detach().cpu().numpy()
        radius = int(self.model_config.get("marker_radius", 4))
        for label, (x_coord, y_coord) in enumerate(coordinates, start=1):
            x_coord = int(round(float(x_coord)))
            y_coord = int(round(float(y_coord)))
            if 0 <= y_coord < mask.shape[0] and 0 <= x_coord < mask.shape[1]:
                cv2.circle(mask, (x_coord, y_coord), radius, int(label), thickness=-1)
        return mask

    @staticmethod
    def _metadata_from_mask(mask: np.ndarray) -> dict:
        """Calculate summary metrics from a DeepScratch mask."""
        labels = np.unique(mask)
        num_objects = int(len(labels[labels > 0]))
        return {
            "num_cells": num_objects,
            "cell_coverage": float(np.sum(mask > 0) / mask.size),
        }
    
    def unload_model(self):
        """Release model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("DeepScratch model unloaded")
