#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyMarAI model wrapper for spheroid segmentation using nnU-Net
"""

from pathlib import Path
import os
import tempfile
import numpy as np
import torch
import logging
import cv2

from .base import BasePredictor, ModelType, SegmentationResult
from app.utils.devices import resolve_device

logger = logging.getLogger(__name__)


class PyMarAIPredictor(BasePredictor):
    """Wrapper for pyMarAI spheroid segmentation (nnU-Net v2)"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__(ModelType.PYMARAI, device)
        self.device_torch = self._resolve_device(device)
        self.nnunet_predictor = None

    @property
    def is_loaded(self) -> bool:
        """Return True when the nnU-Net predictor has been initialized and is ready."""
        if self.nnunet_predictor is None:
            return False
        
        # Verify the network is actually loaded
        try:
            return hasattr(self.nnunet_predictor, 'network') and self.nnunet_predictor.network is not None
        except Exception:
            return False

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve configured device names to torch.device instances."""
        return torch.device(resolve_device(device))
    
    def load_model(
        self,
        model_path: str,
        model_name: str = "Dataset001_spheroids_V1",
        fold: int = 0,
        input_channels: int = 1,
        **kwargs
    ) -> bool:
        """Load pyMarAI/nnU-Net model
        
        Args:
            model_path: Path to nnU-Net model directory
            model_name: nnU-Net model name/task
            fold: Which cross-validation fold to use (0-4)
            **kwargs: Additional model arguments
            
        Returns:
            Success flag
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists() or not model_path.is_dir():
                raise FileNotFoundError(f"nnU-Net model directory not found: {model_path}")

            self._prepare_runtime_environment(model_path)
            
            # Import nnU-Net predictor
            try:
                from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            except ImportError:
                logger.error("nnU-Net not installed. Install via: pip install nnunetv2")
                return False
            
            logger.info(f"Loading pyMarAI/nnU-Net model from {model_path}")
            
            # Initialize nnU-Net predictor
            self.nnunet_predictor = nnUNetPredictor(
                tile_step_size=0.5,
                perform_everything_on_device=self.device_torch.type == "cuda",
                device=self.device_torch,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False
            )
            
            # Load model - verify checkpoint exists first
            checkpoint_path = model_path / f'fold_{fold}' / 'checkpoint_final.pth'
            if not checkpoint_path.exists():
                # Try alternative path
                checkpoint_path = model_path / 'fold_0' / 'checkpoint_final.pth'
                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"nnU-Net checkpoint not found. Checked:\n"
                        f"  {model_path / f'fold_{fold}' / 'checkpoint_final.pth'}\n"
                        f"  {model_path / 'fold_0' / 'checkpoint_final.pth'}"
                    )
                logger.warning(f"Fold {fold} checkpoint not found, using fold_0 instead")
                fold = 0
            
            logger.info(f"Initializing nnU-Net from {model_path} with fold {fold}")
            self.nnunet_predictor.initialize_from_trained_model_folder(
                str(model_path),
                (fold,),
                'checkpoint_final.pth'
            )
            
            # Verify model is loaded
            if not hasattr(self.nnunet_predictor, 'network') or self.nnunet_predictor.network is None:
                raise RuntimeError("nnU-Net model failed to initialize (network is None)")
            
            logger.info(f"nnU-Net model loaded successfully from fold {fold}")
            
            self.model_config.update({
                'model_path': str(model_path),
                'model_name': model_name,
                'fold': fold,
                'input_channels': input_channels,
                'device': str(self.device_torch),
                **kwargs
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load pyMarAI model: {e}", exc_info=True)
            self.nnunet_predictor = None
            return False

    @staticmethod
    def _prepare_runtime_environment(model_path: Path):
        """Set local writable paths expected by nnU-Net and matplotlib."""
        base_dir = model_path.parents[2] if len(model_path.parents) > 2 else model_path.parent
        runtime_dir = base_dir / "_runtime"
        for env_name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            path = runtime_dir / env_name
            path.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault(env_name, str(path))

        mpl_dir = runtime_dir / "matplotlib"
        try:
            mpl_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
        except OSError:
            os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    
    def predict(self, image: np.ndarray) -> SegmentationResult:
        """Run spheroid segmentation on image
        
        Args:
            image: Input microscopy image (HxW or HxWxC)
            
        Returns:
            SegmentationResult with segmentation mask
        """
        try:
            if self.nnunet_predictor is None:
                raise RuntimeError("Model not loaded. Call load_model first.")
            
            logger.info(f"Starting nnU-Net prediction on image shape {image.shape}")
            
            # Preprocess
            image_array = self.preprocess_image(image)
            logger.debug(f"After preprocess: shape={image_array.shape}, dtype={image_array.dtype}")
            
            image_array = self._to_nnunet_array(image_array)
            logger.debug(f"After nnU-Net conversion: shape={image_array.shape}, dtype={image_array.dtype}")
            
            # Create image properties for nnU-Net
            # nnU-Net expects spacing info (one value per spatial dimension)
            image_properties = {"spacing": [1.0] * (image_array.ndim - 1)}
            
            # Predict using nnU-Net
            # The correct nnU-Net v2 API signature is:
            # predict_single_npy_array(data, properties, segmentation_previous_stage)
            logger.debug("Calling nnU-Net predict_single_npy_array...")
            predicted_segmentation = self.nnunet_predictor.predict_single_npy_array(
                image_array,                    # Input array (CxHxW)
                image_properties,               # Spacing metadata
                None                            # segmentation_previous_stage (not used here)
            )
            
            logger.debug(f"Raw output type: {type(predicted_segmentation)}, shape: {getattr(predicted_segmentation, 'shape', 'N/A')}")
            
            if predicted_segmentation is None:
                raise RuntimeError("nnU-Net prediction returned None")
            
            # Postprocess
            mask = self.postprocess_output(predicted_segmentation)
            logger.debug(f"Final mask shape: {mask.shape}, dtype: {mask.dtype}, unique values: {np.unique(mask)[:10]}")
            
            # Calculate metrics
            num_objects, _ = cv2.connectedComponents((mask > 0).astype(np.uint8))
            num_objects = max(0, int(num_objects) - 1)
            
            metadata = {
                "num_spheroids": num_objects,
                "segmentation_coverage": float(np.sum(mask > 0) / mask.size),
                "fold": self.model_config.get('fold', 0)
            }
            
            logger.info(f"Prediction complete: {num_objects} spheroids detected")
            
            return SegmentationResult(
                model_type=ModelType.PYMARAI,
                segmentation_mask=mask,
                raw_output={"output": predicted_segmentation},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image for nnU-Net
        
        Args:
            image: Input image (HxW or HxWxC)
            
        Returns:
            Normalized numpy array
        """
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        input_channels = int(self.model_config.get("input_channels", 1))
        if input_channels == 1 and image.ndim == 3:
            image = np.mean(image[:, :, :3], axis=2)
        elif input_channels == 3 and image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        # Convert to float and normalize
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Normalize to [0, 1]
        max_value = float(np.max(image)) if image.size else 0.0
        if max_value > 1.0:
            image = image / max_value
        
        return image

    @staticmethod
    def _to_nnunet_array(image: np.ndarray) -> np.ndarray:
        """Convert HxW/HxWxC arrays to nnU-Net channel-first arrays without a batch axis."""
        if image.ndim == 2:
            return image[np.newaxis, :, :]

        if image.ndim == 3:
            # PIL/Qt images are HxWxC; nnU-Net expects CxHxW for 2D data.
            if image.shape[-1] in (1, 3):
                return np.moveaxis(image, -1, 0)
            return image

        raise ValueError(f"Unsupported image shape for nnU-Net prediction: {image.shape}")
    
    def postprocess_output(self, raw_output: np.ndarray) -> np.ndarray:
        """Convert nnU-Net output to final segmentation mask
        
        Args:
            raw_output: Raw nnU-Net output
            
        Returns:
            Segmentation mask as numpy array
        """
        if isinstance(raw_output, tuple):
            raw_output = raw_output[0]

        # nnU-Net output is usually (1, H, W) for binary or (C, H, W) for multi-class
        if raw_output.ndim == 3:
            if raw_output.shape[0] == 1:
                mask = raw_output[0]
            else:
                # Multi-class: take argmax
                mask = np.argmax(raw_output, axis=0)
        else:
            mask = raw_output
        
        return mask.astype(np.uint16 if np.max(mask) > 255 else np.uint8)
    
    def unload_model(self):
        """Release model from memory"""
        if self.nnunet_predictor is not None:
            del self.nnunet_predictor
            self.nnunet_predictor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("pyMarAI model unloaded")
