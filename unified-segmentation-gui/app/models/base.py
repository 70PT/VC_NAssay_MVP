#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base model interface for unified segmentation
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image


class ModelType(Enum):
    """Supported model types"""
    DEEPSCRATH = "deepscrath"  # Cell detection
    PYMARAI = "pymarai"         # Spheroid segmentation


class SegmentationResult:
    """Standardized result container for segmentation predictions"""
    
    def __init__(
        self,
        model_type: ModelType,
        segmentation_mask: np.ndarray,
        raw_output: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ):
        self.model_type = model_type
        self.segmentation_mask = segmentation_mask
        self.raw_output = raw_output
        self.metadata = metadata or {}
    
    def get_metrics(self) -> Dict[str, float]:
        """Extract metrics from segmentation"""
        metrics = {}
        mask = self.segmentation_mask
        
        # Basic metrics
        metrics['num_objects'] = int(np.max(mask))
        metrics['mask_coverage'] = float(np.sum(mask > 0) / mask.size)
        
        return metrics


class BasePredictor(ABC):
    """Abstract base class for model predictors"""
    
    def __init__(self, model_type: ModelType, device: str = "cuda"):
        self.model_type = model_type
        self.device = device
        self.model = None
        self.model_config = {}

    @property
    def is_loaded(self) -> bool:
        """Return True when the predictor has an initialized model."""
        return self.model is not None
    
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> bool:
        """Load model weights from path
        
        Args:
            model_path: Path to model weights/checkpoint
            **kwargs: Additional model-specific arguments
            
        Returns:
            Success flag
        """
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> SegmentationResult:
        """Run segmentation prediction on image
        
        Args:
            image: Input image as numpy array (HxWxC or HxW)
            
        Returns:
            SegmentationResult object with mask and metadata
        """
        pass
    
    @abstractmethod
    def unload_model(self):
        """Release model from memory"""
        pass
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for model input (normalize, resize, etc.)
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        return image
    
    def postprocess_output(self, raw_output: Any) -> np.ndarray:
        """Convert raw model output to segmentation mask
        
        Args:
            raw_output: Raw model output
            
        Returns:
            Segmentation mask as numpy array
        """
        return raw_output
