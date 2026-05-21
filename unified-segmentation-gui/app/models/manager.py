#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified model manager for both DeepScratch and pyMarAI
"""

import logging
from typing import Dict, Optional, Any
from pathlib import Path
import numpy as np

from .base import ModelType, BasePredictor, SegmentationResult
from .deepscrath import DeepScratchPredictor
from .pymarai import PyMarAIPredictor

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and running both segmentation models"""
    
    def __init__(self, device: str = "cuda"):
        """Initialize model manager
        
        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device
        self.predictors: Dict[ModelType, Optional[BasePredictor]] = {
            ModelType.DEEPSCRATH: None,
            ModelType.PYMARAI: None
        }
        self.active_model: Optional[ModelType] = None
    
    def load_model(
        self,
        model_type: ModelType,
        model_path: str,
        **kwargs
    ) -> bool:
        """Load a model
        
        Args:
            model_type: Type of model (DEEPSCRATH or PYMARAI)
            model_path: Path to model weights/directory
            **kwargs: Model-specific arguments
            
        Returns:
            Success flag
        """
        try:
            logger.info(f"Loading {model_type.value} model from {model_path}")
            
            # Create predictor if needed
            if self.predictors[model_type] is None:
                if model_type == ModelType.DEEPSCRATH:
                    self.predictors[model_type] = DeepScratchPredictor(device=self.device)
                elif model_type == ModelType.PYMARAI:
                    self.predictors[model_type] = PyMarAIPredictor(device=self.device)
            
            # Load model
            success = self.predictors[model_type].load_model(model_path, **kwargs)
            
            if success:
                self.active_model = model_type
                logger.info(f"Successfully loaded {model_type.value}")
            else:
                logger.error(f"Failed to load {model_type.value}")
                if self.predictors[model_type] is not None:
                    self.predictors[model_type].unload_model()
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(
        self,
        image: np.ndarray,
        model_type: Optional[ModelType] = None
    ) -> Optional[SegmentationResult]:
        """Run prediction with specified or active model
        
        Args:
            image: Input image as numpy array
            model_type: Model to use (None = use active model)
            
        Returns:
            SegmentationResult or None on failure
        """
        model_type = model_type or self.active_model
        
        if model_type is None:
            logger.error("No model selected. Load a model first.")
            return None
        
        predictor = self.predictors.get(model_type)
        if predictor is None:
            logger.error(f"Model {model_type.value} not loaded")
            return None
        
        if not predictor.is_loaded:
            logger.error(f"Model {model_type.value} is not fully initialized. Try reloading the model.")
            return None
        
        try:
            logger.debug(f"Running {model_type.value} prediction on image shape {image.shape}")
            result = predictor.predict(image)
            
            if result is None:
                logger.error(f"Model {model_type.value} returned None result")
                return None
            
            logger.debug(f"Prediction successful: {model_type.value}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {model_type.value}: {e}", exc_info=True)
            return None
    
    def unload_model(self, model_type: Optional[ModelType] = None):
        """Unload a model to free memory
        
        Args:
            model_type: Model to unload (None = unload all)
        """
        if model_type is None:
            # Unload all
            for mtype in ModelType:
                if self.predictors[mtype] is not None:
                    self.predictors[mtype].unload_model()
            self.active_model = None
        else:
            if self.predictors[model_type] is not None:
                self.predictors[model_type].unload_model()
            
            if self.active_model == model_type:
                self.active_model = None
    
    def get_active_model(self) -> Optional[ModelType]:
        """Get currently active model"""
        return self.active_model
    
    def is_model_loaded(self, model_type: ModelType) -> bool:
        """Check if a model is loaded"""
        predictor = self.predictors[model_type]
        return predictor is not None and predictor.is_loaded
    
    def get_model_config(self, model_type: ModelType) -> Dict[str, Any]:
        """Get configuration of a loaded model"""
        if not self.is_model_loaded(model_type):
            return {}
        return self.predictors[model_type].model_config
