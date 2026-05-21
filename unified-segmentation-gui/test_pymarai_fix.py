#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic script to test PyMarAI model loading and prediction
Run this to identify why predictions are returning None
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_model_loading():
    """Test PyMarAI model loading"""
    logger.info("=" * 60)
    logger.info("Testing PyMarAI Model Loading")
    logger.info("=" * 60)
    
    try:
        from app.models.pymarai import PyMarAIPredictor
        from app.models.base import ModelType
        
        logger.info("✓ Imports successful")
        
        # Create predictor
        predictor = PyMarAIPredictor(device="cpu")
        logger.info("✓ PyMarAIPredictor created")
        logger.info(f"  - is_loaded: {predictor.is_loaded}")
        logger.info(f"  - nnunet_predictor: {predictor.nnunet_predictor}")
        
        # Try loading a model
        from app.utils.model_paths import DEFAULT_PYMARAI_MODEL_DIR
        
        model_path = Path(DEFAULT_PYMARAI_MODEL_DIR)
        logger.info(f"\nAttempting to load model from: {model_path}")
        logger.info(f"  - Path exists: {model_path.exists()}")
        logger.info(f"  - Is directory: {model_path.is_dir()}")
        
        if model_path.exists() and model_path.is_dir():
            logger.info(f"  - Contents: {list(model_path.iterdir())[:5]}")
        
        # Check if nnunetv2 is installed
        try:
            import nnunetv2
            logger.info("✓ nnU-Net v2 is installed")
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            logger.info("✓ Can import nnUNetPredictor")
        except ImportError as e:
            logger.error(f"✗ nnU-Net v2 not available: {e}")
            return False
        
        # Try loading the model
        logger.info("\nAttempting model load...")
        if model_path.exists():
            success = predictor.load_model(str(model_path))
            logger.info(f"Load result: {success}")
            logger.info(f"  - is_loaded: {predictor.is_loaded}")
            logger.info(f"  - nnunet_predictor: {predictor.nnunet_predictor}")
            
            if predictor.nnunet_predictor:
                logger.info(f"  - Has network: {hasattr(predictor.nnunet_predictor, 'network')}")
                if hasattr(predictor.nnunet_predictor, 'network'):
                    logger.info(f"  - Network is None: {predictor.nnunet_predictor.network is None}")
            
            return success
        else:
            logger.warning("Model path does not exist - skipping load test")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error during model loading test: {e}", exc_info=True)
        return False


def test_prediction():
    """Test PyMarAI prediction with synthetic image"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing PyMarAI Prediction")
    logger.info("=" * 60)
    
    try:
        from app.models.pymarai import PyMarAIPredictor
        from app.models.base import ModelType
        from app.utils.model_paths import DEFAULT_PYMARAI_MODEL_DIR
        
        model_path = Path(DEFAULT_PYMARAI_MODEL_DIR)
        
        if not model_path.exists():
            logger.warning(f"Model path does not exist: {model_path}")
            return False
        
        # Create and load predictor
        predictor = PyMarAIPredictor(device="cpu")
        success = predictor.load_model(str(model_path))
        
        if not success:
            logger.error("Failed to load model")
            return False
        
        logger.info("✓ Model loaded successfully")
        
        # Create synthetic test image
        logger.info("\nCreating synthetic test image...")
        test_image = np.random.randint(0, 256, size=(256, 256, 1), dtype=np.uint8).astype(np.float32)
        logger.info(f"  - Image shape: {test_image.shape}")
        logger.info(f"  - Image dtype: {test_image.dtype}")
        logger.info(f"  - Image range: [{test_image.min()}, {test_image.max()}]")
        
        # Run prediction
        logger.info("\nRunning prediction...")
        try:
            result = predictor.predict(test_image)
            
            if result is None:
                logger.error("✗ Prediction returned None")
                return False
            
            logger.info("✓ Prediction successful")
            logger.info(f"  - Result type: {type(result)}")
            logger.info(f"  - Mask shape: {result.segmentation_mask.shape}")
            logger.info(f"  - Mask dtype: {result.segmentation_mask.dtype}")
            logger.info(f"  - Mask range: [{result.segmentation_mask.min()}, {result.segmentation_mask.max()}]")
            logger.info(f"  - Metadata: {result.metadata}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Prediction failed: {e}", exc_info=True)
            return False
            
    except Exception as e:
        logger.error(f"✗ Error during prediction test: {e}", exc_info=True)
        return False


def test_manager():
    """Test ModelManager with PyMarAI"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing ModelManager")
    logger.info("=" * 60)
    
    try:
        from app.models.manager import ModelManager
        from app.models.base import ModelType
        from app.utils.model_paths import DEFAULT_PYMARAI_MODEL_DIR
        
        model_path = Path(DEFAULT_PYMARAI_MODEL_DIR)
        
        if not model_path.exists():
            logger.warning(f"Model path does not exist: {model_path}")
            return False
        
        # Create manager
        manager = ModelManager(device="cpu")
        logger.info("✓ ModelManager created")
        
        # Load model
        logger.info(f"\nLoading PyMarAI model from {model_path}...")
        success = manager.load_model(ModelType.PYMARAI, str(model_path))
        logger.info(f"Load result: {success}")
        
        if not success:
            logger.error("Failed to load model via manager")
            return False
        
        # Check if model is loaded
        is_loaded = manager.is_model_loaded(ModelType.PYMARAI)
        logger.info(f"  - is_model_loaded: {is_loaded}")
        
        # Create synthetic test image
        logger.info("\nCreating synthetic test image...")
        test_image = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8).astype(np.float32)
        logger.info(f"  - Image shape: {test_image.shape}")
        
        # Run prediction
        logger.info("\nRunning prediction via manager...")
        result = manager.predict(test_image, model_type=ModelType.PYMARAI)
        
        if result is None:
            logger.error("✗ Prediction returned None")
            return False
        
        logger.info("✓ Prediction successful")
        logger.info(f"  - Result type: {type(result)}")
        logger.info(f"  - Mask shape: {result.segmentation_mask.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error during manager test: {e}", exc_info=True)
        return False


def main():
    """Run all diagnostic tests"""
    logger.info("Starting PyMarAI diagnostic tests...")
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Prediction", test_prediction),
        ("ModelManager", test_manager),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}", exc_info=True)
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("✓ All tests passed!")
    else:
        logger.info("✗ Some tests failed. Check logs above for details.")
    logger.info("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
