#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Default local model locations for the unified GUI."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_PYMARAI_MODEL_DIR = (
    MODELS_DIR
    / "pymarai"
    / "Dataset001_spheroids_V1"
    / "nnUNetTrainer__nnUNetPlans__2d"
)

DEFAULT_DEEPSCRATCH_MODEL_PATH = (
    MODELS_DIR
    / "deepscratch"
    / "model_best.pth.tar"
)
