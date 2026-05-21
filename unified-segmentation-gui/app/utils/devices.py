#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime device discovery for PyTorch-backed predictors."""

import logging
from typing import List

logger = logging.getLogger(__name__)


DEVICE_LABELS = {
    "cpu": "CPU",
    "cuda": "NVIDIA GPU (CUDA)",
    "mps": "Apple GPU (Metal/MPS)",
}


def available_devices() -> List[str]:
    """Return the PyTorch devices available in this Python environment."""
    devices = ["cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            devices.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
    except Exception as exc:
        logger.warning("Could not inspect PyTorch devices: %s", exc)
    return devices


def resolve_device(requested: str) -> str:
    """Resolve a configured device name to a currently available backend."""
    if requested == "auto":
        devices = available_devices()
        if "cuda" in devices:
            return "cuda"
        if "mps" in devices:
            return "mps"
        return "cpu"

    return requested if requested in available_devices() else "cpu"
