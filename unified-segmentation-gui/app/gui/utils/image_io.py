#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image loading and export helpers used by the GUI tabs."""

import csv
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PIL import Image, ImageOps


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def is_supported_image(path: Path) -> bool:
    """Return True if path looks like a supported image file."""
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def load_image_array(file_path: str) -> np.ndarray:
    """Load an image as a numpy array using PIL's RGB channel order."""
    with Image.open(file_path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode == "P":
            img = img.convert("RGB")
        return np.array(img)


def mask_for_image_export(mask: np.ndarray) -> np.ndarray:
    """Prepare a segmentation mask for PNG/TIFF export without losing labels unnecessarily."""
    mask = np.asarray(mask)

    if mask.dtype == np.bool_:
        return mask.astype(np.uint8) * 255

    if np.issubdtype(mask.dtype, np.floating):
        finite_mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        max_value = float(np.max(finite_mask)) if finite_mask.size else 0.0
        if max_value <= 1.0:
            return np.clip(finite_mask * 255, 0, 255).astype(np.uint8)
        return np.clip(finite_mask, 0, 65535).astype(np.uint16)

    max_value = int(np.max(mask)) if mask.size else 0
    if max_value <= 255:
        return mask.astype(np.uint8, copy=False)
    return mask.astype(np.uint16, copy=False)


def save_mask(file_path: str, mask: np.ndarray):
    """Save a mask to NPY, CSV, PNG, TIFF, or another PIL-supported image format."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        np.save(str(path), mask)
    elif suffix == ".csv":
        np.savetxt(str(path), np.asarray(mask), delimiter=",", fmt="%s")
    else:
        Image.fromarray(mask_for_image_export(mask)).save(str(path))


def save_metrics(file_path: str, metadata: Dict[str, Any]):
    """Save prediction metrics next to an exported mask."""
    path = Path(file_path)
    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, default=str)
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metadata.items():
            writer.writerow([key, value])
