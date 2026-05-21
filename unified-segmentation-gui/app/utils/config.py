#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management for unified segmentation GUI
"""

import json
import logging
import copy
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from PyQt5.QtCore import QSettings

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages application settings and configuration"""
    
    DEFAULT_CONFIG = {
        "device": "auto",
        "workspace_path": str(Path.home() / "segmentation_workspace"),
        "recent_images": [],
        "models": {
            "deepscrath": {
                "model_path": None,
                "model_name": "unet-flat-48",
                "enabled": True,
            },
            "pymarai": {
                "model_path": None,
                "model_name": "Dataset001_spheroids_V1",
                "fold": 0,
                "enabled": True,
            }
        },
        "ui": {
            "window_geometry": None,
            "window_state": None,
            "theme": "light",
            "default_zoom": 1.0,
        },
        "export": {
            "export_masks": True,
            "export_metrics": True,
            "export_format": "PNG",
        }
    }
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize config manager
        
        Args:
            config_file: Path to config JSON file
        """
        if config_file is None:
            config_file = Path.home() / ".unified_segmentation_gui" / "config.json"
        
        self.config_file = self._resolve_config_file(Path(config_file))
        
        self.qsettings = QSettings("unified-segmentation-gui", "unified-segmentation-gui")
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        
        # Load existing config
        self.load()
    
    def load(self) -> bool:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                    # Merge with defaults
                    self._deep_merge(self.config, loaded)
                logger.info(f"Loaded config from {self.config_file}")
                return True
            else:
                logger.info("Using default configuration")
                return False
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False

    @staticmethod
    def _resolve_config_file(config_file: Path) -> Path:
        """Return a writable config path, falling back for sandboxed runs."""
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=".write-test-",
                dir=str(config_file.parent),
                delete=True
            ):
                pass
            return config_file
        except OSError:
            fallback = Path(tempfile.gettempdir()) / "unified_segmentation_gui" / "config.json"
            fallback.parent.mkdir(parents=True, exist_ok=True)
            logger.warning("Config path is not writable; using %s", fallback)
            return fallback
    
    def save(self) -> bool:
        """Save configuration to file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            logger.info(f"Saved config to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)
        
        Args:
            key: Configuration key (e.g., 'models.deepscrath.model_path')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key (supports dot notation)
        
        Args:
            key: Configuration key (e.g., 'models.deepscrath.model_path')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model
        
        Args:
            model_name: Model name ('deepscrath' or 'pymarai')
            
        Returns:
            Model configuration dict
        """
        return self.get(f"models.{model_name}", {})
    
    def set_model_config(self, model_name: str, config: Dict[str, Any]):
        """Set configuration for a specific model
        
        Args:
            model_name: Model name
            config: Configuration dict to set
        """
        existing = self.get_model_config(model_name).copy()
        existing.update(config)
        self.set(f"models.{model_name}", existing)
    
    def add_recent_image(self, image_path: str, max_recent: int = 10):
        """Add image to recent files list
        
        Args:
            image_path: Path to image file
            max_recent: Maximum number of recent files to keep
        """
        recent = self.get("recent_images", [])
        recent = [img for img in recent if img != image_path]
        recent.insert(0, image_path)
        self.set("recent_images", recent[:max_recent])
    
    def get_recent_images(self) -> list:
        """Get list of recently opened images
        
        Returns:
            List of image paths
        """
        return self.get("recent_images", [])
    
    @staticmethod
    def _deep_merge(base: dict, update: dict):
        """Recursively merge update dict into base dict
        
        Args:
            base: Base configuration dict
            update: Update configuration dict
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigManager._deep_merge(base[key], value)
            else:
                base[key] = value
