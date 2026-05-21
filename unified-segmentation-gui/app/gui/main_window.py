#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main window for unified segmentation GUI
"""

import logging
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QMessageBox, QStatusBar, QFileDialog, QComboBox
)
from PyQt5.QtCore import Qt, QSettings, QSize
from PyQt5.QtGui import QIcon, QFont

from app.gui.tabs.deepscrath import DeepScratchTab
from app.gui.tabs.pymarai import PyMarAITab
from app.models.manager import ModelManager
from app.models.base import ModelType
from app.utils.config import ConfigManager
from app.utils.devices import DEVICE_LABELS, available_devices, resolve_device

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize managers
        self.config = ConfigManager()
        device = self._resolve_configured_device(self.config.get("device", "cuda"))
        self.config.set("device", device)
        self.model_manager = ModelManager(device=device)
        
        # Setup UI
        self.setup_ui()
        self.restore_window_state()
        
        # Set title and window properties
        self.setWindowTitle("Unified Segmentation GUI - DeepScratch & pyMarAI")
        self.setMinimumSize(QSize(1400, 900))
        self.auto_load_models()
    
    def setup_ui(self):
        """Setup the main UI"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Header with info
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Unified Image Segmentation Platform"))
        header_layout.addStretch()
        
        self.device_label = QLabel(f"Device: {self._device_display_name(self.config.get('device', 'cpu'))}")
        self.device_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self.device_label)
        
        layout.addLayout(header_layout)
        
        # Tabs
        self.tabs = QTabWidget()
        
        # DeepScratch tab
        self.deepscrath_tab = DeepScratchTab(self.model_manager, self.config)
        self.tabs.addTab(self.deepscrath_tab, "Cell Detection (DeepScratch)")
        
        # pyMarAI tab
        self.pymarai_tab = PyMarAITab(self.model_manager, self.config)
        self.tabs.addTab(self.pymarai_tab, "Spheroid Segmentation (pyMarAI)")
        
        # Settings tab
        self.settings_tab = QWidget()
        self.setup_settings_tab()
        self.tabs.addTab(self.settings_tab, "Settings")
        
        layout.addWidget(self.tabs)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        self.deepscrath_tab.status_updated.connect(self.status_bar.showMessage)
        self.pymarai_tab.status_updated.connect(self.status_bar.showMessage)
        
        central_widget.setLayout(layout)
        
        # Menu bar
        self.setup_menu()
    
    def setup_settings_tab(self):
        """Setup settings tab"""
        layout = QVBoxLayout()
        
        # Device settings
        self.settings_device_label = QLabel(f"Current Device: {self._device_display_name(self.config.get('device', 'cpu'))}")
        layout.addWidget(self.settings_device_label)
        
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto", "auto")
        for device in available_devices():
            self.device_combo.addItem(DEVICE_LABELS.get(device, device.upper()), device)
        configured_device = self.config.get("device", "auto")
        combo_index = self.device_combo.findData(configured_device)
        if combo_index < 0:
            combo_index = self.device_combo.findData("auto")
        self.device_combo.setCurrentIndex(combo_index)
        self.device_combo.currentIndexChanged.connect(self.change_device)
        self.device_combo.setToolTip(
            "Select where PyTorch runs inference. On this Mac, CUDA is unavailable; "
            "Apple GPU appears only when this Python/PyTorch build supports Metal/MPS."
        )
        layout.addWidget(self.device_combo)

        self.device_help_label = QLabel(self._device_help_text())
        self.device_help_label.setWordWrap(True)
        layout.addWidget(self.device_help_label)
        
        # Workspace settings
        layout.addWidget(QLabel("\nWorkspace:"))
        self.workspace_label = QLabel(f"Path: {self.config.get('workspace_path')}")
        layout.addWidget(self.workspace_label)
        
        btn_workspace = QPushButton("Change Workspace")
        btn_workspace.clicked.connect(self.change_workspace)
        layout.addWidget(btn_workspace)
        
        # Model paths
        layout.addWidget(QLabel("\nModel Paths:"))
        
        ds_path = self.config.get('models.deepscrath.model_path', 'Not set')
        self.deepscrath_model_label = QLabel(f"DeepScratch: {ds_path}")
        layout.addWidget(self.deepscrath_model_label)
        
        pm_path = self.config.get('models.pymarai.model_path', 'Not set')
        self.pymarai_model_label = QLabel(f"pyMarAI: {pm_path}")
        layout.addWidget(self.pymarai_model_label)
        
        # Clear cache button
        layout.addWidget(QLabel(""))
        btn_clear = QPushButton("Clear Model Cache")
        btn_clear.clicked.connect(self.clear_cache)
        layout.addWidget(btn_clear)
        
        layout.addStretch()
        self.settings_tab.setLayout(layout)

    @staticmethod
    def _resolve_configured_device(device: str) -> str:
        """Resolve a configured device into an available runtime backend."""
        return resolve_device(device)

    @staticmethod
    def _device_display_name(device: str) -> str:
        """Return a short display name for a device."""
        return DEVICE_LABELS.get(device, device.upper())

    @staticmethod
    def _device_help_text() -> str:
        """Explain why GPU choices may not be available."""
        devices = available_devices()
        if "cuda" in devices or "mps" in devices:
            return "Changing device unloads loaded models and reloads default models on the selected backend."
        return "No GPU backend was detected in this Python environment, so inference will run on CPU."
    
    def setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        action_quit = file_menu.addAction("Quit")
        action_quit.triggered.connect(self.close)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        action_about = help_menu.addAction("About")
        action_about.triggered.connect(self.show_about)
        
        action_doc = help_menu.addAction("Documentation")
        action_doc.triggered.connect(self.show_documentation)

        action_options = help_menu.addAction("Option Guide")
        action_options.triggered.connect(self.show_option_guide)
    
    def change_device(self):
        """Change the PyTorch inference device from the settings combo."""
        requested_device = self.device_combo.currentData()
        new_device = self._resolve_configured_device(requested_device)
        
        # Unload all models
        self.model_manager.unload_model()
        
        # Create new manager with new device
        self.model_manager = ModelManager(device=new_device)
        self.config.set("device", new_device)
        self.config.save()
        self._refresh_settings_labels()
        
        self.status_bar.showMessage(f"Switched to {new_device.upper()}")
        QMessageBox.information(self, "Device Switched", f"Now using {new_device.upper()}")
        
        # Refresh tabs
        self.deepscrath_tab.model_manager = self.model_manager
        self.pymarai_tab.model_manager = self.model_manager
        self._disable_model_actions()
        self.auto_load_models()
    
    def change_workspace(self):
        """Change workspace directory"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Workspace Directory",
            self.config.get("workspace_path")
        )
        
        if folder:
            self.config.set("workspace_path", folder)
            self.config.save()
            self._refresh_settings_labels()
            self.status_bar.showMessage(f"Workspace changed to {folder}")
    
    def clear_cache(self):
        """Clear model cache"""
        self.model_manager.unload_model()
        self._disable_model_actions()
        self.status_bar.showMessage("Model cache cleared")
        QMessageBox.information(self, "Cache Cleared", "All models have been unloaded from memory")

    def _disable_model_actions(self):
        """Disable prediction actions after models are unloaded."""
        for tab in (self.deepscrath_tab, self.pymarai_tab):
            tab.btn_predict.setEnabled(False)
            tab.btn_batch.setEnabled(False)
            tab.btn_export.setEnabled(False)

    def _refresh_settings_labels(self):
        """Refresh settings labels after config changes."""
        device = self.config.get("device", "cpu")
        self.device_label.setText(f"Device: {self._device_display_name(device)}")
        self.settings_device_label.setText(f"Current Device: {self._device_display_name(device)}")
        if hasattr(self, "device_help_label"):
            self.device_help_label.setText(self._device_help_text())
        self.workspace_label.setText(f"Path: {self.config.get('workspace_path')}")
        self.deepscrath_model_label.setText(
            f"DeepScratch: {self.config.get('models.deepscrath.model_path', 'Not set')}"
        )
        self.pymarai_model_label.setText(
            f"pyMarAI: {self.config.get('models.pymarai.model_path', 'Not set')}"
        )

    def auto_load_models(self):
        """Load configured/default models at startup when their files are available."""
        self.deepscrath_tab.auto_load_model()
        self.pymarai_tab.auto_load_model()
        self._refresh_settings_labels()
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Unified Segmentation GUI",
            "Unified Segmentation GUI v0.1.0\n\n"
            "A unified PyQt5 interface for:\n"
            "- DeepScratch: Cell detection in microscopy images\n"
            "- pyMarAI: Spheroid segmentation using nnU-Net\n\n"
            "© 2025 - Research Use Only"
        )
    
    def show_documentation(self):
        """Show documentation"""
        QMessageBox.information(
            self,
            "Documentation",
            "Documentation and guides:\n\n"
            "DeepScratch: github.com/sailem-group/DeepScratch\n"
            "pyMarAI: github.com/hzdr-MedImaging/pyMarAI\n\n"
            "For detailed usage, see the README.md files in each repository."
        )

    def show_option_guide(self):
        """Explain model and segmentation options."""
        QMessageBox.information(
            self,
            "Option Guide",
            "Device: CPU is always available. NVIDIA GPU requires CUDA. Apple GPU requires a PyTorch build with Metal/MPS.\n\n"
            "DeepScratch task: Cell detection returns cell-center detections; semantic segmentation returns a class mask.\n"
            "DeepScratch model name: must match the checkpoint architecture.\n"
            "Input channels: 1 for grayscale, 3 for RGB.\n"
            "Output classes: number of output maps/classes in the checkpoint.\n"
            "Loss type: training loss identifier; detection usually uses l2-G1.5, segmentation usually uses crossentropy.\n\n"
            "pyMarAI model name: Dataset001_spheroids_V1 is the bundled spheroid model.\n"
            "Fold: trained cross-validation fold to use; one fold is faster and lighter than loading an ensemble.\n"
            "Microscope type: pixel-size metadata from the pyMarAI workflow.\n\n"
            "Opacity and color affect only the preview overlay. Export writes the mask and metrics."
        )
    
    def save_window_state(self):
        """Save window geometry and state"""
        settings = QSettings("unified-segmentation-gui", "unified-segmentation-gui")
        settings.setValue("window_geometry", self.saveGeometry())
        settings.setValue("window_state", self.saveState())
        settings.setValue("active_tab", self.tabs.currentIndex())
    
    def restore_window_state(self):
        """Restore window geometry and state"""
        settings = QSettings("unified-segmentation-gui", "unified-segmentation-gui")
        
        geometry = settings.value("window_geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        state = settings.value("window_state")
        if state:
            self.restoreState(state)
        
        active_tab = settings.value("active_tab", 0)
        if active_tab is not None:
            self.tabs.setCurrentIndex(int(active_tab))
    
    def closeEvent(self, event):
        """Handle window close"""
        self.save_window_state()
        self.model_manager.unload_model()
        self.config.save()
        event.accept()
