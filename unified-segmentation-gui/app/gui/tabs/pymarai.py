#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyMarAI spheroid segmentation tab
"""

import logging
from pathlib import Path
import numpy as np
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QSpinBox, QComboBox, QMessageBox, QProgressBar, QGroupBox, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from app.gui.utils.components import (
    ImageBrowser, ImagePreviewCanvas, MaskVisualizationControls, ExportDialog
)
from app.gui.utils.image_io import load_image_array, save_mask, save_metrics
from app.gui.utils.threading import BatchPredictionWorker, PredictionWorker
from app.models.manager import ModelManager
from app.models.base import ModelType
from app.utils.config import ConfigManager
from app.utils.model_paths import DEFAULT_PYMARAI_MODEL_DIR

logger = logging.getLogger(__name__)


class PyMarAITab(QWidget):
    """Tab for pyMarAI spheroid segmentation using nnU-Net"""
    
    status_updated = pyqtSignal(str)
    
    def __init__(self, model_manager: ModelManager, config: ConfigManager):
        super().__init__()
        self.model_manager = model_manager
        self.config = config
        self.current_image = None
        self.current_mask = None
        self.current_metadata = {}
        self.current_image_path = None
        self.prediction_worker = None
        self.batch_worker = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI"""
        main_layout = QHBoxLayout()
        
        # Left: Controls
        left_layout = QVBoxLayout()
        
        # Model loading section
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        
        self.btn_load_model = QPushButton("Change nnU-Net Model")
        self.btn_load_model.setToolTip(
            "Choose an nnU-Net trained model folder. The bundled pyMarAI model is loaded automatically when present."
        )
        self.btn_load_model.clicked.connect(self.load_model)
        model_layout.addWidget(self.btn_load_model)
        
        model_layout.addWidget(QLabel("Model Name:"))
        self.model_name_combo = QComboBox()
        self.model_name_combo.addItems([
            "Dataset001_spheroids_V1",
            "Dataset002_advanced",
        ])
        self.model_name_combo.setToolTip(
            "Dataset/model identifier. Dataset001_spheroids_V1 is the bundled pyMarAI spheroid model."
        )
        model_layout.addWidget(self.model_name_combo)
        
        model_layout.addWidget(QLabel("Cross-validation Fold:"))
        self.fold_spinbox = QSpinBox()
        self.fold_spinbox.setRange(0, 4)
        self.fold_spinbox.setValue(0)
        self.fold_spinbox.setToolTip(
            "Cross-validation fold to use. A single fold is faster and lighter; folds 0-4 are available in the bundled model."
        )
        model_layout.addWidget(self.fold_spinbox)
        
        model_layout.addWidget(QLabel("Input Channels:"))
        self.input_channels_spinbox = QSpinBox()
        self.input_channels_spinbox.setRange(1, 3)
        self.input_channels_spinbox.setValue(1)
        self.input_channels_spinbox.setToolTip(
            "Number of image channels passed to nnU-Net. Use 1 for grayscale microscopy, 3 for RGB images."
        )
        model_layout.addWidget(self.input_channels_spinbox)

        self.model_status_label = QLabel("Model: not loaded")
        self.model_status_label.setWordWrap(True)
        model_layout.addWidget(self.model_status_label)
        
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)
        
        # Image browser
        left_layout.addWidget(QLabel("Images:"))
        self.image_browser = ImageBrowser()
        self.image_browser.image_selected.connect(self.load_image)
        left_layout.addWidget(self.image_browser)
        
        # Microscope settings
        microscope_group = QGroupBox("Microscope Settings")
        microscope_layout = QVBoxLayout()
        
        microscope_layout.addWidget(QLabel("Microscope Type:"))
        self.microscope_combo = QComboBox()
        self.microscope_combo.addItems(self.load_microscope_options())
        self.microscope_combo.setToolTip(
            "Pixel size metadata from pyMarAI. It matters for the original ECAT pipeline; direct image-array inference keeps it as reference metadata."
        )
        microscope_layout.addWidget(self.microscope_combo)
        
        microscope_group.setLayout(microscope_layout)
        left_layout.addWidget(microscope_group)
        
        # Segmentation controls
        seg_group = QGroupBox("Segmentation")
        seg_layout = QVBoxLayout()
        
        self.btn_predict = QPushButton("Run Segmentation")
        self.btn_predict.setToolTip("Run the loaded pyMarAI/nnU-Net model on the selected image.")
        self.btn_predict.clicked.connect(self.run_prediction)
        self.btn_predict.setEnabled(False)
        seg_layout.addWidget(self.btn_predict)

        self.btn_batch = QPushButton("Run Folder Batch")
        self.btn_batch.setToolTip("Run the loaded pyMarAI/nnU-Net model on every image listed in the folder browser.")
        self.btn_batch.clicked.connect(self.run_batch_prediction)
        self.btn_batch.setEnabled(False)
        seg_layout.addWidget(self.btn_batch)
        
        seg_group.setLayout(seg_layout)
        left_layout.addWidget(seg_group)
        
        # Visualization controls
        left_layout.addWidget(QLabel("Visualization:"))
        self.viz_controls = MaskVisualizationControls()
        self.viz_controls.mask_settings_changed.connect(self.update_visualization)
        left_layout.addWidget(self.viz_controls)
        
        # Export section
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        
        self.btn_export = QPushButton("Export Results")
        self.btn_export.setToolTip("Save the current spheroid mask and a CSV file with summary metrics.")
        self.btn_export.clicked.connect(self.export_results)
        self.btn_export.setEnabled(False)
        export_layout.addWidget(self.btn_export)
        
        export_group.setLayout(export_layout)
        left_layout.addWidget(export_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        left_layout.addWidget(self.status_label)
        
        left_layout.addStretch()
        
        # Right: Image preview
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Preview:"))
        self.image_canvas = ImagePreviewCanvas()
        right_layout.addWidget(self.image_canvas)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        btn_zoom_in = QPushButton("Zoom In")
        btn_zoom_in.clicked.connect(self.image_canvas.zoom_in)
        btn_zoom_out = QPushButton("Zoom Out")
        btn_zoom_out.clicked.connect(self.image_canvas.zoom_out)
        btn_reset_zoom = QPushButton("Reset Zoom")
        btn_reset_zoom.clicked.connect(self.image_canvas.reset_zoom)
        
        zoom_layout.addWidget(btn_zoom_in)
        zoom_layout.addWidget(btn_zoom_out)
        zoom_layout.addWidget(btn_reset_zoom)
        zoom_layout.addStretch()
        right_layout.addLayout(zoom_layout)
        
        # Add left and right to main
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)
        
        self.setLayout(main_layout)
        
        # Connect status updates
        self.status_updated.connect(self.update_status_label)

    def load_microscope_options(self) -> list:
        """Load microscope options from pyMarAI config when available."""
        candidates = [
            Path("/usr/local/etc/pymarai.yml"),
            Path(__file__).resolve().parents[4] / "pyMarAI" / "pymarai.yml",
        ]
        for config_path in candidates:
            if not config_path.exists():
                continue
            try:
                import yaml
                with config_path.open("r", encoding="utf-8") as handle:
                    config = yaml.safe_load(handle) or {}
                microscopes = config.get("microscopes") or []
                if microscopes:
                    return microscopes
            except Exception as exc:
                logger.warning("Could not load pyMarAI microscope config %s: %s", config_path, exc)
        return [
            "-: Please select resolution of used microscope",
            "1: 0.2012 pixel/um",
            "2: 0.3534 pixel/um",
            "3: 1.079 pixel/um",
            "4: 1.449 pixel/um",
        ]

    def auto_load_model(self):
        """Load the configured pyMarAI model or the bundled default model."""
        model_config = self.config.get_model_config("pymarai")
        configured_path = model_config.get("model_path")
        model_path = configured_path if configured_path and Path(configured_path).is_dir() else str(DEFAULT_PYMARAI_MODEL_DIR)

        if not Path(model_path).is_dir():
            self.model_status_label.setText(
                f"Model: no pyMarAI model found at {DEFAULT_PYMARAI_MODEL_DIR}"
            )
            self.status_updated.emit("pyMarAI model not auto-loaded: model folder missing")
            return False

        self._apply_model_config(model_config)
        return self._load_model_from_path(model_path, show_error=False)

    def _apply_model_config(self, model_config: dict):
        """Populate controls from saved model configuration."""
        model_name = model_config.get("model_name")
        if model_name:
            index = self.model_name_combo.findText(model_name)
            if index >= 0:
                self.model_name_combo.setCurrentIndex(index)
        self.fold_spinbox.setValue(int(model_config.get("fold", self.fold_spinbox.value())))
        self.input_channels_spinbox.setValue(
            int(model_config.get("input_channels", self.input_channels_spinbox.value()))
        )

    def _current_model_options(self) -> dict:
        """Return model-loading options from current controls."""
        return {
            "model_name": self.model_name_combo.currentText(),
            "fold": self.fold_spinbox.value(),
            "input_channels": self.input_channels_spinbox.value(),
        }

    def _load_model_from_path(self, folder_path: str, show_error: bool = True) -> bool:
        """Load a pyMarAI/nnU-Net model from a concrete folder."""
        self.status_updated.emit("Loading pyMarAI model...")
        options = self._current_model_options()
        success = self.model_manager.load_model(ModelType.PYMARAI, folder_path, **options)

        if success:
            self.btn_predict.setEnabled(True)
            self.btn_batch.setEnabled(True)
            self.model_status_label.setText(f"Model: loaded {Path(folder_path).name} fold {options['fold']}")
            self.status_updated.emit("✓ pyMarAI model loaded")
            self.config.set_model_config('pymarai', {'model_path': folder_path, **options})
            self.config.save()
            return True

        self.btn_predict.setEnabled(False)
        self.btn_batch.setEnabled(False)
        self.model_status_label.setText("Model: failed to load")
        self.status_updated.emit("✗ Failed to load pyMarAI model")
        if show_error:
            QMessageBox.critical(self, "Error", "Failed to load model. Make sure nnU-Net is installed.")
        return False
    
    def load_model(self):
        """Load pyMarAI/nnU-Net model"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select nnU-Net Model Directory",
            str(Path.home())
        )
        
        if folder_path:
            self._load_model_from_path(folder_path)
    
    def load_image(self, file_path: str):
        """Load image for segmentation
        
        Args:
            file_path: Path to image file
        """
        try:
            self.current_image = load_image_array(file_path)
            self.current_image_path = file_path
            
            # Display image
            self.image_canvas.set_image(self.current_image)
            self.current_mask = None
            self.current_metadata = {}
            self.btn_export.setEnabled(False)
            
            self.status_updated.emit(f"Loaded: {Path(file_path).name}")
            self.config.add_recent_image(file_path)
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            self.status_updated.emit("✗ Failed to load image")
    
    def run_prediction(self):
        """Run segmentation prediction"""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return
        
        if not self.model_manager.is_model_loaded(ModelType.PYMARAI):
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return
        
        # Start prediction worker
        self.prediction_worker = PredictionWorker(
            self.model_manager.predict,
            self.current_image.copy(),
            model_type=ModelType.PYMARAI
        )
        
        self.prediction_worker.progress.connect(self.on_prediction_progress)
        self.prediction_worker.completed.connect(self.on_prediction_complete)
        self.prediction_worker.error.connect(self.on_prediction_error)
        
        self.btn_predict.setEnabled(False)
        self.btn_batch.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        self.status_updated.emit("Running nnU-Net segmentation...")
        self.prediction_worker.start()
    
    def on_prediction_progress(self, message: str):
        """Handle prediction progress"""
        self.status_updated.emit(message)
    
    def on_prediction_complete(self, result: dict):
        """Handle prediction completion
        
        Args:
            result: Result dict with mask, metadata, etc.
        """
        self.current_mask = result['mask']
        self.current_metadata = result.get('metadata', {})
        
        # Update visualization
        self.image_canvas.set_overlay_mask(self.current_mask)
        self.update_visualization()
        
        # Show metrics
        metrics = result.get('metadata', {})
        metrics_str = ", ".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" 
                                for k, v in metrics.items()])
        self.status_updated.emit(f"✓ Segmentation complete. {metrics_str}")
        
        self.btn_predict.setEnabled(True)
        self.btn_batch.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.btn_export.setEnabled(True)
    
    def on_prediction_error(self, error_msg: str):
        """Handle prediction error"""
        logger.error(f"Prediction error: {error_msg}")
        self.status_updated.emit(f"✗ Error: {error_msg}")
        self.btn_predict.setEnabled(True)
        self.btn_batch.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Prediction Error", error_msg)
    
    def update_visualization(self):
        """Update mask visualization"""
        if self.current_mask is not None:
            color = self.viz_controls.get_color()
            self.image_canvas.set_visualization(color, self.viz_controls.show_contours())
    
    def export_results(self):
        """Export segmentation results"""
        if self.current_mask is None:
            QMessageBox.warning(self, "Warning", "No segmentation to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Segmentation",
            str(Path.home()),
            "PNG Image (*.png);;TIFF Image (*.tiff);;NumPy Array (*.npy)"
        )
        
        if file_path:
            try:
                save_mask(file_path, self.current_mask)
                if self.current_metadata:
                    metrics_path = Path(file_path).with_suffix(".metrics.csv")
                    save_metrics(str(metrics_path), self.current_metadata)
                
                self.status_updated.emit(f"✓ Exported to {Path(file_path).name}")
                
            except Exception as e:
                logger.error(f"Export failed: {e}")
                QMessageBox.critical(self, "Export Error", str(e))

    def run_batch_prediction(self):
        """Run spheroid segmentation on all images loaded in the browser."""
        image_paths = self.image_browser.get_all_files()
        if not image_paths:
            QMessageBox.warning(self, "Warning", "Please load an image folder first")
            return

        if not self.model_manager.is_model_loaded(ModelType.PYMARAI):
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Batch Output Directory",
            self.config.get("workspace_path")
        )
        if not output_dir:
            return

        self.batch_worker = BatchPredictionWorker(
            self.model_manager.predict,
            image_paths,
            output_dir,
            "pymarai_mask",
            model_type=ModelType.PYMARAI
        )
        self.batch_worker.progress.connect(self.on_prediction_progress)
        self.batch_worker.item_completed.connect(self.on_batch_item_complete)
        self.batch_worker.item_error.connect(self.on_batch_item_error)
        self.batch_worker.completed.connect(self.on_batch_complete)
        self.batch_worker.error.connect(self.on_prediction_error)

        self.btn_predict.setEnabled(False)
        self.btn_batch.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_updated.emit("Running folder batch...")
        self.batch_worker.start()

    def on_batch_item_complete(self, file_path: str, result: dict):
        """Update browser state after one batch item succeeds."""
        self.image_browser.update_item_status(file_path, "Done")

    def on_batch_item_error(self, file_path: str, error_msg: str):
        """Update browser state after one batch item fails."""
        self.image_browser.update_item_status(file_path, "Error")
        logger.error("Batch item failed for %s: %s", file_path, error_msg)

    def on_batch_complete(self, summary: dict):
        """Handle batch completion."""
        self.btn_predict.setEnabled(True)
        self.btn_batch.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.status_updated.emit(
            "✓ Batch complete. {succeeded}/{total} done, {failed} failed".format(**summary)
        )
    
    def update_status_label(self, message: str):
        """Update status label"""
        self.status_label.setText(message)
