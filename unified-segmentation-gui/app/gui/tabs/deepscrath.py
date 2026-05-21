#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepScratch segmentation tab
"""

import logging
from pathlib import Path
import numpy as np
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QSpinBox, QComboBox, QMessageBox, QProgressBar, QGroupBox
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
from app.utils.model_paths import DEFAULT_DEEPSCRATCH_MODEL_PATH

logger = logging.getLogger(__name__)


class DeepScratchTab(QWidget):
    """Tab for DeepScratch cell detection segmentation"""
    
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
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout()
        
        self.btn_load_model = QPushButton("Change Model")
        self.btn_load_model.setToolTip(
            "Choose a DeepScratch checkpoint. The app auto-loads models/deepscratch/model_best.pth.tar when present."
        )
        self.btn_load_model.clicked.connect(self.load_model)
        model_layout.addWidget(self.btn_load_model)
        
        model_layout.addWidget(QLabel("Task:"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Cell detection", "Semantic segmentation"])
        self.task_combo.currentTextChanged.connect(self.on_task_changed)
        self.task_combo.setToolTip(
            "Cell detection returns cell-center detections. Semantic segmentation returns class labels for every pixel."
        )
        model_layout.addWidget(self.task_combo)

        model_layout.addWidget(QLabel("Model Name:"))
        self.model_name_combo = QComboBox()
        self.model_name_combo.addItems([
            "unet-flat-48",
            "unet-flat-96",
            "unet-simple",
            "unet-wide",
            "unet-resnet18",
            "unet-densenet121",
            "dense-unet"
        ])
        self.model_name_combo.setToolTip(
            "Network architecture used by the checkpoint. This must match the architecture used during training."
        )
        model_layout.addWidget(self.model_name_combo)

        model_layout.addWidget(QLabel("Input Channels:"))
        self.input_channels_spinbox = QSpinBox()
        self.input_channels_spinbox.setRange(1, 4)
        self.input_channels_spinbox.setValue(1)
        self.input_channels_spinbox.setToolTip(
            "Number of input image channels expected by the checkpoint. Use 1 for grayscale, 3 for RGB."
        )
        model_layout.addWidget(self.input_channels_spinbox)

        model_layout.addWidget(QLabel("Output Classes:"))
        self.output_channels_spinbox = QSpinBox()
        self.output_channels_spinbox.setRange(1, 32)
        self.output_channels_spinbox.setValue(1)
        self.output_channels_spinbox.setToolTip(
            "Number of output maps/classes. Cell detection is usually 1; semantic segmentation is usually the number of classes."
        )
        model_layout.addWidget(self.output_channels_spinbox)

        model_layout.addWidget(QLabel("Loss Type:"))
        self.loss_combo = QComboBox()
        self.loss_combo.setEditable(True)
        self.loss_combo.addItems(["l2-G1.5", "l1smooth-G1.5", "crossentropy", "crossentropy+W1-1-10"])
        self.loss_combo.setToolTip(
            "Training loss identifier stored in the checkpoint setup. Detection checkpoints commonly use l2-G1.5; segmentation uses crossentropy."
        )
        model_layout.addWidget(self.loss_combo)

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
        
        # Segmentation controls
        seg_group = QGroupBox("Segmentation")
        seg_layout = QVBoxLayout()
        
        self.btn_predict = QPushButton("Run Segmentation")
        self.btn_predict.setToolTip("Run the loaded DeepScratch model on the selected image.")
        self.btn_predict.clicked.connect(self.run_prediction)
        self.btn_predict.setEnabled(False)
        seg_layout.addWidget(self.btn_predict)

        self.btn_batch = QPushButton("Run Folder Batch")
        self.btn_batch.setToolTip("Run the loaded DeepScratch model on every image listed in the folder browser.")
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
        self.btn_export.setToolTip("Save the current mask and a CSV file with summary metrics.")
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

    def on_task_changed(self, task_name: str):
        """Keep model defaults aligned with the selected DeepScratch task."""
        if task_name == "Semantic segmentation":
            self.output_channels_spinbox.setValue(3)
            index = self.loss_combo.findText("crossentropy")
        else:
            self.output_channels_spinbox.setValue(1)
            index = self.loss_combo.findText("l2-G1.5")
        if index >= 0:
            self.loss_combo.setCurrentIndex(index)

    def auto_load_model(self):
        """Load the configured DeepScratch checkpoint or the default bundled location."""
        model_config = self.config.get_model_config("deepscrath")
        model_path = model_config.get("model_path") or str(DEFAULT_DEEPSCRATCH_MODEL_PATH)
        if not model_path or not Path(model_path).is_file():
            self.model_status_label.setText(
                f"Model: no DeepScratch checkpoint found at {DEFAULT_DEEPSCRATCH_MODEL_PATH}"
            )
            self.status_updated.emit("DeepScratch model not auto-loaded: checkpoint missing")
            return False

        self._apply_model_config(model_config)
        return self._load_model_from_path(model_path, show_error=False)

    def _apply_model_config(self, model_config: dict):
        """Populate controls from saved model configuration."""
        task = model_config.get("task")
        if task:
            self.task_combo.setCurrentText("Semantic segmentation" if task == "segmentation" else "Cell detection")

        model_name = str(model_config.get("model_name", "")).replace("seg+", "")
        if model_name:
            index = self.model_name_combo.findText(model_name)
            if index >= 0:
                self.model_name_combo.setCurrentIndex(index)

        self.input_channels_spinbox.setValue(int(model_config.get("n_ch_in", self.input_channels_spinbox.value())))
        self.output_channels_spinbox.setValue(int(model_config.get("n_ch_out", self.output_channels_spinbox.value())))

        loss_type = model_config.get("loss_type")
        if loss_type:
            index = self.loss_combo.findText(loss_type)
            if index < 0:
                self.loss_combo.addItem(loss_type)
                index = self.loss_combo.findText(loss_type)
            self.loss_combo.setCurrentIndex(index)

    def _current_model_options(self) -> dict:
        """Return model-loading options from the current controls."""
        return {
            "model_name": self.model_name_combo.currentText(),
            "n_ch_in": self.input_channels_spinbox.value(),
            "n_ch_out": self.output_channels_spinbox.value(),
            "loss_type": self.loss_combo.currentText().strip(),
            "task": "segmentation" if self.task_combo.currentText() == "Semantic segmentation" else "detection",
        }

    def _load_model_from_path(self, file_path: str, show_error: bool = True) -> bool:
        """Load a DeepScratch model from a concrete path."""
        self.status_updated.emit("Loading DeepScratch model...")
        options = self._current_model_options()
        success = self.model_manager.load_model(ModelType.DEEPSCRATH, file_path, **options)

        if success:
            self.btn_predict.setEnabled(True)
            self.btn_batch.setEnabled(True)
            self.model_status_label.setText(f"Model: loaded {Path(file_path).name}")
            self.status_updated.emit("✓ DeepScratch model loaded")
            self.config.set_model_config('deepscrath', {'model_path': file_path, **options})
            self.config.save()
            return True

        self.btn_predict.setEnabled(False)
        self.btn_batch.setEnabled(False)
        self.model_status_label.setText("Model: failed to load")
        self.status_updated.emit("✗ Failed to load DeepScratch model")
        if show_error:
            QMessageBox.critical(self, "Error", "Failed to load DeepScratch model")
        return False
    
    def load_model(self):
        """Load DeepScratch model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Checkpoint",
            str(Path.home()),
            "PyTorch Checkpoint (*.pt *.pth *.checkpoint);;All Files (*)"
        )
        
        if file_path:
            self._load_model_from_path(file_path)
    
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
        
        if not self.model_manager.is_model_loaded(ModelType.DEEPSCRATH):
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return
        
        # Start prediction worker
        self.prediction_worker = PredictionWorker(
            self.model_manager.predict,
            self.current_image.copy(),
            model_type=ModelType.DEEPSCRATH
        )
        
        self.prediction_worker.progress.connect(self.on_prediction_progress)
        self.prediction_worker.completed.connect(self.on_prediction_complete)
        self.prediction_worker.error.connect(self.on_prediction_error)
        
        self.btn_predict.setEnabled(False)
        self.btn_batch.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        self.status_updated.emit("Running segmentation...")
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
        metrics_str = ", ".join([f"{k}: {v:.2f}" for k, v in metrics.items()])
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
            "PNG Image (*.png);;TIFF Image (*.tiff);;NumPy Array (*.npy);;CSV (*.csv)"
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
        """Run segmentation on all images loaded in the browser."""
        image_paths = self.image_browser.get_all_files()
        if not image_paths:
            QMessageBox.warning(self, "Warning", "Please load an image folder first")
            return

        if not self.model_manager.is_model_loaded(ModelType.DEEPSCRATH):
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
            "deepscratch_mask",
            model_type=ModelType.DEEPSCRATH
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
