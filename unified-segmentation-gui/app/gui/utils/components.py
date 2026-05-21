#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared GUI components for both segmentation models
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QTreeWidget, QTreeWidgetItem, QScrollArea, QComboBox, QSpinBox,
    QCheckBox, QSlider, QColorDialog
)
from PyQt5.QtCore import Qt, QSize, QRect, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QIcon, QColor, QFont
from PyQt5.QtCore import QTimer

from app.gui.utils.image_io import SUPPORTED_IMAGE_EXTENSIONS, is_supported_image

logger = logging.getLogger(__name__)


class ImageBrowser(QWidget):
    """File browser for image selection with preview"""
    
    image_selected = pyqtSignal(str)  # Emits file path
    
    def __init__(self):
        super().__init__()
        self.current_directory = Path.home()
        self.image_files = []
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI"""
        layout = QVBoxLayout()
        
        # File and directory browser buttons
        btn_layout = QHBoxLayout()
        self.btn_browse_image = QPushButton("Browse Image")
        self.btn_browse_image.clicked.connect(self.browse_image)
        btn_layout.addWidget(self.btn_browse_image)

        self.btn_browse = QPushButton("Browse Folder")
        self.btn_browse.clicked.connect(self.browse_folder)
        btn_layout.addWidget(self.btn_browse)
        layout.addLayout(btn_layout)
        
        # File tree
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["Filename", "Status"])
        self.file_tree.itemClicked.connect(self.on_file_selected)
        layout.addWidget(self.file_tree)
        
        self.setLayout(layout)

    def browse_image(self):
        """Open a single image and list the rest of its folder."""
        filters = "Images ({})".format(
            " ".join(f"*{ext}" for ext in sorted(SUPPORTED_IMAGE_EXTENSIONS))
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            str(self.current_directory),
            f"{filters};;All Files (*)"
        )
        if file_path:
            path = Path(file_path)
            self.load_folder(str(path.parent))
            self.select_file(str(path))
            self.image_selected.emit(str(path))
    
    def browse_folder(self):
        """Open folder browser dialog"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder",
            str(self.current_directory)
        )
        if folder:
            self.load_folder(folder)
    
    def load_folder(self, folder_path: str):
        """Load images from folder
        
        Args:
            folder_path: Path to folder containing images
        """
        self.current_directory = Path(folder_path)
        self.image_files = [
            path for path in self.current_directory.iterdir()
            if is_supported_image(path)
        ]
        
        self.file_tree.clear()
        for img_file in sorted(self.image_files):
            item = QTreeWidgetItem([img_file.name, "Pending"])
            item.setData(0, Qt.UserRole, str(img_file))
            self.file_tree.addTopLevelItem(item)
        self.file_tree.resizeColumnToContents(0)
    
    def on_file_selected(self, item: QTreeWidgetItem, column: int):
        """Handle file selection"""
        file_path = item.data(0, Qt.UserRole)
        if file_path:
            self.image_selected.emit(file_path)
    
    def update_item_status(self, file_path: str, status: str):
        """Update status of a file item
        
        Args:
            file_path: File path
            status: Status string ('Done', 'Error', etc.)
        """
        for i in range(self.file_tree.topLevelItemCount()):
            item = self.file_tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == file_path:
                item.setText(1, status)
                break

    def select_file(self, file_path: str):
        """Select a file in the tree if it is present."""
        for i in range(self.file_tree.topLevelItemCount()):
            item = self.file_tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == file_path:
                self.file_tree.setCurrentItem(item)
                break
    
    def get_selected_file(self) -> Optional[str]:
        """Get currently selected file"""
        item = self.file_tree.currentItem()
        if item:
            return item.data(0, Qt.UserRole)
        return None

    def get_all_files(self) -> list:
        """Get all loaded image file paths."""
        return [str(path) for path in sorted(self.image_files)]


class ImagePreviewCanvas(QLabel):
    """Interactive image preview with zoom and pan"""
    
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.segmentation_mask = None
        self.overlay_color = QColor(0, 255, 0, 128)  # Green
        self.show_contours = False
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        self.setMinimumSize(400, 400)
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.CrossCursor)
    
    def set_image(self, image: np.ndarray):
        """Set the main image to display
        
        Args:
            image: Image as numpy array
        """
        self.original_image = image
        self.segmentation_mask = None
        self.update_display()
    
    def set_overlay_mask(self, mask: Optional[np.ndarray]):
        """Set segmentation mask overlay
        
        Args:
            mask: Segmentation mask as numpy array
        """
        self.segmentation_mask = mask
        self.update_display()
    
    def set_overlay_color(self, color: QColor):
        """Set color for mask overlay
        
        Args:
            color: QColor object
        """
        self.overlay_color = color
        self.update_display()

    def set_visualization(self, color: QColor, show_contours: bool = False):
        """Set mask overlay color and contour mode."""
        self.overlay_color = color
        self.show_contours = show_contours
        self.update_display()
    
    def update_display(self):
        """Update the displayed image"""
        if self.original_image is None:
            return
        
        # Create display image in RGB order. PIL-loaded arrays are already RGB.
        display_img = self._to_uint8_rgb(self.original_image)
        
        # Add mask overlay if available
        if self.segmentation_mask is not None:
            display_img = self._overlay_mask(display_img, self.segmentation_mask)
        
        # Convert to QPixmap
        if display_img.dtype != np.uint8:
            display_img = np.clip(display_img * 255, 0, 255).astype(np.uint8)
        
        display_img = np.ascontiguousarray(display_img)
        h, w, _ = display_img.shape
        bytes_per_line = 3 * w
        qt_image = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        
        pixmap = QPixmap.fromImage(qt_image)
        
        # Apply zoom
        if self.zoom_level != 1.0:
            new_size = QSize(int(pixmap.width() * self.zoom_level),
                            int(pixmap.height() * self.zoom_level))
            pixmap = pixmap.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.setPixmap(pixmap)

    @staticmethod
    def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
        """Convert grayscale/RGB/RGBA input into uint8 RGB for preview."""
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        if image.dtype == np.uint8:
            return image.copy()

        image = image.astype(np.float32, copy=False)
        if image.size == 0:
            return image.astype(np.uint8)

        min_value = float(np.nanmin(image))
        max_value = float(np.nanmax(image))
        if max_value <= 1.0 and min_value >= 0.0:
            image = image * 255.0
        elif max_value > min_value:
            image = (image - min_value) / (max_value - min_value) * 255.0
        else:
            image = np.zeros_like(image)
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def _overlay_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Overlay segmentation mask on image
        
        Args:
            image: Original image
            mask: Segmentation mask
            
        Returns:
            Image with overlay
        """
        image = self._to_uint8_rgb(image)

        # Resize mask if needed
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Create overlay
        overlay = image.copy().astype(np.float32)
        
        # Get color components
        mask_colored = np.zeros_like(image, dtype=np.float32)
        mask_colored[:, :, 0] = self.overlay_color.red()
        mask_colored[:, :, 1] = self.overlay_color.green()
        mask_colored[:, :, 2] = self.overlay_color.blue()
        
        # Apply overlay where mask is positive
        alpha = self.overlay_color.alpha() / 255.0
        mask_region = mask > 0
        overlay[mask_region] = overlay[mask_region] * (1 - alpha) + mask_colored[mask_region] * alpha

        if self.show_contours and np.any(mask_region):
            contour_mask = (mask_region.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_color = (
                int(self.overlay_color.red()),
                int(self.overlay_color.green()),
                int(self.overlay_color.blue())
            )
            cv2.drawContours(overlay, contours, -1, contour_color, 2)
        
        return np.clip(overlay, 0, 255).astype(np.uint8)
    
    def zoom_in(self):
        """Zoom in"""
        self.zoom_level = min(self.zoom_level * 1.2, 20.0)
        self.update_display()
    
    def zoom_out(self):
        """Zoom out"""
        self.zoom_level = max(self.zoom_level / 1.2, 0.05)
        self.update_display()
    
    def reset_zoom(self):
        """Reset zoom to 1.0"""
        self.zoom_level = 1.0
        self.update_display()
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom"""
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()


class MaskVisualizationControls(QWidget):
    """Controls for mask visualization options"""
    
    mask_settings_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.color = QColor(0, 255, 0)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI"""
        layout = QHBoxLayout()
        
        # Alpha slider
        layout.addWidget(QLabel("Opacity:"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.setMaximumWidth(100)
        self.alpha_slider.valueChanged.connect(self.mask_settings_changed.emit)
        layout.addWidget(self.alpha_slider)
        
        # Color picker
        layout.addWidget(QLabel("Color:"))
        self.btn_color = QPushButton()
        self.btn_color.setMaximumWidth(50)
        self._update_color_button()
        self.btn_color.clicked.connect(self.pick_color)
        layout.addWidget(self.btn_color)
        
        # Show contours checkbox
        self.chk_contours = QCheckBox("Show Contours")
        self.chk_contours.toggled.connect(self.mask_settings_changed.emit)
        layout.addWidget(self.chk_contours)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def pick_color(self):
        """Open color picker dialog"""
        color = QColorDialog.getColor(self.color, self)
        if color.isValid():
            self.color = color
            self._update_color_button()
            self.mask_settings_changed.emit()
    
    def get_color(self) -> QColor:
        """Get selected color"""
        color = QColor(self.color)
        color.setAlpha(int(self.get_alpha() * 255))
        return color

    def _update_color_button(self):
        """Refresh color button stylesheet from the stored QColor."""
        self.btn_color.setStyleSheet(f"background-color: {self.color.name()};")
    
    def get_alpha(self) -> float:
        """Get opacity as 0-1"""
        return self.alpha_slider.value() / 100.0
    
    def show_contours(self) -> bool:
        """Check if contours should be shown"""
        return self.chk_contours.isChecked()


class ExportDialog(QWidget):
    """Export options selector"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout()
        
        # Format selection
        layout.addWidget(QLabel("Export Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "TIFF", "NPY", "CSV"])
        layout.addWidget(self.format_combo)
        
        # Export options
        self.chk_mask = QCheckBox("Export Segmentation Mask")
        self.chk_mask.setChecked(True)
        layout.addWidget(self.chk_mask)
        
        self.chk_metrics = QCheckBox("Export Metrics")
        self.chk_metrics.setChecked(True)
        layout.addWidget(self.chk_metrics)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def get_export_options(self) -> dict:
        """Get export options"""
        return {
            'format': self.format_combo.currentText(),
            'export_mask': self.chk_mask.isChecked(),
            'export_metrics': self.chk_metrics.isChecked(),
        }
