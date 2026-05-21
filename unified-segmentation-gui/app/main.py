#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point for Unified Segmentation GUI
"""

import sys
import logging
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "unified_segmentation_gui" / "matplotlib")
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

LOG_HANDLERS = [logging.StreamHandler(sys.stdout)]
try:
    LOG_DIR = Path.home() / ".unified_segmentation_gui"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_HANDLERS.insert(0, logging.FileHandler(LOG_DIR / "app.log"))
except OSError:
    # Some sandboxed runs cannot write to the user's home directory.
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=LOG_HANDLERS
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    try:
        from PyQt5.QtWidgets import QApplication
        from app.gui.main_window import MainWindow
        
        logger.info("Starting Unified Segmentation GUI")
        
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        
        logger.info("GUI window displayed")
        sys.exit(app.exec_())
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"Error: Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
