import math
import os
from typing import Any, Dict, List, Optional
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import (
    QAction,
    QAbstractSpinBox,
    QCheckBox,
    QFrame,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from pedi_oku_landslide.ui.scenes.ui3_preview_scene import ZoomableGraphicsView
from pedi_oku_landslide.ui.controllers.ui3_line_controller import WORKFLOW_GROUPING_PARAMS
from pedi_oku_landslide.ui.widgets.ui3_widgets import KeyboardOnlyDoubleSpinBox, KeyboardOnlySpinBox, NoWheelComboBox
from pedi_oku_landslide.ui.layout_constants import (
    LEFT_MARGINS,
    PANEL_SPACING,
    PREVIEW_FIT_BUTTON_H,
    PREVIEW_FIT_BUTTON_W,
    PREVIEW_MIN_H,
    PREVIEW_VIEWPORT_STYLE,
    RIGHT_MARGINS,
    RIGHT_MIN_W,
    STATUS_PANEL_H,
    CONTROL_HEIGHT,
    PROJECT_H_SPACING,
    PROJECT_LABEL_W,
    PROJECT_MARGINS,
    PROJECT_V_SPACING,
    ROOT_MARGINS,
    ROOT_SPACING,
)


class UI3StatusMixin:
    @staticmethod
    def _status_brief(msg: str, fallback: str) -> str:
        skip_prefixes = (
            "project:",
            "run:",
            "output:",
            "folder:",
            "dem:",
            "dx:",
            "dy:",
            "dz:",
            "mask:",
        )
        for raw in str(msg or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("[ui3] "):
                line = line[6:].strip()
                low = line.lower()
            if any(low.startswith(prefix) for prefix in skip_prefixes):
                continue
            if "\\" in line or "/" in line:
                if ":" in line:
                    line = line.split(":", 1)[0].strip()
                else:
                    continue
            return line
        return fallback

    def _append_status(self, text: str) -> None:
        self.status.append(text)

    def _ok(self, msg: str) -> None:
        self._append_status(f"OK: {self._status_brief(msg, 'Completed.')}")

    def _info(self, msg: str) -> None:
        return

    def _warn(self, msg: str) -> None:
        self._append_status(f"ERROR: {self._status_brief(msg, 'Action required.')}")

    def _err(self, msg: str) -> None:
        self._append_status(f"ERROR: {self._status_brief(msg, 'Error.')}")

    def _log(self, msg: str) -> None:
        return
