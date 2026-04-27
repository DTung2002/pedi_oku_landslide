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


class UI3LayoutMixin:
    def _left_max_w(self) -> int:
        base_w = self.width()
        if self._splitter is not None and self._splitter.width() > 0:
            base_w = self._splitter.width()
        if base_w < (self._left_min_w * 2):
            return -1
        return max(self._left_min_w, int(base_w * 0.5))

    def _try_apply_initial_splitter_width(self) -> None:
        if not self._pending_init_splitter or self._splitter is None:
            return
        max_w = self._left_max_w()
        if max_w < 0:
            return
        init_left = max(self._left_min_w, min(self._left_default_w, max_w))
        total = sum(self._splitter.sizes())
        if total <= 0:
            total = max(self._splitter.width(), self.width(), init_left + 1)
        self._splitter.setSizes([init_left, max(1, total - init_left)])
        self._pending_init_splitter = False

    def _enforce_left_pane_bounds(self) -> None:
        if self._splitter is None:
            return
        self._try_apply_initial_splitter_width()
        sizes = self._splitter.sizes()
        if len(sizes) != 2:
            return
        left_w, right_w = sizes
        total = left_w + right_w
        max_w = self._left_max_w()
        if max_w < 0:
            return
        clamped_left = max(self._left_min_w, min(left_w, max_w))
        if clamped_left != left_w and total > 0:
            self._splitter.setSizes([clamped_left, max(1, total - clamped_left)])

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._enforce_left_pane_bounds()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._enforce_left_pane_bounds()
