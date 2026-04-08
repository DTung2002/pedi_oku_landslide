import os
from typing import Dict, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QGroupBox,
    QSplitter,
    QComboBox,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QSizePolicy,
)

from pedi_oku_landslide.pipeline.runners.ui4_backend import (
    collect_ui4_run_inputs,
    list_ui4_preview_pngs,
    load_ui4_summary_for_run,
    render_ui4_contours_for_run,
    run_ui4_kriging_for_run,
    summary_range_for_kind,
)
from pedi_oku_landslide.ui.controllers.ui4_preview_controller import UI4PreviewControllerMixin
from pedi_oku_landslide.ui.controllers.ui4_run_controller import UI4RunControllerMixin


class ZoomableImageView(QGraphicsView):
    def __init__(self, parent=None):
        self._scene = QGraphicsScene()
        super().__init__(self._scene, parent)
        self._pix_item = None
        self._zoom_steps = 0
        self._loaded_path = ""
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(Qt.white)
        self.setStyleSheet("QGraphicsView { border: 1px solid #dddddd; border-radius: 8px; }")
        self.setMinimumHeight(420)

    def clear_image(self) -> None:
        self._scene.clear()
        self._pix_item = None
        self._zoom_steps = 0
        self._loaded_path = ""

    def load_image(self, path: str) -> bool:
        pm = QPixmap(path)
        if pm.isNull():
            self.clear_image()
            return False
        self._scene.clear()
        self._pix_item = QGraphicsPixmapItem(pm)
        self._scene.addItem(self._pix_item)
        self._scene.setSceneRect(self._pix_item.boundingRect())
        self._loaded_path = path
        self.fit_to_image()
        return True

    def fit_to_image(self) -> None:
        if self._pix_item is None:
            return
        self.resetTransform()
        rect = self._pix_item.boundingRect()
        if not rect.isNull():
            self.fitInView(rect, Qt.KeepAspectRatio)
        self._zoom_steps = 0

    def zoom_in(self) -> None:
        if self._pix_item is None:
            return
        self.scale(1.2, 1.2)
        self._zoom_steps += 1

    def zoom_out(self) -> None:
        if self._pix_item is None:
            return
        self.scale(1 / 1.2, 1 / 1.2)
        self._zoom_steps -= 1

    def wheelEvent(self, event):
        if self._pix_item is None:
            super().wheelEvent(event)
            return
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()


class UI4FrontendTab(UI4PreviewControllerMixin, UI4RunControllerMixin, QWidget):
    """
    Minimal UI4 connector tab.
    Purpose for now: receive run context from UI1/UI2/UI3 and show whether
    upstream files for UI4 kriging are ready.
    """

    def __init__(self, base_dir: str):
        super().__init__()
        self.base_dir = base_dir
        self._ctx: Dict[str, str] = {"project": "", "run_label": "", "run_dir": ""}
        self._last_info: Dict = {}
        self._preview_png_paths = []
        self._last_ui4_summary: Dict = {}
        self._splitter: Optional[QSplitter] = None
        self._left_min_w = 380
        self._left_default_w = 490
        self._pending_init_splitter = True
        self._ui_shown_once = False
        self._backend_collect_ui4_run_inputs = collect_ui4_run_inputs
        self._backend_list_ui4_preview_pngs = list_ui4_preview_pngs
        self._backend_load_ui4_summary = load_ui4_summary_for_run
        self._backend_summary_range = summary_range_for_kind
        self._backend_render_ui4_contours_for_run = render_ui4_contours_for_run
        self._backend_run_ui4_kriging_for_run = run_ui4_kriging_for_run

        self.lbl_project_value = QLabel("-")
        self.lbl_run_label_value = QLabel("-")
        self.lbl_input_status_value = QLabel("Not Ready")
        self.lbl_preview_status = QLabel("Preview: -")
        self.lbl_preview_status.hide()
        self.preview_file_combo = QComboBox()
        self.preview_file_combo.currentIndexChanged.connect(self._on_preview_file_changed)
        self.preview_view = ZoomableImageView()
        self.status_box = QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMaximumHeight(140)
        self.btn_refresh = QPushButton("Reload Inputs")
        self.btn_refresh.clicked.connect(self.refresh_from_context)
        self.btn_run_ui4 = QPushButton("Calculate Slip Surface")
        self.btn_run_ui4.clicked.connect(self._on_run_ui4)
        self.btn_make_contours = QPushButton("Preview")
        self.btn_make_contours.clicked.connect(self._on_generate_contours)
        self.btn_preview_fit = QPushButton("Fit")
        self.btn_preview_fit.clicked.connect(self.preview_view.fit_to_image)
        self.btn_preview_zoom_in = QPushButton("Zoom +")
        self.btn_preview_zoom_in.clicked.connect(self.preview_view.zoom_in)
        self.btn_preview_zoom_out = QPushButton("Zoom -")
        self.btn_preview_zoom_out.clicked.connect(self.preview_view.zoom_out)
        self.surface_auto_range = QCheckBox("Auto")
        self.surface_auto_range.setChecked(True)
        self.surface_auto_range.toggled.connect(self._on_surface_auto_range_toggled)
        self.surface_zmin = self._make_z_spin()
        self.surface_zmax = self._make_z_spin()
        self.surface_step = self._make_step_spin(1.0)
        self.depth_auto_range = QCheckBox("Auto")
        self.depth_auto_range.setChecked(True)
        self.depth_auto_range.toggled.connect(self._on_depth_auto_range_toggled)
        self.depth_zmin = self._make_z_spin()
        self.depth_zmax = self._make_z_spin()
        self.depth_step = self._make_step_spin(1.0)

        self._build_ui()
        self._update_contour_range_controls()
        self._apply_ui4_panel_spacing()

    def _apply_ui4_panel_spacing(self) -> None:
        # Make panel rows more readable: taller controls and more spacing.
        ctrl_h = 32
        for w in (
            self.preview_file_combo,
            self.surface_zmin,
            self.surface_zmax,
            self.surface_step,
            self.depth_zmin,
            self.depth_zmax,
            self.depth_step,
            self.btn_refresh,
            self.btn_run_ui4,
            self.btn_make_contours,
            self.btn_preview_zoom_out,
            self.btn_preview_zoom_in,
            self.btn_preview_fit,
        ):
            w.setMinimumHeight(ctrl_h)
        self.surface_auto_range.setMinimumHeight(ctrl_h)
        self.depth_auto_range.setMinimumHeight(ctrl_h)
        for lbl in (
            self.lbl_project_value,
            self.lbl_run_label_value,
            self.lbl_input_status_value,
        ):
            lbl.setMinimumHeight(28)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self._splitter = splitter
        splitter.splitterMoved.connect(lambda *_: self._enforce_left_pane_bounds())

        # Left pane (controls / status / info) ~30%
        left_host = QWidget()
        left_lay = QVBoxLayout(left_host)
        left_lay.setContentsMargins(6, 6, 6, 6)
        left_lay.setSpacing(8)

        box_info = QGroupBox("Project")
        info_lay = QGridLayout(box_info)
        info_lay.setContentsMargins(12, 14, 12, 12)
        info_lay.setHorizontalSpacing(12)
        info_lay.setVerticalSpacing(10)
        info_lay.addWidget(QLabel("Project"), 0, 0)
        info_lay.addWidget(self.lbl_project_value, 0, 1)
        info_lay.addWidget(QLabel("Run label"), 1, 0)
        info_lay.addWidget(self.lbl_run_label_value, 1, 1)
        info_lay.addWidget(QLabel("Input Status"), 2, 0)
        info_lay.addWidget(self.lbl_input_status_value, 2, 1)
        info_lay.setColumnStretch(1, 1)
        for r in range(3):
            info_lay.setRowMinimumHeight(r, 30)
        left_lay.addWidget(box_info)

        box_display = QGroupBox("Contour Lines Display")
        display_lay = QVBoxLayout(box_display)
        display_lay.setContentsMargins(12, 14, 12, 12)
        display_lay.setSpacing(10)

        cfg_lay = QGridLayout()
        cfg_lay.setHorizontalSpacing(10)
        cfg_lay.setVerticalSpacing(10)
        cfg_lay.addWidget(QLabel("Type"), 0, 0)
        cfg_lay.addWidget(QLabel("Auto Range"), 0, 1)
        cfg_lay.addWidget(QLabel("Z min (m)"), 0, 2)
        cfg_lay.addWidget(QLabel("Z max (m)"), 0, 3)
        cfg_lay.addWidget(QLabel("Step (m)"), 0, 4)
        cfg_lay.addWidget(QLabel("Surface"), 1, 0)
        cfg_lay.addWidget(self.surface_auto_range, 1, 1)
        cfg_lay.addWidget(self.surface_zmin, 1, 2)
        cfg_lay.addWidget(self.surface_zmax, 1, 3)
        cfg_lay.addWidget(self.surface_step, 1, 4)
        cfg_lay.addWidget(QLabel("Depth"), 2, 0)
        cfg_lay.addWidget(self.depth_auto_range, 2, 1)
        cfg_lay.addWidget(self.depth_zmin, 2, 2)
        cfg_lay.addWidget(self.depth_zmax, 2, 3)
        cfg_lay.addWidget(self.depth_step, 2, 4)
        for r in range(3):
            cfg_lay.setRowMinimumHeight(r, 32)
        display_lay.addLayout(cfg_lay)
        preview_ctl = QHBoxLayout()
        preview_ctl.setSpacing(10)
        preview_ctl.addWidget(QLabel("File"))
        preview_ctl.addWidget(self.preview_file_combo, 1)
        display_lay.addLayout(preview_ctl)
        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        for b in (self.btn_refresh, self.btn_run_ui4, self.btn_make_contours):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            action_row.addWidget(b, 1)
        display_lay.addLayout(action_row)

        left_lay.addWidget(box_display)

        box_status = QGroupBox("Status")
        box_status_lay = QVBoxLayout(box_status)
        box_status_lay.setContentsMargins(12, 14, 12, 12)
        box_status_lay.addWidget(self.status_box)
        left_lay.addWidget(box_status)
        left_lay.addStretch(1)

        # Right pane (image only) ~70%
        right_host = QWidget()
        right_lay = QVBoxLayout(right_host)
        right_lay.setContentsMargins(6, 6, 6, 6)
        right_lay.setSpacing(8)
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(10)
        zoom_row.addWidget(self.btn_preview_zoom_out)
        zoom_row.addWidget(self.btn_preview_zoom_in)
        zoom_row.addWidget(self.btn_preview_fit)
        zoom_row.addStretch(1)
        right_lay.addLayout(zoom_row)
        box_image = QGroupBox("Contour Lines Preview Image")
        box_image_lay = QVBoxLayout(box_image)
        box_image_lay.addWidget(self.preview_view, 1)
        right_lay.addWidget(box_image, 1)

        splitter.addWidget(left_host)
        splitter.addWidget(right_host)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        # Match UI3 left pane default width (_left_default_w ~= 490 px).
        splitter.setSizes([self._left_default_w, 900])
        left_host.setMinimumWidth(self._left_min_w)
        right_host.setMinimumWidth(500)

        root.addWidget(splitter, 1)

    def _append(self, msg: str) -> None:
        self.status_box.append(msg)

    def _left_max_w(self) -> int:
        # Left pane can occupy at most 50% of current splitter/window width.
        base_w = self.width()
        if self._splitter is not None and self._splitter.width() > 0:
            base_w = self._splitter.width()
        if base_w < (self._left_min_w * 2):
            return -1
        return max(self._left_min_w, int(base_w * 0.5))

    def _try_apply_initial_splitter_width(self) -> None:
        if not self._pending_init_splitter or self._splitter is None:
            return
        # Defer until layout settles at a width that can actually host the
        # requested default (490 px) while respecting max = 1/2 window.
        base_w = self._splitter.width() if self._splitter.width() > 0 else self.width()
        if base_w < (self._left_default_w * 2):
            return
        max_w = self._left_max_w()
        if max_w < 0:
            return
        init_left = max(self._left_min_w, min(self._left_default_w, max_w))
        total = max(self._splitter.width(), sum(self._splitter.sizes()), self.width(), init_left + 1)
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

    def _make_z_spin(self) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(-1_000_000.0, 1_000_000.0)
        s.setDecimals(3)
        s.setSingleStep(1.0)
        s.setKeyboardTracking(False)
        s.setValue(0.0)
        return s

    def _make_step_spin(self, default: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(0.01, 1_000_000.0)
        s.setDecimals(3)
        s.setSingleStep(0.5)
        s.setKeyboardTracking(False)
        s.setValue(float(default))
        return s

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._pending_init_splitter:
            return
        self._enforce_left_pane_bounds()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._ui_shown_once = True
        if self._pending_init_splitter:
            # Apply after Qt finishes initial layout passes; early setSizes()
            # calls can be overwritten and appear as a 50/50 split.
            QTimer.singleShot(0, self._enforce_left_pane_bounds)
            QTimer.singleShot(80, self._enforce_left_pane_bounds)
            QTimer.singleShot(180, self._enforce_left_pane_bounds)
            return
        self._enforce_left_pane_bounds()

    def _update_contour_range_controls(self, *_args) -> None:
        surf_manual = not self.surface_auto_range.isChecked()
        self.surface_zmin.setEnabled(surf_manual)
        self.surface_zmax.setEnabled(surf_manual)
        depth_manual = not self.depth_auto_range.isChecked()
        self.depth_zmin.setEnabled(depth_manual)
        self.depth_zmax.setEnabled(depth_manual)

    def _on_surface_auto_range_toggled(self, checked: bool) -> None:
        if not checked:
            self._populate_manual_range_from_summary("surface")
        self._update_contour_range_controls()

    def _on_depth_auto_range_toggled(self, checked: bool) -> None:
        if not checked:
            self._populate_manual_range_from_summary("depth")
        self._update_contour_range_controls()
