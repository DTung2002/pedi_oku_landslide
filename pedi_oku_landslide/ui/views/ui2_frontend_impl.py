# pedi_oku_landslide/ui/views/ui2_frontend_impl.py
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from shapely.geometry import LineString, Point
from PyQt5.QtCore import QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsSimpleTextItem,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pedi_oku_landslide.pipeline.runners.ui2_backend import (
    UI2BackendService,
)
from pedi_oku_landslide.pipeline.runners.ui2.ui2_intersections import (
    line_order_key as backend_line_order_key,
    normalize_line_role as backend_normalize_line_role,
    parse_auto_line_id as backend_parse_auto_line_id,
    pick_intersection_point as backend_pick_intersection_point,
)
from pedi_oku_landslide.pipeline.runners.ui2.ui2_paths import file_exists as _file_exists
from pedi_oku_landslide.pipeline.runners.ui2.ui2_sections_storage import (
    canonical_section_csv_row as storage_canonical_section_csv_row,
    reverse_section_points as storage_reverse_section_points,
)
from pedi_oku_landslide.ui.dialogs.ui2_dialogs import AutoLineDialog
from .ui2_layered_viewer import _LayeredViewer
from pedi_oku_landslide.ui.widgets.ui2_widgets import HBox, NoWheelComboBox as _NoWheelComboBox
from pedi_oku_landslide.ui.layout_constants import (
    LEFT_DEFAULT_W,
    LEFT_MARGINS,
    LEFT_MIN_W,
    PANEL_SPACING,
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

# ---------- UI2: Section Selection tab ----------

class SectionSelectionTab(QWidget):
    sections_confirmed = pyqtSignal(str, str, str)  # project, run_label, run_dir
    def __init__(self, base_dir: str, parent=None) -> None:
        super().__init__(parent)
        self.base_dir = base_dir
        self._backend = UI2BackendService()
        self._splitter: Optional[QSplitter] = None
        self._left_min_w = LEFT_MIN_W
        self._left_default_w = LEFT_DEFAULT_W
        self._pending_init_splitter = True

        # run context
        self._ctx_ready: bool = False
        self._ctx_project: str = ""
        self._ctx_runlabel: str = ""
        self._ctx_run_dir: str = ""
        self._ui1_dir: str = ""
        self._ui2_dir: str = ""

        # viewer items list init
        self._grid_items = []
        self._vec_items = []
        self._hill_item = None
        self._heat_item = None

        self.project: Optional[str] = None
        self.run_label: Optional[str] = None
        self.run_dir: Optional[str] = None
        self._section_lines: list[object] = []
        self._section_line_labels: list[QGraphicsSimpleTextItem] = []
        self._poly_section_lines: list[object] = []
        self._poly_section_line_labels: list[QGraphicsSimpleTextItem] = []
        self._preview_line: Optional[object] = None
        self._preview_label: Optional[QGraphicsSimpleTextItem] = None
        self._preview_source: Optional[str] = None

        # caches
        self._tr: Optional[Affine] = None
        self._inv_tr: Optional[Affine] = None
        self._dz: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._dy: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None  # uint8 0/1 aligned to dz grid
        self._dem_path: Optional[str] = None

        # vector drawing params (UI2-local, độc lập UI1)
        self._vec_step: int = 25
        self._vec_scale: float = 1.0
        self._vec_size_pct: int = 100
        self._vec_opacity_pct: int = 100
        self._vec_color: str = "blue"
        self._vec_pen_base: int = 1
        self._vec_arrow_base: float = 12.0

        self._sections: List[Any] = []
        self._section_meta: List[Dict[str, Any]] = []
        self._poly_sections: List[List[Tuple[float, float]]] = []
        self._poly_section_meta: List[Dict[str, Any]] = []

        self._updating_table: bool = False

        self._build_ui()
        self._wire()

    # ---- public API ----
    # ---- public API ----
    def set_context(
            self,
            project: str,
            run_label: str,
            run_dir: str,
            vec_step: Optional[int] = None,
            vec_scale: Optional[float] = None,
    ) -> None:
        # lưu vào “ctx_*” cho nội bộ
        self._ctx_project = project or ""
        self._ctx_runlabel = run_label or ""
        self._ctx_run_dir = run_dir or ""
        self._ui1_dir = os.path.join(run_dir, "ui1")
        self._ui2_dir = os.path.join(run_dir, "ui2")

        # UI2 dùng bộ thông số Vector Display riêng; không sync từ UI1.
        _ = vec_step
        _ = vec_scale

        # ✨ lưu vào các thuộc tính public để các hàm khác đọc
        self.project = self._ctx_project
        self.run_label = self._ctx_runlabel
        self.run_dir = self._ctx_run_dir

        backend_ctx = self._backend.set_context(project, run_label, run_dir, base_dir=self.base_dir)
        self._ui1_dir = str(backend_ctx.get("ui1_dir", "") or self._ui1_dir)
        self._ui2_dir = str(backend_ctx.get("ui2_dir", "") or self._ui2_dir)
        self._ctx_ready = bool(backend_ctx.get("ready", False))
        missing = list(backend_ctx.get("missing", []) or [])

        # cập nhật text vào ô hiển thị (read-only)
        if hasattr(self, "edit_project"):
            self.edit_project.setText(self.project)
            self.edit_project.setReadOnly(True)
        if hasattr(self, "edit_runlabel"):
            self.edit_runlabel.setText(self.run_label)
            self.edit_runlabel.setReadOnly(True)

        if not self._ctx_ready:
            self._clear_sections_state()
            missing_txt = ", ".join(str(p) for p in missing if p) or "required UI2 inputs"
            self._err(f"[UI2] Context incomplete. Missing files: {missing_txt}")
            return

        # nạp layer
        try:
            self._load_layers_and_show()
        except Exception as e:
            self._ctx_ready = False
            self._clear_sections_state()
            self._err(f"[UI2] Cannot load run context: {e}")
            return
        self._ok("[UI2] Context set OK.")

    # ---- UI ----
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(*ROOT_MARGINS)
        root.setSpacing(ROOT_SPACING)
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        self._splitter = splitter
        splitter.splitterMoved.connect(lambda *_: self._enforce_left_pane_bounds())
        root.addWidget(splitter)
        self.viewer = _LayeredViewer(self)

        # left pane
        left = QWidget()
        left.setMinimumWidth(self._left_min_w)
        left_shell = QVBoxLayout(left)
        left_shell.setContentsMargins(0, 0, 0, 0)
        left_shell.setSpacing(0)
        left_scroll = QScrollArea(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFrameShape(left_scroll.NoFrame)
        left_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_container = QWidget()
        left_lo = QVBoxLayout(left_container)
        left_lo.setContentsMargins(*LEFT_MARGINS)
        left_lo.setSpacing(PANEL_SPACING)
        left_scroll.setWidget(left_container)
        left_shell.addWidget(left_scroll)

        grp_proj = QGroupBox("Project")
        gl = QGridLayout(grp_proj)
        gl.setContentsMargins(*PROJECT_MARGINS)
        gl.setHorizontalSpacing(PROJECT_H_SPACING)
        gl.setVerticalSpacing(PROJECT_V_SPACING)
        gl.setColumnStretch(1, 1)
        lbl_name = QLabel("Name:")
        lbl_run = QLabel("Run label:")
        lbl_name.setFixedWidth(PROJECT_LABEL_W)
        lbl_run.setFixedWidth(PROJECT_LABEL_W)
        gl.setColumnMinimumWidth(0, PROJECT_LABEL_W)
        self.edit_project = QLineEdit()
        self.edit_project.setPlaceholderText("—")
        self.edit_project.setFixedHeight(CONTROL_HEIGHT)
        self.edit_project.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.edit_runlabel = QLineEdit()
        self.edit_runlabel.setPlaceholderText("—")
        self.edit_runlabel.setFixedHeight(CONTROL_HEIGHT)
        self.edit_runlabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        gl.addWidget(lbl_name, 0, 0)
        gl.addWidget(self.edit_project, 0, 1)
        gl.addWidget(lbl_run, 1, 0)
        gl.addWidget(self.edit_runlabel, 1, 1)
        left_lo.addWidget(grp_proj)

        grp_layers = QGroupBox("Layers")
        layers_layout = QVBoxLayout(grp_layers)
        layers_display_header = QHBoxLayout()
        layers_display_header.setContentsMargins(0, 0, 0, 0)
        layers_display_header.setSpacing(6)
        self.btn_layers_display = QPushButton("Display ˅")
        self.btn_layers_display.setCheckable(True)
        self.btn_layers_display.setChecked(False)
        self.btn_layers_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layers_display_header.addWidget(self.btn_layers_display, 1)
        layers_layout.addLayout(layers_display_header)

        self.layers_display_panel = QWidget()
        ll = QGridLayout(self.layers_display_panel)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setHorizontalSpacing(6)
        ll.setVerticalSpacing(6)
        ll.setColumnStretch(1, 1)
        self.layers_display_panel.setVisible(False)
        self.btn_layers_display.toggled.connect(
            lambda checked: (
                self.layers_display_panel.setVisible(checked),
                self.btn_layers_display.setText("Display ˄" if checked else "Display ˅"),
            )
        )

        lbl_gf = QLabel("Grid font size:")
        lbl_hs = QLabel("Hillshade opacity:")
        lbl_hm = QLabel("Heatmap opacity:")
        ll.setColumnMinimumWidth(0, max(
            lbl_gf.sizeHint().width(),
            lbl_hs.sizeHint().width(),
            lbl_hm.sizeHint().width(),
        ))

        self.sld_grid_font = QSlider(Qt.Horizontal); self.sld_grid_font.setRange(8, 72); self.sld_grid_font.setValue(12)
        self.sld_hill = QSlider(Qt.Horizontal); self.sld_hill.setRange(0, 100); self.sld_hill.setValue(100)
        self.sld_heat = QSlider(Qt.Horizontal); self.sld_heat.setRange(0, 100); self.sld_heat.setValue(75)

        ll.addWidget(lbl_gf, 0, 0); ll.addWidget(self.sld_grid_font, 0, 1)
        ll.addWidget(lbl_hs, 1, 0); ll.addWidget(self.sld_hill, 1, 1)
        ll.addWidget(lbl_hm, 2, 0); ll.addWidget(self.sld_heat, 2, 1)
        layers_layout.addWidget(self.layers_display_panel)
        left_lo.addWidget(grp_layers)

        grp_vec = QGroupBox("Vector Display"); vvl = QVBoxLayout(grp_vec)
        vec_actions = QHBoxLayout()
        vec_actions.setContentsMargins(0, 0, 0, 0)
        vec_actions.setSpacing(6)
        self.btn_vec_options = QPushButton("Display ˅")
        self.btn_vec_options.setCheckable(True)
        self.btn_vec_options.setChecked(False)
        self.btn_vec_options.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_render_vectors = QPushButton("Render Vectors")
        self.btn_render_vectors.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        vec_actions.addWidget(self.btn_vec_options, 1)
        vec_actions.addWidget(self.btn_render_vectors, 1)
        vvl.addLayout(vec_actions)

        self.vec_options_panel = QWidget()
        vec_options_layout = QVBoxLayout(self.vec_options_panel)
        vec_options_layout.setContentsMargins(0, 0, 0, 0)
        vec_options_layout.setSpacing(6)
        self.vec_options_panel.setVisible(False)
        self.btn_vec_options.toggled.connect(
            lambda checked: (
                self.vec_options_panel.setVisible(checked),
                self.btn_vec_options.setText("Display ˄" if checked else "Display ˅"),
            )
        )

        grid_vec_top = QGridLayout()
        grid_vec_top.setHorizontalSpacing(8)
        grid_vec_top.setVerticalSpacing(6)
        grid_vec_top.setColumnStretch(1, 1)

        lbl_step = QLabel("Step:")
        lbl_scale = QLabel("Scale:")
        lbl_color = QLabel("Color:")
        lbl_size = QLabel("Size:")
        lbl_opacity = QLabel("Opacity:")

        self.spin_vec_step = QSpinBox()
        self.spin_vec_step.setRange(1, 200)
        self.spin_vec_step.setValue(self._vec_step)
        self.spin_vec_step.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.spin_vec_scale = QDoubleSpinBox()
        self.spin_vec_scale.setRange(0.01, 10.0)
        self.spin_vec_scale.setSingleStep(1.0)
        self.spin_vec_scale.setValue(self._vec_scale)
        self.spin_vec_scale.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.combo_vec_color = QComboBox()
        self.combo_vec_color.addItems(["Blue", "Red", "Green", "White", "Yellow", "Magenta"])
        self.combo_vec_color.setCurrentText("Blue")
        self.combo_vec_color.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.sld_vec_size = QSlider(Qt.Horizontal)
        self.sld_vec_size.setRange(80, 500)
        self.sld_vec_size.setValue(self._vec_size_pct)
        self.sld_vec_size.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.sld_vec_opacity = QSlider(Qt.Horizontal)
        self.sld_vec_opacity.setRange(0, 100)
        self.sld_vec_opacity.setValue(self._vec_opacity_pct)
        self.sld_vec_opacity.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        label_col_w = max(
            lbl_step.sizeHint().width(),
            lbl_scale.sizeHint().width(),
            lbl_color.sizeHint().width(),
        )
        grid_vec_top.setColumnMinimumWidth(0, label_col_w)
        input_min_w = max(
            self.spin_vec_step.sizeHint().width(),
            self.spin_vec_scale.sizeHint().width(),
            self.combo_vec_color.sizeHint().width(),
        )
        for w in (self.spin_vec_step, self.spin_vec_scale, self.combo_vec_color):
            w.setMinimumWidth(input_min_w)

        grid_vec_top.addWidget(lbl_step, 0, 0)
        grid_vec_top.addWidget(self.spin_vec_step, 0, 1)
        grid_vec_top.addWidget(lbl_scale, 1, 0)
        grid_vec_top.addWidget(self.spin_vec_scale, 1, 1)
        grid_vec_top.addWidget(lbl_color, 2, 0)
        grid_vec_top.addWidget(self.combo_vec_color, 2, 1)
        vec_options_layout.addLayout(grid_vec_top)

        grid_vec_sliders = QGridLayout()
        grid_vec_sliders.setHorizontalSpacing(8)
        grid_vec_sliders.setVerticalSpacing(6)
        grid_vec_sliders.setColumnStretch(1, 1)
        grid_vec_sliders.addWidget(lbl_size, 0, 0)
        grid_vec_sliders.addWidget(self.sld_vec_size, 0, 1)
        grid_vec_sliders.addWidget(lbl_opacity, 1, 0)
        grid_vec_sliders.addWidget(self.sld_vec_opacity, 1, 1)
        vec_options_layout.addLayout(grid_vec_sliders)
        vvl.addWidget(self.vec_options_panel)

        left_lo.addWidget(grp_vec)

        grp_secs = QGroupBox("Sections");
        sl = QVBoxLayout(grp_secs)
        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["ID", "Start (x,y)", "End (x,y)", "Role"])
        self.tbl.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl.viewport().setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl.verticalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl.verticalHeader().setVisible(False)
        hdr = self.tbl.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, hdr.Fixed)  # cột #
        hdr.setSectionResizeMode(1, hdr.Stretch)  # Start
        hdr.setSectionResizeMode(2, hdr.Stretch)  # End
        hdr.setSectionResizeMode(3, hdr.Fixed)    # Role
        self.tbl.setColumnWidth(0, 56)
        self.tbl.setColumnWidth(3, 100)
        self.tbl.setMinimumHeight(220)
        self.tbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.tbl.itemChanged.connect(self._on_table_item_changed)
        self.lbl_straight_sections = QLabel("Straight Sections")
        sl.addWidget(self.lbl_straight_sections)
        sl.addWidget(self.tbl)

        self.tbl_poly = QTableWidget(0, 6)
        self.tbl_poly.setHorizontalHeaderLabels(["ID", "Start (x,y)", "End (x,y)", "Points", "Vertices", "Role"])
        self.tbl_poly.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_poly.viewport().setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_poly.verticalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_poly.verticalHeader().setVisible(False)
        hdr_poly = self.tbl_poly.horizontalHeader()
        hdr_poly.setStretchLastSection(False)
        hdr_poly.setSectionResizeMode(0, hdr_poly.Fixed)
        hdr_poly.setSectionResizeMode(1, hdr_poly.Stretch)
        hdr_poly.setSectionResizeMode(2, hdr_poly.Stretch)
        hdr_poly.setSectionResizeMode(3, hdr_poly.Fixed)
        hdr_poly.setSectionResizeMode(4, hdr_poly.Stretch)
        hdr_poly.setSectionResizeMode(5, hdr_poly.Fixed)
        self.tbl_poly.setColumnWidth(0, 56)
        self.tbl_poly.setColumnWidth(3, 70)
        self.tbl_poly.setColumnWidth(5, 100)
        self.tbl_poly.setMinimumHeight(220)
        self.tbl_poly.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.tbl_poly.itemChanged.connect(self._on_polyline_table_item_changed)
        self.lbl_polyline_sections = QLabel("Polyline Sections")
        sl.addWidget(self.lbl_polyline_sections)
        sl.addWidget(self.tbl_poly)

        # Một hàng nút thao tác Section: mode chọn + Auto Line, Preview, Clear All, Confirm
        row_actions = HBox()
        self.combo_draw_mode = QComboBox()
        self.combo_draw_mode.addItem("Straight Section", "straight")
        self.combo_draw_mode.addItem("Polyline Section", "polyline")
        self.btn_auto = QPushButton("Auto Line")
        self.btn_prev = QPushButton("Draw")
        self.btn_clear = QPushButton("Clear All")
        self.btn_confirm = QPushButton("Confirm")

        row_actions.addWidget(self.combo_draw_mode, 1)
        for b in (self.btn_auto, self.btn_prev, self.btn_clear, self.btn_confirm):
            # stretch=1 → 4 nút chia đều chiều ngang
            row_actions.addWidget(b, 1)
        sl.addLayout(row_actions)
        self._update_section_table_visibility()

        left_lo.addWidget(grp_secs)

        grp_status = QGroupBox("Status"); sv = QVBoxLayout(grp_status)
        self.lbl_cursor = QLabel("Cursor: —")
        self.lbl_cursor.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        from PyQt5.QtWidgets import QPlainTextEdit
        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.status_text.setMaximumBlockCount(2000)  # tránh phình bộ nhớ khi log dài
        self.status_text.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")
        self.status_text.setFixedHeight(200)
        self.status_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sv.addWidget(self.status_text)
        left_lo.addWidget(grp_status)

        left_lo.addStretch(1)
        # right pane: viewer
        right = QWidget()
        right.setMinimumWidth(RIGHT_MIN_W)
        right_lo = QVBoxLayout(right)
        right_lo.setContentsMargins(*RIGHT_MARGINS)
        right_lo.setSpacing(PANEL_SPACING)
        cursor_row = QHBoxLayout()
        cursor_row.setContentsMargins(0, 0, 0, 0)
        cursor_row.addWidget(self.lbl_cursor, 1)
        cursor_row.addWidget(self.viewer.btn_zoom_fit, 0)
        right_lo.addLayout(cursor_row)
        right_lo.addWidget(self.viewer, 1)

        splitter.addWidget(left)  # khung trái (form, table, status)
        splitter.addWidget(right)  # khung phải (map)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([self._left_default_w, 700])

        # Áp style nút
        self._apply_button_style()

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


    def _apply_button_style(self) -> None:
        """Áp style tab Section Selection (nền trắng + nút xanh)."""
        style = """
        QWidget {
            background-color: #ffffff;
        }
        QGroupBox {
            font-weight: bold;
            font-size: 9pt;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 3px 6px;
        }
        QPushButton {
            background: #056832;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 6px 10px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #6fa34a, stop:1 #4f7a34
            );
        }
        QPushButton:pressed {
            background: #4f7a34;
        }
        QPushButton:disabled {
            background: #9dbb86;
            color: #eeeeee;
        }
        """
        # self.setStyleSheet(style)

    def _wire(self) -> None:
        self.sld_grid_font.valueChanged.connect(self.viewer.set_grid_font_size)
        self.sld_hill.valueChanged.connect(lambda v: self.viewer.set_hillshade_opacity(v / 100.0))
        self.sld_heat.valueChanged.connect(lambda v: self.viewer.set_heatmap_opacity(v / 100.0))
        self.sld_vec_size.valueChanged.connect(self._on_vec_size_changed)
        self.sld_vec_opacity.valueChanged.connect(self._on_vec_opacity_changed)
        self.btn_render_vectors.clicked.connect(self._on_render_vectors)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_prev.clicked.connect(self._on_preview)
        self.btn_confirm.clicked.connect(self._on_confirm_sections)
        self.btn_auto.clicked.connect(self._on_auto_lines)
        self.tbl.customContextMenuRequested.connect(lambda pos: self._on_sections_table_context_menu(self.tbl, "straight", pos))
        self.tbl.viewport().customContextMenuRequested.connect(lambda pos: self._on_sections_table_context_menu(self.tbl, "straight", pos))
        self.tbl.verticalHeader().customContextMenuRequested.connect(lambda pos: self._on_sections_table_header_context_menu(self.tbl, "straight", pos))
        self.tbl_poly.customContextMenuRequested.connect(lambda pos: self._on_sections_table_context_menu(self.tbl_poly, "polyline", pos))
        self.tbl_poly.viewport().customContextMenuRequested.connect(lambda pos: self._on_sections_table_context_menu(self.tbl_poly, "polyline", pos))
        self.tbl_poly.verticalHeader().customContextMenuRequested.connect(lambda pos: self._on_sections_table_header_context_menu(self.tbl_poly, "polyline", pos))

        self.combo_draw_mode.currentIndexChanged.connect(self._on_draw_mode_changed)
        self.viewer.sectionPicked.connect(self._on_section_picked)
        self.viewer.polylinePicked.connect(self._on_polyline_picked)
        self.viewer.cursorMoved.connect(lambda x, y: self.lbl_cursor.setText(f"Cursor: X={x:.2f}, Y={y:.2f}"))

    def _on_render_vectors(self) -> None:
        self._vec_step = int(self.spin_vec_step.value())
        self._vec_scale = float(self.spin_vec_scale.value())
        self._vec_size_pct = int(self.sld_vec_size.value())
        self._vec_opacity_pct = int(self.sld_vec_opacity.value())
        self._vec_color = str(self.combo_vec_color.currentText() or "Blue").strip().lower()

        if not self._ctx_ready:
            self._info("[UI2] Vector display updated (no active context yet).")
            return

        self._load_dx_dy_and_draw(self._ui1_dir, step=self._vec_step, scale=self._vec_scale)
        self._ok(
            f"[UI2] Vectors rendered (step={self._vec_step}, scale={self._vec_scale:.2f}, "
            f"size={self._vec_size_pct}%, opacity={self._vec_opacity_pct}%, color={self._vec_color})."
        )

    def _on_vec_size_changed(self, v: int) -> None:
        self._vec_size_pct = int(v)
        # đồng bộ các input hiện tại trước khi redraw
        self._vec_step = int(self.spin_vec_step.value())
        self._vec_scale = float(self.spin_vec_scale.value())
        self._vec_color = str(self.combo_vec_color.currentText() or "Blue").strip().lower()
        if self._ctx_ready:
            self._load_dx_dy_and_draw(self._ui1_dir, step=self._vec_step, scale=self._vec_scale)

    def _on_vec_opacity_changed(self, v: int) -> None:
        self._vec_opacity_pct = int(v)
        self.viewer.set_vector_opacity(float(self._vec_opacity_pct) / 100.0)

    # ---- core loading/drawing ----
    def _load_layers_and_show(self) -> None:
        # Lấy run_dir từ context đã set
        rd = getattr(self, "run_dir", "") or self._ctx_run_dir
        if not rd or not os.path.isdir(rd):
            return

        ui1 = os.path.join(rd, "ui1")
        payload = self._backend.load_layers(
            rd,
            {"heat_alpha": float(self.sld_heat.value()) / 100.0},
        )
        self._dz = payload["dz"]
        self._dx = payload["dx"]
        self._dy = payload["dy"]
        self._mask = payload["mask"]
        self._tr = payload["transform"]
        self._inv_tr = payload["inv_transform"]
        self._dem_path = payload["dem_path"]
        W = int(payload["width"])
        H = int(payload["height"])
        hs8 = payload["hillshade"]
        self.viewer.set_transform(self._tr)
        self.viewer.set_hillshade(hs8)  # chốt scene rect & fit
        self.viewer.set_hillshade_opacity(self.sld_hill.value() / 100.0)
        self.viewer.set_grid(W, H, step_m=20.0)
        self.viewer.set_grid_font_size(self.sld_grid_font.value())
        alpha = self.sld_heat.value() / 100.0
        self.viewer.set_heatmap_rgba(payload["heat_rgba"], alpha)

        # --- 7) Vẽ vector (dùng dx/dy vừa đọc) theo tham số Vector Display của UI2 ---
        self._load_dx_dy_and_draw(
            ui1,
            step=self._vec_step,
            scale=self._vec_scale,
        )

        # default native transform (khong fit-to-view)
        self.viewer.view.resetTransform()
        self.viewer.view.centerOn(self.viewer.scene.itemsBoundingRect().center())
        self._ok("[UI2] Layers loaded & aligned.")
        self._load_saved_sections()

    def _load_saved_sections(self) -> None:
        """
        Đọc lại ui2/sections.csv + ui2/polylines.json (nếu tồn tại) và vẽ lại các tuyến lên map + bảng.
        """
        if not self.run_dir:
            return

        self._clear_sections_state()
        result = self._backend.load_sections(self.run_dir)
        rows = result.get("rows", [])
        migrated = bool(result.get("migrated", False))
        csv_path = str(result.get("csv_path", ""))
        poly_result = self._backend.load_polylines(self.run_dir)
        poly_rows = list(poly_result.get("rows", []) or [])
        poly_path = str(poly_result.get("json_path", ""))
        if not rows and not poly_rows and not os.path.isfile(csv_path) and not os.path.isfile(poly_path):
            self._info("[UI2] No saved sections.csv or polylines.json – start with empty sections.")
            return

        # Append từng section vào bảng + vẽ line
        count = 0
        for row in rows:
            try:
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
            except Exception:
                continue

            meta = {
                "line_id": str(row.get("line_id", "")).strip(),
                "line_role": str(row.get("line_role", "")).strip(),
            }
            if not meta["line_id"]:
                meta["line_id"] = str(row.get("name", "")).strip()
            self._append_section((x1, y1), (x2, y2), meta=meta)
            count += 1

        for row in poly_rows:
            vertices = [tuple(map(float, v)) for v in list(row.get("vertices", []) or [])]
            meta = {
                "line_id": str(row.get("line_id", "")).strip(),
                "line_role": str(row.get("line_role", "")).strip(),
            }
            if vertices:
                self._append_polyline(vertices, meta=meta)
                count += 1

        if migrated:
            self._ok("[UI2] Migrated legacy sections.csv to current direction version and cleared old UI3 outputs.")
        self._ok(f"[UI2] Loaded {count} sections from saved UI2 files")

    def _on_draw_mode_changed(self, _index: int) -> None:
        mode = "straight"
        if hasattr(self, "combo_draw_mode"):
            mode = str(self.combo_draw_mode.currentData() or "straight")
        self.viewer.set_draw_mode(mode)
        self._update_section_table_visibility()
        self._info(f"[UI2] Draw mode: {mode}")

    def _update_section_table_visibility(self) -> None:
        mode = "straight"
        if hasattr(self, "combo_draw_mode"):
            mode = str(self.combo_draw_mode.currentData() or "straight")
        show_polyline = mode == "polyline"
        if hasattr(self, "lbl_straight_sections"):
            self.lbl_straight_sections.setVisible(not show_polyline)
        if hasattr(self, "tbl"):
            self.tbl.setVisible(not show_polyline)
        if hasattr(self, "lbl_polyline_sections"):
            self.lbl_polyline_sections.setVisible(show_polyline)
        if hasattr(self, "tbl_poly"):
            self.tbl_poly.setVisible(show_polyline)

    @staticmethod
    def _polyline_length(vertices: List[Tuple[float, float]]) -> float:
        if len(vertices) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(vertices)):
            total += float(math.hypot(vertices[i][0] - vertices[i - 1][0], vertices[i][1] - vertices[i - 1][1]))
        return total

    @staticmethod
    def _section_type(meta: Optional[Dict[str, Any]]) -> str:
        return "polyline" if str((meta or {}).get("section_type", "")).strip().lower() == "polyline" else "straight"

    def _section_vertices(self, row: int) -> List[Tuple[float, float]]:
        if not (0 <= row < len(self._sections)):
            return []
        geom = self._sections[row]
        if isinstance(geom, list):
            out: List[Tuple[float, float]] = []
            for vertex in geom:
                try:
                    out.append((float(vertex[0]), float(vertex[1])))
                except Exception:
                    continue
            return out
        if isinstance(geom, tuple) and len(geom) == 2:
            try:
                p0 = (float(geom[0][0]), float(geom[0][1]))
                p1 = (float(geom[1][0]), float(geom[1][1]))
                return [p0, p1]
            except Exception:
                return []
        return []

    def _polyline_vertices(self, row: int) -> List[Tuple[float, float]]:
        if not (0 <= row < len(self._poly_sections)):
            return []
        out: List[Tuple[float, float]] = []
        for vertex in self._poly_sections[row]:
            try:
                out.append((float(vertex[0]), float(vertex[1])))
            except Exception:
                continue
        return out

    @staticmethod
    def _format_polyline_vertices(vertices: List[Tuple[float, float]]) -> str:
        return "; ".join(f"{float(x):.2f}, {float(y):.2f}" for x, y in vertices)

    @staticmethod
    def _parse_polyline_vertices(text: str) -> List[Tuple[float, float]]:
        raw = str(text or "").strip()
        if not raw:
            raise ValueError("Vertices cannot be empty.")
        vertices: List[Tuple[float, float]] = []
        for chunk in raw.split(";"):
            part = chunk.strip()
            if not part:
                continue
            coords = [s.strip() for s in part.split(",")]
            if len(coords) != 2:
                raise ValueError("Each vertex must be in 'x, y' format.")
            x = float(coords[0])
            y = float(coords[1])
            vertices.append((x, y))
        if len(vertices) < 2:
            raise ValueError("Polyline must contain at least 2 vertices.")
        return vertices

    def _current_selection(self) -> Tuple[Optional[str], int]:
        if hasattr(self, "tbl") and self.tbl.currentRow() >= 0 and self.tbl.hasFocus():
            return "straight", int(self.tbl.currentRow())
        if hasattr(self, "tbl_poly") and self.tbl_poly.currentRow() >= 0 and self.tbl_poly.hasFocus():
            return "polyline", int(self.tbl_poly.currentRow())
        if hasattr(self, "tbl_poly") and self.tbl_poly.currentRow() >= 0 and self.tbl_poly.selectedIndexes():
            return "polyline", int(self.tbl_poly.currentRow())
        if hasattr(self, "tbl") and self.tbl.currentRow() >= 0 and self.tbl.selectedIndexes():
            return "straight", int(self.tbl.currentRow())
        return None, -1

    def _section_display_type(self, meta: Dict[str, Any], vertices: List[Tuple[float, float]]) -> str:
        if self._section_type(meta) == "polyline":
            return f"Polyline ({len(vertices)} pts)"
        return "Straight"

    def _build_scene_path(self, vertices: List[Tuple[float, float]]) -> Optional[QPainterPath]:
        if self._inv_tr is None or len(vertices) < 2:
            return None
        c0, r0 = self._inv_tr * vertices[0]
        path = QPainterPath(QPointF(c0, r0))
        for pt in vertices[1:]:
            c, r = self._inv_tr * pt
            path.lineTo(c, r)
        return path

    def _label_anchor_pixels(self, vertices: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
        if self._inv_tr is None or len(vertices) < 2:
            return None
        c0, r0 = self._inv_tr * vertices[0]
        c1, r1 = self._inv_tr * vertices[1]
        return float(c0), float(r0), float(c1), float(r1)

    def _render_section_item(self, vertices: List[Tuple[float, float]], label_text: str) -> Tuple[Optional[object], Optional[QGraphicsSimpleTextItem]]:
        if self._inv_tr is None or len(vertices) < 2:
            return None, None
        is_polyline = len(vertices) > 2
        pen = QPen(QColor(220, 40, 40, 220) if is_polyline else QColor(30, 200, 30, 200))
        pen.setCosmetic(True)
        pen.setWidth(2)
        if len(vertices) == 2:
            c0, r0 = self._inv_tr * vertices[0]
            c1, r1 = self._inv_tr * vertices[1]
            item = QGraphicsLineItem(c0, r0, c1, r1)
            item.setPen(pen)
            item.setZValue(3)
            self.viewer.scene.addItem(item)
        else:
            path = self._build_scene_path(vertices)
            if path is None:
                return None, None
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setZValue(3)
            self.viewer.scene.addItem(item)
        label_item = None
        label_anchor = self._label_anchor_pixels(vertices)
        if label_anchor is not None:
            label_item = self._add_line_label(label_text, *label_anchor, z=5)
        return item, label_item

    def _sync_polyline_row_geometry(self, row: int, vertices: List[Tuple[float, float]]) -> None:
        if not (0 <= row < self.tbl_poly.rowCount()):
            return
        if len(vertices) < 2:
            return
        if 0 <= row < len(self._poly_sections):
            self._poly_sections[row] = list(vertices)
        while len(self._poly_section_meta) <= row:
            self._poly_section_meta.append({})
        if not isinstance(self._poly_section_meta[row], dict):
            self._poly_section_meta[row] = {}
        self._poly_section_meta[row]["vertex_count"] = int(len(vertices))
        self._poly_section_meta[row]["length_m"] = float(self._polyline_length(vertices))

        start_text = f"{vertices[0][0]:.2f}, {vertices[0][1]:.2f}"
        end_text = f"{vertices[-1][0]:.2f}, {vertices[-1][1]:.2f}"
        points_text = str(len(vertices))
        vertices_text = self._format_polyline_vertices(vertices)

        self._updating_table = True
        try:
            start_item = self.tbl_poly.item(row, 1)
            if start_item is None:
                start_item = QTableWidgetItem(start_text)
                start_item.setFlags(start_item.flags() & ~Qt.ItemIsEditable)
                self.tbl_poly.setItem(row, 1, start_item)
            else:
                start_item.setText(start_text)

            end_item = self.tbl_poly.item(row, 2)
            if end_item is None:
                end_item = QTableWidgetItem(end_text)
                end_item.setFlags(end_item.flags() & ~Qt.ItemIsEditable)
                self.tbl_poly.setItem(row, 2, end_item)
            else:
                end_item.setText(end_text)

            points_item = self.tbl_poly.item(row, 3)
            if points_item is None:
                points_item = QTableWidgetItem(points_text)
                points_item.setFlags(points_item.flags() & ~Qt.ItemIsEditable)
                self.tbl_poly.setItem(row, 3, points_item)
            else:
                points_item.setText(points_text)

            vertices_item = self.tbl_poly.item(row, 4)
            if vertices_item is None:
                vertices_item = QTableWidgetItem(vertices_text)
                self.tbl_poly.setItem(row, 4, vertices_item)
            else:
                vertices_item.setText(vertices_text)
        finally:
            self._updating_table = False

        label_text = self.tbl_poly.item(row, 0).text().strip() if self.tbl_poly.item(row, 0) else f"{row + 1}"
        new_item, new_label = self._render_section_item(vertices, label_text)
        if 0 <= row < len(self._poly_section_lines):
            old_item = self._poly_section_lines[row]
            if old_item is not None:
                self.viewer.scene.removeItem(old_item)
            self._poly_section_lines[row] = new_item
        if 0 <= row < len(self._poly_section_line_labels):
            old_label = self._poly_section_line_labels[row]
            if old_label is not None:
                self.viewer.scene.removeItem(old_label)
            self._poly_section_line_labels[row] = new_label

        if self._preview_source == "polyline" and self.tbl_poly.currentRow() == row:
            if self._preview_line is not None:
                self.viewer.scene.removeItem(self._preview_line)
                self._preview_line = None
            if self._preview_label is not None:
                self.viewer.scene.removeItem(self._preview_label)
                self._preview_label = None
            pen = QPen(QColor(200, 30, 30, 220))
            pen.setCosmetic(True)
            pen.setWidth(2)
            if len(vertices) == 2 and self._inv_tr is not None:
                c0, r0 = self._inv_tr * vertices[0]
                c1, r1 = self._inv_tr * vertices[1]
                item = QGraphicsLineItem(c0, r0, c1, r1)
                item.setPen(pen)
                item.setZValue(4)
                self.viewer.scene.addItem(item)
                self._preview_line = item
            else:
                path = self._build_scene_path(vertices)
                if path is not None:
                    item = QGraphicsPathItem(path)
                    item.setPen(pen)
                    item.setZValue(4)
                    self.viewer.scene.addItem(item)
                    self._preview_line = item
            anchor = self._label_anchor_pixels(vertices)
            if anchor is not None:
                self._preview_label = self._add_line_label(label_text, *anchor, z=6)

    def _load_dx_dy_and_draw(self, ui1_dir: str, step: int = 25, scale: float = 1.0) -> None:
        self.viewer.clear_vectors()
        # read & align
        def _read_align(name: str) -> Optional[np.ndarray]:
            p = os.path.join(ui1_dir, name)
            if not _file_exists(p):
                return None
            with rasterio.open(p) as ds:
                arr = ds.read(1).astype("float32")
                if ds.transform != self._tr or (ds.width, ds.height) != self._dz.shape[::-1]:
                    arr = ds.read(1, out_shape=self._dz.shape, resampling=Resampling.bilinear).astype("float32")
            return arr

        self._dx = _read_align("dx.tif")
        self._dy = _read_align("dy.tif")
        if self._dx is None or self._dy is None:
            self._info("[UI2] dx/dy not found → skip vectors.")
            return

        H, W = self._dz.shape
        ys = np.arange(step // 2, H, step)
        xs = np.arange(step // 2, W, step)
        xs, ys = np.meshgrid(xs, ys)
        xs = xs.ravel(); ys = ys.ravel()

        dx = self._dx[ys, xs]
        dy = self._dy[ys, xs]
        msk = self._mask[ys, xs] > 0
        ok = np.isfinite(dx) & np.isfinite(dy) & msk
        xs, ys, dx, dy = xs[ok], ys[ok], dx[ok], dy[ok]

        pix_w = float(abs(self._tr.a)) if self._tr is not None else 1.0
        pix_h = float(abs(self._tr.e)) if self._tr is not None else 1.0

        # scale càng lớn → vector càng dài, giống UI1
        # size_pct cũng scale theo cả chiều dài + chiều ngang (to lên đồng đều)
        size_mul = max(0.2, float(self._vec_size_pct) / 100.0)
        k = 0.4  # hệ số hiệu chỉnh, có thể chỉnh 0.3–0.6 tuỳ mắt nhìn
        vx_pix = (dx / (pix_w + 1e-9)) * scale * k * size_mul
        vy_pix = (-dy / (pix_h + 1e-9)) * scale * k * size_mul  # y pixel hướng xuống

        pts_pix = np.stack([xs, ys], axis=1).astype("float32")
        vec_pix = np.stack([vx_pix, vy_pix], axis=1).astype("float32")
        # vẽ vector liền khối (thân + đầu mũi tên là một polygon)
        color_name = str(getattr(self, "_vec_color", "blue") or "blue").strip().lower()
        color = QColor(color_name)
        if not color.isValid():
            color = QColor("blue")
        shaft_half_w = max(1.2, float(self._vec_pen_base) * 0.5 * size_mul)
        base_head_len = max(3.0, shaft_half_w * 3.2)
        base_head_half_w = max(2.0, shaft_half_w * 2.75)

        for (x, y), (vx, vy) in zip(pts_pix, vec_pix):
            # đảo hướng y vì raster y+ xuống
            end_x = float(x + vx)
            end_y = float(y - vy)
            dxv = end_x - float(x)
            dyv = end_y - float(y)
            length = float(np.hypot(dxv, dyv))
            if not np.isfinite(length) or length <= 1e-6:
                continue

            ux = dxv / length
            uy = dyv / length
            nx = -uy
            ny = ux

            head_len = min(base_head_len, length * 0.45); head_len = max(head_len, shaft_half_w * 1.8)
            head_half_w = max(base_head_half_w, shaft_half_w * 1.05)
            neck_x = end_x - ux * head_len
            neck_y = end_y - uy * head_len

            p0 = QPointF(float(x + nx * shaft_half_w), float(y + ny * shaft_half_w))
            p1 = QPointF(neck_x + nx * shaft_half_w, neck_y + ny * shaft_half_w)
            p2 = QPointF(neck_x + nx * head_half_w, neck_y + ny * head_half_w)
            p3 = QPointF(end_x, end_y)
            p4 = QPointF(neck_x - nx * head_half_w, neck_y - ny * head_half_w)
            p5 = QPointF(neck_x - nx * shaft_half_w, neck_y - ny * shaft_half_w)
            p6 = QPointF(float(x - nx * shaft_half_w), float(y - ny * shaft_half_w))

            path = QPainterPath(p0)
            path.lineTo(p1)
            path.lineTo(p2)
            path.lineTo(p3)
            path.lineTo(p4)
            path.lineTo(p5)
            path.lineTo(p6)
            path.closeSubpath()

            arrow = QGraphicsPathItem(path)
            arrow.setPen(QPen(Qt.NoPen))
            arrow.setBrush(QBrush(color))
            arrow.setZValue(2)
            self.viewer.scene.addItem(arrow)
            self.viewer._vec_items.append(arrow)

        self.viewer.set_vector_opacity(float(self._vec_opacity_pct) / 100.0)

    # ---- section picking ----
    def _on_section_picked(self, x1: float, y1: float, x2: float, y2: float) -> None:
        p0, p1 = self._reverse_section_points((x1, y1), (x2, y2))
        self._append_section(p0, p1)
        self._ok(f"Section added: ({x1:.2f},{y1:.2f}) → ({x2:.2f},{y2:.2f})")

    def _on_polyline_picked(self, points: object) -> None:
        vertices: List[Tuple[float, float]] = []
        for pt in list(points or []):
            try:
                vertices.append((float(pt[0]), float(pt[1])))
            except Exception:
                continue
        if len(vertices) < 2:
            self._warn("[UI2] Polyline needs at least 2 valid points.")
            return
        self._append_polyline(vertices)
        self._ok(f"[UI2] Polyline added: {len(vertices)} points, length={self._polyline_length(vertices):.2f} m")

    @staticmethod
    def _reverse_section_points(
        p0: Tuple[float, float],
        p1: Tuple[float, float],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return storage_reverse_section_points(p0, p1)

    def _append_section(
            self,
            p0: Tuple[float, float],
            p1: Tuple[float, float],
            label: Optional[str] = None,
            meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Lưu section vào bảng + vẽ line cố định trên map."""
        # tránh trigger _on_table_item_changed khi đang chèn row
        self._updating_table = True

        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        hdr = self.tbl.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, hdr.Fixed)
        hdr.setSectionResizeMode(1, hdr.Stretch)
        hdr.setSectionResizeMode(2, hdr.Stretch)
        hdr.setSectionResizeMode(3, hdr.Fixed)

        meta_row = dict(meta or {})
        role_txt = self._role_combo_text_from_value(str(meta_row.get("line_role", "") or ""), str(meta_row.get("line_id", "") or ""))
        meta_row["line_role"] = self._role_value_from_combo_text(role_txt)
        line_id = str(meta_row.get("line_id", "") or "").strip()
        if not line_id:
            line_id = self._next_line_id_for_role(meta_row["line_role"])
        meta_row["line_id"] = line_id
        line_label = label if label is not None else line_id

        self.tbl.setItem(r, 0, QTableWidgetItem(line_label))
        self.tbl.setItem(r, 1, QTableWidgetItem(f"{p0[0]:.2f}, {p0[1]:.2f}"))
        self.tbl.setItem(r, 2, QTableWidgetItem(f"{p1[0]:.2f}, {p1[1]:.2f}"))
        role_combo = _NoWheelComboBox(self.tbl)
        role_combo.addItems(["Main", "Cross"])
        role_combo.setCurrentText(role_txt)
        self.tbl.setCellWidget(r, 3, role_combo)
        self.tbl.verticalHeader().setDefaultSectionSize(30)  # chiều cao mỗi row
        self.tbl.setColumnWidth(0, 56)
        self.tbl.setColumnWidth(3, 100)

        self._updating_table = False
        # cột cuối cùng đang stretch, giữ nguyên

        # lưu section
        self._sections.append((p0, p1))
        self._section_meta.append(meta_row)
        role_combo.currentIndexChanged.connect(self._on_role_combo_changed)

        line_item, label_item = self._render_section_item([p0, p1], line_label)
        self._section_lines.append(line_item)
        self._section_line_labels.append(label_item)
        self._ok("Section line drawn on map.")

    def _append_polyline(
            self,
            vertices: List[Tuple[float, float]],
            label: Optional[str] = None,
            meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(vertices) < 2:
            return
        self._updating_table = True
        r = self.tbl_poly.rowCount()
        self.tbl_poly.insertRow(r)
        hdr = self.tbl_poly.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, hdr.Fixed)
        hdr.setSectionResizeMode(1, hdr.Stretch)
        hdr.setSectionResizeMode(2, hdr.Stretch)
        hdr.setSectionResizeMode(3, hdr.Fixed)
        hdr.setSectionResizeMode(4, hdr.Stretch)
        hdr.setSectionResizeMode(5, hdr.Fixed)

        meta_row = dict(meta or {})
        meta_row["vertex_count"] = int(len(vertices))
        meta_row["length_m"] = float(self._polyline_length(vertices))
        role_txt = self._role_combo_text_from_value(str(meta_row.get("line_role", "") or ""), str(meta_row.get("line_id", "") or ""))
        meta_row["line_role"] = self._role_value_from_combo_text(role_txt)
        line_id = str(meta_row.get("line_id", "") or "").strip()
        if not line_id:
            line_id = self._next_line_id_for_role(meta_row["line_role"])
        meta_row["line_id"] = line_id
        line_label = label if label is not None else line_id
        p0 = vertices[0]
        p1 = vertices[-1]

        id_item = QTableWidgetItem(line_label)
        self.tbl_poly.setItem(r, 0, id_item)
        start_item = QTableWidgetItem(f"{p0[0]:.2f}, {p0[1]:.2f}")
        end_item = QTableWidgetItem(f"{p1[0]:.2f}, {p1[1]:.2f}")
        for item in (start_item, end_item):
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.tbl_poly.setItem(r, 1, start_item)
        self.tbl_poly.setItem(r, 2, end_item)
        points_item = QTableWidgetItem(str(len(vertices)))
        points_item.setFlags(points_item.flags() & ~Qt.ItemIsEditable)
        self.tbl_poly.setItem(r, 3, points_item)
        self.tbl_poly.setItem(r, 4, QTableWidgetItem(self._format_polyline_vertices(vertices)))
        role_combo = _NoWheelComboBox(self.tbl_poly)
        role_combo.addItems(["Main", "Cross"])
        role_combo.setCurrentText(role_txt)
        self.tbl_poly.setCellWidget(r, 5, role_combo)
        self.tbl_poly.verticalHeader().setDefaultSectionSize(30)
        self.tbl_poly.setColumnWidth(0, 56)
        self.tbl_poly.setColumnWidth(3, 70)
        self.tbl_poly.setColumnWidth(5, 100)
        self._updating_table = False

        self._poly_sections.append(list(vertices))
        self._poly_section_meta.append(meta_row)
        role_combo.currentIndexChanged.connect(self._on_role_combo_changed)

        path_item, label_item = self._render_section_item(vertices, line_label)
        self._poly_section_lines.append(path_item)
        self._poly_section_line_labels.append(label_item)
        self._ok("Polyline drawn on map.")

    def _delete_section_row(self, source: str, row: int, log_msg: Optional[str] = None) -> bool:
        table = self.tbl_poly if source == "polyline" else self.tbl
        geom_list = self._poly_sections if source == "polyline" else self._sections
        meta_list = self._poly_section_meta if source == "polyline" else self._section_meta
        item_list = self._poly_section_lines if source == "polyline" else self._section_lines
        label_list = self._poly_section_line_labels if source == "polyline" else self._section_line_labels
        if row < 0 or row >= table.rowCount():
            return False

        # remove map line
        if 0 <= row < len(item_list):
            it = item_list[row]
            if it is not None:
                self.viewer.scene.removeItem(it)
            item_list.pop(row)

        # remove map label
        if 0 <= row < len(label_list):
            it = label_list[row]
            if it is not None:
                self.viewer.scene.removeItem(it)
            label_list.pop(row)

        # remove section data/meta
        if 0 <= row < len(geom_list):
            geom_list.pop(row)
        if 0 <= row < len(meta_list):
            meta_list.pop(row)

        # remove preview if this row was previewed
        if self._preview_line is not None:
            try:
                self.viewer.scene.removeItem(self._preview_line)
            except Exception:
                pass
            self._preview_line = None
        if self._preview_label is not None:
            try:
                self.viewer.scene.removeItem(self._preview_label)
            except Exception:
                pass
            self._preview_label = None
            self._preview_source = None

        table.removeRow(row)
        if log_msg:
            self._ok(log_msg)
        return True

    def _on_sections_table_context_menu(self, table: QTableWidget, source: str, pos) -> None:
        if table is None:
            return
        idx = table.indexAt(pos)
        if not idx.isValid():
            item = table.itemAt(pos)
            if item is None:
                return
            row = int(item.row())
        else:
            row = int(idx.row())
        if row < 0 or row >= table.rowCount():
            return
        table.selectRow(row)
        menu = QMenu(table)
        act_delete = menu.addAction("Delete")
        chosen = menu.exec_(table.viewport().mapToGlobal(pos))
        if chosen is act_delete:
            self._delete_section_row(source, row, log_msg=f"Deleted section #{row + 1}.")

    def _on_sections_table_header_context_menu(self, table: QTableWidget, source: str, pos) -> None:
        if table is None:
            return
        row = int(table.rowAt(pos.y()))
        if row < 0 or row >= table.rowCount():
            return
        table.selectRow(row)
        menu = QMenu(table)
        act_delete = menu.addAction("Delete")
        chosen = menu.exec_(table.verticalHeader().viewport().mapToGlobal(pos))
        if chosen is act_delete:
            self._delete_section_row(source, row, log_msg=f"Deleted section #{row + 1}.")

    def _add_line_label(
            self,
            label_text: str,
            start_x: float,
            start_y: float,
            end_x: float,
            end_y: float,
            z: float = 5,
    ) -> QGraphicsSimpleTextItem:
        lbl = QGraphicsSimpleTextItem(str(label_text))
        lbl.setFont(QFont("Arial", 15))
        lbl.setBrush(QBrush(QColor(0, 170, 0)))

        # Offset label away from the line (perpendicular direction)
        dx = end_x - start_x
        dy = end_y - start_y
        length = (dx * dx + dy * dy) ** 0.5
        if length > 1e-6:
            nx = -dy / length
            ny = dx / length
        else:
            nx, ny = 0.0, -1.0
        offset = 10.0

        br = lbl.boundingRect()
        x = start_x - br.width() * 0.5 + nx * offset
        y = start_y - br.height() * 0.5 + ny * offset
        lbl.setPos(x, y)
        lbl.setZValue(z)
        self.viewer.scene.addItem(lbl)
        return lbl

    def _on_table_item_changed(self, item) -> None:
        """
        Khi user sửa Start/End (x,y) trong bảng, cập nhật ngay line tương ứng trên viewer.
        """
        if self._updating_table:
            return  # bỏ qua khi đang cập nhật bằng code

        if item is None:
            return

        row = item.row()
        col = item.column()

        # chỉ quan tâm cột 1 (Start), 2 (End), 0 (Label)
        if col not in (0, 1, 2):
            return

        # lấy cả start & end của dòng hiện tại
        start_item = self.tbl.item(row, 1)
        end_item = self.tbl.item(row, 2)
        if start_item is None or end_item is None:
            return

        try:
            p0 = tuple(map(float, start_item.text().split(",")))
            p1 = tuple(map(float, end_item.text().split(",")))
            if self._inv_tr is None:
                return
            c0, r0 = self._inv_tr * p0
            c1, r1 = self._inv_tr * p1
        except Exception:
            # nếu user gõ dở dang (ví dụ "120,"), tạm bỏ qua
            return

        # cập nhật list _sections
        if 0 <= row < len(self._sections):
            self._sections[row] = (p0, p1)
        if col == 0:
            while len(self._section_meta) <= row:
                self._section_meta.append({})
            if not isinstance(self._section_meta[row], dict):
                self._section_meta[row] = {}
            self._section_meta[row]["line_id"] = (self.tbl.item(row, 0).text().strip() if self.tbl.item(row, 0) else "")

        # cập nhật line xanh trên map
        if 0 <= row < len(self._section_lines):
            line_item = self._section_lines[row]
            if isinstance(line_item, QGraphicsLineItem):
                line_item.setLine(c0, r0, c1, r1)

        # cập nhật label (line number)
        if 0 <= row < len(self._section_line_labels):
            lbl = self._section_line_labels[row]
            if lbl is not None:
                label_item = self.tbl.item(row, 0)
                lbl.setText(label_item.text() if label_item else str(row + 1))
                br = lbl.boundingRect()
                dx = c1 - c0
                dy = r1 - r0
                length = (dx * dx + dy * dy) ** 0.5
                if length > 1e-6:
                    nx = -dy / length
                    ny = dx / length
                else:
                    nx, ny = 0.0, -1.0
                offset = 10.0
                lbl.setPos(
                    c0 - br.width() * 0.5 + nx * offset,
                    r0 - br.height() * 0.5 + ny * offset,
                )

        # nếu đang có preview line cho đúng dòng này, cập nhật luôn
        if self._preview_line is not None and self.tbl.currentRow() == row:
            self._preview_line.setLine(c0, r0, c1, r1)
            if self._preview_label is not None:
                label_item = self.tbl.item(row, 0)
                self._preview_label.setText(label_item.text() if label_item else str(row + 1))
                br = self._preview_label.boundingRect()
                dx = c1 - c0
                dy = r1 - r0
                length = (dx * dx + dy * dy) ** 0.5
                if length > 1e-6:
                    nx = -dy / length
                    ny = dx / length
                else:
                    nx, ny = 0.0, -1.0
                offset = 10.0
                self._preview_label.setPos(
                    c0 - br.width() * 0.5 + nx * offset,
                    r0 - br.height() * 0.5 + ny * offset,
                )

        self._ok(f"Updated section #{row + 1} from table edit.")

    def _on_polyline_table_item_changed(self, item) -> None:
        if self._updating_table or item is None:
            return
        row = item.row()
        col = item.column()
        if col not in (0, 4):
            return
        if col == 4:
            try:
                vertices = self._parse_polyline_vertices(item.text())
            except Exception:
                return
            self._sync_polyline_row_geometry(row, vertices)
        if 0 <= row < len(self._poly_section_meta) and isinstance(self._poly_section_meta[row], dict):
            self._poly_section_meta[row]["line_id"] = (self.tbl_poly.item(row, 0).text().strip() if self.tbl_poly.item(row, 0) else "")
        if 0 <= row < len(self._poly_section_line_labels):
            lbl = self._poly_section_line_labels[row]
            if lbl is not None:
                lbl.setText(self.tbl_poly.item(row, 0).text().strip() if self.tbl_poly.item(row, 0) else f"{row + 1}")
        if self._preview_label is not None and self._preview_source == "polyline" and self.tbl_poly.currentRow() == row:
            self._preview_label.setText(self.tbl_poly.item(row, 0).text().strip() if self.tbl_poly.item(row, 0) else f"{row + 1}")
        self._ok(f"Updated polyline #{row + 1} from table edit.")

    def _on_auto_lines(self) -> None:
        """
        Handler cho nút 'Auto Line Generation'.

        Bản này dùng trực tiếp dx/dy/mask đã load trong SectionSelectionTab
        (self._dx, self._dy, self._mask, self._tr) và gọi
        generate_auto_lines_from_arrays(...) để đảm bảo hướng tuyến
        giống vector dXY ở UI1.
        """
        # 0) Kiểm tra context
        if self.run_dir is None or self._tr is None:
            self._info("[UI2] Auto line: run context not ready. Please run Analyze first.")
            return

        if self._dx is None or self._dy is None or self._mask is None:
            self._info("[UI2] Auto line: dx/dy/mask not ready. Please run Analyze first.")
            return

        # 1) Hiện dialog nhập tham số
        dlg = AutoLineDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return  # người dùng Cancel

        main_num_even = int(dlg.main_num.value())
        main_offset_m = float(dlg.main_off.value())
        cross_num_even = int(dlg.cross_num.value())
        cross_offset_m = float(dlg.cross_off.value())

        # 2) Ép số chẵn + kiểm tra offset >= 0 (giống UI2 cũ)
        if main_num_even % 2 == 1:
            main_num_even -= 1
            self._info("[UI2] Main lines: số tuyến phải là số chẵn – đã tự động giảm 1.")

        if cross_num_even % 2 == 1:
            cross_num_even -= 1
            self._info("[UI2] Cross lines: số tuyến phải là số chẵn – đã tự động giảm 1.")

        if main_offset_m < 0 or cross_offset_m < 0:
            self._info("[UI2] Auto line: Offset must be ≥ 0.")
            return

        # 3) Gọi backend dạng mảng – không phụ thuộc tên file detect_mask.tif nữa
        try:
            outs = self._backend.generate_auto_lines(
                dx=self._dx,
                dy=self._dy,
                mask=self._mask,
                transform=self._tr,
                params={
                    "main_num_even": main_num_even,
                    "main_offset_m": main_offset_m,
                    "cross_num_even": cross_num_even,
                    "cross_offset_m": cross_offset_m,
                    "base_length_m": None,
                },
            )
        except Exception as e:
            self._info(f"[UI2] Auto line: generation failed: {e}")
            return

        mains = outs.get("main", [])
        crosses = outs.get("cross", [])
        if not mains and not crosses:
            self._info("[UI2] Auto line: no lines returned.")
            return

        # redraw vectors after auto line generation
        self._load_dx_dy_and_draw(self._ui1_dir, step=self._vec_step, scale=self._vec_scale)
        debug = outs.get("debug", {})
        ang = debug.get("ang_main_deg", None)
        if ang is not None:
            self._ok(f"[UI2] Main direction ≈ {ang:.1f} deg")

        # 4) Xoá tất cả section cũ (bảng + line trên map)
        self._clear_sections_state()

        # 5) Helper: convert feat (LineString) -> 1 dòng trong bảng + vẽ line
        def _add_feat(feat: dict) -> None:
            geom = feat.get("geom")
            if not isinstance(geom, LineString) or geom.is_empty:
                return
            x1, y1 = geom.coords[0]
            x2, y2 = geom.coords[-1]
            p0, p1 = self._reverse_section_points((x1, y1), (x2, y2))
            self._append_section(
                p0, p1,
                meta={
                    "line_id": str(feat.get("name", "")).strip(),
                    "line_role": str(feat.get("type", "")).strip(),
                    "offset_m": float(feat.get("offset_m", 0.0)) if feat.get("offset_m", None) is not None else None,
                    "angle_deg": float(feat.get("angle_deg", 0.0)) if feat.get("angle_deg", None) is not None else None,
                }
            )

        for f in mains:
            _add_feat(f)
        for f in crosses:
            _add_feat(f)

        self._ok(f"[UI2] Auto line OK: {len(mains)} main + {len(crosses)} cross lines.")

    def _on_preview(self) -> None:
        source, r = self._current_selection()
        if source is None or r < 0 or self._inv_tr is None:
            self._info("Select a row to preview.")
            return

        # Xoá preview cũ nếu có
        if self._preview_line is not None:
            self.viewer.scene.removeItem(self._preview_line)
            self._preview_line = None
        if self._preview_label is not None:
            self.viewer.scene.removeItem(self._preview_label)
            self._preview_label = None
        self._preview_source = source

        pen = QPen(QColor(200, 30, 30, 220))
        pen.setCosmetic(True)
        pen.setWidth(2)
        vertices = self._polyline_vertices(r) if source == "polyline" else self._section_vertices(r)
        if len(vertices) < 2:
            self._info("Selected row has invalid geometry.")
            return
        if len(vertices) == 2:
            c0, r0 = self._inv_tr * vertices[0]
            c1, r1 = self._inv_tr * vertices[1]
            item = QGraphicsLineItem(c0, r0, c1, r1)
            item.setPen(pen)
            item.setZValue(4)
            self.viewer.scene.addItem(item)
            self._preview_line = item
        else:
            path = self._build_scene_path(vertices)
            if path is None:
                return
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setZValue(4)
            self.viewer.scene.addItem(item)
            self._preview_line = item
        label_item = self.tbl_poly.item(r, 0) if source == "polyline" else self.tbl.item(r, 0)
        anchor = self._label_anchor_pixels(vertices)
        if anchor is not None:
            self._preview_label = self._add_line_label(
                label_item.text() if label_item else str(r + 1),
                *anchor,
                z=6,
            )

        self._ok("Preview line drawn.")

    def _clear_sections_state(self) -> None:
        if hasattr(self, "tbl"):
            self.tbl.setRowCount(0)
        if hasattr(self, "tbl_poly"):
            self.tbl_poly.setRowCount(0)
        self._sections.clear()
        self._section_meta.clear()
        self._poly_sections.clear()
        self._poly_section_meta.clear()

        for it in getattr(self, "_section_lines", []):
            if it is not None:
                self.viewer.scene.removeItem(it)
        self._section_lines.clear()

        for it in getattr(self, "_section_line_labels", []):
            if it is not None:
                self.viewer.scene.removeItem(it)
        self._section_line_labels.clear()

        for it in getattr(self, "_poly_section_lines", []):
            if it is not None:
                self.viewer.scene.removeItem(it)
        self._poly_section_lines.clear()

        for it in getattr(self, "_poly_section_line_labels", []):
            if it is not None:
                self.viewer.scene.removeItem(it)
        self._poly_section_line_labels.clear()

        if getattr(self, "_preview_line", None) is not None:
            self.viewer.scene.removeItem(self._preview_line)
            self._preview_line = None
        if getattr(self, "_preview_label", None) is not None:
            self.viewer.scene.removeItem(self._preview_label)
            self._preview_label = None
        self._preview_source = None

    def _on_clear(self) -> None:
        """Xoá toàn bộ sections + line trên map."""
        reply = QMessageBox.question(
            self,
            "Confirm",
            "本当に消しますか？\nAre you sure you want to delete it?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._clear_sections_state()
        self._ok("Cleared sections.")

    @staticmethod
    def _normalize_line_role(line_role: str, line_id: str = "") -> str:
        return backend_normalize_line_role(line_role, line_id)

    @staticmethod
    def _role_combo_text_from_value(line_role: str, line_id: str = "") -> str:
        role = SectionSelectionTab._normalize_line_role(line_role, line_id)
        return "Cross" if role == "cross" else "Main"

    @staticmethod
    def _role_value_from_combo_text(text: str) -> str:
        t = str(text or "").strip().lower()
        return "cross" if t == "cross" else "main"

    @staticmethod
    def _parse_auto_line_id(line_id: str) -> Tuple[str, int]:
        return backend_parse_auto_line_id(line_id)

    def _next_line_id_for_role(self, line_role: str, exclude_row: int = -1, exclude_source: str = "straight") -> str:
        role = self._normalize_line_role(line_role, "")
        prefix = "CL" if role == "cross" else "ML"
        used = set()
        for source, table, meta_list in (
            ("straight", self.tbl, self._section_meta),
            ("polyline", self.tbl_poly, self._poly_section_meta),
        ):
            for r in range(table.rowCount()):
                if source == exclude_source and r == exclude_row:
                    continue
                cand = ""
                if 0 <= r < len(meta_list) and isinstance(meta_list[r], dict):
                    cand = str(meta_list[r].get("line_id", "") or "").strip()
                if not cand:
                    item = table.item(r, 0)
                    cand = item.text().strip() if item else ""
                parsed_role, parsed_idx = self._parse_auto_line_id(cand)
                if parsed_role == role and parsed_idx > 0:
                    used.add(parsed_idx)
        n = 1
        while n in used:
            n += 1
        return f"{prefix}{n}"

    def _find_role_combo_row(self, combo: QComboBox) -> Tuple[Optional[str], int]:
        for r in range(self.tbl.rowCount()):
            if self.tbl.cellWidget(r, 3) is combo:
                return "straight", r
        for r in range(self.tbl_poly.rowCount()):
            if self.tbl_poly.cellWidget(r, 5) is combo:
                return "polyline", r
        return None, -1

    def _on_role_combo_changed(self, _index: int) -> None:
        combo = self.sender()
        if not isinstance(combo, QComboBox):
            return
        source, row = self._find_role_combo_row(combo)
        if source is None or row < 0:
            return
        table = self.tbl_poly if source == "polyline" else self.tbl
        meta_list = self._poly_section_meta if source == "polyline" else self._section_meta
        while len(meta_list) <= row:
            meta_list.append({})
        meta = meta_list[row]
        if not isinstance(meta, dict):
            meta = {}
            meta_list[row] = meta
        label = (table.item(row, 0).text().strip() if table.item(row, 0) else f"{row + 1}")
        cur_line_id = str(meta.get("line_id", "") or "").strip() or label
        new_role = self._role_value_from_combo_text(combo.currentText())
        auto_role, auto_idx = self._parse_auto_line_id(cur_line_id)
        if (not cur_line_id) or (auto_idx > 0 and auto_role in ("main", "cross")):
            new_line_id = self._next_line_id_for_role(new_role, exclude_row=row, exclude_source=source)
            if new_line_id != cur_line_id:
                item0 = table.item(row, 0)
                if item0 is None:
                    item0 = QTableWidgetItem(new_line_id)
                    table.setItem(row, 0, item0)
                else:
                    item0.setText(new_line_id)
                cur_line_id = new_line_id
        meta["line_id"] = cur_line_id
        meta["line_role"] = new_role
        self._ok(f"[UI2] Line role set: row {row + 1} -> {meta['line_role']}")

    def _get_row_line_role(self, source: str, row: int) -> str:
        table = self.tbl_poly if source == "polyline" else self.tbl
        combo_col = 5 if source == "polyline" else 3
        meta_list = self._poly_section_meta if source == "polyline" else self._section_meta
        combo = table.cellWidget(row, combo_col) if table is not None else None
        if isinstance(combo, QComboBox):
            return self._role_value_from_combo_text(combo.currentText())
        meta = meta_list[row] if 0 <= row < len(meta_list) else {}
        line_id = str((meta or {}).get("line_id", "")).strip()
        return self._normalize_line_role(str((meta or {}).get("line_role", "")).strip(), line_id)

    @staticmethod
    def _line_order_key(line_id: str, fallback_idx: int) -> Tuple[int, str]:
        return backend_line_order_key(line_id, fallback_idx)

    @staticmethod
    def _pick_intersection_point(geom_a: LineString, geom_b: LineString) -> Tuple[Optional[Point], str]:
        return backend_pick_intersection_point(geom_a, geom_b)

    def _save_main_cross_intersections(self, ui2_dir: str) -> Optional[str]:
        records: List[Dict[str, Any]] = []
        skipped_polyline = len(self._poly_sections)
        for r in range(self.tbl.rowCount()):
            meta = self._section_meta[r] if 0 <= r < len(self._section_meta) else {}
            try:
                p0 = tuple(map(float, self.tbl.item(r, 1).text().split(",")))
                p1 = tuple(map(float, self.tbl.item(r, 2).text().split(",")))
            except Exception:
                continue
            line_id = str((meta or {}).get("line_id", "")).strip()
            label = (self.tbl.item(r, 0).text().strip() if self.tbl.item(r, 0) else f"{r + 1}")
            if not line_id:
                line_id = label
                if 0 <= r < len(self._section_meta) and isinstance(self._section_meta[r], dict):
                    self._section_meta[r]["line_id"] = line_id
            line_role = self._normalize_line_role(self._get_row_line_role("straight", r), line_id)
            if 0 <= r < len(self._section_meta) and isinstance(self._section_meta[r], dict):
                self._section_meta[r]["line_role"] = line_role
            try:
                geom = LineString([p0, p1])
            except Exception:
                continue
            records.append({
                "row_index": int(r),
                "table_label": label,
                "line_id": line_id,
                "line_role": line_role,
                "x1": float(p0[0]),
                "y1": float(p0[1]),
                "x2": float(p1[0]),
                "y2": float(p1[1]),
                "geom": geom,
            })
        _ = ui2_dir
        if skipped_polyline > 0:
            self._info(f"[UI2] Skipped {skipped_polyline} polyline section(s) for intersections export.")
        return self._backend.save_main_cross_intersections(self.run_dir or "", records)

    def _on_confirm_sections(self) -> None:
        """Ghi ui2/sections.csv và phát 1 signal sang MainWindow/Curve tab."""
        # Bắt buộc phải có context (Analyze tab đã chạy)
        if not getattr(self, "_ctx_ready", False):
            self._log("[UI2] Please run Analyze first (no context).")
            return

        # Đọc context đang lưu trong UI2
        project = self._ctx_project or (
            self.edit_project.text().strip() if hasattr(self, "edit_project") else ""
        )
        run_label = self._ctx_runlabel or (
            self.edit_runlabel.text().strip() if hasattr(self, "edit_runlabel") else ""
        )
        run_dir = self._ctx_run_dir

        if not (project and run_label and run_dir):
            self._err("Missing project/run context. Please render vectors in Analyze tab first.")
            return

        # Ghi ui2/sections.csv
        ui2_dir = os.path.join(run_dir, "ui2")
        os.makedirs(ui2_dir, exist_ok=True)
        csv_path = os.path.join(ui2_dir, "sections.csv")

        try:
            rows_to_save: List[Dict[str, Any]] = []
            polylines_to_save: List[Dict[str, Any]] = []
            for r in range(self.tbl.rowCount()):
                meta = self._section_meta[r] if 0 <= r < len(self._section_meta) else {}
                line_id = str((meta or {}).get("line_id", "")).strip()
                if not line_id:
                    label = (self.tbl.item(r, 0).text().strip() if self.tbl.item(r, 0) else f"{r + 1}")
                    line_id = label
                    if 0 <= r < len(self._section_meta) and isinstance(self._section_meta[r], dict):
                        self._section_meta[r]["line_id"] = line_id
                line_role = self._normalize_line_role(self._get_row_line_role("straight", r), line_id)
                if 0 <= r < len(self._section_meta) and isinstance(self._section_meta[r], dict):
                    self._section_meta[r]["line_role"] = line_role
                p0 = tuple(map(float, self.tbl.item(r, 1).text().split(",")))
                p1 = tuple(map(float, self.tbl.item(r, 2).text().split(",")))
                rows_to_save.append(storage_canonical_section_csv_row(
                    len(rows_to_save) + 1,
                    p0,
                    p1,
                    line_id=line_id,
                    line_role=line_role,
                ))
            for r in range(self.tbl_poly.rowCount()):
                meta = self._poly_section_meta[r] if 0 <= r < len(self._poly_section_meta) else {}
                line_id = str((meta or {}).get("line_id", "")).strip()
                if not line_id:
                    label = (self.tbl_poly.item(r, 0).text().strip() if self.tbl_poly.item(r, 0) else f"{r + 1}")
                    line_id = label
                    if 0 <= r < len(self._poly_section_meta) and isinstance(self._poly_section_meta[r], dict):
                        self._poly_section_meta[r]["line_id"] = line_id
                line_role = self._normalize_line_role(self._get_row_line_role("polyline", r), line_id)
                if 0 <= r < len(self._poly_section_meta) and isinstance(self._poly_section_meta[r], dict):
                    self._poly_section_meta[r]["line_role"] = line_role
                vertices = self._polyline_vertices(r)
                if len(vertices) >= 2:
                    polylines_to_save.append({
                        "idx": int(len(polylines_to_save) + 1),
                        "line_id": line_id,
                        "line_role": line_role,
                        "vertices": [[float(x), float(y)] for x, y in vertices],
                    })
            self._backend.save_sections(run_dir, rows_to_save)
            poly_path = self._backend.save_polylines(run_dir, polylines_to_save)

            self._ok(f"Sections saved: {csv_path}")
            self._ok(f"Polylines saved: {poly_path}")
            try:
                inter_path = self._save_main_cross_intersections(ui2_dir)
                if inter_path:
                    self._ok(f"[UI2] Intersections saved: {inter_path}")
            except Exception as e:
                self._err(f"[UI2] Save intersections error: {e}")

        except Exception as e:
            self._err(f"Confirm Sections error: {e}")
            return

        # 👉 Emit DUY NHẤT 1 lần, dùng context chuẩn
        try:
            self.sections_confirmed.emit(project, run_label, run_dir)
            self._log(f"[UI2] sections_confirmed emitted: {project}, {run_label}, {run_dir}")
        except Exception as e:
            self._err(f"Emit sections_confirmed error: {e}")

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and getattr(self.viewer, "is_polyline_active", lambda: False)():
            if self.viewer.finish_polyline_pick():
                self._ok("[UI2] Polyline completed.")
                event.accept()
                return
        if event.key() == Qt.Key_Delete:
            source, _ = self._current_selection()
            table = self.tbl_poly if source == "polyline" else self.tbl
            sel = table.selectedIndexes() if source else []
            if not sel:
                return

            rows = sorted(set(i.row() for i in sel))

            # nếu chọn ít nhất 1 cột Start/End → coi như xóa cả dòng
            if len(rows) >= 1:
                for r in reversed(rows):
                    self._delete_section_row(str(source or "straight"), r)
                self._ok("Deleted selected section(s).")
                event.accept()
                return

        super().keyPressEvent(event)
    # ---------- Public API cho MainWindow ----------

    def reset_session(self) -> None:
        """
        Reset tab Section:
        - Xoá project/run label hiển thị
        - Xoá sections trong bảng + line trên map
        - Xoá status
        - Xoá context run_dir / transform
        """
        # Clear thông tin project/run
        if hasattr(self, "edit_project"):
            self.edit_project.clear()
        if hasattr(self, "edit_runlabel"):
            self.edit_runlabel.clear()

        # Reset context
        self.project = None
        self.run_label = None
        self.run_dir = None
        self._ctx_project = ""
        self._ctx_runlabel = ""
        self._ctx_run_dir = ""
        self._ui1_dir = ""
        self._ui2_dir = ""
        self._backend.reset_context()
        self._ctx_ready = False
        self._dx = None
        self._dy = None
        self._tr = None
        self._inv_tr = None
        self._dz = None
        self._mask = None

        # reset vector params về mặc định
        self._vec_step = 25
        self._vec_scale = 1.0
        self._vec_size_pct = 100
        self._vec_opacity_pct = 100
        self._vec_color = "blue"
        if hasattr(self, "spin_vec_step"):
            self.spin_vec_step.setValue(25)
        if hasattr(self, "spin_vec_scale"):
            self.spin_vec_scale.setValue(1.0)
        if hasattr(self, "combo_vec_color"):
            self.combo_vec_color.setCurrentText("Blue")
        if hasattr(self, "sld_vec_size"):
            self.sld_vec_size.setValue(100)
        if hasattr(self, "sld_vec_opacity"):
            self.sld_vec_opacity.setValue(100)
        if hasattr(self, "combo_draw_mode"):
            self.combo_draw_mode.setCurrentIndex(0)
        self.viewer.set_draw_mode("straight")
        self._update_section_table_visibility()

        # Xoá sections & line
        self._clear_sections_state()

        # Xoá status
        if hasattr(self, "status_text"):
            self.status_text.clear()

        # Xoá layer / vector / lưới (nếu viewer có các hàm này)
        try:
            # Nếu bạn có các hàm clear_vectors / set_heatmap_rgba..., có thể dùng thêm
            self.viewer.clear_vectors()
            self.viewer.set_heatmap_rgba(
                None, 0.0
            )  # nếu hàm này không chấp nhận None thì bỏ dòng này
        except Exception:
            pass

    # ---- status helpers ----
    @staticmethod
    def _status_brief(msg: str, fallback: str) -> str:
        skip_prefixes = ("project:", "run:", "output:", "folder:", "paths:")
        for raw in str(msg or "").splitlines():
            line = raw.strip()
            if not line:
                continue
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
        self.status_text.appendPlainText(text)
        self.status_text.moveCursor(self.status_text.textCursor().End)

    def _info(self, msg: str) -> None:
        return

    def _ok(self, msg: str) -> None:
        self._append_status(f"[UI2] OK: {self._status_brief(msg, 'Completed.')}")

    def _warn(self, msg: str) -> None:
        self._append_status(f"[UI2] ERROR: {self._status_brief(msg, 'Action required.')}")

    def _err(self, msg: str) -> None:
        """Append error to status + popup."""
        self._append_status(f"[UI2] ERROR: {self._status_brief(msg, 'Error.')}")
        try:
            QMessageBox.critical(self, "Section Selection", msg)
        except Exception:
            pass  # phòng trường hợp gọi khi app chưa sẵn sàng

    # Back-compat: vài nơi cũ còn gọi self._log(...)
    def _log(self, msg: str) -> None:
        return
