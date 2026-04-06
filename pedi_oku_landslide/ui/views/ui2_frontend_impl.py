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
from .ui2_dialogs import AutoLineDialog
from .ui2_layered_viewer import _LayeredViewer
from .ui2_widgets import HBox, NoWheelComboBox as _NoWheelComboBox

# ---------- UI2: Section Selection tab ----------

class SectionSelectionTab(QWidget):
    sections_confirmed = pyqtSignal(str, str, str)  # project, run_label, run_dir
    def __init__(self, base_dir: str, parent=None) -> None:
        super().__init__(parent)
        self.base_dir = base_dir
        self._backend = UI2BackendService()
        self._splitter: Optional[QSplitter] = None
        self._left_min_w = 380
        self._left_default_w = 490
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
        self._section_lines: list[QGraphicsLineItem] = []
        self._section_line_labels: list[QGraphicsSimpleTextItem] = []
        self._preview_line: Optional[QGraphicsLineItem] = None
        self._preview_label: Optional[QGraphicsSimpleTextItem] = None

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

        self._sections: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        self._section_meta: List[Dict[str, Any]] = []

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
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        self._splitter = splitter
        splitter.splitterMoved.connect(lambda *_: self._enforce_left_pane_bounds())
        root.addWidget(splitter)
        self.viewer = _LayeredViewer(self)

        # left pane
        left = QWidget(); left.setMinimumWidth(self._left_min_w); left_lo = QVBoxLayout(left)

        grp_proj = QGroupBox("Project"); gl = QHBoxLayout(grp_proj)
        gl.setContentsMargins(8, 8, 8, 8)
        gl.setSpacing(6)
        proj_input_h = 30
        lbl_name = QLabel("Name:")
        lbl_run = QLabel("Run label:")
        fm = lbl_name.fontMetrics()
        proj_label_w = max(fm.horizontalAdvance("Name:"), fm.horizontalAdvance("Run label:")) + 8
        self.edit_project = QLineEdit(); self.edit_project.setPlaceholderText("—")
        self.edit_project.setFixedHeight(proj_input_h)
        self.edit_project.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lbl_name.setFixedWidth(proj_label_w)
        gl.addWidget(lbl_name)
        gl.addWidget(self.edit_project, 1)
        self.edit_runlabel = QLineEdit(); self.edit_runlabel.setPlaceholderText("—")
        self.edit_runlabel.setFixedHeight(proj_input_h)
        self.edit_runlabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lbl_run.setFixedWidth(proj_label_w)
        gl.addWidget(lbl_run)
        gl.addWidget(self.edit_runlabel, 1)
        left_lo.addWidget(grp_proj)

        grp_layers = QGroupBox("Layers"); ll = QGridLayout(grp_layers)
        ll.setHorizontalSpacing(6)
        ll.setVerticalSpacing(6)
        ll.setColumnStretch(1, 1)

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
        left_lo.addWidget(grp_layers)

        grp_vec = QGroupBox("Vector Display"); vvl = QVBoxLayout(grp_vec)
        grid_vec_top = QGridLayout()
        grid_vec_top.setHorizontalSpacing(8)
        grid_vec_top.setVerticalSpacing(6)
        grid_vec_top.setColumnStretch(1, 1)
        grid_vec_top.setColumnStretch(3, 1)
        grid_vec_top.setColumnStretch(5, 1)

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
        for col in (0, 2, 4):
            grid_vec_top.setColumnMinimumWidth(col, label_col_w)
        input_min_w = max(
            self.spin_vec_step.sizeHint().width(),
            self.spin_vec_scale.sizeHint().width(),
            self.combo_vec_color.sizeHint().width(),
        )
        for w in (self.spin_vec_step, self.spin_vec_scale, self.combo_vec_color):
            w.setMinimumWidth(input_min_w)

        grid_vec_top.addWidget(lbl_step, 0, 0)
        grid_vec_top.addWidget(self.spin_vec_step, 0, 1)
        grid_vec_top.addWidget(lbl_scale, 0, 2)
        grid_vec_top.addWidget(self.spin_vec_scale, 0, 3)
        grid_vec_top.addWidget(lbl_color, 0, 4)
        grid_vec_top.addWidget(self.combo_vec_color, 0, 5)
        vvl.addLayout(grid_vec_top)

        grid_vec_sliders = QGridLayout()
        grid_vec_sliders.setHorizontalSpacing(8)
        grid_vec_sliders.setVerticalSpacing(6)
        grid_vec_sliders.setColumnStretch(1, 1)
        grid_vec_sliders.addWidget(lbl_size, 0, 0)
        grid_vec_sliders.addWidget(self.sld_vec_size, 0, 1)
        grid_vec_sliders.addWidget(lbl_opacity, 1, 0)
        grid_vec_sliders.addWidget(self.sld_vec_opacity, 1, 1)
        vvl.addLayout(grid_vec_sliders)

        row_vec_btn = HBox()
        self.btn_render_vectors = QPushButton("Render Vectors")
        row_vec_btn.addWidget(self.btn_render_vectors, 1)
        vvl.addLayout(row_vec_btn)
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
        self.tbl.itemChanged.connect(self._on_table_item_changed)
        sl.addWidget(self.tbl)

        # Một hàng nút thao tác Section: Auto Line, Draw Line, Clear All, Confirm
        row_actions = HBox()
        self.btn_auto = QPushButton("Auto Line")
        self.btn_prev = QPushButton("Draw Line")
        self.btn_clear = QPushButton("Clear All")
        self.btn_confirm = QPushButton("Confirm")

        for b in (self.btn_auto, self.btn_prev, self.btn_clear, self.btn_confirm):
            # stretch=1 → 4 nút chia đều chiều ngang
            row_actions.addWidget(b, 1)
        sl.addLayout(row_actions)

        left_lo.addWidget(grp_secs)

        grp_status = QGroupBox("Status"); sv = QVBoxLayout(grp_status)
        self.lbl_cursor = QLabel("Cursor: —")
        from PyQt5.QtWidgets import QPlainTextEdit
        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.status_text.setMaximumBlockCount(2000)  # tránh phình bộ nhớ khi log dài
        self.status_text.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")
        sv.addWidget(self.lbl_cursor); sv.addWidget(self.status_text)
        left_lo.addWidget(grp_status)

        left_lo.addStretch(1)
        # right pane: viewer
        splitter.addWidget(left)  # khung trái (form, table, status)
        splitter.addWidget(self.viewer)  # khung phải (map)
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
        self.tbl.customContextMenuRequested.connect(self._on_sections_table_context_menu)
        self.tbl.viewport().customContextMenuRequested.connect(self._on_sections_table_context_menu)
        self.tbl.verticalHeader().customContextMenuRequested.connect(self._on_sections_table_header_context_menu)

        self.viewer.sectionPicked.connect(self._on_section_picked)
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

        # default zoom = 100% (không fit-to-view)
        self.viewer.view.resetTransform()
        self.viewer.view.centerOn(self.viewer.scene.itemsBoundingRect().center())
        self._ok("[UI2] Layers loaded & aligned.")
        self._load_saved_sections()

    def _load_saved_sections(self) -> None:
        """
        Đọc lại ui2/sections.csv (nếu tồn tại) và vẽ lại các tuyến lên map + bảng.
        """
        if not self.run_dir:
            return

        self._clear_sections_state()
        result = self._backend.load_sections(self.run_dir)
        rows = result.get("rows", [])
        migrated = bool(result.get("migrated", False))
        csv_path = str(result.get("csv_path", ""))
        if not rows and not os.path.isfile(csv_path):
            self._info("[UI2] No saved sections.csv – start with empty sections.")
            return

        if not rows:
            self._info("[UI2] sections.csv is empty.")
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

        if migrated:
            self._ok("[UI2] Migrated legacy sections.csv to direction_version=2 and cleared old UI3 outputs.")
        self._ok(f"[UI2] Loaded {count} sections from sections.csv")

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

        # 2) vẽ line lên viewer (map → pixel)
        line_item = None
        label_item = None
        if self._inv_tr is not None:
            c0, r0 = self._inv_tr * p0
            c1, r1 = self._inv_tr * p1
            pen = QPen(QColor(30, 200, 30, 200))
            pen.setCosmetic(True)
            pen.setWidth(2)
            line_item = QGraphicsLineItem(c0, r0, c1, r1)
            line_item.setPen(pen)
            line_item.setZValue(3)
            self.viewer.scene.addItem(line_item)
            label_item = self._add_line_label(line_label, c0, r0, c1, r1, z=5)

        self._section_lines.append(line_item)
        self._section_line_labels.append(label_item)
        self._ok("Section line drawn on map.")

    def _delete_section_row(self, row: int, log_msg: Optional[str] = None) -> bool:
        if row < 0 or row >= self.tbl.rowCount():
            return False

        # remove map line
        if 0 <= row < len(self._section_lines):
            it = self._section_lines[row]
            if it is not None:
                self.viewer.scene.removeItem(it)
            self._section_lines.pop(row)

        # remove map label
        if 0 <= row < len(self._section_line_labels):
            it = self._section_line_labels[row]
            if it is not None:
                self.viewer.scene.removeItem(it)
            self._section_line_labels.pop(row)

        # remove section data/meta
        if 0 <= row < len(self._sections):
            self._sections.pop(row)
        if 0 <= row < len(self._section_meta):
            self._section_meta.pop(row)

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

        self.tbl.removeRow(row)
        if log_msg:
            self._ok(log_msg)
        return True

    def _on_sections_table_context_menu(self, pos) -> None:
        if self.tbl is None:
            return
        idx = self.tbl.indexAt(pos)
        if not idx.isValid():
            item = self.tbl.itemAt(pos)
            if item is None:
                return
            row = int(item.row())
        else:
            row = int(idx.row())
        if row < 0 or row >= self.tbl.rowCount():
            return
        self.tbl.selectRow(row)
        menu = QMenu(self.tbl)
        act_delete = menu.addAction("Delete")
        chosen = menu.exec_(self.tbl.viewport().mapToGlobal(pos))
        if chosen is act_delete:
            self._delete_section_row(row, log_msg=f"Deleted section #{row + 1}.")

    def _on_sections_table_header_context_menu(self, pos) -> None:
        if self.tbl is None:
            return
        row = int(self.tbl.rowAt(pos.y()))
        if row < 0 or row >= self.tbl.rowCount():
            return
        self.tbl.selectRow(row)
        menu = QMenu(self.tbl)
        act_delete = menu.addAction("Delete")
        chosen = menu.exec_(self.tbl.verticalHeader().viewport().mapToGlobal(pos))
        if chosen is act_delete:
            self._delete_section_row(row, log_msg=f"Deleted section #{row + 1}.")

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
            if line_item is not None:
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
        self.tbl.setRowCount(0)
        self._sections.clear()
        self._section_meta.clear()
        for it in getattr(self, "_section_lines", []):
            if it is not None:
                self.viewer.scene.removeItem(it)
        self._section_lines.clear()
        for it in getattr(self, "_section_line_labels", []):
            if it is not None:
                self.viewer.scene.removeItem(it)
        self._section_line_labels.clear()
        if getattr(self, "_preview_line", None) is not None:
            self.viewer.scene.removeItem(self._preview_line)
            self._preview_line = None
        if getattr(self, "_preview_label", None) is not None:
            self.viewer.scene.removeItem(self._preview_label)
            self._preview_label = None

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
        r = self.tbl.currentRow()
        if r < 0 or self._inv_tr is None:
            self._info("Select a row to preview.")
            return

        # Xoá preview cũ nếu có
        if self._preview_line is not None:
            self.viewer.scene.removeItem(self._preview_line)
            self._preview_line = None
        if self._preview_label is not None:
            self.viewer.scene.removeItem(self._preview_label)
            self._preview_label = None

        p0 = tuple(map(float, self.tbl.item(r, 1).text().split(",")))
        p1 = tuple(map(float, self.tbl.item(r, 2).text().split(",")))
        c0, r0 = self._inv_tr * p0
        c1, r1 = self._inv_tr * p1

        pen = QPen(QColor(200, 30, 30, 220))
        pen.setCosmetic(True)
        pen.setWidth(2)
        item = QGraphicsLineItem(c0, r0, c1, r1)
        item.setPen(pen)
        item.setZValue(4)  # cao hơn line xanh
        self.viewer.scene.addItem(item)
        self._preview_line = item
        label_item = self.tbl.item(r, 0)
        self._preview_label = self._add_line_label(
            label_item.text() if label_item else str(r + 1),
            c0,
            r0,
            c1,
            r1,
            z=6,
        )

        self._ok("Preview line drawn.")

    def _clear_sections_state(self) -> None:
        if hasattr(self, "tbl"):
            self.tbl.setRowCount(0)
        self._sections.clear()
        self._section_meta.clear()

        for it in getattr(self, "_section_lines", []):
            if it is not None:
                self.viewer.scene.removeItem(it)
        self._section_lines.clear()

        for it in getattr(self, "_section_line_labels", []):
            if it is not None:
                self.viewer.scene.removeItem(it)
        self._section_line_labels.clear()

        if getattr(self, "_preview_line", None) is not None:
            self.viewer.scene.removeItem(self._preview_line)
            self._preview_line = None
        if getattr(self, "_preview_label", None) is not None:
            self.viewer.scene.removeItem(self._preview_label)
            self._preview_label = None

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

    def _next_line_id_for_role(self, line_role: str, exclude_row: int = -1) -> str:
        role = self._normalize_line_role(line_role, "")
        prefix = "CL" if role == "cross" else "ML"
        used = set()
        for r in range(self.tbl.rowCount()):
            if r == exclude_row:
                continue
            cand = ""
            if 0 <= r < len(self._section_meta) and isinstance(self._section_meta[r], dict):
                cand = str(self._section_meta[r].get("line_id", "") or "").strip()
            if not cand:
                item = self.tbl.item(r, 0)
                cand = item.text().strip() if item else ""
            parsed_role, parsed_idx = self._parse_auto_line_id(cand)
            if parsed_role == role and parsed_idx > 0:
                used.add(parsed_idx)
        n = 1
        while n in used:
            n += 1
        return f"{prefix}{n}"

    def _find_role_combo_row(self, combo: QComboBox) -> int:
        for r in range(self.tbl.rowCount()):
            if self.tbl.cellWidget(r, 3) is combo:
                return r
        return -1

    def _on_role_combo_changed(self, _index: int) -> None:
        combo = self.sender()
        if not isinstance(combo, QComboBox):
            return
        row = self._find_role_combo_row(combo)
        if row < 0:
            return
        while len(self._section_meta) <= row:
            self._section_meta.append({})
        meta = self._section_meta[row]
        if not isinstance(meta, dict):
            meta = {}
            self._section_meta[row] = meta
        label = (self.tbl.item(row, 0).text().strip() if self.tbl.item(row, 0) else f"{row + 1}")
        cur_line_id = str(meta.get("line_id", "") or "").strip() or label
        new_role = self._role_value_from_combo_text(combo.currentText())
        auto_role, auto_idx = self._parse_auto_line_id(cur_line_id)
        if (not cur_line_id) or (auto_idx > 0 and auto_role in ("main", "cross")):
            new_line_id = self._next_line_id_for_role(new_role, exclude_row=row)
            if new_line_id != cur_line_id:
                item0 = self.tbl.item(row, 0)
                if item0 is None:
                    item0 = QTableWidgetItem(new_line_id)
                    self.tbl.setItem(row, 0, item0)
                else:
                    item0.setText(new_line_id)
                cur_line_id = new_line_id
        meta["line_id"] = cur_line_id
        meta["line_role"] = new_role
        self._ok(f"[UI2] Line role set: row {row + 1} -> {meta['line_role']}")

    def _get_row_line_role(self, row: int) -> str:
        combo = self.tbl.cellWidget(row, 3) if self.tbl is not None else None
        if isinstance(combo, QComboBox):
            return self._role_value_from_combo_text(combo.currentText())
        meta = self._section_meta[row] if 0 <= row < len(self._section_meta) else {}
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
        for r in range(self.tbl.rowCount()):
            try:
                p0 = tuple(map(float, self.tbl.item(r, 1).text().split(",")))
                p1 = tuple(map(float, self.tbl.item(r, 2).text().split(",")))
            except Exception:
                continue
            meta = self._section_meta[r] if 0 <= r < len(self._section_meta) else {}
            line_id = str((meta or {}).get("line_id", "")).strip()
            label = (self.tbl.item(r, 0).text().strip() if self.tbl.item(r, 0) else f"{r + 1}")
            if not line_id:
                line_id = label
                if 0 <= r < len(self._section_meta) and isinstance(self._section_meta[r], dict):
                    self._section_meta[r]["line_id"] = line_id
            line_role = self._normalize_line_role(self._get_row_line_role(r), line_id)
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
            for r in range(self.tbl.rowCount()):
                p0 = tuple(map(float, self.tbl.item(r, 1).text().split(",")))
                p1 = tuple(map(float, self.tbl.item(r, 2).text().split(",")))
                meta = self._section_meta[r] if 0 <= r < len(self._section_meta) else {}
                line_id = str((meta or {}).get("line_id", "")).strip()
                if not line_id:
                    label = (self.tbl.item(r, 0).text().strip() if self.tbl.item(r, 0) else f"{r + 1}")
                    line_id = label
                    if 0 <= r < len(self._section_meta) and isinstance(self._section_meta[r], dict):
                        self._section_meta[r]["line_id"] = line_id
                line_role = self._normalize_line_role(self._get_row_line_role(r), line_id)
                if 0 <= r < len(self._section_meta) and isinstance(self._section_meta[r], dict):
                    self._section_meta[r]["line_role"] = line_role
                rows_to_save.append(storage_canonical_section_csv_row(
                    r + 1,
                    p0,
                    p1,
                    line_id=line_id,
                    line_role=line_role,
                ))
            self._backend.save_sections(run_dir, rows_to_save)

            self._ok(f"Sections saved: {csv_path}")
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
        if event.key() == Qt.Key_Delete:
            sel = self.tbl.selectedIndexes()
            if not sel:
                return

            rows = sorted(set(i.row() for i in sel))

            # nếu chọn ít nhất 1 cột Start/End → coi như xóa cả dòng
            if len(rows) >= 1:
                for r in reversed(rows):
                    # xoá line trên map
                    if 0 <= r < len(self._section_lines):
                        it = self._section_lines[r]
                        if it is not None:
                            self.viewer.scene.removeItem(it)
                        self._section_lines.pop(r)

                    if 0 <= r < len(self._section_line_labels):
                        it = self._section_line_labels[r]
                        if it is not None:
                            self.viewer.scene.removeItem(it)
                        self._section_line_labels.pop(r)

                    if 0 <= r < len(self._sections):
                        self._sections.pop(r)
                    if 0 <= r < len(self._section_meta):
                        self._section_meta.pop(r)

                    self.tbl.removeRow(r)

                if self._preview_line is not None:
                    self.viewer.scene.removeItem(self._preview_line)
                    self._preview_line = None
                if self._preview_label is not None:
                    self.viewer.scene.removeItem(self._preview_label)
                    self._preview_label = None

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
    def _append_status(self, text: str) -> None:
        self.status_text.appendPlainText(text)
        self.status_text.moveCursor(self.status_text.textCursor().End)

    def _info(self, msg: str) -> None:
        self._append_status(f"[UI2] INFO: {msg}")

    def _ok(self, msg: str) -> None:
        self._append_status(f"[UI2] OK: {msg}")

    def _err(self, msg: str) -> None:
        """Append error to status + popup."""
        self._append_status(f"[UI2] ERROR: {msg}")
        try:
            QMessageBox.critical(self, "Section Selection", msg)
        except Exception:
            pass  # phòng trường hợp gọi khi app chưa sẵn sàng

    # Back-compat: vài nơi cũ còn gọi self._log(...)
    def _log(self, msg: str) -> None:
        self._info(msg)
