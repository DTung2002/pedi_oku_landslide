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
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from pedi_oku_landslide.pipeline.runners.ui3_backend import UI3BackendService
from pedi_oku_landslide.ui.scenes.ui3_preview_scene import ZoomableGraphicsView
from pedi_oku_landslide.ui.controllers.ui3_preview_controller import UI3PreviewControllerMixin
from pedi_oku_landslide.ui.controllers.ui3_group_panel import UI3GroupPanelMixin
from pedi_oku_landslide.ui.controllers.ui3_curve_panel import UI3CurvePanelMixin
from pedi_oku_landslide.ui.controllers.ui3_line_controller import UI3LineControllerMixin, WORKFLOW_GROUPING_PARAMS
from pedi_oku_landslide.ui.widgets.ui3_widgets import KeyboardOnlyDoubleSpinBox, KeyboardOnlySpinBox, NoWheelComboBox

class CurveAnalyzeTab(
    UI3PreviewControllerMixin,
    UI3GroupPanelMixin,
    UI3CurvePanelMixin,
    UI3LineControllerMixin,
    QWidget,
):
    """
    UI3 (refactor): khung làm việc phân tích đường cong.
    - Nhận context từ Analyze/Section (project/run/run_dir)
    - Đọc danh sách sections từ UI2/sections.csv
    - Cho phép chọn line; hiển thị status; vẽ placeholder đồ thị (dz & slope)
    """

    # (Optional) khi bạn muốn phát tín hiệu đã lưu JSON v.v.
    curve_saved = pyqtSignal(str)  # emit path
    _GROUND_EXPORT_STEP_M = 0.2

    def __init__(self, base_dir: str):
        super().__init__()
        self.base_dir = base_dir
        self._backend = UI3BackendService(base_dir=base_dir)
        self._ctx: Dict[str, str] = {"project": "", "run_label": "", "run_dir": ""}
        self._splitter: Optional[QSplitter] = None
        self._left_scroll: Optional[QScrollArea] = None
        self._left_min_w = 380
        self._left_default_w = 490
        self._pending_init_splitter = True

        self._ax_top = None  # dict: {x_min,x_max,left_px,top_px,width_px,height_px}
        self._ax_bot = None

        # paths từ UI1/UI2
        self.dem_path = ""
        self.dem_path_raw = ""
        self.dem_path_smooth = ""
        self.ground_export_dem_path = ""
        self._default_profile_step_m = 0.20
        self.dx_path = ""
        self.dy_path = ""
        self.dz_path = ""
        self.lines_path = ""
        self.slip_path = ""
        self.profile_source_combo = None
        self.rdp_eps_spin = None

        # UI widgets chính (để dùng lại)
        self.line_combo = None
        self.status = None
        self.scene = None
        self.view = None
        self.group_table = None
        # --- state for grouping/guide overlays (phải ở CurveAnalyzeTab) ---
        self._px_per_m: Optional[float] = None  # pixels per meter
        self._sec_len_m: Optional[float] = None  # chiều dài tuyến (m)
        self._group_bounds: Dict[str, List[float]] = {}  # {line_id: [x_m ...]}
        self._guide_lines_top: List[QGraphicsLineItem] = []
        self._guide_lines_bot: List[QGraphicsLineItem] = []
        self._group_bands_bot: List[QGraphicsRectItem] = []
        self._img_ground: Optional[QGraphicsPixmapItem] = None
        self._img_rate0: Optional[QGraphicsPixmapItem] = None
        self._curve_method_by_line: Dict[str, str] = {}
        self._active_prof: Optional[dict] = None
        self._active_groups: List[dict] = []
        self._active_base_curve: Optional[dict] = None
        self._active_curve: Optional[dict] = None
        self._curve_overlay_item: Optional[QGraphicsPathItem] = None
        self._cp_overlay_items: List[Any] = []
        self._anchor_overlay_items: List[Any] = []
        self._ui2_intersections_cache: Optional[Dict[str, Any]] = None
        self._anchors_xyz_cache: Optional[Dict[str, Any]] = None
        self._boring_holes_data: Dict[str, Any] = {"version": 1, "distance_tolerance_m": 1.0, "items": []}
        self._nurbs_params_by_line: Dict[str, Dict[str, Any]] = {}
        self._nurbs_seed_method_by_line: Dict[str, str] = {}
        self._group_table_updating: bool = False
        self._boring_table_updating: bool = False
        self._nurbs_updating_ui: bool = False
        # True when background image already has a baked slip-curve (profile_*_nurbs.png).
        self._static_nurbs_bg_loaded: bool = False
        self._nurbs_live_timer = QTimer(self)
        self._nurbs_live_timer.setSingleShot(True)
        self._nurbs_live_timer.setInterval(30)
        self._nurbs_live_timer.timeout.connect(self._on_nurbs_live_tick)

        self._plot_x0_px = None  # ax_left_px trong PNG
        self._plot_w_px = None  # ax_width_px trong PNG
        self._x_min = None  # trục x (chainage) min trên hình
        self._x_max = None  # trục x (chainage) max trên hình
        self._profile_cursor_label = None

        self._build_ui()

    def _apply_button_style(self) -> None:
        """Áp style tab Curve Analyze (nền trắng + nút xanh, gồm cả QToolButton)."""
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

        QPushButton,
        QToolButton {
            background: #056832;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 6px 10px;
            font-weight: 600;
        }
        QPushButton:hover,
        QToolButton:hover {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #6fa34a, stop:1 #4f7a34
            );
        }
        QPushButton:pressed,
        QToolButton:pressed {
            background: #4f7a34;
        }
        QPushButton:disabled,
        QToolButton:disabled {
            background: #9dbb86;
            color: #eeeeee;
        }
        """
    # -------------------- UI --------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ===== BODY: dùng QSplitter để panel trái/phải kéo được =====
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        self._splitter = splitter
        splitter.splitterMoved.connect(lambda *_: self._enforce_left_pane_bounds())
        root.addWidget(splitter)

        # ===== LEFT: controls (scrollable) =====
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setMinimumWidth(self._left_min_w)
        self._left_scroll = left_scroll
        splitter.addWidget(left_scroll)

        left_container = QWidget()
        left_container.setMinimumWidth(self._left_min_w)
        left = QVBoxLayout(left_container)
        left.setContentsMargins(6, 6, 6, 6)
        left.setSpacing(8)
        left_scroll.setWidget(left_container)

        # Project info – giống Section tab
        box_proj = QGroupBox("Project")
        lp = QVBoxLayout(box_proj)
        proj_input_h = 30
        fm = self.fontMetrics()
        proj_label_w = max(
            fm.horizontalAdvance("Name:"),
            fm.horizontalAdvance("Run label:")
        ) + 15

        def _fit_proj_label(text: str) -> QLabel:
            lb = QLabel(text)
            lb.setFixedWidth(proj_label_w)
            return lb

        row_proj = QHBoxLayout()
        row_proj.addWidget(_fit_proj_label("Name:"))
        self.edit_project = QLineEdit()
        self.edit_project.setPlaceholderText("—")
        self.edit_project.setReadOnly(True)
        self.edit_project.setFixedHeight(proj_input_h)
        self.edit_project.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_proj.addWidget(self.edit_project, 1)
        row_proj.addSpacing(6)
        row_proj.addWidget(_fit_proj_label("Run label:"))
        self.edit_runlabel = QLineEdit()
        self.edit_runlabel.setPlaceholderText("—")
        self.edit_runlabel.setReadOnly(True)
        self.edit_runlabel.setFixedHeight(proj_input_h)
        self.edit_runlabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_proj.addWidget(self.edit_runlabel, 1)
        lp.addLayout(row_proj)

        left.addWidget(box_proj)

        # Sections + Advanced display
        box_sel = QGroupBox("Sections Display")
        lsd = QVBoxLayout(box_sel)
        ls = QHBoxLayout()
        self.line_combo = NoWheelComboBox()
        self.line_combo.currentIndexChanged.connect(self._on_line_changed)
        btn_render = QPushButton("Render Section")
        btn_render.clicked.connect(self._render_current_safe)
        ls.addWidget(self.line_combo)
        ls.addWidget(btn_render)
        lsd.addLayout(ls)

        # Advanced controls
        la = QHBoxLayout()
        la.setSpacing(6)

        def _fit_adv_label(text: str) -> QLabel:
            lb = QLabel(text)
            min_w = lb.fontMetrics().horizontalAdvance(text) + 8
            lb.setFixedWidth(min_w)
            return lb

        lbl_step = _fit_adv_label("Step (m):")
        la.addWidget(lbl_step)
        self.step_box = KeyboardOnlyDoubleSpinBox()
        self.step_box.setDecimals(4)
        self.step_box.setValue(float(self._default_profile_step_m))
        self.step_box.setMaximum(1e6)
        self.step_box.setMinimumWidth(56)
        self.step_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        la.addWidget(self.step_box, 1)
        lbl_scale = _fit_adv_label("Scale:")
        la.addWidget(lbl_scale)
        self.vscale = KeyboardOnlyDoubleSpinBox()
        self.vscale.setDecimals(3)
        self.vscale.setValue(0.1)
        self.vscale.setMaximum(1e6)
        self.vscale.setMinimumWidth(56)
        self.vscale.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        la.addWidget(self.vscale, 1)
        lbl_width = _fit_adv_label("Width:")
        la.addWidget(lbl_width)
        self.vwidth = KeyboardOnlyDoubleSpinBox()
        self.vwidth.setDecimals(4)
        self.vwidth.setValue(0.0015)
        self.vwidth.setMaximum(1.0)
        self.vwidth.setMinimumWidth(56)
        self.vwidth.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        la.addWidget(self.vwidth, 1)
        lsd.addLayout(la)

        left.addWidget(box_sel)

        def _set_table_visible_rows(tbl: QTableWidget, rows: int, row_h: int) -> None:
            """Fix table height to show exactly `rows` data rows (+ header)."""
            tbl.verticalHeader().setDefaultSectionSize(int(row_h))
            header_h = int(tbl.horizontalHeader().sizeHint().height())
            frame_h = int(tbl.frameWidth()) * 2
            total_h = header_h + frame_h + int(rows) * int(row_h) + 2
            tbl.setMinimumHeight(total_h)
            tbl.setMaximumHeight(total_h)

        # Group table
        box_grp = QGroupBox("Group")
        lg = QVBoxLayout(box_grp)
        row_curv = QHBoxLayout()
        row_curv.addWidget(QLabel("RDP eps (m):"))
        self.rdp_eps_spin = KeyboardOnlyDoubleSpinBox()
        self.rdp_eps_spin.setDecimals(3)
        self.rdp_eps_spin.setRange(0.0, 1000.0)
        self.rdp_eps_spin.setSingleStep(0.1)
        self.rdp_eps_spin.setValue(float(WORKFLOW_GROUPING_PARAMS.get("rdp_eps_m", 0.5)))
        self.rdp_eps_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        row_curv.addWidget(self.rdp_eps_spin)
        lg.addLayout(row_curv)

        self.group_table = QTableWidget(0, 4)
        self.group_table.setHorizontalHeaderLabels(
            ["Group ID", "Start (m)", "End (m)", "Color"]
        )
        self.group_table.verticalHeader().setVisible(False)
        self.group_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.group_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.group_table.cellDoubleClicked.connect(self._on_group_cell_double_clicked)
        self.group_table.itemChanged.connect(self._on_group_table_item_changed)
        _set_table_visible_rows(self.group_table, rows=4, row_h=30)
        lg.addWidget(self.group_table)

        rowg = QHBoxLayout()
        self.btn_add_g = QPushButton("Add")
        self.btn_add_g.clicked.connect(self._on_add_group)
        self.btn_del_g = QPushButton("Delete")
        self.btn_del_g.clicked.connect(self._on_delete_group)
        self.btn_draw_curve = QPushButton("Draw Curve")
        self.btn_draw_curve.clicked.connect(self._on_draw_curve)
        self.btn_auto_group = QPushButton("Auto Group")
        self.btn_auto_group.clicked.connect(self._on_auto_group)
        self.btn_load_group = QPushButton("Load Group")
        self.btn_load_group.clicked.connect(self._on_load_group_info)

        for btn in (self.btn_auto_group, self.btn_load_group, self.btn_add_g, self.btn_del_g, self.btn_draw_curve):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            rowg.addWidget(btn, 1)
        lg.addLayout(rowg)
        left.addWidget(box_grp, 1)

        box_bh = QGroupBox("Boring Holes")
        lbh = QVBoxLayout(box_bh)
        self.boring_table = QTableWidget(0, 4)
        self.boring_table.setHorizontalHeaderLabels(["BH", "X", "Y", "Z"])
        self.boring_table.verticalHeader().setVisible(False)
        self.boring_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.boring_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.boring_table.itemChanged.connect(self._on_boring_holes_table_item_changed)
        _set_table_visible_rows(self.boring_table, rows=3, row_h=30)
        lbh.addWidget(self.boring_table)

        row_bh_btn = QHBoxLayout()
        self.btn_add_bh = QPushButton("Add")
        self.btn_add_bh.clicked.connect(self._on_add_boring_hole)
        self.btn_del_bh = QPushButton("Delete")
        self.btn_del_bh.clicked.connect(self._on_delete_boring_hole)
        self.btn_save_bh = QPushButton("Save")
        self.btn_save_bh.clicked.connect(self._on_save_boring_holes)
        for btn in (self.btn_add_bh, self.btn_del_bh, self.btn_save_bh):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row_bh_btn.addWidget(btn, 1)
        lbh.addLayout(row_bh_btn)
        left.addWidget(box_bh, 0)

        # NURBS controls
        box_nurbs = QGroupBox("NURBS")
        ln = QVBoxLayout(box_nurbs)
        ln.setContentsMargins(8, 8, 8, 8)
        ln.setSpacing(6)

        row_cfg = QHBoxLayout()
        row_cfg.addWidget(QLabel("Control points:"))
        self.nurbs_cp_spin = KeyboardOnlySpinBox()
        self.nurbs_cp_spin.setRange(2, 20)
        self.nurbs_cp_spin.setValue(4)
        self.nurbs_cp_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        row_cfg.addWidget(self.nurbs_cp_spin)
        row_cfg.addSpacing(8)
        row_cfg.addWidget(QLabel("Degree:"))
        self.nurbs_deg_spin = KeyboardOnlySpinBox()
        self.nurbs_deg_spin.setRange(1, 10)
        self.nurbs_deg_spin.setValue(3)
        self.nurbs_deg_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        row_cfg.addWidget(self.nurbs_deg_spin)
        row_cfg.addSpacing(8)
        row_cfg.addWidget(QLabel("Seed:"))
        self.nurbs_seed_method_combo = NoWheelComboBox()
        self.nurbs_seed_method_combo.addItem("Bezier-like", "bezier_like")
        self.nurbs_seed_method_combo.addItem("Slope-guided", "slope_guided")
        row_cfg.addWidget(self.nurbs_seed_method_combo, 1)
        ln.addLayout(row_cfg)

        self.nurbs_table = QTableWidget(0, 4)
        self.nurbs_table.setHorizontalHeaderLabels(["CP", "Chainage (m)", "Elev (m)", "Weight"])
        self.nurbs_table.verticalHeader().setVisible(False)
        self.nurbs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.nurbs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        _set_table_visible_rows(self.nurbs_table, rows=4, row_h=34)
        ln.addWidget(self.nurbs_table)

        row_nurbs_btn = QHBoxLayout()
        self.btn_nurbs_load = QPushButton("Load NURBS")
        self.btn_nurbs_reset = QPushButton("Reset NURBS")
        self.btn_nurbs_save = QPushButton("Save")
        row_nurbs_btn.addWidget(self.btn_nurbs_load, 1)
        row_nurbs_btn.addWidget(self.btn_nurbs_reset, 1)
        row_nurbs_btn.addWidget(self.btn_nurbs_save, 1)
        ln.addLayout(row_nurbs_btn)

        self.nurbs_cp_spin.valueChanged.connect(self._on_nurbs_cp_spin_changed)
        self.nurbs_deg_spin.valueChanged.connect(self._on_nurbs_deg_spin_changed)
        self.nurbs_seed_method_combo.currentIndexChanged.connect(self._on_nurbs_seed_method_changed)
        self.btn_nurbs_load.clicked.connect(self._on_load_nurbs_info)
        self.btn_nurbs_reset.clicked.connect(self._on_nurbs_reset_defaults)
        self.btn_nurbs_save.clicked.connect(self._on_nurbs_save)

        left.addWidget(box_nurbs, 0)

        # Status
        box_st = QGroupBox("Status")
        ls = QVBoxLayout(box_st)
        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setMinimumHeight(170)
        self.status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ls.addWidget(self.status)
        left.addWidget(box_st, 1)

        # ===== RIGHT: preview (zoomable like AnalyzeTab) =====
        right_container = QWidget()
        right_wrap = QVBoxLayout(right_container)
        right_wrap.setContentsMargins(0, 0, 0, 0)

        zoombar = QToolBar()
        act_in = QAction("Zoom +", self)
        act_in.triggered.connect(lambda: self.view.zoom_in())
        act_out = QAction("Zoom –", self)
        act_out.triggered.connect(lambda: self.view.zoom_out())
        act_fit = QAction("Fit", self)
        act_fit.triggered.connect(lambda: self.view.fit_to_scene())
        act_100 = QAction("100%", self)
        act_100.triggered.connect(lambda: self.view.set_100())
        zoombar.addAction(act_in)
        zoombar.addAction(act_out)
        zoombar.addAction(act_fit)
        zoombar.addAction(act_100)
        zoombar.setIconSize(QSize(22, 22))
        zoombar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        zoombar.setContentsMargins(0, 0, 0, 0)
        zoombar.setStyleSheet("QToolBar { spacing: 6px; background: transparent; border: none; }")
        right_wrap.addWidget(zoombar)

        self._profile_cursor_label = QLabel("Cursor: —")
        self._profile_cursor_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        right_wrap.addWidget(self._profile_cursor_label)

        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.sceneMouseMoved.connect(self._on_profile_scene_mouse_moved)
        self.view.hoverExited.connect(self._clear_profile_cursor_readout)
        right_wrap.addWidget(self.view, 1)

        splitter.addWidget(right_container)

        # Tỉ lệ ban đầu giữa panel trái/phải
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([self._left_default_w, 900])

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

    # -------------------- Status helpers --------------------
    def _append_status(self, text: str) -> None:
        self.status.append(text)

    def _ok(self, msg: str) -> None:
        self._append_status(f"✅ {msg}")

    def _info(self, msg: str) -> None:
        self._append_status(f"[UI3] INFO: {msg}")

    def _warn(self, msg: str) -> None:
        self._append_status(f"⚠️ {msg}")

    def _err(self, msg: str) -> None:
        self._append_status(f"❌ {msg}")

    def _log(self, msg: str) -> None:
        self._append_status(msg)

