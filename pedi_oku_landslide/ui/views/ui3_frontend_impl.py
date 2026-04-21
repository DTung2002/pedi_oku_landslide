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
from pedi_oku_landslide.pipeline.runners.ui3_backend import UI3BackendService
from pedi_oku_landslide.ui.scenes.ui3_preview_scene import ZoomableGraphicsView
from pedi_oku_landslide.ui.controllers.ui3_preview_controller import UI3PreviewControllerMixin
from pedi_oku_landslide.ui.controllers.ui3_group_panel import UI3GroupPanelMixin
from pedi_oku_landslide.ui.controllers.ui3_curve_panel import UI3CurvePanelMixin
from pedi_oku_landslide.ui.controllers.ui3_line_controller import UI3LineControllerMixin, WORKFLOW_GROUPING_PARAMS
from pedi_oku_landslide.ui.widgets.ui3_widgets import KeyboardOnlyDoubleSpinBox, KeyboardOnlySpinBox, NoWheelComboBox
from pedi_oku_landslide.ui.layout_constants import (
    LEFT_DEFAULT_W,
    LEFT_MARGINS,
    LEFT_MIN_W,
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
        self._left_min_w = LEFT_MIN_W
        self._left_default_w = LEFT_DEFAULT_W
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
        self.table_selector_combo = None
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
        self._active_global_fit_result: Optional[Dict[str, Any]] = None
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
        self._slope_table_updating: bool = False
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
        root.setContentsMargins(*ROOT_MARGINS)
        root.setSpacing(ROOT_SPACING)

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
        left.setContentsMargins(*LEFT_MARGINS)
        left.setSpacing(PANEL_SPACING)
        left.setAlignment(Qt.AlignTop)
        left_scroll.setWidget(left_container)

        # Project info – giống Section tab
        box_proj = QGroupBox("Project")
        box_proj.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lp = QGridLayout(box_proj)
        lp.setContentsMargins(*PROJECT_MARGINS)
        lp.setHorizontalSpacing(PROJECT_H_SPACING)
        lp.setVerticalSpacing(PROJECT_V_SPACING)
        lp.setColumnStretch(1, 1)
        lbl_name = QLabel("Name:")
        lbl_run = QLabel("Run label:")
        lbl_name.setFixedWidth(PROJECT_LABEL_W)
        lbl_run.setFixedWidth(PROJECT_LABEL_W)
        lp.setColumnMinimumWidth(0, PROJECT_LABEL_W)
        self.edit_project = QLineEdit()
        self.edit_project.setPlaceholderText("—")
        self.edit_project.setReadOnly(True)
        self.edit_project.setFixedHeight(CONTROL_HEIGHT)
        self.edit_project.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.edit_runlabel = QLineEdit()
        self.edit_runlabel.setPlaceholderText("—")
        self.edit_runlabel.setReadOnly(True)
        self.edit_runlabel.setFixedHeight(CONTROL_HEIGHT)
        self.edit_runlabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lp.addWidget(lbl_name, 0, 0)
        lp.addWidget(self.edit_project, 0, 1)
        lp.addWidget(lbl_run, 1, 0)
        lp.addWidget(self.edit_runlabel, 1, 1)

        left.addWidget(box_proj)

        # Sections + Advanced display
        box_sel = QGroupBox("Sections Display")
        box_sel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lsd = QVBoxLayout(box_sel)
        ls = QHBoxLayout()
        self.line_combo = NoWheelComboBox()
        self.line_combo.currentIndexChanged.connect(self._on_line_changed)
        btn_render = QPushButton("Render Section")
        btn_render.clicked.connect(self._render_current_safe)
        self.btn_section_advanced = QPushButton("Display ▸")
        self.btn_section_advanced.setCheckable(True)
        self.btn_section_advanced.toggled.connect(self._on_sections_advanced_toggled)
        ls.addWidget(self.line_combo)
        ls.addWidget(btn_render)
        ls.addWidget(self.btn_section_advanced)
        lsd.addLayout(ls)

        # Advanced controls
        self.section_advanced_widget = QWidget()
        self.section_advanced_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        adv_layout = QVBoxLayout(self.section_advanced_widget)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.setSpacing(6)

        def _fit_adv_label(text: str) -> QLabel:
            lb = QLabel(text)
            min_w = max(70, lb.fontMetrics().horizontalAdvance(text) + 8)
            lb.setFixedWidth(min_w)
            return lb

        row_step = QHBoxLayout()
        lbl_step = _fit_adv_label("Step (m):")
        row_step.addWidget(lbl_step)
        self.step_box = KeyboardOnlyDoubleSpinBox()
        self.step_box.setDecimals(4)
        self.step_box.setValue(float(self._default_profile_step_m))
        self.step_box.setMaximum(1e6)
        self.step_box.setMinimumWidth(56)
        self.step_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_step.addWidget(self.step_box, 1)
        adv_layout.addLayout(row_step)

        row_scale = QHBoxLayout()
        lbl_scale = _fit_adv_label("Scale:")
        row_scale.addWidget(lbl_scale)
        self.vscale = KeyboardOnlyDoubleSpinBox()
        self.vscale.setDecimals(3)
        self.vscale.setValue(0.1)
        self.vscale.setMaximum(1e6)
        self.vscale.setMinimumWidth(56)
        self.vscale.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_scale.addWidget(self.vscale, 1)
        adv_layout.addLayout(row_scale)

        row_width = QHBoxLayout()
        lbl_width = _fit_adv_label("Width:")
        row_width.addWidget(lbl_width)
        self.vwidth = KeyboardOnlyDoubleSpinBox()
        self.vwidth.setDecimals(4)
        self.vwidth.setValue(0.0015)
        self.vwidth.setMaximum(1.0)
        self.vwidth.setMinimumWidth(56)
        self.vwidth.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_width.addWidget(self.vwidth, 1)
        adv_layout.addLayout(row_width)
        self.section_advanced_widget.hide()
        lsd.addWidget(self.section_advanced_widget)

        left.addWidget(box_sel)

        def _set_table_visible_rows(tbl: QTableWidget, rows: int, row_h: int, fixed_h: Optional[int] = None) -> None:
            """Fix table height to show exactly `rows` data rows (+ header)."""
            tbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            tbl.verticalHeader().setDefaultSectionSize(int(row_h))
            tbl.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
            header_h = int(tbl.horizontalHeader().sizeHint().height())
            frame_h = int(tbl.frameWidth()) * 2
            total_h = int(fixed_h) if fixed_h is not None else header_h + frame_h + int(rows) * int(row_h) + 2
            tbl.setMinimumHeight(total_h)
            tbl.setMaximumHeight(total_h)

        table_visible_rows = 4
        table_row_h = 30
        table_fixed_h = 287

        # Group table
        box_grp = QGroupBox("Group")
        box_grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.box_group = box_grp
        lg = QVBoxLayout(box_grp)
        row_curv = QHBoxLayout()
        row_curv.addWidget(QLabel("Table:"))
        self.table_selector_combo = NoWheelComboBox()
        self.table_selector_combo.addItem("Group", "group")
        self.table_selector_combo.addItem("Boring holes", "boring")
        self.table_selector_combo.addItem("Nurbs control points", "nurbs")
        self.table_selector_combo.currentIndexChanged.connect(self._on_ui3_table_selector_changed)
        self.table_selector_combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.table_selector_combo.setFixedWidth(self.table_selector_combo.sizeHint().width())
        row_curv.addWidget(self.table_selector_combo)
        row_curv.addStretch(1)
        row_curv.addSpacing(8)
        self.rdp_eps_label = QLabel("RDP eps (m):")
        row_curv.addWidget(self.rdp_eps_label)
        self.rdp_eps_spin = KeyboardOnlyDoubleSpinBox()
        self.rdp_eps_spin.setDecimals(3)
        self.rdp_eps_spin.setRange(0.0, 1000.0)
        self.rdp_eps_spin.setSingleStep(0.1)
        self.rdp_eps_spin.setValue(float(WORKFLOW_GROUPING_PARAMS.get("rdp_eps_m", 0.5)))
        self.rdp_eps_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        row_curv.addWidget(self.rdp_eps_spin)
        lg.addLayout(row_curv)

        self.group_table = QTableWidget(0, 5)
        self.group_table.setHorizontalHeaderLabels(
            ["Group ID", "Start (m)", "End (m)", "Theta (deg)", "Color"]
        )
        self.group_table.verticalHeader().setVisible(False)
        self.group_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.group_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.group_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.group_table.cellDoubleClicked.connect(self._on_group_cell_double_clicked)
        self.group_table.itemChanged.connect(self._on_group_table_item_changed)
        _set_table_visible_rows(self.group_table, rows=table_visible_rows, row_h=table_row_h, fixed_h=table_fixed_h)
        lg.addWidget(self.group_table)

        self.slope_table = QTableWidget(2, 2)
        self.slope_table.setHorizontalHeaderLabels(["Distance (m)", "Slope (deg)"])
        self.slope_table.setVerticalHeaderLabels(["Start", "End"])
        self.slope_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.slope_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.slope_table.verticalHeader().setDefaultSectionSize(table_row_h)
        for r in range(2):
            for c in range(2):
                self.slope_table.setItem(r, c, QTableWidgetItem(""))
        self.slope_table.itemChanged.connect(self._on_slope_table_item_changed)
        _set_table_visible_rows(self.slope_table, rows=table_visible_rows, row_h=table_row_h, fixed_h=table_fixed_h)
        self.slope_table.hide()
        lg.addWidget(self.slope_table)

        rowg = QHBoxLayout()
        self.btn_add_g = QPushButton("Add")
        self.btn_add_g.clicked.connect(self._on_add_group)
        self.btn_del_g = QPushButton("Delete")
        self.btn_del_g.clicked.connect(self._on_delete_group)
        self.btn_draw_curve = QPushButton("Draw Curve")
        self.btn_draw_curve.clicked.connect(self._on_draw_curve)
        self.btn_auto_group = QPushButton("Auto Group")
        self.btn_auto_group.clicked.connect(self._on_auto_group)

        for btn in (self.btn_auto_group, self.btn_add_g, self.btn_del_g, self.btn_draw_curve):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            rowg.addWidget(btn, 1)
        lg.addLayout(rowg)
        self._group_table_widgets = [
            self.group_table,
            self.btn_auto_group,
            self.btn_add_g,
            self.btn_del_g,
            self.btn_draw_curve,
        ]
        self._slope_table_widgets = [self.slope_table]
        box_bh = QWidget()
        box_bh.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.box_boring_holes = box_bh
        lbh = QVBoxLayout(box_bh)
        lbh.setContentsMargins(0, 0, 0, 0)
        lbh.setSpacing(6)
        self.boring_table = QTableWidget(0, 4)
        self.boring_table.setHorizontalHeaderLabels(["BH", "X", "Y", "Z"])
        self.boring_table.verticalHeader().setVisible(False)
        self.boring_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.boring_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.boring_table.itemChanged.connect(self._on_boring_holes_table_item_changed)
        _set_table_visible_rows(self.boring_table, rows=table_visible_rows, row_h=table_row_h, fixed_h=table_fixed_h)
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
        lg.addWidget(box_bh)

        box_nurbs = QWidget()
        box_nurbs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.box_nurbs_control_points = box_nurbs
        ln = QVBoxLayout(box_nurbs)
        ln.setContentsMargins(0, 0, 0, 0)
        ln.setSpacing(6)

        self.nurbs_cp_spin = KeyboardOnlySpinBox()
        self.nurbs_cp_spin.setRange(4, 200)
        self.nurbs_cp_spin.setValue(4)
        self.nurbs_cp_spin.hide()
        self.nurbs_deg_spin = KeyboardOnlySpinBox()
        self.nurbs_deg_spin.setRange(3, 3)
        self.nurbs_deg_spin.setValue(3)
        self.nurbs_deg_spin.hide()
        self.nurbs_seed_method_combo = NoWheelComboBox()
        self.nurbs_seed_method_combo.addItem("Bezier-like", "bezier_like")
        self.nurbs_seed_method_combo.addItem("Slope-guided", "slope_guided")
        self.nurbs_seed_method_combo.hide()
        self.nurbs_table = QTableWidget(0, 4)
        self.nurbs_table.setHorizontalHeaderLabels(["CP", "Chainage (m)", "Elev (m)", "Weight"])
        self.nurbs_table.verticalHeader().setVisible(False)
        self.nurbs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.nurbs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        _set_table_visible_rows(self.nurbs_table, rows=table_visible_rows, row_h=table_row_h, fixed_h=table_fixed_h)
        ln.addWidget(self.nurbs_table)

        row_nurbs_btn = QHBoxLayout()
        self.btn_add_nurbs_cp = QPushButton("Add")
        self.btn_add_nurbs_cp.clicked.connect(self._on_add_nurbs_control_point)
        self.btn_del_nurbs_cp = QPushButton("Delete")
        self.btn_del_nurbs_cp.clicked.connect(self._on_delete_nurbs_control_point)
        self.btn_convert_nurbs = QPushButton("Convert")
        self.btn_convert_nurbs.setEnabled(False)
        self.btn_convert_nurbs.clicked.connect(self._on_convert_to_nurbs)
        for btn in (self.btn_add_nurbs_cp, self.btn_del_nurbs_cp, self.btn_convert_nurbs):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row_nurbs_btn.addWidget(btn, 1)
        ln.addLayout(row_nurbs_btn)

        self.btn_nurbs_save = QPushButton("Save")
        self.btn_nurbs_save.hide()
        self.btn_nurbs_load = QPushButton("Load NURBS")
        self.btn_nurbs_load.hide()
        self.btn_nurbs_reset = QPushButton("Reset NURBS")
        self.btn_nurbs_reset.hide()
        lg.addWidget(box_nurbs)
        left.addWidget(box_grp, 0)
        self._on_ui3_table_selector_changed(0)

        # Status
        box_st = QGroupBox("Status")
        box_st.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ls = QVBoxLayout(box_st)
        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setFixedHeight(200)
        self.status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ls.addWidget(self.status)
        left.addWidget(box_st, 0)

        # ===== RIGHT: preview (zoomable like AnalyzeTab) =====
        right_container = QWidget()
        right_container.setMinimumWidth(RIGHT_MIN_W)
        right_wrap = QVBoxLayout(right_container)
        right_wrap.setContentsMargins(*RIGHT_MARGINS)
        right_wrap.setSpacing(PANEL_SPACING)

        self._profile_cursor_label = QLabel("Cursor: —")
        self._profile_cursor_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        cursor_row = QHBoxLayout()
        cursor_row.setContentsMargins(0, 0, 0, 0)
        cursor_row.addWidget(self._profile_cursor_label, 1)
        self.btn_profile_fit = QPushButton("Fit")
        self.btn_profile_fit.setFixedSize(PREVIEW_FIT_BUTTON_W, PREVIEW_FIT_BUTTON_H)
        self.btn_profile_fit.clicked.connect(lambda: self.view.fit_to_scene())
        cursor_row.addWidget(self.btn_profile_fit, 0)
        right_wrap.addLayout(cursor_row)

        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setMinimumHeight(PREVIEW_MIN_H)
        self.view.setStyleSheet(PREVIEW_VIEWPORT_STYLE)
        self.view.sceneMouseMoved.connect(self._on_profile_scene_mouse_moved)
        self.view.hoverExited.connect(self._clear_profile_cursor_readout)
        right_wrap.addWidget(self.view, 1)

        splitter.addWidget(right_container)

        # Tỉ lệ ban đầu giữa panel trái/phải
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([self._left_default_w, 900])

        self._apply_button_style()

    def _on_sections_advanced_toggled(self, checked: bool) -> None:
        widget = getattr(self, "section_advanced_widget", None)
        if widget is not None:
            widget.setVisible(bool(checked))
        btn = getattr(self, "btn_section_advanced", None)
        if btn is not None:
            btn.setText("Display ▾" if checked else "Display ▸")

    def _on_ui3_table_selector_changed(self, _idx: int) -> None:
        combo = getattr(self, "table_selector_combo", None)
        selected = combo.currentData() if combo is not None else "group"
        selected = str(selected or "group")
        is_cross = False
        try:
            is_cross = self._current_ui2_line_role() == "cross"
        except Exception:
            is_cross = False
        if is_cross and selected == "group":
            selected = "slope"

        titles = {
            "group": "Group",
            "slope": "Slope",
            "boring": "Boring Holes",
            "nurbs": "NURBS Control Points",
        }
        box_group = getattr(self, "box_group", None)
        if box_group is not None:
            box_group.setTitle(titles.get(selected, "Group"))

        show_group = selected == "group"
        show_slope = selected == "slope"
        rdp_label = getattr(self, "rdp_eps_label", None)
        if rdp_label is not None:
            rdp_label.setVisible(show_group)
        rdp_spin = getattr(self, "rdp_eps_spin", None)
        if rdp_spin is not None:
            rdp_spin.setVisible(show_group)
        for widget in getattr(self, "_group_table_widgets", []) or []:
            widget.setVisible(show_group)
        for widget in getattr(self, "_slope_table_widgets", []) or []:
            widget.setVisible(show_slope)

        box_bh = getattr(self, "box_boring_holes", None)
        if box_bh is not None:
            box_bh.setVisible(selected == "boring")

        box_nurbs = getattr(self, "box_nurbs_control_points", None)
        if box_nurbs is not None:
            box_nurbs.setVisible(selected == "nurbs")

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

