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


class UI3BuildMixin:
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
        self.table_selector_combo.addItem("NURBS", "nurbs")
        self.table_selector_combo.addItem("Boring Holes", "boring")
        self.table_selector_combo.currentIndexChanged.connect(self._on_ui3_table_selector_changed)
        self.table_selector_combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._update_ui3_table_selector_width()
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

    def _update_ui3_table_selector_width(self) -> None:
        combo = getattr(self, "table_selector_combo", None)
        if combo is None:
            return
        texts = [combo.itemText(i) for i in range(combo.count())]
        texts.extend(["Group", "NURBS", "Boring Holes"])
        fm = combo.fontMetrics()
        max_text_w = max((fm.horizontalAdvance(str(t)) for t in texts if str(t)), default=0)
        width = max(combo.sizeHint().width(), max_text_w + 56)
        combo.setMinimumWidth(int(width))
        combo.setFixedWidth(int(width))

    def _sync_ui3_table_selector_items_for_role(self, is_cross: bool) -> None:
        combo = getattr(self, "table_selector_combo", None)
        if combo is None:
            return
        desired = [("NURBS", "nurbs"), ("Boring Holes", "boring")] if is_cross else [
            ("Group", "group"),
            ("NURBS", "nurbs"),
            ("Boring Holes", "boring"),
        ]
        current_data = str(combo.currentData() or "")
        existing = [(combo.itemText(i), str(combo.itemData(i) or "")) for i in range(combo.count())]
        if existing != desired:
            old = combo.blockSignals(True)
            try:
                combo.clear()
                for text, data in desired:
                    combo.addItem(text, data)
                valid_data = [data for _, data in desired]
                next_data = current_data if current_data in valid_data else desired[0][1]
                idx = combo.findData(next_data)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            finally:
                combo.blockSignals(old)
        elif current_data not in [data for _, data in desired] and combo.count() > 0:
            combo.setCurrentIndex(0)
        self._update_ui3_table_selector_width()
        self._on_ui3_table_selector_changed(combo.currentIndex())

    def _on_ui3_table_selector_changed(self, _idx: int) -> None:
        combo = getattr(self, "table_selector_combo", None)
        selected = combo.currentData() if combo is not None else "group"
        selected = str(selected or "group")
        try:
            is_cross = self._current_ui2_line_role() == "cross"
        except Exception:
            is_cross = False
        titles = {
            "group": "Group",
            "boring": "Boring Holes",
            "nurbs": "NURBS",
        }
        box_group = getattr(self, "box_group", None)
        if box_group is not None:
            box_group.setTitle("Cross Line" if (is_cross and selected == "group") else titles.get(selected, "Group"))

        show_group = (selected == "group") and (not is_cross)
        rdp_label = getattr(self, "rdp_eps_label", None)
        if rdp_label is not None:
            rdp_label.setVisible(show_group)
        rdp_spin = getattr(self, "rdp_eps_spin", None)
        if rdp_spin is not None:
            rdp_spin.setVisible(show_group)
        for widget in getattr(self, "_group_table_widgets", []) or []:
            widget.setVisible(show_group)

        box_bh = getattr(self, "box_boring_holes", None)
        if box_bh is not None:
            box_bh.setVisible(selected == "boring")

        box_nurbs = getattr(self, "box_nurbs_control_points", None)
        if box_nurbs is not None:
            box_nurbs.setVisible(selected == "nurbs")
        btn_convert = getattr(self, "btn_convert_nurbs", None)
        if btn_convert is not None:
            btn_convert.setText("Draw" if (is_cross and selected == "nurbs") else "Convert")
            if is_cross and selected == "nurbs":
                btn_convert.setEnabled(True)
