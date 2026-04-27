import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import LineString, Point
from PyQt5.QtCore import QPointF, Qt
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
    LEFT_MARGINS,
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


class UI2BuildMixin:
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
