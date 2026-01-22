# ui3_frontend.py
# ui3_frontend.py — CLEAN HEADER
import os, sys, datetime
from typing import Optional
import numpy as np
import traceback
# High-DPI auto scale (đặt trước khi import Qt)
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

# DLL path (Conda/OSGeo)
dll_dir = os.path.join(sys.prefix, "Library", "bin")
if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(dll_dir)
    except FileNotFoundError:
        pass
else:
    os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

# GDAL/PROJ data (Conda)
os.environ.setdefault("GDAL_DATA", os.path.join(sys.prefix, "Library", "share", "gdal"))
os.environ.setdefault("PROJ_LIB",  os.path.join(sys.prefix, "Library", "share", "proj"))
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QTextEdit, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QDoubleSpinBox, QComboBox, QHeaderView, QTableWidget, QTableWidgetItem,
    QGroupBox, QScrollArea, QFrame, QSizePolicy, QFormLayout
)

from shapely.ops import nearest_points   # thêm dòng này

from pipeline.runners.ui3_backend import (
    auto_paths, list_lines, compute_profile,
    render_profile_png, export_csv, auto_group_profile,
    estimate_slip_curve, fit_bezier_smooth_curve,
    clamp_groups_to_slip,
    fit_bezier_with_intersection      # <-- THÊM
)

class ToggleableGroup(QGroupBox):
    def __init__(self, title, inner_widget):
        super().__init__(title)
        self.setCheckable(True)
        self.setChecked(False)
        lay = QVBoxLayout(self)
        lay.addWidget(inner_widget)
        self.toggled.connect(lambda s: inner_widget.setVisible(s))
        inner_widget.setVisible(False)

class UI3App(QWidget):
    def __init__(self, json_path: Optional[str] = None):
        super().__init__()

        # style
        self.setWindowTitle("UI3 – Cross Section Analysis")
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                font-family: 'Public Sans', sans-serif;
                font-size: 11pt;          /* ↑ tăng base font (11–12pt là đẹp) */
                font-weight: 500;
                color: #000000;
            }
            /* Tiêu đề nhóm */
            QGroupBox::title { padding: 2px 4px; font-size: 12pt; font-weight: 600; }
            QLabel { margin: 0 0 2px 0; font-size: 12pt; }  /* nhãn hơi to hơn 1 chút */

            QPushButton {
                background-color: #000000; color: white; border: none;
                padding: 7px 12px;          /* ↑ thêm 1px cho phù hợp font to hơn */
                border-radius: 10px;
                font-size: 11.2pt;
            }
            QPushButton:hover { background-color: #222222; }

            QDoubleSpinBox, QComboBox, QTableWidget, QTextEdit {
                border: 1px solid #cccccc; border-radius: 8px;
                background-color: #ffffff; padding: 3px 8px;  /* ↑ padding ngang 8px */
                font-size: 11.2pt;
            }
        """)

        # === Workspace & paths (standardized to output/UI3/...) ===
        self.workspace   = os.path.join("output", "UI3")
        self.preview_dir = os.path.join(self.workspace, "preview")
        self.export_dir  = os.path.join(self.workspace, "exports")
        os.makedirs(self.preview_dir, exist_ok=True)
        os.makedirs(self.export_dir,  exist_ok=True)

        # log file inside output/UI3
        self.log_path = os.path.join(self.workspace, f"log_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")

        # ensure attributes exist
        self.dem_path = ""
        self.dx_path = ""
        self.dy_path = ""
        self.dz_path = ""
        self.lines_path = ""
        self.transform = None
        self.crs = None
        self.lines_gdf = None

        # auto paths
        paths = auto_paths()
        self.dem_path   = paths.get("dem", "")
        self.dx_path    = paths.get("dx", "")
        self.dy_path    = paths.get("dy", "")
        self.dz_path    = paths.get("dz", "")
        self.lines_path = paths.get("lines", "")
        self.slip_path = paths.get("slip", "")  # <-- thêm dòng này
        self._main_curves = {}  # {row_index: {"chain": np.ndarray, "z": np.ndarray, "geom": LineString}}
        self._line_roles = {}  # {row_index: "main"|"aux"}

        # allow JSON override
        if json_path and os.path.exists(json_path):
            try:
                import json
                with open(json_path, "r", encoding="utf-8") as f:
                    js = json.load(f)
                self.dem_path   = js.get("dem_ground_path", js.get("dem", self.dem_path))
                self.dx_path    = js.get("dx_path", self.dx_path)
                self.dy_path    = js.get("dy_path", self.dy_path)
                self.dz_path    = js.get("dz_path", self.dz_path)
                self.lines_path = js.get("lines_path", self.lines_path)
            except Exception as e:
                print(f"[UI3] Warn: cannot read json overrides: {e}")

        self._build_ui()
        self._auto_load_lines()

    # UI layout
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QHBoxLayout()
        title = QLabel("UI3 – Cross Section Analysis")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        header.addWidget(title); header.addStretch()
        btn_back = QPushButton("Back"); btn_back.clicked.connect(self.go_to_back)
        header.addWidget(btn_back)
        layout.addLayout(header)

        # Body
        body = QHBoxLayout(); layout.addLayout(body)
        # QScrollArea
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        body.addWidget(left_scroll, 3)
        left_host = QWidget()
        left_scroll.setWidget(left_host)
        left = QVBoxLayout(left_host)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(6)
        left.setAlignment(Qt.AlignTop)
        left.setContentsMargins(6, 6, 6, 6)
        left.setSpacing(4)  # 6 -> 4: khoảng cách dọc nhỏ lại
        left_scroll.setMinimumWidth(320)  # cột trái gọn ~320–380 px
        left_scroll.setMaximumWidth(380)

        # Select line
        from PyQt5.QtWidgets import QFormLayout, QGroupBox
        sel_box = QGroupBox("Line selection")
        sel_form = QFormLayout(sel_box)
        sel_form.setContentsMargins(6, 6, 6, 6)
        sel_form.setHorizontalSpacing(8)
        sel_form.setVerticalSpacing(4)
        self.line_combo = QComboBox()
        self.line_combo.currentIndexChanged.connect(self._on_line_changed)
        sel_form.addRow("Select line from UI2:", self.line_combo)
        left.addWidget(sel_box)

        # Render Section button
        self.btn_render = QPushButton("Render Section")
        self.btn_render.setToolTip("Render with settings in Advanced Display")
        self.btn_render.clicked.connect(self._render_current)
        left.addWidget(self.btn_render)

        # Advanced Display
        adv = QWidget(); adv_l = QVBoxLayout(adv); adv_l.setContentsMargins(0,0,0,0)

        adv_l.addWidget(QLabel("Sampling step (m):"))
        self.step_box = QDoubleSpinBox(); self.step_box.setDecimals(2); self.step_box.setMaximum(1e6); self.step_box.setValue(0.20)
        adv_l.addWidget(self.step_box)

        adv_l.addWidget(QLabel("Vector scale (quiver 'scale'):"))
        self.vscale_box = QDoubleSpinBox(); self.vscale_box.setDecimals(3); self.vscale_box.setMaximum(1e6); self.vscale_box.setValue(0.1)
        adv_l.addWidget(self.vscale_box)

        adv_l.addWidget(QLabel("Vector width:"))
        self.vwidth_box = QDoubleSpinBox(); self.vwidth_box.setDecimals(4); self.vwidth_box.setMaximum(1.0); self.vwidth_box.setValue(0.0015)
        adv_l.addWidget(self.vwidth_box)

        adv_l.addWidget(QLabel("Head length:"))
        self.hlen_box = QDoubleSpinBox(); self.hlen_box.setDecimals(1); self.hlen_box.setMaximum(1000); self.hlen_box.setValue(5.0)
        adv_l.addWidget(self.hlen_box)

        adv_l.addWidget(QLabel("Head width:"))
        self.hwid_box = QDoubleSpinBox(); self.hwid_box.setDecimals(1); self.hwid_box.setMaximum(1000); self.hwid_box.setValue(3.0)
        adv_l.addWidget(self.hwid_box)

        adv_l.addWidget(QLabel("Highlight θ (deg):"))
        self.hth_box = QDoubleSpinBox(); self.hth_box.setDecimals(1); self.hth_box.setMaximum(90); self.hth_box.setValue(10.0)
        adv_l.addWidget(self.hth_box)

        adv_l.addWidget(QLabel("X min / X max (chainage, m):"))
        xrow = QHBoxLayout()
        self.xmin_box = QDoubleSpinBox(); self.xmin_box.setDecimals(2); self.xmin_box.setMaximum(1e9); self.xmin_box.setValue(0.0)
        self.xmax_box = QDoubleSpinBox(); self.xmax_box.setDecimals(2); self.xmax_box.setMaximum(1e9); self.xmax_box.setValue(0.0)
        xrow.addWidget(self.xmin_box); xrow.addWidget(self.xmax_box)
        adv_l.addLayout(xrow)

        adv_l.addWidget(QLabel("Y min / Y max (elevation, m):"))
        yrow = QHBoxLayout()
        self.ymin_box = QDoubleSpinBox(); self.ymin_box.setDecimals(2); self.ymin_box.setMaximum(1e6); self.ymin_box.setValue(0.0)
        self.ymax_box = QDoubleSpinBox(); self.ymax_box.setDecimals(2); self.ymax_box.setMaximum(1e6); self.ymax_box.setValue(0.0)
        yrow.addWidget(self.ymin_box); yrow.addWidget(self.ymax_box)
        adv_l.addLayout(yrow)

        # render
        adv_scroll = QScrollArea()
        adv_scroll.setWidgetResizable(True)
        adv_scroll.setFrameShape(QFrame.NoFrame)
        adv_scroll.setMinimumHeight(200)
        adv_scroll.setMaximumHeight(360)
        adv_scroll.setWidget(adv)
        left.addWidget(ToggleableGroup("Advanced Display", adv_scroll))

        # Log
        stat_box = QGroupBox("Status")
        stat_lay = QVBoxLayout(stat_box);
        stat_lay.setContentsMargins(6, 6, 6, 6);
        stat_lay.setSpacing(4)
        self.status = QTextEdit();
        self.status.setReadOnly(True);
        self.status.setLineWrapMode(QTextEdit.NoWrap)
        self.status.setFixedHeight(96)
        stat_lay.addWidget(self.status)
        left.addWidget(stat_box)
        left.addStretch(1)
        # Preview pane
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        body.addWidget(self.view, 5)

        # Group Vectors
        grp_box = QGroupBox("Group Vectors")
        grp_lay = QVBoxLayout(grp_box)

        # Bảng: Num. | Group ID | Start | End | Color
        self.group_table = QTableWidget(0, 5)
        self.group_table.setHorizontalHeaderLabels(["Num.", "Group ID", "Start chainage", "End chainage", "Color"])
        self.group_table.verticalHeader().setVisible(False)
        self.group_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.group_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Num.
        self.group_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Color
        self.group_table.setMinimumHeight(180)

        # Palette màu
        self._color_palette = [
            ("Blue", "#1f77b4"), ("Orange", "#ff7f0e"), ("Green", "#2ca02c"),
            ("Red", "#d62728"), ("Purple", "#9467bd"), ("Brown", "#8c564b"),
            ("Pink", "#e377c2"), ("Gray", "#7f7f7f"), ("Olive", "#bcbd22"),
            ("Cyan", "#17becf"),
        ]


        # for _ in range(3):
        #     r = self.group_table.rowCount()
        #     self._insert_group_row(r, gid=f"G{r + 1}")
        # self._renumber_group_table()
        # self._refresh_all_color_combos()

        # nút
        row_btns = QHBoxLayout()
        self.btn_add_group = QPushButton("Add Group")
        self.btn_add_group.clicked.connect(self.on_add_group)
        self.btn_del_group = QPushButton("Delete Group")
        self.btn_del_group.clicked.connect(self.on_delete_group)
        self.btn_confirm_groups = QPushButton("Confirm Group Vector")
        self.btn_confirm_groups.clicked.connect(self.on_confirm_groups)
        row_btns.addWidget(self.btn_add_group)
        row_btns.addWidget(self.btn_del_group)
        row_btns.addStretch(1)
        row_btns.addWidget(self.btn_confirm_groups)

        grp_lay.addWidget(self.group_table)
        grp_lay.addLayout(row_btns)

        self.btn_draw_curve = QPushButton("Draw Slip Curve")
        self.btn_draw_curve.setToolTip("Draw Slip curve for slip zone")
        self.btn_draw_curve.clicked.connect(self.on_draw_slip_curve)
        row_btns.addWidget(self.btn_draw_curve)

        grp_scroll = QScrollArea()
        grp_scroll.setWidgetResizable(True)
        grp_scroll.setFrameShape(QFrame.NoFrame)
        grp_scroll.setWidget(grp_box)
        grp_scroll.setMinimumHeight(200)
        grp_scroll.setMaximumHeight(380)
        layout.addWidget(grp_scroll)

    #  Data loading
    def _auto_load_lines(self):
        try:
            labels, gdf, _ = list_lines(self.lines_path, self.dem_path)
            self.lines_gdf = gdf
            self.line_combo.clear(); self.line_combo.addItems(labels)
            # KHÔNG auto render ở đây nữa
            if labels:
                self._log(f"[i] Loaded {len(labels)} lines. Select a line and click 'Render Section'.")
            else:
                self._log("[!] No lines found.")
        except Exception as e:
            self._log(f"[!] Cannot load lines: {e}")

    # Actions
    def _on_line_changed(self, idx: int):
        # Không auto-render; chỉ ghi log để nhắc người dùng bấm nút
        self._classify_lines_main_aux()
        if idx < 0 or self.lines_gdf is None:
            return
        self._log(f"[i] Line selected: {self.line_combo.currentText()}. Click 'Render Section' to update preview.")
        self._load_groups_for_current_line()

    def _render_current(self):
        if self.lines_gdf is None or self.lines_gdf.empty:
            self._log("[!] No lines.")
            return
        row = self.line_combo.currentIndex()
        if row < 0:
            self._log("[!] Please select a line first.")
            return
        geom = self.lines_gdf.geometry.iloc[row]

        prof = compute_profile(
            self.dem_path, self.dx_path, self.dy_path, self.dz_path,
            geom, step_m=self.step_box.value(),
            smooth_win=11, smooth_poly=2,
            slip_mask_path=self.slip_path, slip_only=True
        )

        if not prof:
            self._log("[!] Empty profile.")
            return

        y_min = self.ymin_box.value() or None
        y_max = self.ymax_box.value() or None

        cmin = float(prof["chain"].min())
        cmax = float(prof["chain"].max())
        xmin_ui = self.xmin_box.value()
        xmax_ui = self.xmax_box.value()

        x_min = None
        x_max = None
        if (xmin_ui > 0.0) or (xmax_ui > 0.0):
            x_min = float(xmin_ui) if xmin_ui > 0.0 else cmin
            x_max = float(xmax_ui) if xmax_ui > 0.0 else cmax
            if x_max <= x_min:
                self._log("[!] X max must be > X min.")
                x_min = x_max = None

        # save preview to output/UI3/preview
        out_png = os.path.join(self.preview_dir, f"profile_{row+1:03d}.png")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        group_ranges = None
        try:
            import json
            gpath = self._groups_json_path()
            if os.path.exists(gpath):
                with open(gpath, "r", encoding="utf-8") as f:
                    group_ranges = json.load(f).get("groups", None)
        except Exception:
            group_ranges = None

        msg, path = render_profile_png(
            prof, out_png,
            y_min=y_min, y_max=y_max,
            x_min=x_min, x_max=x_max,
            vec_scale=self.vscale_box.value(),
            vec_width=self.vwidth_box.value(),
            head_len=self.hlen_box.value(),
            head_w=self.hwid_box.value(),
            highlight_theta=None,
            group_ranges=group_ranges  # <--- thêm
        )

        self._log(msg)
        self._show_png(path)

    def _export_current(self):
        if self.lines_gdf is None or self.lines_gdf.empty: return
        row = self.line_combo.currentIndex()
        geom = self.lines_gdf.geometry.iloc[row]
        prof = compute_profile(
            self.dem_path, self.dx_path, self.dy_path, self.dz_path,
            geom, step_m=self.step_box.value(),
            smooth_win=11, smooth_poly=2,
            slip_mask_path=self.slip_path, slip_only=True
        )

        if not prof:
            self._log("[!] Empty profile when exporting.")
            return

        out_dir = self.export_dir
        os.makedirs(out_dir, exist_ok=True)
        png_path = os.path.join(out_dir, f"profile_{row+1:03d}.png")
        csv_path = os.path.join(out_dir, f"profile_{row+1:03d}.csv")

        cmin = float(prof["chain"].min())
        cmax = float(prof["chain"].max())
        xmin_ui = self.xmin_box.value()
        xmax_ui = self.xmax_box.value()

        x_min = None
        x_max = None
        if (xmin_ui > 0.0) or (xmax_ui > 0.0):
            x_min = float(xmin_ui) if xmin_ui > 0.0 else cmin
            x_max = float(xmax_ui) if xmax_ui > 0.0 else cmax
            if x_max <= x_min:
                self._log("[!] X max phải lớn hơn X min – export sẽ dùng auto X.")
                x_min = x_max = None

        msg_png, _ = render_profile_png(
            prof, png_path,
            y_min=self.ymin_box.value() or None,
            y_max=self.ymax_box.value() or None,
            x_min=x_min, x_max=x_max,
            vec_scale=self.vscale_box.value(),
            vec_width=self.vwidth_box.value(),
            head_len=self.hlen_box.value(),
            head_w=self.hwid_box.value(),
            highlight_theta=self.hth_box.value()
        )
        msg_csv, _ = export_csv(prof, csv_path)

        self._log(msg_png); self._log(msg_csv)

        # Nếu không dùng bảng liệt kê exports, bỏ khối try/except này
        try:
            i = self.table.rowCount(); self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(self.line_combo.currentText()))
            self.table.setItem(i, 1, QTableWidgetItem(csv_path))
            self.table.setItem(i, 2, QTableWidgetItem(png_path))
        except Exception:
            pass

    # Helpers
    # ==== GROUP HELPERS ====
    def _renumber_group_table(self):
        rc = self.group_table.rowCount()
        for r in range(rc):
            num_item = QTableWidgetItem(str(r + 1))
            num_item.setFlags(num_item.flags() & ~Qt.ItemIsEditable)  # khóa edit
            num_item.setTextAlignment(Qt.AlignCenter)
            self.group_table.setItem(r, 0, num_item)

    def _ensure_palette_size(self, needed: int):
        # sinh thêm màu nếu cần (golden ratio)
        while len(self._color_palette) < needed:
            i = len(self._color_palette)
            hue = (i * 0.61803398875) % 1.0
            c = QColor.fromHsvF(hue, 0.7, 0.9)
            self._color_palette.append((f"Color {i + 1}", c.name()))

    def _used_colors(self) -> set:
        used = set()
        rc = self.group_table.rowCount()
        for r in range(rc):
            w = self.group_table.cellWidget(r, 4)
            if isinstance(w, QComboBox):
                val = w.currentData()
                if val: used.add(val)
        return used

    def _create_color_combo(self, preselect_hex=None) -> QComboBox:
        cb = QComboBox()
        for name, hexv in self._color_palette:
            cb.addItem(name, hexv)
            idx = cb.count() - 1
            cb.setItemData(idx, QColor(hexv), Qt.BackgroundRole)
        if preselect_hex:
            idx = next((i for i in range(cb.count()) if cb.itemData(i) == preselect_hex), -1)
            if idx >= 0: cb.setCurrentIndex(idx)
        cb.currentIndexChanged.connect(self._on_color_changed)
        return cb

    def _refresh_all_color_combos(self):
        """Giữ màu duy nhất: mỗi combo chỉ hiện màu chưa dùng (hoặc chính màu nó đang chọn)."""
        used = self._used_colors()
        rc = self.group_table.rowCount()
        for r in range(rc):
            cb = self.group_table.cellWidget(r, 4)
            if not isinstance(cb, QComboBox): continue
            current = cb.currentData()
            cb.blockSignals(True);
            cb.clear()
            for name, hexv in self._color_palette:
                if hexv == current or hexv not in used:
                    cb.addItem(name, hexv)
                    idx = cb.count() - 1
                    cb.setItemData(idx, QColor(hexv), Qt.BackgroundRole)
            found = next((i for i in range(cb.count()) if cb.itemData(i) == current), -1)
            cb.setCurrentIndex(found if found >= 0 else 0)
            cb.blockSignals(False)

    def _insert_group_row(self, r: int, gid: str = "", start: str = "", end: str = "", color_hex: str = None):
        self._ensure_palette_size(r + 1)
        self.group_table.insertRow(r)
        # cột 1–3: ID, Start, End
        self.group_table.setItem(r, 1, QTableWidgetItem(gid))
        self.group_table.setItem(r, 2, QTableWidgetItem(start))
        self.group_table.setItem(r, 3, QTableWidgetItem(end))
        # cột 4: Color (ComboBox)
        if color_hex is None:
            used = self._used_colors()
            color_hex = next((h for _, h in self._color_palette if h not in used), self._color_palette[0][1])
        cb = self._create_color_combo(preselect_hex=color_hex)
        self.group_table.setCellWidget(r, 4, cb)

    def _on_color_changed(self, _idx):
        self._refresh_all_color_combos()

    def on_add_group(self):
        r = self.group_table.rowCount()
        self._insert_group_row(r, gid=f"G{r + 1}")
        self._renumber_group_table()
        self._refresh_all_color_combos()

    def on_delete_group(self):
        rows = sorted({idx.row() for idx in self.group_table.selectedIndexes()}, reverse=True)
        if not rows:
            self._log("[!] Select at least one row to delete.")
            return
        for r in rows:
            self.group_table.removeRow(r)
        if self.group_table.rowCount() == 0:
            self._insert_group_row(0, gid="G1")
        self._renumber_group_table()
        self._refresh_all_color_combos()

    def _groups_json_path(self) -> str:
        # lưu mỗi line một file JSON
        line_label = self.line_combo.currentText().strip() or f"line_{self.line_combo.currentIndex()+1:03d}"
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in line_label)
        gdir = os.path.join(self.workspace, "groups")
        os.makedirs(gdir, exist_ok=True)
        return os.path.join(gdir, f"{safe}.json")

    def _read_group_table(self):
        groups = []
        rc = self.group_table.rowCount()
        for r in range(rc):
            id_item = self.group_table.item(r, 1)
            s_item = self.group_table.item(r, 2)
            e_item = self.group_table.item(r, 3)
            cb = self.group_table.cellWidget(r, 4)

            gid = (id_item.text().strip() if id_item else "")
            s = (s_item.text().strip() if s_item else "")
            e = (e_item.text().strip() if e_item else "")
            if not gid and not s and not e:
                continue
            try:
                s_val = float(s); e_val = float(e)
            except Exception:
                self._log(f"[!] Row {r+1}: start/end không hợp lệ.")
                continue
            if e_val < s_val: s_val, e_val = e_val, s_val
            groups.append({
                "id": gid or f"G{len(groups) + 1}",
                "start": s_val, "end": e_val,
                "color": cb.currentData() if isinstance(cb, QComboBox) else None
            })
        return groups

    def _load_groups_for_current_line(self):
        path = self._groups_json_path()
        self.group_table.setRowCount(0)
        loaded = 0
        if os.path.exists(path):
            try:
                import json
                with open(path, "r", encoding="utf-8") as f:
                    js = json.load(f)
                for g in js.get("groups", []):
                    r = self.group_table.rowCount()
                    self._insert_group_row(
                        r,
                        gid=str(g.get("id", "")),
                        start=f'{float(g.get("start", 0.0)):.3f}',
                        end=f'{float(g.get("end", 0.0)):.3f}',
                        color_hex=g.get("color", None)
                    )
                    loaded += 1
                if loaded:
                    self._renumber_group_table()
                    self._refresh_all_color_combos()
                    self._log(f"[i] Loaded {loaded} groups for current line.")
            except Exception as e:
                self._log(f"[!] Cannot load group file: {e}")
        if loaded == 0:
            for _ in range(3):
                r = self.group_table.rowCount()
                self._insert_group_row(r, gid=f"G{r + 1}")
            self._renumber_group_table()
            self._refresh_all_color_combos()

    def on_confirm_groups(self):
        # 1) Luôn tính profile (chỉ trong slip zone)
        if self.lines_gdf is None or self.lines_gdf.empty:
            self._log("[!] No lines.")
            return
        row = self.line_combo.currentIndex()
        if row < 0:
            self._log("[!] Please select a line first.")
            return
        geom = self.lines_gdf.geometry.iloc[row]
        prof = compute_profile(
            self.dem_path, self.dx_path, self.dy_path, self.dz_path,
            geom, step_m=self.step_box.value(),
            smooth_win=11, smooth_poly=2,
            slip_mask_path=self.slip_path, slip_only=True
        )
        if not prof:
            self._log("[!] Empty profile. Cannot confirm groups.")
            return

        # 2) Lấy groups từ bảng; nếu trống -> auto-group
        groups = self._read_group_table()
        if not groups:
            groups = auto_group_profile(prof)
            self._log(f"[i] Auto-grouped {len(groups)} segments in slip zone.")

        # 3) CLAMP về đúng slip zone
        groups = clamp_groups_to_slip(prof, groups)
        if not groups:
            self._log("[!] All groups fall outside slip zone after clamping.")
            return

        # 4) Lưu JSON
        path = self._groups_json_path()
        try:
            import json, datetime
            js = {
                "line": self.line_combo.currentText(),
                "timestamp": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                "groups": groups
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(js, f, ensure_ascii=False, indent=2)
            self._log(f"[✓] Saved group definition: {path}")
        except Exception as e:
            self._log(f"[!] Cannot save group file: {e}")

        # 5) Re-render ảnh với màu theo group (đã clamp)
        y_min = self.ymin_box.value() or None
        y_max = self.ymax_box.value() or None
        out_png = os.path.join(self.preview_dir, f"profile_{row + 1:03d}.png")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        msg, path_png = render_profile_png(
            prof, out_png,
            y_min=y_min, y_max=y_max,
            vec_scale=self.vscale_box.value(),
            vec_width=self.vwidth_box.value(),
            head_len=self.hlen_box.value(),
            head_w=self.hwid_box.value(),
            highlight_theta=None,
            group_ranges=groups
        )
        self._log(msg)
        self._show_png(path_png)
        self._log(f"[✓] Applied {len(groups)} groups to vectors.")

    def _line_azimuth_deg(self, geom):
        try:
            x0, y0 = list(geom.coords)[0]
            x1, y1 = list(geom.coords)[-1]
            return float(np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180.0)
        except Exception:
            return 0.0

    def _classify_lines_main_aux(self):
        """Ưu tiên metadata từ UI2 (cột role/type/axis/class/kind), thiếu thì suy từ góc."""
        self._line_roles = {}
        if self.lines_gdf is None or self.lines_gdf.empty: return

        meta_cols = [c for c in self.lines_gdf.columns if str(c).lower() in ("role", "type", "axis", "class", "kind")]
        if meta_cols:
            col = meta_cols[0]
            for idx, row in self.lines_gdf.iterrows():
                val = str(row.get(col, "")).lower()
                if "main" in val or "chinh" in val:
                    self._line_roles[idx] = "main"
                elif "aux" in val or "phu" in val:
                    self._line_roles[idx] = "aux"

        if len(self._line_roles) < len(self.lines_gdf):
            azs = [(idx, self._line_azimuth_deg(row.geometry)) for idx, row in self.lines_gdf.iterrows()]
            if not azs: return
            med = float(np.median([a for _, a in azs]))
            for idx, ang in azs:
                d = abs(((ang - med + 90) % 180) - 90)
                self._line_roles[idx] = "main" if d <= 15 else (
                    "aux" if abs(d - 90) <= 15 else ("main" if d < 45 else "aux"))

    def _z_on_main_at_intersection(self, aux_row, main_row):
        """Trả (sM_aux, zM) – chain trên AUX tại giao; zM nội suy từ curve CHÍNH đã cache."""
        try:
            geom_aux = self.lines_gdf.geometry.iloc[aux_row]
            geom_main = self.lines_gdf.geometry.iloc[main_row]
            inter = geom_aux.intersection(geom_main)
            if inter.is_empty:
                p_aux, _ = nearest_points(geom_aux, geom_main)
                pt = p_aux
            else:
                pt = inter if inter.geom_type != "MultiPoint" else min(list(inter.geoms), key=lambda p: p.distance(
                    geom_aux.interpolate(0.5 * geom_aux.length)))
            sM_aux = float(geom_aux.project(pt))
            sM_main = float(geom_main.project(pt))
            mc = self._main_curves.get(main_row)
            if not mc: return None
            zM = float(np.interp(sM_main, mc["chain"], mc["z"]))
            return sM_aux, zM
        except Exception:
            return None

    def on_draw_slip_curve(self):
        try:
            if self.lines_gdf is None or self.lines_gdf.empty:
                self._log("[!] No lines loaded.");
                return
            row = self.line_combo.currentIndex()
            if row < 0:
                self._log("[!] Please select a line first.");
                return

            self._classify_lines_main_aux()
            role = self._line_roles.get(row, "main")

            # 1) Profile CHỈ TRONG SLIP ZONE
            geom = self.lines_gdf.geometry.iloc[row]
            prof = compute_profile(
                self.dem_path, self.dx_path, self.dy_path, self.dz_path,
                geom, step_m=self.step_box.value(),
                smooth_win=11, smooth_poly=2,
                slip_mask_path=self.slip_path, slip_only=True
            )
            if not prof or len(prof.get("chain", [])) < 5:
                self._log("[!] Empty/too-short slip profile.");
                return

            # 2) Lấy group (ưu tiên bảng → file → auto) và CLAMP vào slip-zone
            groups = self._read_group_table() or []
            if not groups:
                try:
                    import json
                    gpath = self._groups_json_path()
                    if os.path.exists(gpath):
                        with open(gpath, "r", encoding="utf-8") as f:
                            js = json.load(f);
                            groups = js.get("groups", []) or []
                except Exception:
                    groups = []
            if not groups:
                groups = auto_group_profile(prof)
                self._log(f"[i] Auto-grouped {len(groups)} segments in slip zone.")
            groups = clamp_groups_to_slip(prof, groups)
            if not groups:
                self._log("[!] No groups within slip zone.");
                return

            # 3) Đường mục tiêu trong SLIP-ZONE
            base = estimate_slip_curve(
                prof, groups,
                ds=0.2, smooth_factor=0.1,
                depth_gain=8, min_depth=2
            )
            if len(base.get("chain", [])) < 6:
                self._log("[!] Not enough points to fit curve.");
                return

            # Extent dùng cho Bézier = đúng span slip-zone (từ groups)
            sA = min(g["start"] for g in groups)
            sB = max(g["end"] for g in groups)

            # 4) Fit theo vai trò
            if role == "main":
                bez = fit_bezier_smooth_curve(
                    chain=np.asarray(prof["chain"]),
                    elevg=np.asarray(prof["elev_s"]),
                    target_s=np.asarray(base["chain"]),
                    target_z=np.asarray(base["elev"]),
                    c0=0.2, c1=0.4, clearance=0.2
                )

                self._main_curves[row] = {
                    "chain": np.asarray(bez["chain"], float),
                    "z": np.asarray(bez["elev"], float),
                    "geom": geom
                }
            else:
                # tìm tuyến CHÍNH thích hợp
                main_rows = [idx for idx, rl in self._line_roles.items() if rl == "main"]
                chosen, zinfo = None, None
                for mr in main_rows:
                    zinfo = self._z_on_main_at_intersection(aux_row=row, main_row=mr)
                    if zinfo is not None: chosen = mr; break
                if chosen is None and main_rows:
                    # không giao hình học? chọn tuyến CHÍNH gần nhất rồi lấy nearest_points
                    dmin = 1e18
                    for mr in main_rows:
                        d = self.lines_gdf.geometry.iloc[row].distance(self.lines_gdf.geometry.iloc[mr])
                        if d < dmin: dmin, chosen = d, mr
                    zinfo = self._z_on_main_at_intersection(aux_row=row, main_row=chosen)

                if (chosen is None) or (zinfo is None):
                    self._log("[i] No usable main curve; fitting free Bezier.")
                    bez = fit_bezier_smooth_curve(
                        chain=np.asarray(prof["chain"]),
                        elevg=np.asarray(prof["elev_s"]),
                        target_s=np.asarray(base["chain"]),
                        target_z=np.asarray(base["elev"]),
                        c0=0.30, c1=0.30, clearance=0.12
                    )
                else:
                    sM_aux, zM = zinfo
                    bez = fit_bezier_with_intersection(
                        chain=np.asarray(prof["chain"]),
                        elevg=np.asarray(prof["elev_s"]),
                        target_s=np.asarray(base["chain"]),
                        target_z=np.asarray(base["elev"]),
                        sA=sA, sB=sB,
                        pass_point=(sM_aux, zM),
                        c0=0.30, c1=0.30, clearance=0.12, hard_weight=2500.0
                    )
                    self._log(f"[✓] Constrained at intersection s={sM_aux:.2f}, z={zM:.2f}")

            # 5) Vẽ preview (ground + slip curve) — chỉ trong slip-zone
            out_png = os.path.join(self.preview_dir, f"profile_{row + 1:03d}_curve.png")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
            ax.plot(prof["chain"], prof["elev_s"], lw=1.2, label="Ground", color="#4e79a7")
            ax.plot(bez["chain"], bez["elev"], lw=2.4, label="Slip curve", color="#f28e2b")
            ax.set_xlabel("Chain (m)");
            ax.set_ylabel("Elevation (m)")
            ax.grid(True, alpha=0.25);
            ax.legend(loc="best");
            fig.tight_layout()
            fig.savefig(out_png);
            plt.close(fig)
            self._log(f"[✓] Slip curve saved → {out_png}")
            self._show_png(out_png)

        except Exception as e:
            import traceback
            self._log(f"[!] Draw Slip Curve crashed: {e}")
            self._log(traceback.format_exc())

    def _show_png(self, path: Optional[str]):
        if not path or not os.path.exists(path): return
        self.scene.clear()
        self.scene.addItem(QGraphicsPixmapItem(QPixmap(path)))
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def _log(self, text: str):
        self.status.append(text)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def go_to_back(self):
        try:
            from ui2_frontend import UI2App
        except Exception as e:
            self._log(f"[!] Cannot import UI2App: {e}")
            return
        self._log("[←] Back to UI2")
        self.win2 = UI2App(); self.win2.showMaximized(); self.close()
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.scene.items():
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    scr = app.primaryScreen()
    dpi = scr.logicalDotsPerInch() if scr else 96.
    base_pt = 12 if dpi > 120 else 11
    app.setFont(QFont("Public Sans", base_pt))
    win = UI3App()
    geo = app.primaryScreen().availableGeometry()
    win.resize(int(geo.width() * 0.85), int(geo.height() * 0.85))
    win.show()
    win.showMaximized()
    sys.exit(app.exec_())




