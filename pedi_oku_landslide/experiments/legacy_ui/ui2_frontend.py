# == ui2_frontend.py — CLEAN HEADER ==
import os, sys, datetime
dll_dir = os.path.join(sys.prefix, "Library", "bin")
if hasattr(os, "add_dll_directory"):
    try: os.add_dll_directory(dll_dir)
    except FileNotFoundError: pass
else:
    os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

os.environ.setdefault("GDAL_DATA", os.path.join(sys.prefix, "Library", "share", "gdal"))
os.environ.setdefault("PROJ_LIB",  os.path.join(sys.prefix, "Library", "share", "proj"))
from PyQt5.QtWidgets import QGraphicsLineItem
from pipeline.runners.ui2_backend import generate_auto_lines_from_slipzone
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPen, QImage, QPainter
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QTableWidgetItem, QHeaderView, QTextEdit,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QAbstractItemView,
    QComboBox, QTableWidget, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QMessageBox, QScrollArea
)
from PyQt5.QtWidgets import QSizePolicy
from shapely.geometry import LineString
import numpy as np
import rasterio

from pipeline.runners.ui2_backend import (
    default_paths, generate_vector_overlay_image, save_selected_lines_gpkg,
    read_vector_lines, map_to_xy, write_shared_json
)

try:
    from ui1_frontend import UI1App
except Exception as e:
    UI1App = None
    print("[UI2] Warning: cannot import UI1App:", e)

def _densify_line(line: LineString, step_m: float) -> tuple[np.ndarray, np.ndarray]:
    if line is None or getattr(line, "is_empty", False):
        return np.array([]), np.array([])
    try:
        length_m = float(line.length)
    except Exception:
        return np.array([]), np.array([])
    if not np.isfinite(length_m) or length_m <= 0:
        return np.array([]), np.array([])
    step = float(step_m) if step_m and np.isfinite(step_m) else 1.0
    n = max(2, int(np.ceil(length_m / step)) + 1)
    s = np.linspace(0.0, length_m, n)
    xs = np.empty(n); ys = np.empty(n)
    for i, d in enumerate(s):
        p = line.interpolate(d)
        xs[i], ys[i] = p.x, p.y
    return xs, ys

def _export_lines_dem_json(dem_path: str, lines: list, labels: list, out_json: str, step_m: float | None) -> str:
    import json
    if not dem_path or not os.path.exists(dem_path):
        return "[!] DEM path not found for JSON export."
    if not lines:
        return "[!] No lines to export."
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with rasterio.open(dem_path) as ds:
        if step_m is None:
            px = abs(float(ds.transform.a)) if np.isfinite(ds.transform.a) else 0.0
            py = abs(float(ds.transform.e)) if np.isfinite(ds.transform.e) else 0.0
            step_m = max(px, py, 1.0)
        nodata = ds.nodata
        out_lines = []
        for i, line in enumerate(lines):
            xs, ys = _densify_line(line, step_m)
            if xs.size == 0:
                out_lines.append({"index": i, "label": labels[i], "points": []})
                continue
            vals = list(ds.sample(list(zip(xs.tolist(), ys.tolist())), indexes=1))
            pts = []
            for x, y, v in zip(xs.tolist(), ys.tolist(), vals):
                z = v[0] if len(v) else np.nan
                if nodata is not None:
                    if np.isnan(nodata):
                        if not np.isfinite(z):
                            z = None
                    else:
                        if np.isclose(z, float(nodata)):
                            z = None
                if z is not None and not np.isfinite(z):
                    z = None
                pts.append({"x": float(x), "y": float(y), "z": None if z is None else float(z)})
            out_lines.append({"index": i, "label": labels[i], "points": pts})
    payload = {
        "dem_path": os.path.abspath(dem_path),
        "step_m": float(step_m),
        "lines": out_lines,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return f"[✓] Saved DEM points JSON → {os.path.abspath(out_json)}"

class DeletableTableWidget(QTableWidget):
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            selected_rows = sorted(set(index.row() for index in self.selectedIndexes()), reverse=True)
            for r in selected_rows:
                self.removeRow(r)
        else:
            super().keyPressEvent(event)

class LinePickerView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setInteractive(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._map_to_xy = None
        self.coord_label = None
        self.last_click_xy = None
        self.pen_hover = QPen(Qt.green)
        self.pen_hover.setWidth(4)
        self.hover_dot = None
        self.click_dots = []

    def set_mapper(self, mapper):
        self._map_to_xy = mapper

    def set_coord_label(self, label):
        self.coord_label = label

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if not self._map_to_xy or not self.coord_label:
            return
        pos_scene = self.mapToScene(event.pos())
        col, row = pos_scene.x(), pos_scene.y()
        x, y = self._map_to_xy(row, col)
        self.coord_label.setText(f"X: {x:.3f}  Y: {y:.3f}")
        if self.hover_dot:
            self.scene().removeItem(self.hover_dot)
        self.hover_dot = self.scene().addEllipse(pos_scene.x()-2, pos_scene.y()-2, 4, 4, self.pen_hover)
        self.hover_dot.setZValue(5)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton and self._map_to_xy:
            pos_scene = self.mapToScene(event.pos())
            col, row = pos_scene.x(), pos_scene.y()
            x, y = self._map_to_xy(row, col)
            self.last_click_xy = (x, y)
            for dot in self.click_dots:
                self.scene().removeItem(dot)
            self.click_dots.clear()
            dot = self.scene().addEllipse(pos_scene.x()-4, pos_scene.y()-4, 8, 8, self.pen_hover)
            dot.setZValue(10)
            self.click_dots.append(dot)


class UI2App(QWidget):
    def __init__(self, json_path=None):
        super().__init__()
        self.before_file = ""
        self.after_file = ""
        self.workspace = "output/UI2"  # standard root for UI2

        self.setWindowTitle("UI2 – Slip Zone Cross Section Selector")
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                font-family: 'Public Sans', sans-serif;
                font-size: 11.5pt;      /* ↑ từ 10.5pt */
                font-weight: 500;       /* 600 -> 500 cho đỡ “nở” */
                color: #000000;
            }
            QGroupBox::title { padding: 2px 4px; font-size: 12pt; font-weight: 600; }
            QLabel { margin: 0 0 2px 0; font-size: 11.2pt; }
            QPushButton {
                background-color: #000000; color: white; border: none;
                padding: 7px 12px; border-radius: 10px; font-size: 11.2pt;
            }
            QPushButton:hover { background-color: #222222; }
            QTableWidget, QComboBox, QTextEdit {
                border: 1px solid #cccccc; border-radius: 8px;
                background-color: #ffffff; padding: 3px 8px; font-size: 11.2pt;
            }
            QHeaderView::section {
                background-color: #f6f6f6; padding: 4px; border: 0; border-bottom: 1px solid #e6e6e6;
            }
            QGraphicsView { background-color: #ffffff; border: 1px solid #eeeeee; border-radius: 8px; }
            QTextEdit { background-color: #ffffff; }
        """)

        os.makedirs(self.workspace, exist_ok=True)
        self.log_path = os.path.join(self.workspace, f"log_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")
        self.transform = None
        self.crs = None
        self.dem_path = ""
        self._build_ui()
        self._auto_load_default()

    def log(self, msg):
        self.status_box.append(msg)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg+"\n")

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # trong _build_ui
        header = QHBoxLayout()
        title = QLabel("UI2 – Slip Zone Cross Section Selector")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()
        
        btn_back = QPushButton("Back")
        btn_back.clicked.connect(self.go_to_back)
        header.addWidget(btn_back)
        
        btn_next = QPushButton("Next Step")
        btn_next.clicked.connect(self.go_to_next)
        header.addWidget(btn_next)
        
        layout.addLayout(header)

        content = QHBoxLayout()
        layout.addLayout(content)

        self.scene = QGraphicsScene()
        self.view = LinePickerView(self.scene)
        content.addWidget(self.view, 6)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.view.setAlignment(Qt.AlignCenter)

        #  QScrollArea
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        content.addWidget(right_scroll, 6)

        right_host = QWidget()
        right_scroll.setWidget(right_host)
        right = QVBoxLayout(right_host)
        right.setContentsMargins(6, 6, 6, 6)
        right.setSpacing(6)
        right_scroll.setMinimumWidth(520)  # 320–380 tuỳ ý
        right_scroll.setMaximumWidth(700)

        self.mode_box = QComboBox()
        self.mode_box.addItems(["Coordinate based Input", "Load File"])
        self.mode_box.currentIndexChanged.connect(self._on_mode_changed)
        right.addWidget(QLabel("Input mode"))
        right.addWidget(self.mode_box)

        self.table = DeletableTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Line", "First X", "First Y", "Second X", "Second Y"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        right.addWidget(QLabel("Input coordinates:"))
        right.addWidget(self.table, 5)
        self.table.setSizeAdjustPolicy(QAbstractItemView.AdjustToContents)
        auto_grp = QGroupBox("Automatic Lines")
        grid = QGridLayout(auto_grp)
        
        self.main_num = QSpinBox()
        self.main_num.setRange(0, 200)
        self.main_num.setSingleStep(2)   # chỉ bước chẵn
        self.main_off = QDoubleSpinBox()
        self.main_off.setRange(0.0, 1e6)
        self.main_off.setDecimals(2)
        self.main_off.setValue(20.0)
        
        self.cross_num = QSpinBox()
        self.cross_num.setRange(0, 200)
        self.cross_num.setSingleStep(2)
        self.cross_off = QDoubleSpinBox()
        self.cross_off.setRange(0.0, 1e6)
        self.cross_off.setDecimals(2)
        self.cross_off.setValue(20.0)
        
        grid.addWidget(QLabel("Main lines:"),        0, 0)
        grid.addWidget(QLabel("Line Number:"),       0, 1)
        grid.addWidget(self.main_num,                0, 2)
        grid.addWidget(QLabel("Offset (m):"),        0, 3)
        grid.addWidget(self.main_off,                0, 4)
        
        grid.addWidget(QLabel("Cross lines:"),       1, 0)
        grid.addWidget(QLabel("Line Number:"),       1, 1)
        grid.addWidget(self.cross_num,               1, 2)
        grid.addWidget(QLabel("Offset (m):"),        1, 3)
        grid.addWidget(self.cross_off,               1, 4)
        
        self.btn_gen_lines = QPushButton("Generate Lines")
        self.btn_gen_lines.clicked.connect(self.on_generate_lines)
        grid.addWidget(self.btn_gen_lines,           2, 0, 1, 5)
        
        right.addWidget(auto_grp)
        row1 = QHBoxLayout()
        self.btn_add = QPushButton("+ Add Line")
        self.btn_add.clicked.connect(self._add_line)
        row1.addWidget(self.btn_add)
        self.btn_pick = QPushButton("Pick point")
        self.btn_pick.clicked.connect(self._pick_point)
        row1.addWidget(self.btn_pick)
        right.addLayout(row1)

        self.btn_upload = QPushButton("Upload vector file")
        self.btn_upload.clicked.connect(self._upload_vector)
        right.addWidget(self.btn_upload)
        self.btn_confirm = QPushButton("Confirm Input")
        self.btn_confirm.clicked.connect(self._confirm)
        right.addWidget(self.btn_confirm)

        self.coord_label = QLabel("X: ---  Y: ---")
        self.view.set_coord_label(self.coord_label)
        right.addWidget(self.coord_label)

        stat_box = QGroupBox("Status")
        stat_lay = QVBoxLayout(stat_box);
        stat_lay.setContentsMargins(6, 6, 6, 6);
        stat_lay.setSpacing(4)
        self.status_box = QTextEdit();
        self.status_box.setReadOnly(True)
        self.status_box.setLineWrapMode(QTextEdit.NoWrap)
        self.status_box.setFixedHeight(96)
        stat_lay.addWidget(self.status_box)
        right.addWidget(stat_box)
        right.addStretch(1)

    def _display_png(self, png_path: str, dem_path: str):
        # Read transform+crs from DEM, image from PNG
        with rasterio.open(dem_path) as src:
            self.transform = src.transform
            self.crs = src.crs
            h, w = src.height, src.width
        self.dem_path = dem_path
        pixmap = QPixmap(png_path)
        self.scene.clear()
        item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(item)
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.view.setSceneRect(self.scene.sceneRect())
        self.view.set_mapper(lambda r, c: map_to_xy(self.transform, r, c))
        self.view.centerOn(pixmap.width() // 2, pixmap.height() // 2)
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def _auto_load_default(self):
        paths = default_paths(base_root="output")
        dem_path = paths["dem_path"]
        dx_path = paths["dx_path"]
        dy_path = paths["dy_path"]
        slip_path = paths["slip_path"]

        if all(os.path.exists(p) for p in [dem_path, dx_path, dy_path]):
            status, outs = generate_vector_overlay_image(
                dem_path, dx_path, dy_path,
                slip_mask_path=slip_path,
                stride=25, scale=5.0, vector_color="red",
                output_dir="output/UI2/step1_vector_overlay",
                output_name="vector_overlay.png",
                show_grid=True,          
                grid_interval=20.0,       
            )
            self.log(status)
            self._display_png(outs["png_path"], dem_path)
        else:
            self.log("[!] Missing DEM or dX/dY for overlay display.")

    def _add_line(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(str(r + 1)))

    def _pick_point(self):
        xy = self.view.last_click_xy
        if xy is None:
            self.log("[!] No point selected – click on map first.")
            return
        for dot in self.view.click_dots:
            self.scene.removeItem(dot)
        self.view.click_dots.clear()
        x, y = xy
        selected = self.table.selectedIndexes()
        if selected:
            xs = [idx for idx in selected if idx.column() in [1, 2, 3, 4]]
            if len(xs) >= 2:
                xs.sort(key=lambda i: i.column())
                self.table.setItem(xs[0].row(), xs[0].column(), QTableWidgetItem(f"{x:.3f}"))
                self.table.setItem(xs[1].row(), xs[1].column(), QTableWidgetItem(f"{y:.3f}"))
                return
        r = self.table.rowCount() - 1
        if r < 0 or (self.table.item(r, 3) and self.table.item(r, 4)):
            self._add_line()
            r += 1
        if not self.table.item(r, 1):
            self.table.setItem(r, 1, QTableWidgetItem(f"{x:.3f}"))
            self.table.setItem(r, 2, QTableWidgetItem(f"{y:.3f}"))
        elif not self.table.item(r, 3):
            self.table.setItem(r, 3, QTableWidgetItem(f"{x:.3f}"))
            self.table.setItem(r, 4, QTableWidgetItem(f"{y:.3f}"))

    def _upload_vector(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select vector file", "", "Vector (*.gpkg *.geojson)")
        if not path:
            return
        lines, crs = read_vector_lines(path)
        if crs is None:
            # fallback to DEM crs
            crs = self.crs or "EPSG:4326"
        self.table.setRowCount(0)
        for i, geom in enumerate(lines):
            x0, y0 = geom.coords[0][:2]
            x1, y1 = geom.coords[-1][:2]
            self._add_line()
            r = self.table.rowCount() - 1
            for c, v in enumerate([i + 1, x0, y0, x1, y1]):
                self.table.setItem(r, c, QTableWidgetItem(f"{v:.3f}" if c else str(v)))
        self.log(f"[✓] Loaded {self.table.rowCount()} lines from {os.path.basename(path)}")

    def _confirm(self):
        try:
            # (2) chặn nếu chưa có transform
            if self.transform is None:
                QMessageBox.warning(self, "Confirm Input", "Missing DEM/image to transform – input image first.")
                return

            # 1) Xoá overlay line cũ
            for item in list(self.scene.items()):
                if isinstance(item, QGraphicsLineItem) and item.zValue() == 2:
                    self.scene.removeItem(item)

            # 2) Đọc bảng -> LineString + vẽ overlay
            lines = []
            labels = []
            for r in range(self.table.rowCount()):
                try:
                    x0 = float(self.table.item(r, 1).text());
                    y0 = float(self.table.item(r, 2).text())
                    x1 = float(self.table.item(r, 3).text());
                    y1 = float(self.table.item(r, 4).text())
                except Exception:
                    self.log(f"[!] Row {r + 1}: Coordinates unavailble, reject and continue.");
                    continue

                lines.append(LineString([(x0, y0), (x1, y1)]))
                label_item = self.table.item(r, 0)
                labels.append(label_item.text() if label_item else str(r + 1))
                c0, r0 = ~self.transform * (x0, y0)
                c1, r1 = ~self.transform * (x1, y1)
                it = self.scene.addLine(c0, r0, c1, r1, QPen(Qt.white, 2));
                it.setZValue(2)

            if not lines:
                self.log("[!] No valid lines to save.")
                QMessageBox.warning(self, "Confirm Input", "No avaliable lines to save.")
                return

            # --- 3) Lưu GPKG ---
            gpkg_out = os.path.join("output", "UI2", "step2_selected_lines", "selected_lines.gpkg")
            os.makedirs(os.path.dirname(gpkg_out), exist_ok=True)

            # LẤY CRS TỪ DEM ĐÃ NẠP Ở _display_png; nếu chưa có thì cảnh báo
            if self.crs is None:
                self.log("[!] CRS is None – cannot save lines with valid projection.")
                QMessageBox.warning(self, "Confirm Input", "CRS from DEM unavailable - input image first.")
                return

            status, meta = save_selected_lines_gpkg(lines, self.crs, gpkg_out)
            gpkg_path = os.path.abspath(meta["gpkg_path"])
            self.log(status)
            self.log(f"[✓] Saved {len(lines)} lines → {gpkg_path}")

            # --- 4) Ghi JSON chia sẻ (KHÔNG dùng write_shared_json cũ) ---
            import json
            def _write_json(path, payload):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                old = {}
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            old = json.load(f) or {}
                    except Exception:
                        old = {}
                old.update(payload)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(old, f, ensure_ascii=False, indent=2)

            payload = {"lines_path": gpkg_path}
            _write_json(os.path.join("output", "ui_shared_data.json"), payload)
            self.log("[i] Updated output/ui_shared_data.json")
            _write_json(os.path.join("output", "UI2", "ui_shared_data.json"), payload)
            self.log("[i] Updated output/UI2/ui_shared_data.json")

            # --- 5) Export DEM points JSON for drawn lines ---
            out_json = os.path.join("output", "UI2", "step2_selected_lines", "selected_lines_dem_points.json")
            msg = _export_lines_dem_json(self.dem_path, lines, labels, out_json, step_m=None)
            self.log(msg)

            # --- 6) Fit view + dọn chấm chọn như cũ ---
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            for d in self.view.click_dots:
                self.scene.removeItem(d)
            self.view.click_dots.clear()

        except Exception as e:
            self.log(f"[✗] Confirm Input failed: {e}")
            QMessageBox.critical(self, "Confirm Input", f"Đã xảy ra lỗi:\n{e}")

    def _on_mode_changed(self, idx):
        coord_mode = idx == 0
        self.btn_upload.setEnabled(not coord_mode)
        self.btn_pick.setEnabled(coord_mode)
        self.btn_add.setEnabled(coord_mode)

    
    def _warn(self, msg: str):
        QMessageBox.warning(self, "Automatic Lines", msg)

    def _append_lines_to_table(self, feats: list):
        for feat in feats:
            geom = feat["geom"]
            (x0, y0), (x1, y1) = list(geom.coords)[0], list(geom.coords)[-1]
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(feat["name"]))
            self.table.setItem(r, 1, QTableWidgetItem(f"{x0:.3f}"))
            self.table.setItem(r, 2, QTableWidgetItem(f"{y0:.3f}"))
            self.table.setItem(r, 3, QTableWidgetItem(f"{x1:.3f}"))
            self.table.setItem(r, 4 , QTableWidgetItem(f"{y1:.3f}"))

    def _draw_lines_overlay(self, feats: list, color=Qt.white, width=2):
        pen = QPen(color)
        pen.setWidth(width)
        for feat in feats:
            (x0, y0), (x1, y1) = list(feat["geom"].coords)[0], list(feat["geom"].coords)[-1]
            c0, r0 = ~self.transform * (x0, y0)
            c1, r1 = ~self.transform * (x1, y1)
            it = self.scene.addLine(c0, r0, c1, r1, pen)
            it.setZValue(2)

    def on_generate_lines(self):
        # Lấy tham số UI
        m_num = int(self.main_num.value())
        c_num = int(self.cross_num.value())
        m_off = float(self.main_off.value())
        c_off = float(self.cross_off.value())

        # Ép số chẵn (nếu người dùng nhập lẻ)
        if m_num % 2 == 1:
            m_num -= 1
            self._warn("Main lines: Line Number must be even. It has been automatically rounded down to the nearest even number.")
        if c_num % 2 == 1:
            c_num -= 1
            self._warn("Cross lines: Line Number must be even. It has been automatically rounded down to the nearest even number.")

        if m_off < 0 or c_off < 0:
            self._warn("Offset must be ≥ 0.")
            return

        # Đường dẫn dữ liệu mặc định từ UI1/UI2
        paths = default_paths(base_root="output")
        dem_path = paths["dem_path"]
        dx_path  = paths["dx_path"]
        dy_path  = paths["dy_path"]
        slip_path= paths["slip_path"]

        ok = all(os.path.exists(p) for p in [dem_path, dx_path, dy_path, slip_path])
        if not ok:
            self._warn("Lost data (DEM/DX/DY/SlipZone). Please back to UI1 and check output.")
            return

        try:
            outs = generate_auto_lines_from_slipzone(
                dem_path=dem_path, dx_path=dx_path, dy_path=dy_path, slip_mask_path=slip_path,
                main_num_even=m_num, main_offset_m=m_off,
                cross_num_even=c_num, cross_offset_m=c_off,
                base_length_m=None
            )
        except Exception as e:
            self._warn(f"Failed to auto generate lines: {e}")
            return

        main_feats = outs.get("main", [])
        cross_feats = outs.get("cross", [])
        # APPEND vào bảng (sau các line có sẵn)
        self._append_lines_to_table(main_feats + cross_feats)
        # Vẽ overlay ngay
        self._draw_lines_overlay(main_feats + cross_feats)
        self.log(f"[✓] Generated {len(main_feats)} main + {len(cross_feats)} cross lines (appended).")

    def go_to_back(self):
        # Mở lại UI1
        try:
            from ui1_frontend import UI1App
        except Exception as e:
            self.log(f"[!] Can't open UI1 – import UI1App failed: {e}")
            return
        try:
            self.log("[←] Returning to UI1")
            self.win1 = UI1App()
            self.win1.showMaximized()
            self.close()
        except Exception as e:
            self.log(f"[!] Failed to return to UI1: {e}")


    def go_to_next(self):
        self.log("[→] Switching to UI3 – Cross Section Analysis")
        try:
            from ui3_frontend import UI3App
            json_path = os.path.join("output", "UI2", "ui_shared_data.json")
            self.win3 = UI3App(json_path)
            self.win3.showMaximized()
            self.close()
        except Exception as e:
            self.log(f"[!] Failed to open UI3: {e}")
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.scene.items():
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    from PyQt5.QtGui import QFont
    scr = app.primaryScreen()
    dpi = scr.logicalDotsPerInch() if scr else 96.0
    base_pt = 12 if dpi > 120 else 11
    app.setFont(QFont("Public Sans", base_pt))

    win = UI2App()
    win.showMaximized()
    sys.exit(app.exec_())

