#ui1_frontend.py
import os, sys, datetime
dll_dir = os.path.join(sys.prefix, "Library", "bin")
if hasattr(os, "add_dll_directory"):
    try: os.add_dll_directory(dll_dir)
    except FileNotFoundError: pass
else:
    os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

# GDAL/PROJ + thread caps
os.environ.setdefault("GDAL_DATA", os.path.join(sys.prefix, "Library", "share", "gdal"))
os.environ.setdefault("PROJ_LIB",  os.path.join(sys.prefix, "Library", "share", "proj"))
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
os.environ.setdefault("OPENCV_OPENCL_DEVICE","disabled")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME","disabled")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QDoubleSpinBox, QComboBox, QTextEdit,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGroupBox, QFormLayout, QScrollArea, QMessageBox
)

from pedi_oku_landslide.pipeline.runners.ui1_backend import (
    run_UI1_1_crop, run_UI1_2_sad, run_UI1_2_sad_original, run_UI1_3_plot_vector_geotiff,
    run_UI1_4_dz, run_UI1_5_detect_slipzone
)

class ToggleableGroup(QGroupBox):
    def __init__(self, title, content_widget):
        super().__init__(title)
        self.setCheckable(True)
        self.setChecked(False)
        self.content_widget = content_widget
        layout = QVBoxLayout()
        layout.addWidget(self.content_widget)
        self.setLayout(layout)
        self.toggled.connect(self.on_toggled)
        self.on_toggled(self.isChecked())

    def on_toggled(self, checked):
        self.content_widget.setVisible(checked)

class UI1App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UI1 – Landslide Displacement Viewer")
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                font-family: 'Public Sans', sans-serif;
                font-size: 11.5pt;          /* ↑ base font */
                font-weight: 500;
                color: #000000;
            }
            QGroupBox::title { padding: 2px 4px; font-size: 12pt; font-weight: 600; }
            QLabel { margin: 0 0 2px 0; font-size: 11.2pt; }
            QPushButton {
                background-color: #000000; color: white; border: none;
                padding: 7px 12px; border-radius: 10px; font-size: 11.2pt;
            }
            QPushButton:hover { background-color: #222222; }
            QDoubleSpinBox, QComboBox, QTextEdit {
                border: 1px solid #cccccc; border-radius: 8px;
                background-color: #ffffff; padding: 3px 8px; font-size: 11.2pt;
            }
        """)

        self.workspace = "UI1_workspace"
        os.makedirs(self.workspace, exist_ok=True)
        self.log_path = os.path.join(self.workspace, f"log_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")

        self.before_file = self.after_file = self.before_pz_file = self.after_pz_file = self.before_ground_file = ""
        self.vector_color = "blue"
        self.initUI()

    def log(self, text):
        self.status_box.append(text)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def initUI(self):
        main_layout = QVBoxLayout(self)

        header = QHBoxLayout()
        title_label = QLabel("UI1 – Landslide Displacement Viewer")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        header.addWidget(title_label)
        header.addStretch()
        btn_next = QPushButton("Next Step")
        btn_next.clicked.connect(self.go_to_next)
        header.addWidget(btn_next)
        main_layout.addLayout(header)

        body_layout = QHBoxLayout()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_host = QWidget()
        left_scroll.setWidget(left_host)
        left_panel = QVBoxLayout(left_host)
        left_panel.setContentsMargins(6, 6, 6, 6)
        left_panel.setSpacing(6)
        left_scroll.setMinimumWidth(320)
        left_scroll.setMaximumWidth(420)

        left_panel.addWidget(QLabel("Upload BEFORE DEM for ground profile"))
        self._add_file_button("Upload BEFORE DEM ", left_panel, "before_ground")
        left_panel.addWidget(QLabel("Upload Data"))
        self._add_file_button("Upload BEFORE Data", left_panel, "before")
        self._add_file_button("Upload AFTER Data", left_panel, "after")
        left_panel.addWidget(QLabel("Upload Vertical Data"))
        self._add_file_button("Upload BEFORE_PZ", left_panel, "before_pz")
        self._add_file_button("Upload AFTER_PZ", left_panel, "after_pz")
        

        self._add_button("Confirm Input", self.run_confirm_input, left_panel)
        self.smooth_size_spin = self._add_labeled_spinbox("Smoothing kernel size (pixels):", left_panel, 10)
        self._add_button("Show Smooth DEM", self.run_show_smooth, left_panel)
        self.pixel_size_spin = self._add_labeled_spinbox("DEM pixel size (mm):", left_panel, 200)

        # === SAD engine selector (single) ===
        left_panel.addWidget(QLabel("SAD engine:"))
        self.sad_engine_combo = QComboBox()
        self.sad_engine_combo.addItems(["OpenCV (fast)", "Original (accurate)"])
        self.sad_engine_combo.setCurrentIndex(0)  # default: OpenCV
        left_panel.addWidget(self.sad_engine_combo)
        # --- Disable "Original (accurate)" option ---
        m = self.sad_engine_combo.model()
        it = m.item(1)  # index 1 = "Original (accurate)"
        if it is not None:
            it.setEnabled(False)  # bôi xám
            it.setSelectable(False)  # không cho chọn
            it.setForeground(Qt.gray)  # màu chữ xám
        # tooltip để người dùng biết lý do
        self.sad_engine_combo.setItemData(1, "Disabled: using OpenCV SAD for now", Qt.ToolTipRole)
        # đảm bảo luôn đang ở OpenCV
        self.sad_engine_combo.setCurrentIndex(0)

        self._add_button("Run SAD", self.run_sad, left_panel)
        self.threshold_spin = self._add_labeled_spinbox("Threshold height change (mm):", left_panel, 800)
        self._add_button("Display Slip Zone", self.run_slip_zone, left_panel)

        adv_widget = QWidget()
        adv_layout = QFormLayout(adv_widget)
        self.vector_scale_spin = QDoubleSpinBox(); self.vector_scale_spin.setValue(25.0)
        self.vector_stride_spin = QDoubleSpinBox(); self.vector_stride_spin.setValue(25)
        self.vector_stride_spin.setDecimals(0)
        self.vector_stride_spin.setMinimum(1)
        self.vector_headwidth_spin = QDoubleSpinBox(); self.vector_headwidth_spin.setValue(6)
        self.vector_headlength_spin = QDoubleSpinBox(); self.vector_headlength_spin.setValue(5)
        self.vector_headaxislength_spin = QDoubleSpinBox(); self.vector_headaxislength_spin.setValue(3)
        self.vector_width_spin = QDoubleSpinBox(); self.vector_width_spin.setValue(3.0)
        self.vector_color_combo = QComboBox(); self.vector_color_combo.addItems(["blue", "red", "green", "black", "yellow"])
        self.vector_color_combo.currentTextChanged.connect(lambda c: setattr(self, 'vector_color', c))

        adv_layout.addRow("Vector Scale:", self.vector_scale_spin)
        adv_layout.addRow("Vector Stride (px):", self.vector_stride_spin)
        adv_layout.addRow("Head Width:", self.vector_headwidth_spin)
        adv_layout.addRow("Head Length:", self.vector_headlength_spin)
        adv_layout.addRow("Head Axis Length:", self.vector_headaxislength_spin)
        adv_layout.addRow("Vector Line Width:", self.vector_width_spin)
        adv_layout.addRow("Vector Color:", self.vector_color_combo)

        adv_group = ToggleableGroup("Advanced Vector Display", adv_widget)
        left_panel.addWidget(adv_group)

        self._add_button("Display Vector", self.run_vector_overlay, left_panel)

        stat_box = QGroupBox("Status")
        stat_lay = QVBoxLayout(stat_box);
        stat_lay.setContentsMargins(6, 6, 6, 6);
        stat_lay.setSpacing(4)
        self.status_box = QTextEdit();
        self.status_box.setReadOnly(True)
        self.status_box.setLineWrapMode(QTextEdit.NoWrap)
        self.status_box.setFixedHeight(100)
        stat_lay.addWidget(self.status_box)
        left_panel.addWidget(stat_box)
        left_panel.addStretch(1)

        self.scene = QGraphicsScene(); self.view = QGraphicsView(self.scene); self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        body_layout.addWidget(left_scroll, 4)
        body_layout.addWidget(self.view, 6)
        main_layout.addLayout(body_layout)

    def _add_file_button(self, text, layout, attr):
        btn = QPushButton(text)
        btn.clicked.connect(lambda: self._select_file(attr))
        layout.addWidget(btn)

    def _select_file(self, attr):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", "Raster files (*.tif *.asc)")
        setattr(self, f"{attr}_file", path)
        if path:
            self.log(f"[✓] Loaded {attr.upper()}: {os.path.basename(path)}")

    def _add_button(self, text, func, layout):
        btn = QPushButton(text)
        btn.clicked.connect(func)
        layout.addWidget(btn)

    def run_confirm_input(self):
        out1 = "output/UI1/step1_crop"
        status, outputs = run_UI1_1_crop(
            self.before_file,
            self.after_file,
            self.before_pz_file,
            self.after_pz_file,
            output_dir=out1,
            smooth_size=self.smooth_size_spin.value()
        )
        self.log(status)
        if outputs:
            self.display_images_side_by_side(
                outputs.get("hill_before"),
                outputs.get("hill_after")
            )
        try:
            if getattr(self, "before_ground_file", ""):
                import os, json, pathlib, rasterio
                from rasterio.crs import CRS
        
                out_dir = os.path.join("output", "UI1", "step1_crop")
                os.makedirs(out_dir, exist_ok=True)
        
                src = os.path.abspath(self.before_ground_file)
                ext = pathlib.Path(src).suffix.lower()
                dest = os.path.abspath(os.path.join(out_dir, f"before_ground{ext}"))
        
                if ext in [".tif", ".tiff"]:
 
                    with rasterio.open(src) as src_ds:
                        data = src_ds.read()
                        profile = src_ds.profile
                        if src_ds.crs is None:
                            profile["crs"] = CRS.from_epsg(6677)  
                        profile["driver"] = "GTiff"
                    with rasterio.open(dest, "w", **profile) as dst:
                        dst.write(data)
                    self.log(f"[✓] Saved ground GeoTIFF with CRS=EPSG:6677 → {dest}")
                else:
                    if src != dest:
                        import shutil
                        shutil.copyfile(src, dest)
                    self.log(f"[i] Ground DEM copied → {dest}")
        

                shared_json = os.path.join("output", "ui_shared_data.json")
                payload = {"dem_ground_path": dest}
                if os.path.exists(shared_json):
                    try:
                        with open(shared_json, "r", encoding="utf-8") as f:
                            old = json.load(f)
                    except Exception:
                        old = {}
                    old.update(payload)
                    payload = old
                with open(shared_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                self.log(f"[i] Updated {shared_json}")
            else:
                self.log("[!] No 'BEFORE DEM for ground profile' selected.")
        except Exception as e:
            self.log(f"[!] Failed to save ground DEM: {e}")


    def display_images_side_by_side(self, path1, path2):
        if path1 and os.path.exists(path1) and path2 and os.path.exists(path2):
            self.scene.clear()
            pixmap1 = QPixmap(path1)
            pixmap2 = QPixmap(path2)
            combined_width = pixmap1.width() + pixmap2.width()
            combined_height = max(pixmap1.height(), pixmap2.height())
            combined = QPixmap(combined_width, combined_height)
            combined.fill(Qt.white)
            painter = QPainter(combined)
            painter.drawPixmap(0, 0, pixmap1)
            painter.drawPixmap(pixmap1.width(), 0, pixmap2)
            painter.end()
            self.scene.addItem(QGraphicsPixmapItem(combined))
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

            self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.view.setRenderHint(QPainter.Antialiasing, True)

    def run_show_smooth(self):
        out1 = "output/UI1/step1_crop"
        os.makedirs(out1, exist_ok=True)
        _, outputs = run_UI1_1_crop(
            self.before_file,
            self.after_file,
            self.before_pz_file,
            self.after_pz_file,
            output_dir=out1,
            smooth_size=self.smooth_size_spin.value()
        )
        if outputs:
            self.display_images_side_by_side(
                outputs.get("hill_before_smooth"),
                outputs.get("hill_after_smooth")
            )

    def run_sad(self):
        # Safety guard: nếu vì lý do nào đó combobox đang ở index 1, ép về OpenCV
        if self.sad_engine_combo.currentIndex() == 1:
            self.sad_engine_combo.setCurrentIndex(0)
        out1 = "output/UI1/step1_crop"
        out2 = "output/UI1/step2_sad"
        out5dz = "output/UI1/step5_dz"
        os.makedirs(out2, exist_ok=True)
        os.makedirs(out5dz, exist_ok=True)

        # NEW: chặn chạy nếu chưa Confirm Input
        before_crop = os.path.join(out1, "before_crop.asc")
        after_crop = os.path.join(out1, "after_crop.asc")
        if not (os.path.exists(before_crop) and os.path.exists(after_crop)):
            self.log("[!] Click 'Confirm Input' before Run SAD (missing before_crop.asc / after_crop.asc).")
            QMessageBox.warning(self, "Missing data", "You must to click Confirm Input before Run SAD.")
            return

        cellsize_m = float(self.pixel_size_spin.value()) / 1000.0

        engine = self.sad_engine_combo.currentText() if hasattr(self, "sad_engine_combo") else "OpenCV (fast)"

        if "Original" in engine:
            status, outputs = run_UI1_2_sad_original(
                before_path=before_crop,
                after_path=after_crop,
                output_dir=out2,
                cellsize=cellsize_m
            )
        else:
            status, outputs = run_UI1_2_sad(
                before_path=before_crop,
                after_path=after_crop,
                output_dir=out2,
                cellsize=cellsize_m
            )

        self.log(f"{status} [Engine={engine}]")

        if outputs:
            dz_before = self.before_pz_file if self.before_pz_file else before_crop
            dz_after = self.after_pz_file if self.after_pz_file else after_crop

            status_dz, _ = run_UI1_4_dz(
                before_path=dz_before,
                after_path=dz_after,
                dx_path=outputs["dX_path"],
                dy_path=outputs["dY_path"],
                output_dir=out5dz,
                cellsize=cellsize_m
            )
            self.log(status_dz)

    def run_slip_zone(self):
        out2 = "output/UI1/step2_sad"
        out5dz = "output/UI1/step5_dz"
        out7 = "output/UI1/step7_slipzone"
        os.makedirs(out7, exist_ok=True)
    
        dxp = f"{out2}/dX.asc"
        dyp = f"{out2}/dY.asc"
        dzp = f"{out5dz}/dZ.asc"
    
        status, outputs = run_UI1_5_detect_slipzone(
            dxp, dyp, dzp,
            threshold_mm=self.threshold_spin.value(),
            output_dir=out7
        )
        self.log(status)
        if outputs:
            self.display_image(outputs["slip_mask_png"])
    
            try:
                from pedi_oku_landslide.pipeline.runners.ui1_backend import run_UI1_5_extract_boundary
                out8_dir = "output/UI1/step8_boundary"
                os.makedirs(out8_dir, exist_ok=True)
                boundary_gpkg = f"{out8_dir}/slip_boundary.gpkg"
                status_b, out_gpkg = run_UI1_5_extract_boundary(
                    slip_zone_path=f"{out7}/slip_zone.asc",
                    output_path=boundary_gpkg
                )
                self.log(status_b)
                self.log(f"[→] Boundary saved: {out_gpkg}")
            except Exception as e:
                self.log(f"[✗] Export boundary failed: {e}")

    def run_vector_overlay(self):
        out1 = "output/UI1/step1_crop"
        out2 = "output/UI1/step2_sad"
        out7 = "output/UI1/step7_slipzone"
        out4png = "output/UI1/step4_vector_geotiff.png"
    
        dem_path  = f"{out1}/before_crop.asc"
        dx_path   = f"{out2}/dX.asc"
        dy_path   = f"{out2}/dY.asc"
        slip_path = f"{out7}/slip_zone.asc"
    
        os.makedirs(os.path.dirname(out4png), exist_ok=True)
    
        status, image_path = run_UI1_3_plot_vector_geotiff(
            dem_path=dem_path,
            dx_path=dx_path,
            dy_path=dy_path,
            slip_zone_path=slip_path,
            output_png=out4png,
            vector_scale=self.vector_scale_spin.value(),
            stride=int(self.vector_stride_spin.value()),
            headwidth=self.vector_headwidth_spin.value(),
            headlength=self.vector_headlength_spin.value(),
            headaxislength=self.vector_headaxislength_spin.value(),
            vector_width=self.vector_width_spin.value(),
            vector_color=self.vector_color
        )
        self.log(status)
        self.display_image(image_path)

    def display_image(self, path):
        if path and os.path.exists(path):
            self.scene.clear()
            self.scene.addItem(QGraphicsPixmapItem(QPixmap(path)))
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def _add_labeled_spinbox(self, label, layout, default):
        box = QDoubleSpinBox(); box.setDecimals(4); box.setMaximum(99999999); box.setValue(default)
        layout.addWidget(QLabel(label)); layout.addWidget(box)
        return box
    
    def go_to_next(self):
        try:
            from .ui2_frontend import UI2App
        except Exception as e:
            self.log(f"[!] Can't access UI2 – import UI2App error: {e}")
            return
        try:
            self.log("[→] Switching to UI2 – Vector Overlay")
            json_path = os.path.join(self.workspace, "ui_shared_data.json")
            self.win2 = UI2App(json_path=json_path)
            self.win2.showMaximized()
            self.close()
        except Exception as e:
            self.log(f"[!] Failed to open UI2: {e}")

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    from PyQt5.QtGui import QFont
    scr = app.primaryScreen()
    dpi = scr.logicalDotsPerInch() if scr else 96.0
    base_pt = 12 if dpi > 120 else 11
    app.setFont(QFont("Public Sans", base_pt))
    win = UI1App()
    win.showMaximized()
    sys.exit(app.exec_())

