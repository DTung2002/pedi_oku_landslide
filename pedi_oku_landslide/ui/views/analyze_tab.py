# pedi_oku_landslide/ui/views/analyze_tab.py
from typing import Optional
from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QMessageBox, QGroupBox, QTextEdit, QSplitter, QSizePolicy,
    QDoubleSpinBox, QComboBox, QSpinBox, QFileDialog, QScrollArea, QGridLayout
)
import os, sys
from datetime import datetime
from pedi_oku_landslide.project.path_manager import (
    create_context, AnalysisContext, load_context_from_run_dir, is_valid_run_dir
)
from ..widgets.file_picker import FilePicker
from ..views.ui.ui1_viewer import UI1Viewer
from pedi_oku_landslide.pipeline.ingest import run_ingest
from pedi_oku_landslide.pipeline.steps.step_smooth import run_smooth
from pedi_oku_landslide.pipeline.steps.step_sad import run_sad
from pedi_oku_landslide.pipeline.steps.step_detect import run_detect, render_vectors


class AnalyzeTab(QWidget):
    # phát cho MainWindow khi đã render vectors xong → enable Section tab
    vectors_rendered = pyqtSignal(str, str, str)  # project, run_label, run_dir

    """
    Left: Project/Run + file pickers + Confirm Input + Processing + Status
    Right: UI1 preview (viewer with zoom toolbar inside)
    """

    def __init__(self, base_dir: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.base_dir = base_dir
        self._last_run_dir: Optional[str] = None
        self._splitter: Optional[QSplitter] = None
        self._left_min_w = 380
        self._left_default_w = 490
        self._pending_init_splitter = True
        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        self._splitter = splitter
        splitter.splitterMoved.connect(lambda *_: self._enforce_left_pane_bounds())
        root.addWidget(splitter)

        # ----- Left pane -----
        left_container = QWidget()
        left_container.setMinimumWidth(self._left_min_w)
        left_layout = QVBoxLayout(left_container)

        # Project group
        grp_proj = QGroupBox("Project")
        proj_layout = QGridLayout(grp_proj)
        proj_layout.setHorizontalSpacing(8)
        proj_layout.setVerticalSpacing(6)
        proj_layout.setColumnStretch(1, 1)

        lbl_name = QLabel("Name:")
        lbl_run = QLabel("Run label:")
        label_col_w = max(lbl_name.sizeHint().width(), lbl_run.sizeHint().width())
        proj_layout.setColumnMinimumWidth(0, label_col_w)

        proj_input_h = 30
        self.edit_project = QLineEdit()
        self.edit_project.setPlaceholderText("e.g. Jimba_01")
        self.edit_project.setFixedHeight(proj_input_h)
        self.edit_project.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_load_run = QPushButton("Load Run")
        self.btn_load_run.setToolTip("Open an existing run folder under output/<Project>/<RunID>")
        self.btn_load_run.clicked.connect(self._on_open_existing_run)
        self.edit_runlabel = QLineEdit()
        self.edit_runlabel.setPlaceholderText("e.g. baseline")
        self.edit_runlabel.setFixedHeight(proj_input_h)
        self.edit_runlabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Reserve the same right column width on both rows so inputs end next to Load Run.
        proj_layout.setColumnMinimumWidth(2, self.btn_load_run.sizeHint().width())
        proj_layout.addWidget(lbl_name, 0, 0)
        proj_layout.addWidget(self.edit_project, 0, 1)
        proj_layout.addWidget(self.btn_load_run, 0, 2)
        proj_layout.addWidget(lbl_run, 1, 0)
        proj_layout.addWidget(self.edit_runlabel, 1, 1)

        left_layout.addWidget(grp_proj)

        # Inputs group
        grp_inputs = QGroupBox("Inputs")
        inputs_layout = QVBoxLayout(grp_inputs)
        inputs_layout.setContentsMargins(6, 8, 6, 8)
        inputs_layout.setSpacing(6)

        self.fp_bdem = FilePicker("BEFORE DEM.tif", "GeoTIFF (*.tif *.tiff)")
        self.fp_basc = FilePicker("BEFORE.asc", "ASC (*.asc)")
        self.fp_aasc = FilePicker("AFTER.asc", "ASC (*.asc)")
        self.fp_bpz  = FilePicker("BEFORE_PZ.asc", "ASC (*.asc)")
        self.fp_apz  = FilePicker("AFTER_PZ.asc", "ASC (*.asc)")
        
        for w in (self.fp_bdem, self.fp_basc, self.fp_aasc, self.fp_bpz, self.fp_apz):
            inputs_layout.addWidget(w)
            
        # Actions row (inside Inputs panel)
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 2, 0, 0)
        actions.setSpacing(6)

        self.btn_confirm = QPushButton("Confirm Input")
        self.btn_confirm.clicked.connect(self._on_confirm_input)

        self.btn_open_run = QPushButton("Open run folder")
        self.btn_open_run.clicked.connect(self._on_open_run)
        self.btn_open_run.setEnabled(False)
        actions.addWidget(self.btn_confirm, 1)
        actions.addWidget(self.btn_open_run, 1)
        inputs_layout.addLayout(actions)

        # Keep action buttons at their natural size; only normalize file-picker buttons.
        file_buttons = [self.fp_bdem.btn, self.fp_basc.btn, self.fp_aasc.btn, self.fp_bpz.btn, self.fp_apz.btn]
        max_w = max(btn.sizeHint().width() for btn in file_buttons) + 36
        max_h = max(btn.sizeHint().height() for btn in file_buttons)
        for btn in file_buttons:
            btn.setFixedSize(max_w, max_h)

        left_layout.addWidget(grp_inputs)

        # ===================== Detect Landslide Zone (combined) =====================
        grp_detect = QGroupBox("Detect Landslide Zone")
        lay_detect = QVBoxLayout(grp_detect)

        lab_smooth = QLabel("Gaussian σ (px):")
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.0, 50.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.setDecimals(1)
        self.spin_sigma.setValue(2.0)
        self.spin_sigma.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_smooth = QPushButton("Smooth")
        self.btn_smooth.setEnabled(False)  # bật sau Confirm Input
        self.btn_smooth.clicked.connect(self._on_smooth)

        # ---- Calculate SAD ----
        lab_method = QLabel("SAD Method:")
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["OpenCV", "Traditional"])
        self.cmb_method.setCurrentText("OpenCV")
        self.cmb_method.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_calc_sad = QPushButton("Calculate")
        self.btn_calc_sad.setEnabled(False)  # bật sau Confirm Input
        self.btn_calc_sad.clicked.connect(self._on_calc_sad)

        # ---- Landslide Zone ----
        lab_detect = QLabel("Displacement (m):")
        self.spin_detect_thr = QDoubleSpinBox()
        self.spin_detect_thr.setRange(0.0, 10.0)
        self.spin_detect_thr.setSingleStep(0.1)
        self.spin_detect_thr.setValue(0.8)
        self.spin_detect_thr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_detect = QPushButton("Detect")
        self.btn_detect.setEnabled(False)  # bật sau SAD
        self.btn_detect.clicked.connect(self._on_detect)
        # Three aligned rows: label | input | action button.
        grid_detect = QGridLayout()
        grid_detect.setHorizontalSpacing(8)
        grid_detect.setVerticalSpacing(6)
        grid_detect.setColumnStretch(1, 1)
        grid_detect.addWidget(lab_smooth, 0, 0)
        grid_detect.addWidget(self.spin_sigma, 0, 1)
        grid_detect.addWidget(self.btn_smooth, 0, 2)
        grid_detect.addWidget(lab_method, 1, 0)
        grid_detect.addWidget(self.cmb_method, 1, 1)
        grid_detect.addWidget(self.btn_calc_sad, 1, 2)
        grid_detect.addWidget(lab_detect, 2, 0)
        grid_detect.addWidget(self.spin_detect_thr, 2, 1)
        grid_detect.addWidget(self.btn_detect, 2, 2)
        lay_detect.addLayout(grid_detect)
        # ---- Vectors Adjustment ----
        grp_vectors = QGroupBox("Vectors Adjustment")
        lay_vectors = QVBoxLayout(grp_vectors)
        grid_vec = QGridLayout()
        grid_vec.setHorizontalSpacing(8)
        grid_vec.setVerticalSpacing(6)
        grid_vec.setColumnStretch(1, 1)
        grid_vec.setColumnStretch(3, 1)

        lab_step = QLabel("Step:")
        lab_scale = QLabel("Scale:")
        lab_size = QLabel("Size:")
        lab_color = QLabel("Color:")

        self.spin_vec_step = QSpinBox()
        self.spin_vec_step.setRange(1, 200)
        self.spin_vec_step.setValue(25)  # mặc định gọn
        self.spin_vec_step.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.spin_vec_scale = QDoubleSpinBox()
        self.spin_vec_scale.setRange(0.01, 10.0)
        self.spin_vec_scale.setSingleStep(1.0)
        self.spin_vec_scale.setValue(1.0)  # theo mét
        self.spin_vec_scale.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.spin_vec_width = QDoubleSpinBox()
        self.spin_vec_width.setDecimals(4)
        self.spin_vec_width.setRange(0.0002, 0.02)
        self.spin_vec_width.setSingleStep(0.0002)
        self.spin_vec_width.setValue(0.001)
        self.spin_vec_width.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.combo_vec_color = QComboBox()
        self.combo_vec_color.addItems(["Blue", "Red", "Green", "Black", "Yellow", "Magenta"])
        self.combo_vec_color.setCurrentText("Blue")
        self.combo_vec_color.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Row 1: Step + Scale
        grid_vec.addWidget(lab_step, 0, 0)
        grid_vec.addWidget(self.spin_vec_step, 0, 1)
        grid_vec.addWidget(lab_scale, 0, 2)
        grid_vec.addWidget(self.spin_vec_scale, 0, 3)
        # Row 2: Size + Color
        grid_vec.addWidget(lab_size, 1, 0)
        grid_vec.addWidget(self.spin_vec_width, 1, 1)
        grid_vec.addWidget(lab_color, 1, 2)
        grid_vec.addWidget(self.combo_vec_color, 1, 3)
        lay_vectors.addLayout(grid_vec)

        row_v3 = QHBoxLayout()
        self.btn_vectors = QPushButton("Render Vectors")
        self.btn_vectors.setEnabled(False)  # bật sau SAD
        self.btn_vectors.clicked.connect(self._on_render_vectors)
        self.btn_vectors.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_v3.addWidget(self.btn_vectors)
        lay_vectors.addLayout(row_v3)
        # Force same width as "Detect Landslide Zone" for action buttons.
        btn_detect_w = max(
            self.btn_detect.sizeHint().width(),
            self.btn_smooth.sizeHint().width(),
            self.btn_calc_sad.sizeHint().width(),
            120,  # keep labels fully visible on narrow panes
        )
        self.btn_smooth.setFixedWidth(btn_detect_w)
        self.btn_calc_sad.setFixedWidth(btn_detect_w)
        self.btn_detect.setFixedWidth(btn_detect_w)
        left_layout.addWidget(grp_detect)
        left_layout.addWidget(grp_vectors)

        # Status group
        grp_status = QGroupBox("Status")
        status_layout = QVBoxLayout(grp_status)
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(120)
        status_layout.addWidget(self.status_text)
        left_layout.addWidget(grp_status)

        left_layout.addStretch(1)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(self._left_min_w)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setWidget(left_container)
        splitter.addWidget(left_scroll)

        # ----- Right pane: Preview -----
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)

        # title = QLabel("Hillshade Preview (Before / After)")
        # title.setStyleSheet("font-weight: 600;")
        # right_layout.addWidget(title)

        self.viewer = UI1Viewer(right_container)
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.viewer, 1)

        splitter.addWidget(right_container)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([self._left_default_w, 900])

        self._apply_button_style(self)

    def _left_max_w(self) -> int:
        # Left pane can occupy at most 50% of current window width.
        base_w = self.width()
        if self._splitter is not None and self._splitter.width() > 0:
            base_w = self._splitter.width()
        # During early layout, widths can be tiny/unstable; defer clamping then.
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

    def _apply_button_style(self, container: QWidget | None = None) -> None:
        """Áp style chung cho tab Analyze (nền trắng + nút xanh)."""
        target = container or self
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
        # target.setStyleSheet(style)

    def _set_project_run_locked(self, locked: bool) -> None:
        """Khóa/Mở 2 ô Project & Run label."""
        self.edit_project.setReadOnly(locked)
        self.edit_project.setStyleSheet("background:#f3f3f3;" if locked else "")
        self.edit_runlabel.setReadOnly(locked)
        self.edit_runlabel.setStyleSheet("background:#f3f3f3;" if locked else "")

    @staticmethod
    def _parse_label_from_run_id(run_id: str) -> str:
        # RunID format: YYYYmmdd_HHMMSS or YYYYmmdd_HHMMSS_<label>
        parts = run_id.split("_", 2)
        return parts[2] if len(parts) == 3 else ""

    # ---------- Actions ----------
    def _on_confirm_input(self) -> None:
        project = (self.edit_project.text() or "").strip()
        run_label = (self.edit_runlabel.text() or "").strip()

        if not project:
            self._warn("Project name is required.")
            return

        files = {
            "before_dem": self.fp_bdem.path,
            "before_asc": self.fp_basc.path,
            "after_asc":  self.fp_aasc.path,
            "before_pz":  self.fp_bpz.path,
            "after_pz":   self.fp_apz.path,
        }
        if not all(files.values()):
            self._warn("Please select all 5 input files.")
            return

        try:
            ctx = create_context(self.base_dir, project, run_label if run_label else None)
            info = run_ingest(ctx, files)

            self._last_run_dir = info.get("run_dir")

            self.btn_open_run.setEnabled(True)
            self.btn_smooth.setEnabled(True)  # bật smooth sau khi confirm
            self.btn_calc_sad.setEnabled(True)

            self._ok(
                "Confirm Input completed.\n"
                f"Project: {ctx.project_id}\n"
                f"Run: {ctx.run_id}\n"
                f"Output: {self._last_run_dir}"
            )

            preview = info.get("preview", {})
            self.viewer.show_pair(
                preview.get("before_asc_hillshade_png"),
                preview.get("after_asc_hillshade_png"),
            )

        except FileExistsError:
            self._err("Run folder already exists (rare). Please try again.")
        except Exception as e:
            self._err(f"Error: {e}")

    def _on_smooth(self) -> None:
        if not self._last_run_dir:
            self._warn("Please run 'Confirm Input' first.")
            return
        try:
            sigma = float(self.spin_sigma.value())

            # reconstruct ctx from self._last_run_dir
            parts = os.path.normpath(self._last_run_dir).split(os.sep)
            if len(parts) < 2:
                raise RuntimeError("Unexpected run folder structure.")
            run_id = parts[-1]
            project = parts[-2]

            ctx = AnalysisContext(
                project_id=project,
                run_id=run_id,
                base_dir=self.base_dir,
                project_dir=os.path.join(self.base_dir, "output", project),
                run_dir=self._last_run_dir,
                in_dir=os.path.join(self._last_run_dir, "input"),
                out_ui1=os.path.join(self._last_run_dir, "ui1"),
                out_ui2=os.path.join(self._last_run_dir, "ui2"),
                out_ui3=os.path.join(self._last_run_dir, "ui3"),
            )

            out = run_smooth(ctx, sigma_px=sigma)

            self._ok(
                f"Smooth completed (σ={sigma}px).\n"
                f"Outputs:\n"
                f" - {out['before_tif']}\n"
                f" - {out['after_tif']}"
            )
            self.viewer.show_pair(out["before_png"], out["after_png"])

        except Exception as e:
            self._err(f"Smooth error: {e}")

    def _on_calc_sad(self) -> None:
        if not self._last_run_dir:
            self._warn("Please run 'Confirm Input' first.")
            return
        try:
            # reconstruct ctx từ run_dir
            parts = os.path.normpath(self._last_run_dir).split(os.sep)
            run_id = parts[-1]
            project = parts[-2]

            ctx = AnalysisContext(
                project_id=project,
                run_id=run_id,
                base_dir=self.base_dir,
                project_dir=os.path.join(self.base_dir, "output", project),
                run_dir=self._last_run_dir,
                in_dir=os.path.join(self._last_run_dir, "input"),
                out_ui1=os.path.join(self._last_run_dir, "ui1"),
                out_ui2=os.path.join(self._last_run_dir, "ui2"),
                out_ui3=os.path.join(self._last_run_dir, "ui3"),
            )

            method = self.cmb_method.currentText().lower()  # "opencv" | "traditional"

            # --- disable nút khi đang chạy ---
            self.btn_calc_sad.setEnabled(False)
            self.btn_detect.setEnabled(False)
            self.btn_vectors.setEnabled(False)
            self._info("Calculating SAD + dZ in background...")

            # --- chạy nền bằng QThread ---
            self._sad_thread = QThread(self)
            self._sad_worker = _SadWorker(
                ctx=ctx,
                method=method,
                patch_size_m=20.0,
                search_radius_m=2.0,
                use_smoothed=True
            )
            self._sad_worker.moveToThread(self._sad_thread)

            self._sad_thread.started.connect(self._sad_worker.run)
            self._sad_worker.finished.connect(self._on_sad_done)
            self._sad_worker.error.connect(self._on_sad_error)
            self._sad_worker.error.connect(self._sad_thread.quit)
            self._sad_worker.error.connect(self._sad_worker.deleteLater)
            self._sad_worker.finished.connect(self._sad_thread.quit)
            self._sad_worker.finished.connect(self._sad_worker.deleteLater)
            self._sad_thread.finished.connect(self._sad_thread.deleteLater)

            self._sad_thread.start()

        except Exception as e:
            self._err(f"SAD+dZ error: {e}")

    def _on_sad_done(self, out: dict, method: str) -> None:
        try:
            # bật lại nút
            self.btn_calc_sad.setEnabled(True)
            self.btn_detect.setEnabled(True)
            self.btn_vectors.setEnabled(True)

            # thông báo
            self._ok(
                f"SAD + dZ completed (method: {method}).\n"
                f"- dX: {out.get('dx_tif', '')}\n"
                f"- dY: {out.get('dy_tif', '')}\n"
                f"- dZ: {out.get('dz_tif', '')}"
            )

            # hiển thị mặc định sau khi xong
            overlay = os.path.join(self._last_run_dir, "ui1", "landslide_overlay.png")
            vectors = os.path.join(self._last_run_dir, "ui1", "vectors_overlay.png")
            dz_png = os.path.join(self._last_run_dir, "ui1", "dz.png")

            left_img = overlay if os.path.exists(overlay) else dz_png
            right_img = vectors if os.path.exists(vectors) else dz_png

            self.viewer.show_pair(left_img, right_img)

        except Exception as e:
            self._err(f"Post-process error: {e}")

    def _on_sad_error(self, msg: str) -> None:
        self.btn_calc_sad.setEnabled(True)
        self.btn_detect.setEnabled(True)
        self.btn_vectors.setEnabled(True)
        self._err(f"SAD+dZ error: {msg}")

    def _on_open_run(self) -> None:
        if not self._last_run_dir:
            self._warn("No run folder to open.")
            return
        try:
            if os.name == "nt":
                os.startfile(self._last_run_dir)  # type: ignore[attr-defined]
            else:
                import subprocess
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.Popen([opener, self._last_run_dir])
        except Exception as e:
            self._err(f"Cannot open folder: {e}")

    # ---------- Status helpers ----------
    def _append_status(self, text: str) -> None:
        """
        Ghi 1 dòng vào khung Status; nếu chưa có self.status_text thì in ra console.
        Tự động cuộn xuống cuối.
        """
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        box = getattr(self, "status_text", None)
        if box is not None:
            try:
                box.append(line)
                box.moveCursor(box.textCursor().End)
                return
            except Exception:
                pass
        print(line)

    def _info(self, msg: str) -> None:
        self._append_status(f"INFO: {msg}")

    def _ok(self, msg: str) -> None:
        self._append_status(f"OK: {msg}")

    def _warn(self, msg: str) -> None:
        self._append_status(f"WARN: {msg}")
        QMessageBox.warning(self, "Warning", msg)

    def _err(self, msg: str) -> None:
        self._append_status(f"ERROR: {msg}")
        QMessageBox.critical(self, "Error", msg)

    def _on_detect(self) -> None:
        if not self._last_run_dir:
            self._warn("Please run 'Confirm Input' first.")
            return
        try:
            # reconstruct ctx
            parts = os.path.normpath(self._last_run_dir).split(os.sep)
            if len(parts) < 2:
                raise RuntimeError("Unexpected run folder structure.")
            run_id = parts[-1]
            project = parts[-2]

            ctx = AnalysisContext(
                project_id=project,
                run_id=run_id,
                base_dir=self.base_dir,
                project_dir=os.path.join(self.base_dir, "output", project),
                run_dir=self._last_run_dir,
                in_dir=os.path.join(self._last_run_dir, "input"),
                out_ui1=os.path.join(self._last_run_dir, "ui1"),
                out_ui2=os.path.join(self._last_run_dir, "ui2"),
                out_ui3=os.path.join(self._last_run_dir, "ui3"),
            )

            thr_m = float(self.spin_detect_thr.value())
            out = run_detect(ctx, method="threshold", threshold_m=thr_m)

            self._ok(f"Detected with threshold = {thr_m:.2f} m\n - {out['mask_tif']}")
            dz_png = os.path.join(ctx.out_ui1, "dz.png")
            overlay = out["mask_png"]  # bước detect đã lưu heatmap overlay
            self.viewer.show_pair(dz_png, overlay)

        except Exception as e:
            self._err(f"Detect error: {e}")

    def _on_render_vectors(self) -> None:
        if not self._last_run_dir:
            self._warn("Please run 'Confirm Input' first.")
            return
        try:
            step = int(self.spin_vec_step.value())
            scale = float(self.spin_vec_scale.value())
            color = str(self.combo_vec_color.currentText())
            width = float(self.spin_vec_width.value())

            # reconstruct ctx
            parts = os.path.normpath(self._last_run_dir).split(os.sep)
            if len(parts) < 2:
                raise RuntimeError("Unexpected run folder structure.")
            run_id = parts[-1]
            project = parts[-2]

            ctx = AnalysisContext(
                project_id=project,
                run_id=run_id,
                base_dir=self.base_dir,
                project_dir=os.path.join(self.base_dir, "output", project),
                run_dir=self._last_run_dir,
                in_dir=os.path.join(self._last_run_dir, "input"),
                out_ui1=os.path.join(self._last_run_dir, "ui1"),
                out_ui2=os.path.join(self._last_run_dir, "ui2"),
                out_ui3=os.path.join(self._last_run_dir, "ui3"),
            )

            out = render_vectors(
                ctx,
                step=step,
                scale=scale,
                vector_color=color,
                vector_width=width,
            )

            # Hiển thị: trái = overlay (nếu có) hoặc dz, phải = vectors overlay
            overlay = os.path.join(ctx.out_ui1, "landslide_overlay.png")
            left_img = overlay if os.path.exists(overlay) else os.path.join(ctx.out_ui1, "dz.png")
            self.viewer.show_pair(left_img, out["vectors_png"])

            self._ok(f"Vectors rendered (step={step}, scale={scale}).")

            # emit để MainWindow enable tab Section Selection
            project = (self.edit_project.text() or "").strip()
            run_label = (self.edit_runlabel.text() or "").strip()
            self.vectors_rendered.emit(project, run_label, self._last_run_dir)

        except Exception as e:
            self._err(f"Render vectors error: {e}")

    def _on_open_existing_run(self) -> None:
        """
        Let user pick an existing run folder under output/<Project>/<RunID>,
        then load context, enable actions, and show previews.
        """
        try:
            root = os.path.join(self.base_dir, "output")
            sel = QFileDialog.getExistingDirectory(self, "Open existing run", root)
            if not sel:
                return

            # Validate selection
            if not is_valid_run_dir(sel):
                # Nếu user chọn nhầm thư mục project
                proj_candidate = os.path.join(sel, "input")
                if os.path.isdir(proj_candidate):
                    self._err(f"Selected folder is not a run folder:\n{sel}\n"
                              f"It must be .../output/<Project>/<RunID> and contain 'input' and 'ui*'.")
                    return
                self._err("Please select a run folder like:\n.../output/<Project>/<YYYYmmdd_HHMMSS[_label]>")
                return

            # Build context
            ctx = load_context_from_run_dir(self.base_dir, sel)
            self._last_run_dir = ctx.run_dir

            # Fill fields
            self.edit_project.setText(ctx.project_id)
            self.edit_runlabel.setText(self._parse_label_from_run_id(ctx.run_id))

            # Khóa 2 ô khi đã load run
            self._set_project_run_locked(True)
            self._ok("Fields locked (loaded run).")

            # Enable buttons based on what exists
            self.btn_smooth.setEnabled(True)
            self.btn_calc_sad.setEnabled(True)

            dx_ok = os.path.exists(os.path.join(ctx.out_ui1, "dx.tif"))
            dy_ok = os.path.exists(os.path.join(ctx.out_ui1, "dy.tif"))
            self.btn_detect.setEnabled(dx_ok and dy_ok)
            self.btn_vectors.setEnabled(dx_ok and dy_ok)

            # Bật Open run
            self.btn_open_run.setEnabled(True)

            # Show best available preview
            before_hs = os.path.join(ctx.out_ui1, "before_asc_hillshade.png")
            after_hs = os.path.join(ctx.out_ui1, "after_asc_hillshade.png")
            dz_png = os.path.join(ctx.out_ui1, "dz.png")
            overlay = os.path.join(ctx.out_ui1, "landslide_overlay.png")
            vectors = os.path.join(ctx.out_ui1, "vectors_overlay.png")

            def _exists(p): return os.path.isfile(p)

            left_img = right_img = None
            if _exists(dz_png) and _exists(overlay):
                left_img, right_img = dz_png, overlay
            elif _exists(before_hs) and _exists(after_hs):
                left_img, right_img = before_hs, after_hs
            elif _exists(dz_png) and _exists(vectors):
                left_img, right_img = dz_png, vectors
            elif _exists(dz_png):
                left_img, right_img = dz_png, dz_png

            if left_img and right_img:
                self.viewer.show_pair(left_img, right_img)

            self._ok(
                "Opened existing run.\n"
                f"Project: {ctx.project_id}\n"
                f"Run:     {ctx.run_id}\n"
                f"Folder:  {ctx.run_dir}"
            )

            # Nếu run đã có SAD (dx/dy) → enable UI2 ngay
            if dx_ok and dy_ok:
                project = ctx.project_id
                run_label = (self.edit_runlabel.text() or "").strip()
                self.vectors_rendered.emit(project, run_label, ctx.run_dir)

        except Exception as e:
            self._err(f"Open error: {e}")

    def reset_session(self) -> None:
        """
        Đưa tab Analyze (UI1) về trạng thái ban đầu cho New Session.
        Được MainWindow gọi khi user tạo session mới.
        """
        # 0) Best-effort dừng thread SAD nếu còn chạy
        if hasattr(self, "_sad_thread"):
            try:
                if self._sad_thread and self._sad_thread.isRunning():
                    self._sad_thread.requestInterruption()
                    self._sad_thread.quit()
                    self._sad_thread.wait(2000)
            except Exception:
                pass

        # 1) Quên run hiện tại
        self._last_run_dir = None

        # 2) Mở khoá & clear Project / Run label
        self._set_project_run_locked(False)
        self.edit_project.clear()
        self.edit_runlabel.clear()

        # 3) Clear 5 ô input (FilePicker)
        for fp in (self.fp_bdem, self.fp_basc, self.fp_aasc, self.fp_bpz, self.fp_apz):
            try:
                if hasattr(fp, "clear"):
                    fp.clear()
                elif hasattr(fp, "set_path"):
                    fp.set_path("")
                elif hasattr(fp, "edit_path"):
                    fp.edit_path.clear()
            except Exception:
                # Không fatal, chỉ cố gắng hết sức
                pass

        # 4) Đưa các nút về trạng thái ban đầu
        self.btn_open_run.setEnabled(False)
        self.btn_smooth.setEnabled(False)
        self.btn_calc_sad.setEnabled(False)
        self.btn_detect.setEnabled(False)
        self.btn_vectors.setEnabled(False)
        # Nút Confirm Input luôn bật
        self.btn_confirm.setEnabled(True)

        # 5) Reset các thông số xử lý
        try:
            self.spin_sigma.setValue(2.0)
        except Exception:
            pass
        try:
            self.cmb_method.setCurrentText("OpenCV")
        except Exception:
            pass
        try:
            self.spin_detect_thr.setValue(0.8)
        except Exception:
            pass
        try:
            self.spin_vec_step.setValue(25)
        except Exception:
            pass
        try:
            self.spin_vec_scale.setValue(1.0)
        except Exception:
            pass

        # 6) Clear viewer (ảnh bên phải)
        try:
            if hasattr(self.viewer, "scene"):
                self.viewer.scene.clear()
            if hasattr(self.viewer, "caption"):
                self.viewer.caption.setText("")
            if hasattr(self.viewer, "view") and hasattr(self.viewer.view, "set_zoom_100"):
                self.viewer.view.set_zoom_100()
        except Exception:
            pass

        # 7) Clear status log
        try:
            self.status_text.clear()
        except Exception:
            pass

        # 8) Thông báo nhẹ để debug
        try:
            self._info("Session reset.")
        except Exception:
            pass

from time import perf_counter

class _SadWorker(QObject):
    finished = pyqtSignal(dict, str)  # (out_dict, method)
    error = pyqtSignal(str)
    t0 = perf_counter()
    def __init__(self, ctx, method: str, patch_size_m: float, search_radius_m: float, use_smoothed: bool):
        super().__init__()
        self.ctx = ctx
        self.method = method
        self.patch_size_m = patch_size_m
        self.search_radius_m = search_radius_m
        self.use_smoothed = use_smoothed

    def run(self):
        try:
            t0 = perf_counter()
            print("[SAD] run_sad start")
            out = run_sad(
                self.ctx,
                method=self.method,
                patch_size_m=self.patch_size_m,
                search_radius_m=self.search_radius_m,
                use_smoothed=self.use_smoothed,
                vlim_dz=None
            )
            dt = perf_counter() - t0
            print(f"[SAD] run_sad done in {dt:.2f}s")
            self.finished.emit(out, self.method)
        except Exception as e:
            self.error.emit(str(e))
