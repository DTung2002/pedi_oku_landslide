# pedi_oku_landslide/ui/views/ui1_frontend_impl.py
from typing import Optional
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QMessageBox, QGroupBox, QTextEdit, QSplitter, QSizePolicy,
    QDoubleSpinBox, QComboBox, QSpinBox, QFileDialog, QScrollArea, QGridLayout, QSlider, QFrame
)
import os, sys
from datetime import datetime
from ..widgets.file_picker import FilePicker
from .ui1_viewer import UI1Viewer
from .ui1_workers import UI1SadWorker
from pedi_oku_landslide.pipeline.runners.ui1_backend import AnalysisContext, UI1BackendService
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
        self._backend = UI1BackendService(base_dir)
        self._last_run_dir: Optional[str] = None
        self._splitter: Optional[QSplitter] = None
        self._left_min_w = LEFT_MIN_W
        self._left_default_w = LEFT_DEFAULT_W
        self._pending_init_splitter = True
        self._vec_live_timer = QTimer(self)
        self._vec_live_timer.setSingleShot(True)
        self._vec_live_timer.setInterval(80)
        self._vec_live_timer.timeout.connect(self._on_vec_live_tick)
        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(*ROOT_MARGINS)
        root.setSpacing(ROOT_SPACING)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        self._splitter = splitter
        splitter.splitterMoved.connect(lambda *_: self._enforce_left_pane_bounds())
        root.addWidget(splitter)

        # ----- Left pane -----
        left_container = QWidget()
        left_container.setMinimumWidth(self._left_min_w)
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(*LEFT_MARGINS)
        left_layout.setSpacing(PANEL_SPACING)

        # Project group
        grp_proj = QGroupBox("Project")
        proj_layout = QGridLayout(grp_proj)
        proj_layout.setContentsMargins(*PROJECT_MARGINS)
        proj_layout.setHorizontalSpacing(PROJECT_H_SPACING)
        proj_layout.setVerticalSpacing(PROJECT_V_SPACING)
        proj_layout.setColumnStretch(1, 1)

        lbl_name = QLabel("Name:")
        lbl_run = QLabel("Run label:")
        lbl_name.setFixedWidth(PROJECT_LABEL_W)
        lbl_run.setFixedWidth(PROJECT_LABEL_W)
        proj_layout.setColumnMinimumWidth(0, PROJECT_LABEL_W)

        self.edit_project = QLineEdit()
        self.edit_project.setPlaceholderText("e.g. Jimba_01")
        self.edit_project.setFixedHeight(CONTROL_HEIGHT)
        self.edit_project.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_load_run = QPushButton("Load Run")
        self.btn_load_run.setFixedHeight(CONTROL_HEIGHT)
        self.btn_load_run.setToolTip("Open an existing run folder under output/<Project>/<RunID>")
        self.btn_load_run.clicked.connect(self._on_open_existing_run)
        self.edit_runlabel = QLineEdit()
        self.edit_runlabel.setPlaceholderText("e.g. baseline")
        self.edit_runlabel.setFixedHeight(CONTROL_HEIGHT)
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

        self.fp_adem = FilePicker("AFTER DEM.tif", "GeoTIFF (*.tif *.tiff)")
        self.fp_basc = FilePicker("BEFORE.asc", "ASC (*.asc)")
        self.fp_aasc = FilePicker("AFTER.asc", "ASC (*.asc)")
        self.fp_bpz  = FilePicker("BEFORE_PZ.asc", "ASC (*.asc)")
        self.fp_apz  = FilePicker("AFTER_PZ.asc", "ASC (*.asc)")
        self.fp_mask_dxf = FilePicker("Boundary.dxf", "DXF (*.dxf)")
        
        for w in (self.fp_adem, self.fp_basc, self.fp_aasc, self.fp_bpz, self.fp_apz):
            inputs_layout.addWidget(w)

        row_mask = QHBoxLayout()
        row_mask.setContentsMargins(0, 0, 0, 0)
        row_mask.setSpacing(6)
        row_mask.addWidget(self.fp_mask_dxf, 1)
        inputs_layout.addLayout(row_mask)
            
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
        file_buttons = [self.fp_adem.btn, self.fp_basc.btn, self.fp_aasc.btn, self.fp_bpz.btn, self.fp_apz.btn, self.fp_mask_dxf.btn]
        max_w = max(btn.sizeHint().width() for btn in file_buttons) + 36
        max_h = max(btn.sizeHint().height() for btn in file_buttons)
        for btn in file_buttons:
            btn.setFixedSize(max_w, max_h)

        left_layout.addWidget(grp_inputs)

        # ===================== Detect Landslide Zone (combined) =====================
        grp_detect = QGroupBox("Detect Landslide Zone")
        lay_detect = QVBoxLayout(grp_detect)

        # ---- Smooth ----
        self.lab_smooth_param = QLabel("Gaussian filter (m):")
        self.spin_smooth_param = QDoubleSpinBox()
        self.spin_smooth_param.setRange(0.0, 50.0)
        self.spin_smooth_param.setSingleStep(0.5)
        self.spin_smooth_param.setDecimals(1)
        self.spin_smooth_param.setValue(2.0)
        self.spin_smooth_param.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_smooth = QPushButton("Smooth")
        self.btn_smooth.setEnabled(False)  # bật sau Confirm Input
        self.btn_smooth.clicked.connect(self._on_smooth)

        # ---- Calculate SAD ----
        lab_method = QLabel("Template Matching:")
        self.cmb_method = QComboBox()
        self.cmb_method.addItem("SAD", "traditional")
        self.cmb_method.addItem("SSD (OpenCV)", "ssd_opencv")
        self.cmb_method.setCurrentIndex(0)
        self.cmb_method.setEnabled(False)
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
        self.spin_detect_thr.setFixedWidth(90)
        self.cmb_detect_source = QComboBox()
        self.cmb_detect_source.addItem("Displacement", "displacement")
        self.cmb_detect_source.addItem("DXF", "dxf")
        self.cmb_detect_source.setEnabled(False)
        self.cmb_detect_source.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_detect = QPushButton("Detect")
        self.btn_detect.setEnabled(False)  # bật sau SAD
        self.btn_detect.clicked.connect(self._on_detect_requested)

        detect_buttons = (self.btn_smooth, self.btn_calc_sad, self.btn_detect)
        detect_button_w = max(btn.sizeHint().width() for btn in detect_buttons) + 16
        for btn in detect_buttons:
            btn.setFixedSize(detect_button_w, CONTROL_HEIGHT)

        row_detect_inputs = QWidget()
        row_detect_inputs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_detect_inputs_lay = QHBoxLayout(row_detect_inputs)
        row_detect_inputs_lay.setContentsMargins(0, 0, 0, 0)
        row_detect_inputs_lay.setSpacing(6)
        row_detect_inputs_lay.addWidget(self.spin_detect_thr)
        row_detect_inputs_lay.addWidget(self.cmb_detect_source, 1)

        # Aligned rows: label | input | action button.
        grid_detect = QGridLayout()
        grid_detect.setHorizontalSpacing(8)
        grid_detect.setVerticalSpacing(6)
        grid_detect.setColumnStretch(1, 1)
        grid_detect.addWidget(self.lab_smooth_param, 0, 0)
        grid_detect.addWidget(self.spin_smooth_param, 0, 1)
        grid_detect.addWidget(self.btn_smooth, 0, 2)
        grid_detect.addWidget(lab_method, 1, 0)
        grid_detect.addWidget(self.cmb_method, 1, 1)
        grid_detect.addWidget(self.btn_calc_sad, 1, 2)
        grid_detect.addWidget(lab_detect, 2, 0)
        grid_detect.addWidget(row_detect_inputs, 2, 1)
        grid_detect.addWidget(self.btn_detect, 2, 2)
        lay_detect.addLayout(grid_detect)

        self.lbl_mask_source = QLabel("Mask source: not set")
        self.lbl_mask_source.setWordWrap(True)
        self.lbl_mask_source.setVisible(False)

        # ---- Vector Display ----
        grp_vectors = QGroupBox("Vector Display")
        lay_vectors = QVBoxLayout(grp_vectors)
        vec_actions = QHBoxLayout()
        vec_actions.setContentsMargins(0, 0, 0, 0)
        vec_actions.setSpacing(6)
        self.btn_vec_options = QPushButton("Display ˅")
        self.btn_vec_options.setCheckable(True)
        self.btn_vec_options.setChecked(False)
        self.btn_vec_options.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_vectors = QPushButton("Render Vectors")
        self.btn_vectors.setEnabled(False)  # bật sau SAD
        self.btn_vectors.clicked.connect(self._on_render_vectors)
        self.btn_vectors.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        vec_actions.addWidget(self.btn_vec_options, 1)
        vec_actions.addWidget(self.btn_vectors, 1)
        lay_vectors.addLayout(vec_actions)

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

        lab_step = QLabel("Step:")
        lab_scale = QLabel("Scale:")
        lab_color = QLabel("Color:")
        lab_size = QLabel("Size:")
        lab_opacity = QLabel("Opacity:")

        self.spin_vec_step = QSpinBox()
        self.spin_vec_step.setRange(1, 200)
        self.spin_vec_step.setValue(25)  # mặc định gọn
        self.spin_vec_step.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.spin_vec_scale = QDoubleSpinBox()
        self.spin_vec_scale.setRange(0.01, 10.0)
        self.spin_vec_scale.setSingleStep(1.0)
        self.spin_vec_scale.setValue(1.0)  # theo mét
        self.spin_vec_scale.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Size slider giống UI2
        self.sld_vec_size = QSlider(Qt.Horizontal)
        self.sld_vec_size.setRange(80, 500)
        self.sld_vec_size.setValue(100)
        self.sld_vec_size.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Opacity slider giống UI2
        self.sld_vec_opacity = QSlider(Qt.Horizontal)
        self.sld_vec_opacity.setRange(0, 100)
        self.sld_vec_opacity.setValue(100)
        self.sld_vec_opacity.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.sld_vec_size.valueChanged.connect(lambda _v: self._on_vec_display_slider_changed())
        self.sld_vec_opacity.valueChanged.connect(lambda _v: self._on_vec_display_slider_changed())

        self.combo_vec_color = QComboBox()
        self.combo_vec_color.addItems(["Blue", "Red", "Green", "White", "Yellow", "Magenta"])
        self.combo_vec_color.setCurrentText("Blue")
        self.combo_vec_color.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Step/Scale/Color each on its own row, with aligned labels.
        label_col_w = max(lab_step.sizeHint().width(), lab_scale.sizeHint().width(), lab_color.sizeHint().width())
        grid_vec_top.setColumnMinimumWidth(0, label_col_w)
        input_min_w = max(
            self.spin_vec_step.sizeHint().width(),
            self.spin_vec_scale.sizeHint().width(),
            self.combo_vec_color.sizeHint().width(),
        )
        for w in (self.spin_vec_step, self.spin_vec_scale, self.combo_vec_color):
            w.setMinimumWidth(input_min_w)

        grid_vec_top.addWidget(lab_step, 0, 0)
        grid_vec_top.addWidget(self.spin_vec_step, 0, 1)
        grid_vec_top.addWidget(lab_scale, 1, 0)
        grid_vec_top.addWidget(self.spin_vec_scale, 1, 1)
        grid_vec_top.addWidget(lab_color, 2, 0)
        grid_vec_top.addWidget(self.combo_vec_color, 2, 1)
        vec_options_layout.addLayout(grid_vec_top)

        grid_vec_sliders = QGridLayout()
        grid_vec_sliders.setHorizontalSpacing(8)
        grid_vec_sliders.setVerticalSpacing(6)
        grid_vec_sliders.setColumnStretch(1, 1)
        grid_vec_sliders.addWidget(lab_size, 0, 0)
        grid_vec_sliders.addWidget(self.sld_vec_size, 0, 1)
        grid_vec_sliders.addWidget(lab_opacity, 1, 0)
        grid_vec_sliders.addWidget(self.sld_vec_opacity, 1, 1)
        vec_options_layout.addLayout(grid_vec_sliders)
        lay_vectors.addWidget(self.vec_options_panel)

        left_layout.addWidget(grp_detect)
        left_layout.addWidget(grp_vectors)

        # Status group
        grp_status = QGroupBox("Status")
        status_layout = QVBoxLayout(grp_status)
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setFixedHeight(200)
        status_layout.addWidget(self.status_text)
        left_layout.addWidget(grp_status)

        left_layout.addStretch(1)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        left_scroll.setMinimumWidth(self._left_min_w)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setWidget(left_container)
        splitter.addWidget(left_scroll)

        # ----- Right pane: Preview -----
        right_container = QWidget()
        right_container.setMinimumWidth(RIGHT_MIN_W)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(*RIGHT_MARGINS)
        right_layout.setSpacing(PANEL_SPACING)

        # title = QLabel("Hillshade Preview (Before / After)")
        # title.setStyleSheet("font-weight: 600;")
        # right_layout.addWidget(title)

        self.viewer = UI1Viewer(right_container)
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        fit_row = QHBoxLayout()
        fit_row.setContentsMargins(0, 0, 0, 0)
        fit_row.addStretch(1)
        fit_row.addWidget(self.viewer.btn_zoom_fit)
        right_layout.addLayout(fit_row)
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
        base_w = self._splitter.width() if self._splitter.width() > 0 else self.width()
        if base_w < (self._left_default_w * 2):
            return
        max_w = self._left_max_w()
        if max_w < 0:
            return
        init_left = max(self._left_min_w, min(self._left_default_w, max_w))
        total = max(self._splitter.width(), sum(self._splitter.sizes()), self.width(), init_left + 1)
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
        if self._pending_init_splitter:
            QTimer.singleShot(0, self._enforce_left_pane_bounds)
        self._enforce_left_pane_bounds()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._pending_init_splitter:
            QTimer.singleShot(0, self._enforce_left_pane_bounds)
            QTimer.singleShot(80, self._enforce_left_pane_bounds)
            QTimer.singleShot(180, self._enforce_left_pane_bounds)
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

    def _refresh_mask_source_ui(self, run_dir: Optional[str] = None) -> None:
        rd = (run_dir or self._last_run_dir or "").strip()
        info = self._backend.mask_source_info(rd)
        self.lbl_mask_source.setText(str(info.get("label") or "Mask source: not set"))
        dxf_path = str(info.get("dxf_path") or "").strip()
        if dxf_path:
            try:
                self.fp_mask_dxf.set_path(dxf_path)
            except Exception:
                pass

    # ---------- Actions ----------
    def _on_confirm_input(self) -> None:
        project = (self.edit_project.text() or "").strip()
        run_label = (self.edit_runlabel.text() or "").strip()

        if not project:
            self._warn("Project name is required.")
            return

        files = {
            "after_dem":  self.fp_adem.path,
            "before_asc": self.fp_basc.path,
            "after_asc":  self.fp_aasc.path,
            "before_pz":  self.fp_bpz.path,
            "after_pz":   self.fp_apz.path,
        }
        if not all(files.values()):
            self._warn("Please select all 5 input files.")
            return

        try:
            ctx, info = self._backend.confirm_input(project, run_label if run_label else None, files)

            self._last_run_dir = info.get("run_dir")

            self.btn_open_run.setEnabled(True)
            self.btn_smooth.setEnabled(True)  # bật smooth sau khi confirm
            self.cmb_method.setEnabled(True)
            self.btn_calc_sad.setEnabled(True)
            self.btn_detect.setEnabled(False)
            self.cmb_detect_source.setEnabled(False)
            self.btn_vectors.setEnabled(False)

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
            self._refresh_mask_source_ui(self._last_run_dir)

        except FileExistsError:
            self._err("Run folder already exists (rare). Please try again.")
        except Exception as e:
            self._err(f"Error: {e}")

    def _on_smooth(self) -> None:
        if not self._last_run_dir:
            self._warn("Please run 'Confirm Input' first.")
            return
        try:
            method = "Gaussian"
            param_m = float(self.spin_smooth_param.value())

            ctx = self._backend.context_from_run_dir(self._last_run_dir)
            out = self._backend.run_smooth(
                ctx,
                param_m=param_m,
                method=method,
                gaussian_sigma_percent=50.0,
            )

            smooth_note = f"filter={method}, radius={param_m}m"
            if method.lower() == "gaussian":
                smooth_note += ", sigma=50% radius, kernel=circle"

            self._ok(
                f"Smooth completed ({smooth_note}).\n"
                f"Outputs:\n"
                f" - {out['before_tif']}\n"
                f" - {out['after_tif']}\n"
                f" - {out['after_dem_tif']}"
            )
            self.viewer.show_pair(out["before_png"], out["after_png"])

        except Exception as e:
            self._err(f"Smooth error: {e}")

    def _on_calc_sad(self) -> None:
        if not self._last_run_dir:
            self._warn("Please run 'Confirm Input' first.")
            return
        try:
            ctx = self._backend.context_from_run_dir(self._last_run_dir)

            # --- disable nút khi đang chạy ---
            self.btn_calc_sad.setEnabled(False)
            self.btn_detect.setEnabled(False)
            self.btn_vectors.setEnabled(False)
            self.cmb_method.setEnabled(False)
            self._info("Calculating SAD + dZ in background...")

            # --- chạy nền bằng QThread ---
            self._sad_thread = QThread(self)
            self._sad_worker = UI1SadWorker(
                backend=self._backend,
                ctx=ctx,
                method=self._selected_sad_method_key(),
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
            self.cmb_method.setEnabled(True)
            self.btn_calc_sad.setEnabled(True)
            self.btn_detect.setEnabled(True)
            self.cmb_detect_source.setEnabled(True)
            self.btn_vectors.setEnabled(True)

            # thông báo
            self._ok(
                f"SAD + dZ completed (method: {method}).\n"
                f"- dX: {out.get('dx_tif', '')}\n"
                f"- dY: {out.get('dy_tif', '')}\n"
                f"- dZ: {out.get('dz_tif', '')}"
            )

            # hiển thị mặc định sau khi xong
            left_img, right_img = self._backend.post_sad_preview(str(self._last_run_dir or ""))
            self.viewer.show_pair(left_img, right_img)
            self._refresh_mask_source_ui(self._last_run_dir)

        except Exception as e:
            self._err(f"Post-process error: {e}")

    def _on_sad_error(self, msg: str) -> None:
        self.cmb_method.setEnabled(True)
        self.btn_calc_sad.setEnabled(True)
        self.btn_detect.setEnabled(True)
        self.btn_vectors.setEnabled(True)
        if self._last_run_dir:
            dx_ok = os.path.exists(os.path.join(self._last_run_dir, "ui1", "dx.tif"))
            dy_ok = os.path.exists(os.path.join(self._last_run_dir, "ui1", "dy.tif"))
            self.btn_detect.setEnabled(dx_ok and dy_ok)
            self.cmb_detect_source.setEnabled(dx_ok and dy_ok)
        else:
            self.btn_detect.setEnabled(False)
            self.cmb_detect_source.setEnabled(False)
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
    @staticmethod
    def _status_brief(msg: str, fallback: str) -> str:
        skip_prefixes = (
            "project:",
            "run:",
            "output:",
            "folder:",
            "outputs:",
            "- dx:",
            "- dy:",
            "- dz:",
            "- mask:",
            "- polygons:",
            "- dX:",
            "- dY:",
            "- dZ:",
        )
        for raw in str(msg or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if any(low.startswith(prefix.lower()) for prefix in skip_prefixes):
                continue
            if "\\" in line or "/" in line:
                if ":" in line:
                    line = line.split(":", 1)[0].strip()
                else:
                    continue
            return line
        return fallback

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
        return

    def _ok(self, msg: str) -> None:
        self._append_status(f"OK: {self._status_brief(msg, 'Completed.')}")

    def _warn(self, msg: str) -> None:
        self._append_status(f"ERROR: {self._status_brief(msg, 'Action required.')}")
        QMessageBox.warning(self, "Warning", msg)

    def _err(self, msg: str) -> None:
        self._append_status(f"ERROR: {self._status_brief(msg, 'Error.')}")
        QMessageBox.critical(self, "Error", msg)

    def _on_detect_requested(self) -> None:
        source = self.cmb_detect_source.currentData()
        if source == "dxf":
            self._on_import_dxf_mask()
        else:
            self._on_detect()

    def _on_detect(self) -> None:
        if not self._last_run_dir:
            self._warn("Please run 'Confirm Input' first.")
            return
        try:
            ctx = self._backend.context_from_run_dir(self._last_run_dir)

            thr_m = float(self.spin_detect_thr.value())
            out = self._backend.run_detect(ctx, method="threshold", threshold_m=thr_m)

            # Auto detect overrides manual DXF mask metadata.
            meta_json = os.path.join(ctx.out_ui1, "mask_from_dxf_meta.json")
            if os.path.exists(meta_json):
                try:
                    os.remove(meta_json)
                except Exception:
                    pass

            self._ok(f"Detected with threshold = {thr_m:.2f} m\n - {out['mask_tif']}")
            dz_png = os.path.join(ctx.out_ui1, "dz.png")
            overlay = out["mask_png"]  # bước detect đã lưu heatmap overlay
            self.viewer.show_pair(dz_png, overlay)
            self._refresh_mask_source_ui(ctx.run_dir)

        except Exception as e:
            self._err(f"Detect error: {e}")

    def _on_import_dxf_mask(self) -> None:
        if not self._last_run_dir:
            self._warn("Please run 'Confirm Input' first.")
            return
        dxf_path = (self.fp_mask_dxf.path or "").strip()
        if not dxf_path:
            self._warn("Please select a DXF boundary file first.")
            return
        try:
            ctx = self._backend.context_from_run_dir(self._last_run_dir)
            out = self._backend.run_mask_from_dxf(ctx, dxf_path)

            self._ok(
                f"DXF mask created.\n"
                f" - mask: {out.get('mask_tif', '')}\n"
                f" - polygons: {out.get('polygon_count', 0)}"
            )
            self._info(f"Mask pixels in-zone: {out.get('mask_pixels_positive', 0)}")

            dz_png = os.path.join(ctx.out_ui1, "dz.png")
            left_img = dz_png if os.path.exists(dz_png) else os.path.join(ctx.out_ui1, "after_asc_hillshade.png")
            right_img = out.get("mask_png", "")
            if left_img and right_img and os.path.exists(right_img):
                self.viewer.show_pair(left_img, right_img)
            self._refresh_mask_source_ui(ctx.run_dir)
        except Exception as e:
            self._err(f"DXF mask import error: {e}")

    def _on_render_vectors(self, quiet: bool = False, emit_signal: bool = True) -> None:
        if not self._last_run_dir:
            if not quiet:
                self._warn("Please run 'Confirm Input' first.")
            return
        try:
            step = int(self.spin_vec_step.value())
            scale = float(self.spin_vec_scale.value())
            color = str(self.combo_vec_color.currentText())
            size_mul = max(0.2, float(self.sld_vec_size.value()) / 100.0)
            width = 0.003 * size_mul
            opacity = max(0.0, min(1.0, float(self.sld_vec_opacity.value()) / 100.0))

            ctx = self._backend.context_from_run_dir(self._last_run_dir)
            out = self._backend.render_vectors(
                ctx,
                step=step,
                scale=scale,
                vector_color=color,
                vector_width=width,
                vector_opacity=opacity,
            )

            # Hiển thị: trái = overlay (nếu có) hoặc dz, phải = vectors overlay
            overlay = os.path.join(ctx.out_ui1, "landslide_overlay.png")
            left_img = overlay if os.path.exists(overlay) else os.path.join(ctx.out_ui1, "dz.png")
            self.viewer.show_pair(left_img, out["vectors_png"])

            if not quiet:
                self._ok(
                    f"Vectors rendered (step={step}, scale={scale}, "
                    f"size={self.sld_vec_size.value()}%, opacity={self.sld_vec_opacity.value()}%)."
                )

            # emit để MainWindow enable tab Section Selection
            if emit_signal:
                project = (self.edit_project.text() or "").strip()
                run_label = (self.edit_runlabel.text() or "").strip()
                self.vectors_rendered.emit(project, run_label, self._last_run_dir)

        except Exception as e:
            if quiet:
                return
            else:
                self._err(f"Render vectors error: {e}")

    def _on_vec_display_slider_changed(self) -> None:
        if not self._last_run_dir:
            return
        if not hasattr(self, "btn_vectors") or not self.btn_vectors.isEnabled():
            return
        self._vec_live_timer.start()

    def _on_vec_live_tick(self) -> None:
        self._on_render_vectors(quiet=True, emit_signal=False)

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
            if not self._backend.is_valid_run_dir(sel):
                # Nếu user chọn nhầm thư mục project
                proj_candidate = os.path.join(sel, "input")
                if os.path.isdir(proj_candidate):
                    self._err(f"Selected folder is not a run folder:\n{sel}\n"
                              f"It must be .../output/<Project>/<RunID> and contain 'input' and 'ui*'.")
                    return
                self._err("Please select a run folder like:\n.../output/<Project>/<YYYYmmdd_HHMMSS[_label]>")
                return

            # Build context
            ctx = self._backend.load_context_from_run_dir(sel)
            self._last_run_dir = ctx.run_dir

            # Fill fields
            self.edit_project.setText(ctx.project_id)
            self.edit_runlabel.setText(self._backend.parse_label_from_run_id(ctx.run_id))

            # Khóa 2 ô khi đã load run
            self._set_project_run_locked(True)
            self._ok("Fields locked (loaded run).")

            # Enable buttons based on what exists
            self.btn_smooth.setEnabled(True)
            self.cmb_method.setEnabled(True)
            self.btn_calc_sad.setEnabled(True)

            dx_ok = os.path.exists(os.path.join(ctx.out_ui1, "dx.tif"))
            dy_ok = os.path.exists(os.path.join(ctx.out_ui1, "dy.tif"))
            self.btn_detect.setEnabled(dx_ok and dy_ok)
            self.cmb_detect_source.setEnabled(dx_ok and dy_ok)
            self.btn_vectors.setEnabled(dx_ok and dy_ok)

            # Bật Open run
            self.btn_open_run.setEnabled(True)

            # Show best available preview
            left_img, right_img = self._backend.existing_run_preview(ctx)
            if left_img and right_img:
                self.viewer.show_pair(left_img, right_img)
            self._refresh_mask_source_ui(ctx.run_dir)
            self._restore_sad_method_from_run(ctx.run_dir)

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

        # 3) Clear input file pickers
        for fp in (self.fp_adem, self.fp_basc, self.fp_aasc, self.fp_bpz, self.fp_apz):
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
        try:
            self.fp_mask_dxf.clear()
        except Exception:
            pass

        # 4) Đưa các nút về trạng thái ban đầu
        self.btn_open_run.setEnabled(False)
        self.btn_smooth.setEnabled(False)
        self.cmb_method.setEnabled(False)
        self.btn_calc_sad.setEnabled(False)
        self.btn_detect.setEnabled(False)
        self.cmb_detect_source.setEnabled(False)
        self.btn_vectors.setEnabled(False)
        # Nút Confirm Input luôn bật
        self.btn_confirm.setEnabled(True)

        # 5) Reset các thông số xử lý
        try:
            self.spin_smooth_param.setValue(2.0)
        except Exception:
            pass
        try:
            self._set_sad_method_combo("traditional")
        except Exception:
            pass
        try:
            self.spin_detect_thr.setValue(0.8)
        except Exception:
            pass
        try:
            self.cmb_detect_source.setCurrentIndex(0)
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
        try:
            self.combo_vec_color.setCurrentText("Blue")
        except Exception:
            pass
        try:
            self.sld_vec_size.setValue(100)
        except Exception:
            pass
        try:
            self.sld_vec_opacity.setValue(100)
        except Exception:
            pass

        # 6) Clear viewer (ảnh bên phải)
        try:
            if hasattr(self.viewer, "scene"):
                self.viewer.scene.clear()
            if hasattr(self.viewer, "caption"):
                self.viewer.caption.setText("")
            if hasattr(self.viewer, "view") and hasattr(self.viewer.view, "reset_view_transform"):
                self.viewer.view.reset_view_transform()
        except Exception:
            pass

        # 7) Clear status log
        try:
            self.status_text.clear()
        except Exception:
            pass
        try:
            self.lbl_mask_source.setText("Mask source: not set")
        except Exception:
            pass

        # 8) Thông báo nhẹ để debug
        try:
            self._info("Session reset.")
        except Exception:
            pass

    def _selected_sad_method_key(self) -> str:
        data = self.cmb_method.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip()
        txt = (self.cmb_method.currentText() or "").strip().lower()
        return "ssd_opencv" if "opencv" in txt or "ssd" in txt else "traditional"

    def _set_sad_method_combo(self, method_key: str) -> None:
        idx = self.cmb_method.findData(method_key)
        if idx < 0:
            idx = self.cmb_method.findText("SAD")
        if idx >= 0:
            self.cmb_method.setCurrentIndex(idx)

    def _restore_sad_method_from_run(self, run_dir: str) -> None:
        self._set_sad_method_combo(self._backend.read_sad_method(run_dir, default="traditional"))
