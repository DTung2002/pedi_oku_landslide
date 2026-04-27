from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QGroupBox, QTextEdit, QSplitter, QSizePolicy,
    QDoubleSpinBox, QComboBox, QSpinBox, QScrollArea, QGridLayout, QSlider, QFrame
)

from ..widgets.file_picker import FilePicker
from .ui1_viewer import UI1Viewer
from pedi_oku_landslide.ui.layout_constants import (
    LEFT_MARGINS,
    PANEL_SPACING,
    RIGHT_MARGINS,
    RIGHT_MIN_W,
    CONTROL_HEIGHT,
    PROJECT_H_SPACING,
    PROJECT_LABEL_W,
    PROJECT_MARGINS,
    PROJECT_V_SPACING,
    ROOT_MARGINS,
    ROOT_SPACING,
)


class UI1BuildMixin:
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

