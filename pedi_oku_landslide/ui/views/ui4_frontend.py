import json
import os
from typing import Dict, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QGroupBox,
    QSplitter,
    QComboBox,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QSizePolicy,
)

from pedi_oku_landslide.pipeline.runners.ui4_backend import (
    collect_ui4_run_inputs,
    render_ui4_contours_for_run,
    run_ui4_kriging_for_run,
)


class ZoomableImageView(QGraphicsView):
    def __init__(self, parent=None):
        self._scene = QGraphicsScene()
        super().__init__(self._scene, parent)
        self._pix_item = None
        self._zoom_steps = 0
        self._loaded_path = ""
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(Qt.white)
        self.setStyleSheet("QGraphicsView { border: 1px solid #dddddd; border-radius: 8px; }")
        self.setMinimumHeight(420)

    def clear_image(self) -> None:
        self._scene.clear()
        self._pix_item = None
        self._zoom_steps = 0
        self._loaded_path = ""

    def load_image(self, path: str) -> bool:
        pm = QPixmap(path)
        if pm.isNull():
            self.clear_image()
            return False
        self._scene.clear()
        self._pix_item = QGraphicsPixmapItem(pm)
        self._scene.addItem(self._pix_item)
        self._scene.setSceneRect(self._pix_item.boundingRect())
        self._loaded_path = path
        self.fit_to_image()
        return True

    def fit_to_image(self) -> None:
        if self._pix_item is None:
            return
        self.resetTransform()
        rect = self._pix_item.boundingRect()
        if not rect.isNull():
            self.fitInView(rect, Qt.KeepAspectRatio)
        self._zoom_steps = 0

    def zoom_in(self) -> None:
        if self._pix_item is None:
            return
        self.scale(1.2, 1.2)
        self._zoom_steps += 1

    def zoom_out(self) -> None:
        if self._pix_item is None:
            return
        self.scale(1 / 1.2, 1 / 1.2)
        self._zoom_steps -= 1

    def wheelEvent(self, event):
        if self._pix_item is None:
            super().wheelEvent(event)
            return
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()


class UI4FrontendTab(QWidget):
    """
    Minimal UI4 connector tab.
    Purpose for now: receive run context from UI1/UI2/UI3 and show whether
    upstream files for UI4 kriging are ready.
    """

    def __init__(self, base_dir: str):
        super().__init__()
        self.base_dir = base_dir
        self._ctx: Dict[str, str] = {"project": "", "run_label": "", "run_dir": ""}
        self._last_info: Dict = {}
        self._preview_png_paths = []
        self._last_ui4_summary: Dict = {}
        self._splitter: Optional[QSplitter] = None
        self._left_min_w = 380
        self._left_default_w = 490
        self._pending_init_splitter = True
        self._ui_shown_once = False

        self.lbl_project_value = QLabel("-")
        self.lbl_run_label_value = QLabel("-")
        self.lbl_input_status_value = QLabel("Not Ready")
        self.lbl_preview_status = QLabel("Preview: -")
        self.lbl_preview_status.hide()
        self.preview_file_combo = QComboBox()
        self.preview_file_combo.currentIndexChanged.connect(self._on_preview_file_changed)
        self.preview_view = ZoomableImageView()
        self.status_box = QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMaximumHeight(140)
        self.btn_refresh = QPushButton("Reload Inputs")
        self.btn_refresh.clicked.connect(self.refresh_from_context)
        self.btn_run_ui4 = QPushButton("Calculate Kriging")
        self.btn_run_ui4.clicked.connect(self._on_run_ui4)
        self.btn_make_contours = QPushButton("Preview")
        self.btn_make_contours.clicked.connect(self._on_generate_contours)
        self.btn_preview_fit = QPushButton("Fit")
        self.btn_preview_fit.clicked.connect(self.preview_view.fit_to_image)
        self.btn_preview_zoom_in = QPushButton("Zoom +")
        self.btn_preview_zoom_in.clicked.connect(self.preview_view.zoom_in)
        self.btn_preview_zoom_out = QPushButton("Zoom -")
        self.btn_preview_zoom_out.clicked.connect(self.preview_view.zoom_out)
        self.surface_auto_range = QCheckBox("Auto")
        self.surface_auto_range.setChecked(True)
        self.surface_auto_range.toggled.connect(self._on_surface_auto_range_toggled)
        self.surface_zmin = self._make_z_spin()
        self.surface_zmax = self._make_z_spin()
        self.surface_step = self._make_step_spin(5.0)
        self.depth_auto_range = QCheckBox("Auto")
        self.depth_auto_range.setChecked(True)
        self.depth_auto_range.toggled.connect(self._on_depth_auto_range_toggled)
        self.depth_zmin = self._make_z_spin()
        self.depth_zmax = self._make_z_spin()
        self.depth_step = self._make_step_spin(2.0)

        self._build_ui()
        self._update_contour_range_controls()
        self._apply_ui4_panel_spacing()

    def _apply_ui4_panel_spacing(self) -> None:
        # Make panel rows more readable: taller controls and more spacing.
        ctrl_h = 32
        for w in (
            self.preview_file_combo,
            self.surface_zmin,
            self.surface_zmax,
            self.surface_step,
            self.depth_zmin,
            self.depth_zmax,
            self.depth_step,
            self.btn_refresh,
            self.btn_run_ui4,
            self.btn_make_contours,
            self.btn_preview_zoom_out,
            self.btn_preview_zoom_in,
            self.btn_preview_fit,
        ):
            w.setMinimumHeight(ctrl_h)
        self.surface_auto_range.setMinimumHeight(ctrl_h)
        self.depth_auto_range.setMinimumHeight(ctrl_h)
        for lbl in (
            self.lbl_project_value,
            self.lbl_run_label_value,
            self.lbl_input_status_value,
        ):
            lbl.setMinimumHeight(28)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self._splitter = splitter
        splitter.splitterMoved.connect(lambda *_: self._enforce_left_pane_bounds())

        # Left pane (controls / status / info) ~30%
        left_host = QWidget()
        left_lay = QVBoxLayout(left_host)
        left_lay.setContentsMargins(6, 6, 6, 6)
        left_lay.setSpacing(8)

        box_info = QGroupBox("Project")
        info_lay = QGridLayout(box_info)
        info_lay.setContentsMargins(12, 14, 12, 12)
        info_lay.setHorizontalSpacing(12)
        info_lay.setVerticalSpacing(10)
        info_lay.addWidget(QLabel("Project"), 0, 0)
        info_lay.addWidget(self.lbl_project_value, 0, 1)
        info_lay.addWidget(QLabel("Run label"), 1, 0)
        info_lay.addWidget(self.lbl_run_label_value, 1, 1)
        info_lay.addWidget(QLabel("Input Status"), 2, 0)
        info_lay.addWidget(self.lbl_input_status_value, 2, 1)
        info_lay.setColumnStretch(1, 1)
        for r in range(3):
            info_lay.setRowMinimumHeight(r, 30)
        left_lay.addWidget(box_info)

        box_display = QGroupBox("Contour Lines Display")
        display_lay = QVBoxLayout(box_display)
        display_lay.setContentsMargins(12, 14, 12, 12)
        display_lay.setSpacing(10)

        cfg_lay = QGridLayout()
        cfg_lay.setHorizontalSpacing(10)
        cfg_lay.setVerticalSpacing(10)
        cfg_lay.addWidget(QLabel("Type"), 0, 0)
        cfg_lay.addWidget(QLabel("Auto Range"), 0, 1)
        cfg_lay.addWidget(QLabel("Z min (m)"), 0, 2)
        cfg_lay.addWidget(QLabel("Z max (m)"), 0, 3)
        cfg_lay.addWidget(QLabel("Step (m)"), 0, 4)
        cfg_lay.addWidget(QLabel("Surface"), 1, 0)
        cfg_lay.addWidget(self.surface_auto_range, 1, 1)
        cfg_lay.addWidget(self.surface_zmin, 1, 2)
        cfg_lay.addWidget(self.surface_zmax, 1, 3)
        cfg_lay.addWidget(self.surface_step, 1, 4)
        cfg_lay.addWidget(QLabel("Depth"), 2, 0)
        cfg_lay.addWidget(self.depth_auto_range, 2, 1)
        cfg_lay.addWidget(self.depth_zmin, 2, 2)
        cfg_lay.addWidget(self.depth_zmax, 2, 3)
        cfg_lay.addWidget(self.depth_step, 2, 4)
        for r in range(3):
            cfg_lay.setRowMinimumHeight(r, 32)
        display_lay.addLayout(cfg_lay)
        preview_ctl = QHBoxLayout()
        preview_ctl.setSpacing(10)
        preview_ctl.addWidget(QLabel("File"))
        preview_ctl.addWidget(self.preview_file_combo, 1)
        display_lay.addLayout(preview_ctl)
        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        for b in (self.btn_refresh, self.btn_run_ui4, self.btn_make_contours):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            action_row.addWidget(b, 1)
        display_lay.addLayout(action_row)

        left_lay.addWidget(box_display)

        box_status = QGroupBox("Status")
        box_status_lay = QVBoxLayout(box_status)
        box_status_lay.setContentsMargins(12, 14, 12, 12)
        box_status_lay.addWidget(self.status_box)
        left_lay.addWidget(box_status)
        left_lay.addStretch(1)

        # Right pane (image only) ~70%
        right_host = QWidget()
        right_lay = QVBoxLayout(right_host)
        right_lay.setContentsMargins(6, 6, 6, 6)
        right_lay.setSpacing(8)
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(10)
        zoom_row.addWidget(self.btn_preview_zoom_out)
        zoom_row.addWidget(self.btn_preview_zoom_in)
        zoom_row.addWidget(self.btn_preview_fit)
        zoom_row.addStretch(1)
        right_lay.addLayout(zoom_row)
        box_image = QGroupBox("Contour Lines Preview Image")
        box_image_lay = QVBoxLayout(box_image)
        box_image_lay.addWidget(self.preview_view, 1)
        right_lay.addWidget(box_image, 1)

        splitter.addWidget(left_host)
        splitter.addWidget(right_host)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        # Match UI3 left pane default width (_left_default_w ~= 490 px).
        splitter.setSizes([self._left_default_w, 900])
        left_host.setMinimumWidth(self._left_min_w)
        right_host.setMinimumWidth(500)

        root.addWidget(splitter, 1)

    def _append(self, msg: str) -> None:
        self.status_box.append(msg)

    def _left_max_w(self) -> int:
        # Left pane can occupy at most 50% of current splitter/window width.
        base_w = self.width()
        if self._splitter is not None and self._splitter.width() > 0:
            base_w = self._splitter.width()
        if base_w < (self._left_min_w * 2):
            return -1
        return max(self._left_min_w, int(base_w * 0.5))

    def _try_apply_initial_splitter_width(self) -> None:
        if not self._pending_init_splitter or self._splitter is None:
            return
        # Defer until layout settles at a width that can actually host the
        # requested default (490 px) while respecting max = 1/2 window.
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

    def _make_z_spin(self) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(-1_000_000.0, 1_000_000.0)
        s.setDecimals(3)
        s.setSingleStep(1.0)
        s.setKeyboardTracking(False)
        s.setValue(0.0)
        return s

    def _make_step_spin(self, default: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(0.01, 1_000_000.0)
        s.setDecimals(3)
        s.setSingleStep(0.5)
        s.setKeyboardTracking(False)
        s.setValue(float(default))
        return s

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._pending_init_splitter:
            return
        self._enforce_left_pane_bounds()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._ui_shown_once = True
        if self._pending_init_splitter:
            # Apply after Qt finishes initial layout passes; early setSizes()
            # calls can be overwritten and appear as a 50/50 split.
            QTimer.singleShot(0, self._enforce_left_pane_bounds)
            QTimer.singleShot(80, self._enforce_left_pane_bounds)
            QTimer.singleShot(180, self._enforce_left_pane_bounds)
            return
        self._enforce_left_pane_bounds()

    def _update_contour_range_controls(self, *_args) -> None:
        surf_manual = not self.surface_auto_range.isChecked()
        self.surface_zmin.setEnabled(surf_manual)
        self.surface_zmax.setEnabled(surf_manual)
        depth_manual = not self.depth_auto_range.isChecked()
        self.depth_zmin.setEnabled(depth_manual)
        self.depth_zmax.setEnabled(depth_manual)

    def _on_surface_auto_range_toggled(self, checked: bool) -> None:
        if not checked:
            self._populate_manual_range_from_summary("surface")
        self._update_contour_range_controls()

    def _on_depth_auto_range_toggled(self, checked: bool) -> None:
        if not checked:
            self._populate_manual_range_from_summary("depth")
        self._update_contour_range_controls()

    def _read_json_file(self, path: str) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _load_ui4_summary_for_current_run(self) -> Dict:
        run_dir = (self._ctx.get("run_dir") or "").strip()
        if not run_dir:
            self._last_ui4_summary = {}
            return {}
        ui4_dir = os.path.join(run_dir, "ui4")
        shared_path = os.path.join(ui4_dir, "ui_shared_data.json")
        shared = self._read_json_file(shared_path) if os.path.exists(shared_path) else {}
        summary_path = str(shared.get("ui4_summary_json") or os.path.join(ui4_dir, "ui4_kriging_summary.json"))
        if not os.path.exists(summary_path):
            self._last_ui4_summary = {}
            return {}
        self._last_ui4_summary = self._read_json_file(summary_path)
        return self._last_ui4_summary

    def _summary_range_for_kind(self, kind: str) -> Optional[tuple]:
        summary = self._last_ui4_summary or self._load_ui4_summary_for_current_run()
        if not isinstance(summary, dict):
            return None
        raster_stats = summary.get("raster_stats", {}) if isinstance(summary.get("raster_stats"), dict) else {}
        for key in (f"{kind}_masked", kind):
            rs = raster_stats.get(key, {})
            if not isinstance(rs, dict):
                continue
            zmin = rs.get("min")
            zmax = rs.get("max")
            try:
                zmin_f = float(zmin)
                zmax_f = float(zmax)
            except Exception:
                continue
            if zmax_f > zmin_f:
                return (zmin_f, zmax_f)

        stats = summary.get("stats", {}) if isinstance(summary.get("stats"), dict) else {}
        legacy_min = stats.get(f"{kind}_min_m")
        legacy_max = stats.get(f"{kind}_max_m")
        try:
            legacy_min_f = float(legacy_min)
            legacy_max_f = float(legacy_max)
        except Exception:
            return None
        if legacy_max_f > legacy_min_f:
            return (legacy_min_f, legacy_max_f)
        return None

    def _populate_manual_range_from_summary(self, kind: str) -> None:
        rng = self._summary_range_for_kind(kind)
        if rng is None:
            self._append(f"[UI4] Cannot auto-fill {kind} manual range: summary min/max not available.")
            return
        zmin, zmax = rng
        if kind == "surface":
            self.surface_zmin.setValue(zmin)
            self.surface_zmax.setValue(zmax)
        else:
            self.depth_zmin.setValue(zmin)
            self.depth_zmax.setValue(zmax)
        self._append(f"[UI4] {kind} manual range auto-filled from raster stats: {zmin:g} .. {zmax:g}")

    def _validate_contour_params(self, params: Dict[str, float | None]) -> Optional[str]:
        if params.get("surface_z_min") is not None and params.get("surface_z_max") is not None:
            if float(params["surface_z_max"]) <= float(params["surface_z_min"]):
                return "Invalid surface manual range: zmax must be greater than zmin."
        if params.get("depth_z_min") is not None and params.get("depth_z_max") is not None:
            if float(params["depth_z_max"]) <= float(params["depth_z_min"]):
                return "Invalid depth manual range: zmax must be greater than zmin."
        return None

    def _contour_param_values(self) -> Dict[str, float | None]:
        surf_zmin = None if self.surface_auto_range.isChecked() else float(self.surface_zmin.value())
        surf_zmax = None if self.surface_auto_range.isChecked() else float(self.surface_zmax.value())
        depth_zmin = None if self.depth_auto_range.isChecked() else float(self.depth_zmin.value())
        depth_zmax = None if self.depth_auto_range.isChecked() else float(self.depth_zmax.value())
        return {
            "surface_interval_m": float(self.surface_step.value()),
            "depth_interval_m": float(self.depth_step.value()),
            "surface_z_min": surf_zmin,
            "surface_z_max": surf_zmax,
            "depth_z_min": depth_zmin,
            "depth_z_max": depth_zmax,
        }

    def _on_preview_file_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._preview_png_paths):
            self.preview_view.clear_image()
            return
        path = self._preview_png_paths[idx]
        ok = self.preview_view.load_image(path)
        if ok:
            self.lbl_preview_status.setText(
                f"Preview: {len(self._preview_png_paths)} PNG file(s) | Showing: {os.path.basename(path)} "
                f"(wheel/Zoom +/-/Fit)"
            )
        else:
            self.lbl_preview_status.setText(f"Preview: cannot load image ({path})")

    def _refresh_preview_pngs(self) -> None:
        run_dir = (self._ctx.get("run_dir") or "").strip()
        prev_selected = self.preview_file_combo.currentData()
        self.preview_file_combo.blockSignals(True)
        self.preview_file_combo.clear()
        self.preview_file_combo.blockSignals(False)
        self._preview_png_paths = []
        self.preview_view.clear_image()
        if not run_dir:
            self.lbl_preview_status.setText("Preview: missing run context")
            return

        preview_dir = os.path.join(run_dir, "ui4", "preview")
        if not os.path.isdir(preview_dir):
            self.lbl_preview_status.setText(f"Preview: folder not found ({preview_dir})")
            return

        pngs = sorted(
            os.path.join(preview_dir, n)
            for n in os.listdir(preview_dir)
            if n.lower().endswith(".png")
        )
        self._preview_png_paths = pngs
        self.lbl_preview_status.setText(f"Preview: {len(pngs)} PNG file(s) in {preview_dir}")
        if not pngs:
            return

        self.preview_file_combo.blockSignals(True)
        for p in pngs:
            self.preview_file_combo.addItem(os.path.basename(p), p)
        self.preview_file_combo.blockSignals(False)

        idx = 0
        if prev_selected:
            try:
                idx = pngs.index(prev_selected)
            except ValueError:
                idx = 0
        self.preview_file_combo.setCurrentIndex(idx)
        self._on_preview_file_changed(idx)

    def set_context(self, project: str, run_label: str, run_dir: str) -> None:
        self._ctx = {
            "project": str(project or ""),
            "run_label": str(run_label or ""),
            "run_dir": str(run_dir or ""),
        }
        self.lbl_project_value.setText(self._ctx["project"] or "-")
        self.lbl_run_label_value.setText(self._ctx["run_label"] or "-")
        self.refresh_from_context()

    def on_upstream_curve_saved(self, curve_json_path: str = "") -> None:
        if curve_json_path:
            self._append(f"[UI3] curve_saved -> {curve_json_path}")
        self.refresh_from_context()

    def refresh_from_context(self) -> None:
        run_dir = (self._ctx.get("run_dir") or "").strip()
        if not run_dir:
            self.lbl_input_status_value.setText("Not Ready")
            self._append("[UI4] Waiting for run context from previous tabs.")
            self._refresh_preview_pngs()
            return

        info = collect_ui4_run_inputs(run_dir)
        self._last_info = info
        self._load_ui4_summary_for_current_run()
        if not info.get("ok", False):
            self.lbl_input_status_value.setText("Not Ready")
            self._append(f"[UI4] Error: {info.get('error', 'unknown error')}")
            self._refresh_preview_pngs()
            return

        ready = bool(info.get("ready_for_ui4"))
        self.lbl_input_status_value.setText("Ready" if ready else "Not Ready")

        paths = info.get("paths", {})
        counts = info.get("counts", {})
        missing = info.get("missing_required", [])

        lines = [
            f"[UI4] Refresh run: {os.path.basename(run_dir)}",
            f"  Input dir: {paths.get('input_dir') or '-'}",
            f"  DEM: {paths.get('dem') or '-'}",
            f"  Mask (optional): {paths.get('mask_tif') or '-'}",
            f"  UI3 curve dir: {paths.get('ui3_curve_dir') or '-'}",
            f"  NURBS curves (CL/ML pattern): {counts.get('nurbs_curves', 0)}",
            f"  Groups: {counts.get('groups', 0)}",
            f"  NURBS info: {counts.get('nurbs_info', 0)}",
        ]
        if missing:
            lines.append("  Missing required: " + ", ".join(missing))
        else:
            lines.append("  Required inputs look ready for UI4 (DEM .tif + NURBS CL/ML curves).")

        self.status_box.clear()
        for ln in lines:
            self._append(ln)
        self._refresh_preview_pngs()

    def _on_generate_contours(self) -> None:
        run_dir = (self._ctx.get("run_dir") or "").strip()
        if not run_dir:
            self._append("[UI4] Cannot generate contours: missing run context.")
            return
        try:
            self._append("[UI4] Generating contour previews...")
            contour_kwargs = self._contour_param_values()
            err = self._validate_contour_params(contour_kwargs)
            if err:
                self._append(f"[UI4] {err}")
                return
            self._append(
                "[UI4] Contour settings: "
                f"surface(step={contour_kwargs['surface_interval_m']}, "
                f"zmin={contour_kwargs['surface_z_min']}, zmax={contour_kwargs['surface_z_max']}), "
                f"depth(step={contour_kwargs['depth_interval_m']}, "
                f"zmin={contour_kwargs['depth_z_min']}, zmax={contour_kwargs['depth_z_max']})"
            )
            res = render_ui4_contours_for_run(run_dir, log_fn=self._append, **contour_kwargs)
            if not res.get("ok", False):
                self._append(f"[UI4] Contours failed: {res.get('error', 'unknown error')}")
                if "No UI4 kriging rasters found" in str(res.get("error", "")):
                    self._append("[UI4] Hint: click 'Calculate Kriging' first, then 'Preview'.")
                return
            items = res.get("items", {})
            for key in ("surface", "depth"):
                it = items.get(key, {})
                if it.get("ok"):
                    self._append(f"[UI4] {key} contours PNG: {it.get('png_path')}")
                elif it:
                    self._append(f"[UI4] {key} contours error: {it.get('error')}")
            if res.get("summary_json"):
                self._append(f"[UI4] Contour summary: {res.get('summary_json')}")
            self._refresh_preview_pngs()
        except Exception as e:
            self._append(f"[UI4] Contours exception: {e}")

    def _on_run_ui4(self) -> None:
        run_dir = (self._ctx.get("run_dir") or "").strip()
        if not run_dir:
            self._append("[UI4] Cannot run UI4: missing run context.")
            return
        try:
            self._append("[UI4] Running kriging backend...")
            res = run_ui4_kriging_for_run(run_dir, log_fn=self._append)
            if not res.get("ok", False):
                self._append(f"[UI4] Kriging failed: {res.get('error', 'unknown error')}")
                missing = res.get("missing_required", [])
                if missing:
                    self._append("[UI4] Missing required inputs: " + ", ".join(map(str, missing)))
                return
            outputs = res.get("outputs", {})
            self._append(f"[UI4] Surface raster: {outputs.get('slip_surface_tif')}")
            self._append(f"[UI4] Depth raster: {outputs.get('slip_depth_tif')}")
            self._append(f"[UI4] Variance raster: {outputs.get('slip_depth_variance_tif')}")
            if outputs.get("slip_surface_masked_tif"):
                self._append(f"[UI4] Surface raster (masked): {outputs.get('slip_surface_masked_tif')}")
            if outputs.get("slip_depth_masked_tif"):
                self._append(f"[UI4] Depth raster (masked): {outputs.get('slip_depth_masked_tif')}")
            if outputs.get("summary_json"):
                self._append(f"[UI4] Summary: {outputs.get('summary_json')}")
            self._load_ui4_summary_for_current_run()
            self.refresh_from_context()
            self._refresh_preview_pngs()
        except Exception as e:
            self._append(f"[UI4] Kriging exception: {e}")

    def reset_session(self) -> None:
        self._ctx = {"project": "", "run_label": "", "run_dir": ""}
        self._last_info = {}
        self._last_ui4_summary = {}
        self.lbl_project_value.setText("-")
        self.lbl_run_label_value.setText("-")
        self.lbl_input_status_value.setText("Not Ready")
        self.status_box.clear()
        self.lbl_preview_status.setText("Preview: -")
        self.preview_file_combo.blockSignals(True)
        self.preview_file_combo.clear()
        self.preview_file_combo.blockSignals(False)
        self._preview_png_paths = []
        self.preview_view.clear_image()
        self.surface_auto_range.setChecked(True)
        self.depth_auto_range.setChecked(True)
        self.surface_step.setValue(5.0)
        self.depth_step.setValue(2.0)
        self.surface_zmin.setValue(0.0)
        self.surface_zmax.setValue(0.0)
        self.depth_zmin.setValue(0.0)
        self.depth_zmax.setValue(0.0)
