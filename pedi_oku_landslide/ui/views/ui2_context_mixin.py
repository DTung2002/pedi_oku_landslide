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


class UI2ContextMixin:
    def set_context(
            self,
            project: str,
            run_label: str,
            run_dir: str,
            vec_step: Optional[int] = None,
            vec_scale: Optional[float] = None,
    ) -> None:
        # lưu vào “ctx_*” cho nội bộ
        self._ctx_project = project or ""
        self._ctx_runlabel = run_label or ""
        self._ctx_run_dir = run_dir or ""
        self._ui1_dir = os.path.join(run_dir, "ui1")
        self._ui2_dir = os.path.join(run_dir, "ui2")

        # UI2 dùng bộ thông số Vector Display riêng; không sync từ UI1.
        _ = vec_step
        _ = vec_scale

        # ✨ lưu vào các thuộc tính public để các hàm khác đọc
        self.project = self._ctx_project
        self.run_label = self._ctx_runlabel
        self.run_dir = self._ctx_run_dir

        backend_ctx = self._backend.set_context(project, run_label, run_dir, base_dir=self.base_dir)
        self._ui1_dir = str(backend_ctx.get("ui1_dir", "") or self._ui1_dir)
        self._ui2_dir = str(backend_ctx.get("ui2_dir", "") or self._ui2_dir)
        self._ctx_ready = bool(backend_ctx.get("ready", False))
        missing = list(backend_ctx.get("missing", []) or [])

        # cập nhật text vào ô hiển thị (read-only)
        if hasattr(self, "edit_project"):
            self.edit_project.setText(self.project)
            self.edit_project.setReadOnly(True)
        if hasattr(self, "edit_runlabel"):
            self.edit_runlabel.setText(self.run_label)
            self.edit_runlabel.setReadOnly(True)

        if not self._ctx_ready:
            self._clear_sections_state()
            missing_txt = ", ".join(str(p) for p in missing if p) or "required UI2 inputs"
            self._err(f"[UI2] Context incomplete. Missing files: {missing_txt}")
            return

        # nạp layer
        try:
            self._load_layers_and_show()
        except Exception as e:
            self._ctx_ready = False
            self._clear_sections_state()
            self._err(f"[UI2] Cannot load run context: {e}")
            return
        self._ok("[UI2] Context set OK.")

    def _on_render_vectors(self) -> None:
        self._vec_step = int(self.spin_vec_step.value())
        self._vec_scale = float(self.spin_vec_scale.value())
        self._vec_size_pct = int(self.sld_vec_size.value())
        self._vec_opacity_pct = int(self.sld_vec_opacity.value())
        self._vec_color = str(self.combo_vec_color.currentText() or "Blue").strip().lower()

        if not self._ctx_ready:
            self._info("[UI2] Vector display updated (no active context yet).")
            return

        self._load_dx_dy_and_draw(self._ui1_dir, step=self._vec_step, scale=self._vec_scale)
        self._ok(
            f"[UI2] Vectors rendered (step={self._vec_step}, scale={self._vec_scale:.2f}, "
            f"size={self._vec_size_pct}%, opacity={self._vec_opacity_pct}%, color={self._vec_color})."
        )

    def _on_vec_size_changed(self, v: int) -> None:
        self._vec_size_pct = int(v)
        # đồng bộ các input hiện tại trước khi redraw
        self._vec_step = int(self.spin_vec_step.value())
        self._vec_scale = float(self.spin_vec_scale.value())
        self._vec_color = str(self.combo_vec_color.currentText() or "Blue").strip().lower()
        if self._ctx_ready:
            self._load_dx_dy_and_draw(self._ui1_dir, step=self._vec_step, scale=self._vec_scale)

    def _on_vec_opacity_changed(self, v: int) -> None:
        self._vec_opacity_pct = int(v)
        self.viewer.set_vector_opacity(float(self._vec_opacity_pct) / 100.0)

    # ---- core loading/drawing ----
    def _load_layers_and_show(self) -> None:
        # Lấy run_dir từ context đã set
        rd = getattr(self, "run_dir", "") or self._ctx_run_dir
        if not rd or not os.path.isdir(rd):
            return

        ui1 = os.path.join(rd, "ui1")
        payload = self._backend.load_layers(
            rd,
            {"heat_alpha": float(self.sld_heat.value()) / 100.0},
        )
        self._dz = payload["dz"]
        self._dx = payload["dx"]
        self._dy = payload["dy"]
        self._mask = payload["mask"]
        self._tr = payload["transform"]
        self._inv_tr = payload["inv_transform"]
        self._dem_path = payload["dem_path"]
        W = int(payload["width"])
        H = int(payload["height"])
        hs8 = payload["hillshade"]
        self.viewer.set_transform(self._tr)
        self.viewer.set_hillshade(hs8)  # chốt scene rect & fit
        self.viewer.set_hillshade_opacity(self.sld_hill.value() / 100.0)
        self.viewer.set_grid(W, H, step_m=20.0)
        self.viewer.set_grid_font_size(self.sld_grid_font.value())
        alpha = self.sld_heat.value() / 100.0
        self.viewer.set_heatmap_rgba(payload["heat_rgba"], alpha)

        # --- 7) Vẽ vector (dùng dx/dy vừa đọc) theo tham số Vector Display của UI2 ---
        self._load_dx_dy_and_draw(
            ui1,
            step=self._vec_step,
            scale=self._vec_scale,
        )

        # default native transform (khong fit-to-view)
        self.viewer.view.resetTransform()
        self.viewer.view.centerOn(self.viewer.scene.itemsBoundingRect().center())
        self._ok("[UI2] Layers loaded & aligned.")
        self._load_saved_sections()

    def _load_dx_dy_and_draw(self, ui1_dir: str, step: int = 25, scale: float = 1.0) -> None:
        self.viewer.clear_vectors()
        # read & align
        def _read_align(name: str) -> Optional[np.ndarray]:
            p = os.path.join(ui1_dir, name)
            if not _file_exists(p):
                return None
            with rasterio.open(p) as ds:
                arr = ds.read(1).astype("float32")
                if ds.transform != self._tr or (ds.width, ds.height) != self._dz.shape[::-1]:
                    arr = ds.read(1, out_shape=self._dz.shape, resampling=Resampling.bilinear).astype("float32")
            return arr

        self._dx = _read_align("dx.tif")
        self._dy = _read_align("dy.tif")
        if self._dx is None or self._dy is None:
            self._info("[UI2] dx/dy not found → skip vectors.")
            return

        H, W = self._dz.shape
        ys = np.arange(step // 2, H, step)
        xs = np.arange(step // 2, W, step)
        xs, ys = np.meshgrid(xs, ys)
        xs = xs.ravel(); ys = ys.ravel()

        dx = self._dx[ys, xs]
        dy = self._dy[ys, xs]
        msk = self._mask[ys, xs] > 0
        ok = np.isfinite(dx) & np.isfinite(dy) & msk
        xs, ys, dx, dy = xs[ok], ys[ok], dx[ok], dy[ok]

        pix_w = float(abs(self._tr.a)) if self._tr is not None else 1.0
        pix_h = float(abs(self._tr.e)) if self._tr is not None else 1.0

        # scale càng lớn → vector càng dài, giống UI1
        # size_pct cũng scale theo cả chiều dài + chiều ngang (to lên đồng đều)
        size_mul = max(0.2, float(self._vec_size_pct) / 100.0)
        k = 0.4  # hệ số hiệu chỉnh, có thể chỉnh 0.3–0.6 tuỳ mắt nhìn
        vx_pix = (dx / (pix_w + 1e-9)) * scale * k * size_mul
        vy_pix = (-dy / (pix_h + 1e-9)) * scale * k * size_mul  # y pixel hướng xuống

        pts_pix = np.stack([xs, ys], axis=1).astype("float32")
        vec_pix = np.stack([vx_pix, vy_pix], axis=1).astype("float32")
        # vẽ vector liền khối (thân + đầu mũi tên là một polygon)
        color_name = str(getattr(self, "_vec_color", "blue") or "blue").strip().lower()
        color = QColor(color_name)
        if not color.isValid():
            color = QColor("blue")
        shaft_half_w = max(1.2, float(self._vec_pen_base) * 0.5 * size_mul)
        base_head_len = max(3.0, shaft_half_w * 3.2)
        base_head_half_w = max(2.0, shaft_half_w * 2.75)

        for (x, y), (vx, vy) in zip(pts_pix, vec_pix):
            # đảo hướng y vì raster y+ xuống
            end_x = float(x + vx)
            end_y = float(y - vy)
            dxv = end_x - float(x)
            dyv = end_y - float(y)
            length = float(np.hypot(dxv, dyv))
            if not np.isfinite(length) or length <= 1e-6:
                continue

            ux = dxv / length
            uy = dyv / length
            nx = -uy
            ny = ux

            head_len = min(base_head_len, length * 0.45); head_len = max(head_len, shaft_half_w * 1.8)
            head_half_w = max(base_head_half_w, shaft_half_w * 1.05)
            neck_x = end_x - ux * head_len
            neck_y = end_y - uy * head_len

            p0 = QPointF(float(x + nx * shaft_half_w), float(y + ny * shaft_half_w))
            p1 = QPointF(neck_x + nx * shaft_half_w, neck_y + ny * shaft_half_w)
            p2 = QPointF(neck_x + nx * head_half_w, neck_y + ny * head_half_w)
            p3 = QPointF(end_x, end_y)
            p4 = QPointF(neck_x - nx * head_half_w, neck_y - ny * head_half_w)
            p5 = QPointF(neck_x - nx * shaft_half_w, neck_y - ny * shaft_half_w)
            p6 = QPointF(float(x - nx * shaft_half_w), float(y - ny * shaft_half_w))

            path = QPainterPath(p0)
            path.lineTo(p1)
            path.lineTo(p2)
            path.lineTo(p3)
            path.lineTo(p4)
            path.lineTo(p5)
            path.lineTo(p6)
            path.closeSubpath()

            arrow = QGraphicsPathItem(path)
            arrow.setPen(QPen(Qt.NoPen))
            arrow.setBrush(QBrush(color))
            arrow.setZValue(2)
            self.viewer.scene.addItem(arrow)
            self.viewer._vec_items.append(arrow)

        self.viewer.set_vector_opacity(float(self._vec_opacity_pct) / 100.0)
