# --- ADD/UPDATE imports ở đầu file ---
import math
import os, json
from typing import Optional, Dict, Any, List
import os
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QScrollArea, QFrame, QTextEdit, QComboBox, QDoubleSpinBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy,
    QSplitter, QLineEdit, QMessageBox, QColorDialog
)
from PyQt5.QtGui import QPixmap, QPixmapCache
from typing import Tuple, List, Dict, Optional
# backend UI3 đã có sẵn
from pedi_oku_landslide.pipeline.runners.ui3_backend import (
    auto_paths, list_lines, compute_profile, render_profile_png,
    clamp_groups_to_slip, auto_group_profile,
    estimate_slip_curve, fit_bezier_smooth_curve   # <-- thêm 2 hàm này
)
from pedi_oku_landslide.pipeline.runners.ui3_backend import rdp_indices_from_profile, rdp_points_from_profile
from PyQt5.QtGui import QPen, QColor
# ===================== ZOOMABLE GRAPHICS VIEW =====================
from PyQt5.QtWidgets import QGraphicsView, QToolBar, QAction
from typing import List, Tuple, Dict, Optional
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QGraphicsLineItem, QGraphicsRectItem
import csv
import traceback
import geopandas as gpd
from shapely.geometry import LineString
import rasterio

def _build_gdf_from_sections_csv(csv_path: str, dem_path: str) -> gpd.GeoDataFrame:
    """
    Đọc ui2/sections.csv (idx, x1, y1, x2, y2) và tạo GeoDataFrame lines.
    CRS lấy từ DEM (dem_path).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"sections.csv not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row.get("idx") or 0)
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
            except Exception:
                continue
            rows.append((idx, x1, y1, x2, y2))

    if not rows:
        return gpd.GeoDataFrame(columns=["idx", "name", "length_m", "geometry"], geometry="geometry")

    # Lấy CRS từ DEM
    crs = None
    try:
        with rasterio.open(dem_path) as ds:
            crs = ds.crs
    except Exception:
        pass

    idxs, xs1, ys1, xs2, ys2 = zip(*rows)
    geoms = [LineString([(x1, y1), (x2, y2)]) for (_, x1, y1, x2, y2) in rows]

    lengths = []
    for g in geoms:
        try:
            L = float(g.length)
        except Exception:
            L = float("nan")
        lengths.append(L)

    names = [f"Line {i}" for i in idxs]

    gdf = gpd.GeoDataFrame(
        {"idx": list(idxs), "name": names, "length_m": lengths},
        geometry=geoms,
        crs=crs,
    )
    return gdf

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom = 0

    def wheelEvent(self, e):
        zoom_in = 1.25
        zoom_out = 0.8
        if e.angleDelta().y() > 0:
            factor = zoom_in;  self._zoom += 1
        else:
            factor = zoom_out; self._zoom -= 1
        self.scale(factor, factor)

    def fit_to_scene(self):
        if not self.scene() or not self.scene().items():
            return
        r = self.scene().itemsBoundingRect()
        if r.isNull(): return
        self.resetTransform()
        self.fitInView(r, Qt.KeepAspectRatio)
        self._zoom = 0

    def set_100(self):
        self.resetTransform()
        self._zoom = 0

    def zoom_in(self):
        self.scale(1.25, 1.25)
        self._zoom += 1

    def zoom_out(self):
        self.scale(0.8, 0.8)
        self._zoom -= 1


Section = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


class CurveAnalyzeTab(QWidget):
    """
    UI3 (refactor): khung làm việc phân tích đường cong.
    - Nhận context từ Analyze/Section (project/run/run_dir)
    - Đọc danh sách sections từ UI2/sections.csv
    - Cho phép chọn line; hiển thị status; vẽ placeholder đồ thị (dz & slope)
    """

    # (Optional) khi bạn muốn phát tín hiệu đã lưu JSON v.v.
    curve_saved = pyqtSignal(str)  # emit path

    def __init__(self, base_dir: str):
        super().__init__()
        self.base_dir = base_dir
        self._ctx: Dict[str, str] = {"project": "", "run_label": "", "run_dir": ""}

        self._ax_top = None  # dict: {x_min,x_max,left_px,top_px,width_px,height_px}
        self._ax_bot = None

        # paths từ UI1/UI2
        self.dem_path = ""
        self.dx_path = ""
        self.dy_path = ""
        self.dz_path = ""
        self.lines_path = ""
        self.slip_path = ""

        # UI widgets chính (để dùng lại)
        self.line_combo = None
        self.status = None
        self.scene = None
        self.view = None
        self.group_table = None
        # --- state for grouping/guide overlays (phải ở CurveAnalyzeTab) ---
        self._px_per_m: Optional[float] = None  # pixels per meter
        self._sec_len_m: Optional[float] = None  # chiều dài tuyến (m)
        self._group_bounds: Dict[str, List[float]] = {}  # {line_id: [x_m ...]}
        self._guide_lines_top: List[QGraphicsLineItem] = []
        self._guide_lines_bot: List[QGraphicsLineItem] = []
        self._group_bands_bot: List[QGraphicsRectItem] = []
        self._img_ground: Optional[QGraphicsPixmapItem] = None
        self._img_rate0: Optional[QGraphicsPixmapItem] = None

        self._plot_x0_px = None  # ax_left_px trong PNG
        self._plot_w_px = None  # ax_width_px trong PNG
        self._x_min = None  # trục x (chainage) min trên hình
        self._x_max = None  # trục x (chainage) max trên hình

        self._build_ui()
    #
    # def _clear_group_guides(self) -> None:
    #     # Chỉ remove nếu item vẫn còn thuộc scene hiện tại
    #     cur_scene = getattr(self, "scene", None)
    #     for it in self._guide_lines_top:
    #         sc = it.scene()
    #         if sc is not None and (cur_scene is None or sc is cur_scene):
    #             sc.removeItem(it)
    #     for it in self._guide_lines_bot:
    #         sc = it.scene()
    #         if sc is not None and (cur_scene is None or sc is cur_scene):
    #             sc.removeItem(it)
    #     for it in self._group_bands_bot:
    #         sc = it.scene()
    #         if sc is not None and (cur_scene is None or sc is cur_scene):
    #             sc.removeItem(it)
    #     self._guide_lines_top.clear()
    #     self._guide_lines_bot.clear()
    #     self._group_bands_bot.clear()
    # #
    # def _chainage_to_xpx(self, x_m: float, panel: str = "top") -> float:
    #     ax = self._ax_top if panel == "top" else self._ax_bot
    #     if not ax or self._img_ground is None:
    #         return 0.0
    #     x_min = float(ax["x_min"]);
    #     x_max = float(ax["x_max"])
    #     left = float(ax["left_px"]);
    #     width = float(ax["width_px"])
    #     if x_max <= x_min or width <= 0.0:
    #         return 0.0
    #     px_per_m = width / (x_max - x_min)
    #     x_local = left + (float(x_m) - x_min) * px_per_m
    #     return float(self._img_ground.pos().x()) + x_local

    # def _line_id_current(self) -> str:
    #     """Trả về nhãn line hiện tại; fallback theo index nếu label trống."""
    #     if self.line_combo is None or self.line_combo.count() == 0:
    #         return "line_000"
    #     label = (self.line_combo.currentText() or "").strip()
    #     if label:
    #         return label
    #     return f"line_{self.line_combo.currentIndex() + 1:03d}"
    #
    # def _draw_group_guides_for_current_line(self) -> None:
    #     # guard
    #     if self.scene is None or self._img_ground is None:
    #         return
    #     if not self._ax_top or not self._ax_bot:
    #         return
    #
    #     line_id = self._line_id_current()
    #     bounds = self._group_bounds.get(line_id)
    #     if not bounds or len(bounds) < 2:
    #         self._clear_group_guides()
    #         return
    #
    #     self._clear_group_guides()
    #
    #     # Toạ độ gốc pixmap trong scene
    #     x0_img = float(self._img_ground.pos().x())
    #     y0_img = float(self._img_ground.pos().y())
    #
    #     # Pen dashed
    #     pen = QPen(QColor("#444444"))
    #     pen.setCosmetic(True)
    #     pen.setWidth(0)
    #     pen.setStyle(Qt.DashLine)
    #
    #     # Panel TOP
    #     t = self._ax_top  # {x_min,x_max,left_px,top_px,width_px,height_px}
    #     y_top = y0_img + float(t["top_px"])
    #     h_top = float(t["height_px"])
    #
    #     # Panel BOT
    #     b = self._ax_bot
    #     y_bot = y0_img + float(b["top_px"])
    #     h_bot = float(b["height_px"])
    #
    #     for x_m in bounds:
    #         # x theo từng panel (vì có thể x_min/x_max/bbox khác nhau)
    #         x_px_top = self._chainage_to_xpx(x_m, panel="top")
    #         x_px_bot = self._chainage_to_xpx(x_m, panel="bot")
    #
    #         ln_top = self.scene.addLine(x_px_top, y_top, x_px_top, y_top + h_top, pen)
    #         ln_bot = self.scene.addLine(x_px_bot, y_bot, x_px_bot, y_bot + h_bot, pen)
    #         self._guide_lines_top.append(ln_top)
    #         self._guide_lines_bot.append(ln_bot)
    #
    #     # Không tô band
    #     self._group_bands_bot.clear()

    def _save_groups_to_ui(self, groups: list, prof: dict, line_id: str, log_text: Optional[str] = None) -> None:
        try:
            js = {"line": self.line_combo.currentText(), "groups": groups}
            with open(self._groups_json_path(), "w", encoding="utf-8") as f:
                json.dump(js, f, ensure_ascii=False, indent=2)
            self._log(f"[✓] Saved group definition: {self._groups_json_path()}")
        except Exception as e:
            self._warn(f"[UI3] Cannot save groups JSON: {e}")

        if self.group_table is not None:
            self.group_table.setRowCount(0)
            for i, g in enumerate(groups, 1):
                self.group_table.insertRow(self.group_table.rowCount())
                self.group_table.setItem(i - 1, 0, QTableWidgetItem(str(g.get("id", f"G{i}"))))
                self.group_table.setItem(i - 1, 1, QTableWidgetItem(f'{float(g.get("start", 0.0)):.3f}'))
                self.group_table.setItem(i - 1, 2, QTableWidgetItem(f'{float(g.get("end", 0.0)):.3f}'))
                self._set_color_cell(i - 1, str(g.get("color", "")).strip())

        if "length_m" in prof and prof["length_m"] is not None:
            length_m = float(prof["length_m"])
        else:
            ch = prof.get("chain")
            length_m = float(ch[-1] - ch[0]) if ch is not None and len(ch) >= 2 else None
        self._append_ungrouped_row(groups, length_m)

        bounds_set = set()
        for g in groups:
            s = float(g.get("start", 0.0))
            e = float(g.get("end", 0.0))
            if e < s:
                s, e = e, s
            if length_m:
                s = max(0.0, min(length_m, s))
                e = max(0.0, min(length_m, e))
            bounds_set.add(s)
            bounds_set.add(e)
        bounds_m = sorted(bounds_set)

        self._group_bounds[line_id] = bounds_m
        self._sec_len_m = length_m

        if self._px_per_m is None and getattr(self, "_img_ground", None) and self._sec_len_m:
            W = self._img_ground.pixmap().width()
            self._px_per_m = float(W) / float(self._sec_len_m)

        if log_text:
            self._ok(log_text)

    def _on_auto_group(self) -> None:
        """Sinh group tự động như UI3, lưu JSON, cập nhật bảng, và vẽ guide (không re-render)."""
        try:
            # 1) Dữ liệu tuyến
            if not hasattr(self, "_gdf") or self._gdf is None or self._gdf.empty:
                self._warn("[UI3] No lines loaded from UI2.")
                return
            row = self.line_combo.currentIndex()
            if row < 0:
                self._warn("[UI3] No line selected.")
                return
            geom = self._gdf.geometry.iloc[row]

            # 2) Profile chỉ trong slip-zone
            prof = compute_profile(
                self.dem_path, self.dx_path, self.dy_path, self.dz_path,
                geom,
                step_m=self.step_box.value(),
                smooth_win=11, smooth_poly=2,
                slip_mask_path=self.slip_path,  # phải đúng path
                slip_only=True
            )
            if not prof:
                self._err("[UI3] Empty profile.")
                return

            # 3) Auto-group + clamp (y như UI3)
            groups = auto_group_profile(prof)
            groups = clamp_groups_to_slip(prof, groups)
            if not groups:
                self._warn("[UI3] Auto grouping produced no segments within slip zone.")
                return

            line_id = self._line_id_current()
            self._save_groups_to_ui(
                groups,
                prof,
                line_id,
                log_text=f"[UI3] Auto Group done for '{line_id}': {len(groups)} groups.",
            )
            self._render_current_safe()
        except Exception:
            pass

    def _on_draw_curve(self) -> None:
        """Tính và vẽ đường cong (overlay) vào PNG preview hiện tại."""
        try:
            # 1) Lấy line và profile TRONG slip-zone
            if not hasattr(self, "_gdf") or self._gdf is None or self._gdf.empty:
                self._warn("[UI3] No lines.");
                return
            row = self.line_combo.currentIndex()
            if row < 0:
                self._warn("[UI3] Select a line first.");
                return

            geom = self._gdf.geometry.iloc[row]
            prof = compute_profile(
                self.dem_path, self.dx_path, self.dy_path, self.dz_path,
                geom, step_m=self.step_box.value(),
                smooth_win=11, smooth_poly=2,
                slip_mask_path=self.slip_path, slip_only=False
            )
            if not prof or len(prof.get("chain", [])) < 6:
                self._warn("[UI3] Empty/too-short slip profile.");
                return

            # Save RDP indices (numpy-based _rdp) to output/RDP_indicies.json
            try:
                out_dir = os.path.join(self.base_dir, "output")
                os.makedirs(out_dir, exist_ok=True)

                rdp_indices = rdp_indices_from_profile(prof, rdp_eps_m=0.5)
                out_json_idx = os.path.join(out_dir, "RDP_indicies.json")
                payload_idx = {
                    "line": self._line_id_current(),
                    "eps": 0.5,
                    "rdp_indices": rdp_indices,
                }
                with open(out_json_idx, "w", encoding="utf-8") as f:
                    json.dump(payload_idx, f, ensure_ascii=False, indent=2)
                self._log(f"[UI3] RDP indices saved: {out_json_idx} (n={len(rdp_indices)})")

                rdp_points = rdp_points_from_profile(prof, rdp_eps_m=0.5)
                out_json_pts = os.path.join(out_dir, "RDP_points.json")
                payload_pts = {
                    "line": self._line_id_current(),
                    "eps": 0.5,
                    "rdp_points": rdp_points,
                }
                with open(out_json_pts, "w", encoding="utf-8") as f:
                    json.dump(payload_pts, f, ensure_ascii=False, indent=2)
                self._log(f"[UI3] RDP points saved: {out_json_pts} (n={len(rdp_points)})")
            except Exception as e:
                self._warn(f"[UI3] RDP save failed: {e}")

            # 2) Lấy groups (ưu tiên bảng → file), rồi clamp vào slip-zone
            groups = self._load_groups_for_current_line()
            if not groups:
                try:
                    line_id = self._line_id_current()
                    gpath = self._groups_json_path_for(line_id)
                    # hoặc:
                    # gpath = self._groups_json_path()
                    if os.path.exists(gpath):
                        with open(gpath, "r", encoding="utf-8") as f:
                            groups = (json.load(f) or {}).get("groups", []) or []
                except Exception:
                    groups = []
            if not groups:
                groups = auto_group_profile(prof)
                groups = clamp_groups_to_slip(prof, groups)
                if not groups:
                    self._warn("[UI3] Auto grouping produced no segments within slip zone.");
                    return
                line_id = self._line_id_current()
                self._save_groups_to_ui(
                    groups,
                    prof,
                    line_id,
                    log_text=f"[UI3] Auto Group (implicit) for '{line_id}': {len(groups)} groups.",
                )
            else:
                groups = clamp_groups_to_slip(prof, groups)
                if not groups:
                    self._warn("[UI3] No groups within slip zone.");
                    return
            #
            # # 3) Base curve trong slip-zone (theo UI3 gốc)
            # base = estimate_slip_curve(
            #     prof, groups,
            #     ds=0.2, smooth_factor=0.1,
            #     depth_gain=8, min_depth=2
            # )
            # if len(base.get("chain", [])) < 6:
            #     self._warn("[UI3] Not enough points to fit curve.");
            #     return
            #
            # # 4) Fit Bezier “main” (đơn giản như nhánh main trong UI3 gốc)
            # bez = fit_bezier_smooth_curve(
            #     chain=np.asarray(prof["chain"]),
            #     elevg=np.asarray(prof["elev_s"]),
            #     target_s=np.asarray(base["chain"]),
            #     target_z=np.asarray(base["elev"]),
            #     c0=0.20, c1=0.40, clearance=0.20
            # )
            # 3) Base curve trong slip-zone (polyline đơn giản – giống backbone)
            base = estimate_slip_curve(
                prof, groups,
                ds=0.2, smooth_factor=0.1,
                depth_gain=8, min_depth=2
            )
            x_base = np.asarray(base.get("chain", []), dtype=float)
            z_base = np.asarray(base.get("elev", []), dtype=float)

            # Lọc NaN / inf
            mask = np.isfinite(x_base) & np.isfinite(z_base)
            x_base = x_base[mask]
            z_base = z_base[mask]

            if x_base.size < 2:
                self._warn("[UI3] Slip curve has too few valid points.")
                return

            # Sắp xếp theo chainage tăng dần cho chắc
            order = np.argsort(x_base)
            x_base = x_base[order]
            z_base = z_base[order]

            self._log(
                f"[UI3] Slip curve pts={x_base.size}, "
                f"chain=[{x_base.min():.2f}, {x_base.max():.2f}]"
            )
            # 4) Thử fit Bezier mượt giống UI3 cũ
            x_smooth, z_smooth = x_base, z_base  # fallback mặc định

            try:
                bez = fit_bezier_smooth_curve(
                    chain=np.asarray(prof["chain"], dtype=float),
                    elevg=np.asarray(prof["elev_s"], dtype=float),
                    target_s=x_base,
                    target_z=z_base,
                    c0=0.20,
                    c1=0.40,
                    clearance=0.20,
                )

                xb = np.asarray(bez.get("chain", []), dtype=float)
                zb = np.asarray(bez.get("elev", []), dtype=float)
                m2 = np.isfinite(xb) & np.isfinite(zb)
                xb = xb[m2]
                zb = zb[m2]

                if xb.size >= 2:
                    x_smooth, z_smooth = xb, zb
                    self._log(f"[UI3] Bezier slip curve OK: n={xb.size}")
                else:
                    self._warn("[UI3] Bezier returned too few points; using base curve.")

            except Exception as e:
                # Lỗi DGELSD/SVD... thì báo warning nhưng vẫn không crash
                self._warn(f"[UI3] Bezier fit failed, using base curve. ({e})")

            # 5) Re-render PNG hiện tại + overlay curve
            line_id = self._line_id_current()
            out_png = self._profile_png_path_for(line_id)

            msg, path = render_profile_png(
                prof, out_png,
                y_min=None, y_max=None,
                x_min=None, x_max=None,
                vec_scale=self.vscale.value(),
                vec_width=self.vwidth.value(),
                head_len=6.0, head_w=4.0,
                highlight_theta=None,
                group_ranges=groups,
                overlay_curves=[(x_smooth, z_smooth, "#bf00ff", "Slip curve")]
            )
            self._log(msg)
            if not path or not os.path.exists(path):
                return

            # 6) Nạp PNG mới (clear cache để tránh ảnh cũ)
            from PyQt5.QtGui import QPixmap, QPixmapCache
            QPixmapCache.clear()
            self.scene.clear()
            item = QGraphicsPixmapItem(QPixmap(path))
            self.scene.addItem(item)
            self._img_ground = item
            self._img_rate0 = item

            if getattr(self, "_first_show", True):
                self.view.fit_to_scene()
                self._first_show = False

            self._ok("[UI3] Curve drawn on current section.")
        except Exception as e:
            self._err(f"[UI3] Draw Curve error: {e}")
            raise

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
        # self.setStyleSheet(style)

    # -------------------- UI --------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ===== BODY: dùng QSplitter để panel trái/phải kéo được =====
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        # ===== LEFT: controls (KHÔNG dùng QScrollArea nữa) =====
        left_container = QWidget()
        left = QVBoxLayout(left_container)
        left.setContentsMargins(6, 6, 6, 6)
        left.setSpacing(8)
        splitter.addWidget(left_container)

        # Project info – giống Section tab
        box_proj = QGroupBox("Project")
        lp = QVBoxLayout(box_proj)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Name:"))
        self.edit_project = QLineEdit()
        self.edit_project.setPlaceholderText("—")
        self.edit_project.setReadOnly(True)
        row1.addWidget(self.edit_project, 1)
        lp.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Run label:"))
        self.edit_runlabel = QLineEdit()
        self.edit_runlabel.setPlaceholderText("—")
        self.edit_runlabel.setReadOnly(True)
        row2.addWidget(self.edit_runlabel, 1)
        lp.addLayout(row2)

        left.addWidget(box_proj)

        # Line selection
        box_sel = QGroupBox("Sections")
        ls = QHBoxLayout(box_sel)
        self.line_combo = QComboBox()
        self.line_combo.currentIndexChanged.connect(self._on_line_changed)
        btn_render = QPushButton("Render Section")
        btn_render.clicked.connect(self._render_current_safe)
        ls.addWidget(self.line_combo)
        ls.addWidget(btn_render)
        left.addWidget(box_sel)

        # Advanced (quiver + axes)
        box_adv = QGroupBox("Advanced display")
        la = QVBoxLayout(box_adv)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Step (m):"))
        self.step_box = QDoubleSpinBox()
        self.step_box.setDecimals(2)
        self.step_box.setValue(0.20)
        self.step_box.setMaximum(1e6)
        row1.addWidget(self.step_box)
        row1.addStretch(1)
        la.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Vec scale:"))
        self.vscale = QDoubleSpinBox()
        self.vscale.setDecimals(3)
        self.vscale.setValue(0.1)
        self.vscale.setMaximum(1e6)
        row2.addWidget(self.vscale)
        row2.addWidget(QLabel("Width:"))
        self.vwidth = QDoubleSpinBox()
        self.vwidth.setDecimals(4)
        self.vwidth.setValue(0.0015)
        self.vwidth.setMaximum(1.0)
        row2.addWidget(self.vwidth)
        la.addLayout(row2)

        left.addWidget(box_adv)

        # Group table
        box_grp = QGroupBox("Group")
        lg = QVBoxLayout(box_grp)
        self.group_table = QTableWidget(0, 4)
        self.group_table.setHorizontalHeaderLabels(
            ["Group ID", "Start (m)", "End (m)", "Color"]
        )
        self.group_table.verticalHeader().setVisible(False)
        self.group_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.group_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.group_table.cellDoubleClicked.connect(self._on_group_cell_double_clicked)
        lg.addWidget(self.group_table)

        rowg = QHBoxLayout()
        self.btn_add_g = QPushButton("Add")
        self.btn_add_g.clicked.connect(self._on_add_group)
        self.btn_del_g = QPushButton("Delete")
        self.btn_del_g.clicked.connect(self._on_delete_group)
        self.btn_draw_curve = QPushButton("Draw Curve")
        self.btn_draw_curve.clicked.connect(self._on_draw_curve)
        self.btn_auto_group = QPushButton("Auto-Gene Vector Group")
        self.btn_auto_group.clicked.connect(self._on_auto_group)

        rowg.addWidget(self.btn_auto_group)
        rowg.addWidget(self.btn_add_g)
        rowg.addWidget(self.btn_del_g)
        rowg.addStretch(1)
        rowg.addWidget(self.btn_draw_curve)
        lg.addLayout(rowg)
        left.addWidget(box_grp)

        # Status
        box_st = QGroupBox("Status")
        ls = QVBoxLayout(box_st)
        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setFixedHeight(110)
        ls.addWidget(self.status)
        left.addWidget(box_st)

        left.addStretch(1)

        # ===== RIGHT: preview (zoomable like AnalyzeTab) =====
        right_container = QWidget()
        right_wrap = QVBoxLayout(right_container)
        right_wrap.setContentsMargins(0, 0, 0, 0)

        zoombar = QToolBar()
        act_in = QAction("Zoom +", self)
        act_in.triggered.connect(lambda: self.view.zoom_in())
        act_out = QAction("Zoom –", self)
        act_out.triggered.connect(lambda: self.view.zoom_out())
        act_fit = QAction("Fit", self)
        act_fit.triggered.connect(lambda: self.view.fit_to_scene())
        act_100 = QAction("100%", self)
        act_100.triggered.connect(lambda: self.view.set_100())
        zoombar.addAction(act_in)
        zoombar.addAction(act_out)
        zoombar.addAction(act_fit)
        zoombar.addAction(act_100)
        zoombar.setIconSize(QSize(22, 22))
        zoombar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        zoombar.setContentsMargins(0, 0, 0, 0)
        zoombar.setStyleSheet("QToolBar { spacing: 6px; background: transparent; border: none; }")
        right_wrap.addWidget(zoombar)

        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_wrap.addWidget(self.view, 1)

        splitter.addWidget(right_container)

        # Tỉ lệ ban đầu giữa panel trái/phải
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 900])

        self._apply_button_style()

    def _row(self) -> QHBoxLayout:
        r = QHBoxLayout()
        r.setContentsMargins(0, 0, 0, 0)
        return r

    # -------------------- Context --------------------
    def set_context(self, project: str, run_label: str, run_dir: str) -> None:
        """Du?c g?i t? MainWindow sau khi Analyze/Section xong."""
        self._ctx.update({"project": project, "run_label": run_label, "run_dir": run_dir})
        # uu tiˆn d?c t? ui_shared_data.json (do Analyze/Section ghi)
        shared_jsons = [
            os.path.join(run_dir, "ui_shared_data.json"),
            os.path.join(self.base_dir, "output", "ui_shared_data.json"),
            os.path.join(self.base_dir, "output", "UI1", "ui_shared_data.json"),
        ]
        js = {}
        for p in shared_jsons:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        js.update(json.load(f))
                except Exception:
                    pass

        # fallback to auto_paths when json missing
        ap = auto_paths()

        meta_inputs = {}
        meta_processed = {}
        try:
            meta_path = os.path.join(run_dir, "ingest_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}
                meta_inputs = meta.get("inputs") or {}
                meta_processed = meta.get("processed") or {}
        except Exception:
            pass

        def _pick_first(*cands: str) -> str:
            for p in cands:
                if p and os.path.exists(p):
                    return p
            return ""

        # Prefer raw DEM inputs over smoothed/cropped outputs
        self.dem_path = _pick_first(
            meta_inputs.get("before_dem") or "",
            js.get("dem_ground_path") or "",
            ap.get("dem", ""),
            meta_inputs.get("before_asc") or "",
            os.path.join(run_dir, "input", "before_dem.tif"),
            os.path.join(run_dir, "input", "before.asc"),
            meta_processed.get("dem_cropped") or "",
        )
        self._log(f"[UI3] Ground DEM input: {self.dem_path}")
        self.dx_path = _pick_first(
            js.get("dx_path") or "",
            ap.get("dx", ""),
            os.path.join(run_dir, "ui1", "dx.tif"),
        )
        self.dy_path = _pick_first(
            js.get("dy_path") or "",
            ap.get("dy", ""),
            os.path.join(run_dir, "ui1", "dy.tif"),
        )
        self.dz_path = _pick_first(
            js.get("dz_path") or "",
            ap.get("dz", ""),
            os.path.join(run_dir, "ui1", "dz.tif"),
        )
        # QUAN TR?NG: lines t? UI2
        # QUAN TR?NG: kh“ng set lines_path legacy n?a
        self.lines_path = ""  # d? loader t? quy?t d?nh t? sections.csv

        # slip-zone mask (d? ch? v? trong v-ng tru?t)
        self.slip_path = js.get("slip_path") or ap.get("slip", "")
        # Prefer run-scoped slip mask from ingest_meta.json (pipeline step_detect)
        try:
            slip_mask = meta_processed.get("slip_mask")
            if slip_mask:
                self.slip_path = slip_mask
        except Exception:
            pass

        # Fallback to current run's ui1 output if still missing
        if not self.slip_path:
            self.slip_path = os.path.join(run_dir, "ui1", "landslide_mask.tif")
        # sau khi self.slip_path = os.path.join(..., "ui1", "step7_slipzone", "slip_zone.asc")
        if not self.slip_path or not os.path.exists(self.slip_path):
            self._warn("[UI3] Slip-zone mask not found. Vectors outside landslide may appear.")

        if not os.path.exists(self.slip_path):
            alt = self.slip_path.replace(".asc", ".tif")
            if os.path.exists(alt):
                self.slip_path = alt

        self._load_lines_into_combo()

    def reset_session(self) -> None:
        """
        Reset tab Curve Analyze cho New Session:
        - Xoá context project/run
        - Xoá line combo, bảng group
        - Clear hình nền / guide trên QGraphicsScene
        - Clear status
        """
        # 1) Reset context
        self._ctx = {"project": "", "run_label": "", "run_dir": ""}

        # 2) Reset thông tin hiển thị project/run nếu có label
        # 2) Reset thông tin hiển thị project/run nếu có
        if hasattr(self, "edit_project") and self.edit_project is not None:
            self.edit_project.clear()
        if hasattr(self, "edit_runlabel") and self.edit_runlabel is not None:
            self.edit_runlabel.clear()

        # 3) Reset các path dữ liệu
        self.dem_path = ""
        self.dx_path = ""
        self.dy_path = ""
        self.dz_path = ""
        self.lines_path = ""
        self.slip_path = ""

        # 4) Clear combo line (list tuyến cắt)
        try:
            if self.line_combo is not None:
                self.line_combo.blockSignals(True)
                self.line_combo.clear()
                self.line_combo.blockSignals(False)
        except Exception:
            pass

        # 5) Clear bảng group
        try:
            if self.group_table is not None:
                self.group_table.setRowCount(0)
        except Exception:
            pass

        # 6) Clear scene & reset zoom viewer
        try:
            if self.scene is not None:
                self.scene.clear()
        except Exception:
            pass
        try:
            if self.view is not None:
                self.view.resetTransform()
        except Exception:
            pass

        # 7) Reset state nội bộ cho guide / overlay
        self._px_per_m = None
        self._sec_len_m = None
        self._group_bounds.clear()
        self._guide_lines_top.clear()
        self._guide_lines_bot.clear()
        self._group_bands_bot.clear()
        self._img_ground = None
        self._img_rate0 = None
        self._plot_x0_px = None
        self._plot_w_px = None
        self._x_min = None
        self._x_max = None
        self._current_idx = 0

        # 8) Clear status log
        try:
            if self.status is not None:
                self.status.clear()
        except Exception:
            pass

        # 9) Log nhẹ cho dễ debug
        try:
            self._log("[UI3] Curve tab session reset.")
        except Exception:
            pass

    def _load_lines_into_combo(self) -> None:
        """
        Đọc ui2/sections.csv của run hiện tại, build GeoDataFrame line,
        đổ vào combo và set self._gdf.
        (Không dùng selected_lines.gpkg cũ nữa.)
        """
        self.line_combo.blockSignals(True)
        self.line_combo.clear()
        try:
            run_dir = self._ctx.get("run_dir") or ""
            if not run_dir:
                self._log("[!] Run context is empty – cannot load sections.")
                return

            csv_path = os.path.join(run_dir, "ui2", "sections.csv")
            gdf = _build_gdf_from_sections_csv(csv_path, self.dem_path)
            if gdf is None or gdf.empty:
                self._log(f"[!] No sections in csv:\n{csv_path}")
                return

            # Gán lại cho UI3 dùng
            self._gdf = gdf

            # Tạo label cho combo: "Line i  (xxx.x m)"
            labels: List[str] = []
            for _, row in gdf.iterrows():
                base = row.get("name") or f"Line {int(row.get('idx', 0) or 0)}"
                L = float(row.get("length_m", float("nan")))
                if math.isfinite(L) and L > 0:
                    labels.append(f"{base}  ({L:.1f} m)")
                else:
                    labels.append(base)

            self.line_combo.addItems(labels)
            self._log(f"[i] Loaded {len(labels)} lines from ui2/sections.csv.")

            # (Tuỳ chọn) nếu bạn vẫn muốn có GPKG để các chỗ khác dùng:
            # try:
            #     os.makedirs(os.path.join(run_dir, "UI2", "step2_selected_lines"), exist_ok=True)
            #     gpkg_path = os.path.join(run_dir, "UI2", "step2_selected_lines", "selected_lines.gpkg")
            #     gdf.to_file(gpkg_path, driver="GPKG")
            #     self.lines_path = gpkg_path
            # except Exception:
            #     pass

        except Exception as e:
            self._log(f"[!] Cannot load lines from sections.csv: {e}")
        finally:
            self.line_combo.blockSignals(False)

        # cập nhật nhãn project
        # cập nhật ô Project / Run label giống các tab khác
        pj = self._ctx.get("project") or "—"
        rl = self._ctx.get("run_label") or "—"

        if hasattr(self, "edit_project") and self.edit_project is not None:
            self.edit_project.setText(pj)
        if hasattr(self, "edit_runlabel") and self.edit_runlabel is not None:
            self.edit_runlabel.setText(rl)

    def _on_line_changed(self, _idx: int) -> None:
        self._log("[i] Line changed. Click 'Render Section' to preview.")
        self._populate_group_table_for_current_line()  # <-- đổi tên ở đây

    # def _groups_json_path(self) -> str:
    #     line_label = self.line_combo.currentText().strip() or f"line_{self.line_combo.currentIndex() + 1:03d}"
    #     safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in line_label)
    #     gdir = os.path.join(self.base_dir, "output", "UI3", "groups");
    #     os.makedirs(gdir, exist_ok=True)
    #     return os.path.join(gdir, f"{safe}.json")

    def _populate_group_table_for_current_line(self) -> None:
        path = self._groups_json_path()
        self.group_table.setRowCount(0)
        loaded = 0
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    js = json.load(f)
                for g in js.get("groups", []):
                    r = self.group_table.rowCount();
                    self.group_table.insertRow(r)
                    self.group_table.setItem(r, 0, QTableWidgetItem(str(g.get("id", ""))))
                    self.group_table.setItem(r, 1, QTableWidgetItem(f'{float(g.get("start", 0.0)):.3f}'))
                    self.group_table.setItem(r, 2, QTableWidgetItem(f'{float(g.get("end", 0.0)):.3f}'))
                    self._set_color_cell(r, str(g.get("color", "")).strip())
                    loaded += 1
            except Exception as e:
                self._log(f"[!] Cannot read groups: {e}")
        if loaded:
            self._append_ungrouped_row(self._read_groups_from_table(), self._sec_len_m)
        if loaded == 0:
            # 3 dòng trống mặc định
            for _ in range(3):
                r = self.group_table.rowCount();
                self.group_table.insertRow(r)
                self.group_table.setItem(r, 0, QTableWidgetItem(""))

    def _read_group_table(self) -> List[dict]:
        out = []
        rc = self.group_table.rowCount()
        for r in range(rc):
            gid = (self.group_table.item(r, 0).text().strip() if self.group_table.item(r, 0) else "")
            if gid.upper() == "UNGROUPED":
                continue
            s = (self.group_table.item(r, 1).text().strip() if self.group_table.item(r, 1) else "")
            e = (self.group_table.item(r, 2).text().strip() if self.group_table.item(r, 2) else "")
            color = self._get_color_cell_value(r)
            if not gid and not s and not e:
                continue
            try:
                s_val = float(s);
                e_val = float(e)
                if e_val < s_val: s_val, e_val = e_val, s_val
                out.append({"id": gid or f"G{len(out) + 1}", "start": s_val, "end": e_val, "color": color or None})
            except:
                self._log(f"[!] Row {r + 1}: start/end invalid.")
        return out

    def _get_groups_for_current_line(self):
        """Alias để tương thích với các chỗ gọi cũ; trả về list nhóm hiện hành."""
        return self._load_groups_for_current_line()

    # --- run-scoped path helpers (save under <run_dir>/ui3/...) ---
    def _ui3_run_dir(self) -> str:
        run_dir = (self._ctx.get("run_dir") or "").strip()
        if not run_dir:
            raise RuntimeError("[UI3] Run context is empty. Call set_context() first.")
        path = os.path.join(run_dir, "ui3")
        os.makedirs(path, exist_ok=True)
        return path

    def _preview_dir(self) -> str:
        path = os.path.join(self._ui3_run_dir(), "preview")
        os.makedirs(path, exist_ok=True)
        return path

    def _groups_dir(self) -> str:
        path = os.path.join(self._ui3_run_dir(), "groups")
        os.makedirs(path, exist_ok=True)
        return path

    def _line_id_current(self) -> str:
        # dùng id ổn định để tên file không đụng nhau (ưu tiên tên trong combo)
        if hasattr(self, "line_combo"):
            txt = self.line_combo.currentText().strip() or f"line_{self.line_combo.currentIndex() + 1:03d}"
            return txt.replace(" ", "_")
        # fallback
        row = getattr(self, "line_combo", None).currentIndex() if hasattr(self, "line_combo") else 0
        return f"line_{(row or 0) + 1:03d}"

    def _profile_png_path_for(self, line_id: str) -> str:
        return os.path.join(self._preview_dir(), f"profile_{line_id}.png")

    def _groups_json_path_for(self, line_id: str) -> str:
        return os.path.join(self._groups_dir(), f"{line_id}.json")

    # tiện gọi ở mọi nơi hiện tại (giữ chữ ký cũ không tham số)
    def _profile_png_path(self) -> str:
        return self._profile_png_path_for(self._line_id_current())

    def _groups_json_path(self) -> str:
        return self._groups_json_path_for(self._line_id_current())

    # def _groups_json_path(self, line_id: str) -> str:
    #     return os.path.join(self._groups_dir(), f"{line_id}.json")

    def _render_current(self) -> None:
        # 1) Kiểm tra dữ liệu tuyến
        if not hasattr(self, "_gdf") or self._gdf is None or self._gdf.empty:
            self._log("[!] No lines.")
            return
        row = self.line_combo.currentIndex()
        if row < 0:
            self._log("[!] Select a line first.")
            return

        # Log resolved inputs for debugging
        self._log(f"[UI3] DEM: {self.dem_path}")
        self._log(f"[UI3] DX:  {self.dx_path}")
        self._log(f"[UI3] DY:  {self.dy_path}")
        self._log(f"[UI3] DZ:  {self.dz_path}")
        self._log(f"[UI3] MASK:{self.slip_path}")

        # 2) Tính profile (ground full line, không giới hạn slip-zone)
        geom = self._gdf.geometry.iloc[row]
        prof = compute_profile(
            self.dem_path, self.dx_path, self.dy_path, self.dz_path,
            geom,
            step_m=self.step_box.value(),
            smooth_win=11, smooth_poly=2,
            slip_mask_path=self.slip_path, slip_only=False
        )
        if not prof:
            self._log("[!] Empty profile.")
            return

        # 3) Lấy nhóm hiện hành (ưu tiên JSON, fallback bảng), rồi clamp theo slip
        groups = self._get_groups_for_current_line()  # <- dùng helper trả list
        if groups:
            groups = clamp_groups_to_slip(prof, groups)

        # 4) Gọi backend vẽ PNG (tô màu theo group nếu có)
        line_id = self._line_id_current()
        out_png = self._profile_png_path_for(line_id)

        msg, path = render_profile_png(
            prof, out_png,
            y_min=None, y_max=None,
            x_min=None, x_max=None,
            vec_scale=self.vscale.value(),
            vec_width=self.vwidth.value(),
            head_len=6.0, head_w=4.0,
            highlight_theta=None,
            group_ranges=groups if groups else None
        )
        self._log(msg)
        if not path or not os.path.exists(path):
            return

        # 5) Đưa PNG lên scene (luôn cập nhật reference mới)
        QPixmapCache.clear()

        pm = QPixmap(path)
        if pm.isNull():
            self._err("[UI3] Cannot load PNG with curve overlay.")
            return

        self.scene.clear()
        item = QGraphicsPixmapItem(pm)
        self.scene.addItem(item)
        self._img_ground = item
        self._img_rate0 = item

        # Fit lần đầu
        if getattr(self, "_first_show", True):
            self.view.fit_to_scene()
            self._first_show = False

        # KHÔNG vẽ overlay guide nữa vì đã vẽ sẵn trong backend

    def _render_current_safe(self) -> None:
        try:
            self._render_current()
        except Exception:
            msg = "[UI3] Render Section failed. See log for details."
            self._append_status(msg)
            log_path = ""
            try:
                run_dir = (self._ctx.get("run_dir") or "").strip()
                if run_dir:
                    log_dir = os.path.join(run_dir, "ui3")
                    os.makedirs(log_dir, exist_ok=True)
                    log_path = os.path.join(log_dir, "ui3_render_error.log")
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(traceback.format_exc())
                        f.write("\n")
            except Exception:
                pass

            try:
                details = msg
                if log_path:
                    details += f"\nLog: {log_path}"
                QMessageBox.critical(self, "UI3 Render Section Error", details)
            except Exception:
                pass

    def _on_add_group(self):
        r = self._find_ungrouped_row()
        if r is None:
            r = self.group_table.rowCount()
        self.group_table.insertRow(r)
        n_groups = len(self._read_groups_from_table()) + 1
        self.group_table.setItem(r, 0, QTableWidgetItem(f"G{n_groups}"))

    def _on_delete_group(self):
        rows = sorted({i.row() for i in self.group_table.selectedIndexes()}, reverse=True)
        if not rows:
            self._log("[!] Select row(s) to delete.");
            return
        for r in rows:
            gid = self.group_table.item(r, 0)
            if gid and gid.text().strip().upper() == "UNGROUPED":
                continue
            self.group_table.removeRow(r)


    def _read_groups_from_table(self):
        """Đọc nhóm từ bảng UI (cột: Group ID | Start | End | Color)."""
        rows = self.group_table.rowCount()
        out = []
        for r in range(rows):
            gid = self.group_table.item(r, 0)
            s = self.group_table.item(r, 1)
            e = self.group_table.item(r, 2)
            try:
                gid = gid.text().strip() if gid else f"G{r + 1}"
                if gid.upper() == "UNGROUPED":
                    continue
                s = float(s.text()) if s and s.text() not in ("", None) else None
                e = float(e.text()) if e and e.text() not in ("", None) else None
                if s is None or e is None:
                    continue
                if e < s: s, e = e, s
                out.append({"id": gid, "start": s, "end": e, "color": self._get_color_cell_value(r)})
            except Exception:
                continue
        return out

    def _find_ungrouped_row(self) -> Optional[int]:
        rows = self.group_table.rowCount()
        for r in range(rows):
            gid = self.group_table.item(r, 0)
            if gid and gid.text().strip().upper() == "UNGROUPED":
                return r
        return None

    def _compute_ungrouped_ranges(self, groups: list, smin: float, smax: float) -> List[tuple]:
        if smin is None or smax is None:
            return []
        if smax <= smin:
            return []
        norm = []
        for g in (groups or []):
            try:
                s = float(g.get("start", 0.0))
                e = float(g.get("end", 0.0))
            except Exception:
                continue
            if e < s:
                s, e = e, s
            s = max(s, smin)
            e = min(e, smax)
            if e > s:
                norm.append((s, e))
        norm.sort(key=lambda x: x[0])

        gaps = []
        cur = smin
        for s, e in norm:
            if s > cur:
                gaps.append((cur, s))
            if e > cur:
                cur = e
        if cur < smax:
            gaps.append((cur, smax))
        return gaps

    def _append_ungrouped_row(self, groups: list, length_m: Optional[float]) -> None:
        r = self._find_ungrouped_row()
        if r is not None:
            self.group_table.removeRow(r)

        if length_m is not None:
            smin, smax = 0.0, float(length_m)
        else:
            starts = [float(g.get("start", 0.0)) for g in (groups or []) if g.get("start", None) is not None]
            ends = [float(g.get("end", 0.0)) for g in (groups or []) if g.get("end", None) is not None]
            if not starts or not ends:
                return
            smin, smax = min(starts), max(ends)

        gaps = self._compute_ungrouped_ranges(groups, smin, smax)
        if not gaps:
            return

        starts = "; ".join([f"{s:.3f}" for s, _ in gaps])
        ends = "; ".join([f"{e:.3f}" for _, e in gaps])

        r = self.group_table.rowCount()
        self.group_table.insertRow(r)
        item_id = QTableWidgetItem("UNGROUPED")
        item_s = QTableWidgetItem(starts)
        item_e = QTableWidgetItem(ends)
        for it in (item_id, item_s, item_e):
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        self.group_table.setItem(r, 0, item_id)
        self.group_table.setItem(r, 1, item_s)
        self.group_table.setItem(r, 2, item_e)
        self._set_color_cell(r, "")
        color_item = self.group_table.item(r, 3)
        if color_item:
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)

    def _normalize_color_hex(self, color_hex: str) -> str:
        if not color_hex:
            return ""
        c = color_hex.strip()
        if not c:
            return ""
        if not c.startswith("#"):
            c = f"#{c}"
        qc = QColor(c)
        return qc.name() if qc.isValid() else ""

    def _set_color_cell(self, row: int, color_hex: str) -> None:
        c = self._normalize_color_hex(color_hex)
        item = self.group_table.item(row, 3)
        if item is None:
            item = QTableWidgetItem("")
            self.group_table.setItem(row, 3, item)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        if c:
            item.setData(Qt.UserRole, c)
            item.setToolTip(c)
            item.setText("")
            item.setBackground(QColor(c))
        else:
            item.setData(Qt.UserRole, "")
            item.setToolTip("")
            item.setText("")
            item.setBackground(QColor(0, 0, 0, 0))

    def _get_color_cell_value(self, row: int) -> str:
        item = self.group_table.item(row, 3)
        if item is None:
            return ""
        val = item.data(Qt.UserRole)
        if isinstance(val, str) and val.strip():
            return val.strip()
        txt = item.text().strip() if item.text() else ""
        return self._normalize_color_hex(txt)

    def _on_group_cell_double_clicked(self, row: int, col: int) -> None:
        if col != 3:
            return
        gid = self.group_table.item(row, 0)
        if gid and gid.text().strip().upper() == "UNGROUPED":
            return
        current = self._get_color_cell_value(row)
        initial = QColor(current) if current else QColor(255, 255, 255)
        color = QColorDialog.getColor(initial, self, "Select group color")
        if color.isValid():
            self._set_color_cell(row, color.name())
            # Re-render to reflect new group colors on vectors
            self._render_current_safe()

    def _load_groups_for_current_line(self):
        """Ưu tiên đọc từ bảng (phản ánh chỉnh sửa/delete mới nhất); nếu trống thì đọc JSON."""
        table_groups = self._read_groups_from_table()
        if table_groups:
            return table_groups

        js_path = self._groups_json_path()
        if os.path.exists(js_path):
            try:
                import json
                with open(js_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                gs = data.get("groups", [])
                # chuẩn hoá khóa
                norm = []
                for i, g in enumerate(gs, 1):
                    s = g.get("start", g.get("start_chainage"))
                    e = g.get("end", g.get("end_chainage"))
                    if s is None or e is None:
                        continue
                    s = float(s)
                    e = float(e)
                    if e < s:
                        s, e = e, s
                    norm.append({
                        "id": g.get("id", f"G{i}"),
                        "start": s, "end": e,
                        "color": g.get("color", "")
                    })
                if norm:
                    return norm
            except Exception:
                pass
        return []

    def _on_confirm_groups(self):
        """Lưu JSON nhóm và re-render."""
        if self.line_combo.count() == 0:
            self._log("[!] No line.");
            return
        groups = self._read_group_table()
        # nếu trống → auto
        if not groups:
            # cần profile để auto
            row = self.line_combo.currentIndex()
            geom = self._gdf.geometry.iloc[row]
            prof = compute_profile(
                self.dem_path, self.dx_path, self.dy_path, self.dz_path,
                geom, step_m=self.step_box.value(),
                smooth_win=11, smooth_poly=2,
                slip_mask_path=self.slip_path, slip_only=True
            )
            if prof:
                groups = auto_group_profile(prof)

        # clamp trong slip-zone
        try:
            row = self.line_combo.currentIndex()
            geom = self._gdf.geometry.iloc[row]
            prof = compute_profile(
                self.dem_path, self.dx_path, self.dy_path, self.dz_path,
                geom, step_m=self.step_box.value(),
                smooth_win=11, smooth_poly=2,
                slip_mask_path=self.slip_path, slip_only=True
            )
            if prof:
                groups = clamp_groups_to_slip(prof, groups)
        except Exception:
            pass

        # save
        js = {"line": self.line_combo.currentText(), "groups": groups}
        try:
            with open(self._groups_json_path(), "w", encoding="utf-8") as f:
                json.dump(js, f, ensure_ascii=False, indent=2)
            self._log(f"[✓] Saved groups: {self._groups_json_path()}")
        except Exception as e:
            self._log(f"[!] Save groups failed: {e}")

        # re-render
        self._render_current_safe()
        # --- UPDATE: build bounds + đảm bảo scale + vẽ guides ---
        try:
            line_id = self._line_id_current()

            # 7.1) Thu groups (từ table bạn vừa confirm) → bounds [x0, x1, ...]
            groups = self._read_group_table()  # hoặc đọc lại từ JSON nếu bạn lưu ở trên
            bounds_set = set()
            for g in groups:
                try:
                    s = float(g.get("start", "nan"))
                    e = float(g.get("end", "nan"))
                    if not math.isnan(s) and not math.isnan(e):
                        if e < s: s, e = e, s
                        # clamp vào đoạn [0, _sec_len_m] nếu đã biết
                        if self._sec_len_m:
                            s = max(0.0, min(self._sec_len_m, s))
                            e = max(0.0, min(self._sec_len_m, e))
                        bounds_set.add(s);
                        bounds_set.add(e)
                except Exception:
                    pass
            bounds_m = sorted(bounds_set)
            if bounds_m:
                self._group_bounds[line_id] = bounds_m

            # 7.2) Đảm bảo có px_per_m
            if self._px_per_m is None and getattr(self, "_img_ground", None) and self._sec_len_m:
                W = self._img_ground.pixmap().width()
                if W and self._sec_len_m > 0:
                    self._px_per_m = float(W) / float(self._sec_len_m)

            # 7.3) Vẽ guides (gọi sau khi render)
            # self._draw_group_guides_for_current_line()
            self._ok("[UI3] Groups confirmed and guides updated.")
        except Exception as e:
            self._warn(f"[UI3] Confirm Groups: cannot update guides ({e})")

    # -------------------- Events --------------------
    # def _on_select_line(self, idx: int) -> None:
    #     if not self._sections:
    #         return
    #     self._current_idx = max(0, min(idx, len(self._sections) - 1))
    #     self._log(f"Selected line #{self._current_idx+1}.")
    #     self._draw_placeholder()
    #
    # def _on_auto_generate(self) -> None:
    #
    #     # TODO: gắn thuật toán auto-generate group/curve tại đây
    #     self._info("Auto generate curve – not implemented yet.")
    #
    # def _on_save_json(self) -> None:
    #
    #     # TODO: serialize kết quả đường cong đang hiển thị ra JSON
    #     self._info("Save curve JSON – not implemented yet.")

    # -------------------- Status helpers --------------------
    def _append_status(self, text: str) -> None:
        self.status.append(text)

    def _ok(self, msg: str) -> None:
        self._append_status(f"✅ {msg}")

    def _info(self, msg: str) -> None:
        self._append_status(f"[UI3] INFO: {msg}")

    def _warn(self, msg: str) -> None:
        self._append_status(f"⚠️ {msg}")

    def _err(self, msg: str) -> None:
        self._append_status(f"❌ {msg}")

    def _log(self, msg: str) -> None:
        self._append_status(msg)
