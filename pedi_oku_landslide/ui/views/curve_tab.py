# --- ADD/UPDATE imports ở đầu file ---
import math
import os, json
from typing import Optional, Dict, Any, List
import os
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor, QPainterPath
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QScrollArea, QFrame, QTextEdit, QComboBox, QDoubleSpinBox, QSpinBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy,
    QSplitter, QLineEdit, QMessageBox, QColorDialog, QAbstractSpinBox
)
from PyQt5.QtGui import QPixmap, QPixmapCache
from typing import Tuple, List, Dict, Optional
# backend UI3 đã có sẵn
from pedi_oku_landslide.pipeline.runners.ui3_backend import (
    auto_paths, list_lines, compute_profile, render_profile_png,
    clamp_groups_to_slip, auto_group_profile_by_criteria, auto_group_profile,
    estimate_slip_curve, fit_bezier_smooth_curve, evaluate_nurbs_curve
)
from pedi_oku_landslide.pipeline.runners.ui3_backend import rdp_indices_from_profile, rdp_points_from_profile
from PyQt5.QtGui import QPen, QColor
# ===================== ZOOMABLE GRAPHICS VIEW =====================
from PyQt5.QtWidgets import QGraphicsView, QToolBar, QAction
from typing import List, Tuple, Dict, Optional
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import (
    QGraphicsLineItem, QGraphicsRectItem, QGraphicsPathItem,
    QGraphicsEllipseItem, QGraphicsSimpleTextItem
)
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


class KeyboardOnlySpinBox(QSpinBox):
    """Disable wheel changes; allow keyboard input only."""
    def wheelEvent(self, event):
        event.ignore()


class KeyboardOnlyDoubleSpinBox(QDoubleSpinBox):
    """Disable wheel changes; allow keyboard input only."""
    def wheelEvent(self, event):
        event.ignore()


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
        self._splitter: Optional[QSplitter] = None
        self._left_min_w = 380
        self._left_default_w = 490
        self._pending_init_splitter = True

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
        self._curve_method_by_line: Dict[str, str] = {}
        self._active_prof: Optional[dict] = None
        self._active_groups: List[dict] = []
        self._active_base_curve: Optional[dict] = None
        self._active_curve: Optional[dict] = None
        self._curve_overlay_item: Optional[QGraphicsPathItem] = None
        self._cp_overlay_items: List[Any] = []
        self._nurbs_params_by_line: Dict[str, Dict[str, Any]] = {}
        self._group_table_updating: bool = False
        self._nurbs_updating_ui: bool = False
        self._nurbs_live_timer = QTimer(self)
        self._nurbs_live_timer.setSingleShot(True)
        self._nurbs_live_timer.setInterval(30)
        self._nurbs_live_timer.timeout.connect(self._on_nurbs_live_tick)

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

    @staticmethod
    def _normalize_curve_method(method: Optional[str]) -> str:
        m = str(method or "").strip().lower()
        if m == "nurbs":
            return "nurbs"
        return "bezier"

    @staticmethod
    def _curve_method_from_group_method(group_method: Optional[str]) -> str:
        gm = str(group_method or "").strip().lower()
        if gm == "traditional":
            return "nurbs"
        return "bezier"

    def _set_curve_method_for_line(self, line_id: str, curve_method: Optional[str]) -> str:
        cm = self._normalize_curve_method(curve_method)
        self._curve_method_by_line[line_id] = cm
        return cm

    def _get_curve_method_for_line(self, line_id: str) -> str:
        cm = self._curve_method_by_line.get(line_id, "")
        if cm:
            return self._normalize_curve_method(cm)

        js_path = self._groups_json_path_for(line_id)
        if os.path.exists(js_path):
            try:
                with open(js_path, "r", encoding="utf-8") as f:
                    js = json.load(f) or {}
                cm = str(js.get("curve_method", "")).strip().lower()
                if not cm:
                    cm = self._curve_method_from_group_method(js.get("group_method"))
                cm = self._set_curve_method_for_line(line_id, cm)
                return cm
            except Exception:
                pass

        return self._set_curve_method_for_line(line_id, "bezier")

    def _save_groups_to_ui(
        self,
        groups: list,
        prof: dict,
        line_id: str,
        log_text: Optional[str] = None,
        curve_method: Optional[str] = None,
        group_method: Optional[str] = None,
    ) -> None:
        cm = self._set_curve_method_for_line(line_id, curve_method or self._curve_method_by_line.get(line_id))
        try:
            js = {"line": self.line_combo.currentText(), "groups": groups, "curve_method": cm}
            if group_method:
                js["group_method"] = str(group_method)
            with open(self._groups_json_path(), "w", encoding="utf-8") as f:
                json.dump(js, f, ensure_ascii=False, indent=2)
            self._log(f"[✓] Saved group definition: {self._groups_json_path()}")
        except Exception as e:
            self._warn(f"[UI3] Cannot save groups JSON: {e}")

        if self.group_table is not None:
            self._group_table_updating = True
            try:
                self.group_table.setRowCount(0)
                for i, g in enumerate(groups, 1):
                    self.group_table.insertRow(self.group_table.rowCount())
                    self.group_table.setItem(i - 1, 0, QTableWidgetItem(str(g.get("id", f"G{i}"))))
                    self.group_table.setItem(i - 1, 1, QTableWidgetItem(f'{float(g.get("start", 0.0)):.3f}'))
                    self.group_table.setItem(i - 1, 2, QTableWidgetItem(f'{float(g.get("end", 0.0)):.3f}'))
                    self._set_color_cell(i - 1, str(g.get("color", "")).strip())
            finally:
                self._group_table_updating = False

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

    def _load_axes_meta(self, png_path: str) -> None:
        self._ax_top = None
        self._ax_bot = None
        try:
            meta_path = png_path.rsplit(".", 1)[0] + ".json"
            if not os.path.exists(meta_path):
                return
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
            top = meta.get("top") or {}
            bot = meta.get("bot") or {}
            if top:
                self._ax_top = top
            if bot:
                self._ax_bot = bot
        except Exception:
            self._ax_top = None
            self._ax_bot = None

    def _clear_curve_overlay(self) -> None:
        it = self._curve_overlay_item
        if it is not None:
            try:
                sc = it.scene()
                if sc is not None:
                    sc.removeItem(it)
            except Exception:
                pass
        self._curve_overlay_item = None
        self._clear_control_points_overlay()

    def _clear_control_points_overlay(self) -> None:
        for it in (self._cp_overlay_items or []):
            try:
                sc = it.scene()
                if sc is not None:
                    sc.removeItem(it)
            except Exception:
                pass
        self._cp_overlay_items = []

    def _chain_elev_to_scene_xy(self, chain_m: float, elev_m: float) -> Optional[Tuple[float, float]]:
        if self._img_ground is None or self._ax_top is None:
            return None
        try:
            ax = self._ax_top
            x_min = float(ax.get("x_min"))
            x_max = float(ax.get("x_max"))
            y_min = float(ax.get("y_min"))
            y_max = float(ax.get("y_max"))
            left_px = float(ax.get("left_px"))
            top_px = float(ax.get("top_px"))
            w_px = float(ax.get("width_px"))
            h_px = float(ax.get("height_px"))
            if not (x_max > x_min and y_max > y_min and w_px > 0 and h_px > 0):
                return None
            xr = (float(chain_m) - x_min) / (x_max - x_min)
            yr = (y_max - float(elev_m)) / (y_max - y_min)
            x_local = left_px + xr * w_px
            y_local = top_px + yr * h_px
            x_scene = float(self._img_ground.pos().x()) + x_local
            y_scene = float(self._img_ground.pos().y()) + y_local
            return x_scene, y_scene
        except Exception:
            return None

    def _draw_curve_overlay(self, chain_arr: np.ndarray, elev_arr: np.ndarray, color: str = "#bf00ff") -> None:
        # keep control-point markers; only refresh curve path here
        if self._curve_overlay_item is not None:
            try:
                sc = self._curve_overlay_item.scene()
                if sc is not None:
                    sc.removeItem(self._curve_overlay_item)
            except Exception:
                pass
            self._curve_overlay_item = None
        if self.scene is None:
            return
        ch = np.asarray(chain_arr, dtype=float)
        zz = np.asarray(elev_arr, dtype=float)
        m = np.isfinite(ch) & np.isfinite(zz)
        ch = ch[m]
        zz = zz[m]
        if ch.size < 2:
            return
        order = np.argsort(ch)
        ch = ch[order]
        zz = zz[order]

        path = QPainterPath()
        started = False
        for s, z in zip(ch, zz):
            pt = self._chain_elev_to_scene_xy(float(s), float(z))
            if pt is None:
                continue
            x, y = pt
            if not started:
                path.moveTo(x, y)
                started = True
            else:
                path.lineTo(x, y)
        if not started:
            return
        item = QGraphicsPathItem(path)
        pen = QPen(QColor(color))
        pen.setWidth(3)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setZValue(120.0)
        self.scene.addItem(item)
        self._curve_overlay_item = item

    def _draw_control_points_overlay(self, params: Optional[Dict[str, Any]] = None) -> None:
        self._clear_control_points_overlay()
        if self.scene is None:
            return
        p = params or self._collect_nurbs_params_from_ui()
        if not p:
            return
        cps = np.asarray(p.get("control_points", []), dtype=float)
        if cps.ndim != 2 or cps.shape[0] < 2:
            return

        for i, cp in enumerate(cps):
            pt = self._chain_elev_to_scene_xy(float(cp[0]), float(cp[1]))
            if pt is None:
                continue
            x, y = pt
            r = 10.0  # 2.5x from previous 4.0
            marker = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
            if i in (0, cps.shape[0] - 1):
                marker.setBrush(QColor("#ff4d4f"))
            else:
                marker.setBrush(QColor("#00a8ff"))
            marker.setPen(QPen(QColor("#ffffff")))
            marker.setZValue(130.0)
            self.scene.addItem(marker)
            self._cp_overlay_items.append(marker)

            lbl = QGraphicsSimpleTextItem(f"P{i}")
            lbl.setBrush(QColor("#111111"))
            fnt = lbl.font()
            psz = fnt.pointSizeF()
            if psz <= 0:
                psz = 10.0
            fnt.setPointSizeF(psz * 2.5)
            lbl.setFont(fnt)
            lbl.setPos(x + 20.0, y - 34.0)
            lbl.setZValue(131.0)
            self.scene.addItem(lbl)
            self._cp_overlay_items.append(lbl)

    @staticmethod
    def _profile_endpoints(prof: dict) -> Optional[Tuple[float, float, float, float]]:
        chain = np.asarray(prof.get("chain", []), dtype=float)
        elev = np.asarray(prof.get("elev_s", []), dtype=float)
        m = np.isfinite(chain) & np.isfinite(elev)
        chain = chain[m]
        elev = elev[m]
        if chain.size < 2:
            return None
        order = np.argsort(chain)
        chain = chain[order]
        elev = elev[order]
        return float(chain[0]), float(elev[0]), float(chain[-1]), float(elev[-1])

    def _build_default_nurbs_params(self, line_id: str, prof: dict, groups: list, base_curve: dict) -> Dict[str, Any]:
        ends = self._profile_endpoints(prof)
        if ends is None:
            return {"degree": 1, "control_points": [], "weights": []}
        s0, z0, s1, z1 = ends

        gs = []
        for g in (groups or []):
            try:
                st = float(g.get("start", g.get("start_chainage", np.nan)))
                en = float(g.get("end", g.get("end_chainage", np.nan)))
            except Exception:
                continue
            if not (np.isfinite(st) and np.isfinite(en)):
                continue
            if en < st:
                st, en = en, st
            gs.append((st, en))
        gs.sort(key=lambda x: (x[0], x[1]))
        # Default count rule: number of groups + 3
        n_ctrl = max(2, len(gs) + 3)

        # Default chainage: profile endpoints + group boundaries (end of each group).
        # This keeps control points tied to group boundaries.
        cp_chain = [float(s0)]
        cp_chain.extend([float(e) for _, e in gs])
        cp_chain.append(float(s1))
        if len(cp_chain) != n_ctrl:
            cp_chain = np.linspace(s0, s1, n_ctrl).tolist()

        xb = np.asarray((base_curve or {}).get("chain", []), dtype=float)
        zb = np.asarray((base_curve or {}).get("elev", []), dtype=float)
        mb = np.isfinite(xb) & np.isfinite(zb)
        xb = xb[mb]
        zb = zb[mb]
        if xb.size >= 2:
            cp_elev = np.interp(np.asarray(cp_chain, dtype=float), xb, zb).tolist()
        else:
            chain = np.asarray(prof.get("chain", []), dtype=float)
            elev = np.asarray(prof.get("elev_s", []), dtype=float)
            m = np.isfinite(chain) & np.isfinite(elev)
            chain = chain[m]
            elev = elev[m]
            if chain.size >= 2:
                cp_elev = np.interp(np.asarray(cp_chain, dtype=float), chain, elev).tolist()
            else:
                cp_elev = np.linspace(z0, z1, len(cp_chain)).tolist()

        # Keep default interior CP always below ground profile.
        gch = np.asarray(prof.get("chain", []), dtype=float)
        gz = np.asarray(prof.get("elev_s", []), dtype=float)
        mg = np.isfinite(gch) & np.isfinite(gz)
        gch = gch[mg]
        gz = gz[mg]
        clearance = 0.2
        if gch.size >= 2 and len(cp_elev) >= 3:
            g_at_cp = np.interp(np.asarray(cp_chain, dtype=float), gch, gz)
            cp_elev_arr = np.asarray(cp_elev, dtype=float)
            cp_elev_arr[1:-1] = np.minimum(cp_elev_arr[1:-1], g_at_cp[1:-1] - clearance)
            cp_elev = cp_elev_arr.tolist()

        # Endpoint lock to first/last vectors (profile endpoints).
        cp_elev[0] = z0
        cp_elev[-1] = z1
        cps = [[float(s), float(z)] for s, z in zip(cp_chain, cp_elev)]
        w = [1.0] * len(cps)
        deg = min(3, max(1, len(cps) - 1))
        params = {"degree": int(deg), "control_points": cps, "weights": w}
        self._nurbs_params_by_line[line_id] = params
        return params

    def _get_nurbs_params_for_line(self, line_id: str) -> Optional[Dict[str, Any]]:
        return self._nurbs_params_by_line.get(line_id)

    def _set_nurbs_params_for_line(self, line_id: str, params: Dict[str, Any]) -> None:
        self._nurbs_params_by_line[line_id] = params

    def _try_load_nurbs_params_file(self, line_id: str) -> Optional[Dict[str, Any]]:
        path = self._nurbs_json_path_for(line_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                js = json.load(f) or {}
            params = {
                "degree": int(js.get("degree", 3)),
                "control_points": js.get("control_points", []),
                "weights": js.get("weights", []),
            }
            cps = np.asarray(params.get("control_points", []), dtype=float)
            ws = np.asarray(params.get("weights", []), dtype=float)
            if cps.ndim != 2 or cps.shape[0] < 2:
                return None
            if ws.ndim != 1 or ws.size != cps.shape[0]:
                params["weights"] = np.ones(cps.shape[0], dtype=float).tolist()
            return params
        except Exception:
            return None

    def _sync_nurbs_panel_for_current_line(self, reset_defaults: bool = False) -> None:
        if self.line_combo is None or self.line_combo.count() == 0:
            return
        line_id = self._line_id_current()
        prof = self._active_prof
        if not prof:
            return
        groups = self._active_groups or []
        base = self._active_base_curve or {}

        params = None if reset_defaults else self._get_nurbs_params_for_line(line_id)
        if (not params) and (not reset_defaults):
            params = self._try_load_nurbs_params_file(line_id)
        if not params:
            params = self._build_default_nurbs_params(line_id, prof, groups, base)

        cps = params.get("control_points", []) or []
        ww = params.get("weights", []) or []
        deg = int(params.get("degree", 3))
        n_ctrl = max(2, len(cps))
        deg = max(1, min(deg, n_ctrl - 1))
        params["degree"] = deg
        if len(ww) != n_ctrl:
            ww = [1.0] * n_ctrl
            params["weights"] = ww
        self._set_nurbs_params_for_line(line_id, params)

        self._nurbs_updating_ui = True
        try:
            self.nurbs_cp_spin.setValue(n_ctrl)
            self.nurbs_deg_spin.setMaximum(max(1, n_ctrl - 1))
            self.nurbs_deg_spin.setValue(deg)
            self._populate_nurbs_table(params)
        finally:
            self._nurbs_updating_ui = False
        self._draw_control_points_overlay(params)

    def _populate_nurbs_table(self, params: Dict[str, Any]) -> None:
        cps = params.get("control_points", []) or []
        ww = params.get("weights", []) or []
        self.nurbs_table.setRowCount(0)
        for i, cp in enumerate(cps):
            r = self.nurbs_table.rowCount()
            self.nurbs_table.insertRow(r)
            item = QTableWidgetItem(f"P{i}")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.nurbs_table.setItem(r, 0, item)

            sbox = KeyboardOnlyDoubleSpinBox()
            sbox.setDecimals(3)
            sbox.setRange(-1e6, 1e6)
            sbox.setSingleStep(0.1)
            sbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
            sbox.setValue(float(cp[0]))
            sbox.valueChanged.connect(lambda _v, _r=r: self._on_nurbs_table_changed(_r))
            self.nurbs_table.setCellWidget(r, 1, sbox)

            zbox = KeyboardOnlyDoubleSpinBox()
            zbox.setDecimals(3)
            zbox.setRange(-1e6, 1e6)
            zbox.setSingleStep(0.1)
            zbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
            zbox.setValue(float(cp[1]))
            zbox.valueChanged.connect(lambda _v, _r=r: self._on_nurbs_table_changed(_r))
            self.nurbs_table.setCellWidget(r, 2, zbox)

            wbox = KeyboardOnlyDoubleSpinBox()
            wbox.setDecimals(3)
            wbox.setRange(0.001, 1e6)
            wbox.setSingleStep(0.1)
            wbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
            wbox.setValue(float(ww[i] if i < len(ww) else 1.0))
            wbox.valueChanged.connect(lambda _v, _r=r: self._on_nurbs_table_changed(_r))
            self.nurbs_table.setCellWidget(r, 3, wbox)

        self._enforce_nurbs_endpoint_lock()

    def _collect_nurbs_params_from_ui(self) -> Optional[Dict[str, Any]]:
        rc = self.nurbs_table.rowCount()
        if rc < 2:
            return None
        cps = []
        ws = []
        for r in range(rc):
            sbox = self.nurbs_table.cellWidget(r, 1)
            zbox = self.nurbs_table.cellWidget(r, 2)
            wbox = self.nurbs_table.cellWidget(r, 3)
            if not isinstance(sbox, QDoubleSpinBox) or not isinstance(zbox, QDoubleSpinBox) or not isinstance(wbox, QDoubleSpinBox):
                return None
            cps.append([float(sbox.value()), float(zbox.value())])
            ws.append(float(max(0.001, wbox.value())))
        cps_arr = np.asarray(cps, dtype=float)
        order = np.argsort(cps_arr[:, 0])
        cps_arr = cps_arr[order]
        ws_arr = np.asarray(ws, dtype=float)[order]
        deg = int(self.nurbs_deg_spin.value())
        deg = max(1, min(deg, len(cps) - 1))
        return {
            "degree": deg,
            "control_points": cps_arr.tolist(),
            "weights": ws_arr.tolist(),
        }

    def _enforce_nurbs_endpoint_lock(self) -> None:
        prof = self._active_prof
        if not prof:
            return
        ends = self._profile_endpoints(prof)
        if ends is None:
            return
        s0, z0, s1, z1 = ends
        rc = self.nurbs_table.rowCount()
        if rc < 2:
            return
        for row, s_val, z_val in ((0, s0, z0), (rc - 1, s1, z1)):
            sbox = self.nurbs_table.cellWidget(row, 1)
            zbox = self.nurbs_table.cellWidget(row, 2)
            if isinstance(sbox, QDoubleSpinBox):
                sbox.blockSignals(True)
                sbox.setValue(float(s_val))
                sbox.setEnabled(False)
                sbox.blockSignals(False)
            if isinstance(zbox, QDoubleSpinBox):
                zbox.blockSignals(True)
                zbox.setValue(float(z_val))
                zbox.setEnabled(False)
                zbox.blockSignals(False)

    def _resize_nurbs_control_points(self, new_count: int) -> None:
        line_id = self._line_id_current()
        params = self._get_nurbs_params_for_line(line_id)
        prof = self._active_prof
        if params is None or prof is None:
            return
        ends = self._profile_endpoints(prof)
        if ends is None:
            return
        s0, z0, s1, z1 = ends
        cps = np.asarray(params.get("control_points", []), dtype=float)
        ws = np.asarray(params.get("weights", []), dtype=float)
        if cps.ndim != 2 or cps.shape[0] < 2:
            params = self._build_default_nurbs_params(line_id, prof, self._active_groups, self._active_base_curve or {})
            cps = np.asarray(params.get("control_points", []), dtype=float)
            ws = np.asarray(params.get("weights", []), dtype=float)
        old_s = cps[:, 0]
        old_z = cps[:, 1]
        new_s = np.linspace(s0, s1, int(max(2, new_count)))
        new_z = np.interp(new_s, old_s, old_z)
        new_w = np.interp(new_s, old_s, ws if ws.size == old_s.size else np.ones_like(old_s))
        new_z[0] = z0
        new_z[-1] = z1
        new_params = {
            "degree": int(min(int(params.get("degree", 3)), len(new_s) - 1)),
            "control_points": np.vstack([new_s, new_z]).T.tolist(),
            "weights": np.where(np.isfinite(new_w) & (new_w > 0), new_w, 1.0).tolist(),
        }
        self._set_nurbs_params_for_line(line_id, new_params)
        self._sync_nurbs_panel_for_current_line(reset_defaults=False)
        self._schedule_nurbs_live_update()

    def _schedule_nurbs_live_update(self) -> None:
        if self._nurbs_updating_ui:
            return
        self._nurbs_live_timer.start()

    def _on_nurbs_live_tick(self) -> None:
        line_id = self._line_id_current()
        params = self._collect_nurbs_params_from_ui()
        if not params:
            return
        self._set_nurbs_params_for_line(line_id, params)
        self._draw_control_points_overlay(params)
        curve_method = self._get_curve_method_for_line(line_id)
        if curve_method != "nurbs":
            return
        curve = self._compute_nurbs_curve_from_params(params)
        if curve is None:
            return
        self._active_curve = curve
        self._draw_curve_overlay(np.asarray(curve["chain"], dtype=float), np.asarray(curve["elev"], dtype=float))

    def _compute_nurbs_curve_from_params(self, params: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        prof = self._active_prof
        if not prof:
            return None
        p = dict(params or {})
        cps = np.asarray(p.get("control_points", []), dtype=float)
        ww = np.asarray(p.get("weights", []), dtype=float)
        if cps.ndim != 2 or cps.shape[0] < 2:
            return None
        if ww.ndim != 1 or ww.size != cps.shape[0]:
            ww = np.ones(cps.shape[0], dtype=float)

        ends = self._profile_endpoints(prof)
        if ends is None:
            return None
        s0, z0, s1, z1 = ends
        cps = cps.copy()
        cps[0, 0], cps[0, 1] = s0, z0
        cps[-1, 0], cps[-1, 1] = s1, z1

        deg = int(max(1, min(int(p.get("degree", 3)), cps.shape[0] - 1)))
        ch = np.asarray(prof.get("chain", []), dtype=float)
        n_samples = int(max(120, np.count_nonzero(np.isfinite(ch)) * 2))
        out = evaluate_nurbs_curve(
            chain_ctrl=cps[:, 0],
            elev_ctrl=cps[:, 1],
            weights=ww,
            degree=deg,
            n_samples=n_samples,
        )
        sx = np.asarray(out.get("chain", []), dtype=float)
        sz = np.asarray(out.get("elev", []), dtype=float)
        m = np.isfinite(sx) & np.isfinite(sz)
        sx = sx[m]
        sz = sz[m]
        if sx.size < 2:
            return None
        return {"chain": sx, "elev": sz}

    def _on_nurbs_cp_spin_changed(self, val: int) -> None:
        if self._nurbs_updating_ui:
            return
        self.nurbs_deg_spin.setMaximum(max(1, int(val) - 1))
        if self.nurbs_deg_spin.value() > self.nurbs_deg_spin.maximum():
            self.nurbs_deg_spin.setValue(self.nurbs_deg_spin.maximum())
        self._resize_nurbs_control_points(int(val))

    def _on_nurbs_deg_spin_changed(self, val: int) -> None:
        if self._nurbs_updating_ui:
            return
        max_deg = max(1, self.nurbs_cp_spin.value() - 1)
        if val > max_deg:
            self._nurbs_updating_ui = True
            try:
                self.nurbs_deg_spin.setValue(max_deg)
            finally:
                self._nurbs_updating_ui = False
        self._schedule_nurbs_live_update()

    def _on_nurbs_table_changed(self, _row: int) -> None:
        self._enforce_nurbs_endpoint_lock()
        self._schedule_nurbs_live_update()

    def _on_nurbs_reset_defaults(self) -> None:
        if not self._active_prof:
            self._warn("[UI3] Render/Draw curve first to initialize NURBS.")
            return
        self._sync_nurbs_panel_for_current_line(reset_defaults=True)
        self._schedule_nurbs_live_update()

    def _nurbs_png_path_for(self, line_id: str) -> str:
        return os.path.join(self._preview_dir(), f"profile_{line_id}_nurbs.png")

    def _nurbs_json_path_for(self, line_id: str) -> str:
        return os.path.join(self._preview_dir(), f"profile_{line_id}_nurbs.json")

    def _on_nurbs_save(self) -> None:
        if not self._active_prof:
            self._warn("[UI3] No active profile to save NURBS.")
            return
        line_id = self._line_id_current()
        params = self._collect_nurbs_params_from_ui()
        if not params:
            self._warn("[UI3] Invalid NURBS parameters.")
            return
        curve = self._compute_nurbs_curve_from_params(params)
        if curve is None:
            self._warn("[UI3] Cannot evaluate NURBS curve.")
            return
        out_png = self._nurbs_png_path_for(line_id)
        msg, path = render_profile_png(
            self._active_prof, out_png,
            y_min=None, y_max=None,
            x_min=None, x_max=None,
            vec_scale=self.vscale.value(),
            vec_width=self.vwidth.value(),
            head_len=6.0, head_w=4.0,
            highlight_theta=None,
            group_ranges=self._active_groups if self._active_groups else None,
            ungrouped_color=self._get_ungrouped_color(),
            overlay_curves=[(curve["chain"], curve["elev"], "#bf00ff", "Slip curve")]
        )
        self._log(msg)
        if path and os.path.exists(path):
            payload = {
                "line_id": line_id,
                "curve_method": "nurbs",
                "degree": int(params.get("degree", 3)),
                "control_points": params.get("control_points", []),
                "weights": params.get("weights", []),
                "curve": {
                    "chain": np.asarray(curve["chain"], dtype=float).tolist(),
                    "elev": np.asarray(curve["elev"], dtype=float).tolist(),
                },
            }
            jpath = self._nurbs_json_path_for(line_id)
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._ok(f"[UI3] Saved NURBS: {path}")
            self._log(f"[UI3] Saved NURBS params: {jpath}")

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

            # 3) Chọn thuật toán Auto-group
            dlg = QMessageBox(self)
            dlg.setIcon(QMessageBox.Question)
            dlg.setWindowTitle("Auto Group Method")
            dlg.setText("Select grouping method:")
            dlg.setStandardButtons(QMessageBox.NoButton)
            btn_traditional = dlg.addButton("Traditional", QMessageBox.ActionRole)
            btn_new = dlg.addButton("New", QMessageBox.ActionRole)
            btn_cancel = dlg.addButton("Cancel", QMessageBox.RejectRole)
            dlg.exec_()

            clicked = dlg.clickedButton()
            clicked_text = clicked.text().strip() if clicked is not None else ""
            if clicked_text == "Traditional":
                # Theo yêu cầu: Traditional -> hàm mới
                self._log("[UI3] Auto Group method: Traditional -> new criteria (curve: NURBS)")
                groups = auto_group_profile_by_criteria(prof)
                group_method = "traditional"
            elif clicked_text == "New":
                # Theo yêu cầu: New -> hàm/tiêu chí cũ
                self._log("[UI3] Auto Group method: New -> legacy criteria (curve: Bezier)")
                groups = auto_group_profile(prof)
                group_method = "new"
            else:
                if clicked == btn_cancel:
                    self._log("[UI3] Auto Group canceled.")
                return

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
                curve_method=self._curve_method_from_group_method(group_method),
                group_method=group_method,
            )
            # Re-render ngay để vẽ vector và tô màu theo group vừa tạo.
            self._render_current_safe()
        except Exception as e:
            self._err(f"[UI3] Auto Group error: {e}")

    def _on_draw_curve(self) -> None:
        """Tính và vẽ đường cong (overlay) vào PNG preview hiện tại."""
        try:
            line_id = self._line_id_current()
            curve_method = self._get_curve_method_for_line(line_id)
            self._log(f"[UI3] Curve method for '{line_id}': {curve_method.upper()}")

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
                if curve_method == "nurbs":
                    groups = auto_group_profile_by_criteria(prof)
                else:
                    groups = auto_group_profile(prof)
                groups = clamp_groups_to_slip(prof, groups)
                if not groups:
                    self._warn("[UI3] Auto grouping produced no segments within slip zone.");
                    return
                self._save_groups_to_ui(
                    groups,
                    prof,
                    line_id,
                    log_text=f"[UI3] Auto Group (implicit) for '{line_id}': {len(groups)} groups.",
                    curve_method=curve_method,
                )
            else:
                groups = clamp_groups_to_slip(prof, groups)
                if not groups:
                    self._warn("[UI3] No groups within slip zone.");
                    return

            # 3) Base curve để lấy mục tiêu mặc định cho NURBS/Bezier
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

            self._active_prof = prof
            self._active_groups = groups
            self._active_base_curve = {"chain": x_base, "elev": z_base}
            self._sync_nurbs_panel_for_current_line(reset_defaults=False)

            # 4) Fit curve theo method
            curve = {"chain": x_base, "elev": z_base}  # fallback mặc định

            def _fit_bezier_curve() -> Optional[dict]:
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
                        self._log(f"[UI3] Bezier slip curve OK: n={xb.size}")
                        return {"chain": xb, "elev": zb}
                except Exception as e:
                    self._warn(f"[UI3] Bezier fit failed, using base curve. ({e})")
                return None

            if curve_method == "nurbs":
                params = self._collect_nurbs_params_from_ui()
                if params:
                    nurbs_curve = self._compute_nurbs_curve_from_params(params)
                    if nurbs_curve is not None:
                        curve = nurbs_curve
                        self._log(f"[UI3] NURBS slip curve OK: n={len(nurbs_curve['chain'])}")
                    else:
                        self._warn("[UI3] NURBS fit failed; fallback to Bezier.")
                        bez_curve = _fit_bezier_curve()
                        if bez_curve is not None:
                            curve = bez_curve
                else:
                    self._warn("[UI3] Invalid NURBS params; fallback to Bezier.")
                    bez_curve = _fit_bezier_curve()
                    if bez_curve is not None:
                        curve = bez_curve
            else:
                bez_curve = _fit_bezier_curve()
                if bez_curve is not None:
                    curve = bez_curve
                else:
                    self._warn("[UI3] Bezier returned too few points; using base curve.")

            # 5) Re-render base PNG (no curve baked-in), then draw overlay in scene
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
                ungrouped_color=self._get_ungrouped_color()
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
            self._load_axes_meta(path)

            self._active_curve = {
                "chain": np.asarray(curve["chain"], dtype=float),
                "elev": np.asarray(curve["elev"], dtype=float),
            }
            self._draw_curve_overlay(self._active_curve["chain"], self._active_curve["elev"])

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
        self._splitter = splitter
        splitter.splitterMoved.connect(lambda *_: self._enforce_left_pane_bounds())
        root.addWidget(splitter)

        # ===== LEFT: controls (KHÔNG dùng QScrollArea nữa) =====
        left_container = QWidget()
        left_container.setMinimumWidth(self._left_min_w)
        left = QVBoxLayout(left_container)
        left.setContentsMargins(6, 6, 6, 6)
        left.setSpacing(8)
        splitter.addWidget(left_container)

        # Project info – giống Section tab
        box_proj = QGroupBox("Project")
        lp = QVBoxLayout(box_proj)
        proj_input_h = 30
        fm = self.fontMetrics()
        proj_label_w = max(
            fm.horizontalAdvance("Name:"),
            fm.horizontalAdvance("Run label:")
        ) + 15

        def _fit_proj_label(text: str) -> QLabel:
            lb = QLabel(text)
            lb.setFixedWidth(proj_label_w)
            return lb

        row_proj = QHBoxLayout()
        row_proj.addWidget(_fit_proj_label("Name:"))
        self.edit_project = QLineEdit()
        self.edit_project.setPlaceholderText("—")
        self.edit_project.setReadOnly(True)
        self.edit_project.setFixedHeight(proj_input_h)
        self.edit_project.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_proj.addWidget(self.edit_project, 1)
        row_proj.addSpacing(6)
        row_proj.addWidget(_fit_proj_label("Run label:"))
        self.edit_runlabel = QLineEdit()
        self.edit_runlabel.setPlaceholderText("—")
        self.edit_runlabel.setReadOnly(True)
        self.edit_runlabel.setFixedHeight(proj_input_h)
        self.edit_runlabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_proj.addWidget(self.edit_runlabel, 1)
        lp.addLayout(row_proj)

        left.addWidget(box_proj)

        # Sections + Advanced display
        box_sel = QGroupBox("Sections Display")
        lsd = QVBoxLayout(box_sel)
        ls = QHBoxLayout()
        self.line_combo = QComboBox()
        self.line_combo.currentIndexChanged.connect(self._on_line_changed)
        btn_render = QPushButton("Render Section")
        btn_render.clicked.connect(self._render_current_safe)
        ls.addWidget(self.line_combo)
        ls.addWidget(btn_render)
        lsd.addLayout(ls)

        # Advanced controls
        la = QHBoxLayout()
        la.setSpacing(6)

        def _fit_adv_label(text: str) -> QLabel:
            lb = QLabel(text)
            min_w = lb.fontMetrics().horizontalAdvance(text) + 8
            lb.setFixedWidth(min_w)
            return lb

        lbl_step = _fit_adv_label("Step (m):")
        la.addWidget(lbl_step)
        self.step_box = QDoubleSpinBox()
        self.step_box.setDecimals(2)
        self.step_box.setValue(0.20)
        self.step_box.setMaximum(1e6)
        self.step_box.setMinimumWidth(56)
        self.step_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        la.addWidget(self.step_box, 1)
        lbl_scale = _fit_adv_label("Scale:")
        la.addWidget(lbl_scale)
        self.vscale = QDoubleSpinBox()
        self.vscale.setDecimals(3)
        self.vscale.setValue(0.1)
        self.vscale.setMaximum(1e6)
        self.vscale.setMinimumWidth(56)
        self.vscale.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        la.addWidget(self.vscale, 1)
        lbl_width = _fit_adv_label("Width:")
        la.addWidget(lbl_width)
        self.vwidth = QDoubleSpinBox()
        self.vwidth.setDecimals(4)
        self.vwidth.setValue(0.0015)
        self.vwidth.setMaximum(1.0)
        self.vwidth.setMinimumWidth(56)
        self.vwidth.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        la.addWidget(self.vwidth, 1)
        lsd.addLayout(la)

        left.addWidget(box_sel)

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
        self.group_table.itemChanged.connect(self._on_group_table_item_changed)
        self.group_table.setMinimumHeight(220)
        lg.addWidget(self.group_table)

        rowg = QHBoxLayout()
        self.btn_add_g = QPushButton("Add")
        self.btn_add_g.clicked.connect(self._on_add_group)
        self.btn_del_g = QPushButton("Delete")
        self.btn_del_g.clicked.connect(self._on_delete_group)
        self.btn_draw_curve = QPushButton("Draw Curve")
        self.btn_draw_curve.clicked.connect(self._on_draw_curve)
        self.btn_auto_group = QPushButton("Auto Group")
        self.btn_auto_group.clicked.connect(self._on_auto_group)

        for btn in (self.btn_auto_group, self.btn_add_g, self.btn_del_g, self.btn_draw_curve):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            rowg.addWidget(btn, 1)
        lg.addLayout(rowg)
        left.addWidget(box_grp, 1)

        # NURBS controls
        box_nurbs = QGroupBox("NURBS")
        ln = QVBoxLayout(box_nurbs)
        ln.setContentsMargins(8, 8, 8, 8)
        ln.setSpacing(6)

        row_cfg = QHBoxLayout()
        row_cfg.addWidget(QLabel("Control points:"))
        self.nurbs_cp_spin = KeyboardOnlySpinBox()
        self.nurbs_cp_spin.setRange(2, 20)
        self.nurbs_cp_spin.setValue(4)
        self.nurbs_cp_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        row_cfg.addWidget(self.nurbs_cp_spin)
        row_cfg.addSpacing(8)
        row_cfg.addWidget(QLabel("Degree:"))
        self.nurbs_deg_spin = KeyboardOnlySpinBox()
        self.nurbs_deg_spin.setRange(1, 10)
        self.nurbs_deg_spin.setValue(3)
        self.nurbs_deg_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        row_cfg.addWidget(self.nurbs_deg_spin)
        ln.addLayout(row_cfg)

        self.nurbs_table = QTableWidget(0, 4)
        self.nurbs_table.setHorizontalHeaderLabels(["CP", "Chainage (m)", "Elev (m)", "Weight"])
        self.nurbs_table.verticalHeader().setVisible(False)
        self.nurbs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.nurbs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.nurbs_table.setMinimumHeight(120)
        self.nurbs_table.setMaximumHeight(170)
        ln.addWidget(self.nurbs_table)

        row_nurbs_btn = QHBoxLayout()
        self.btn_nurbs_reset = QPushButton("Reset NURBS")
        self.btn_nurbs_save = QPushButton("Save")
        row_nurbs_btn.addWidget(self.btn_nurbs_reset, 1)
        row_nurbs_btn.addWidget(self.btn_nurbs_save, 1)
        ln.addLayout(row_nurbs_btn)

        self.nurbs_cp_spin.valueChanged.connect(self._on_nurbs_cp_spin_changed)
        self.nurbs_deg_spin.valueChanged.connect(self._on_nurbs_deg_spin_changed)
        self.btn_nurbs_reset.clicked.connect(self._on_nurbs_reset_defaults)
        self.btn_nurbs_save.clicked.connect(self._on_nurbs_save)

        left.addWidget(box_nurbs, 0)

        # Status
        box_st = QGroupBox("Status")
        ls = QVBoxLayout(box_st)
        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ls.addWidget(self.status)
        left.addWidget(box_st, 1)

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
        splitter.setSizes([self._left_default_w, 900])

        self._apply_button_style()

    def _left_max_w(self) -> int:
        base_w = self.width()
        if self._splitter is not None and self._splitter.width() > 0:
            base_w = self._splitter.width()
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
        self._clear_curve_overlay()
        self._active_prof = None
        self._active_groups = []
        self._active_base_curve = None
        self._active_curve = None
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
        self._clear_curve_overlay()
        self._active_prof = None
        self._active_groups = []
        self._active_base_curve = None
        self._active_curve = None
        self._populate_group_table_for_current_line()  # <-- đổi tên ở đây

    # def _groups_json_path(self) -> str:
    #     line_label = self.line_combo.currentText().strip() or f"line_{self.line_combo.currentIndex() + 1:03d}"
    #     safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in line_label)
    #     gdir = os.path.join(self.base_dir, "output", "UI3", "groups");
    #     os.makedirs(gdir, exist_ok=True)
    #     return os.path.join(gdir, f"{safe}.json")

    def _populate_group_table_for_current_line(self) -> None:
        path = self._groups_json_path()
        self._group_table_updating = True
        try:
            self.group_table.setRowCount(0)
            loaded = 0
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        js = json.load(f)
                    cm = self._normalize_curve_method(js.get("curve_method"))
                    self._set_curve_method_for_line(self._line_id_current(), cm)
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
        finally:
            self._group_table_updating = False

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

    def _vectors_dir(self) -> str:
        path = os.path.join(self._ui3_run_dir(), "vectors")
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

    def _vectors_json_path_for(self, line_id: str) -> str:
        return os.path.join(self._vectors_dir(), f"{line_id}.json")

    @staticmethod
    def _group_for_chainage(groups: List[dict], chainage: float) -> Tuple[Optional[str], Optional[str]]:
        for g in (groups or []):
            try:
                s = float(g.get("start", g.get("start_chainage", 0.0)))
                e = float(g.get("end", g.get("end_chainage", 0.0)))
                if e < s:
                    s, e = e, s
                if s <= chainage <= e:
                    gid = str(g.get("id", "")) or None
                    color = g.get("color", None)
                    return gid, color
            except Exception:
                continue
        return None, None

    def _save_vectors_json_for_line(self, line_id: str, prof: dict, groups: Optional[List[dict]]) -> Optional[str]:
        def _to_float(arr, i: int) -> Optional[float]:
            if arr is None:
                return None
            try:
                v = float(arr[i])
            except Exception:
                return None
            return v if np.isfinite(v) else None

        chain = np.asarray(prof.get("chain", []), dtype=float)
        n = int(chain.size)
        if n == 0:
            return None

        x = np.asarray(prof.get("x", []), dtype=float) if prof.get("x", None) is not None else None
        y = np.asarray(prof.get("y", []), dtype=float) if prof.get("y", None) is not None else None
        elev = np.asarray(prof.get("elev", []), dtype=float) if prof.get("elev", None) is not None else None
        elev_s = np.asarray(prof.get("elev_s", []), dtype=float) if prof.get("elev_s", None) is not None else None
        dx = np.asarray(prof.get("dx", []), dtype=float) if prof.get("dx", None) is not None else None
        dy = np.asarray(prof.get("dy", []), dtype=float) if prof.get("dy", None) is not None else None
        dz = np.asarray(prof.get("dz", []), dtype=float) if prof.get("dz", None) is not None else None
        d_para = np.asarray(prof.get("d_para", []), dtype=float) if prof.get("d_para", None) is not None else None
        theta = np.asarray(prof.get("theta", []), dtype=float) if prof.get("theta", None) is not None else None
        slip_mask = np.asarray(prof.get("slip_mask", [])) if prof.get("slip_mask", None) is not None else None

        rows: List[dict] = []
        for i in range(n):
            ch = _to_float(chain, i)
            if ch is None:
                continue
            gid, gcolor = self._group_for_chainage(groups or [], ch)
            in_slip = None
            if slip_mask is not None and i < slip_mask.size:
                try:
                    in_slip = bool(slip_mask[i])
                except Exception:
                    in_slip = None
            rows.append({
                "index": i,
                "chain_m": ch,
                "x": _to_float(x, i),
                "y": _to_float(y, i),
                "elev_raw_m": _to_float(elev, i),
                "elev_s_m": _to_float(elev_s, i),
                "dx_m": _to_float(dx, i),
                "dy_m": _to_float(dy, i),
                "dz_m": _to_float(dz, i),
                "d_para_m": _to_float(d_para, i),
                "theta_deg": _to_float(theta, i),
                "in_slip_zone": in_slip,
                "group_id": gid,
                "group_color": gcolor,
            })

        payload = {
            "line_id": line_id,
            "count": len(rows),
            "groups": groups or [],
            "vectors": rows,
        }
        out_json = self._vectors_json_path_for(line_id)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out_json

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
            group_ranges=groups if groups else None,
            ungrouped_color=self._get_ungrouped_color()
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
        self._clear_curve_overlay()
        self._load_axes_meta(path)
        self._active_prof = prof
        self._active_groups = groups if groups else []
        self._active_base_curve = None
        self._active_curve = None

        # Fit lần đầu
        if getattr(self, "_first_show", True):
            self.view.fit_to_scene()
            self._first_show = False

        try:
            vec_json = self._save_vectors_json_for_line(line_id, prof, groups if groups else [])
            if vec_json:
                self._log(f"[UI3] Saved vectors JSON: {vec_json}")
        except Exception as e:
            self._warn(f"[UI3] Cannot save vectors JSON: {e}")

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

    def _sync_nurbs_defaults_from_group_table(self) -> None:
        """Rebuild default NURBS control points when Group table changes."""
        if self._group_table_updating:
            return
        if not self._active_prof:
            return
        try:
            line_id = self._line_id_current()
            groups = self._read_groups_from_table()
            if groups:
                groups = clamp_groups_to_slip(self._active_prof, groups)
            self._active_groups = groups or []
            params = self._build_default_nurbs_params(
                line_id=line_id,
                prof=self._active_prof,
                groups=self._active_groups,
                base_curve=self._active_base_curve or {},
            )
            self._set_nurbs_params_for_line(line_id, params)
            self._sync_nurbs_panel_for_current_line(reset_defaults=False)
            self._schedule_nurbs_live_update()
        except Exception as e:
            self._warn(f"[UI3] Cannot sync NURBS from groups: {e}")

    def _on_group_table_item_changed(self, _item) -> None:
        self._sync_nurbs_defaults_from_group_table()

    def _on_add_group(self):
        r = self._find_ungrouped_row()
        if r is None:
            r = self.group_table.rowCount()
        self.group_table.insertRow(r)
        n_groups = len(self._read_groups_from_table()) + 1
        self.group_table.setItem(r, 0, QTableWidgetItem(f"G{n_groups}"))
        self._sync_nurbs_defaults_from_group_table()

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
        self._sync_nurbs_defaults_from_group_table()


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
        self._set_color_cell(r, "#bbbbbb")
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
        current = self._get_color_cell_value(row)
        initial = QColor(current) if current else QColor(255, 255, 255)
        color = QColorDialog.getColor(initial, self, "Select group color")
        if color.isValid():
            self._set_color_cell(row, color.name())
            # Re-render to reflect new group colors on vectors
            self._render_current_safe()

    def _get_ungrouped_color(self) -> str:
        r = self._find_ungrouped_row()
        if r is None:
            return "#bbbbbb"
        c = self._get_color_cell_value(r)
        return c or "#bbbbbb"

    def _load_groups_for_current_line(self):
        """Ưu tiên đọc từ bảng (phản ánh chỉnh sửa/delete mới nhất); nếu trống thì đọc JSON."""
        line_id = self._line_id_current()
        table_groups = self._read_groups_from_table()
        if table_groups:
            return table_groups

        js_path = self._groups_json_path()
        if os.path.exists(js_path):
            try:
                import json
                with open(js_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cm = self._normalize_curve_method(data.get("curve_method"))
                self._set_curve_method_for_line(line_id, cm)
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
                groups = auto_group_profile_by_criteria(prof)

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
        line_id = self._line_id_current()
        curve_method = self._get_curve_method_for_line(line_id)
        js = {"line": self.line_combo.currentText(), "groups": groups, "curve_method": curve_method}
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
