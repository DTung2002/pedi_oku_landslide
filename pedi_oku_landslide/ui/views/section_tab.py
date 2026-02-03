# pedi_oku_landslide/ui/views/section_tab.py
import os, math
from typing import Optional, Tuple, List

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, xy as rio_xy
from shapely.geometry import LineString
from PyQt5.QtWidgets import QPlainTextEdit
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QEvent
from PyQt5.QtGui import QPen, QBrush, QColor, QImage, QPixmap, QPainterPath, QPainter, QFont, QFontMetrics
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QTextEdit, QSplitter, QSlider,
    QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsPathItem,
    QGraphicsSimpleTextItem, QGraphicsView, QMessageBox, QDialog, QDialogButtonBox,
    QSpinBox, QDoubleSpinBox, QGridLayout,
)

from .ui.ui1_viewer import UI1Viewer

# --- tiny layout helpers (alias) ---
def HBox():
    return QHBoxLayout()

# ---------- helpers ----------

def _file_exists(p: str) -> bool:
    return bool(p) and os.path.isfile(p)


def _np_to_qimage_gray(img8: np.ndarray) -> QImage:
    """img8: HxW uint8"""
    h, w = img8.shape
    qimg = QImage(img8.data, w, h, w, QImage.Format_Grayscale8)
    return qimg.copy()


def _np_to_qimage_rgba(rgba: np.ndarray) -> QImage:
    """rgba: HxWx4 uint8 (RGBA) -> ARGB32 for Qt"""
    h, w, _ = rgba.shape
    bgra = np.empty_like(rgba)
    bgra[..., 0] = rgba[..., 2]  # B
    bgra[..., 1] = rgba[..., 1]  # G
    bgra[..., 2] = rgba[..., 0]  # R
    bgra[..., 3] = rgba[..., 3]  # A
    qimg = QImage(bgra.data, w, h, 4 * w, QImage.Format_ARGB32)
    return qimg.copy()


def _hillshade(z: np.ndarray, cell: float) -> np.ndarray:
    """Return hillshade 0..255 (uint8). z may contain NaN."""
    zf = z.astype("float32")
    mask = ~np.isfinite(zf)
    if mask.any():
        zf[mask] = np.nanmean(zf)
    gy, gx = np.gradient(zf, cell, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gy, gx)
    az = np.deg2rad(315.0)
    alt = np.deg2rad(45.0)
    hs = (np.sin(alt) * np.cos(slope) +
          np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    hs = np.clip(hs, 0, 1)
    return (hs * 255).astype(np.uint8)


def _rgba_from_scalar(a: np.ndarray, cm: str = "turbo", alpha: float = 0.75,
                      vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """Map scalar -> RGBA (uint8). NaN -> alpha=0."""
    import matplotlib.cm as cm_mod
    a = a.astype("float32")
    nan_mask = ~np.isfinite(a)
    if vmin is None:
        vmin = np.nanpercentile(a, 2)
    if vmax is None:
        vmax = np.nanpercentile(a, 98)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.nanmin(a), np.nanmax(a)
    rng = float(max(1e-9, vmax - vmin))
    x = np.clip((a - vmin) / rng, 0.0, 1.0)
    rgba = cm_mod.get_cmap(cm)(x, bytes=True).astype(np.uint8)
    a_byte = int(max(0, min(255, round(alpha * 255.0))))
    rgba[..., 3] = a_byte
    if nan_mask.any():
        rgba[..., 3][nan_mask] = 0
    return rgba

# ---------- Auto-lines helpers (giống logic UI2 cũ) ----------

def _unit(v: np.ndarray, eps=1e-9):
    n = float(np.hypot(v[0], v[1]))
    if n < eps:
        return np.array([1.0, 0.0], dtype=float), n
    return (v / n), n


def _centroid_from_mask(mask: np.ndarray, transform: Affine) -> Optional[Tuple[float, float]]:
    """Centroid (X,Y) của vùng mask>0 trong toạ độ map."""
    rr, cc = np.nonzero(mask)
    if len(rr) == 0:
        return None
    r0 = rr.mean()
    c0 = cc.mean()
    x0, y0 = rio_xy(transform, r0, c0, offset="center")
    return float(x0), float(y0)


def _pca_dir_from_mask(mask: np.ndarray, transform: Affine) -> np.ndarray:
    """Fallback: dùng PCA shape vùng trượt để lấy hướng chính."""
    rr, cc = np.nonzero(mask)
    if len(rr) < 2:
        return np.array([1.0, 0.0], dtype=float)

    xs, ys = rio_xy(transform, rr, cc, offset="center")
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    X = np.vstack([xs - xs.mean(), ys - ys.mean()]).T  # N x 2
    C = np.cov(X.T)
    vals, vecs = np.linalg.eigh(C)     # eigenvalues tăng dần
    d = vecs[:, np.argmax(vals)]       # eigenvector eigenvalue lớn nhất
    u, _ = _unit(d)
    return u


def _build_line(center_xy: Tuple[float, float], u_dir: np.ndarray, length_m: float) -> LineString:
    """Tạo LineString tâm tại center_xy, hướng u_dir (unit), chiều dài length_m."""
    cx, cy = center_xy
    half = 0.5 * float(length_m)
    dx, dy = u_dir * half
    return LineString([(cx - dx, cy - dy), (cx + dx, cy + dy)])

def _densify_line(line: LineString, step_m: float) -> Tuple[np.ndarray, np.ndarray]:
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


def _bbox_length_from_mask(
    mask: np.ndarray,
    transform: Affine,
    scale: float = 1.0,
    min_length: float = 0.0,
) -> float:
    """
    Chiều dài tuyến đặc trưng ≈ max(kích thước bbox vùng trượt) * scale.

    - scale = 1.0: đúng bằng bbox landslide (không nới rộng).
    - min_length: chiều dài tối thiểu (m), dùng để tránh L = 0.
    """
    rr, cc = np.nonzero(mask)
    if len(rr) == 0:
        return 0.0

    xs, ys = rio_xy(transform, rr, cc, offset="center")
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    Lx = xs.max() - xs.min()
    Ly = ys.max() - ys.min()
    L = max(Lx, Ly) * float(scale)

    if not np.isfinite(L) or L <= 0.0:
        return 0.0

    return max(L, float(min_length))


def generate_auto_lines_from_arrays(
    dx: np.ndarray,
    dy: np.ndarray,
    mask: np.ndarray,
    transform: Affine,
    main_num_even: int,
    main_offset_m: float,
    cross_num_even: int,
    cross_offset_m: float,
    base_length_m: Optional[float] = None,
    min_mag_thresh: float = 1e-4,
) -> dict:
    """
    Sinh main/cross lines từ trường vector (dx,dy) + mask.

    Logic bám sát generate_auto_lines_from_slipzone() bản UI2 gốc:
    - Slip zone: mask > 0.
    - Hướng ML-001:
        + lấy vector trung bình <dx, dy> trên slip zone,
        + chuẩn hoá thành u_main,
        + nếu độ lớn quá nhỏ -> fallback sang PCA shape của mask.
    """

    # 1) mask bool
    m_mask = (mask > 0)

    # nếu không có pixel nào trong slipzone → trả về rỗng
    if not np.any(m_mask):
        return {"main": [], "cross": [], "debug": {}}

    # 2) centroid vùng trượt
    center = _centroid_from_mask(m_mask, transform)
    if center is None:
        return {"main": [], "cross": [], "debug": {}}

    # 3) vector trung bình trong slip zone (theo dX/dY)
    m_valid = m_mask & np.isfinite(dx) & np.isfinite(dy)
    if not np.any(m_valid):
        # không có vector hợp lệ → lấy mặc định, độ lớn 0
        v_map = np.array([1.0, 0.0], dtype=float)
        u_main, mag_mean = _unit(v_map)
        mag_mean = 0.0
    else:
        v_pix = np.array(
            [np.nanmean(dx[m_valid]), np.nanmean(dy[m_valid])],
            dtype=float,
        )
        # x = a*col + b*row + c ; y = d*col + e*row + f
        a, b, _, d, e, _ = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        v_map = np.array([a * v_pix[0] + b * v_pix[1], d * v_pix[0] + e * v_pix[1]], dtype=float)
        u_main, mag_mean = _unit(v_map)

    # nếu độ lớn quá nhỏ → fallback PCA shape
    if mag_mean < float(min_mag_thresh):
        u_main = _pca_dir_from_mask(m_mask, transform)
        u_main, _ = _unit(u_main)

    # 4) pháp tuyến
    u_norm = np.array([-u_main[1], u_main[0]], dtype=float)

    # 5) chiều dài tuyến (giống bản gốc: từ bbox mask)
    if base_length_m is not None and base_length_m > 0:
        L = float(base_length_m)
    else:
        L = _bbox_length_from_mask(m_mask, transform, scale=1.0, min_length=0.0)

    if not np.isfinite(L) or L <= 0.0:
        return {"main": [], "cross": [], "debug": {}}

    feats_main: List[dict] = []
    feats_cross: List[dict] = []

    # 6) Tuyến gốc ML-001 & CL-001 đi qua centroid
    main1 = _build_line(center, u_main, L)
    cross1 = _build_line(center, u_norm, L)

    ang_main = float(np.degrees(np.arctan2(u_main[1], u_main[0])))
    ang_cross = float(np.degrees(np.arctan2(u_norm[1], u_norm[0])))

    feats_main.append({
        "name": "ML-001",
        "type": "main",
        "offset_m": 0.0,
        "angle_deg": ang_main,
        "geom": main1,
    })
    feats_cross.append({
        "name": "CL-001",
        "type": "cross",
        "offset_m": 0.0,
        "angle_deg": ang_cross,
        "geom": cross1,
    })

    # 7) Các ML-xxx bổ sung (song song ML-001, offset theo pháp tuyến)
    idx = 2
    for k in range(1, max(0, int(main_num_even)) // 2 + 1):
        for sgn in (+1, -1):
            off = sgn * float(main_offset_m) * k
            cx = center[0] + off * u_norm[0]
            cy = center[1] + off * u_norm[1]
            geom = _build_line((cx, cy), u_main, L)
            feats_main.append({
                "name": f"ML-{idx:03d}",
                "type": "main",
                "offset_m": off,
                "angle_deg": ang_main,
                "geom": geom,
            })
            idx += 1

    # 8) Các CL-xxx bổ sung (song song CL-001, offset theo u_main)
    idx = 2
    for k in range(1, max(0, int(cross_num_even)) // 2 + 1):
        for sgn in (+1, -1):
            off = sgn * float(cross_offset_m) * k
            cx = center[0] + off * u_main[0]
            cy = center[1] + off * u_main[1]
            geom = _build_line((cx, cy), u_norm, L)
            feats_cross.append({
                "name": f"CL-{idx:03d}",
                "type": "cross",
                "offset_m": off,
                "angle_deg": ang_cross,
                "geom": geom,
            })
            idx += 1

    debug = {
        "mag_mean": float(mag_mean),
        "u_main": u_main.tolist(),
        "ang_main_deg": ang_main,
    }

    return {
        "main": feats_main,
        "cross": feats_cross,
        "debug": debug,
    }

# ---------- Layered viewer (one canvas, 4 layers) ----------
class _LayeredViewer(UI1Viewer):
    """
    Một viewer dùng chung cho UI2:
    - Nền: hillshade (before.asc)  → layer 0
    - Heatmap: dZ masked           → layer 1
    - Vectors: dx/dy (mũi tên)     → layer 2
    - Lưới toạ độ + text           → layer 10–11
    - Crosshair, dot, marker start, rubber line
    """

    cursorMoved = pyqtSignal(float, float)                  # X,Y theo map
    sectionPicked = pyqtSignal(float, float, float, float)  # x1,y1,x2,y2 (map)

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- run context (được MainWindow set sau khi Analyze xong) ---
        self._ctx_ready: bool = False
        self._ctx_project: str = ""
        self._ctx_runlabel: str = ""
        self._ctx_run_dir: str = ""
        self._ui1_dir: str = ""
        self._ui2_dir: str = ""

        # đảm bảo các list chứa item của viewer luôn tồn tại (tránh None/crash)
        self._grid_items = []
        self._grid_w = None
        self._grid_h = None
        self._grid_step_m = 20.0
        self._grid_font_size = 32
        self._vec_items = []
        self._hill_item = None
        self._heat_item = None

        # georef
        self._tr: Optional[Affine] = None
        self._cell: float = 1.0

        # base & overlays
        self._hill_item: Optional[QGraphicsPixmapItem] = None
        self._heat_item: Optional[QGraphicsPixmapItem] = None
        self._vec_items: list = []
        self._grid_items: list = []

        # crosshair
        pen_cross = QPen(QColor("#27ae60"))
        pen_cross.setWidth(0)
        pen_cross.setStyle(Qt.DotLine)

        self._cross_h = self.scene.addLine(0, 0, 0, 0, pen_cross)
        self._cross_v = self.scene.addLine(0, 0, 0, 0, pen_cross)
        self._cross_h.setZValue(1100)
        self._cross_v.setZValue(1100)

        # marker start
        self._start_marker = QGraphicsEllipseItem(-4, -4, 8, 8)
        self._start_marker.setBrush(QBrush(QColor("#2ecc71")))
        self._start_marker.setPen(QPen(Qt.NoPen))
        self._start_marker.setZValue(1200)
        self._start_marker.hide()
        self.scene.addItem(self._start_marker)

        # rubber line start→cursor
        pen_rb = QPen(QColor("#27ae60"))
        pen_rb.setWidth(0)
        self._rubber = self.scene.addLine(0, 0, 0, 0, pen_rb)
        self._rubber.setZValue(1150)
        self._rubber.hide()

        # hover dot
        self._hover_dot = QGraphicsEllipseItem(-3, -3, 6, 6)
        self._hover_dot.setBrush(QBrush(QColor("#2ecc71")))
        self._hover_dot.setPen(QPen(Qt.NoPen))
        self._hover_dot.setZValue(1200)
        self._hover_dot.hide()
        self.scene.addItem(self._hover_dot)

        # picking state
        self._picking: bool = True
        self._p1_pix: Optional[Tuple[float, float]] = None  # (col,row) trong raster

        # behaviour view
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.setMouseTracking(True)
        self.view.viewport().setMouseTracking(True)
        self.view.viewport().installEventFilter(self)
        self.view.viewport().setCursor(Qt.ArrowCursor)

        self.view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.view.setRenderHints(self.view.renderHints() | QPainter.Antialiasing)

    # ---- georef helpers ----
    def set_transform(self, tr: Affine) -> None:
        self._tr = tr
        self._cell = float(abs(tr.a)) if tr is not None else 1.0

    def pix_to_map(self, c: float, r: float) -> Tuple[float, float]:
        if self._tr is None:
            return c, r
        x = self._tr.c + self._tr.a * c
        y = self._tr.f + self._tr.e * r
        return float(x), float(y)

    # ---- drawing API ----
    def set_hillshade(self, hs8: np.ndarray) -> None:
        """Đặt hillshade (uint8 HxW) làm nền & cố định scene rect (có margin)."""
        if getattr(self, "_hill_item", None) is not None:
            self.scene.removeItem(self._hill_item)
            self._hill_item = None

        qimg = _np_to_qimage_gray(hs8)
        pm = QPixmap.fromImage(qimg)
        self._hill_item = QGraphicsPixmapItem(pm)
        self._hill_item.setZValue(0)
        self._hill_item.setPos(0, 0)
        self.scene.addItem(self._hill_item)

        w, h = pm.width(), pm.height()

        # Thêm margin để số toạ độ vẽ ra ngoài vẫn nhìn thấy
        pad_x = 60  # lề trái/phải
        pad_y = 30  # lề trên/dưới
        self.scene.setSceneRect(-pad_x, -pad_y, w + 2 * pad_x, h + 2 * pad_y)
        self.view.setSceneRect(self.scene.sceneRect())
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def set_hillshade_opacity(self, val_0_1: float) -> None:
        if self._hill_item is not None:
            self._hill_item.setOpacity(float(max(0.0, min(1.0, val_0_1))))

    def set_grid_font_size(self, size_px: int) -> None:
        self._grid_font_size = int(size_px)
        if self._grid_w is not None and self._grid_h is not None:
            self.set_grid(self._grid_w, self._grid_h, step_m=self._grid_step_m)

    def set_grid(self, width: int, height: int, step_m: float = 20.0) -> None:
        """Vẽ lưới tọa độ 20 m + nhãn X,Y."""
        for it in getattr(self, "_grid_items", []):
            self.scene.removeItem(it)
        self._grid_items = []

        if self._tr is None:
            return

        self._grid_w = int(width)
        self._grid_h = int(height)
        self._grid_step_m = float(step_m)

        # pixel / mét
        pix_per_m_x = 1.0 / abs(self._tr.a)
        pix_per_m_y = 1.0 / abs(self._tr.e)

        step_px_x = step_m * pix_per_m_x
        step_px_y = step_m * pix_per_m_y

        pen = QPen(QColor(200, 0, 0, 180))
        pen.setStyle(Qt.DashLine)
        pen.setCosmetic(True)
        pen.setWidth(0)

        text_brush = QBrush(QColor(180, 0, 0))

        # Vertical lines + X labels
        x = 0.0
        while x <= width:
            line = QGraphicsLineItem(x, 0, x, height)
            line.setPen(pen)
            line.setZValue(10)
            self.scene.addItem(line)
            self._grid_items.append(line)

            mx, my = self.pix_to_map(x, height)
            lbl = QGraphicsSimpleTextItem(f"{mx:.0f}")
            #set font chu ui2 to hon
            lbl.setFont(QFont("Arial", self._grid_font_size))
            lbl.setBrush(text_brush)
            lbl.setPos(x + 2, height + 2)
            lbl.setZValue(11)
            self.scene.addItem(lbl)
            self._grid_items.append(lbl)

            x += step_px_x

        # Horizontal lines + Y labels
        y = 0.0
        while y <= height:
            line = QGraphicsLineItem(0, y, width, y)
            line.setPen(pen)
            line.setZValue(10)
            self.scene.addItem(line)
            self._grid_items.append(line)

            mx, my = self.pix_to_map(0, y)
            lbl = QGraphicsSimpleTextItem(f"{my:.0f}")
            #set font chu ui2 to hon
            lbl.setFont(QFont("Arial", self._grid_font_size))
            lbl.setBrush(text_brush)
            # Đặt ra ngoài bên trái ảnh
            fm = QFontMetrics(lbl.font())
            text_w = fm.horizontalAdvance(lbl.text())
            y_off = int(self._grid_font_size * 0.25)
            lbl.setPos(-text_w - 10, y - y_off)
            lbl.setZValue(11)
            self.scene.addItem(lbl)
            self._grid_items.append(lbl)
            y += step_px_y

    def set_heatmap_rgba(self, rgba: Optional[np.ndarray], opacity: float) -> None:
        """Đặt/clear lớp heatmap (RGBA uint8) tại (0,0), z=1."""
        if getattr(self, "_heat_item", None) is not None:
            self.scene.removeItem(self._heat_item)
            self._heat_item = None

        if rgba is None:
            return

        pm = QPixmap.fromImage(_np_to_qimage_rgba(rgba))
        self._heat_item = QGraphicsPixmapItem(pm)
        self._heat_item.setPos(0, 0)
        self._heat_item.setOpacity(float(max(0.0, min(1.0, opacity))))
        self._heat_item.setZValue(1)
        self.scene.addItem(self._heat_item)

    def set_heatmap_opacity(self, val_0_1: float) -> None:
        if self._heat_item is not None:
            self._heat_item.setOpacity(float(max(0.0, min(1.0, val_0_1))))

    def clear_vectors(self) -> None:
        for it in self._vec_items:
            self.scene.removeItem(it)
        self._vec_items.clear()

    def set_vector_opacity(self, val_0_1: float) -> None:
        a = float(max(0.0, min(1.0, val_0_1)))
        for it in self._vec_items:
            it.setOpacity(a)

    # ---- picking / cursor ----
    def start_pick(self) -> None:
        self._picking = True
        self._p1_pix = None
        self._start_marker.hide()
        self._rubber.hide()
        self._cross_show(True)

    def cancel_pick(self) -> None:
        self._p1_pix = None
        self._start_marker.hide()
        self._rubber.hide()
        # vẫn cho pick tiếp

    def _cross_show(self, on: bool) -> None:
        self._cross_h.setVisible(on)
        self._cross_v.setVisible(on)
        self._hover_dot.setVisible(on)

    def eventFilter(self, obj, ev):
        if obj is self.view.viewport():
            if ev.type() == QEvent.MouseMove:
                sp = self.view.mapToScene(ev.pos())
                c, r = float(sp.x()), float(sp.y())

                # crosshair + dot
                self._cross_h.setLine(0, r, self.scene.width(), r)
                self._cross_v.setLine(c, 0, c, self.scene.height())
                self._hover_dot.setPos(c, r)
                self._hover_dot.show()

                # rubber line nếu đã có start
                if self._picking and self._p1_pix is not None:
                    c1, r1 = self._p1_pix
                    self._rubber.setLine(c1, r1, c, r)
                    self._rubber.show()

                x, y = self.pix_to_map(c, r)
                self.cursorMoved.emit(x, y)
                return True

            if ev.type() == QEvent.MouseButtonPress and ev.button() == Qt.RightButton:
                if self._picking and self._p1_pix is not None:
                    self.cancel_pick()
                return True

            if ev.type() == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                sp = self.view.mapToScene(ev.pos())
                c, r = float(sp.x()), float(sp.y())

                if not self._picking:
                    return True

                if self._p1_pix is None:
                    # click 1 → start
                    self._p1_pix = (c, r)
                    self._start_marker.setPos(c, r)
                    self._start_marker.show()
                else:
                    # click 2 → end
                    x1, y1 = self.pix_to_map(*self._p1_pix)
                    x2, y2 = self.pix_to_map(c, r)
                    self.sectionPicked.emit(x1, y1, x2, y2)
                    self.cancel_pick()
                return True

        return super().eventFilter(obj, ev)

from PyQt5.QtWidgets import QDialog, QGridLayout, QSpinBox, QDoubleSpinBox

class AutoLineDialog(QDialog):
    """
    Dialog nhập tham số Auto Line Generation:
      - Số main lines (tuyến chính, chẵn, tính cả 2 bên)
      - Offset main (m)
      - Số cross lines (tuyến phụ, chẵn)
      - Offset cross (m)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Line Generation")

        layout = QVBoxLayout(self)

        grp = QGroupBox("Automatic Lines")
        grid = QGridLayout(grp)

        # Main lines
        self.main_num = QSpinBox()
        self.main_num.setRange(0, 200)
        self.main_num.setSingleStep(2)
        self.main_num.setValue(4)

        self.main_off = QDoubleSpinBox()
        self.main_off.setRange(0.0, 1e6)
        self.main_off.setDecimals(2)
        self.main_off.setValue(20.0)

        # Cross lines
        self.cross_num = QSpinBox()
        self.cross_num.setRange(0, 200)
        self.cross_num.setSingleStep(2)
        self.cross_num.setValue(4)

        self.cross_off = QDoubleSpinBox()
        self.cross_off.setRange(0.0, 1e6)
        self.cross_off.setDecimals(2)
        self.cross_off.setValue(20.0)

        # Layout grid
        grid.addWidget(QLabel("Main lines:"),        0, 0)
        grid.addWidget(QLabel("Line number:"),       0, 1)
        grid.addWidget(self.main_num,                0, 2)
        grid.addWidget(QLabel("Offset (m):"),        0, 3)
        grid.addWidget(self.main_off,                0, 4)

        grid.addWidget(QLabel("Cross lines:"),       1, 0)
        grid.addWidget(QLabel("Line number:"),       1, 1)
        grid.addWidget(self.cross_num,               1, 2)
        grid.addWidget(QLabel("Offset (m):"),        1, 3)
        grid.addWidget(self.cross_off,               1, 4)


        layout.addWidget(grp)

        # OK / Cancel
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

# ---------- UI2: Section Selection tab ----------

class SectionSelectionTab(QWidget):
    sections_confirmed = pyqtSignal(str, str, str)  # project, run_label, run_dir
    def __init__(self, base_dir: str, parent=None) -> None:
        super().__init__(parent)
        self.base_dir = base_dir

        # run context
        self._ctx_ready: bool = False
        self._ctx_project: str = ""
        self._ctx_runlabel: str = ""
        self._ctx_run_dir: str = ""
        self._ui1_dir: str = ""
        self._ui2_dir: str = ""

        # viewer items list init
        self._grid_items = []
        self._vec_items = []
        self._hill_item = None
        self._heat_item = None

        self.project: Optional[str] = None
        self.run_label: Optional[str] = None
        self.run_dir: Optional[str] = None
        self._section_lines: list[QGraphicsLineItem] = []
        self._preview_line: Optional[QGraphicsLineItem] = None

        # caches
        self._tr: Optional[Affine] = None
        self._inv_tr: Optional[Affine] = None
        self._dz: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._dy: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None  # uint8 0/1 aligned to dz grid
        self._dem_path: Optional[str] = None

        # vector drawing params (sync với UI1)
        self._vec_step: int = 25
        self._vec_scale: float = 0.5
        self._vec_size_pct: int = 100
        self._vec_pen_base: int = 1
        self._vec_arrow_base: float = 12.0

        self._sections: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

        self._updating_table: bool = False

        self._build_ui()
        self._wire()

    # ---- public API ----
    # ---- public API ----
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

        # cập nhật thông số vector nếu được truyền từ UI1
        if vec_step is not None:
            self._vec_step = int(vec_step)
        if vec_scale is not None:
            self._vec_scale = float(vec_scale)

        # ✨ lưu vào các thuộc tính public để các hàm khác đọc
        self.project = self._ctx_project
        self.run_label = self._ctx_runlabel
        self.run_dir = self._ctx_run_dir

        self._ctx_ready = (
                os.path.isdir(self._ui1_dir)
                and os.path.isfile(os.path.join(self._ui1_dir, "dx.tif"))
                and os.path.isfile(os.path.join(self._ui1_dir, "dy.tif"))
        )

        # cập nhật text vào ô hiển thị (read-only)
        if hasattr(self, "edit_project"):
            self.edit_project.setText(self.project)
            self.edit_project.setReadOnly(True)
        if hasattr(self, "edit_runlabel"):
            self.edit_runlabel.setText(self.run_label)
            self.edit_runlabel.setReadOnly(True)

        # nạp layer
        self._load_layers_and_show()
        self._ok("[UI2] Context set OK.")


    # ---- UI ----
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)
        self.viewer = _LayeredViewer(self)

        # left pane
        left = QWidget(); left_lo = QVBoxLayout(left)

        grp_proj = QGroupBox("Project"); gl = QVBoxLayout(grp_proj)
        row1 = HBox(); row1.addWidget(QLabel("Name:"))
        self.edit_project = QLineEdit(); self.edit_project.setPlaceholderText("—")
        row1.addWidget(self.edit_project, 1); gl.addLayout(row1)
        row2 = HBox(); row2.addWidget(QLabel("Run label:"))
        self.edit_runlabel = QLineEdit(); self.edit_runlabel.setPlaceholderText("—")
        row2.addWidget(self.edit_runlabel, 1); gl.addLayout(row2)
        left_lo.addWidget(grp_proj)

        grp_layers = QGroupBox("Layers"); ll = QVBoxLayout(grp_layers)
        r_gf = HBox(); r_gf.addWidget(QLabel("Grid font size:"))
        self.sld_grid_font = QSlider(Qt.Horizontal); self.sld_grid_font.setRange(8, 72); self.sld_grid_font.setValue(12)
        r_gf.addWidget(self.sld_grid_font, 1); ll.addLayout(r_gf)
        r_hs = HBox(); r_hs.addWidget(QLabel("Hillshade opacity:"))
        self.sld_hill = QSlider(Qt.Horizontal); self.sld_hill.setRange(0, 100); self.sld_hill.setValue(100)
        r_hs.addWidget(self.sld_hill, 1); ll.addLayout(r_hs)
        r_hm = HBox(); r_hm.addWidget(QLabel("Heatmap opacity:"))
        self.sld_heat = QSlider(Qt.Horizontal); self.sld_heat.setRange(0, 100); self.sld_heat.setValue(75)
        r_hm.addWidget(self.sld_heat, 1); ll.addLayout(r_hm)
        r_vs = HBox(); r_vs.addWidget(QLabel("Vector size:"))
        self.sld_vec_size = QSlider(Qt.Horizontal); self.sld_vec_size.setRange(50, 200); self.sld_vec_size.setValue(50)
        r_vs.addWidget(self.sld_vec_size, 1); ll.addLayout(r_vs)
        r_vc = HBox(); r_vc.addWidget(QLabel("Vectors opacity:"))
        self.sld_vec = QSlider(Qt.Horizontal); self.sld_vec.setRange(0, 100); self.sld_vec.setValue(100)
        r_vc.addWidget(self.sld_vec, 1); ll.addLayout(r_vc)
        left_lo.addWidget(grp_layers)

        grp_secs = QGroupBox("Sections");
        sl = QVBoxLayout(grp_secs)
        self.tbl = QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["#", "Start (x,y)", "End (x,y)"])
        hdr = self.tbl.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, hdr.Fixed)  # cột #
        hdr.setSectionResizeMode(1, hdr.Fixed)  # Start
        hdr.setSectionResizeMode(2, hdr.Stretch)  # End (cột cuối cùng fill phần dư)
        self.tbl.itemChanged.connect(self._on_table_item_changed)
        sl.addWidget(self.tbl)

        # Hàng trên: Auto Line Generation + Preview + Clear (dàn đều chiều ngang)
        row_top = HBox()
        self.btn_auto = QPushButton("Auto Line Generation")
        self.btn_prev = QPushButton("Preview line")
        self.btn_clear = QPushButton("Clear All")

        for b in (self.btn_auto, self.btn_prev, self.btn_clear):
            # stretch=1 → 3 nút chia đều chiều ngang
            row_top.addWidget(b, 1)
        sl.addLayout(row_top)

        # Hàng dưới: Confirm sections, nút kéo dài hết chiều ngang layout
        row_confirm = HBox()
        self.btn_confirm = QPushButton("Confirm sections")
        row_confirm.addWidget(self.btn_confirm, 1)  # stretch=1 → full width
        sl.addLayout(row_confirm)

        left_lo.addWidget(grp_secs)

        grp_status = QGroupBox("Status"); sv = QVBoxLayout(grp_status)
        self.lbl_cursor = QLabel("Cursor: —")
        from PyQt5.QtWidgets import QPlainTextEdit
        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.status_text.setMaximumBlockCount(2000)  # tránh phình bộ nhớ khi log dài
        self.status_text.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")
        sv.addWidget(self.lbl_cursor); sv.addWidget(self.status_text)
        left_lo.addWidget(grp_status)

        left_lo.addStretch(1)
        # right pane: viewer
        splitter.addWidget(left)  # khung trái (form, table, status)
        splitter.addWidget(self.viewer)  # khung phải (map)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([600, 700])

        # Áp style nút
        self._apply_button_style()


    def _apply_button_style(self) -> None:
        """Áp style tab Section Selection (nền trắng + nút xanh)."""
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
        # self.setStyleSheet(style)

    def _wire(self) -> None:
        self.sld_vec_size.valueChanged.connect(self._on_vec_size_changed)
        self.sld_grid_font.valueChanged.connect(self.viewer.set_grid_font_size)
        self.sld_hill.valueChanged.connect(lambda v: self.viewer.set_hillshade_opacity(v / 100.0))
        self.sld_heat.valueChanged.connect(lambda v: self.viewer.set_heatmap_opacity(v / 100.0))
        self.sld_vec.valueChanged.connect(lambda v: self.viewer.set_vector_opacity(v / 100.0))
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_prev.clicked.connect(self._on_preview)
        self.btn_confirm.clicked.connect(self._on_confirm_sections)
        self.btn_auto.clicked.connect(self._on_auto_lines)

        self.viewer.sectionPicked.connect(self._on_section_picked)
        self.viewer.cursorMoved.connect(lambda x, y: self.lbl_cursor.setText(f"Cursor: X={x:.2f}, Y={y:.2f}"))

    def _on_vec_size_changed(self, v: int) -> None:
        self._vec_size_pct = int(v)
        if self._ctx_ready:
            self._load_dx_dy_and_draw(self._ui1_dir, step=self._vec_step, scale=self._vec_scale)

    # ---- core loading/drawing ----
    def _load_layers_and_show(self) -> None:
        # Lấy run_dir từ context đã set
        rd = getattr(self, "run_dir", "") or self._ctx_run_dir
        if not rd or not os.path.isdir(rd):
            return

        ui1 = os.path.join(rd, "ui1")

        dz_tif = os.path.join(ui1, "dz.tif")
        before_tif = os.path.join(ui1, "before_asc.tif")
        before_smooth = os.path.join(ui1, "before_asc_smooth.tif")
        if _file_exists(before_smooth):
            before_tif = before_smooth
        self._dem_path = before_tif

        # Có thể có các tên mask khác nhau
        # Có thể có các tên mask khác nhau (ưu tiên detect_mask.tif của UI1 mới)
        mask_tif = None
        for name in ("detect_mask.tif", "mask.tif", "landslide_mask.tif", "mask_binary.tif"):
            p = os.path.join(ui1, name)
            if _file_exists(p):
                mask_tif = p
                break

        # --- 1) đọc dZ (làm base grid & transform) ---
        with rasterio.open(dz_tif) as ds:
            self._dz = ds.read(1).astype("float32")
            self._tr = ds.transform
            self._inv_tr = ~self._tr
            W, H = ds.width, ds.height

        # --- 2) đọc BEFORE & resample về grid dZ nếu cần ---
        with rasterio.open(before_tif) as ds:
            if (ds.width, ds.height) != (W, H) or ds.transform != self._tr:
                before = ds.read(
                    1, out_shape=(H, W), resampling=Resampling.bilinear
                ).astype("float32")
            else:
                before = ds.read(1).astype("float32")

        # --- 3) đọc dx/dy & align về grid dZ để tính dXY ---
        self._dx, self._dy = None, None

        def _read_align(name: str) -> Optional[np.ndarray]:
            p = os.path.join(ui1, name)
            if not _file_exists(p):
                return None
            with rasterio.open(p) as ds:
                arr = ds.read(1).astype("float32")
                if ds.transform != self._tr or (ds.width, ds.height) != (W, H):
                    arr = ds.read(
                        1,
                        out_shape=(H, W),
                        resampling=Resampling.bilinear
                    ).astype("float32")
            return arr

        self._dx = _read_align("dx.tif")
        self._dy = _read_align("dy.tif")

        # --- 4) đọc mask (uint8 0/1) & align về grid dZ ---
        if mask_tif and _file_exists(mask_tif):
            with rasterio.open(mask_tif) as ds:
                if (ds.width, ds.height) != (W, H) or ds.transform != self._tr:
                    mask = ds.read(
                        1, out_shape=(H, W), resampling=Resampling.nearest
                    ).astype("uint8")
                else:
                    mask = ds.read(1).astype("uint8")
            self._mask = (mask > 0).astype("uint8")
        else:
            # nếu chưa có mask → tạm coi tất cả là 1
            self._mask = np.ones_like(self._dz, dtype="uint8")
            self._info("[UI2] mask.tif not found → using full extent as mask.")

        # --- 5) hillshade từ BEFORE đã align ---
        cell = float(abs(self._tr.a))
        hs8 = _hillshade(before, cell)
        self.viewer.set_transform(self._tr)
        self.viewer.set_hillshade(hs8)  # chốt scene rect & fit
        self.viewer.set_hillshade_opacity(self.sld_hill.value() / 100.0)
        self.viewer.set_grid(W, H, step_m=20.0)
        self.viewer.set_grid_font_size(self.sld_grid_font.value())

        # --- 6) Heatmap = landslide/dXY ---
        if self._dx is not None and self._dy is not None:
            mag = np.hypot(self._dx, self._dy)  # √(dx²+dy²)
            mag[self._mask == 0] = np.nan  # chỉ giữ vùng landslide
            heat_scalar = mag
        else:
            # fallback: dZ chỉ trong vùng mask
            heat_scalar = self._dz.copy()
            heat_scalar[self._mask == 0] = np.nan

        alpha = self.sld_heat.value() / 100.0
        rgba = _rgba_from_scalar(
            heat_scalar,
            cm="turbo",
            alpha=alpha,
        )
        self.viewer.set_heatmap_rgba(rgba, alpha)

        # --- 7) Vẽ vector (dùng dx/dy vừa đọc) theo tham số từ UI1 ---
        self._load_dx_dy_and_draw(
            ui1,
            step=self._vec_step,
            scale=self._vec_scale,
        )

        # fit view
        self.viewer.view.fitInView(
            self.viewer.scene.itemsBoundingRect(), Qt.KeepAspectRatio
        )
        self._ok("[UI2] Layers loaded & aligned.")
        self._load_saved_sections()

    def _load_saved_sections(self) -> None:
        """
        Đọc lại ui2/sections.csv (nếu tồn tại) và vẽ lại các tuyến lên map + bảng.
        """
        if not self.run_dir:
            return

        ui2_dir = os.path.join(self.run_dir, "ui2")
        csv_path = os.path.join(ui2_dir, "sections.csv")
        if not os.path.isfile(csv_path):
            self._info("[UI2] No saved sections.csv – start with empty sections.")
            return

        try:
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            self._info(f"[UI2] Cannot read sections.csv: {e}")
            return

        if not rows:
            self._info("[UI2] sections.csv is empty.")
            return

        # Xoá mọi thứ hiện tại trước khi load lại
        self.tbl.setRowCount(0)
        self._sections.clear()

        # Xoá line cũ trên map (nếu có)
        for it in self._section_lines:
            if it is not None:
                self.viewer.scene.removeItem(it)
        self._section_lines.clear()

        # Append từng section vào bảng + vẽ line
        count = 0
        for row in rows:
            try:
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
            except Exception:
                continue

            self._append_section((x1, y1), (x2, y2))
            count += 1

        self._ok(f"[UI2] Loaded {count} sections from sections.csv")

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
        k = 0.4  # hệ số hiệu chỉnh, có thể chỉnh 0.3–0.6 tuỳ mắt nhìn
        vx_pix = (dx / (pix_w + 1e-9)) * scale * k
        vy_pix = (-dy / (pix_h + 1e-9)) * scale * k  # y pixel hướng xuống

        pts_pix = np.stack([xs, ys], axis=1).astype("float32")
        vec_pix = np.stack([vx_pix, vy_pix], axis=1).astype("float32")
        # vẽ vector có mũi tên — giống UI1
        # vẽ vector màu magenta
        pen = QPen(QColor("#ffffff"))  # hoặc QColor(191, 0, 255)
        pen.setCosmetic(True)
        pen.setWidth(max(1, int(round(self._vec_pen_base * (self._vec_size_pct / 100.0)))))

        for (x, y), (vx, vy) in zip(pts_pix, vec_pix):
            # đảo hướng y vì raster y+ xuống
            end_x = float(x + vx)
            end_y = float(y - vy)
            # thân vector
            line = QGraphicsLineItem(float(x), float(y), end_x, end_y)
            line.setPen(pen)
            line.setZValue(2)
            self.viewer.scene.addItem(line)
            self.viewer._vec_items.append(line)

            # mũi tên (tam giác nhỏ)
            arr_len = float(self._vec_arrow_base) * (self._vec_size_pct / 100.0)
            arr_ang = np.deg2rad(45)
            dx, dy = end_x - x, end_y - y
            ang = np.arctan2(dy, dx)
            p1 = QPointF(end_x - arr_len * np.cos(ang - arr_ang),
                         end_y - arr_len * np.sin(ang - arr_ang))
            p2 = QPointF(end_x - arr_len * np.cos(ang + arr_ang),
                         end_y - arr_len * np.sin(ang + arr_ang))
            arrow = QGraphicsPathItem()
            path = QPainterPath(QPointF(end_x, end_y))
            path.lineTo(p1)
            path.lineTo(p2)
            path.lineTo(QPointF(end_x, end_y))
            arrow.setPath(path)
            arrow.setPen(QPen(Qt.NoPen))
            arrow.setBrush(QBrush(QColor("#ffffff")))  # cùng màu với thân
            arrow.setZValue(2)
            self.viewer.scene.addItem(arrow)
            self.viewer._vec_items.append(arrow)

    # ---- section picking ----
    def _on_section_picked(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self._append_section((x1, y1), (x2, y2))
        self._ok(f"Section added: ({x1:.2f},{y1:.2f}) → ({x2:.2f},{y2:.2f})")

    def _append_section(
            self,
            p0: Tuple[float, float],
            p1: Tuple[float, float],
            label: Optional[str] = None,
    ) -> None:
        """Lưu section vào bảng + vẽ line cố định trên map."""
        # tránh trigger _on_table_item_changed khi đang chèn row
        self._updating_table = True

        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        hdr = self.tbl.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, hdr.Fixed)
        hdr.setSectionResizeMode(1, hdr.Fixed)
        hdr.setSectionResizeMode(2, hdr.Stretch)

        line_label = label if label is not None else str(r + 1)
        self.tbl.setItem(r, 0, QTableWidgetItem(line_label))
        self.tbl.setItem(r, 1, QTableWidgetItem(f"{p0[0]:.2f}, {p0[1]:.2f}"))
        self.tbl.setItem(r, 2, QTableWidgetItem(f"{p1[0]:.2f}, {p1[1]:.2f}"))
        self.tbl.verticalHeader().setDefaultSectionSize(22)  # chiều cao mỗi row
        self.tbl.setColumnWidth(0, 40)  # cột chỉ số
        self.tbl.setColumnWidth(1, 240)  # Start

        self._updating_table = False
        # cột cuối cùng đang stretch, giữ nguyên

        # lưu section
        self._sections.append((p0, p1))

        # 2) vẽ line lên viewer (map → pixel)
        line_item = None
        if self._inv_tr is not None:
            c0, r0 = self._inv_tr * p0
            c1, r1 = self._inv_tr * p1
            pen = QPen(QColor(30, 200, 30, 200))
            pen.setCosmetic(True)
            pen.setWidth(2)
            line_item = QGraphicsLineItem(c0, r0, c1, r1)
            line_item.setPen(pen)
            line_item.setZValue(3)
            self.viewer.scene.addItem(line_item)

        self._section_lines.append(line_item)
        self._ok("Section line drawn on map.")

    def _on_table_item_changed(self, item) -> None:
        """
        Khi user sửa Start/End (x,y) trong bảng, cập nhật ngay line tương ứng trên viewer.
        """
        if self._updating_table:
            return  # bỏ qua khi đang cập nhật bằng code

        if item is None:
            return

        row = item.row()
        col = item.column()

        # chỉ quan tâm cột 1 (Start) và 2 (End)
        if col not in (1, 2):
            return

        # lấy cả start & end của dòng hiện tại
        start_item = self.tbl.item(row, 1)
        end_item = self.tbl.item(row, 2)
        if start_item is None or end_item is None:
            return

        try:
            p0 = tuple(map(float, start_item.text().split(",")))
            p1 = tuple(map(float, end_item.text().split(",")))
            if self._inv_tr is None:
                return
            c0, r0 = self._inv_tr * p0
            c1, r1 = self._inv_tr * p1
        except Exception:
            # nếu user gõ dở dang (ví dụ "120,"), tạm bỏ qua
            return

        # cập nhật list _sections
        if 0 <= row < len(self._sections):
            self._sections[row] = (p0, p1)

        # cập nhật line xanh trên map
        if 0 <= row < len(self._section_lines):
            line_item = self._section_lines[row]
            if line_item is not None:
                line_item.setLine(c0, r0, c1, r1)

        # nếu đang có preview line cho đúng dòng này, cập nhật luôn
        if self._preview_line is not None and self.tbl.currentRow() == row:
            self._preview_line.setLine(c0, r0, c1, r1)

        self._ok(f"Updated section #{row + 1} from table edit.")

    def _on_auto_lines(self) -> None:
        """
        Handler cho nút 'Auto Line Generation'.

        Bản này dùng trực tiếp dx/dy/mask đã load trong SectionSelectionTab
        (self._dx, self._dy, self._mask, self._tr) và gọi
        generate_auto_lines_from_arrays(...) để đảm bảo hướng tuyến
        giống vector dXY ở UI1.
        """
        # 0) Kiểm tra context
        if self.run_dir is None or self._tr is None:
            self._info("[UI2] Auto line: run context not ready. Please run Analyze first.")
            return

        if self._dx is None or self._dy is None or self._mask is None:
            self._info("[UI2] Auto line: dx/dy/mask not ready. Please run Analyze first.")
            return

        # 1) Hiện dialog nhập tham số
        dlg = AutoLineDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return  # người dùng Cancel

        main_num_even = int(dlg.main_num.value())
        main_offset_m = float(dlg.main_off.value())
        cross_num_even = int(dlg.cross_num.value())
        cross_offset_m = float(dlg.cross_off.value())

        # 2) Ép số chẵn + kiểm tra offset >= 0 (giống UI2 cũ)
        if main_num_even % 2 == 1:
            main_num_even -= 1
            self._info("[UI2] Main lines: số tuyến phải là số chẵn – đã tự động giảm 1.")

        if cross_num_even % 2 == 1:
            cross_num_even -= 1
            self._info("[UI2] Cross lines: số tuyến phải là số chẵn – đã tự động giảm 1.")

        if main_offset_m < 0 or cross_offset_m < 0:
            self._info("[UI2] Auto line: Offset must be ≥ 0.")
            return

        # 3) Gọi backend dạng mảng – không phụ thuộc tên file detect_mask.tif nữa
        try:
            outs = generate_auto_lines_from_arrays(
                dx=self._dx,
                dy=self._dy,
                mask=self._mask,
                transform=self._tr,
                main_num_even=main_num_even,
                main_offset_m=main_offset_m,
                cross_num_even=cross_num_even,
                cross_offset_m=cross_offset_m,
                base_length_m=None,
            )
        except Exception as e:
            self._info(f"[UI2] Auto line: generation failed: {e}")
            return

        mains = outs.get("main", [])
        crosses = outs.get("cross", [])
        if not mains and not crosses:
            self._info("[UI2] Auto line: no lines returned.")
            return

        # redraw vectors after auto line generation
        self._load_dx_dy_and_draw(self._ui1_dir, step=self._vec_step, scale=self._vec_scale)
        debug = outs.get("debug", {})
        ang = debug.get("ang_main_deg", None)
        if ang is not None:
            self._ok(f"[UI2] Main direction ≈ {ang:.1f} deg")

        # 4) Xoá tất cả section cũ (bảng + line trên map)
        self.tbl.setRowCount(0)
        self._sections.clear()
        for it in getattr(self, "_section_lines", []):
            if it is not None:
                self.viewer.scene.removeItem(it)
        self._section_lines.clear()
        if getattr(self, "_preview_line", None) is not None:
            self.viewer.scene.removeItem(self._preview_line)
            self._preview_line = None

        # 5) Helper: convert feat (LineString) -> 1 dòng trong bảng + vẽ line
        def _add_feat(feat: dict) -> None:
            geom = feat.get("geom")
            if not isinstance(geom, LineString) or geom.is_empty:
                return
            x1, y1 = geom.coords[0]
            x2, y2 = geom.coords[-1]
            self._append_section((x1, y1), (x2, y2))

        for f in mains:
            _add_feat(f)
        for f in crosses:
            _add_feat(f)

        self._ok(f"[UI2] Auto line OK: {len(mains)} main + {len(crosses)} cross lines.")

    def _on_preview(self) -> None:
        r = self.tbl.currentRow()
        if r < 0 or self._inv_tr is None:
            self._info("Select a row to preview.")
            return

        # Xoá preview cũ nếu có
        if self._preview_line is not None:
            self.viewer.scene.removeItem(self._preview_line)
            self._preview_line = None

        p0 = tuple(map(float, self.tbl.item(r, 1).text().split(",")))
        p1 = tuple(map(float, self.tbl.item(r, 2).text().split(",")))
        c0, r0 = self._inv_tr * p0
        c1, r1 = self._inv_tr * p1

        pen = QPen(QColor(200, 30, 30, 220))
        pen.setCosmetic(True)
        pen.setWidth(2)
        item = QGraphicsLineItem(c0, r0, c1, r1)
        item.setPen(pen)
        item.setZValue(4)  # cao hơn line xanh
        self.viewer.scene.addItem(item)
        self._preview_line = item

        self._ok("Preview line drawn.")

    def _on_clear(self) -> None:
        """Xoá toàn bộ sections + line trên map."""
        reply = QMessageBox.question(
            self,
            "Confirm",
            "本当に消しますか？\nAre you sure you want to delete it?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.tbl.setRowCount(0)
        self._sections.clear()

        # xoá line khỏi scene
        for it in self._section_lines:
            if it is not None:
                self.viewer.scene.removeItem(it)
        self._section_lines.clear()

        self._ok("Cleared sections.")
        if self._preview_line is not None:
            self.viewer.scene.removeItem(self._preview_line)
            self._preview_line = None

    def _on_confirm_sections(self) -> None:
        """Ghi ui2/sections.csv và phát 1 signal sang MainWindow/Curve tab."""
        # Bắt buộc phải có context (Analyze tab đã chạy)
        if not getattr(self, "_ctx_ready", False):
            self._log("[UI2] Please run Analyze first (no context).")
            return

        # Đọc context đang lưu trong UI2
        project = self._ctx_project or (
            self.edit_project.text().strip() if hasattr(self, "edit_project") else ""
        )
        run_label = self._ctx_runlabel or (
            self.edit_runlabel.text().strip() if hasattr(self, "edit_runlabel") else ""
        )
        run_dir = self._ctx_run_dir

        if not (project and run_label and run_dir):
            self._err("Missing project/run context. Please render vectors in Analyze tab first.")
            return

        # Ghi ui2/sections.csv
        ui2_dir = os.path.join(run_dir, "ui2")
        os.makedirs(ui2_dir, exist_ok=True)
        csv_path = os.path.join(ui2_dir, "sections.csv")

        try:
            import csv
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["idx", "x1", "y1", "x2", "y2"])
                for r in range(self.tbl.rowCount()):
                    p0 = tuple(map(float, self.tbl.item(r, 1).text().split(",")))
                    p1 = tuple(map(float, self.tbl.item(r, 2).text().split(",")))
                    w.writerow([r + 1, p0[0], p0[1], p1[0], p1[1]])

            self._ok(f"Sections saved: {csv_path}")

        except Exception as e:
            self._err(f"Confirm Sections error: {e}")
            return

        # 👉 Emit DUY NHẤT 1 lần, dùng context chuẩn
        try:
            self.sections_confirmed.emit(project, run_label, run_dir)
            self._log(f"[UI2] sections_confirmed emitted: {project}, {run_label}, {run_dir}")
        except Exception as e:
            self._err(f"Emit sections_confirmed error: {e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            sel = self.tbl.selectedIndexes()
            if not sel:
                return

            rows = sorted(set(i.row() for i in sel))

            # nếu chọn ít nhất 1 cột Start/End → coi như xóa cả dòng
            if len(rows) >= 1:
                for r in reversed(rows):
                    # xoá line trên map
                    if 0 <= r < len(self._section_lines):
                        it = self._section_lines[r]
                        if it is not None:
                            self.viewer.scene.removeItem(it)
                        self._section_lines.pop(r)

                    if 0 <= r < len(self._sections):
                        self._sections.pop(r)

                    self.tbl.removeRow(r)

                if self._preview_line is not None:
                    self.viewer.scene.removeItem(self._preview_line)
                    self._preview_line = None

                self._ok("Deleted selected section(s).")
                event.accept()
                return

        super().keyPressEvent(event)
    # ---------- Public API cho MainWindow ----------

    def reset_session(self) -> None:
        """
        Reset tab Section:
        - Xoá project/run label hiển thị
        - Xoá sections trong bảng + line trên map
        - Xoá status
        - Xoá context run_dir / transform
        """
        # Clear thông tin project/run
        if hasattr(self, "edit_project"):
            self.edit_project.clear()
        if hasattr(self, "edit_runlabel"):
            self.edit_runlabel.clear()

        # Reset context
        self.project = None
        self.run_label = None
        self.run_dir = None
        self._ctx_project = ""
        self._ctx_runlabel = ""
        self._ctx_run_dir = ""
        self._ui1_dir = ""
        self._ui2_dir = ""
        self._ctx_ready = False
        self._dx = None
        self._dy = None
        self._tr = None
        self._inv_tr = None
        self._dz = None
        self._mask = None

        # reset vector params về mặc định
        self._vec_step = 25
        self._vec_scale = 0.5

        # Xoá sections & line
        if hasattr(self, "tbl"):
            self.tbl.setRowCount(0)

        # Xoá line section đã vẽ
        if hasattr(self, "_section_lines"):
            for it in self._section_lines:
                if it is not None:
                    self.viewer.scene.removeItem(it)
            self._section_lines.clear()

        # Xoá preview line nếu có
        if hasattr(self, "_preview_line") and self._preview_line is not None:
            self.viewer.scene.removeItem(self._preview_line)
            self._preview_line = None

        # Xoá status
        if hasattr(self, "status_text"):
            self.status_text.clear()

        # Xoá layer / vector / lưới (nếu viewer có các hàm này)
        try:
            # Nếu bạn có các hàm clear_vectors / set_heatmap_rgba..., có thể dùng thêm
            self.viewer.clear_vectors()
            self.viewer.set_heatmap_rgba(
                None, 0.0
            )  # nếu hàm này không chấp nhận None thì bỏ dòng này
        except Exception:
            pass

    # ---- status helpers ----
    def _append_status(self, text: str) -> None:
        self.status_text.appendPlainText(text)
        self.status_text.moveCursor(self.status_text.textCursor().End)

    def _info(self, msg: str) -> None:
        self._append_status(f"[UI2] INFO: {msg}")

    def _ok(self, msg: str) -> None:
        self._append_status(f"[UI2] OK: {msg}")

    def _err(self, msg: str) -> None:
        """Append error to status + popup."""
        self._append_status(f"[UI2] ERROR: {msg}")
        try:
            QMessageBox.critical(self, "Section Selection", msg)
        except Exception:
            pass  # phòng trường hợp gọi khi app chưa sẵn sàng

    # Back-compat: vài nơi cũ còn gọi self._log(...)
    def _log(self, msg: str) -> None:
        self._info(msg)
