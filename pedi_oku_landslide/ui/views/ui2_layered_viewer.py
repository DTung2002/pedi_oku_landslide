from typing import Optional, Tuple

import numpy as np
from rasterio.transform import Affine
from PyQt5.QtCore import QPointF, QEvent, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QFontMetrics, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from pedi_oku_landslide.ui.components.image_pair_viewer import UI1Viewer


def _np_to_qimage_gray(img8: np.ndarray) -> QImage:
    h, w = img8.shape
    qimg = QImage(img8.data, w, h, w, QImage.Format_Grayscale8)
    return qimg.copy()


def _np_to_qimage_rgba(rgba: np.ndarray) -> QImage:
    h, w, _ = rgba.shape
    bgra = np.empty_like(rgba)
    bgra[..., 0] = rgba[..., 2]
    bgra[..., 1] = rgba[..., 1]
    bgra[..., 2] = rgba[..., 0]
    bgra[..., 3] = rgba[..., 3]
    qimg = QImage(bgra.data, w, h, 4 * w, QImage.Format_ARGB32)
    return qimg.copy()


class _LayeredViewer(UI1Viewer):
    cursorMoved = pyqtSignal(float, float)
    sectionPicked = pyqtSignal(float, float, float, float)
    polylinePicked = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid_items = []
        self._grid_w = None
        self._grid_h = None
        self._grid_step_m = 20.0
        self._grid_font_size = 32
        self._vec_items = []
        self._hill_item = None
        self._heat_item = None
        self._tr: Optional[Affine] = None
        self._cell: float = 1.0
        self._hill_item: Optional[QGraphicsPixmapItem] = None
        self._heat_item: Optional[QGraphicsPixmapItem] = None
        self._vec_items: list = []
        self._grid_items: list = []

        pen_cross = QPen(QColor("#27ae60"))
        pen_cross.setWidth(0)
        pen_cross.setStyle(Qt.DotLine)
        self._cross_h = self.scene.addLine(0, 0, 0, 0, pen_cross)
        self._cross_v = self.scene.addLine(0, 0, 0, 0, pen_cross)
        self._cross_h.setZValue(1100)
        self._cross_v.setZValue(1100)

        self._start_marker = QGraphicsEllipseItem(-4, -4, 8, 8)
        self._start_marker.setBrush(QBrush(QColor("#2ecc71")))
        self._start_marker.setPen(QPen(Qt.NoPen))
        self._start_marker.setZValue(1200)
        self._start_marker.hide()
        self.scene.addItem(self._start_marker)

        pen_rb = QPen(QColor("#27ae60"))
        pen_rb.setWidth(0)
        self._rubber = self.scene.addLine(0, 0, 0, 0, pen_rb)
        self._rubber.setZValue(1150)
        self._rubber.hide()
        self._polyline_preview = QGraphicsPathItem()
        self._polyline_preview.setPen(pen_rb)
        self._polyline_preview.setZValue(1150)
        self._polyline_preview.hide()
        self.scene.addItem(self._polyline_preview)

        self._hover_dot = QGraphicsEllipseItem(-3, -3, 6, 6)
        self._hover_dot.setBrush(QBrush(QColor("#2ecc71")))
        self._hover_dot.setPen(QPen(Qt.NoPen))
        self._hover_dot.setZValue(1200)
        self._hover_dot.hide()
        self.scene.addItem(self._hover_dot)

        self._picking = True
        self._draw_mode = "straight"
        self._p1_pix: Optional[Tuple[float, float]] = None
        self._polyline_pix: list[Tuple[float, float]] = []
        self._last_mouse_scene: Optional[QPointF] = None

        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.setMouseTracking(True)
        self.view.viewport().setMouseTracking(True)
        self.view.viewport().installEventFilter(self)
        self.view.viewport().setCursor(Qt.ArrowCursor)
        self.view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.view.setRenderHints(self.view.renderHints() | QPainter.Antialiasing)
        try:
            self.btn_zoom_in.clicked.disconnect()
            self.btn_zoom_out.clicked.disconnect()
        except Exception:
            pass
        self.btn_zoom_in.clicked.connect(lambda: self._zoom_by_factor(1.15))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_by_factor(1 / 1.15))

    def set_transform(self, tr: Affine) -> None:
        self._tr = tr
        self._cell = float(abs(tr.a)) if tr is not None else 1.0

    def pix_to_map(self, c: float, r: float) -> Tuple[float, float]:
        if self._tr is None:
            return c, r
        x = self._tr.c + self._tr.a * c
        y = self._tr.f + self._tr.e * r
        return float(x), float(y)

    def set_hillshade(self, hs8: np.ndarray) -> None:
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
        pad_x = 60
        pad_y = 30
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
        for it in getattr(self, "_grid_items", []):
            self.scene.removeItem(it)
        self._grid_items = []
        if self._tr is None:
            return
        self._grid_w = int(width)
        self._grid_h = int(height)
        self._grid_step_m = float(step_m)
        pix_per_m_x = 1.0 / abs(self._tr.a)
        pix_per_m_y = 1.0 / abs(self._tr.e)
        step_px_x = step_m * pix_per_m_x
        step_px_y = step_m * pix_per_m_y
        pen = QPen(QColor(140, 140, 140, 180))
        pen.setStyle(Qt.DashLine)
        pen.setCosmetic(True)
        pen.setWidth(0)
        text_brush = QBrush(QColor(0, 0, 0))

        x = 0.0
        while x <= width:
            line = QGraphicsLineItem(x, 0, x, height)
            line.setPen(pen)
            line.setZValue(10)
            self.scene.addItem(line)
            self._grid_items.append(line)
            mx, _ = self.pix_to_map(x, height)
            lbl = QGraphicsSimpleTextItem(f"{mx:.0f}")
            lbl.setFont(QFont("Arial", self._grid_font_size))
            lbl.setBrush(text_brush)
            lbl.setPos(x + 2, height + 2)
            lbl.setZValue(11)
            self.scene.addItem(lbl)
            self._grid_items.append(lbl)
            x += step_px_x

        y = 0.0
        while y <= height:
            line = QGraphicsLineItem(0, y, width, y)
            line.setPen(pen)
            line.setZValue(10)
            self.scene.addItem(line)
            self._grid_items.append(line)
            _, my = self.pix_to_map(0, y)
            lbl = QGraphicsSimpleTextItem(f"{my:.0f}")
            lbl.setFont(QFont("Arial", self._grid_font_size))
            lbl.setBrush(text_brush)
            fm = QFontMetrics(lbl.font())
            text_w = fm.horizontalAdvance(lbl.text())
            y_off = int(self._grid_font_size * 0.25)
            lbl.setPos(-text_w - 10, y - y_off)
            lbl.setZValue(11)
            self.scene.addItem(lbl)
            self._grid_items.append(lbl)
            y += step_px_y

    def set_heatmap_rgba(self, rgba: Optional[np.ndarray], opacity: float) -> None:
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

    def start_pick(self) -> None:
        self._picking = True
        self._p1_pix = None
        self._polyline_pix = []
        self._start_marker.hide()
        self._rubber.hide()
        self._polyline_preview.hide()
        self._cross_show(True)

    def cancel_pick(self) -> None:
        self._p1_pix = None
        self._polyline_pix = []
        self._start_marker.hide()
        self._rubber.hide()
        self._polyline_preview.hide()

    def set_draw_mode(self, mode: str) -> None:
        draw_mode = str(mode or "straight").strip().lower()
        self._draw_mode = "polyline" if draw_mode == "polyline" else "straight"
        self.cancel_pick()

    def is_polyline_active(self) -> bool:
        return self._draw_mode == "polyline" and len(self._polyline_pix) >= 2

    def finish_polyline_pick(self) -> bool:
        if self._draw_mode != "polyline" or len(self._polyline_pix) < 2:
            return False
        pts = [self.pix_to_map(c, r) for c, r in self._polyline_pix]
        self.polylinePicked.emit(pts)
        self.cancel_pick()
        return True

    def _append_polyline_point(self, c: float, r: float) -> None:
        pt = (float(c), float(r))
        if self._polyline_pix and abs(self._polyline_pix[-1][0] - pt[0]) <= 1e-6 and abs(self._polyline_pix[-1][1] - pt[1]) <= 1e-6:
            return
        self._polyline_pix.append(pt)
        first = self._polyline_pix[0]
        self._start_marker.setPos(first[0], first[1])
        self._start_marker.show()

    def _update_polyline_preview(self, current_pt: Optional[Tuple[float, float]] = None) -> None:
        pts = list(self._polyline_pix)
        if current_pt is not None:
            pts.append((float(current_pt[0]), float(current_pt[1])))
        if len(pts) < 2:
            self._polyline_preview.hide()
            return
        path = QPainterPath(QPointF(pts[0][0], pts[0][1]))
        for c, r in pts[1:]:
            path.lineTo(c, r)
        self._polyline_preview.setPath(path)
        self._polyline_preview.show()

    def _cross_show(self, on: bool) -> None:
        self._cross_h.setVisible(on)
        self._cross_v.setVisible(on)
        self._hover_dot.setVisible(on)

    def eventFilter(self, obj, ev):
        if obj is self.view.viewport():
            if ev.type() == QEvent.Wheel:
                dy = ev.angleDelta().y()
                if dy == 0:
                    return True
                factor = 1.15 if dy > 0 else (1 / 1.15)
                self._zoom_at(ev.pos(), factor)
                return True
            if ev.type() == QEvent.MouseMove:
                sp = self.view.mapToScene(ev.pos())
                c, r = float(sp.x()), float(sp.y())
                self._last_mouse_scene = sp
                self._cross_h.setLine(0, r, self.scene.width(), r)
                self._cross_v.setLine(c, 0, c, self.scene.height())
                self._hover_dot.setPos(c, r)
                self._hover_dot.show()
                if self._picking:
                    if self._draw_mode == "straight" and self._p1_pix is not None:
                        c1, r1 = self._p1_pix
                        self._rubber.setLine(c1, r1, c, r)
                        self._rubber.show()
                    elif self._draw_mode == "polyline" and self._polyline_pix:
                        self._update_polyline_preview((c, r))
                x, y = self.pix_to_map(c, r)
                self.cursorMoved.emit(x, y)
                return True
            if ev.type() == QEvent.MouseButtonPress and ev.button() == Qt.RightButton:
                if self._picking and (self._p1_pix is not None or self._polyline_pix):
                    self.cancel_pick()
                    return True
                return True
            if ev.type() == QEvent.MouseButtonDblClick and ev.button() == Qt.LeftButton:
                sp = self.view.mapToScene(ev.pos())
                c, r = float(sp.x()), float(sp.y())
                if self._picking and self._draw_mode == "polyline":
                    self._append_polyline_point(c, r)
                    self.finish_polyline_pick()
                    return True
            if ev.type() == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                sp = self.view.mapToScene(ev.pos())
                c, r = float(sp.x()), float(sp.y())
                if not self._picking:
                    return True
                if self._draw_mode == "polyline":
                    self._append_polyline_point(c, r)
                    self._update_polyline_preview()
                else:
                    if self._p1_pix is None:
                        self._p1_pix = (c, r)
                        self._start_marker.setPos(c, r)
                        self._start_marker.show()
                    else:
                        x1, y1 = self.pix_to_map(*self._p1_pix)
                        x2, y2 = self.pix_to_map(c, r)
                        self.sectionPicked.emit(x1, y1, x2, y2)
                        self.cancel_pick()
                return True
        return super().eventFilter(obj, ev)

    def _zoom_by_factor(self, factor: float) -> None:
        if self._last_mouse_scene is not None:
            vp = self.view.mapFromScene(self._last_mouse_scene)
        else:
            vp = self.view.viewport().rect().center()
        self._zoom_at(vp, factor)

    def _zoom_at(self, viewport_pos, factor: float) -> None:
        old_pos = self.view.mapToScene(viewport_pos)
        self.view.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.view.scale(factor, factor)
        new_pos = self.view.mapToScene(viewport_pos)
        delta = new_pos - old_pos
        self.view.translate(delta.x(), delta.y())
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
