from typing import Optional, Callable

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsView


class AnchorMarkerItem(QGraphicsEllipseItem):
    def __init__(
        self,
        x: float,
        y: float,
        r: float,
        tooltip: str,
        on_click: Optional[Callable[[], None]] = None,
    ):
        super().__init__(x - r, y - r, 2 * r, 2 * r)
        self._on_click = on_click
        self.setAcceptHoverEvents(True)
        self.setToolTip(tooltip)

    def mousePressEvent(self, event):
        try:
            if self._on_click is not None:
                self._on_click()
        except Exception:
            pass
        super().mousePressEvent(event)


class ZoomableGraphicsView(QGraphicsView):
    sceneMouseMoved = pyqtSignal(float, float)
    hoverExited = pyqtSignal()

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._zoom = 0

    def wheelEvent(self, e):
        zoom_in = 1.25
        zoom_out = 0.8
        if e.angleDelta().y() > 0:
            factor = zoom_in
            self._zoom += 1
        else:
            factor = zoom_out
            self._zoom -= 1
        self.scale(factor, factor)

    def fit_to_scene(self):
        if not self.scene() or not self.scene().items():
            return
        rect = self.scene().itemsBoundingRect()
        if rect.isNull():
            return
        self.resetTransform()
        self.fitInView(rect, Qt.KeepAspectRatio)
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

    def mouseMoveEvent(self, e):
        try:
            sp = self.mapToScene(e.pos())
            self.sceneMouseMoved.emit(float(sp.x()), float(sp.y()))
        except Exception:
            pass
        super().mouseMoveEvent(e)

    def leaveEvent(self, e):
        try:
            self.hoverExited.emit()
        except Exception:
            pass
        super().leaveEvent(e)
