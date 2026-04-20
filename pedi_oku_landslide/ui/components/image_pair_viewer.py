from PyQt5.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QBrush, QPainter, QPixmap
from PyQt5.QtCore import Qt
import os

from pedi_oku_landslide.ui.layout_constants import (
    PREVIEW_FIT_BUTTON_H,
    PREVIEW_FIT_BUTTON_W,
    PREVIEW_MIN_H,
    PREVIEW_VIEWPORT_STYLE,
)


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.TextAntialiasing, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom = 1.0

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self._zoom *= factor

    def reset_view_transform(self):
        self.resetTransform()
        self._zoom = 1.0

    def fit_to_scene(self, scene_rect=None):
        rect = scene_rect or self.scene().itemsBoundingRect()
        if rect.isNull():
            return
        self.reset_view_transform()
        self.fitInView(rect, Qt.KeepAspectRatio)


class UI1Viewer(QWidget):
    """Live shared image viewer used by Analyze and Section tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self.scene, self)
        self.caption = QLabel("")
        self.caption.setStyleSheet("font-weight: 600;")
        self.scene.setBackgroundBrush(QBrush(Qt.white))
        self.view.setStyleSheet(PREVIEW_VIEWPORT_STYLE)
        self.view.setMinimumHeight(PREVIEW_MIN_H)
        self.caption.setStyleSheet("background: #ffffff;")
        self.caption.hide()

        self.btn_zoom_in = QPushButton("Zoom +")
        self.btn_zoom_out = QPushButton("Zoom -")
        self.btn_zoom_fit = QPushButton("Fit")
        self.btn_zoom_fit.setFixedSize(PREVIEW_FIT_BUTTON_W, PREVIEW_FIT_BUTTON_H)

        self.btn_zoom_in.clicked.connect(lambda: self.view.scale(1.15, 1.15))
        self.btn_zoom_out.clicked.connect(lambda: self.view.scale(1 / 1.15, 1 / 1.15))
        self.btn_zoom_fit.clicked.connect(lambda: self.view.fit_to_scene())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.view, 1)

    def show_pair(
        self,
        left_path: str | None,
        right_path: str | None,
        left_title: str | None = None,
        right_title: str | None = None,
    ) -> None:
        self.scene.clear()

        if not left_path or not os.path.exists(left_path):
            self.caption.setText("UI1 - Missing images to show.")
            return

        left_pix = QPixmap(left_path)
        left_item = QGraphicsPixmapItem(left_pix)
        left_item.setZValue(0)
        self.scene.addItem(left_item)

        has_right = bool(right_path and os.path.exists(right_path))
        if has_right:
            right_pix = QPixmap(right_path)
            right_item = QGraphicsPixmapItem(right_pix)
            right_item.setOffset(left_pix.width(), 0)
            right_item.setZValue(0)
            self.scene.addItem(right_item)

        if has_right:
            if left_title or right_title:
                self.caption.setText(f"{left_title or 'Left'} | {right_title or 'Right'}")
            else:
                self.caption.setText("UI1 - Hillshade Preview (Before | After)")
        else:
            self.caption.setText(left_title or "UI1 - Hillshade Preview (Before)")

        self.view.fit_to_scene()
