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

    def set_zoom_100(self):
        self.resetTransform()
        self._zoom = 1.0

    def fit_to_scene(self, scene_rect=None):
        rect = scene_rect or self.scene().itemsBoundingRect()
        if rect.isNull():
            return
        self.set_zoom_100()
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
        self.view.setStyleSheet("background: #ffffff;")
        self.caption.setStyleSheet("background: #ffffff;")

        tool_row = QHBoxLayout()
        self.btn_zoom_in = QPushButton("Zoom +")
        self.btn_zoom_out = QPushButton("Zoom -")
        self.btn_zoom_fit = QPushButton("Fit")
        self.btn_zoom_100 = QPushButton("100%")
        for button in (self.btn_zoom_in, self.btn_zoom_out, self.btn_zoom_fit, self.btn_zoom_100):
            tool_row.addWidget(button)
        tool_row.addStretch(1)

        self.btn_zoom_in.clicked.connect(lambda: self.view.scale(1.15, 1.15))
        self.btn_zoom_out.clicked.connect(lambda: self.view.scale(1 / 1.15, 1 / 1.15))
        self.btn_zoom_fit.clicked.connect(lambda: self.view.fit_to_scene())
        self.btn_zoom_100.clicked.connect(self.view.set_zoom_100)

        layout = QVBoxLayout(self)
        layout.addWidget(self.caption)
        layout.addLayout(tool_row)
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
