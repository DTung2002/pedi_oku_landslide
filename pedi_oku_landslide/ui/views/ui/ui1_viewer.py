# pedi_oku_landslide/ui/views/ui1_viewer.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsScene, QGraphicsView,
    QGraphicsPixmapItem, QLabel, QPushButton
)
from PyQt5.QtGui import QBrush
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt, QPointF
import os

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Render hints cho ảnh mượt
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.TextAntialiasing, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom = 1.0

    def wheelEvent(self, event):
        # Zoom bằng bánh xe chuột
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            factor = zoom_in_factor
        else:
            factor = zoom_out_factor

        self.scale(factor, factor)
        self._zoom *= factor

    def set_zoom_100(self):
        # Reset về 100%
        self.resetTransform()
        self._zoom = 1.0

    def fit_to_scene(self, scene_rect=None):
        rect = scene_rect or self.scene().itemsBoundingRect()
        if rect.isNull():
            return
        self.set_zoom_100()
        # Fit toàn bộ
        self.fitInView(rect, Qt.KeepAspectRatio)


class UI1Viewer(QWidget):
    """
    Side-by-side viewer (Before / After) giống phong cách UI1 cũ,
    có nút zoom +/−, 100%, Fit, và zoom bằng bánh xe chuột.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self.scene, self)
        self.caption = QLabel("")
        self.caption.setStyleSheet("font-weight: 600;")
        self.scene.setBackgroundBrush(QBrush(Qt.white))
        self.view.setStyleSheet("background: #ffffff;")
        self.caption.setStyleSheet("background: #ffffff;")

        # Toolbar zoom
        tool_row = QHBoxLayout()
        self.btn_zoom_in  = QPushButton("Zoom +")
        self.btn_zoom_out = QPushButton("Zoom −")
        self.btn_zoom_fit = QPushButton("Fit")
        self.btn_zoom_100 = QPushButton("100%")
        for b in (self.btn_zoom_in, self.btn_zoom_out, self.btn_zoom_fit, self.btn_zoom_100):
            tool_row.addWidget(b)
        tool_row.addStretch(1)

        self.btn_zoom_in.clicked.connect(lambda: self.view.scale(1.15, 1.15))
        self.btn_zoom_out.clicked.connect(lambda: self.view.scale(1/1.15, 1/1.15))
        self.btn_zoom_fit.clicked.connect(lambda: self.view.fit_to_scene())
        self.btn_zoom_100.clicked.connect(self.view.set_zoom_100)

        lay = QVBoxLayout(self)
        lay.addWidget(self.caption)
        lay.addLayout(tool_row)
        lay.addWidget(self.view, 1)

    def show_pair(
            self,
            left_path: str | None,
            right_path: str | None,
            left_title: str | None = None,
            right_title: str | None = None,
    ) -> None:
        """
        Show left and right images side-by-side; if right missing, show only left.
        Hỗ trợ tiêu đề tuỳ chọn cho caption qua left_title / right_title.
        Backward-compatible với các chỗ gọi cũ (không truyền tiêu đề).
        """
        import os  # đảm bảo có os; bỏ nếu bạn đã import ở đầu file

        self.scene.clear()

        if not left_path or not os.path.exists(left_path):
            self.caption.setText("UI1 – Missing images to show.")
            return

        # Left
        left_pix = QPixmap(left_path)
        left_item = QGraphicsPixmapItem(left_pix)
        left_item.setZValue(0)
        self.scene.addItem(left_item)

        # Right (optional)
        has_right = bool(right_path and os.path.exists(right_path))
        if has_right:
            right_pix = QPixmap(right_path)
            right_item = QGraphicsPixmapItem(right_pix)
            right_item.setOffset(left_pix.width(), 0)
            right_item.setZValue(0)
            self.scene.addItem(right_item)

        # Caption (ưu tiên tiêu đề truyền vào nếu có)
        if has_right:
            if left_title or right_title:
                lt = left_title or "Left"
                rt = right_title or "Right"
                self.caption.setText(f"{lt} | {rt}")
            else:
                self.caption.setText("UI1 – Hillshade Preview (Before | After)")
        else:
            if left_title:
                self.caption.setText(left_title)
            else:
                self.caption.setText("UI1 – Hillshade Preview (Before)")

        # Fit để ảnh to hết khung; người dùng có thể zoom tiếp bằng nút/bánh xe
        self.view.fit_to_scene()

