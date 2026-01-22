# pedi_oku_landslide/ui/widgets/dual_viewer.py
from __future__ import annotations
import os
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy, QScrollArea
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

def _make_panel():
    # panel = Title + Scrollable QLabel
    root = QWidget()
    v = QVBoxLayout(root); v.setContentsMargins(0,0,0,0); v.setSpacing(6)
    title = QLabel(" "); title.setAlignment(Qt.AlignCenter)
    title.setStyleSheet("font-weight:600;")
    area = QScrollArea(); area.setWidgetResizable(True)
    img = QLabel(); img.setAlignment(Qt.AlignCenter)
    img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    area.setWidget(img)
    v.addWidget(title); v.addWidget(area, 1)
    root._title = title
    root._img = img
    root._area = area
    return root

class DualViewer(QWidget):
    """
    Viewer đơn giản: hiển thị 2 ảnh cạnh nhau (trái/phải).
    Cung cấp: show_pair(left_path, right_path, left_title="", right_title="")
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self); h.setContentsMargins(0,0,0,0); h.setSpacing(8)
        self.left = _make_panel(); self.right = _make_panel()
        h.addWidget(self.left, 1); h.addWidget(self.right, 1)

    def _set_panel(self, panel: QWidget, path: str, title: str):
        panel._title.setText(title or " ")
        if isinstance(path, str) and os.path.isfile(path):
            pm = QPixmap(path)
            # scale-to-fit chiều rộng panel, giữ tỉ lệ
            target_w = max(200, panel._area.viewport().width()-20)
            if not pm.isNull():
                pm = pm.scaledToWidth(target_w, Qt.SmoothTransformation)
            panel._img.setPixmap(pm)
        else:
            panel._img.setPixmap(QPixmap())

    def show_pair(self, left_path: str, right_path: str, left_title: str = "", right_title: str = ""):
        self._set_panel(self.left, left_path, left_title)
        self._set_panel(self.right, right_path, right_title)
