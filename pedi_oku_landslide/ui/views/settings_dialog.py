# pedi_oku_landslide/ui/views/settings_dialog.py

from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QWidget,
)
from PyQt5.QtCore import Qt

from pedi_oku_landslide.project.settings_store import (
    load_settings,
    save_settings,
    AppSettings,
)


class SettingsDialog(QDialog):
    """
    Hộp thoại Settings:
    - Chọn UI scale
    - Nút New Session để reset phiên làm việc
    """

    def __init__(self, base_dir: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._base_dir = base_dir
        self._settings: AppSettings = load_settings(base_dir)
        self._new_session: bool = False

        self.setWindowTitle("Settings")
        self.setModal(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # --- UI scale row ---
        row = QHBoxLayout()
        row.addWidget(QLabel("UI scale:"))
        self.cmb_scale = QComboBox()
        for v in (25, 50, 75, 100, 125, 150, 175, 200):
            self.cmb_scale.addItem(f"{v} %", v)
        # chọn scale hiện tại
        cur = self._settings.ui_scale_percent
        idx = next(
            (i for i in range(self.cmb_scale.count())
             if self.cmb_scale.itemData(i) == cur),
            -1,
        )
        if idx >= 0:
            self.cmb_scale.setCurrentIndex(idx)
        row.addWidget(self.cmb_scale)
        row.addStretch(1)
        root.addLayout(row)

        # --- Button row ---
        btn_row = QHBoxLayout()

        # New Session
        # self.btn_new_session = QPushButton("New Session")
        # self.btn_new_session.clicked.connect(self._on_new_session)
        # btn_row.addWidget(self.btn_new_session)
        #
        # btn_row.addStretch(1)

        # Cancel / OK
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self._on_ok)
        btn_row.addWidget(self.btn_ok)

        root.addLayout(btn_row)

    # --------- Internal handlers ---------

    def _on_ok(self) -> None:
        """Chỉ cập nhật scale, KHÔNG reset session."""
        scale = self.selected_scale()
        self.persist(scale)
        self._new_session = False
        self.accept()

    def _on_new_session(self) -> None:
        """
        Cập nhật scale (nếu đổi) + đặt cờ new_session = True.
        MainWindow sẽ đọc cờ này để reset UI.
        """
        scale = self.selected_scale()
        self.persist(scale)
        self._new_session = True
        self.accept()

    # --------- Public API cho MainWindow ---------

    def selected_scale(self) -> int:
        return int(self.cmb_scale.currentData())

    def persist(self, new_scale: int) -> None:
        self._settings.ui_scale_percent = new_scale
        save_settings(self._base_dir, self._settings)

    def wants_new_session(self) -> bool:
        """True nếu người dùng bấm nút New Session."""
        return self._new_session
