from typing import Optional

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget


class UI1LayoutMixin:
    def _left_max_w(self) -> int:
        # Left pane can occupy at most 50% of current window width.
        base_w = self.width()
        if self._splitter is not None and self._splitter.width() > 0:
            base_w = self._splitter.width()
        # During early layout, widths can be tiny/unstable; defer clamping then.
        if base_w < (self._left_min_w * 2):
            return -1
        return max(self._left_min_w, int(base_w * 0.5))

    def _try_apply_initial_splitter_width(self) -> None:
        if not self._pending_init_splitter or self._splitter is None:
            return
        base_w = self._splitter.width() if self._splitter.width() > 0 else self.width()
        if base_w < (self._left_default_w * 2):
            return
        max_w = self._left_max_w()
        if max_w < 0:
            return
        init_left = max(self._left_min_w, min(self._left_default_w, max_w))
        total = max(self._splitter.width(), sum(self._splitter.sizes()), self.width(), init_left + 1)
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
        if self._pending_init_splitter:
            QTimer.singleShot(0, self._enforce_left_pane_bounds)
        self._enforce_left_pane_bounds()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._pending_init_splitter:
            QTimer.singleShot(0, self._enforce_left_pane_bounds)
            QTimer.singleShot(80, self._enforce_left_pane_bounds)
            QTimer.singleShot(180, self._enforce_left_pane_bounds)
        self._enforce_left_pane_bounds()

    def _apply_button_style(self, container: QWidget | None = None) -> None:
        """Áp style chung cho tab Analyze (nền trắng + nút xanh)."""
        target = container or self
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
        # target.setStyleSheet(style)

    def _set_project_run_locked(self, locked: bool) -> None:
        """Khóa/Mở 2 ô Project & Run label."""
        self.edit_project.setReadOnly(locked)
        self.edit_project.setStyleSheet("background:#f3f3f3;" if locked else "")
        self.edit_runlabel.setReadOnly(locked)
        self.edit_runlabel.setStyleSheet("background:#f3f3f3;" if locked else "")

    def _refresh_mask_source_ui(self, run_dir: Optional[str] = None) -> None:
        rd = (run_dir or self._last_run_dir or "").strip()
        info = self._backend.mask_source_info(rd)
        self.lbl_mask_source.setText(str(info.get("label") or "Mask source: not set"))
        dxf_path = str(info.get("dxf_path") or "").strip()
        if dxf_path:
            try:
                self.fp_mask_dxf.set_path(dxf_path)
            except Exception:
                pass
