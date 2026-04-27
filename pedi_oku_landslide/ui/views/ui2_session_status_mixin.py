import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import LineString, Point
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsSimpleTextItem,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pedi_oku_landslide.pipeline.runners.ui2.ui2_intersections import (
    line_order_key as backend_line_order_key,
    normalize_line_role as backend_normalize_line_role,
    parse_auto_line_id as backend_parse_auto_line_id,
    pick_intersection_point as backend_pick_intersection_point,
)
from pedi_oku_landslide.pipeline.runners.ui2.ui2_paths import file_exists as _file_exists
from pedi_oku_landslide.pipeline.runners.ui2.ui2_sections_storage import (
    canonical_section_csv_row as storage_canonical_section_csv_row,
    reverse_section_points as storage_reverse_section_points,
)
from pedi_oku_landslide.ui.dialogs.ui2_dialogs import AutoLineDialog
from .ui2_layered_viewer import _LayeredViewer
from pedi_oku_landslide.ui.widgets.ui2_widgets import HBox, NoWheelComboBox as _NoWheelComboBox
from pedi_oku_landslide.ui.layout_constants import (
    LEFT_MARGINS,
    PANEL_SPACING,
    RIGHT_MARGINS,
    RIGHT_MIN_W,
    STATUS_PANEL_H,
    CONTROL_HEIGHT,
    PROJECT_H_SPACING,
    PROJECT_LABEL_W,
    PROJECT_MARGINS,
    PROJECT_V_SPACING,
    ROOT_MARGINS,
    ROOT_SPACING,
)


class UI2SessionStatusMixin:
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and getattr(self.viewer, "is_polyline_active", lambda: False)():
            if self.viewer.finish_polyline_pick():
                self._ok("[UI2] Polyline completed.")
                event.accept()
                return
        if event.key() == Qt.Key_Delete:
            source, _ = self._current_selection()
            table = self.tbl_poly if source == "polyline" else self.tbl
            sel = table.selectedIndexes() if source else []
            if not sel:
                return

            rows = sorted(set(i.row() for i in sel))

            # nếu chọn ít nhất 1 cột Start/End → coi như xóa cả dòng
            if len(rows) >= 1:
                for r in reversed(rows):
                    self._delete_section_row(str(source or "straight"), r)
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
        self._backend.reset_context()
        self._ctx_ready = False
        self._dx = None
        self._dy = None
        self._tr = None
        self._inv_tr = None
        self._dz = None
        self._mask = None

        # reset vector params về mặc định
        self._vec_step = 25
        self._vec_scale = 1.0
        self._vec_size_pct = 100
        self._vec_opacity_pct = 100
        self._vec_color = "blue"
        if hasattr(self, "spin_vec_step"):
            self.spin_vec_step.setValue(25)
        if hasattr(self, "spin_vec_scale"):
            self.spin_vec_scale.setValue(1.0)
        if hasattr(self, "combo_vec_color"):
            self.combo_vec_color.setCurrentText("Blue")
        if hasattr(self, "sld_vec_size"):
            self.sld_vec_size.setValue(100)
        if hasattr(self, "sld_vec_opacity"):
            self.sld_vec_opacity.setValue(100)
        if hasattr(self, "combo_draw_mode"):
            self.combo_draw_mode.setCurrentIndex(0)
        self.viewer.set_draw_mode("straight")
        self._update_section_table_visibility()

        # Xoá sections & line
        self._clear_sections_state()

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
    @staticmethod
    def _status_brief(msg: str, fallback: str) -> str:
        skip_prefixes = ("project:", "run:", "output:", "folder:", "paths:")
        for raw in str(msg or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if any(low.startswith(prefix) for prefix in skip_prefixes):
                continue
            if "\\" in line or "/" in line:
                if ":" in line:
                    line = line.split(":", 1)[0].strip()
                else:
                    continue
            return line
        return fallback

    def _append_status(self, text: str) -> None:
        self.status_text.appendPlainText(text)
        self.status_text.moveCursor(self.status_text.textCursor().End)

    def _info(self, msg: str) -> None:
        return

    def _ok(self, msg: str) -> None:
        self._append_status(f"[UI2] OK: {self._status_brief(msg, 'Completed.')}")

    def _warn(self, msg: str) -> None:
        self._append_status(f"[UI2] ERROR: {self._status_brief(msg, 'Action required.')}")

    def _err(self, msg: str) -> None:
        """Append error to status + popup."""
        self._append_status(f"[UI2] ERROR: {self._status_brief(msg, 'Error.')}")
        try:
            QMessageBox.critical(self, "Section Selection", msg)
        except Exception:
            pass  # phòng trường hợp gọi khi app chưa sẵn sàng

    # Back-compat: vài nơi cũ còn gọi self._log(...)
    def _log(self, msg: str) -> None:
        return
