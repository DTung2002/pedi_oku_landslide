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


class UI2ConfirmMixin:
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
            rows_to_save: List[Dict[str, Any]] = []
            polylines_to_save: List[Dict[str, Any]] = []
            for r in range(self.tbl.rowCount()):
                meta = self._section_meta[r] if 0 <= r < len(self._section_meta) else {}
                line_id = str((meta or {}).get("line_id", "")).strip()
                if not line_id:
                    label = (self.tbl.item(r, 0).text().strip() if self.tbl.item(r, 0) else f"{r + 1}")
                    line_id = label
                    if 0 <= r < len(self._section_meta) and isinstance(self._section_meta[r], dict):
                        self._section_meta[r]["line_id"] = line_id
                line_role = self._normalize_line_role(self._get_row_line_role("straight", r), line_id)
                if 0 <= r < len(self._section_meta) and isinstance(self._section_meta[r], dict):
                    self._section_meta[r]["line_role"] = line_role
                p0 = tuple(map(float, self.tbl.item(r, 1).text().split(",")))
                p1 = tuple(map(float, self.tbl.item(r, 2).text().split(",")))
                rows_to_save.append(storage_canonical_section_csv_row(
                    len(rows_to_save) + 1,
                    p0,
                    p1,
                    line_id=line_id,
                    line_role=line_role,
                ))
            for r in range(self.tbl_poly.rowCount()):
                meta = self._poly_section_meta[r] if 0 <= r < len(self._poly_section_meta) else {}
                line_id = str((meta or {}).get("line_id", "")).strip()
                if not line_id:
                    label = (self.tbl_poly.item(r, 0).text().strip() if self.tbl_poly.item(r, 0) else f"{r + 1}")
                    line_id = label
                    if 0 <= r < len(self._poly_section_meta) and isinstance(self._poly_section_meta[r], dict):
                        self._poly_section_meta[r]["line_id"] = line_id
                line_role = self._normalize_line_role(self._get_row_line_role("polyline", r), line_id)
                if 0 <= r < len(self._poly_section_meta) and isinstance(self._poly_section_meta[r], dict):
                    self._poly_section_meta[r]["line_role"] = line_role
                vertices = self._polyline_vertices(r)
                if len(vertices) >= 2:
                    polylines_to_save.append({
                        "idx": int(len(polylines_to_save) + 1),
                        "line_id": line_id,
                        "line_role": line_role,
                        "vertices": [[float(x), float(y)] for x, y in vertices],
                    })
            self._backend.save_sections(run_dir, rows_to_save)
            poly_path = self._backend.save_polylines(run_dir, polylines_to_save)

            self._ok(f"Sections saved: {csv_path}")
            self._ok(f"Polylines saved: {poly_path}")
            try:
                inter_path = self._save_main_cross_intersections(ui2_dir)
                if inter_path:
                    self._ok(f"[UI2] Intersections saved: {inter_path}")
            except Exception as e:
                self._err(f"[UI2] Save intersections error: {e}")

        except Exception as e:
            self._err(f"Confirm Sections error: {e}")
            return

        # 👉 Emit DUY NHẤT 1 lần, dùng context chuẩn
        try:
            self.sections_confirmed.emit(project, run_label, run_dir)
            self._log(f"[UI2] sections_confirmed emitted: {project}, {run_label}, {run_dir}")
        except Exception as e:
            self._err(f"Emit sections_confirmed error: {e}")
