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


class UI2SectionsMixin:
    def _load_saved_sections(self) -> None:
        """
        Đọc lại ui2/sections.csv + ui2/polylines.json (nếu tồn tại) và vẽ lại các tuyến lên map + bảng.
        """
        if not self.run_dir:
            return

        self._clear_sections_state()
        result = self._backend.load_sections(self.run_dir)
        rows = result.get("rows", [])
        migrated = bool(result.get("migrated", False))
        csv_path = str(result.get("csv_path", ""))
        poly_result = self._backend.load_polylines(self.run_dir)
        poly_rows = list(poly_result.get("rows", []) or [])
        poly_path = str(poly_result.get("json_path", ""))
        if not rows and not poly_rows and not os.path.isfile(csv_path) and not os.path.isfile(poly_path):
            self._info("[UI2] No saved sections.csv or polylines.json – start with empty sections.")
            return

        # Append từng section vào bảng + vẽ line
        count = 0
        for row in rows:
            try:
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
            except Exception:
                continue

            meta = {
                "line_id": str(row.get("line_id", "")).strip(),
                "line_role": str(row.get("line_role", "")).strip(),
            }
            if not meta["line_id"]:
                meta["line_id"] = str(row.get("name", "")).strip()
            self._append_section((x1, y1), (x2, y2), meta=meta)
            count += 1

        for row in poly_rows:
            vertices = [tuple(map(float, v)) for v in list(row.get("vertices", []) or [])]
            meta = {
                "line_id": str(row.get("line_id", "")).strip(),
                "line_role": str(row.get("line_role", "")).strip(),
            }
            if vertices:
                self._append_polyline(vertices, meta=meta)
                count += 1

        if migrated:
            self._ok("[UI2] Migrated legacy sections.csv to current direction version and cleared old UI3 outputs.")
        self._ok(f"[UI2] Loaded {count} sections from saved UI2 files")

    def _on_draw_mode_changed(self, _index: int) -> None:
        mode = "straight"
        if hasattr(self, "combo_draw_mode"):
            mode = str(self.combo_draw_mode.currentData() or "straight")
        self.viewer.set_draw_mode(mode)
        self._update_section_table_visibility()
        self._info(f"[UI2] Draw mode: {mode}")

    def _update_section_table_visibility(self) -> None:
        mode = "straight"
        if hasattr(self, "combo_draw_mode"):
            mode = str(self.combo_draw_mode.currentData() or "straight")
        show_polyline = mode == "polyline"
        if hasattr(self, "lbl_straight_sections"):
            self.lbl_straight_sections.setVisible(not show_polyline)
        if hasattr(self, "tbl"):
            self.tbl.setVisible(not show_polyline)
        if hasattr(self, "lbl_polyline_sections"):
            self.lbl_polyline_sections.setVisible(show_polyline)
        if hasattr(self, "tbl_poly"):
            self.tbl_poly.setVisible(show_polyline)

    @staticmethod
    def _polyline_length(vertices: List[Tuple[float, float]]) -> float:
        if len(vertices) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(vertices)):
            total += float(math.hypot(vertices[i][0] - vertices[i - 1][0], vertices[i][1] - vertices[i - 1][1]))
        return total

    @staticmethod
    def _section_type(meta: Optional[Dict[str, Any]]) -> str:
        return "polyline" if str((meta or {}).get("section_type", "")).strip().lower() == "polyline" else "straight"

    def _section_vertices(self, row: int) -> List[Tuple[float, float]]:
        if not (0 <= row < len(self._sections)):
            return []
        geom = self._sections[row]
        if isinstance(geom, list):
            out: List[Tuple[float, float]] = []
            for vertex in geom:
                try:
                    out.append((float(vertex[0]), float(vertex[1])))
                except Exception:
                    continue
            return out
        if isinstance(geom, tuple) and len(geom) == 2:
            try:
                p0 = (float(geom[0][0]), float(geom[0][1]))
                p1 = (float(geom[1][0]), float(geom[1][1]))
                return [p0, p1]
            except Exception:
                return []
        return []

    def _polyline_vertices(self, row: int) -> List[Tuple[float, float]]:
        if not (0 <= row < len(self._poly_sections)):
            return []
        out: List[Tuple[float, float]] = []
        for vertex in self._poly_sections[row]:
            try:
                out.append((float(vertex[0]), float(vertex[1])))
            except Exception:
                continue
        return out

    @staticmethod
    def _format_polyline_vertices(vertices: List[Tuple[float, float]]) -> str:
        return "; ".join(f"{float(x):.2f}, {float(y):.2f}" for x, y in vertices)

    @staticmethod
    def _parse_polyline_vertices(text: str) -> List[Tuple[float, float]]:
        raw = str(text or "").strip()
        if not raw:
            raise ValueError("Vertices cannot be empty.")
        vertices: List[Tuple[float, float]] = []
        for chunk in raw.split(";"):
            part = chunk.strip()
            if not part:
                continue
            coords = [s.strip() for s in part.split(",")]
            if len(coords) != 2:
                raise ValueError("Each vertex must be in 'x, y' format.")
            x = float(coords[0])
            y = float(coords[1])
            vertices.append((x, y))
        if len(vertices) < 2:
            raise ValueError("Polyline must contain at least 2 vertices.")
        return vertices

    def _current_selection(self) -> Tuple[Optional[str], int]:
        if hasattr(self, "tbl") and self.tbl.currentRow() >= 0 and self.tbl.hasFocus():
            return "straight", int(self.tbl.currentRow())
        if hasattr(self, "tbl_poly") and self.tbl_poly.currentRow() >= 0 and self.tbl_poly.hasFocus():
            return "polyline", int(self.tbl_poly.currentRow())
        if hasattr(self, "tbl_poly") and self.tbl_poly.currentRow() >= 0 and self.tbl_poly.selectedIndexes():
            return "polyline", int(self.tbl_poly.currentRow())
        if hasattr(self, "tbl") and self.tbl.currentRow() >= 0 and self.tbl.selectedIndexes():
            return "straight", int(self.tbl.currentRow())
        return None, -1

    def _section_display_type(self, meta: Dict[str, Any], vertices: List[Tuple[float, float]]) -> str:
        if self._section_type(meta) == "polyline":
            return f"Polyline ({len(vertices)} pts)"
        return "Straight"

    def _build_scene_path(self, vertices: List[Tuple[float, float]]) -> Optional[QPainterPath]:
        if self._inv_tr is None or len(vertices) < 2:
            return None
        c0, r0 = self._inv_tr * vertices[0]
        path = QPainterPath(QPointF(c0, r0))
        for pt in vertices[1:]:
            c, r = self._inv_tr * pt
            path.lineTo(c, r)
        return path

    def _label_anchor_pixels(self, vertices: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
        if self._inv_tr is None or len(vertices) < 2:
            return None
        c0, r0 = self._inv_tr * vertices[0]
        c1, r1 = self._inv_tr * vertices[1]
        return float(c0), float(r0), float(c1), float(r1)

    def _render_section_item(self, vertices: List[Tuple[float, float]], label_text: str) -> Tuple[Optional[object], Optional[QGraphicsSimpleTextItem]]:
        if self._inv_tr is None or len(vertices) < 2:
            return None, None
        is_polyline = len(vertices) > 2
        pen = QPen(QColor(220, 40, 40, 220) if is_polyline else QColor(30, 200, 30, 200))
        pen.setCosmetic(True)
        pen.setWidth(2)
        if len(vertices) == 2:
            c0, r0 = self._inv_tr * vertices[0]
            c1, r1 = self._inv_tr * vertices[1]
            item = QGraphicsLineItem(c0, r0, c1, r1)
            item.setPen(pen)
            item.setZValue(3)
            self.viewer.scene.addItem(item)
        else:
            path = self._build_scene_path(vertices)
            if path is None:
                return None, None
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setZValue(3)
            self.viewer.scene.addItem(item)
        label_item = None
        label_anchor = self._label_anchor_pixels(vertices)
        if label_anchor is not None:
            label_item = self._add_line_label(label_text, *label_anchor, z=5)
        return item, label_item

    def _sync_polyline_row_geometry(self, row: int, vertices: List[Tuple[float, float]]) -> None:
        if not (0 <= row < self.tbl_poly.rowCount()):
            return
        if len(vertices) < 2:
            return
        if 0 <= row < len(self._poly_sections):
            self._poly_sections[row] = list(vertices)
        while len(self._poly_section_meta) <= row:
            self._poly_section_meta.append({})
        if not isinstance(self._poly_section_meta[row], dict):
            self._poly_section_meta[row] = {}
        self._poly_section_meta[row]["vertex_count"] = int(len(vertices))
        self._poly_section_meta[row]["length_m"] = float(self._polyline_length(vertices))

        start_text = f"{vertices[0][0]:.2f}, {vertices[0][1]:.2f}"
        end_text = f"{vertices[-1][0]:.2f}, {vertices[-1][1]:.2f}"
        points_text = str(len(vertices))
        vertices_text = self._format_polyline_vertices(vertices)

        self._updating_table = True
        try:
            start_item = self.tbl_poly.item(row, 1)
            if start_item is None:
                start_item = QTableWidgetItem(start_text)
                start_item.setFlags(start_item.flags() & ~Qt.ItemIsEditable)
                self.tbl_poly.setItem(row, 1, start_item)
            else:
                start_item.setText(start_text)

            end_item = self.tbl_poly.item(row, 2)
            if end_item is None:
                end_item = QTableWidgetItem(end_text)
                end_item.setFlags(end_item.flags() & ~Qt.ItemIsEditable)
                self.tbl_poly.setItem(row, 2, end_item)
            else:
                end_item.setText(end_text)

            points_item = self.tbl_poly.item(row, 3)
            if points_item is None:
                points_item = QTableWidgetItem(points_text)
                points_item.setFlags(points_item.flags() & ~Qt.ItemIsEditable)
                self.tbl_poly.setItem(row, 3, points_item)
            else:
                points_item.setText(points_text)

            vertices_item = self.tbl_poly.item(row, 4)
            if vertices_item is None:
                vertices_item = QTableWidgetItem(vertices_text)
                self.tbl_poly.setItem(row, 4, vertices_item)
            else:
                vertices_item.setText(vertices_text)
        finally:
            self._updating_table = False

        label_text = self.tbl_poly.item(row, 0).text().strip() if self.tbl_poly.item(row, 0) else f"{row + 1}"
        new_item, new_label = self._render_section_item(vertices, label_text)
        if 0 <= row < len(self._poly_section_lines):
            old_item = self._poly_section_lines[row]
            if old_item is not None:
                self.viewer.scene.removeItem(old_item)
            self._poly_section_lines[row] = new_item
        if 0 <= row < len(self._poly_section_line_labels):
            old_label = self._poly_section_line_labels[row]
            if old_label is not None:
                self.viewer.scene.removeItem(old_label)
            self._poly_section_line_labels[row] = new_label

        if self._preview_source == "polyline" and self.tbl_poly.currentRow() == row:
            if self._preview_line is not None:
                self.viewer.scene.removeItem(self._preview_line)
                self._preview_line = None
            if self._preview_label is not None:
                self.viewer.scene.removeItem(self._preview_label)
                self._preview_label = None
            pen = QPen(QColor(200, 30, 30, 220))
            pen.setCosmetic(True)
            pen.setWidth(2)
            if len(vertices) == 2 and self._inv_tr is not None:
                c0, r0 = self._inv_tr * vertices[0]
                c1, r1 = self._inv_tr * vertices[1]
                item = QGraphicsLineItem(c0, r0, c1, r1)
                item.setPen(pen)
                item.setZValue(4)
                self.viewer.scene.addItem(item)
                self._preview_line = item
            else:
                path = self._build_scene_path(vertices)
                if path is not None:
                    item = QGraphicsPathItem(path)
                    item.setPen(pen)
                    item.setZValue(4)
                    self.viewer.scene.addItem(item)
                    self._preview_line = item
            anchor = self._label_anchor_pixels(vertices)
            if anchor is not None:
                self._preview_label = self._add_line_label(label_text, *anchor, z=6)

    def _on_section_picked(self, x1: float, y1: float, x2: float, y2: float) -> None:
        p0, p1 = self._reverse_section_points((x1, y1), (x2, y2))
        self._append_section(p0, p1)
        self._ok(f"Section added: ({x1:.2f},{y1:.2f}) → ({x2:.2f},{y2:.2f})")

    def _on_polyline_picked(self, points: object) -> None:
        vertices: List[Tuple[float, float]] = []
        for pt in list(points or []):
            try:
                vertices.append((float(pt[0]), float(pt[1])))
            except Exception:
                continue
        if len(vertices) < 2:
            self._warn("[UI2] Polyline needs at least 2 valid points.")
            return
        self._append_polyline(vertices)
        self._ok(f"[UI2] Polyline added: {len(vertices)} points, length={self._polyline_length(vertices):.2f} m")

    @staticmethod
    def _reverse_section_points(
        p0: Tuple[float, float],
        p1: Tuple[float, float],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return storage_reverse_section_points(p0, p1)

    def _append_section(
            self,
            p0: Tuple[float, float],
            p1: Tuple[float, float],
            label: Optional[str] = None,
            meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Lưu section vào bảng + vẽ line cố định trên map."""
        # tránh trigger _on_table_item_changed khi đang chèn row
        self._updating_table = True

        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        hdr = self.tbl.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, hdr.Fixed)
        hdr.setSectionResizeMode(1, hdr.Stretch)
        hdr.setSectionResizeMode(2, hdr.Stretch)
        hdr.setSectionResizeMode(3, hdr.Fixed)

        meta_row = dict(meta or {})
        role_txt = self._role_combo_text_from_value(str(meta_row.get("line_role", "") or ""), str(meta_row.get("line_id", "") or ""))
        meta_row["line_role"] = self._role_value_from_combo_text(role_txt)
        line_id = str(meta_row.get("line_id", "") or "").strip()
        if not line_id:
            line_id = self._next_line_id_for_role(meta_row["line_role"])
        meta_row["line_id"] = line_id
        line_label = label if label is not None else line_id

        self.tbl.setItem(r, 0, QTableWidgetItem(line_label))
        self.tbl.setItem(r, 1, QTableWidgetItem(f"{p0[0]:.2f}, {p0[1]:.2f}"))
        self.tbl.setItem(r, 2, QTableWidgetItem(f"{p1[0]:.2f}, {p1[1]:.2f}"))
        role_combo = _NoWheelComboBox(self.tbl)
        role_combo.addItems(["Main", "Cross"])
        role_combo.setCurrentText(role_txt)
        self.tbl.setCellWidget(r, 3, role_combo)
        self.tbl.verticalHeader().setDefaultSectionSize(30)  # chiều cao mỗi row
        self.tbl.setColumnWidth(0, 56)
        self.tbl.setColumnWidth(3, 100)

        self._updating_table = False
        # cột cuối cùng đang stretch, giữ nguyên

        # lưu section
        self._sections.append((p0, p1))
        self._section_meta.append(meta_row)
        role_combo.currentIndexChanged.connect(self._on_role_combo_changed)

        line_item, label_item = self._render_section_item([p0, p1], line_label)
        self._section_lines.append(line_item)
        self._section_line_labels.append(label_item)
        self._ok("Section line drawn on map.")

    def _append_polyline(
            self,
            vertices: List[Tuple[float, float]],
            label: Optional[str] = None,
            meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(vertices) < 2:
            return
        self._updating_table = True
        r = self.tbl_poly.rowCount()
        self.tbl_poly.insertRow(r)
        hdr = self.tbl_poly.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, hdr.Fixed)
        hdr.setSectionResizeMode(1, hdr.Stretch)
        hdr.setSectionResizeMode(2, hdr.Stretch)
        hdr.setSectionResizeMode(3, hdr.Fixed)
        hdr.setSectionResizeMode(4, hdr.Stretch)
        hdr.setSectionResizeMode(5, hdr.Fixed)

        meta_row = dict(meta or {})
        meta_row["vertex_count"] = int(len(vertices))
        meta_row["length_m"] = float(self._polyline_length(vertices))
        role_txt = self._role_combo_text_from_value(str(meta_row.get("line_role", "") or ""), str(meta_row.get("line_id", "") or ""))
        meta_row["line_role"] = self._role_value_from_combo_text(role_txt)
        line_id = str(meta_row.get("line_id", "") or "").strip()
        if not line_id:
            line_id = self._next_line_id_for_role(meta_row["line_role"])
        meta_row["line_id"] = line_id
        line_label = label if label is not None else line_id
        p0 = vertices[0]
        p1 = vertices[-1]

        id_item = QTableWidgetItem(line_label)
        self.tbl_poly.setItem(r, 0, id_item)
        start_item = QTableWidgetItem(f"{p0[0]:.2f}, {p0[1]:.2f}")
        end_item = QTableWidgetItem(f"{p1[0]:.2f}, {p1[1]:.2f}")
        for item in (start_item, end_item):
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.tbl_poly.setItem(r, 1, start_item)
        self.tbl_poly.setItem(r, 2, end_item)
        points_item = QTableWidgetItem(str(len(vertices)))
        points_item.setFlags(points_item.flags() & ~Qt.ItemIsEditable)
        self.tbl_poly.setItem(r, 3, points_item)
        self.tbl_poly.setItem(r, 4, QTableWidgetItem(self._format_polyline_vertices(vertices)))
        role_combo = _NoWheelComboBox(self.tbl_poly)
        role_combo.addItems(["Main", "Cross"])
        role_combo.setCurrentText(role_txt)
        self.tbl_poly.setCellWidget(r, 5, role_combo)
        self.tbl_poly.verticalHeader().setDefaultSectionSize(30)
        self.tbl_poly.setColumnWidth(0, 56)
        self.tbl_poly.setColumnWidth(3, 70)
        self.tbl_poly.setColumnWidth(5, 100)
        self._updating_table = False

        self._poly_sections.append(list(vertices))
        self._poly_section_meta.append(meta_row)
        role_combo.currentIndexChanged.connect(self._on_role_combo_changed)

        path_item, label_item = self._render_section_item(vertices, line_label)
        self._poly_section_lines.append(path_item)
        self._poly_section_line_labels.append(label_item)
        self._ok("Polyline drawn on map.")

    def _delete_section_row(self, source: str, row: int, log_msg: Optional[str] = None) -> bool:
        table = self.tbl_poly if source == "polyline" else self.tbl
        geom_list = self._poly_sections if source == "polyline" else self._sections
        meta_list = self._poly_section_meta if source == "polyline" else self._section_meta
        item_list = self._poly_section_lines if source == "polyline" else self._section_lines
        label_list = self._poly_section_line_labels if source == "polyline" else self._section_line_labels
        if row < 0 or row >= table.rowCount():
            return False

        # remove map line
        if 0 <= row < len(item_list):
            it = item_list[row]
            if it is not None:
                self.viewer.scene.removeItem(it)
            item_list.pop(row)

        # remove map label
        if 0 <= row < len(label_list):
            it = label_list[row]
            if it is not None:
                self.viewer.scene.removeItem(it)
            label_list.pop(row)

        # remove section data/meta
        if 0 <= row < len(geom_list):
            geom_list.pop(row)
        if 0 <= row < len(meta_list):
            meta_list.pop(row)

        # remove preview if this row was previewed
        if self._preview_line is not None:
            try:
                self.viewer.scene.removeItem(self._preview_line)
            except Exception:
                pass
            self._preview_line = None
        if self._preview_label is not None:
            try:
                self.viewer.scene.removeItem(self._preview_label)
            except Exception:
                pass
            self._preview_label = None
            self._preview_source = None

        table.removeRow(row)
        if log_msg:
            self._ok(log_msg)
        return True

    def _on_sections_table_context_menu(self, table: QTableWidget, source: str, pos) -> None:
        if table is None:
            return
        idx = table.indexAt(pos)
        if not idx.isValid():
            item = table.itemAt(pos)
            if item is None:
                return
            row = int(item.row())
        else:
            row = int(idx.row())
        if row < 0 or row >= table.rowCount():
            return
        table.selectRow(row)
        menu = QMenu(table)
        act_delete = menu.addAction("Delete")
        chosen = menu.exec_(table.viewport().mapToGlobal(pos))
        if chosen is act_delete:
            self._delete_section_row(source, row, log_msg=f"Deleted section #{row + 1}.")

    def _on_sections_table_header_context_menu(self, table: QTableWidget, source: str, pos) -> None:
        if table is None:
            return
        row = int(table.rowAt(pos.y()))
        if row < 0 or row >= table.rowCount():
            return
        table.selectRow(row)
        menu = QMenu(table)
        act_delete = menu.addAction("Delete")
        chosen = menu.exec_(table.verticalHeader().viewport().mapToGlobal(pos))
        if chosen is act_delete:
            self._delete_section_row(source, row, log_msg=f"Deleted section #{row + 1}.")

    def _add_line_label(
            self,
            label_text: str,
            start_x: float,
            start_y: float,
            end_x: float,
            end_y: float,
            z: float = 5,
    ) -> QGraphicsSimpleTextItem:
        lbl = QGraphicsSimpleTextItem(str(label_text))
        lbl.setFont(QFont("Arial", 15))
        lbl.setBrush(QBrush(QColor(0, 170, 0)))

        # Offset label away from the line (perpendicular direction)
        dx = end_x - start_x
        dy = end_y - start_y
        length = (dx * dx + dy * dy) ** 0.5
        if length > 1e-6:
            nx = -dy / length
            ny = dx / length
        else:
            nx, ny = 0.0, -1.0
        offset = 10.0

        br = lbl.boundingRect()
        x = start_x - br.width() * 0.5 + nx * offset
        y = start_y - br.height() * 0.5 + ny * offset
        lbl.setPos(x, y)
        lbl.setZValue(z)
        self.viewer.scene.addItem(lbl)
        return lbl

    def _on_table_item_changed(self, item) -> None:
        """
        Khi user sửa Start/End (x,y) trong bảng, cập nhật ngay line tương ứng trên viewer.
        """
        if self._updating_table:
            return  # bỏ qua khi đang cập nhật bằng code

        if item is None:
            return

        row = item.row()
        col = item.column()

        # chỉ quan tâm cột 1 (Start), 2 (End), 0 (Label)
        if col not in (0, 1, 2):
            return

        # lấy cả start & end của dòng hiện tại
        start_item = self.tbl.item(row, 1)
        end_item = self.tbl.item(row, 2)
        if start_item is None or end_item is None:
            return

        try:
            p0 = tuple(map(float, start_item.text().split(",")))
            p1 = tuple(map(float, end_item.text().split(",")))
            if self._inv_tr is None:
                return
            c0, r0 = self._inv_tr * p0
            c1, r1 = self._inv_tr * p1
        except Exception:
            # nếu user gõ dở dang (ví dụ "120,"), tạm bỏ qua
            return

        # cập nhật list _sections
        if 0 <= row < len(self._sections):
            self._sections[row] = (p0, p1)
        if col == 0:
            while len(self._section_meta) <= row:
                self._section_meta.append({})
            if not isinstance(self._section_meta[row], dict):
                self._section_meta[row] = {}
            self._section_meta[row]["line_id"] = (self.tbl.item(row, 0).text().strip() if self.tbl.item(row, 0) else "")

        # cập nhật line xanh trên map
        if 0 <= row < len(self._section_lines):
            line_item = self._section_lines[row]
            if isinstance(line_item, QGraphicsLineItem):
                line_item.setLine(c0, r0, c1, r1)

        # cập nhật label (line number)
        if 0 <= row < len(self._section_line_labels):
            lbl = self._section_line_labels[row]
            if lbl is not None:
                label_item = self.tbl.item(row, 0)
                lbl.setText(label_item.text() if label_item else str(row + 1))
                br = lbl.boundingRect()
                dx = c1 - c0
                dy = r1 - r0
                length = (dx * dx + dy * dy) ** 0.5
                if length > 1e-6:
                    nx = -dy / length
                    ny = dx / length
                else:
                    nx, ny = 0.0, -1.0
                offset = 10.0
                lbl.setPos(
                    c0 - br.width() * 0.5 + nx * offset,
                    r0 - br.height() * 0.5 + ny * offset,
                )

        # nếu đang có preview line cho đúng dòng này, cập nhật luôn
        if self._preview_line is not None and self.tbl.currentRow() == row:
            self._preview_line.setLine(c0, r0, c1, r1)
            if self._preview_label is not None:
                label_item = self.tbl.item(row, 0)
                self._preview_label.setText(label_item.text() if label_item else str(row + 1))
                br = self._preview_label.boundingRect()
                dx = c1 - c0
                dy = r1 - r0
                length = (dx * dx + dy * dy) ** 0.5
                if length > 1e-6:
                    nx = -dy / length
                    ny = dx / length
                else:
                    nx, ny = 0.0, -1.0
                offset = 10.0
                self._preview_label.setPos(
                    c0 - br.width() * 0.5 + nx * offset,
                    r0 - br.height() * 0.5 + ny * offset,
                )

        self._ok(f"Updated section #{row + 1} from table edit.")

    def _on_polyline_table_item_changed(self, item) -> None:
        if self._updating_table or item is None:
            return
        row = item.row()
        col = item.column()
        if col not in (0, 4):
            return
        if col == 4:
            try:
                vertices = self._parse_polyline_vertices(item.text())
            except Exception:
                return
            self._sync_polyline_row_geometry(row, vertices)
        if 0 <= row < len(self._poly_section_meta) and isinstance(self._poly_section_meta[row], dict):
            self._poly_section_meta[row]["line_id"] = (self.tbl_poly.item(row, 0).text().strip() if self.tbl_poly.item(row, 0) else "")
        if 0 <= row < len(self._poly_section_line_labels):
            lbl = self._poly_section_line_labels[row]
            if lbl is not None:
                lbl.setText(self.tbl_poly.item(row, 0).text().strip() if self.tbl_poly.item(row, 0) else f"{row + 1}")
        if self._preview_label is not None and self._preview_source == "polyline" and self.tbl_poly.currentRow() == row:
            self._preview_label.setText(self.tbl_poly.item(row, 0).text().strip() if self.tbl_poly.item(row, 0) else f"{row + 1}")
        self._ok(f"Updated polyline #{row + 1} from table edit.")
