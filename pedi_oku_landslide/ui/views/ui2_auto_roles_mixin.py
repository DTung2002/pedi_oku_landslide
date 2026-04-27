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


class UI2AutoRolesMixin:
    def _on_auto_lines(self) -> None:
        """
        Handler cho nút 'Auto Line Generation'.

        Bản này dùng trực tiếp dx/dy/mask đã load trong SectionSelectionTab
        (self._dx, self._dy, self._mask, self._tr) và gọi
        generate_auto_lines_from_arrays(...) để đảm bảo hướng tuyến
        giống vector dXY ở UI1.
        """
        # 0) Kiểm tra context
        if self.run_dir is None or self._tr is None:
            self._info("[UI2] Auto line: run context not ready. Please run Analyze first.")
            return

        if self._dx is None or self._dy is None or self._mask is None:
            self._info("[UI2] Auto line: dx/dy/mask not ready. Please run Analyze first.")
            return

        # 1) Hiện dialog nhập tham số
        dlg = AutoLineDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return  # người dùng Cancel

        main_num_even = int(dlg.main_num.value())
        main_offset_m = float(dlg.main_off.value())
        cross_num_even = int(dlg.cross_num.value())
        cross_offset_m = float(dlg.cross_off.value())

        # 2) Ép số chẵn + kiểm tra offset >= 0 (giống UI2 cũ)
        if main_num_even % 2 == 1:
            main_num_even -= 1
            self._info("[UI2] Main lines: số tuyến phải là số chẵn – đã tự động giảm 1.")

        if cross_num_even % 2 == 1:
            cross_num_even -= 1
            self._info("[UI2] Cross lines: số tuyến phải là số chẵn – đã tự động giảm 1.")

        if main_offset_m < 0 or cross_offset_m < 0:
            self._info("[UI2] Auto line: Offset must be ≥ 0.")
            return

        # 3) Gọi backend dạng mảng – không phụ thuộc tên file detect_mask.tif nữa
        try:
            outs = self._backend.generate_auto_lines(
                dx=self._dx,
                dy=self._dy,
                mask=self._mask,
                transform=self._tr,
                params={
                    "main_num_even": main_num_even,
                    "main_offset_m": main_offset_m,
                    "cross_num_even": cross_num_even,
                    "cross_offset_m": cross_offset_m,
                    "base_length_m": None,
                },
            )
        except Exception as e:
            self._info(f"[UI2] Auto line: generation failed: {e}")
            return

        mains = outs.get("main", [])
        crosses = outs.get("cross", [])
        if not mains and not crosses:
            self._info("[UI2] Auto line: no lines returned.")
            return

        # redraw vectors after auto line generation
        self._load_dx_dy_and_draw(self._ui1_dir, step=self._vec_step, scale=self._vec_scale)
        debug = outs.get("debug", {})
        ang = debug.get("ang_main_deg", None)
        if ang is not None:
            self._ok(f"[UI2] Main direction ≈ {ang:.1f} deg")

        # 4) Xoá tất cả section cũ (bảng + line trên map)
        self._clear_sections_state()

        # 5) Helper: convert feat (LineString) -> 1 dòng trong bảng + vẽ line
        def _add_feat(feat: dict) -> None:
            geom = feat.get("geom")
            if not isinstance(geom, LineString) or geom.is_empty:
                return
            x1, y1 = geom.coords[0]
            x2, y2 = geom.coords[-1]
            p0, p1 = self._reverse_section_points((x1, y1), (x2, y2))
            self._append_section(
                p0, p1,
                meta={
                    "line_id": str(feat.get("name", "")).strip(),
                    "line_role": str(feat.get("type", "")).strip(),
                    "offset_m": float(feat.get("offset_m", 0.0)) if feat.get("offset_m", None) is not None else None,
                    "angle_deg": float(feat.get("angle_deg", 0.0)) if feat.get("angle_deg", None) is not None else None,
                }
            )

        for f in mains:
            _add_feat(f)
        for f in crosses:
            _add_feat(f)

        self._ok(f"[UI2] Auto line OK: {len(mains)} main + {len(crosses)} cross lines.")

    def _on_preview(self) -> None:
        source, r = self._current_selection()
        if source is None or r < 0 or self._inv_tr is None:
            self._info("Select a row to preview.")
            return

        # Xoá preview cũ nếu có
        if self._preview_line is not None:
            self.viewer.scene.removeItem(self._preview_line)
            self._preview_line = None
        if self._preview_label is not None:
            self.viewer.scene.removeItem(self._preview_label)
            self._preview_label = None
        self._preview_source = source

        pen = QPen(QColor(200, 30, 30, 220))
        pen.setCosmetic(True)
        pen.setWidth(2)
        vertices = self._polyline_vertices(r) if source == "polyline" else self._section_vertices(r)
        if len(vertices) < 2:
            self._info("Selected row has invalid geometry.")
            return
        if len(vertices) == 2:
            c0, r0 = self._inv_tr * vertices[0]
            c1, r1 = self._inv_tr * vertices[1]
            item = QGraphicsLineItem(c0, r0, c1, r1)
            item.setPen(pen)
            item.setZValue(4)
            self.viewer.scene.addItem(item)
            self._preview_line = item
        else:
            path = self._build_scene_path(vertices)
            if path is None:
                return
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setZValue(4)
            self.viewer.scene.addItem(item)
            self._preview_line = item
        label_item = self.tbl_poly.item(r, 0) if source == "polyline" else self.tbl.item(r, 0)
        anchor = self._label_anchor_pixels(vertices)
        if anchor is not None:
            self._preview_label = self._add_line_label(
                label_item.text() if label_item else str(r + 1),
                *anchor,
                z=6,
            )

        self._ok("Preview line drawn.")

    def _clear_sections_state(self) -> None:
        if hasattr(self, "tbl"):
            self.tbl.setRowCount(0)
        if hasattr(self, "tbl_poly"):
            self.tbl_poly.setRowCount(0)
        self._sections.clear()
        self._section_meta.clear()
        self._poly_sections.clear()
        self._poly_section_meta.clear()

        for it in getattr(self, "_section_lines", []):
            if it is not None:
                self.viewer.scene.removeItem(it)
        self._section_lines.clear()

        for it in getattr(self, "_section_line_labels", []):
            if it is not None:
                self.viewer.scene.removeItem(it)
        self._section_line_labels.clear()

        for it in getattr(self, "_poly_section_lines", []):
            if it is not None:
                self.viewer.scene.removeItem(it)
        self._poly_section_lines.clear()

        for it in getattr(self, "_poly_section_line_labels", []):
            if it is not None:
                self.viewer.scene.removeItem(it)
        self._poly_section_line_labels.clear()

        if getattr(self, "_preview_line", None) is not None:
            self.viewer.scene.removeItem(self._preview_line)
            self._preview_line = None
        if getattr(self, "_preview_label", None) is not None:
            self.viewer.scene.removeItem(self._preview_label)
            self._preview_label = None
        self._preview_source = None

    def _on_clear(self) -> None:
        """Xoá toàn bộ sections + line trên map."""
        reply = QMessageBox.question(
            self,
            "Confirm",
            "本当に消しますか？\nAre you sure you want to delete it?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._clear_sections_state()
        self._ok("Cleared sections.")

    @staticmethod
    def _normalize_line_role(line_role: str, line_id: str = "") -> str:
        return backend_normalize_line_role(line_role, line_id)

    @staticmethod
    def _role_combo_text_from_value(line_role: str, line_id: str = "") -> str:
        role = UI2AutoRolesMixin._normalize_line_role(line_role, line_id)
        return "Cross" if role == "cross" else "Main"

    @staticmethod
    def _role_value_from_combo_text(text: str) -> str:
        t = str(text or "").strip().lower()
        return "cross" if t == "cross" else "main"

    @staticmethod
    def _parse_auto_line_id(line_id: str) -> Tuple[str, int]:
        return backend_parse_auto_line_id(line_id)

    def _next_line_id_for_role(self, line_role: str, exclude_row: int = -1, exclude_source: str = "straight") -> str:
        role = self._normalize_line_role(line_role, "")
        prefix = "CL" if role == "cross" else "ML"
        used = set()
        for source, table, meta_list in (
            ("straight", self.tbl, self._section_meta),
            ("polyline", self.tbl_poly, self._poly_section_meta),
        ):
            for r in range(table.rowCount()):
                if source == exclude_source and r == exclude_row:
                    continue
                cand = ""
                if 0 <= r < len(meta_list) and isinstance(meta_list[r], dict):
                    cand = str(meta_list[r].get("line_id", "") or "").strip()
                if not cand:
                    item = table.item(r, 0)
                    cand = item.text().strip() if item else ""
                parsed_role, parsed_idx = self._parse_auto_line_id(cand)
                if parsed_role == role and parsed_idx > 0:
                    used.add(parsed_idx)
        n = 1
        while n in used:
            n += 1
        return f"{prefix}{n}"

    def _find_role_combo_row(self, combo: QComboBox) -> Tuple[Optional[str], int]:
        for r in range(self.tbl.rowCount()):
            if self.tbl.cellWidget(r, 3) is combo:
                return "straight", r
        for r in range(self.tbl_poly.rowCount()):
            if self.tbl_poly.cellWidget(r, 5) is combo:
                return "polyline", r
        return None, -1

    def _on_role_combo_changed(self, _index: int) -> None:
        combo = self.sender()
        if not isinstance(combo, QComboBox):
            return
        source, row = self._find_role_combo_row(combo)
        if source is None or row < 0:
            return
        table = self.tbl_poly if source == "polyline" else self.tbl
        meta_list = self._poly_section_meta if source == "polyline" else self._section_meta
        while len(meta_list) <= row:
            meta_list.append({})
        meta = meta_list[row]
        if not isinstance(meta, dict):
            meta = {}
            meta_list[row] = meta
        label = (table.item(row, 0).text().strip() if table.item(row, 0) else f"{row + 1}")
        cur_line_id = str(meta.get("line_id", "") or "").strip() or label
        new_role = self._role_value_from_combo_text(combo.currentText())
        auto_role, auto_idx = self._parse_auto_line_id(cur_line_id)
        if (not cur_line_id) or (auto_idx > 0 and auto_role in ("main", "cross")):
            new_line_id = self._next_line_id_for_role(new_role, exclude_row=row, exclude_source=source)
            if new_line_id != cur_line_id:
                item0 = table.item(row, 0)
                if item0 is None:
                    item0 = QTableWidgetItem(new_line_id)
                    table.setItem(row, 0, item0)
                else:
                    item0.setText(new_line_id)
                cur_line_id = new_line_id
        meta["line_id"] = cur_line_id
        meta["line_role"] = new_role
        self._ok(f"[UI2] Line role set: row {row + 1} -> {meta['line_role']}")

    def _get_row_line_role(self, source: str, row: int) -> str:
        table = self.tbl_poly if source == "polyline" else self.tbl
        combo_col = 5 if source == "polyline" else 3
        meta_list = self._poly_section_meta if source == "polyline" else self._section_meta
        combo = table.cellWidget(row, combo_col) if table is not None else None
        if isinstance(combo, QComboBox):
            return self._role_value_from_combo_text(combo.currentText())
        meta = meta_list[row] if 0 <= row < len(meta_list) else {}
        line_id = str((meta or {}).get("line_id", "")).strip()
        return self._normalize_line_role(str((meta or {}).get("line_role", "")).strip(), line_id)

    @staticmethod
    def _line_order_key(line_id: str, fallback_idx: int) -> Tuple[int, str]:
        return backend_line_order_key(line_id, fallback_idx)

    @staticmethod
    def _pick_intersection_point(geom_a: LineString, geom_b: LineString) -> Tuple[Optional[Point], str]:
        return backend_pick_intersection_point(geom_a, geom_b)

    def _save_main_cross_intersections(self, ui2_dir: str) -> Optional[str]:
        records: List[Dict[str, Any]] = []
        skipped_polyline = len(self._poly_sections)
        for r in range(self.tbl.rowCount()):
            meta = self._section_meta[r] if 0 <= r < len(self._section_meta) else {}
            try:
                p0 = tuple(map(float, self.tbl.item(r, 1).text().split(",")))
                p1 = tuple(map(float, self.tbl.item(r, 2).text().split(",")))
            except Exception:
                continue
            line_id = str((meta or {}).get("line_id", "")).strip()
            label = (self.tbl.item(r, 0).text().strip() if self.tbl.item(r, 0) else f"{r + 1}")
            if not line_id:
                line_id = label
                if 0 <= r < len(self._section_meta) and isinstance(self._section_meta[r], dict):
                    self._section_meta[r]["line_id"] = line_id
            line_role = self._normalize_line_role(self._get_row_line_role("straight", r), line_id)
            if 0 <= r < len(self._section_meta) and isinstance(self._section_meta[r], dict):
                self._section_meta[r]["line_role"] = line_role
            try:
                geom = LineString([p0, p1])
            except Exception:
                continue
            records.append({
                "row_index": int(r),
                "table_label": label,
                "line_id": line_id,
                "line_role": line_role,
                "x1": float(p0[0]),
                "y1": float(p0[1]),
                "x2": float(p1[0]),
                "y2": float(p1[1]),
                "geom": geom,
            })
        _ = ui2_dir
        if skipped_polyline > 0:
            self._info(f"[UI2] Skipped {skipped_polyline} polyline section(s) for intersections export.")
        return self._backend.save_main_cross_intersections(self.run_dir or "", records)
