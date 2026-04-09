import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QColorDialog, QTableWidgetItem

WORKFLOW_GROUP_MIN_LEN_M = 0.0


class UI3GroupPanelMixin:
    def _load_group_json_data(
        self,
        path: str,
        *,
        line_id: Optional[str] = None,
        apply_settings: bool = True,
    ) -> Tuple[Dict[str, Any], List[dict]]:
        lid = line_id or self._line_id_current()
        data, groups, cm = self._backend.load_group_json_data(
            path=path,
            line_id=lid,
            apply_settings=bool(apply_settings),
            apply_group_json_settings=self._apply_group_json_settings,
        )
        self._set_curve_method_for_line(lid, cm)
        return data, groups

    def _populate_group_table_rows(self, groups: List[dict], *, length_m: Optional[float]) -> int:
        self.group_table.setRowCount(0)
        loaded = 0
        for g in (groups or []):
            try:
                gid = str(g.get("group_id", g.get("id", ""))).strip()
                s = float(g.get("start", g.get("start_chainage")))
                e = float(g.get("end", g.get("end_chainage")))
            except Exception:
                continue
            if e < s:
                s, e = e, s
            r = self.group_table.rowCount()
            self.group_table.insertRow(r)
            self.group_table.setItem(r, 0, QTableWidgetItem(gid or f"G{r + 1}"))
            self.group_table.setItem(r, 1, QTableWidgetItem(f"{s:.3f}"))
            self.group_table.setItem(r, 2, QTableWidgetItem(f"{e:.3f}"))
            self._set_group_boundary_reason(r, 1, str(g.get("start_reason", "") or ""))
            self._set_group_boundary_reason(r, 2, str(g.get("end_reason", "") or ""))
            self._set_color_cell(r, str(g.get("color", "")).strip())
            loaded += 1
        if loaded:
            self._append_ungrouped_row(self._read_groups_from_table(), length_m)
        return loaded

    def _save_groups_to_ui(
        self,
        groups: list,
        prof: dict,
        line_id: str,
        log_text: Optional[str] = None,
        curve_method: Optional[str] = None,
        group_method: Optional[str] = None,
    ) -> None:
        cm = self._set_curve_method_for_line(line_id, curve_method or self._curve_method_by_line.get(line_id))
        try:
            js = self._backend.build_group_json_payload(
                line_label=self.line_combo.currentText(),
                groups=groups,
                prof=prof,
                chainage_origin=self._ui3_chainage_origin(),
                curve_method=cm,
                profile_dem_source=self._current_profile_source_key(),
                profile_dem_path=str(getattr(self, "dem_path", "") or ""),
                grouping_params=self._grouping_params_current(),
                group_method=group_method,
            )
            saved_path = self._backend.save_group_json(self._groups_json_path(), js)
            self._log(f"[✓] Saved group definition: {saved_path}")
        except Exception as e:
            self._warn(f"[UI3] Cannot save groups JSON: {e}")

        if "length_m" in prof and prof["length_m"] is not None:
            length_m = float(prof["length_m"])
        else:
            ch = prof.get("chain")
            length_m = float(ch[-1] - ch[0]) if ch is not None and len(ch) >= 2 else None

        if self.group_table is not None:
            self._group_table_updating = True
            try:
                self._populate_group_table_rows(groups, length_m=length_m)
            finally:
                self._group_table_updating = False

        bounds_set = set()
        for g in groups:
            s = float(g.get("start", 0.0))
            e = float(g.get("end", 0.0))
            if e < s:
                s, e = e, s
            if length_m:
                s = max(0.0, min(length_m, s))
                e = max(0.0, min(length_m, e))
            bounds_set.add(s)
            bounds_set.add(e)
        self._group_bounds[line_id] = sorted(bounds_set)
        self._sec_len_m = length_m

        if self._px_per_m is None and getattr(self, "_img_ground", None) and self._sec_len_m:
            w = self._img_ground.pixmap().width()
            self._px_per_m = float(w) / float(self._sec_len_m)
        if log_text:
            self._ok(log_text)

    def _set_group_chainage_cell(self, row: int, col: int, val: float) -> None:
        item = self.group_table.item(row, col)
        if item is None:
            item = QTableWidgetItem("")
            self.group_table.setItem(row, col, item)
        item.setText(f"{float(val):.3f}")

    def _set_group_boundary_reason(self, row: int, col: int, reason: str) -> None:
        if col not in (1, 2):
            return
        item = self.group_table.item(row, col)
        if item is None:
            item = QTableWidgetItem("")
            self.group_table.setItem(row, col, item)
        item.setData(Qt.UserRole + 1, str(reason or ""))

    def _get_group_boundary_reason(self, row: int, col: int) -> str:
        if col not in (1, 2):
            return ""
        item = self.group_table.item(row, col)
        if item is None:
            return ""
        try:
            return str(item.data(Qt.UserRole + 1) or "").strip()
        except Exception:
            return ""

    def _nearest_group_row(self, row: int, step: int) -> Optional[int]:
        r = int(row) + int(step)
        while 0 <= r < self.group_table.rowCount():
            gid_item = self.group_table.item(r, 0)
            gid = gid_item.text().strip().upper() if gid_item and gid_item.text() else ""
            if gid != "UNGROUPED":
                return r
            r += int(step)
        return None

    def _link_adjacent_group_boundaries(self, row: int, col: int) -> None:
        if col not in (1, 2):
            return
        gid_item = self.group_table.item(row, 0)
        gid = gid_item.text().strip().upper() if gid_item and gid_item.text() else ""
        if gid == "UNGROUPED":
            return
        cur_item = self.group_table.item(row, col)
        txt = cur_item.text().strip() if cur_item and cur_item.text() else ""
        if not txt:
            return
        try:
            val = float(txt)
        except Exception:
            return
        self._set_group_chainage_cell(row, col, val)
        if col == 2:
            nxt = self._nearest_group_row(row, +1)
            if nxt is not None:
                self._set_group_chainage_cell(nxt, 1, val)
        elif col == 1:
            prv = self._nearest_group_row(row, -1)
            if prv is not None:
                self._set_group_chainage_cell(prv, 2, val)

    def _on_group_table_item_changed(self, _item) -> None:
        if self._group_table_updating:
            return
        changed_group_chainage = bool(_item is not None and _item.column() in (1, 2))
        if _item is not None and _item.column() in (1, 2):
            self._group_table_updating = True
            try:
                self._link_adjacent_group_boundaries(_item.row(), _item.column())
            finally:
                self._group_table_updating = False
        self._sync_nurbs_defaults_from_group_table()
        if changed_group_chainage and self._active_prof:
            self._render_current_safe()

    def _on_add_group(self):
        r = self._find_ungrouped_row()
        if r is None:
            r = self.group_table.rowCount()
        self.group_table.insertRow(r)
        n_groups = len(self._read_groups_from_table()) + 1
        self.group_table.setItem(r, 0, QTableWidgetItem(f"G{n_groups}"))
        self._sync_nurbs_defaults_from_group_table()

    def _on_delete_group(self):
        rows = sorted({i.row() for i in self.group_table.selectedIndexes()}, reverse=True)
        if not rows:
            self._log("[!] Select row(s) to delete.")
            return
        for r in rows:
            gid = self.group_table.item(r, 0)
            if gid and gid.text().strip().upper() == "UNGROUPED":
                continue
            self.group_table.removeRow(r)
        self._sync_nurbs_defaults_from_group_table()

    def _read_groups_from_table(self):
        rows = self.group_table.rowCount()
        out = []
        for r in range(rows):
            gid = self.group_table.item(r, 0)
            s = self.group_table.item(r, 1)
            e = self.group_table.item(r, 2)
            try:
                gid = gid.text().strip() if gid else f"G{r + 1}"
                if gid.upper() == "UNGROUPED":
                    continue
                s = float(s.text()) if s and s.text() not in ("", None) else None
                e = float(e.text()) if e and e.text() not in ("", None) else None
                if s is None or e is None:
                    continue
                if e < s:
                    s, e = e, s
                out.append({
                    "id": gid,
                    "start": s,
                    "end": e,
                    "start_reason": self._get_group_boundary_reason(r, 1),
                    "end_reason": self._get_group_boundary_reason(r, 2),
                    "color": self._get_color_cell_value(r),
                })
            except Exception:
                continue
        return self._backend.normalize_groups(out)

    def _find_ungrouped_row(self) -> Optional[int]:
        rows = self.group_table.rowCount()
        for r in range(rows):
            gid = self.group_table.item(r, 0)
            if gid and gid.text().strip().upper() == "UNGROUPED":
                return r
        return None

    def _compute_ungrouped_ranges(self, groups: list, smin: float, smax: float) -> List[tuple]:
        if smin is None or smax is None or smax <= smin:
            return []
        norm = []
        for g in (groups or []):
            try:
                s = float(g.get("start", 0.0))
                e = float(g.get("end", 0.0))
            except Exception:
                continue
            if e < s:
                s, e = e, s
            s = max(s, smin)
            e = min(e, smax)
            if e > s:
                norm.append((s, e))
        norm.sort(key=lambda x: x[0])
        gaps = []
        cur = smin
        for s, e in norm:
            if s > cur:
                gaps.append((cur, s))
            if e > cur:
                cur = e
        if cur < smax:
            gaps.append((cur, smax))
        return gaps

    def _append_ungrouped_row(self, groups: list, length_m: Optional[float]) -> None:
        r = self._find_ungrouped_row()
        if r is not None:
            self.group_table.removeRow(r)
        if length_m is not None:
            smin, smax = 0.0, float(length_m)
        else:
            starts = [float(g.get("start", 0.0)) for g in (groups or []) if g.get("start", None) is not None]
            ends = [float(g.get("end", 0.0)) for g in (groups or []) if g.get("end", None) is not None]
            if not starts or not ends:
                return
            smin, smax = min(starts), max(ends)
        gaps = self._compute_ungrouped_ranges(groups, smin, smax)
        if not gaps:
            return
        starts = "; ".join([f"{s:.3f}" for s, _ in gaps])
        ends = "; ".join([f"{e:.3f}" for _, e in gaps])
        r = self.group_table.rowCount()
        self.group_table.insertRow(r)
        item_id = QTableWidgetItem("UNGROUPED")
        item_s = QTableWidgetItem(starts)
        item_e = QTableWidgetItem(ends)
        for it in (item_id, item_s, item_e):
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        self.group_table.setItem(r, 0, item_id)
        self.group_table.setItem(r, 1, item_s)
        self.group_table.setItem(r, 2, item_e)
        self._set_color_cell(r, "#bbbbbb")
        color_item = self.group_table.item(r, 3)
        if color_item:
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)

    def _normalize_color_hex(self, color_hex: str) -> str:
        if not color_hex:
            return ""
        c = color_hex.strip()
        if not c:
            return ""
        if not c.startswith("#"):
            c = f"#{c}"
        qc = QColor(c)
        return qc.name() if qc.isValid() else ""

    def _set_color_cell(self, row: int, color_hex: str) -> None:
        c = self._normalize_color_hex(color_hex)
        item = self.group_table.item(row, 3)
        if item is None:
            item = QTableWidgetItem("")
            self.group_table.setItem(row, 3, item)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        if c:
            item.setData(Qt.UserRole, c)
            item.setToolTip(c)
            item.setText("")
            item.setBackground(QColor(c))
        else:
            item.setData(Qt.UserRole, "")
            item.setToolTip("")
            item.setText("")
            item.setBackground(QColor(0, 0, 0, 0))

    def _get_color_cell_value(self, row: int) -> str:
        item = self.group_table.item(row, 3)
        if item is None:
            return ""
        val = item.data(Qt.UserRole)
        if isinstance(val, str) and val.strip():
            return val.strip()
        txt = item.text().strip() if item.text() else ""
        return self._normalize_color_hex(txt)

    def _on_group_cell_double_clicked(self, row: int, col: int) -> None:
        if col != 3:
            return
        current = self._get_color_cell_value(row)
        initial = QColor(current) if current else QColor(255, 255, 255)
        color = QColorDialog.getColor(initial, self, "Select group color")
        if color.isValid():
            self._set_color_cell(row, color.name())
            self._render_current_safe()

    def _get_ungrouped_color(self) -> str:
        r = self._find_ungrouped_row()
        if r is None:
            return "#bbbbbb"
        c = self._get_color_cell_value(r)
        return c or "#bbbbbb"

    def _load_groups_for_current_line(self):
        line_id = self._line_id_current()
        table_groups = self._read_groups_from_table()
        if table_groups:
            return table_groups
        js_path = self._groups_json_path()
        if os.path.exists(js_path):
            try:
                _data, groups = self._load_group_json_data(js_path, line_id=line_id, apply_settings=True)
                if groups:
                    return groups
            except Exception:
                pass
        return []

    def _on_confirm_groups(self):
        if self.line_combo.count() == 0:
            self._log("[!] No line.")
            return
        groups = self._read_groups_from_table()
        prof_for_stats = None
        if not groups:
            row = self.line_combo.currentIndex()
            geom = self._gdf.geometry.iloc[row]
            prof = self._compute_profile_for_geom(geom, slip_only=False)
            if prof:
                groups = self._backend.auto_group_profile(
                    prof,
                    self._grouping_params_current(),
                    min_len=WORKFLOW_GROUP_MIN_LEN_M,
                )
        try:
            row = self.line_combo.currentIndex()
            geom = self._gdf.geometry.iloc[row]
            prof = self._compute_profile_for_geom(geom, slip_only=False)
            if prof:
                prof_for_stats = prof
                groups = self._backend.clamp_groups(prof, groups, min_len=WORKFLOW_GROUP_MIN_LEN_M)
        except Exception:
            pass

        line_id = self._line_id_current()
        curve_method = self._get_curve_method_for_line(line_id)
        try:
            js = self._backend.build_group_json_payload(
                line_label=self.line_combo.currentText(),
                groups=groups,
                prof=prof_for_stats,
                chainage_origin=self._ui3_chainage_origin(),
                curve_method=curve_method,
                profile_dem_source=self._current_profile_source_key(),
                profile_dem_path=str(getattr(self, "dem_path", "") or ""),
                grouping_params=self._grouping_params_current(),
                group_method=None,
            )
            saved_path = self._backend.save_group_json(self._groups_json_path(), js)
            self._log(f"[✓] Saved groups: {saved_path}")
        except Exception as e:
            self._log(f"[!] Save groups failed: {e}")

        self._render_current_safe()
        try:
            line_id = self._line_id_current()
            groups = self._read_groups_from_table()
            bounds_set = set()
            for g in groups:
                try:
                    s = float(g.get("start", "nan"))
                    e = float(g.get("end", "nan"))
                    if not math.isnan(s) and not math.isnan(e):
                        if e < s:
                            s, e = e, s
                        if self._sec_len_m:
                            s = max(0.0, min(self._sec_len_m, s))
                            e = max(0.0, min(self._sec_len_m, e))
                        bounds_set.add(s)
                        bounds_set.add(e)
                except Exception:
                    pass
            bounds_m = sorted(bounds_set)
            if bounds_m:
                self._group_bounds[line_id] = bounds_m
            if self._px_per_m is None and getattr(self, "_img_ground", None) and self._sec_len_m:
                w = self._img_ground.pixmap().width()
                if w and self._sec_len_m > 0:
                    self._px_per_m = float(w) / float(self._sec_len_m)
            self._ok("[UI3] Groups confirmed and guides updated.")
        except Exception as e:
            self._warn(f"[UI3] Confirm Groups: cannot update guides ({e})")

    def _on_auto_group(self) -> None:
        try:
            try:
                self._nurbs_live_timer.stop()
            except Exception:
                pass
            if not hasattr(self, "_gdf") or self._gdf is None or self._gdf.empty:
                self._warn("[UI3] No lines loaded from UI2.")
                return
            if self._current_ui2_line_role() != "main":
                self._warn("[UI3] Auto Group applies only to Main Lines. Cross Lines will use a separate method.")
                return
            row = self.line_combo.currentIndex()
            if row < 0:
                self._warn("[UI3] No line selected.")
                return
            geom = self._gdf.geometry.iloc[row]
            prof = self._compute_profile_for_geom(geom, slip_only=False)
            if not prof:
                self._err("[UI3] Empty profile.")
                return

            include_curvature = self._include_curvature_threshold()
            include_vector_zero = self._include_vector_angle_zero()
            parts = [f"{self._current_profile_source_key()} DEM", "RDP"]
            if include_curvature:
                parts.append("curvature")
            if include_vector_zero:
                parts.append("vector=0")
            self._log("[UI3] Auto Group method: " + " + ".join(parts))
            groups = self._backend.auto_group_profile(
                prof,
                self._grouping_params_current(),
                min_len=WORKFLOW_GROUP_MIN_LEN_M,
            )
            if include_curvature and include_vector_zero:
                group_method = "profile_dem_rdp_curvature_theta0"
            elif include_curvature:
                group_method = "profile_dem_rdp_curvature_only"
            elif include_vector_zero:
                group_method = "profile_dem_rdp_theta0_only"
            else:
                group_method = "profile_dem_rdp_span_only"
            if not groups:
                self._warn("[UI3] Auto grouping produced no segments within slip zone.")
                return

            line_id = self._line_id_current()
            rdp_csv = self._save_rdp_csv_for_line(line_id, prof)
            self._save_groups_to_ui(
                groups,
                prof,
                line_id,
                log_text=f"[UI3] Auto Group done for '{line_id}': {len(groups)} groups.",
                curve_method=self._curve_method_from_group_method(group_method),
                group_method=group_method,
            )
            if rdp_csv:
                self._log(f"[UI3] Saved RDP CSV: {rdp_csv}")
            self._render_current_safe()
            try:
                self._nurbs_live_timer.stop()
            except Exception:
                pass
        except Exception as e:
            self._err(f"[UI3] Auto Group error: {e}")
