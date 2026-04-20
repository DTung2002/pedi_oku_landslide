import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QMessageBox

from pedi_oku_landslide.application.ui3.profile_sampling import parse_nominal_length_m, resample_profile_to_nominal_grid
from pedi_oku_landslide.domain.ui3.boring_holes import BORING_HOLES_DEFAULT_TOLERANCE_M

WORKFLOW_GROUP_MIN_LEN_M = 0.0
WORKFLOW_GROUPING_PARAMS = {
    "rdp_eps_m": 0.5,
    "curvature_thr_abs": 0.02,
    "smooth_radius_m": 0.0,
    "include_curvature_threshold": True,
    "include_vector_angle_zero": True,
}


class UI3LineControllerMixin:
    @staticmethod
    def _pick_existing_path(*cands: str) -> str:
        for p in cands:
            if p and os.path.exists(p):
                return p
        return ""

    def _current_profile_source_key(self) -> str:
        return "raw"

    def _set_profile_source_key(self, key: Optional[str], *, log_paths: bool = False) -> str:
        _ = key
        self._refresh_profile_source_paths(log_paths=log_paths)
        return self._current_profile_source_key()

    def _refresh_profile_source_paths(self, log_paths: bool = False) -> None:
        chosen = str(getattr(self, "dem_path_raw", "") or "").strip()
        if chosen and not os.path.exists(chosen):
            chosen = ""
        self.dem_path = chosen
        self.ground_export_dem_path = chosen
        self._sync_step_to_raw_dem(log_paths=log_paths)
        if log_paths:
            self._log(f"[UI3] Raw DEM: {self.dem_path_raw}")
            self._log(f"[UI3] Smoothed DEM: {self.dem_path_smooth}")
            self._log(f"[UI3] Active profile DEM: {self.dem_path}")
            if not self.dem_path:
                self._warn("[UI3] Raw DEM not found. Render/export will stop until a valid raw DEM is available.")

    def _raw_dem_pixel_step_m(self) -> Optional[float]:
        dem_path = str(getattr(self, "dem_path_raw", "") or "").strip()
        if not dem_path or not os.path.exists(dem_path):
            return None
        try:
            with rasterio.open(dem_path) as ds:
                xres, yres = ds.res
            vals = [float(v) for v in (xres, yres) if np.isfinite(v) and float(v) > 0.0]
            if not vals:
                return None
            return float(min(vals))
        except Exception:
            return None

    def _sync_step_to_raw_dem(self, *, log_paths: bool = False) -> None:
        if self.step_box is None:
            return
        try:
            old = self.step_box.blockSignals(True)
            self.step_box.setValue(float(self._GROUND_EXPORT_STEP_M))
            self.step_box.blockSignals(old)
            self.step_box.setEnabled(False)
        except Exception:
            return
        if log_paths:
            self._log(f"[UI3] Sampling step fixed to export grid: {float(self._GROUND_EXPORT_STEP_M):.4f} m")

    def _include_vector_angle_zero(self) -> bool:
        return True

    def _include_curvature_threshold(self) -> bool:
        return True

    def _current_rdp_eps_m(self) -> float:
        try:
            if self.rdp_eps_spin is not None:
                val = float(self.rdp_eps_spin.value())
                if np.isfinite(val) and val >= 0.0:
                    return float(val)
        except Exception:
            pass
        return float(WORKFLOW_GROUPING_PARAMS.get("rdp_eps_m", 0.5))

    def _current_smooth_radius_m(self) -> float:
        return 0.0

    def _grouping_params_current(self) -> Dict[str, Any]:
        params = dict(WORKFLOW_GROUPING_PARAMS)
        params["rdp_eps_m"] = self._current_rdp_eps_m()
        params["smooth_radius_m"] = 0.0
        params["include_curvature_threshold"] = self._include_curvature_threshold()
        params["include_vector_angle_zero"] = self._include_vector_angle_zero()
        return params

    def _apply_group_json_settings(self, data: Dict[str, Any], *, log_paths: bool = False) -> None:
        try:
            if self.rdp_eps_spin is not None and "rdp_eps_m" in data:
                self.rdp_eps_spin.setValue(max(0.0, float(data.get("rdp_eps_m"))))
        except Exception:
            pass
        try:
            if "profile_dem_source" in data:
                self._set_profile_source_key("raw", log_paths=log_paths)
        except Exception:
            pass

    @staticmethod
    def _group_signature(groups: List[dict]) -> List[Tuple[str, float, float, str, str]]:
        sig: List[Tuple[str, float, float, str, str]] = []
        for i, g in enumerate(groups or [], 1):
            try:
                gid = str(g.get("id", g.get("group_id", f"G{i}")) or f"G{i}").strip()
                s = float(g.get("start", g.get("start_chainage")))
                e = float(g.get("end", g.get("end_chainage")))
            except Exception:
                continue
            if e < s:
                s, e = e, s
            sig.append((gid, round(float(s), 3), round(float(e), 3), str(g.get("start_reason", "") or "").strip(), str(g.get("end_reason", "") or "").strip()))
        return sig

    @staticmethod
    def _ui3_chainage_origin() -> str:
        return "picked"

    def _groups_to_current_chainage(
        self,
        groups: List[dict],
        *,
        source_origin: Optional[str] = None,
        length_m: Optional[float] = None,
    ) -> List[dict]:
        _ = source_origin
        _ = length_m
        out: List[Dict[str, Any]] = []
        for i, g in enumerate(groups or [], 1):
            try:
                s = float(g.get("start", g.get("start_chainage", np.nan)))
                e = float(g.get("end", g.get("end_chainage", np.nan)))
            except Exception:
                continue
            if not (np.isfinite(s) and np.isfinite(e)):
                continue
            gid = str(g.get("id", g.get("group_id", f"G{i}")) or f"G{i}").strip() or f"G{i}"
            start_reason = str(g.get("start_reason", "") or "").strip()
            end_reason = str(g.get("end_reason", "") or "").strip()
            if e < s:
                s, e = e, s
                start_reason, end_reason = end_reason, start_reason
            out.append({"id": gid, "start": float(s), "end": float(e), "start_reason": start_reason, "end_reason": end_reason, "color": str(g.get("color", "") or "").strip()})
        return self._backend.normalize_groups(out)

    @staticmethod
    def _interp_series_y(xs: np.ndarray, ys: np.ndarray, xq: float) -> Optional[float]:
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if xs.ndim != 1 or ys.ndim != 1 or xs.size < 2 or ys.size != xs.size:
            return None
        keep = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[keep]
        ys = ys[keep]
        if xs.size < 2:
            return None
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        if not np.isfinite(float(xq)):
            return None
        if xq < float(xs[0]) or xq > float(xs[-1]):
            return None
        if np.any(np.isclose(xs, float(xq), atol=1e-9)):
            idx = int(np.flatnonzero(np.isclose(xs, float(xq), atol=1e-9))[0])
            return float(ys[idx])
        idx = int(np.searchsorted(xs, float(xq)))
        if idx <= 0 or idx >= xs.size:
            return None
        x0 = float(xs[idx - 1]); x1 = float(xs[idx])
        y0 = float(ys[idx - 1]); y1 = float(ys[idx])
        if not (np.isfinite(x0) and np.isfinite(x1) and np.isfinite(y0) and np.isfinite(y1)) or abs(x1 - x0) <= 1e-12:
            return None
        t = (float(xq) - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0))

    def _repair_saved_curvature_points(self, points: List[dict], prof: Optional[dict]) -> List[dict]:
        repaired = [dict(p) for p in (points or [])]
        if len(repaired) < 2 or not prof:
            return repaired
        try:
            params = self._grouping_params_current()
            nodes = self._backend.extract_curvature_nodes(
                prof,
                rdp_eps_m=float(params.get("rdp_eps_m", 0.5)),
                smooth_radius_m=float(params.get("smooth_radius_m", 0.0)),
                restrict_to_slip_span=False,
            )
            full_x = np.asarray(nodes.get("chain", []), dtype=float)
            full_k = np.asarray(nodes.get("curvature", []), dtype=float)
        except Exception:
            return repaired
        if full_x.size < 2 or full_k.size != full_x.size:
            return repaired

        def _repair_one(item: dict, neighbor: Optional[dict], is_head: bool) -> None:
            try:
                idx = int(item.get("index", -1))
                kval = float(item.get("curvature", np.nan))
                xval = float(item.get("chain_m", np.nan))
            except Exception:
                return
            if not (np.isfinite(kval) and np.isfinite(xval)) or abs(kval) > 1e-12:
                return
            if is_head:
                if idx != 0:
                    return
            else:
                try:
                    if neighbor is None:
                        return
                    nidx = int(neighbor.get("index", -1))
                    if idx != (nidx + 1):
                        return
                except Exception:
                    return
            kval_new = self._interp_series_y(full_x, full_k, float(xval))
            if kval_new is None or not np.isfinite(kval_new):
                return
            item["curvature"] = float(kval_new)
            item["curvature_abs"] = abs(float(kval_new))

        _repair_one(repaired[0], repaired[1] if len(repaired) > 1 else None, True)
        _repair_one(repaired[-1], repaired[-2] if len(repaired) > 1 else None, False)
        return repaired

    def _saved_curvature_series_for_line(self, line_id: str, current_groups: Optional[List[dict]], prof: Optional[dict] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._backend.saved_curvature_series_for_line(
            path=self._groups_json_path_for(line_id),
            current_groups=current_groups,
            prof=prof,
            repair_saved_curvature_points=self._repair_saved_curvature_points,
        )

    def _on_profile_source_changed(self, _idx: int) -> None:
        self._refresh_profile_source_paths(log_paths=True)
        self._active_prof = None
        if self.line_combo is None or self.line_combo.count() == 0:
            return
        self._render_current_safe()

    def _compute_profile_for_geom(self, geom, *, slip_only: bool) -> Optional[dict]:
        dem_path = str(getattr(self, "dem_path", "") or "").strip()
        dem_orig_path = str(getattr(self, "ground_export_dem_path", "") or "").strip()
        if not dem_path:
            self._warn("[UI3] Raw DEM not found. Cannot render profile.")
            return None
        prof = self._backend.compute_profile_for_line(
            self._line_id_current(),
            geom,
            self._current_profile_source_key(),
            float(self._GROUND_EXPORT_STEP_M),
            bool(slip_only),
            dem_path=dem_path,
            dem_orig_path=(dem_orig_path or dem_path),
            dx_path=self.dx_path,
            dy_path=self.dy_path,
            dz_path=self.dz_path,
            slip_mask_path=self.slip_path,
        )
        if not prof:
            return None
        nominal_length_m = parse_nominal_length_m(self._line_id_current())
        if nominal_length_m is None:
            try:
                nominal_length_m = round(float(getattr(geom, "length", np.nan)), 1)
            except Exception:
                nominal_length_m = None
        return resample_profile_to_nominal_grid(
            prof,
            line_id=self._line_id_current(),
            target_step_m=float(self._GROUND_EXPORT_STEP_M),
            nominal_length_m=nominal_length_m,
        )

    def set_context(self, project: str, run_label: str, run_dir: str) -> None:
        self._ctx.update({"project": project, "run_label": run_label, "run_dir": run_dir})
        self._ui2_intersections_cache = None
        self._anchors_xyz_cache = None
        backend_inputs = self._backend.set_context(project, run_label, run_dir, base_dir=self.base_dir)

        self.dem_path_smooth = str(backend_inputs.get("dem_path_smooth", "") or "")
        self.dem_path_raw = str(backend_inputs.get("dem_path_raw", "") or "")
        self._refresh_profile_source_paths(log_paths=True)
        self.dx_path = str(backend_inputs.get("dx_path", "") or "")
        self.dy_path = str(backend_inputs.get("dy_path", "") or "")
        self.dz_path = str(backend_inputs.get("dz_path", "") or "")
        self.lines_path = ""
        self.slip_path = str(backend_inputs.get("slip_path", "") or "")
        self._load_boring_holes_into_ui()
        if not self.slip_path or not os.path.exists(self.slip_path):
            self._warn("[UI3] Slip-zone mask not found. Vectors outside landslide may appear.")
        self._load_lines_into_combo()

    def reset_session(self) -> None:
        self._ctx = {"project": "", "run_label": "", "run_dir": ""}
        if hasattr(self, "edit_project") and self.edit_project is not None:
            self.edit_project.clear()
        if hasattr(self, "edit_runlabel") and self.edit_runlabel is not None:
            self.edit_runlabel.clear()
        self.dem_path = ""
        self.dem_path_raw = ""
        self.dem_path_smooth = ""
        self.ground_export_dem_path = ""
        self.dx_path = ""
        self.dy_path = ""
        self.dz_path = ""
        self.lines_path = ""
        self.slip_path = ""
        try:
            if self.step_box is not None:
                old = self.step_box.blockSignals(True)
                self.step_box.setValue(float(self._default_profile_step_m))
                self.step_box.blockSignals(old)
        except Exception:
            pass
        try:
            if self.rdp_eps_spin is not None:
                self.rdp_eps_spin.setValue(float(WORKFLOW_GROUPING_PARAMS.get("rdp_eps_m", 0.5)))
        except Exception:
            pass
        try:
            if self.line_combo is not None:
                self.line_combo.blockSignals(True)
                self.line_combo.clear()
                self.line_combo.blockSignals(False)
        except Exception:
            pass
        try:
            if self.group_table is not None:
                self.group_table.setRowCount(0)
        except Exception:
            pass
        try:
            if self.boring_table is not None:
                self.boring_table.setRowCount(0)
        except Exception:
            pass
        try:
            if self.scene is not None:
                self.scene.clear()
        except Exception:
            pass
        try:
            if self.view is not None:
                self.view.resetTransform()
        except Exception:
            pass
        self._px_per_m = None
        self._sec_len_m = None
        self._group_bounds.clear()
        self._guide_lines_top.clear()
        self._guide_lines_bot.clear()
        self._group_bands_bot.clear()
        self._img_ground = None
        self._img_rate0 = None
        self._clear_curve_overlay()
        self._active_prof = None
        self._active_groups = []
        self._active_base_curve = None
        self._active_curve = None
        self._active_global_fit_result = None
        self._update_global_fit_debug_panel(None, theta_csv_path=None)
        self._ui2_intersections_cache = None
        self._anchors_xyz_cache = None
        self._boring_holes_data = self._empty_boring_holes_payload()
        self._plot_x0_px = None
        self._plot_w_px = None
        self._x_min = None
        self._x_max = None
        self._current_idx = 0
        try:
            if self.status is not None:
                self.status.clear()
        except Exception:
            pass
        try:
            self._log("[UI3] Curve tab session reset.")
        except Exception:
            pass

    def _paths(self):
        return self._backend.ui3_paths()

    def _ui3_run_dir(self) -> str:
        return self._paths().ui3_run_dir()

    def _preview_dir(self) -> str:
        return self._paths().preview_dir()

    def _groups_dir(self) -> str:
        return self._paths().groups_dir()

    def _curve_dir(self) -> str:
        return self._paths().curve_dir()

    def _ground_dir(self) -> str:
        return self._paths().ground_dir()

    def _curve_nurbs_info_json_path_for(self, line_id: str) -> str:
        return self._paths().curve_nurbs_info_json_path_for(line_id)

    def _curve_nurbs_png_path_for(self, line_id: str) -> str:
        return self._paths().nurbs_png_path_for(line_id)

    def _line_id_current(self) -> str:
        if hasattr(self, "line_combo"):
            txt = self.line_combo.currentText().strip() or f"line_{self.line_combo.currentIndex() + 1:03d}"
            return txt.replace(" ", "_")
        row = getattr(self, "line_combo", None).currentIndex() if hasattr(self, "line_combo") else 0
        return f"line_{(row or 0) + 1:03d}"

    def _profile_png_path_for(self, line_id: str) -> str:
        return self._paths().profile_png_path_for(line_id)

    def _groups_json_path_for(self, line_id: str) -> str:
        return self._paths().groups_json_path_for(line_id)

    def _ui2_intersections_json_path(self) -> str:
        return self._paths().ui2_intersections_json_path()

    def _anchors_xyz_json_path(self) -> str:
        return self._paths().anchors_json_path()

    def _boring_holes_dir(self) -> str:
        return self._paths().boring_holes_dir()

    def _boring_holes_json_path_for(self, line_id: str) -> str:
        return self._paths().boring_holes_json_path_for(line_id)

    def _boring_holes_json_path(self) -> str:
        return self._boring_holes_json_path_for(self._line_id_current())

    @staticmethod
    def _empty_boring_holes_payload() -> Dict[str, Any]:
        return {
            "version": 1,
            "distance_tolerance_m": float(BORING_HOLES_DEFAULT_TOLERANCE_M),
            "items": [],
        }

    def _populate_boring_holes_table(self, payload: Optional[Dict[str, Any]]) -> None:
        if self.boring_table is None:
            return
        data = self._backend.build_boring_holes_payload(
            (payload or {}).get("items", []),
            distance_tolerance_m=(payload or {}).get("distance_tolerance_m", BORING_HOLES_DEFAULT_TOLERANCE_M),
        )
        self._boring_holes_data = data
        self._boring_table_updating = True
        try:
            self.boring_table.setRowCount(0)
            items = list(data.get("items", []) or [])
            if not items:
                items = [
                    {"bh": "BH1", "x": "", "y": "", "z": ""},
                    {"bh": "BH2", "x": "", "y": "", "z": ""},
                    {"bh": "BH3", "x": "", "y": "", "z": ""},
                ]
            for rec in items:
                r = self.boring_table.rowCount()
                self.boring_table.insertRow(r)
                self.boring_table.setItem(r, 0, QTableWidgetItem(str(rec.get("bh", "") or "").strip()))
                for col, key in ((1, "x"), (2, "y"), (3, "z")):
                    val = rec.get(key, "")
                    if isinstance(val, (int, float, np.floating)) and np.isfinite(float(val)):
                        txt = f"{float(val):.3f}"
                    else:
                        txt = ""
                    self.boring_table.setItem(r, col, QTableWidgetItem(txt))
        finally:
            self._boring_table_updating = False

    def _parse_boring_holes_from_table(self, *, strict: bool) -> Tuple[Dict[str, Any], List[str]]:
        errors: List[str] = []
        items: List[Dict[str, Any]] = []
        seen = set()
        if self.boring_table is None:
            return self._empty_boring_holes_payload(), errors

        tol = float((self._boring_holes_data or {}).get("distance_tolerance_m", BORING_HOLES_DEFAULT_TOLERANCE_M))
        for r in range(self.boring_table.rowCount()):
            cells = []
            for c in range(4):
                item = self.boring_table.item(r, c)
                cells.append(item.text().strip() if item and item.text() else "")
            bh_txt, x_txt, y_txt, z_txt = cells
            if not any(cells):
                continue
            if not strict and (not bh_txt or not x_txt or not y_txt or not z_txt):
                continue
            if not bh_txt:
                errors.append(f"Row {r + 1}: BH is required.")
                continue
            if bh_txt in seen:
                if strict:
                    errors.append(f"Row {r + 1}: duplicate BH '{bh_txt}'.")
                continue
            try:
                x_val = float(x_txt)
                y_val = float(y_txt)
                z_val = float(z_txt)
            except Exception:
                if strict:
                    errors.append(f"Row {r + 1}: X, Y, Z must be numeric.")
                continue
            if not (np.isfinite(x_val) and np.isfinite(y_val) and np.isfinite(z_val)):
                if strict:
                    errors.append(f"Row {r + 1}: X, Y, Z must be finite.")
                continue
            seen.add(bh_txt)
            items.append({"bh": bh_txt, "x": float(x_val), "y": float(y_val), "z": float(z_val)})
        payload = self._backend.build_boring_holes_payload(items, distance_tolerance_m=tol)
        return payload, errors

    def _current_boring_holes_payload(self, *, strict: bool = False) -> Dict[str, Any]:
        payload, _errors = self._parse_boring_holes_from_table(strict=bool(strict))
        return payload

    def _load_boring_holes_into_ui(self) -> None:
        line_id = self._line_id_current().strip()
        if not line_id or (hasattr(self, "line_combo") and self.line_combo.currentIndex() < 0):
            self._populate_boring_holes_table(self._empty_boring_holes_payload())
            return
        try:
            payload = self._backend.load_boring_holes(self._boring_holes_json_path_for(line_id))
        except Exception:
            payload = self._empty_boring_holes_payload()
        self._populate_boring_holes_table(payload)

    def _current_line_geom(self):
        try:
            if not hasattr(self, "_gdf") or self._gdf is None or self._gdf.empty:
                return None
            row = self.line_combo.currentIndex()
            if row < 0 or row >= len(self._gdf):
                return None
            return self._gdf.geometry.iloc[row]
        except Exception:
            return None

    def _project_boring_holes_to_line(
        self,
        geom,
        *,
        use_unsaved_table: bool = True,
        log_skips: bool = False,
    ) -> Dict[str, Any]:
        payload = self._current_boring_holes_payload(strict=False) if use_unsaved_table else dict(self._boring_holes_data or {})
        tol = float(payload.get("distance_tolerance_m", BORING_HOLES_DEFAULT_TOLERANCE_M))
        result = self._backend.project_boring_holes_to_line(
            geom,
            payload,
            distance_tolerance_m=tol,
        )
        if log_skips:
            for msg in (result.get("skipped", []) or []):
                self._log(f"[UI3] {msg}")
        return result

    def _project_boring_holes_for_current_line(
        self,
        *,
        use_unsaved_table: bool = True,
        log_skips: bool = False,
    ) -> Dict[str, Any]:
        geom = self._current_line_geom()
        if geom is None:
            return {"distance_tolerance_m": float(BORING_HOLES_DEFAULT_TOLERANCE_M), "items": [], "skipped": []}
        return self._project_boring_holes_to_line(
            geom,
            use_unsaved_table=use_unsaved_table,
            log_skips=log_skips,
        )

    def _on_boring_holes_table_item_changed(self, _item) -> None:
        if self._boring_table_updating:
            return
        self._refresh_anchor_overlay()
        if self._active_prof:
            self._schedule_nurbs_live_update()

    def _on_add_boring_hole(self) -> None:
        if self.boring_table is None:
            return
        r = self.boring_table.rowCount()
        self.boring_table.insertRow(r)
        for c in range(4):
            self.boring_table.setItem(r, c, QTableWidgetItem(""))
        self._refresh_anchor_overlay()
        if self._active_prof:
            self._schedule_nurbs_live_update()

    def _on_delete_boring_hole(self) -> None:
        if self.boring_table is None:
            return
        rows = sorted({i.row() for i in self.boring_table.selectedIndexes()}, reverse=True)
        if not rows:
            self._log("[!] Select boring hole row(s) to delete.")
            return
        for r in rows:
            self.boring_table.removeRow(r)
        self._refresh_anchor_overlay()
        if self._active_prof:
            self._schedule_nurbs_live_update()

    def _on_save_boring_holes(self) -> None:
        payload, errors = self._parse_boring_holes_from_table(strict=True)
        if errors:
            for msg in errors[:8]:
                self._warn(f"[UI3] {msg}")
            if len(errors) > 8:
                self._warn(f"[UI3] {len(errors) - 8} additional boring-hole validation errors omitted.")
            return
        line_id = self._line_id_current().strip()
        if not line_id or (hasattr(self, "line_combo") and self.line_combo.currentIndex() < 0):
            self._warn("[UI3] No line selected for saving boring holes.")
            return
        path = self._backend.save_boring_holes(self._boring_holes_json_path_for(line_id), payload)
        self._boring_holes_data = payload
        self._ok(f"[UI3] Saved boring holes: {path}")
        self._refresh_anchor_overlay()
        if self._active_prof:
            self._schedule_nurbs_live_update()

    @staticmethod
    def _normalize_line_role(line_role: str, line_id: str = "") -> str:
        role = (line_role or "").strip().lower()
        lid = (line_id or "").strip().lower()
        if role in ("main", "ml"):
            return "main"
        if role in ("cross", "aux", "cl"):
            return "cross"
        if lid.startswith("ml") or lid.startswith("main"):
            return "main"
        if lid.startswith("cl") or lid.startswith("cross"):
            return "cross"
        return ""

    def _line_row_meta(self, row: Optional[int] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"row": -1, "line_id": "", "line_role": ""}
        try:
            ridx = self.line_combo.currentIndex() if row is None else int(row)
        except Exception:
            ridx = -1
        out["row"] = ridx
        try:
            if ridx >= 0 and hasattr(self, "_gdf") and self._gdf is not None and ridx < len(self._gdf):
                g = self._gdf.iloc[ridx]
                line_id = str(g.get("line_id", g.get("name", "")) or "").strip()
                line_role = self._normalize_line_role(str(g.get("line_role", "")) or "", line_id)
                out["line_id"] = line_id
                out["line_role"] = line_role
                out["name"] = str(g.get("name", "") or "").strip()
        except Exception:
            pass
        if not out.get("line_id"):
            out["line_id"] = self._line_id_current()
        if not out.get("line_role"):
            out["line_role"] = self._normalize_line_role("", str(out.get("line_id", "")))
        return out

    def _current_ui2_line_id(self) -> str:
        return str(self._line_row_meta().get("line_id", "") or "")

    def _current_ui2_line_role(self) -> str:
        return str(self._line_row_meta().get("line_role", "") or "")

    def _refresh_group_controls_for_line_role(self) -> None:
        btn = getattr(self, "btn_auto_group", None)
        if btn is None:
            return
        role = self._current_ui2_line_role()
        is_main = (role == "main")
        btn.setEnabled(is_main)
        if is_main:
            btn.setToolTip("")
        elif role == "cross":
            btn.setToolTip("Auto Group is available only for Main Lines.")
        else:
            btn.setToolTip("Auto Group requires a line marked as Main.")

    def _load_ui2_intersections(self, force: bool = False) -> Dict[str, Any]:
        if (not force) and isinstance(self._ui2_intersections_cache, dict):
            return self._ui2_intersections_cache
        path = ""
        try:
            path = self._ui2_intersections_json_path()
            data = self._backend.load_anchor_items(path)
        except Exception:
            data = {}
        items = data.get("items", []) if isinstance(data, dict) else []
        if not isinstance(items, list):
            items = []
        data = dict(data or {})
        data["items"] = items
        data["_path"] = path
        self._ui2_intersections_cache = data
        return data

    def _load_anchors_xyz(self, force: bool = False) -> Dict[str, Any]:
        if (not force) and isinstance(self._anchors_xyz_cache, dict):
            return self._anchors_xyz_cache
        path = self._anchors_xyz_json_path()
        data = self._backend.load_anchor_items(path)
        items = data.get("items", []) if isinstance(data, dict) else []
        if not isinstance(items, list):
            items = []
        data = dict(data or {})
        data["items"] = items
        data["_path"] = path
        self._anchors_xyz_cache = data
        return data

    def _save_anchors_xyz(self, data: Dict[str, Any]) -> str:
        path = self._anchors_xyz_json_path()
        payload = dict(data or {})
        saved_path = self._backend.save_anchor_items(path, payload)
        payload.pop("_path", None)
        payload["_path"] = saved_path
        self._anchors_xyz_cache = payload
        return saved_path

    def _anchors_ready_for_cross_constraints(self) -> bool:
        return self._backend.anchors_ready_for_cross_constraints(self._load_ui2_intersections(), self._load_anchors_xyz())

    def _anchors_for_cross_line(self, cross_line_id: str, require_ready: bool = True) -> List[dict]:
        return self._backend.anchors_for_cross_line(
            self._load_ui2_intersections(),
            self._load_anchors_xyz(),
            str(cross_line_id or ""),
            require_ready=bool(require_ready),
        )

    def _update_anchors_xyz_for_saved_main_curve(self, curve: Dict[str, np.ndarray]) -> Tuple[Optional[str], int]:
        if self._current_ui2_line_role() != "main":
            return None, 0
        main_line_id = self._current_ui2_line_id()
        if not main_line_id:
            return None, 0
        result = self._backend.sync_anchor_updates(main_line_id, curve)
        updated = int(result.get("updated", 0) or 0)
        if updated <= 0:
            return None, 0
        self._anchors_xyz_cache = None
        out_path = str(result.get("path", "") or self._anchors_xyz_json_path())
        return out_path, updated

    def _save_ground_csv_for_line(self, line_id: str, geom, step_m: float) -> Optional[str]:
        out_csv = self._ground_csv_path_for(line_id)
        return self._backend.save_ground_csv(
            line_id=line_id,
            geom=geom,
            step_m=float(step_m),
            profile_source=self._current_profile_source_key(),
            ground_export_step_m=float(self._GROUND_EXPORT_STEP_M),
            ground_export_dem_path=str(getattr(self, "ground_export_dem_path", "") or ""),
            dx_path=self.dx_path,
            dy_path=self.dy_path,
            dz_path=self.dz_path,
            slip_path=self.slip_path,
            out_csv=out_csv,
        )

    def _save_rdp_csv_for_line(self, line_id: str, prof: dict) -> Optional[str]:
        params = self._grouping_params_current()
        return self._backend.save_rdp_csv(
            line_id=line_id,
            profile=prof,
            rdp_eps_m=float(params.get("rdp_eps_m", 0.5)),
            smooth_radius_m=float(params.get("smooth_radius_m", 0.0)),
            out_csv=self._rdp_csv_path_for(line_id),
        )

    def _save_theta_csv_for_line(self, line_id: str, prof: dict, groups: Optional[list] = None) -> Optional[str]:
        return self._backend.save_theta_csv(
            line_id=line_id,
            profile=prof,
            groups=groups if groups is not None else (self._active_groups or []),
            out_csv=self._theta_csv_path_for(line_id),
        )

    def _profile_png_path(self) -> str:
        return self._profile_png_path_for(self._line_id_current())

    def _groups_json_path(self) -> str:
        return self._groups_json_path_for(self._line_id_current())

    def _nurbs_png_path_for(self, line_id: str) -> str:
        return os.path.join(self._preview_dir(), f"profile_{line_id}_nurbs.png")

    def _nurbs_json_path_for(self, line_id: str) -> str:
        return os.path.join(self._preview_dir(), f"profile_{line_id}_nurbs.json")

    def _ground_csv_path_for(self, line_id: str) -> str:
        return os.path.join(self._ground_dir(), f"{line_id}_ground.csv")

    def _rdp_csv_path_for(self, line_id: str) -> str:
        return os.path.join(self._ground_dir(), f"{line_id}_RDP.csv")

    def _theta_csv_path_for(self, line_id: str) -> str:
        return os.path.join(self._ground_dir(), f"{line_id}_theta.csv")

    def _vectors_csv_path_for(self, line_id: str) -> str:
        return self._paths().vectors_csv_path_for(line_id)

    def _save_vectors_csv_for_line(self, line_id: str, prof: dict) -> Optional[str]:
        return self._backend.save_vectors_csv(
            line_id=line_id,
            profile=prof,
            out_csv=self._vectors_csv_path_for(line_id),
        )

    def _load_lines_into_combo(self) -> None:
        self.line_combo.blockSignals(True)
        self.line_combo.clear()
        try:
            run_dir = self._ctx.get("run_dir") or ""
            if not run_dir:
                self._log("[!] Run context is empty – cannot load sections.")
                return

            csv_path = os.path.join(run_dir, "ui2", "sections.csv")
            lines_result = self._backend.load_lines(csv_path, self.dem_path)
            migrated = bool(lines_result.get("migrated", False))
            if migrated:
                self._log("[UI3] Migrated legacy sections.csv to current direction version and cleared old UI3 derived outputs.")
            gdf = lines_result.get("gdf")
            if gdf is None or gdf.empty:
                self._log(f"[!] No sections in csv:\n{csv_path}")
                return

            self._gdf = gdf
            labels: List[str] = []
            for _, row in gdf.iterrows():
                base = row.get("name") or f"Line {int(row.get('idx', 0) or 0)}"
                L = float(row.get("length_m", float("nan")))
                if math.isfinite(L) and L > 0:
                    labels.append(f"{base}  ({L:.1f} m)")
                else:
                    labels.append(base)
            self.line_combo.addItems(labels)
            self._log(f"[i] Loaded {len(labels)} lines from ui2/sections.csv.")
        except Exception as e:
            self._log(f"[!] Cannot load lines from sections.csv: {e}")
        finally:
            self.line_combo.blockSignals(False)

        pj = self._ctx.get("project") or "—"
        rl = self._ctx.get("run_label") or "—"
        if hasattr(self, "edit_project") and self.edit_project is not None:
            self.edit_project.setText(pj)
        if hasattr(self, "edit_runlabel") and self.edit_runlabel is not None:
            self.edit_runlabel.setText(rl)
        if self.line_combo.count() > 0:
            self.line_combo.blockSignals(True)
            self.line_combo.setCurrentIndex(0)
            self.line_combo.blockSignals(False)
            self._on_line_changed(0)

    def _on_line_changed(self, _idx: int) -> None:
        self._log("[i] Line changed. Loading saved UI3 data if available.")
        self._clear_curve_overlay()
        self._active_prof = None
        self._active_groups = []
        self._active_base_curve = None
        self._active_curve = None
        self._active_global_fit_result = None
        theta_path = self._theta_csv_path_for(self._line_id_current())
        self._update_global_fit_debug_panel(None, theta_csv_path=(theta_path if os.path.exists(theta_path) else None))
        self._refresh_group_controls_for_line_role()
        self._load_boring_holes_into_ui()
        try:
            row = self.line_combo.currentIndex()
            if row >= 0 and hasattr(self, "_gdf") and self._gdf is not None and not self._gdf.empty:
                geom = self._gdf.geometry.iloc[row]
                self._sec_len_m = float(getattr(geom, "length", np.nan))
        except Exception:
            self._sec_len_m = None

        self._load_saved_curve_state_for_current_line()
        self._refresh_anchor_overlay()

    def _load_group_table_from_path(self, path: str, line_id: str) -> bool:
        if not os.path.exists(path):
            return False
        loaded = 0
        self._group_table_updating = True
        try:
            _data, groups = self._load_group_json_data(path, line_id=line_id, apply_settings=True)
            loaded = self._populate_group_table_rows(groups, length_m=self._sec_len_m)
        except Exception as e:
            self._log(f"[!] Cannot read curve group file: {e}")
            return False
        finally:
            self._group_table_updating = False
        if loaded:
            self._set_curve_method_for_line(line_id, "global_fit_spline")
            self._log(f"[UI3] Loaded group table from: {path}")
            return True
        return False

    def _on_load_group_info(self) -> None:
        if self.line_combo.count() == 0:
            self._warn("[UI3] No line selected.")
            return
        try:
            start_dir = self._curve_dir()
        except Exception:
            start_dir = self.base_dir
        path, _ = QFileDialog.getOpenFileName(self, "Load group_info", start_dir, "JSON files (*.json);;All files (*.*)")
        if not path:
            return
        line_id = self._line_id_current()
        if self._load_group_table_from_path(path, line_id):
            self._sync_nurbs_defaults_from_group_table()
        else:
            self._warn("[UI3] Cannot load group_info file.")

    def _load_saved_global_fit_curve(self, line_id: str) -> Optional[Dict[str, np.ndarray]]:
        curve_path = os.path.join(self._curve_dir(), f"nurbs_{line_id}.json")
        if not os.path.exists(curve_path):
            return None
        try:
            with open(curve_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            points = list(data.get("points", []) or [])
            chain = []
            elev = []
            for pt in points:
                try:
                    s_val = float(pt.get("chainage_m", np.nan))
                    z_val = float(pt.get("z", np.nan))
                except Exception:
                    continue
                if np.isfinite(s_val) and np.isfinite(z_val):
                    chain.append(float(s_val))
                    elev.append(float(z_val))
            if len(chain) < 2:
                return None
            return {
                "chain": np.asarray(chain, dtype=float),
                "elev": np.asarray(elev, dtype=float),
            }
        except Exception:
            return None

    def _apply_loaded_global_fit_state(self, line_id: str, data: Dict[str, Any]) -> bool:
        curve_payload = dict(data.get("curve", {}) or {})
        curve = {
            "chain": np.asarray(curve_payload.get("chain", []), dtype=float),
            "elev": np.asarray(curve_payload.get("elev", []), dtype=float),
        }
        if curve["chain"].size < 2 or curve["elev"].size != curve["chain"].size:
            loaded_curve = self._load_saved_global_fit_curve(line_id)
            if loaded_curve is None:
                return False
            curve = loaded_curve
        mask = np.isfinite(curve["chain"]) & np.isfinite(curve["elev"])
        curve["chain"] = curve["chain"][mask]
        curve["elev"] = curve["elev"][mask]
        if curve["chain"].size < 2:
            return False
        theta_csv_path = str(data.get("theta_csv_path", "") or self._theta_csv_path_for(line_id))
        result = {
            "curve_method": "global_fit_spline",
            "representation": str(data.get("mode", data.get("representation", "global_forward_fit_spline")) or "global_forward_fit_spline"),
            "short_length_m": float(data.get("short_length_m", 0.1)),
            "fit_parameterization": str(data.get("fit_parameterization", "chord_length") or "chord_length"),
            "theta_csv_path": theta_csv_path,
            "theta_rows": list(data.get("theta_rows", []) or []),
            "fit_points": list(data.get("fit_points", []) or []),
            "steps": list(data.get("steps", []) or []),
            "boundary_intersections": list(data.get("boundary_intersections", []) or []),
            "curve": curve,
        }
        self._set_curve_method_for_line(line_id, "global_fit_spline")
        self._active_global_fit_result = result
        self._active_curve = {"chain": curve["chain"], "elev": curve["elev"]}
        self._update_global_fit_debug_panel(result, theta_csv_path=theta_csv_path)
        self._draw_global_fit_debug_overlay(result, draw_curve=not self._static_nurbs_bg_loaded)
        return True

    def _load_nurbs_table_from_path(self, path: str, line_id: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            if str(data.get("curve_method", "") or "").strip().lower() == "global_fit_spline":
                if self._apply_loaded_global_fit_state(line_id, data):
                    self._log(f"[UI3] Loaded global fit spline info: {path}")
                    return True
                return False
            self._set_nurbs_seed_method_for_line(line_id, data.get("nurbs_seed_method"), sync_ui=False)
            cp_items = data.get("control_points", []) or []
            if not cp_items:
                return False
            cp_items = sorted(cp_items, key=lambda d: int(d.get("cp_index", 0)) if str(d.get("cp_index", "")).strip() else 0)
            cps = []
            ws = []
            for cp in cp_items:
                try:
                    s = float(cp.get("chainage_m"))
                    z = float(cp.get("elev_m"))
                    w = float(cp.get("weight", 1.0))
                except Exception:
                    continue
                if not (np.isfinite(s) and np.isfinite(z)):
                    continue
                cps.append([float(s), float(z)])
                ws.append(float(w) if np.isfinite(w) and w > 0 else 1.0)
            if len(cps) < 2:
                return False
            params = {"degree": int(data.get("degree", 3)), "control_points": cps, "weights": ws if len(ws) == len(cps) else [1.0] * len(cps)}
            if not self._active_prof:
                self._active_prof = self._build_profile_for_current_line()
            groups_now = self._read_groups_from_table() or []
            if groups_now and self._active_prof:
                try:
                    groups_now = self._backend.clamp_groups(self._active_prof, groups_now, min_len=WORKFLOW_GROUP_MIN_LEN_M)
                except Exception:
                    pass
            self._active_groups = groups_now
            if self._active_prof:
                params = self._reconcile_nurbs_params_with_groups(line_id, self._active_prof, self._active_groups, self._active_base_curve or {}, params)
            self._set_nurbs_params_for_line(line_id, params)
            n_ctrl = len(params.get("control_points", []) or [])
            deg = max(1, min(int(params.get("degree", 3)), n_ctrl - 1))
            self._nurbs_updating_ui = True
            try:
                self.nurbs_cp_spin.setValue(n_ctrl)
                self.nurbs_deg_spin.setMaximum(max(1, n_ctrl - 1))
                self.nurbs_deg_spin.setValue(deg)
                method_idx = self.nurbs_seed_method_combo.findData(self._get_nurbs_seed_method_for_line(line_id))
                self.nurbs_seed_method_combo.setCurrentIndex(method_idx if method_idx >= 0 else 0)
                self._populate_nurbs_table(params)
            finally:
                self._nurbs_updating_ui = False
            self._draw_control_points_overlay(params)
            self._set_curve_method_for_line(line_id, "nurbs")
            self._schedule_nurbs_live_update()
            self._log(f"[UI3] Loaded NURBS table from: {path}")
            return True
        except Exception as e:
            self._log(f"[!] Cannot read curve nurbs_info file: {e}")
            return False

    def _try_load_nurbs_table_from_curve(self, line_id: str) -> bool:
        path = self._curve_nurbs_info_json_path_for(line_id)
        return self._load_nurbs_table_from_path(path, line_id)

    def _on_load_nurbs_info(self) -> None:
        if self.line_combo.count() == 0:
            self._warn("[UI3] No line selected.")
            return
        try:
            start_dir = self._curve_dir()
        except Exception:
            start_dir = self.base_dir
        path, _ = QFileDialog.getOpenFileName(self, "Load nurbs_info", start_dir, "JSON files (*.json);;All files (*.*)")
        if not path:
            return
        line_id = self._line_id_current()
        if not self._load_nurbs_table_from_path(path, line_id):
            self._warn("[UI3] Cannot load nurbs_info file.")

    def _try_load_nurbs_preview_from_curve(self, line_id: str) -> bool:
        path = self._curve_nurbs_png_path_for(line_id)
        if not os.path.exists(path):
            return False
        try:
            if not self._load_preview_scene_from_path(path, static_nurbs_bg=True):
                return False
            self._log(f"[UI3] Loaded saved curve preview: {path}")
            return True
        except Exception:
            return False

    def _build_profile_for_current_line(self) -> Optional[dict]:
        try:
            if not hasattr(self, "_gdf") or self._gdf is None or self._gdf.empty:
                return None
            row = self.line_combo.currentIndex()
            if row < 0:
                return None
            geom = self._gdf.geometry.iloc[row]
            prof = self._compute_profile_for_geom(geom, slip_only=False)
            if not prof or len(prof.get("chain", [])) < 2:
                return None
            return prof
        except Exception:
            return None

    def _load_saved_curve_state_for_current_line(self) -> None:
        line_id = self._line_id_current()
        self._populate_group_table_for_current_line()
        self._set_curve_method_for_line(line_id, "global_fit_spline")
        prof = self._build_profile_for_current_line()
        if prof is not None:
            self._active_prof = prof

        groups = self._read_groups_from_table() or []
        if groups and self._active_prof:
            try:
                groups = self._backend.clamp_groups(self._active_prof, groups, min_len=WORKFLOW_GROUP_MIN_LEN_M)
            except Exception:
                pass
        self._active_groups = groups
        self._active_global_fit_result = None

        loaded_preview = self._try_load_nurbs_preview_from_curve(line_id)
        self._try_load_nurbs_table_from_curve(line_id)
        if not loaded_preview:
            self._log("[i] No saved curve preview in ui3/curve. Click 'Render Section' to preview.")
        self._refresh_anchor_overlay()

    def _populate_group_table_for_current_line(self) -> None:
        path = self._groups_json_path()
        self._group_table_updating = True
        try:
            loaded = 0
            if os.path.exists(path):
                try:
                    _data, groups = self._load_group_json_data(path, line_id=self._line_id_current(), apply_settings=True)
                    loaded = self._populate_group_table_rows(groups, length_m=self._sec_len_m)
                except Exception as e:
                    self._log(f"[!] Cannot read groups: {e}")
            if loaded == 0:
                self.group_table.setRowCount(0)
                for _ in range(3):
                    r = self.group_table.rowCount()
                    self.group_table.insertRow(r)
                    self.group_table.setItem(r, 0, QTableWidgetItem(""))
        finally:
            self._group_table_updating = False

    def _render_current(self) -> None:
        if not hasattr(self, "_gdf") or self._gdf is None or self._gdf.empty:
            self._log("[!] No lines.")
            return
        row = self.line_combo.currentIndex()
        if row < 0:
            self._log("[!] Select a line first.")
            return

        self._log(f"[UI3] DEM: {self.dem_path}")
        self._log(f"[UI3] DX:  {self.dx_path}")
        self._log(f"[UI3] DY:  {self.dy_path}")
        self._log(f"[UI3] DZ:  {self.dz_path}")
        self._log(f"[UI3] MASK:{self.slip_path}")

        geom = self._gdf.geometry.iloc[row]
        prof = self._compute_profile_for_geom(geom, slip_only=False)
        if not prof:
            self._log("[!] Empty profile.")
            return

        groups = self._load_groups_for_current_line()
        if groups:
            groups = self._backend.clamp_groups(prof, groups, min_len=WORKFLOW_GROUP_MIN_LEN_M)

        line_id = self._line_id_current()
        out_png = self._profile_png_path_for(line_id)
        path = self._render_profile_png_current_settings(prof, out_png, groups=groups if groups else None)
        if not path or not os.path.exists(path):
            return
        if not self._load_preview_scene_from_path(path, static_nurbs_bg=False):
            self._err("[UI3] Cannot load PNG with curve overlay.")
            return
        self._active_prof = prof
        self._active_groups = groups if groups else []
        self._active_base_curve = None
        self._active_curve = None
        self._active_global_fit_result = None

        try:
            ground_csv = self._save_ground_csv_for_line(line_id, geom, step_m=float(self.step_box.value()))
            if ground_csv:
                self._log(f"[UI3] Saved ground CSV: {ground_csv}")
        except Exception as e:
            self._warn(f"[UI3] Cannot save ground CSV: {e}")

        try:
            vectors_csv = self._save_vectors_csv_for_line(line_id, prof)
            if vectors_csv:
                self._log(f"[UI3] Saved vectors CSV: {vectors_csv}")
        except Exception as e:
            self._warn(f"[UI3] Cannot save vectors CSV: {e}")

        try:
            if groups:
                theta_csv = self._save_theta_csv_for_line(line_id, prof, groups)
                if theta_csv:
                    self._log(f"[UI3] Saved theta CSV: {theta_csv}")
                    self._group_table_updating = True
                    try:
                        self._populate_group_table_rows(groups, length_m=self._sec_len_m)
                    finally:
                        self._group_table_updating = False
                    self._update_global_fit_debug_panel(None, theta_csv_path=theta_csv)
        except Exception as e:
            self._warn(f"[UI3] Cannot save theta CSV: {e}")

    def _render_current_safe(self) -> None:
        try:
            self._render_current()
        except Exception:
            msg = "[UI3] Render Section failed. See log for details."
            self._append_status(msg)
            log_path = ""
            try:
                run_dir = (self._ctx.get("run_dir") or "").strip()
                if run_dir:
                    log_dir = os.path.join(run_dir, "ui3")
                    os.makedirs(log_dir, exist_ok=True)
                    log_path = os.path.join(log_dir, "ui3_render_error.log")
                    with open(log_path, "a", encoding="utf-8") as f:
                        import traceback
                        f.write(traceback.format_exc())
                        f.write("\n")
            except Exception:
                pass
            try:
                details = msg
                if log_path:
                    details += f"\nLog: {log_path}"
                QMessageBox.critical(self, "UI3 Render Section Error", details)
            except Exception:
                pass
