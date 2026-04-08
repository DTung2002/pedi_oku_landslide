import json
import os
from typing import Any, Dict, Optional

import numpy as np
from PyQt5.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QTableWidgetItem

from .ui3_widgets import KeyboardOnlyDoubleSpinBox

WORKFLOW_GROUP_MIN_LEN_M = 0.0


class UI3CurvePanelMixin:
    @staticmethod
    def _normalize_curve_method(method: Optional[str]) -> str:
        m = str(method or "").strip().lower()
        if m == "nurbs":
            return "nurbs"
        return "bezier"

    @staticmethod
    def _curve_method_from_group_method(group_method: Optional[str]) -> str:
        gm = str(group_method or "").strip().lower()
        if gm in (
            "traditional",
            "new",
            "test",
            "raw_dem_curvature",
            "profile_dem_rdp_curvature_theta0",
            "profile_dem_rdp_curvature_only",
            "profile_dem_rdp_theta0_only",
            "profile_dem_rdp_span_only",
        ):
            return "nurbs"
        return "nurbs"

    def _set_curve_method_for_line(self, line_id: str, curve_method: Optional[str]) -> str:
        cm = self._normalize_curve_method(curve_method)
        self._curve_method_by_line[line_id] = cm
        return cm

    def _get_curve_method_for_line(self, line_id: str) -> str:
        cm = self._curve_method_by_line.get(line_id, "")
        if cm:
            return self._normalize_curve_method(cm)

        js_path = self._groups_json_path_for(line_id)
        if os.path.exists(js_path):
            try:
                with open(js_path, "r", encoding="utf-8") as f:
                    js = json.load(f) or {}
                cm = str(js.get("curve_method", "")).strip().lower()
                if not cm:
                    cm = self._curve_method_from_group_method(js.get("group_method"))
                cm = self._set_curve_method_for_line(line_id, cm)
                return cm
            except Exception:
                pass
        return self._set_curve_method_for_line(line_id, "bezier")

    def _profile_endpoints(self, prof: dict):
        return self._backend.profile_endpoints(
            prof,
            rdp_eps_m=float(self._grouping_params_current().get("rdp_eps_m", 0.5)),
            smooth_radius_m=float(self._grouping_params_current().get("smooth_radius_m", 0.0)),
        )

    def _nurbs_endpoint_targets(self, prof: dict, groups: Optional[list] = None, line_id: Optional[str] = None):
        lid = line_id or self._line_id_current()
        grouped = self._backend.grouped_vector_endpoints(prof, groups or self._active_groups or [])
        base = grouped if grouped is not None else self._profile_endpoints(prof)
        if base is None:
            return None
        if self._current_ui2_line_role() != "cross":
            return base
        anchors = self._anchors_for_cross_line(self._current_ui2_line_id(), require_ready=True)
        if not anchors:
            return base
        return self._backend.extend_endpoint_targets_with_cross_anchors(prof, base, anchors)

    def _build_default_nurbs_params(self, line_id: str, prof: dict, groups: list, base_curve: dict) -> Dict[str, Any]:
        ends = self._nurbs_endpoint_targets(prof, groups, line_id=line_id)
        if ends is None:
            return {"degree": 1, "control_points": [], "weights": []}
        params = self._backend.build_default_nurbs_params(
            prof=prof,
            groups=groups,
            base_curve=base_curve,
            endpoints=ends,
        )
        self._nurbs_params_by_line[line_id] = params
        return params

    def _reconcile_nurbs_params_with_groups(
        self,
        line_id: str,
        prof: dict,
        groups: list,
        base_curve: dict,
        params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        ends = self._nurbs_endpoint_targets(prof, groups, line_id=line_id)
        if ends is None:
            return self._build_default_nurbs_params(line_id, prof, groups, base_curve)
        return self._backend.reconcile_nurbs_params_with_groups(
            prof=prof,
            groups=groups,
            base_curve=base_curve,
            params=params,
            endpoints=ends,
        )

    def _constrain_curve_to_cross_anchors(self, curve: Optional[Dict[str, np.ndarray]]) -> Optional[Dict[str, np.ndarray]]:
        if not curve:
            return curve
        if self._current_ui2_line_role() != "cross":
            return curve
        anchors = self._anchors_for_cross_line(self._current_ui2_line_id(), require_ready=True)
        return self._backend.constrain_curve_to_cross_anchors(curve, anchors)

    def _get_nurbs_params_for_line(self, line_id: str) -> Optional[Dict[str, Any]]:
        return self._nurbs_params_by_line.get(line_id)

    def _set_nurbs_params_for_line(self, line_id: str, params: Dict[str, Any]) -> None:
        self._nurbs_params_by_line[line_id] = params

    def _try_load_nurbs_params_file(self, line_id: str) -> Optional[Dict[str, Any]]:
        path = self._nurbs_json_path_for(line_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                js = json.load(f) or {}
            params = {
                "degree": int(js.get("degree", 3)),
                "control_points": js.get("control_points", []),
                "weights": js.get("weights", []),
            }
            cps = np.asarray(params.get("control_points", []), dtype=float)
            ws = np.asarray(params.get("weights", []), dtype=float)
            if cps.ndim != 2 or cps.shape[0] < 2:
                return None
            if ws.ndim != 1 or ws.size != cps.shape[0]:
                params["weights"] = np.ones(cps.shape[0], dtype=float).tolist()
            return params
        except Exception:
            return None

    def _sync_nurbs_panel_for_current_line(self, reset_defaults: bool = False) -> None:
        if self.line_combo is None or self.line_combo.count() == 0:
            return
        line_id = self._line_id_current()
        prof = self._active_prof
        if not prof:
            return
        groups = self._active_groups or []
        base = self._active_base_curve or {}

        params = None if reset_defaults else self._get_nurbs_params_for_line(line_id)
        if (not params) and (not reset_defaults):
            params = self._try_load_nurbs_params_file(line_id)
        if not params:
            params = self._build_default_nurbs_params(line_id, prof, groups, base)
        else:
            params = self._reconcile_nurbs_params_with_groups(line_id, prof, groups, base, params)

        cps = params.get("control_points", []) or []
        ww = params.get("weights", []) or []
        deg = int(params.get("degree", 3))
        n_ctrl = max(2, len(cps))
        deg = max(1, min(deg, n_ctrl - 1))
        params["degree"] = deg
        if len(ww) != n_ctrl:
            ww = [1.0] * n_ctrl
            params["weights"] = ww
        self._set_nurbs_params_for_line(line_id, params)

        self._nurbs_updating_ui = True
        try:
            self.nurbs_cp_spin.setValue(n_ctrl)
            self.nurbs_deg_spin.setMaximum(max(1, n_ctrl - 1))
            self.nurbs_deg_spin.setValue(deg)
            self._populate_nurbs_table(params)
        finally:
            self._nurbs_updating_ui = False
        self._draw_control_points_overlay(params)

    def _populate_nurbs_table(self, params: Dict[str, Any]) -> None:
        cps = params.get("control_points", []) or []
        ww = params.get("weights", []) or []
        self.nurbs_table.setRowCount(0)
        for i, cp in enumerate(cps):
            r = self.nurbs_table.rowCount()
            self.nurbs_table.insertRow(r)
            item = QTableWidgetItem(f"CP{i}")
            self.nurbs_table.setItem(r, 0, item)

            sbox = KeyboardOnlyDoubleSpinBox()
            sbox.setDecimals(3)
            sbox.setRange(-1e6, 1e6)
            sbox.setSingleStep(0.1)
            sbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
            sbox.setValue(float(cp[0]))
            sbox.valueChanged.connect(lambda _v, _r=r: self._on_nurbs_table_changed(_r))
            self.nurbs_table.setCellWidget(r, 1, sbox)

            zbox = KeyboardOnlyDoubleSpinBox()
            zbox.setDecimals(3)
            zbox.setRange(-1e6, 1e6)
            zbox.setSingleStep(0.1)
            zbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
            zbox.setValue(float(cp[1]))
            zbox.valueChanged.connect(lambda _v, _r=r: self._on_nurbs_table_changed(_r))
            self.nurbs_table.setCellWidget(r, 2, zbox)

            wbox = KeyboardOnlyDoubleSpinBox()
            wbox.setDecimals(3)
            wbox.setRange(0.001, 1e6)
            wbox.setSingleStep(0.1)
            wbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
            wbox.setValue(float(ww[i] if i < len(ww) else 1.0))
            wbox.valueChanged.connect(lambda _v, _r=r: self._on_nurbs_table_changed(_r))
            self.nurbs_table.setCellWidget(r, 3, wbox)

        self._enforce_nurbs_endpoint_lock()

    def _collect_nurbs_params_from_ui(self) -> Optional[Dict[str, Any]]:
        rc = self.nurbs_table.rowCount()
        if rc < 2:
            return None
        cps = []
        ws = []
        for r in range(rc):
            sbox = self.nurbs_table.cellWidget(r, 1)
            zbox = self.nurbs_table.cellWidget(r, 2)
            wbox = self.nurbs_table.cellWidget(r, 3)
            if not isinstance(sbox, QDoubleSpinBox) or not isinstance(zbox, QDoubleSpinBox) or not isinstance(wbox, QDoubleSpinBox):
                return None
            cps.append([float(sbox.value()), float(zbox.value())])
            ws.append(float(max(0.001, wbox.value())))
        cps_arr = np.asarray(cps, dtype=float)
        order = np.argsort(cps_arr[:, 0])
        cps_arr = cps_arr[order]
        ws_arr = np.asarray(ws, dtype=float)[order]
        deg = int(self.nurbs_deg_spin.value())
        deg = max(1, min(deg, len(cps) - 1))
        return {
            "degree": deg,
            "control_points": cps_arr.tolist(),
            "weights": ws_arr.tolist(),
        }

    def _enforce_nurbs_endpoint_lock(self) -> None:
        prof = self._active_prof
        if not prof:
            return
        ends = self._nurbs_endpoint_targets(prof, self._active_groups, line_id=self._line_id_current())
        if ends is None:
            return
        s0, z0, s1, z1 = ends
        rc = self.nurbs_table.rowCount()
        if rc < 2:
            return
        for row, s_val, z_val in ((0, s0, z0), (rc - 1, s1, z1)):
            sbox = self.nurbs_table.cellWidget(row, 1)
            zbox = self.nurbs_table.cellWidget(row, 2)
            if isinstance(sbox, QDoubleSpinBox):
                sbox.blockSignals(True)
                sbox.setValue(float(s_val))
                sbox.setEnabled(False)
                sbox.blockSignals(False)
            if isinstance(zbox, QDoubleSpinBox):
                zbox.blockSignals(True)
                zbox.setValue(float(z_val))
                zbox.setEnabled(False)
                zbox.blockSignals(False)

    def _resize_nurbs_control_points(self, new_count: int) -> None:
        line_id = self._line_id_current()
        params = self._get_nurbs_params_for_line(line_id)
        prof = self._active_prof
        if params is None or prof is None:
            return
        ends = self._nurbs_endpoint_targets(prof, self._active_groups, line_id=line_id)
        if ends is None:
            return
        s0, z0, s1, z1 = ends
        cps = np.asarray(params.get("control_points", []), dtype=float)
        ws = np.asarray(params.get("weights", []), dtype=float)
        if cps.ndim != 2 or cps.shape[0] < 2:
            params = self._build_default_nurbs_params(line_id, prof, self._active_groups, self._active_base_curve or {})
            cps = np.asarray(params.get("control_points", []), dtype=float)
            ws = np.asarray(params.get("weights", []), dtype=float)
        old_s = cps[:, 0]
        old_z = cps[:, 1]
        new_s = np.linspace(s0, s1, int(max(2, new_count)))
        new_z = np.interp(new_s, old_s, old_z)
        new_w = np.interp(new_s, old_s, ws if ws.size == old_s.size else np.ones_like(old_s))
        new_z[0] = z0
        new_z[-1] = z1
        new_params = {
            "degree": int(min(int(params.get("degree", 3)), len(new_s) - 1)),
            "control_points": np.vstack([new_s, new_z]).T.tolist(),
            "weights": np.where(np.isfinite(new_w) & (new_w > 0), new_w, 1.0).tolist(),
        }
        self._set_nurbs_params_for_line(line_id, new_params)
        self._sync_nurbs_panel_for_current_line(reset_defaults=False)
        self._schedule_nurbs_live_update()

    def _schedule_nurbs_live_update(self) -> None:
        if self._nurbs_updating_ui:
            return
        self._nurbs_live_timer.start()

    def _on_nurbs_live_tick(self) -> None:
        line_id = self._line_id_current()
        params = self._collect_nurbs_params_from_ui()
        if not params:
            return
        if self._static_nurbs_bg_loaded:
            self._render_current_safe()
            self._static_nurbs_bg_loaded = False
        self._set_nurbs_params_for_line(line_id, params)
        self._draw_control_points_overlay(params)
        curve_method = self._get_curve_method_for_line(line_id)
        if curve_method != "nurbs":
            return
        curve = self._compute_nurbs_curve_from_params(params)
        if curve is None:
            return
        self._active_curve = curve
        self._draw_curve_overlay(np.asarray(curve["chain"], dtype=float), np.asarray(curve["elev"], dtype=float))

    def _compute_nurbs_curve_from_params(self, params: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        prof = self._active_prof
        if not prof:
            return None
        p = dict(params or {})
        cps = np.asarray(p.get("control_points", []), dtype=float)
        ww = np.asarray(p.get("weights", []), dtype=float)
        if cps.ndim != 2 or cps.shape[0] < 2:
            return None
        if ww.ndim != 1 or ww.size != cps.shape[0]:
            ww = np.ones(cps.shape[0], dtype=float)

        ends = self._nurbs_endpoint_targets(prof, self._active_groups, line_id=self._line_id_current())
        if ends is None:
            return None
        s0, z0, s1, z1 = ends
        cps = cps.copy()
        cps[0, 0], cps[0, 1] = s0, z0
        cps[-1, 0], cps[-1, 1] = s1, z1

        deg = int(max(1, min(int(p.get("degree", 3)), cps.shape[0] - 1)))
        ch = np.asarray(prof.get("chain", []), dtype=float)
        n_samples = int(max(120, np.count_nonzero(np.isfinite(ch)) * 2))
        out = self._backend.evaluate_nurbs(
            cps[:, 0],
            elev_ctrl=cps[:, 1],
            weights=ww,
            degree=deg,
            n_samples=n_samples,
        )
        sx = np.asarray(out.get("chain", []), dtype=float)
        sz = np.asarray(out.get("elev", []), dtype=float)
        m = np.isfinite(sx) & np.isfinite(sz)
        sx = sx[m]
        sz = sz[m]
        m_span = (sx >= min(s0, s1)) & (sx <= max(s0, s1))
        sx = sx[m_span]
        sz = sz[m_span]
        if sx.size < 2:
            return None
        out = {"chain": sx, "elev": sz}
        out = self._constrain_curve_to_cross_anchors(out)
        return out

    def _on_nurbs_cp_spin_changed(self, val: int) -> None:
        if self._nurbs_updating_ui:
            return
        self.nurbs_deg_spin.setMaximum(max(1, int(val) - 1))
        if self.nurbs_deg_spin.value() > self.nurbs_deg_spin.maximum():
            self.nurbs_deg_spin.setValue(self.nurbs_deg_spin.maximum())
        self._resize_nurbs_control_points(int(val))

    def _on_nurbs_deg_spin_changed(self, val: int) -> None:
        if self._nurbs_updating_ui:
            return
        max_deg = max(1, self.nurbs_cp_spin.value() - 1)
        if val > max_deg:
            self._nurbs_updating_ui = True
            try:
                self.nurbs_deg_spin.setValue(max_deg)
            finally:
                self._nurbs_updating_ui = False
        self._schedule_nurbs_live_update()

    def _on_nurbs_table_changed(self, _row: int) -> None:
        self._enforce_nurbs_endpoint_lock()
        self._schedule_nurbs_live_update()

    def _on_nurbs_reset_defaults(self) -> None:
        if not self._active_prof:
            self._warn("[UI3] Render/Draw curve first to initialize NURBS.")
            return
        self._sync_nurbs_panel_for_current_line(reset_defaults=True)
        self._schedule_nurbs_live_update()

    def _on_nurbs_save(self) -> None:
        if not self._active_prof:
            self._warn("[UI3] No active profile to save NURBS.")
            return
        line_id = self._line_id_current()
        params = self._collect_nurbs_params_from_ui()
        if not params:
            self._warn("[UI3] Invalid NURBS parameters.")
            return
        curve = self._compute_nurbs_curve_from_params(params)
        if curve is None:
            self._warn("[UI3] Cannot evaluate NURBS curve.")
            return
        out_png = self._nurbs_png_path_for(line_id)
        path = self._render_profile_png_current_settings(
            self._active_prof,
            out_png,
            groups=self._active_groups if self._active_groups else None,
            overlay_curves=[(curve["chain"], curve["elev"], "#bf00ff", "Slip curve")],
        )
        if not path or not os.path.exists(path):
            return

        curve_chain = np.asarray(curve.get("chain", []), dtype=float)
        curve_elev = np.asarray(curve.get("elev", []), dtype=float)
        m_curve = np.isfinite(curve_chain) & np.isfinite(curve_elev)
        curve_chain = curve_chain[m_curve]
        curve_elev = curve_elev[m_curve]

        x_curve = np.full(curve_chain.shape, np.nan, dtype=float)
        y_curve = np.full(curve_chain.shape, np.nan, dtype=float)
        prof_chain = np.asarray(self._active_prof.get("chain", []), dtype=float)
        prof_x = np.asarray(self._active_prof.get("x", []), dtype=float)
        prof_y = np.asarray(self._active_prof.get("y", []), dtype=float)
        if prof_chain.size == prof_x.size and prof_chain.size == prof_y.size:
            m_xy = np.isfinite(prof_chain) & np.isfinite(prof_x) & np.isfinite(prof_y)
            if int(np.count_nonzero(m_xy)) >= 2 and curve_chain.size > 0:
                ch = prof_chain[m_xy]
                xx = prof_x[m_xy]
                yy = prof_y[m_xy]
                order = np.argsort(ch)
                ch = ch[order]
                xx = xx[order]
                yy = yy[order]
                ch_u, uniq_idx = np.unique(ch, return_index=True)
                xx_u = xx[uniq_idx]
                yy_u = yy[uniq_idx]
                if ch_u.size >= 2:
                    x_curve = np.interp(curve_chain, ch_u, xx_u)
                    y_curve = np.interp(curve_chain, ch_u, yy_u)

        curve_rows = []
        for i, (s, x, y, z) in enumerate(zip(curve_chain, x_curve, y_curve, curve_elev)):
            curve_rows.append({
                "index": int(i),
                "chainage_m": float(s),
                "x": (float(x) if np.isfinite(x) else None),
                "y": (float(y) if np.isfinite(y) else None),
                "z": float(z),
            })
        groups_ui3_json = self._groups_json_path_for(line_id)
        group_method = None
        if os.path.exists(groups_ui3_json):
            try:
                with open(groups_ui3_json, "r", encoding="utf-8") as f:
                    old_js = json.load(f) or {}
                if old_js.get("group_method", None):
                    group_method = str(old_js.get("group_method"))
            except Exception:
                pass

        groups_ui3_payload = self._backend.build_group_json_payload(
            line_label=self.line_combo.currentText(),
            groups=self._read_groups_from_table(),
            prof=self._active_prof,
            chainage_origin=self._ui3_chainage_origin(),
            curve_method=self._get_curve_method_for_line(line_id),
            profile_dem_source=self._current_profile_source_key(),
            profile_dem_path=str(getattr(self, "dem_path", "") or ""),
            grouping_params=self._grouping_params_current(),
            group_method=group_method,
        )
        cps = np.asarray(params.get("control_points", []), dtype=float)
        ww = np.asarray(params.get("weights", []), dtype=float)
        if ww.ndim != 1 or ww.size != cps.shape[0]:
            ww = np.ones(cps.shape[0], dtype=float)
        cp_rows = []
        for i in range(cps.shape[0]):
            cp_rows.append({
                "cp_index": int(i),
                "chainage_m": float(cps[i, 0]),
                "elev_m": float(cps[i, 1]),
                "weight": float(ww[i]),
            })

        saved = self._backend.save_nurbs_outputs(
            {
                "preview_params": {
                    "path": self._nurbs_json_path_for(line_id),
                    "payload": {
                        "line_id": line_id,
                        "curve_method": "nurbs",
                        "degree": int(params.get("degree", 3)),
                        "control_points": params.get("control_points", []),
                        "weights": params.get("weights", []),
                        "curve": {
                            "chain": np.asarray(curve["chain"], dtype=float).tolist(),
                            "elev": np.asarray(curve["elev"], dtype=float).tolist(),
                        },
                    },
                },
                "curve_json": {
                    "path": os.path.join(self._curve_dir(), f"nurbs_{line_id}.json"),
                    "payload": {
                        "line_id": line_id,
                        "curve_method": "nurbs",
                        "chainage_origin": self._ui3_chainage_origin(),
                        "count": int(len(curve_rows)),
                        "points": curve_rows,
                    },
                },
                "group_json": {"path": groups_ui3_json, "payload": groups_ui3_payload},
                "nurbs_info": {
                    "path": os.path.join(self._curve_dir(), f"nurbs_info_{line_id}.json"),
                    "payload": {
                        "line_id": line_id,
                        "chainage_origin": self._ui3_chainage_origin(),
                        "control_points_count": int(cps.shape[0]),
                        "degree": int(params.get("degree", 3)),
                        "control_points": cp_rows,
                    },
                },
            }
        )
        self._ok(f"[UI3] Saved NURBS: {path}")
        self._log(f"[UI3] Saved NURBS params: {saved.get('preview_params', self._nurbs_json_path_for(line_id))}")
        self._log(f"[UI3] Saved NURBS curve: {saved.get('curve_json', '')}")
        self._log(f"[UI3] Saved groups JSON: {saved.get('group_json', groups_ui3_json)}")
        self._log(f"[UI3] Saved NURBS info: {saved.get('nurbs_info', '')}")
        try:
            anchor_path, n_upd = self._update_anchors_xyz_for_saved_main_curve(curve)
            if anchor_path and n_upd > 0:
                self._log(f"[UI3] Updated anchors_xyz: {anchor_path} (n={n_upd})")
        except Exception as e:
            self._warn(f"[UI3] Cannot update anchors_xyz: {e}")
        self._refresh_anchor_overlay()
        try:
            self.curve_saved.emit(str(saved.get("curve_json", "")))
        except Exception:
            pass

    def _on_draw_curve(self) -> None:
        try:
            line_id = self._line_id_current()
            _prev_curve_method = self._get_curve_method_for_line(line_id)
            curve_method = self._set_curve_method_for_line(line_id, "nurbs")
            if _prev_curve_method != "nurbs":
                self._log(f"[UI3] Curve method for '{line_id}': forced to NURBS (Bezier-like seed)")
            else:
                self._log(f"[UI3] Curve method for '{line_id}': NURBS (Bezier-like seed)")

            if not hasattr(self, "_gdf") or self._gdf is None or self._gdf.empty:
                self._warn("[UI3] No lines.")
                return
            row = self.line_combo.currentIndex()
            if row < 0:
                self._warn("[UI3] Select a line first.")
                return

            geom = self._gdf.geometry.iloc[row]
            prof = self._compute_profile_for_geom(geom, slip_only=False)
            if not prof or len(prof.get("chain", [])) < 6:
                self._warn("[UI3] Empty/too-short slip profile.")
                return

            groups = self._load_groups_for_current_line()
            if not groups:
                groups = self._backend.auto_group_profile(
                    prof,
                    self._grouping_params_current(),
                    min_len=WORKFLOW_GROUP_MIN_LEN_M,
                )
                if not groups:
                    self._warn("[UI3] Auto grouping produced no segments within slip zone.")
                    return
                self._save_groups_to_ui(
                    groups,
                    prof,
                    line_id,
                    log_text=f"[UI3] Auto Group (implicit) for '{line_id}': {len(groups)} groups.",
                    curve_method=curve_method,
                )
            else:
                groups = self._backend.clamp_groups(prof, groups, min_len=WORKFLOW_GROUP_MIN_LEN_M)
                if not groups:
                    self._warn("[UI3] No groups within slip zone.")
                    return

            base = self._backend.build_curve_seed(
                prof,
                groups,
                {"ds": 0.2, "smooth_factor": 0.06, "depth_gain": 14, "min_depth": 5},
            )
            x_base = np.asarray(base.get("chain", []), dtype=float)
            z_base = np.asarray(base.get("elev", []), dtype=float)
            mask = np.isfinite(x_base) & np.isfinite(z_base)
            x_base = x_base[mask]
            z_base = z_base[mask]
            if x_base.size < 2:
                self._warn("[UI3] Slip curve has too few valid points.")
                return
            order = np.argsort(x_base)
            x_base = x_base[order]
            z_base = z_base[order]
            self._log(f"[UI3] Slip curve pts={x_base.size}, chain=[{x_base.min():.2f}, {x_base.max():.2f}]")

            curve = {"chain": x_base, "elev": z_base}
            try:
                bez = self._backend.fit_bezier_curve_seed(
                    np.asarray(prof["chain"], dtype=float),
                    np.asarray(prof["elev_s"], dtype=float),
                    x_base,
                    z_base,
                    {"c0": 0.20, "c1": 0.40, "clearance": 0.35},
                )
                xb = np.asarray(bez.get("chain", []), dtype=float)
                zb = np.asarray(bez.get("elev", []), dtype=float)
                m2 = np.isfinite(xb) & np.isfinite(zb)
                xb = xb[m2]
                zb = zb[m2]
                if xb.size >= 2:
                    self._log(f"[UI3] Bezier-like seed curve OK: n={xb.size}")
                    curve = {"chain": xb, "elev": zb}
                else:
                    self._warn("[UI3] Bezier-like seed has too few points; using base target.")
            except Exception as e:
                self._warn(f"[UI3] Bezier-like seed fit failed, using base target. ({e})")

            self._active_prof = prof
            self._active_groups = groups
            self._active_base_curve = {"chain": np.asarray(curve["chain"], dtype=float), "elev": np.asarray(curve["elev"], dtype=float)}
            self._sync_nurbs_panel_for_current_line(reset_defaults=False)

            params_now = self._collect_nurbs_params_from_ui()
            nurbs_curve = self._compute_nurbs_curve_from_params(params_now) if params_now else None
            if nurbs_curve is None:
                self._warn("[UI3] Current NURBS params invalid/unusable; reset to Bezier-like NURBS seed.")
                self._sync_nurbs_panel_for_current_line(reset_defaults=True)
                params_now = self._collect_nurbs_params_from_ui()
                nurbs_curve = self._compute_nurbs_curve_from_params(params_now) if params_now else None
            if nurbs_curve is not None:
                curve = nurbs_curve
                self._log(f"[UI3] NURBS slip curve OK: n={len(nurbs_curve['chain'])}")
            else:
                self._warn("[UI3] NURBS fit failed after reseed; showing Bezier-like seed curve.")

            out_png = self._profile_png_path_for(line_id)
            path = self._render_profile_png_current_settings(prof, out_png, groups=groups)
            if not path or not os.path.exists(path):
                return
            if not self._load_preview_scene_from_path(path, static_nurbs_bg=False):
                self._err("[UI3] Cannot load PNG with curve overlay.")
                return

            self._active_curve = {"chain": np.asarray(curve["chain"], dtype=float), "elev": np.asarray(curve["elev"], dtype=float)}
            self._draw_curve_overlay(self._active_curve["chain"], self._active_curve["elev"])
            self._draw_control_points_overlay()
            self._ok("[UI3] Curve drawn on current section.")
        except Exception as e:
            self._err(f"[UI3] Draw Curve error: {e}")
            raise

    def _sync_nurbs_defaults_from_group_table(self) -> None:
        if self._group_table_updating:
            return
        if not self._active_prof:
            return
        try:
            line_id = self._line_id_current()
            groups = self._read_groups_from_table()
            if groups:
                groups = self._backend.clamp_groups(self._active_prof, groups, min_len=WORKFLOW_GROUP_MIN_LEN_M)
            self._active_groups = groups or []
            params = self._build_default_nurbs_params(
                line_id=line_id,
                prof=self._active_prof,
                groups=self._active_groups,
                base_curve=self._active_base_curve or {},
            )
            self._set_nurbs_params_for_line(line_id, params)
            self._sync_nurbs_panel_for_current_line(reset_defaults=False)
            self._schedule_nurbs_live_update()
        except Exception as e:
            self._warn(f"[UI3] Cannot sync NURBS from groups: {e}")
