import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt5.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QTableWidgetItem

from pedi_oku_landslide.ui.widgets.ui3_widgets import KeyboardOnlyDoubleSpinBox

WORKFLOW_GROUP_MIN_LEN_M = 0.0


class UI3CurvePanelMixin:
    @staticmethod
    def _normalize_nurbs_seed_method(method: Optional[str]) -> str:
        m = str(method or "").strip().lower()
        if m == "slope_guided":
            return "slope_guided"
        return "bezier_like"

    @staticmethod
    def _normalize_curve_method(method: Optional[str]) -> str:
        m = str(method or "").strip().lower()
        if m in ("global_fit_spline", "global_forward_fit_spline"):
            return "global_fit_spline"
        if m == "nurbs":
            return "nurbs"
        return "global_fit_spline"

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
            return "global_fit_spline"
        return "global_fit_spline"

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
        return self._set_curve_method_for_line(line_id, "global_fit_spline")

    def _update_global_fit_debug_panel(
        self,
        result: Optional[Dict[str, Any]] = None,
        *,
        theta_csv_path: Optional[str] = None,
    ) -> None:
        labels = {
            "global_fit_short_length_value": "0.1 m",
            "global_fit_theta_source_value": str(theta_csv_path or "—"),
            "global_fit_fit_count_value": "0",
            "global_fit_step_count_value": "0",
            "global_fit_curve_method_value": "global_forward_fit_spline",
        }
        if result:
            labels["global_fit_short_length_value"] = f"{float(result.get('short_length_m', 0.1)):.3f} m"
            theta_path = str(theta_csv_path or result.get("theta_csv_path", "") or "—")
            labels["global_fit_theta_source_value"] = theta_path
            labels["global_fit_fit_count_value"] = str(len(result.get("fit_points", []) or []))
            labels["global_fit_step_count_value"] = str(len(result.get("steps", []) or []))
            labels["global_fit_curve_method_value"] = str(result.get("representation", "global_forward_fit_spline"))
        for attr, value in labels.items():
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setText(str(value))

    def _set_nurbs_seed_method_for_line(self, line_id: str, method: Optional[str], *, sync_ui: bool = True) -> str:
        nm = self._normalize_nurbs_seed_method(method)
        self._nurbs_seed_method_by_line[line_id] = nm
        is_current = False
        try:
            is_current = bool(line_id) and (line_id == self._line_id_current())
        except Exception:
            is_current = False
        if sync_ui and is_current:
            self._sync_nurbs_seed_method_combo(line_id)
        return nm

    def _sync_nurbs_seed_method_combo(self, line_id: Optional[str] = None) -> None:
        combo = getattr(self, "nurbs_seed_method_combo", None)
        if combo is None:
            return
        lid = line_id or self._line_id_current()
        method = self._normalize_nurbs_seed_method(self._nurbs_seed_method_by_line.get(lid, "bezier_like"))
        idx = combo.findData(method)
        if idx < 0:
            idx = 0
        combo.blockSignals(True)
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _get_nurbs_seed_method_for_line(self, line_id: str) -> str:
        method = self._nurbs_seed_method_by_line.get(line_id, "")
        if method:
            return self._normalize_nurbs_seed_method(method)
        return self._set_nurbs_seed_method_for_line(line_id, "bezier_like", sync_ui=False)

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
        points = self._active_curve_constraints(log_skips=False)
        if not points:
            return base
        return self._backend.extend_endpoint_targets_with_points(
            prof,
            base,
            points,
            chain_key="s",
            elev_key="z",
        )

    def _build_default_nurbs_params(self, line_id: str, prof: dict, groups: list, base_curve: dict) -> Dict[str, Any]:
        ends = self._nurbs_endpoint_targets(prof, groups, line_id=line_id)
        if ends is None:
            return {"degree": 1, "control_points": [], "weights": []}
        seed_method = self._get_nurbs_seed_method_for_line(line_id)
        params = self._backend.build_default_nurbs_params(
            prof=prof,
            groups=groups,
            base_curve=base_curve,
            endpoints=ends,
            nurbs_seed_method=seed_method,
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
        seed_method = self._get_nurbs_seed_method_for_line(line_id)
        return self._backend.reconcile_nurbs_params_with_groups(
            prof=prof,
            groups=groups,
            base_curve=base_curve,
            params=params,
            endpoints=ends,
            nurbs_seed_method=seed_method,
        )

    def _active_curve_constraints(self, *, log_skips: bool = False) -> List[dict]:
        _ = log_skips
        points: List[dict] = []
        if self._current_ui2_line_role() == "cross":
            anchors = self._anchors_for_cross_line(self._current_ui2_line_id(), require_ready=True)
            for anchor in anchors:
                try:
                    s_val = float(anchor.get("s_on_cross"))
                    z_val = float(anchor.get("z"))
                except Exception:
                    continue
                if not (np.isfinite(s_val) and np.isfinite(z_val)):
                    continue
                points.append(
                    {
                        "label": str(anchor.get("main_label_fixed", anchor.get("main_line_id", "")) or "").strip(),
                        "s": float(s_val),
                        "z": float(z_val),
                        "source": "cross_anchor",
                    }
                )
        points.sort(key=lambda d: (float(d.get("s", 0.0)), str(d.get("label", ""))))
        return points

    def _constrain_curve_to_active_constraints(self, curve: Optional[Dict[str, np.ndarray]], *, log_skips: bool = False) -> Optional[Dict[str, np.ndarray]]:
        if not curve:
            return curve
        points = self._active_curve_constraints(log_skips=log_skips)
        if not points:
            return curve
        return self._backend.constrain_curve_to_points(
            curve,
            points,
            chain_key="s",
            elev_key="z",
        )

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
            self._set_nurbs_seed_method_for_line(line_id, js.get("nurbs_seed_method"), sync_ui=False)
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

        seed_method = self._get_nurbs_seed_method_for_line(line_id)
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
            idx = self.nurbs_seed_method_combo.findData(seed_method)
            self.nurbs_seed_method_combo.setCurrentIndex(idx if idx >= 0 else 0)
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
        try:
            if self._get_curve_method_for_line(self._line_id_current()) != "nurbs":
                return
        except Exception:
            return
        self._nurbs_live_timer.start()

    def _on_nurbs_live_tick(self) -> None:
        line_id = self._line_id_current()
        if self._get_curve_method_for_line(line_id) != "nurbs":
            return
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
        out = self._constrain_curve_to_active_constraints(out)
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

    def _on_nurbs_seed_method_changed(self, _idx: int) -> None:
        if self._nurbs_updating_ui:
            return
        if self.line_combo is None or self.line_combo.count() == 0:
            return
        line_id = self._line_id_current()
        method = self.nurbs_seed_method_combo.currentData()
        method = self._set_nurbs_seed_method_for_line(line_id, method, sync_ui=False)
        if not self._active_prof:
            return
        self._sync_nurbs_panel_for_current_line(reset_defaults=True)
        self._schedule_nurbs_live_update()
        self._log(f"[UI3] NURBS seed method for '{line_id}': {method}")

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
        if not self._active_prof or not self._active_global_fit_result or not self._active_curve:
            self._warn("[UI3] Draw Curve first to generate a global fit spline preview.")
            return
        line_id = self._line_id_current()
        curve = {
            "chain": np.asarray((self._active_curve or {}).get("chain", []), dtype=float),
            "elev": np.asarray((self._active_curve or {}).get("elev", []), dtype=float),
        }
        m_curve = np.isfinite(curve["chain"]) & np.isfinite(curve["elev"])
        curve["chain"] = curve["chain"][m_curve]
        curve["elev"] = curve["elev"][m_curve]
        if curve["chain"].size < 2:
            self._warn("[UI3] Global fit spline preview has too few valid points.")
            return
        boring_result = self._project_boring_holes_for_current_line(use_unsaved_table=True, log_skips=True)
        boring_snapshot = {
            "distance_tolerance_m": float(boring_result.get("distance_tolerance_m", 1.0)),
            "count": int(len(boring_result.get("items", []) or [])),
            "items": list(boring_result.get("items", []) or []),
        }
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
            curve_method="global_fit_spline",
            nurbs_seed_method=self._get_nurbs_seed_method_for_line(line_id),
            profile_dem_source=self._current_profile_source_key(),
            profile_dem_path=str(getattr(self, "dem_path", "") or ""),
            grouping_params=self._grouping_params_current(),
            group_method=group_method,
        )
        theta_csv_path = self._theta_csv_path_for(line_id)
        result = dict(self._active_global_fit_result or {})
        preview_curve = {
            "chain": curve_chain.astype(float).tolist(),
            "elev": curve_elev.astype(float).tolist(),
        }

        saved = self._backend.save_global_fit_spline_outputs(
            {
                "preview_params": {
                    "path": self._nurbs_json_path_for(line_id),
                    "payload": {
                        "line_id": line_id,
                        "curve_method": "global_fit_spline",
                        "mode": "global_forward_fit_spline",
                        "short_length_m": float(result.get("short_length_m", 0.1)),
                        "fit_parameterization": str(result.get("fit_parameterization", "chord_length")),
                        "theta_csv_path": theta_csv_path,
                        "theta_rows": list(result.get("theta_rows", []) or []),
                        "fit_points": list(result.get("fit_points", []) or []),
                        "global_fit_points": list(result.get("global_fit_points", result.get("fit_points", [])) or []),
                        "steps": list(result.get("steps", []) or []),
                        "short_lines": list(result.get("short_lines", []) or []),
                        "markers": list(result.get("markers", []) or []),
                        "boundary_intersections": list(result.get("boundary_intersections", []) or []),
                        "applied_boring_holes": boring_snapshot,
                        "curve": preview_curve,
                    },
                },
                "curve_json": {
                    "path": os.path.join(self._curve_dir(), f"nurbs_{line_id}.json"),
                    "payload": {
                        "line_id": line_id,
                        "curve_method": "global_fit_spline",
                        "mode": "global_forward_fit_spline",
                        "fit_parameterization": str(result.get("fit_parameterization", "chord_length")),
                        "chainage_origin": self._ui3_chainage_origin(),
                        "applied_boring_holes": boring_snapshot,
                        "count": int(len(curve_rows)),
                        "points": curve_rows,
                    },
                },
                "group_json": {"path": groups_ui3_json, "payload": groups_ui3_payload},
                "nurbs_info": {
                    "path": os.path.join(self._curve_dir(), f"nurbs_info_{line_id}.json"),
                    "payload": {
                        "line_id": line_id,
                        "curve_method": "global_fit_spline",
                        "mode": "global_forward_fit_spline",
                        "chainage_origin": self._ui3_chainage_origin(),
                        "short_length_m": float(result.get("short_length_m", 0.1)),
                        "fit_parameterization": str(result.get("fit_parameterization", "chord_length")),
                        "theta_csv_path": theta_csv_path,
                        "theta_rows": list(result.get("theta_rows", []) or []),
                        "fit_points": list(result.get("fit_points", []) or []),
                        "global_fit_points": list(result.get("global_fit_points", result.get("fit_points", [])) or []),
                        "steps": list(result.get("steps", []) or []),
                        "short_lines": list(result.get("short_lines", []) or []),
                        "markers": list(result.get("markers", []) or []),
                        "boundary_intersections": list(result.get("boundary_intersections", []) or []),
                        "fit_point_count": int(len(result.get("fit_points", []) or [])),
                        "curve": preview_curve,
                    },
                },
            }
        )
        self._ok(f"[UI3] Saved global fit spline: {path}")
        self._log(f"[UI3] Saved preview metadata: {saved.get('preview_params', self._nurbs_json_path_for(line_id))}")
        self._log(f"[UI3] Saved curve JSON: {saved.get('curve_json', '')}")
        self._log(f"[UI3] Saved groups JSON: {saved.get('group_json', groups_ui3_json)}")
        self._log(f"[UI3] Saved curve info: {saved.get('nurbs_info', '')}")
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
            curve_method = self._set_curve_method_for_line(line_id, "global_fit_spline")
            self._log(f"[UI3] Curve method for '{line_id}': {curve_method}")

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

            length_m = None
            try:
                if "length_m" in prof and prof["length_m"] is not None:
                    length_m = float(prof["length_m"])
                else:
                    ch = np.asarray(prof.get("chain", []), dtype=float)
                    if ch.size >= 2:
                        length_m = float(ch[-1] - ch[0])
            except Exception:
                length_m = None

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
                self._group_table_updating = True
                try:
                    self._populate_group_table_rows(groups, length_m=length_m)
                finally:
                    self._group_table_updating = False

            theta_csv_path = self._save_theta_csv_for_line(line_id, prof, groups)
            if theta_csv_path:
                self._log(f"[UI3] Saved theta CSV: {theta_csv_path}")
            theta_rows = self._backend.load_theta_csv_group_angles(
                csv_path=self._theta_csv_path_for(line_id),
                groups=groups,
            )
            self._group_table_updating = True
            try:
                self._populate_group_table_rows(groups, length_m=length_m)
            finally:
                self._group_table_updating = False

            result = self._backend.build_global_forward_fit_spline(
                profile=prof,
                groups=groups,
                theta_rows=theta_rows,
            )
            curve = {
                "chain": np.asarray((result.get("curve", {}) or {}).get("chain", []), dtype=float),
                "elev": np.asarray((result.get("curve", {}) or {}).get("elev", []), dtype=float),
            }
            mask = np.isfinite(curve["chain"]) & np.isfinite(curve["elev"])
            curve["chain"] = curve["chain"][mask]
            curve["elev"] = curve["elev"][mask]
            if curve["chain"].size < 2:
                self._warn("[UI3] Global fit spline has too few valid points.")
                return

            out_png = self._profile_png_path_for(line_id)
            path = self._render_profile_png_current_settings(prof, out_png, groups=groups)
            if not path or not os.path.exists(path):
                return
            if not self._load_preview_scene_from_path(path, static_nurbs_bg=False):
                self._err("[UI3] Cannot load PNG with curve overlay.")
                return

            result["curve"] = {"chain": curve["chain"], "elev": curve["elev"]}
            result["theta_csv_path"] = self._theta_csv_path_for(line_id)
            self._active_prof = prof
            self._active_groups = groups
            self._active_base_curve = None
            self._active_curve = {"chain": curve["chain"], "elev": curve["elev"]}
            self._active_global_fit_result = result
            self._update_global_fit_debug_panel(result, theta_csv_path=self._theta_csv_path_for(line_id))
            self._draw_global_fit_debug_overlay(result, draw_curve=True)
            self._ok("[UI3] Global fit spline drawn on current section.")
        except Exception as e:
            self._err(f"[UI3] Draw Curve error: {e}")

    def _sync_nurbs_defaults_from_group_table(self) -> None:
        if self._group_table_updating:
            return
        try:
            groups = self._read_groups_from_table()
            if groups and self._active_prof:
                groups = self._backend.clamp_groups(self._active_prof, groups, min_len=WORKFLOW_GROUP_MIN_LEN_M)
            self._active_groups = groups or []
            self._active_base_curve = None
            self._active_curve = None
            self._active_global_fit_result = None
            self._clear_curve_overlay()
            self._clear_control_points_overlay()
            theta_path = self._theta_csv_path_for(self._line_id_current())
            self._update_global_fit_debug_panel(None, theta_csv_path=(theta_path if os.path.exists(theta_path) else None))
        except Exception as e:
            self._warn(f"[UI3] Cannot sync global fit spline state from groups: {e}")
