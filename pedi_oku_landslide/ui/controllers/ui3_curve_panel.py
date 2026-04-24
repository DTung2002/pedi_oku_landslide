import json
import os
from typing import Any, Dict, List, Optional, Tuple

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
        return "nurbs"

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
        return self._set_curve_method_for_line(line_id, "nurbs")

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
        _ = line_id
        base_curve = self._active_base_curve or {}
        base_chain = np.asarray(base_curve.get("chain", []), dtype=float)
        base_elev = np.asarray(base_curve.get("elev", []), dtype=float)
        base_mask = np.isfinite(base_chain) & np.isfinite(base_elev)
        if int(np.count_nonzero(base_mask)) >= 2:
            base_chain = base_chain[base_mask]
            base_elev = base_elev[base_mask]
            base = (float(base_chain[0]), float(base_elev[0]), float(base_chain[-1]), float(base_elev[-1]))
        else:
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
            span = self._cross_line_slip_span_bounds(self._active_prof) if self._active_prof else None
            anchors = self._anchors_for_cross_line(self._current_ui2_line_id(), require_ready=True)
            for anchor in anchors:
                try:
                    s_val = float(anchor.get("s_on_cross"))
                    z_val = float(anchor.get("z"))
                except Exception:
                    continue
                if not (np.isfinite(s_val) and np.isfinite(z_val)):
                    continue
                if span is not None and (s_val < span[0] or s_val > span[1]):
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
        if self._current_ui2_line_role() == "cross":
            return self._smooth_constrain_curve_to_points(curve, points)
        return self._backend.constrain_curve_to_points(
            curve,
            points,
            chain_key="s",
            elev_key="z",
        )

    @staticmethod
    def _smooth_hermite_interp(x_nodes: np.ndarray, y_nodes: np.ndarray, xq: np.ndarray) -> np.ndarray:
        x = np.asarray(x_nodes, dtype=float)
        y = np.asarray(y_nodes, dtype=float)
        q = np.asarray(xq, dtype=float)
        if x.size < 2:
            return np.full(q.shape, float(y[0]) if y.size else np.nan, dtype=float)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        x, idx = np.unique(x, return_index=True)
        y = y[idx]
        if x.size < 2:
            return np.full(q.shape, float(y[0]) if y.size else np.nan, dtype=float)
        dx = np.diff(x)
        dy = np.diff(y)
        sec = np.divide(dy, dx, out=np.zeros_like(dy), where=np.abs(dx) > 1e-12)
        slope = np.zeros_like(x)
        slope[0] = sec[0]
        slope[-1] = sec[-1]
        for i in range(1, x.size - 1):
            if sec[i - 1] * sec[i] <= 0:
                slope[i] = 0.0
            else:
                slope[i] = 0.5 * (sec[i - 1] + sec[i])
                limit = 3.0 * min(abs(sec[i - 1]), abs(sec[i]))
                slope[i] = float(np.clip(slope[i], -limit, limit))
        out = np.empty(q.shape, dtype=float)
        q_clip = np.clip(q, x[0], x[-1])
        seg = np.searchsorted(x, q_clip, side="right") - 1
        seg = np.clip(seg, 0, x.size - 2)
        h = x[seg + 1] - x[seg]
        t = np.divide(q_clip - x[seg], h, out=np.zeros_like(q_clip), where=np.abs(h) > 1e-12)
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        out[:] = (
            h00 * y[seg]
            + h10 * h * slope[seg]
            + h01 * y[seg + 1]
            + h11 * h * slope[seg + 1]
        )
        return out

    def _smooth_constrain_curve_to_points(self, curve: Dict[str, np.ndarray], points: List[dict]) -> Optional[Dict[str, np.ndarray]]:
        ch = np.asarray((curve or {}).get("chain", []), dtype=float)
        zz = np.asarray((curve or {}).get("elev", []), dtype=float)
        m = np.isfinite(ch) & np.isfinite(zz)
        ch = ch[m]
        zz = zz[m]
        if ch.size < 2:
            return curve
        order = np.argsort(ch)
        ch = ch[order]
        zz = zz[order]
        ch, idx = np.unique(ch, return_index=True)
        zz = zz[idx]
        a_s = []
        a_z = []
        for pt in points:
            try:
                s_val = float(pt.get("s"))
                z_val = float(pt.get("z"))
            except Exception:
                continue
            if np.isfinite(s_val) and np.isfinite(z_val) and ch[0] <= s_val <= ch[-1]:
                a_s.append(s_val)
                a_z.append(z_val)
        if not a_s:
            return {"chain": ch, "elev": zz}
        a_s = np.asarray(a_s, dtype=float)
        a_z = np.asarray(a_z, dtype=float)
        o = np.argsort(a_s)
        a_s = a_s[o]
        a_z = a_z[o]
        ch_aug = np.unique(np.concatenate([ch, a_s]))
        base_zz = np.interp(ch_aug, ch, zz)
        residual = a_z - np.interp(a_s, ch_aug, base_zz)
        node_x = np.concatenate([[float(ch_aug[0])], a_s, [float(ch_aug[-1])]])
        node_r = np.concatenate([[0.0], residual, [0.0]])
        corr = self._smooth_hermite_interp(node_x, node_r, ch_aug)
        zz_adj = base_zz + corr
        for s_val, z_val in zip(a_s, a_z):
            hit = np.isclose(ch_aug, s_val, rtol=0.0, atol=1e-9)
            zz_adj[hit] = z_val
        return {"chain": ch_aug, "elev": zz_adj}

    def _cross_line_slip_span_bounds(self, prof: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        chain = np.asarray((prof or {}).get("chain", []), dtype=float)
        if chain.size < 2:
            return None
        finite = np.isfinite(chain)
        if not bool(np.any(finite)):
            return None
        slip_mask = np.asarray((prof or {}).get("slip_mask", []), dtype=bool)
        if slip_mask.shape == chain.shape and bool(np.any(finite & slip_mask)):
            vals = chain[finite & slip_mask]
            s0, s1 = float(np.nanmin(vals)), float(np.nanmax(vals))
        else:
            span = (prof or {}).get("slip_span", None)
            if isinstance(span, (list, tuple)) and len(span) >= 2:
                try:
                    s0, s1 = float(span[0]), float(span[1])
                except Exception:
                    s0, s1 = float(np.nanmin(chain[finite])), float(np.nanmax(chain[finite]))
            else:
                s0, s1 = float(np.nanmin(chain[finite])), float(np.nanmax(chain[finite]))
        if not (np.isfinite(s0) and np.isfinite(s1)):
            return None
        if s1 < s0:
            s0, s1 = s1, s0
        if s1 <= s0:
            return None
        return s0, s1

    def _interp_profile_elev(self, prof: Dict[str, Any], s_val: float) -> Optional[float]:
        ch = np.asarray((prof or {}).get("chain", []), dtype=float)
        zz = np.asarray((prof or {}).get("elev_s", []), dtype=float)
        if ch.size != zz.size or ch.size < 2:
            return None
        m = np.isfinite(ch) & np.isfinite(zz)
        if int(np.count_nonzero(m)) < 2:
            return None
        ch = ch[m]
        zz = zz[m]
        order = np.argsort(ch)
        ch = ch[order]
        zz = zz[order]
        ch_u, idx = np.unique(ch, return_index=True)
        zz_u = zz[idx]
        if ch_u.size < 2:
            return None
        return float(np.interp(float(s_val), ch_u, zz_u))

    def _cross_anchor_points_in_span(self, line_id: str, s0: float, s1: float) -> List[Dict[str, Any]]:
        anchors = self._anchors_for_cross_line(line_id, require_ready=True)
        out: List[Dict[str, Any]] = []
        lo, hi = (float(s0), float(s1)) if s0 <= s1 else (float(s1), float(s0))
        for anchor in anchors:
            try:
                s = float(anchor.get("s_on_cross"))
                z = float(anchor.get("z"))
            except Exception:
                continue
            if not (np.isfinite(s) and np.isfinite(z)):
                continue
            if s < lo or s > hi:
                continue
            out.append(
                {
                    "s": float(s),
                    "z": float(z),
                    "label": str(anchor.get("main_label_fixed", anchor.get("main_line_id", "")) or "").strip(),
                    "source": "cross_anchor",
                }
            )
        out.sort(key=lambda p: (float(p["s"]), str(p.get("label", ""))))
        dedup: List[Dict[str, Any]] = []
        for p in out:
            if dedup and abs(float(p["s"]) - float(dedup[-1]["s"])) < 1e-6:
                dedup[-1] = p
            else:
                dedup.append(p)
        return dedup

    @staticmethod
    def _ensure_cubic_nurbs_control_points(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return np.empty((0, 2), dtype=float)
        m = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
        pts = pts[m]
        if pts.shape[0] < 2:
            return pts
        pts = pts[np.argsort(pts[:, 0])]
        unique = [pts[0]]
        for p in pts[1:]:
            if abs(float(p[0]) - float(unique[-1][0])) < 1e-6:
                unique[-1] = p
            else:
                unique.append(p)
        pts = np.asarray(unique, dtype=float)
        if pts.shape[0] >= 4:
            return pts
        s0, s1 = float(pts[0, 0]), float(pts[-1, 0])
        if s1 <= s0:
            return pts
        target_s = np.linspace(s0, s1, 4)
        add = []
        for s in target_s:
            if pts.shape[0] + len(add) >= 4:
                break
            if np.any(np.isclose(pts[:, 0], s, atol=1e-6)):
                continue
            add.append([float(s), float(np.interp(s, pts[:, 0], pts[:, 1]))])
        if pts.shape[0] + len(add) < 4:
            for s in np.linspace(s0, s1, 8):
                if pts.shape[0] + len(add) >= 4:
                    break
                if np.any(np.isclose(pts[:, 0], s, atol=1e-6)):
                    continue
                if any(abs(float(a[0]) - float(s)) < 1e-6 for a in add):
                    continue
                add.append([float(s), float(np.interp(s, pts[:, 0], pts[:, 1]))])
        if add:
            pts = np.vstack([pts, np.asarray(add, dtype=float)])
        pts = pts[np.argsort(pts[:, 0])]
        return pts

    def _build_smooth_cross_nurbs_control_points(self, key_points: np.ndarray) -> np.ndarray:
        pts = self._ensure_cubic_nurbs_control_points(np.asarray(key_points, dtype=float))
        if pts.ndim != 2 or pts.shape[0] < 2:
            return pts
        pts = pts[np.argsort(pts[:, 0])]
        s0, s1 = float(pts[0, 0]), float(pts[-1, 0])
        if not (np.isfinite(s0) and np.isfinite(s1)) or s1 <= s0:
            return pts
        desired = int(min(40, max(10, 2 + 4 * (pts.shape[0] - 1))))
        target_s = np.linspace(s0, s1, desired)
        guide_s = np.unique(np.concatenate([target_s, pts[:, 0]]))
        guide_z = self._smooth_hermite_interp(pts[:, 0], pts[:, 1], guide_s)
        cps = np.vstack([guide_s, guide_z]).T.astype(float)
        cps[0] = pts[0]
        cps[-1] = pts[-1]
        cps = cps[np.isfinite(cps[:, 0]) & np.isfinite(cps[:, 1])]
        cps = cps[np.argsort(cps[:, 0])]
        unique = [cps[0]]
        for p in cps[1:]:
            if abs(float(p[0]) - float(unique[-1][0])) < 1e-6:
                unique[-1] = p
            else:
                unique.append(p)
        return np.asarray(unique, dtype=float)

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
                "degree": 3,
                "control_points": js.get("control_points", []),
                "weights": js.get("weights", []),
            }
            cps = np.asarray(params.get("control_points", []), dtype=float)
            ws = np.asarray(params.get("weights", []), dtype=float)
            if cps.ndim != 2 or cps.shape[0] < 4:
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
        deg = 3
        n_ctrl = max(2, len(cps))
        if n_ctrl < 4:
            params = self._build_default_nurbs_params(line_id, prof, groups, base)
            cps = params.get("control_points", []) or []
            ww = params.get("weights", []) or []
            n_ctrl = max(4, len(cps))
        params["degree"] = deg
        if len(ww) != n_ctrl:
            ww = [1.0] * n_ctrl
            params["weights"] = ww
        self._set_nurbs_params_for_line(line_id, params)

        self._nurbs_updating_ui = True
        try:
            self.nurbs_cp_spin.setValue(n_ctrl)
            self.nurbs_deg_spin.setRange(3, 3)
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
        return {
            "degree": 3,
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

    def _apply_nurbs_params_after_edit(self, params: Dict[str, Any], current_row: Optional[int] = None) -> None:
        line_id = self._line_id_current()
        cps = np.asarray(params.get("control_points", []), dtype=float)
        if cps.ndim != 2 or cps.shape[0] < 4:
            self._warn("[UI3] Cubic NURBS requires at least 4 control points.")
            return
        weights = np.asarray(params.get("weights", []), dtype=float)
        if weights.ndim != 1 or weights.size != cps.shape[0]:
            weights = np.ones(cps.shape[0], dtype=float)
        params = {
            "degree": 3,
            "control_points": cps.astype(float).tolist(),
            "weights": np.where(np.isfinite(weights) & (weights > 0), weights, 1.0).astype(float).tolist(),
        }
        self._set_curve_method_for_line(line_id, "nurbs")
        self._set_nurbs_params_for_line(line_id, params)
        self._nurbs_updating_ui = True
        try:
            self.nurbs_cp_spin.setValue(int(cps.shape[0]))
            self.nurbs_deg_spin.setRange(3, 3)
            self.nurbs_deg_spin.setValue(3)
            self._populate_nurbs_table(params)
            if current_row is not None and self.nurbs_table.rowCount() > 0:
                row = max(0, min(int(current_row), self.nurbs_table.rowCount() - 1))
                self.nurbs_table.setCurrentCell(row, 1)
        finally:
            self._nurbs_updating_ui = False
        self._schedule_nurbs_live_update()

    def _on_add_nurbs_control_point(self) -> None:
        if not self._active_prof:
            self._warn("[UI3] Draw Curve and Convert first before adding NURBS control points.")
            return
        params = self._collect_nurbs_params_from_ui() or self._get_nurbs_params_for_line(self._line_id_current())
        if not params:
            self._warn("[UI3] Convert to NURBS first before adding control points.")
            return
        cps = np.asarray(params.get("control_points", []), dtype=float)
        weights = np.asarray(params.get("weights", []), dtype=float)
        if cps.ndim != 2 or cps.shape[0] < 4:
            self._warn("[UI3] Convert to NURBS first before adding control points.")
            return
        if weights.ndim != 1 or weights.size != cps.shape[0]:
            weights = np.ones(cps.shape[0], dtype=float)

        row = int(self.nurbs_table.currentRow())
        n = int(cps.shape[0])
        if row < 0:
            insert_at = n - 1
        elif row <= 0:
            insert_at = 1
        elif row >= n - 1:
            insert_at = n - 1
        else:
            insert_at = row + 1

        new_cp = (cps[insert_at - 1] + cps[insert_at]) / 2.0
        new_w = float((weights[insert_at - 1] + weights[insert_at]) / 2.0)
        new_cps = np.insert(cps, insert_at, new_cp, axis=0)
        new_weights = np.insert(weights, insert_at, max(0.001, new_w), axis=0)
        self._apply_nurbs_params_after_edit(
            {"degree": 3, "control_points": new_cps.tolist(), "weights": new_weights.tolist()},
            current_row=insert_at,
        )

    def _on_delete_nurbs_control_point(self) -> None:
        if not self._active_prof:
            self._warn("[UI3] Draw Curve and Convert first before deleting NURBS control points.")
            return
        params = self._collect_nurbs_params_from_ui() or self._get_nurbs_params_for_line(self._line_id_current())
        if not params:
            self._warn("[UI3] Convert to NURBS first before deleting control points.")
            return
        cps = np.asarray(params.get("control_points", []), dtype=float)
        weights = np.asarray(params.get("weights", []), dtype=float)
        if cps.ndim != 2 or cps.shape[0] <= 4:
            self._warn("[UI3] Cannot delete: cubic NURBS must keep at least 4 control points.")
            return
        if weights.ndim != 1 or weights.size != cps.shape[0]:
            weights = np.ones(cps.shape[0], dtype=float)

        row = int(self.nurbs_table.currentRow())
        if row < 0:
            self._warn("[UI3] Select a control point row to delete.")
            return
        if row == 0 or row == cps.shape[0] - 1:
            self._warn("[UI3] Cannot delete locked NURBS endpoint control points.")
            return

        new_cps = np.delete(cps, row, axis=0)
        new_weights = np.delete(weights, row, axis=0)
        self._apply_nurbs_params_after_edit(
            {"degree": 3, "control_points": new_cps.tolist(), "weights": new_weights.tolist()},
            current_row=min(row, int(new_cps.shape[0]) - 2),
        )

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

        deg = 3
        if cps.shape[0] < 4:
            return None
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
        if int(val) != 3:
            self._nurbs_updating_ui = True
            try:
                self.nurbs_deg_spin.setValue(3)
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
        if not self._active_prof or not self._active_curve:
            self._warn("[UI3] Draw Curve first to generate a NURBS preview.")
            return
        line_id = self._line_id_current()
        params = self._collect_nurbs_params_from_ui() or self._get_nurbs_params_for_line(line_id)
        if not params:
            self._warn("[UI3] Draw Curve first to generate NURBS control points.")
            return
        params["degree"] = 3
        cps = np.asarray(params.get("control_points", []), dtype=float)
        weights = np.asarray(params.get("weights", []), dtype=float)
        if cps.ndim != 2 or cps.shape[0] < 4:
            self._warn("[UI3] NURBS degree 3 requires at least 4 control points.")
            return
        if weights.ndim != 1 or weights.size != cps.shape[0]:
            weights = np.ones(cps.shape[0], dtype=float)
            params["weights"] = weights.tolist()
        curve = {
            "chain": np.asarray((self._active_curve or {}).get("chain", []), dtype=float),
            "elev": np.asarray((self._active_curve or {}).get("elev", []), dtype=float),
        }
        m_curve = np.isfinite(curve["chain"]) & np.isfinite(curve["elev"])
        curve["chain"] = curve["chain"][m_curve]
        curve["elev"] = curve["elev"][m_curve]
        if curve["chain"].size < 2:
            self._warn("[UI3] NURBS preview has too few valid points.")
            return
        boring_result = self._project_boring_holes_for_current_line(use_unsaved_table=True, log_skips=True)
        boring_snapshot = {
            "distance_tolerance_m": float(boring_result.get("distance_tolerance_m", 1.0)),
            "count": int(len(boring_result.get("items", []) or [])),
            "items": list(boring_result.get("items", []) or []),
        }
        is_cross_line = self._current_ui2_line_role() == "cross"
        render_groups = self._active_groups if self._active_groups else None
        if is_cross_line:
            render_groups = self._cross_line_slip_span_render_group(self._active_prof)
        out_png = self._nurbs_png_path_for(line_id)
        path = self._render_profile_png_current_settings(
            self._active_prof,
            out_png,
            groups=render_groups if render_groups else None,
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
        theta_csv_path = self._theta_csv_path_for(line_id)
        result = dict(self._active_global_fit_result or {})
        preview_curve = {
            "chain": curve_chain.astype(float).tolist(),
            "elev": curve_elev.astype(float).tolist(),
        }
        cp_rows = []
        for i, cp in enumerate(cps):
            cp_rows.append({
                "cp_index": int(i),
                "chainage_m": float(cp[0]),
                "elev_m": float(cp[1]),
                "weight": float(weights[i]) if i < weights.size and np.isfinite(weights[i]) else 1.0,
            })

        outputs = {
            "preview_params": {
                "path": self._nurbs_json_path_for(line_id),
                "payload": {
                    "line_id": line_id,
                    "curve_method": "nurbs",
                    "mode": "cubic_nurbs",
                    "degree": 3,
                    "control_points": cps.astype(float).tolist(),
                    "weights": weights.astype(float).tolist(),
                    "theta_csv_path": theta_csv_path,
                    "theta_rows": list(result.get("theta_rows", []) or []),
                    "fit_points": list(result.get("fit_points", []) or []),
                    "global_fit_points": list(result.get("global_fit_points", result.get("fit_points", [])) or []),
                    "boundary_intersections": list(result.get("boundary_intersections", []) or []),
                    "applied_boring_holes": boring_snapshot,
                    "curve": preview_curve,
                },
            },
            "curve_json": {
                "path": os.path.join(self._curve_dir(), f"nurbs_{line_id}.json"),
                "payload": {
                    "line_id": line_id,
                    "curve_method": "nurbs",
                    "mode": "cubic_nurbs",
                    "degree": 3,
                    "control_points": cp_rows,
                    "chainage_origin": self._ui3_chainage_origin(),
                    "applied_boring_holes": boring_snapshot,
                    "count": int(len(curve_rows)),
                    "points": curve_rows,
                },
            },
            "nurbs_info": {
                "path": os.path.join(self._curve_dir(), f"nurbs_info_{line_id}.json"),
                "payload": {
                    "line_id": line_id,
                    "curve_method": "nurbs",
                    "mode": "cubic_nurbs",
                    "chainage_origin": self._ui3_chainage_origin(),
                    "degree": 3,
                    "control_points": cp_rows,
                    "theta_csv_path": theta_csv_path,
                    "theta_rows": list(result.get("theta_rows", []) or []),
                    "fit_points": list(result.get("fit_points", []) or []),
                    "global_fit_points": list(result.get("global_fit_points", result.get("fit_points", [])) or []),
                    "boundary_intersections": list(result.get("boundary_intersections", []) or []),
                    "fit_point_count": int(len(result.get("fit_points", []) or [])),
                    "curve": preview_curve,
                },
            },
        }
        if not is_cross_line:
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
                curve_method="nurbs",
                nurbs_seed_method=self._get_nurbs_seed_method_for_line(line_id),
                profile_dem_source=self._current_profile_source_key(),
                profile_dem_path=str(getattr(self, "dem_path", "") or ""),
                grouping_params=self._grouping_params_current(),
                group_method=group_method,
            )
            outputs["group_json"] = {"path": groups_ui3_json, "payload": groups_ui3_payload}
        saved = self._backend.save_nurbs_outputs(outputs)
        self._ok(f"[UI3] Saved NURBS curve: {path}")
        self._log(f"[UI3] Saved preview metadata: {saved.get('preview_params', self._nurbs_json_path_for(line_id))}")
        self._log(f"[UI3] Saved curve JSON: {saved.get('curve_json', '')}")
        if not is_cross_line:
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
                groups = self._groups_clamped_to_profile_bounds(prof, groups)
                prof = self._profile_with_group_slip_span(prof, groups)
                if not groups:
                    self._warn("[UI3] No groups within profile bounds.")
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
            reference_curve = {
                "chain": np.asarray((result.get("curve", {}) or {}).get("chain", []), dtype=float),
                "elev": np.asarray((result.get("curve", {}) or {}).get("elev", []), dtype=float),
            }
            mask = np.isfinite(reference_curve["chain"]) & np.isfinite(reference_curve["elev"])
            reference_curve["chain"] = reference_curve["chain"][mask]
            reference_curve["elev"] = reference_curve["elev"][mask]
            if reference_curve["chain"].size < 2:
                self._warn("[UI3] Reference spline has too few valid points.")
                return

            self._active_prof = prof
            self._active_groups = groups
            self._active_base_curve = {"chain": reference_curve["chain"], "elev": reference_curve["elev"]}
            self._nurbs_params_by_line.pop(line_id, None)
            self._nurbs_updating_ui = True
            try:
                self.nurbs_deg_spin.setRange(3, 3)
                self.nurbs_deg_spin.setValue(3)
                self.nurbs_cp_spin.setValue(4)
                self.nurbs_table.setRowCount(0)
            finally:
                self._nurbs_updating_ui = False

            out_png = self._profile_png_path_for(line_id)
            path = self._render_profile_png_current_settings(prof, out_png, groups=groups)
            if not path or not os.path.exists(path):
                return
            if not self._load_preview_scene_from_path(path, static_nurbs_bg=False):
                self._err("[UI3] Cannot load PNG with curve overlay.")
                return

            result["curve"] = {"chain": reference_curve["chain"], "elev": reference_curve["elev"]}
            result["theta_csv_path"] = self._theta_csv_path_for(line_id)
            self._active_curve = {"chain": reference_curve["chain"], "elev": reference_curve["elev"]}
            self._active_global_fit_result = result
            self._draw_curve_overlay(reference_curve["chain"], reference_curve["elev"])
            self._clear_control_points_overlay()
            try:
                spline_csv = self._save_spline_curve_csv_for_line(line_id, self._active_curve)
                if spline_csv:
                    self._log(f"[UI3] Saved spline curve CSV: {spline_csv}")
                    group_method = None
                    groups_ui3_json = self._groups_json_path_for(line_id)
                    if os.path.exists(groups_ui3_json):
                        try:
                            with open(groups_ui3_json, "r", encoding="utf-8") as f:
                                old_js = json.load(f) or {}
                            if old_js.get("group_method", None):
                                group_method = str(old_js.get("group_method"))
                        except Exception:
                            pass
                    self._save_groups_to_ui(
                        groups,
                        prof,
                        line_id,
                        curve_method=curve_method,
                        group_method=group_method,
                    )
            except Exception as e:
                self._warn(f"[UI3] Cannot save spline curve CSV: {e}")
            try:
                anchor_path, n_upd = self._update_anchors_xyz_for_saved_main_curve(self._active_curve)
                if anchor_path and n_upd > 0:
                    self._log(f"[UI3] Updated anchors.json from spline curve: {anchor_path} (n={n_upd})")
            except Exception as e:
                self._warn(f"[UI3] Cannot update anchors.json from spline curve: {e}")
            self._refresh_anchor_overlay()
            if hasattr(self, "btn_convert_nurbs"):
                self.btn_convert_nurbs.setEnabled(True)
            self._ok("[UI3] Global fit spline drawn. Click Convert to NURBS to generate control points.")
        except Exception as e:
            self._err(f"[UI3] Draw Curve error: {e}")

    def _on_draw_cross_nurbs(self) -> None:
        try:
            if self._current_ui2_line_role() != "cross":
                return
            if not hasattr(self, "_gdf") or self._gdf is None or self._gdf.empty:
                self._warn("[UI3] No lines.")
                return
            row = self.line_combo.currentIndex()
            if row < 0:
                self._warn("[UI3] Select a Cross Line first.")
                return

            line_id = self._line_id_current()
            prof = self._active_prof
            if not prof or len(prof.get("chain", [])) < 2:
                geom = self._gdf.geometry.iloc[row]
                prof = self._compute_profile_for_geom(geom, slip_only=False)
            if not prof or len(prof.get("chain", [])) < 2:
                self._warn("[UI3] Cannot compute Cross Line profile.")
                return

            span = self._cross_line_slip_span_bounds(prof)
            if span is None:
                self._warn("[UI3] Cannot detect slip span for this Cross Line.")
                return
            s0, s1 = span
            z0 = self._interp_profile_elev(prof, s0)
            z1 = self._interp_profile_elev(prof, s1)
            if z0 is None or z1 is None:
                self._warn("[UI3] Cannot interpolate slip-span endpoint elevations.")
                return

            anchors = self._cross_anchor_points_in_span(self._current_ui2_line_id(), s0, s1)
            if not anchors:
                self._warn("[UI3] No ready anchors inside slip span for this Cross Line.")
                return

            raw_points = [[float(s0), float(z0)]]
            raw_points.extend([[float(a["s"]), float(a["z"])] for a in anchors])
            raw_points.append([float(s1), float(z1)])
            cps = self._build_smooth_cross_nurbs_control_points(np.asarray(raw_points, dtype=float))
            if cps.ndim != 2 or cps.shape[0] < 4:
                self._warn("[UI3] Cubic NURBS requires at least 4 valid control points.")
                return

            params = {
                "degree": 3,
                "control_points": cps.astype(float).tolist(),
                "weights": np.ones(cps.shape[0], dtype=float).tolist(),
            }
            reference_curve = {"chain": cps[:, 0], "elev": cps[:, 1]}
            self._active_prof = prof
            self._active_groups = []
            self._active_base_curve = reference_curve
            self._set_curve_method_for_line(line_id, "nurbs")
            self._set_nurbs_params_for_line(line_id, params)
            self._nurbs_updating_ui = True
            try:
                self.nurbs_cp_spin.setValue(int(cps.shape[0]))
                self.nurbs_deg_spin.setRange(3, 3)
                self.nurbs_deg_spin.setValue(3)
                self._populate_nurbs_table(params)
            finally:
                self._nurbs_updating_ui = False

            curve = self._compute_nurbs_curve_from_params(params)
            if curve is None or np.asarray(curve.get("chain", []), dtype=float).size < 2:
                self._warn("[UI3] Cubic NURBS has too few valid points.")
                return

            result = dict(self._active_global_fit_result or {})
            result["reference_curve"] = reference_curve
            result["curve"] = {"chain": curve["chain"], "elev": curve["elev"]}
            result["fit_points"] = anchors
            result["global_fit_points"] = anchors
            result["theta_csv_path"] = self._theta_csv_path_for(line_id)
            self._active_curve = {"chain": curve["chain"], "elev": curve["elev"]}
            self._active_global_fit_result = result

            render_groups = self._cross_line_slip_span_render_group(prof)
            out_png = self._profile_png_path_for(line_id)
            path = self._render_profile_png_current_settings(
                prof,
                out_png,
                groups=render_groups if render_groups else None,
            )
            if path and os.path.exists(path):
                self._load_preview_scene_from_path(path, static_nurbs_bg=False)
            self._draw_curve_overlay(np.asarray(curve["chain"], dtype=float), np.asarray(curve["elev"], dtype=float))
            self._draw_control_points_overlay(params)
            if hasattr(self, "btn_convert_nurbs"):
                self.btn_convert_nurbs.setEnabled(True)
                self.btn_convert_nurbs.setText("Draw")
            self._ok("[UI3] Drew Cross Line NURBS through ready anchors.")
            self._on_nurbs_save()
        except Exception as e:
            self._err(f"[UI3] Draw Cross Line NURBS error: {e}")

    def _on_convert_to_nurbs(self) -> None:
        try:
            if self._current_ui2_line_role() == "cross":
                self._on_draw_cross_nurbs()
                return
            if not self._active_prof:
                self._warn("[UI3] Draw Curve first before converting to NURBS.")
                return
            line_id = self._line_id_current()
            groups = self._active_groups or self._read_groups_from_table() or []
            reference_curve = self._active_base_curve or (self._active_global_fit_result or {}).get("curve", {})
            reference_curve = {
                "chain": np.asarray((reference_curve or {}).get("chain", []), dtype=float),
                "elev": np.asarray((reference_curve or {}).get("elev", []), dtype=float),
            }
            mask = np.isfinite(reference_curve["chain"]) & np.isfinite(reference_curve["elev"])
            reference_curve["chain"] = reference_curve["chain"][mask]
            reference_curve["elev"] = reference_curve["elev"][mask]
            if reference_curve["chain"].size < 2:
                self._warn("[UI3] Draw Curve first before converting to NURBS.")
                return

            params = self._backend.fit_nurbs_params_from_curve(
                reference_curve,
                groups,
                degree=3,
                min_control_points=4,
            )
            cps = np.asarray(params.get("control_points", []), dtype=float)
            if cps.ndim != 2 or cps.shape[0] < 4:
                self._warn("[UI3] Cannot fit cubic NURBS control points from the current spline.")
                return

            self._active_groups = groups
            self._active_base_curve = {"chain": reference_curve["chain"], "elev": reference_curve["elev"]}
            self._set_curve_method_for_line(line_id, "nurbs")
            self._set_nurbs_params_for_line(line_id, params)
            self._nurbs_updating_ui = True
            try:
                self.nurbs_cp_spin.setValue(int(cps.shape[0]))
                self.nurbs_deg_spin.setRange(3, 3)
                self.nurbs_deg_spin.setValue(3)
                self._populate_nurbs_table(params)
            finally:
                self._nurbs_updating_ui = False

            curve = self._compute_nurbs_curve_from_params(params)
            if curve is None or np.asarray(curve.get("chain", []), dtype=float).size < 2:
                self._warn("[UI3] Cubic NURBS has too few valid points.")
                return

            result = dict(self._active_global_fit_result or {})
            result["reference_curve"] = {"chain": reference_curve["chain"], "elev": reference_curve["elev"]}
            result["curve"] = {"chain": curve["chain"], "elev": curve["elev"]}
            result["theta_csv_path"] = self._theta_csv_path_for(line_id)
            self._active_curve = {"chain": curve["chain"], "elev": curve["elev"]}
            self._active_global_fit_result = result
            self._draw_curve_overlay(np.asarray(curve["chain"], dtype=float), np.asarray(curve["elev"], dtype=float))
            self._draw_control_points_overlay(params)
            if hasattr(self, "btn_convert_nurbs"):
                self.btn_convert_nurbs.setEnabled(True)
            self._ok("[UI3] Converted spline to cubic NURBS.")
            self._on_nurbs_save()
        except Exception as e:
            self._err(f"[UI3] Convert to NURBS error: {e}")

    def _sync_nurbs_defaults_from_group_table(self) -> None:
        if self._group_table_updating:
            return
        try:
            groups = self._read_groups_from_table()
            if groups and self._active_prof:
                groups = self._groups_clamped_to_profile_bounds(self._active_prof, groups)
                self._active_prof = self._profile_with_group_slip_span(self._active_prof, groups)
            self._active_groups = groups or []
            self._active_base_curve = None
            self._active_curve = None
            self._active_global_fit_result = None
            self._clear_curve_overlay()
            self._clear_control_points_overlay()
            if hasattr(self, "btn_convert_nurbs"):
                self.btn_convert_nurbs.setEnabled(False)
            if hasattr(self, "nurbs_table"):
                self.nurbs_table.setRowCount(0)
        except Exception as e:
            self._warn(f"[UI3] Cannot sync NURBS state from groups: {e}")
