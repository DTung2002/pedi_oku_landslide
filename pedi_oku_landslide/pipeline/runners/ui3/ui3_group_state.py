import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .ui3_grouping import renumber_groups_visual_order


def normalize_curve_method(method: Optional[str]) -> str:
    m = str(method or "").strip().lower()
    if m == "nurbs":
        return "nurbs"
    return "bezier"


def curve_method_from_group_method(group_method: Optional[str]) -> str:
    _ = group_method
    return "nurbs"


def groups_to_current_chainage(
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
        out.append(
            {
                "id": gid,
                "start": float(s),
                "end": float(e),
                "start_reason": start_reason,
                "end_reason": end_reason,
                "color": str(g.get("color", "") or "").strip(),
            }
        )
    return renumber_groups_visual_order(out)


def group_signature(groups: List[dict]) -> List[Tuple[str, float, float, str, str]]:
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
        sig.append(
            (
                gid,
                round(float(s), 3),
                round(float(e), 3),
                str(g.get("start_reason", "") or "").strip(),
                str(g.get("end_reason", "") or "").strip(),
            )
        )
    return sig


def group_for_chainage(groups: List[dict], chainage: float) -> Tuple[Optional[str], Optional[str]]:
    for g in (groups or []):
        try:
            s = float(g.get("start", g.get("start_chainage", 0.0)))
            e = float(g.get("end", g.get("end_chainage", 0.0)))
            if e < s:
                s, e = e, s
            if s <= chainage <= e:
                gid = str(g.get("id", "")) or None
                color = g.get("color", None)
                return gid, color
        except Exception:
            continue
    return None, None


def groups_with_median_theta(groups: List[dict], prof: Optional[dict]) -> List[dict]:
    out: List[dict] = []
    if not groups:
        return out

    chain = np.asarray((prof or {}).get("chain", []), dtype=float)
    theta = np.asarray((prof or {}).get("theta", []), dtype=float)
    n = int(min(chain.size, theta.size))
    if n > 0:
        chain = chain[:n]
        theta = theta[:n]

    for g in groups:
        gg = dict(g or {})
        med_theta = None
        try:
            if n > 0:
                s = float(gg.get("start", gg.get("start_chainage", 0.0)))
                e = float(gg.get("end", gg.get("end_chainage", 0.0)))
                if e < s:
                    s, e = e, s
                mask = (chain >= s) & (chain <= e)
                if np.any(mask):
                    vals = theta[mask]
                    vals = vals[np.isfinite(vals)]
                    if vals.size > 0:
                        med_theta = float(np.median(vals))
        except Exception:
            med_theta = None
        gg["median_theta_deg"] = med_theta
        out.append(gg)
    return out


def curvature_points_for_json(
    prof: Optional[dict],
    groups: List[dict],
    *,
    grouping_params: Dict[str, Any],
    extract_curvature_nodes: Callable[..., Dict[str, Any]],
    profile_slip_span_range: Callable[..., Optional[Tuple[float, float]]],
) -> List[dict]:
    out: List[dict] = []
    if not prof:
        return out
    try:
        nodes = extract_curvature_nodes(
            prof,
            rdp_eps_m=float(grouping_params.get("rdp_eps_m", 0.5)),
            smooth_radius_m=float(grouping_params.get("smooth_radius_m", 0.0)),
            restrict_to_slip_span=False,
        )
        chain = np.asarray(nodes.get("chain", []), dtype=float)
        elev = np.asarray(nodes.get("elev", []), dtype=float)
        curv = np.asarray(nodes.get("curvature", []), dtype=float)
        slip_span = profile_slip_span_range(
            prof,
            rdp_eps_m=float(grouping_params.get("rdp_eps_m", 0.5)),
            smooth_radius_m=float(grouping_params.get("smooth_radius_m", 0.0)),
        )
        if slip_span:
            smin, smax = slip_span
            keep = np.isfinite(chain) & np.isfinite(elev) & np.isfinite(curv) & (chain >= float(smin)) & (chain <= float(smax))
            chain = chain[keep]
            elev = elev[keep]
            curv = curv[keep]
        n = int(min(chain.size, elev.size, curv.size))
        curvature_thr_abs = float(grouping_params.get("curvature_thr_abs", 0.02))
        include_curvature = bool(grouping_params.get("include_curvature_threshold", True))
        for i in range(n):
            ch = float(chain[i])
            zz = float(elev[i])
            kk = float(curv[i])
            if not (np.isfinite(ch) and np.isfinite(zz) and np.isfinite(kk)):
                continue
            gid, color = group_for_chainage(groups or [], ch)
            out.append(
                {
                    "index": i,
                    "chain_m": ch,
                    "elev_m": zz,
                    "curvature": kk,
                    "curvature_abs": abs(kk),
                    "group_id": gid,
                    "group_color": color,
                    "passes_curvature_threshold": bool(include_curvature and (abs(kk) > curvature_thr_abs)),
                }
            )
    except Exception:
        return []
    return out


def load_group_json_data(
    path: str,
    *,
    line_id: str,
    apply_settings: bool,
    apply_group_json_settings: Callable[[Dict[str, Any]], None],
) -> Tuple[Dict[str, Any], List[dict], str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f) or {}
    if apply_settings:
        apply_group_json_settings(data)
    curve_method = normalize_curve_method(data.get("curve_method"))
    groups = groups_to_current_chainage(
        data.get("groups", []) or [],
        source_origin=data.get("chainage_origin"),
    )
    _ = line_id
    return data, groups, curve_method


def build_group_json_payload(
    *,
    line_label: str,
    groups: List[dict],
    prof: Optional[dict],
    chainage_origin: str,
    curve_method: str,
    profile_dem_source: str,
    profile_dem_path: str,
    grouping_params: Dict[str, Any],
    group_method: Optional[str],
    extract_curvature_nodes: Callable[..., Dict[str, Any]],
    profile_slip_span_range: Callable[..., Optional[Tuple[float, float]]],
) -> Dict[str, Any]:
    groups_for_json = groups_with_median_theta(groups, prof)
    curvature_points = curvature_points_for_json(
        prof,
        groups_for_json,
        grouping_params=grouping_params,
        extract_curvature_nodes=extract_curvature_nodes,
        profile_slip_span_range=profile_slip_span_range,
    )
    payload = {
        "line": line_label,
        "groups": groups_for_json,
        "curvature_points": curvature_points,
        "chainage_origin": str(chainage_origin or "").strip(),
        "curve_method": normalize_curve_method(curve_method),
        "profile_dem_source": str(profile_dem_source or "").strip(),
        "profile_dem_path": str(profile_dem_path or "").replace("\\", "/"),
        "rdp_eps_m": float(grouping_params.get("rdp_eps_m", 0.5)),
        "smooth_radius_m": float(grouping_params.get("smooth_radius_m", 0.0)),
        "include_curvature_threshold": bool(grouping_params.get("include_curvature_threshold", True)),
        "include_vector_angle_zero": bool(grouping_params.get("include_vector_angle_zero", True)),
    }
    if group_method:
        payload["group_method"] = str(group_method)
    return payload


def save_group_json(path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def saved_curvature_series_for_line(
    path: str,
    *,
    current_groups: Optional[List[dict]],
    prof: Optional[dict],
    repair_saved_curvature_points: Callable[[List[dict], Optional[dict]], List[dict]],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        return None

    saved_groups = groups_to_current_chainage(
        data.get("groups", []) or [],
        source_origin=data.get("chainage_origin"),
        length_m=(prof.get("length_m") if prof else None),
    )
    if current_groups is not None:
        if group_signature(saved_groups) != group_signature(current_groups):
            return None

    pts = repair_saved_curvature_points(data.get("curvature_points", []) or [], prof)
    xs: List[float] = []
    ks: List[float] = []
    for it in pts:
        try:
            x = float(it.get("chain_m"))
            k = float(it.get("curvature"))
        except Exception:
            continue
        if not (np.isfinite(x) and np.isfinite(k)):
            continue
        xs.append(float(x))
        ks.append(float(k))
    if len(xs) < 2:
        return None
    order = np.argsort(np.asarray(xs, dtype=float))
    return np.asarray(xs, dtype=float)[order], np.asarray(ks, dtype=float)[order]
