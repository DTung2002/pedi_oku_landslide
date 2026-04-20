from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_NURBS_SEED_METHOD = "bezier_like"
SEGMENT_HANDLE_RATIO = 1.0 / 3.0


def normalize_nurbs_seed_method(method: Optional[str]) -> str:
    m = str(method or "").strip().lower()
    if m == "slope_guided":
        return "slope_guided"
    return DEFAULT_NURBS_SEED_METHOD


def profile_endpoints(
    prof: dict,
    *,
    profile_slip_span_range,
    rdp_eps_m: float,
    smooth_radius_m: float,
) -> Optional[Tuple[float, float, float, float]]:
    chain = np.asarray(prof.get("chain", []), dtype=float)
    elev = np.asarray(prof.get("elev_s", []), dtype=float)
    m = np.isfinite(chain) & np.isfinite(elev)
    chain = chain[m]
    elev = elev[m]
    if chain.size < 2:
        return None
    order = np.argsort(chain)
    chain = chain[order]
    elev = elev[order]
    eff_span = profile_slip_span_range(
        prof,
        rdp_eps_m=float(rdp_eps_m),
        smooth_radius_m=float(smooth_radius_m),
    )
    if eff_span is not None:
        smin, smax = eff_span
        keep = (chain >= float(smin)) & (chain <= float(smax))
        if int(np.count_nonzero(keep)) >= 2:
            chain = chain[keep]
            elev = elev[keep]
    return float(chain[0]), float(elev[0]), float(chain[-1]), float(elev[-1])


def grouped_vector_endpoints(prof: dict, groups: list) -> Optional[Tuple[float, float, float, float]]:
    chain = np.asarray(prof.get("chain", []), dtype=float)
    elev = np.asarray(prof.get("elev_s", []), dtype=float)
    finite = np.isfinite(chain) & np.isfinite(elev)
    if int(np.count_nonzero(finite)) < 2:
        return None
    chain = chain[finite]
    elev = elev[finite]
    if chain.size < 2:
        return None
    order = np.argsort(chain)
    chain = chain[order]
    elev = elev[order]

    grouped_mask = np.zeros(chain.shape, dtype=bool)
    for g in (groups or []):
        try:
            s = float(g.get("start", g.get("start_chainage", np.nan)))
            e = float(g.get("end", g.get("end_chainage", np.nan)))
        except Exception:
            continue
        if not (np.isfinite(s) and np.isfinite(e)):
            continue
        if e < s:
            s, e = e, s
        grouped_mask |= ((chain >= s) & (chain <= e))

    idx = np.flatnonzero(grouped_mask)
    if idx.size < 2:
        return None
    i0 = int(idx[0])
    i1 = int(idx[-1])
    return float(chain[i0]), float(elev[i0]), float(chain[i1]), float(elev[i1])


def normalize_group_spans(groups: list) -> List[Tuple[float, float]]:
    spans: List[Tuple[float, float]] = []
    for g in (groups or []):
        try:
            s = float(g.get("start", g.get("start_chainage", np.nan)))
            e = float(g.get("end", g.get("end_chainage", np.nan)))
        except Exception:
            continue
        if not (np.isfinite(s) and np.isfinite(e)):
            continue
        if e < s:
            s, e = e, s
        spans.append((float(s), float(e)))
    spans.sort(key=lambda x: (x[0], x[1]))
    return spans


def build_default_nurbs_chainage(s0: float, s1: float, groups: list) -> List[float]:
    spans = normalize_group_spans(groups)
    n_ctrl = max(2, len(spans) + 1)
    if not spans:
        return np.linspace(float(s0), float(s1), n_ctrl).tolist()

    inner: List[float] = []
    for i in range(max(0, len(spans) - 1)):
        b = float(spans[i][1])
        b = max(float(s0), min(float(s1), b))
        inner.append(b)

    cp_chain = [float(s0)] + inner + [float(s1)]
    if len(cp_chain) != n_ctrl:
        cp_chain = np.linspace(float(s0), float(s1), n_ctrl).tolist()
    return cp_chain


def _build_nurbs_params_from_control_points(cp_chain: List[float], cp_elev: List[float]) -> Dict[str, Any]:
    cps = [[float(s), float(z)] for s, z in zip(cp_chain, cp_elev)]
    w = [1.0] * len(cps)
    deg = min(3, max(1, len(cps) - 1))
    return {"degree": int(deg), "control_points": cps, "weights": w}


def _normalize_groups_with_meta(groups: list) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, g in enumerate(groups or []):
        try:
            s = float(g.get("start", g.get("start_chainage", np.nan)))
            e = float(g.get("end", g.get("end_chainage", np.nan)))
        except Exception:
            continue
        if not (np.isfinite(s) and np.isfinite(e)):
            continue
        if e < s:
            s, e = e, s
        rows.append({
            "index": int(idx),
            "id": str(g.get("id", g.get("group_id", "")) or ""),
            "start": float(s),
            "end": float(e),
        })
    rows.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return rows


def _ground_profile_arrays(prof: dict) -> Tuple[np.ndarray, np.ndarray]:
    gch = np.asarray(prof.get("chain", []), dtype=float)
    gz = np.asarray(prof.get("elev_s", []), dtype=float)
    mg = np.isfinite(gch) & np.isfinite(gz)
    gch = gch[mg]
    gz = gz[mg]
    if gch.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(gch)
    gch = gch[order]
    gz = gz[order]
    gch_u, uniq_idx = np.unique(gch, return_index=True)
    return gch_u, gz[uniq_idx]


def _clamp_interior_control_points_below_ground(
    cp_chain: List[float],
    cp_elev: List[float],
    prof: dict,
    *,
    clearance: float = 0.35,
) -> List[float]:
    if len(cp_chain) != len(cp_elev):
        return list(cp_elev)
    cp_elev_arr = np.asarray(cp_elev, dtype=float)
    if cp_elev_arr.size < 3:
        return cp_elev_arr.tolist()
    gch, gz = _ground_profile_arrays(prof)
    if gch.size < 2:
        return cp_elev_arr.tolist()
    g_at_cp = np.interp(np.asarray(cp_chain, dtype=float), gch, gz)
    cp_elev_arr[1:-1] = np.minimum(cp_elev_arr[1:-1], g_at_cp[1:-1] - float(clearance))
    return cp_elev_arr.tolist()


def _profile_surface_for_seed(prof: dict) -> Tuple[np.ndarray, np.ndarray]:
    chain = np.asarray(prof.get("chain", []), dtype=float)
    if chain.ndim != 1 or chain.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    elev = None
    for key in ("elev_orig", "elev", "elev_s"):
        arr = np.asarray(prof.get(key, []), dtype=float)
        if arr.ndim == 1 and arr.size == chain.size:
            elev = arr
            break
    if elev is None:
        return np.array([], dtype=float), np.array([], dtype=float)

    keep = np.isfinite(chain) & np.isfinite(elev)
    chain = chain[keep]
    elev = elev[keep]
    if chain.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(chain)
    chain = chain[order]
    elev = elev[order]
    chain_u, uniq_idx = np.unique(chain, return_index=True)
    elev_u = elev[uniq_idx]
    return chain_u, elev_u


def _local_surface_window_for_seed(
    prof: dict,
    start_chainage: float,
    *,
    neighbors_each_side: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    chain, elev = _profile_surface_for_seed(prof)
    if chain.size < 2 or not np.isfinite(float(start_chainage)):
        return np.array([], dtype=float), np.array([], dtype=float)

    before = np.flatnonzero(chain < float(start_chainage))
    after = np.flatnonzero(chain > float(start_chainage))

    ch_parts: List[np.ndarray] = []
    z_parts: List[np.ndarray] = []
    if before.size > 0:
        idx = before[-int(max(0, neighbors_each_side)):]
        ch_parts.append(chain[idx])
        z_parts.append(elev[idx])

    z0_surface = float(np.interp(float(start_chainage), chain, elev))
    ch_parts.append(np.asarray([float(start_chainage)], dtype=float))
    z_parts.append(np.asarray([z0_surface], dtype=float))

    if after.size > 0:
        idx = after[: int(max(0, neighbors_each_side))]
        ch_parts.append(chain[idx])
        z_parts.append(elev[idx])

    local_chain = np.concatenate(ch_parts) if ch_parts else np.array([], dtype=float)
    local_elev = np.concatenate(z_parts) if z_parts else np.array([], dtype=float)
    if local_chain.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(local_chain)
    local_chain = local_chain[order]
    local_elev = local_elev[order]
    return local_chain, local_elev


def _point_slopes_for_profile_window(
    full_chain: np.ndarray,
    full_elev: np.ndarray,
    local_chain: np.ndarray,
    local_elev: np.ndarray,
) -> np.ndarray:
    n = int(min(local_chain.size, local_elev.size))
    if n < 2:
        return np.array([], dtype=float)
    if int(min(full_chain.size, full_elev.size)) < 2:
        return np.array([], dtype=float)

    slopes = np.full(n, np.nan, dtype=float)
    for idx in range(n):
        ch = float(local_chain[idx])
        zz = float(local_elev[idx])
        next_pos = int(np.searchsorted(full_chain, ch, side="right"))
        if next_pos < full_chain.size:
            # Match the client sheet: slope at a point uses the next point
            # on the full ground profile, not just within the 11-point window.
            dx = float(full_chain[next_pos] - ch)
            dz = float(full_elev[next_pos] - zz)
        else:
            prev_pos = int(np.searchsorted(full_chain, ch, side="left")) - 1
            if prev_pos < 0:
                continue
            dx = float(ch - full_chain[prev_pos])
            dz = float(zz - full_elev[prev_pos])
        if np.isfinite(dx) and np.isfinite(dz) and abs(dx) > 1e-9:
            slopes[idx] = dz / dx
    return slopes[np.isfinite(slopes)]


def _start_percentile_slope(
    prof: dict,
    start_chainage: float,
    *,
    percentile: float = 20.0,
    neighbors_each_side: int = 5,
) -> Optional[float]:
    full_chain, full_elev = _profile_surface_for_seed(prof)
    local_chain, local_elev = _local_surface_window_for_seed(
        prof,
        start_chainage,
        neighbors_each_side=neighbors_each_side,
    )
    slopes = _point_slopes_for_profile_window(full_chain, full_elev, local_chain, local_elev)
    if slopes.size <= 0:
        return None
    slope = float(np.nanpercentile(slopes, float(percentile)))
    return slope if np.isfinite(slope) else None


def start_theta_deg_for_cp1(
    prof: dict,
    start_chainage: float,
    *,
    percentile: float = 20.0,
    neighbors_each_side: int = 5,
) -> Optional[float]:
    full_chain, full_elev = _profile_surface_for_seed(prof)
    local_chain, local_elev = _local_surface_window_for_seed(
        prof,
        start_chainage,
        neighbors_each_side=neighbors_each_side,
    )
    slopes = _point_slopes_for_profile_window(full_chain, full_elev, local_chain, local_elev)
    if slopes.size <= 0:
        return None
    theta_list = np.degrees(np.arctan(slopes))
    theta_list = theta_list[np.isfinite(theta_list)]
    if theta_list.size <= 0:
        return None
    theta_deg = float(np.nanpercentile(theta_list, float(percentile)))
    return theta_deg if np.isfinite(theta_deg) else None


def start_slope_for_cp1(
    prof: dict,
    start_chainage: float,
    *,
    percentile: float = 20.0,
    neighbors_each_side: int = 5,
) -> Optional[float]:
    start_slope = _start_percentile_slope(
        prof,
        start_chainage,
        percentile=percentile,
        neighbors_each_side=neighbors_each_side,
    )
    return float(start_slope) if start_slope is not None and np.isfinite(start_slope) else None


def slope_theta_specs_for_groups(
    prof: dict,
    groups: list,
    *,
    start_chainage: float,
) -> Optional[List[Dict[str, Any]]]:
    rows = _normalize_groups_with_meta(groups)
    if not rows:
        return None

    specs: List[Dict[str, Any]] = []
    for idx, group in enumerate(rows):
        if idx == 0:
            theta_deg = start_theta_deg_for_cp1(
                prof,
                float(start_chainage),
                percentile=20.0,
                neighbors_each_side=5,
            )
            source = "theta_percentile_20"
        else:
            theta_deg = median_displacement_theta_deg_for_group(group, prof)
            source = "median_theta_vector"
        if theta_deg is None or not np.isfinite(theta_deg):
            return None
        specs.append({
            "segment_index": int(idx),
            "group_id": str(group.get("id", "") or ""),
            "theta_deg": float(theta_deg),
            "theta_source": source,
        })
    return specs


def median_displacement_theta_deg_for_group(group: dict, prof: Optional[dict]) -> Optional[float]:
    chain = np.asarray((prof or {}).get("chain", []), dtype=float)
    d_para = np.asarray((prof or {}).get("d_para", []), dtype=float)
    dz = np.asarray((prof or {}).get("dz", []), dtype=float)
    n = int(min(chain.size, d_para.size, dz.size))
    if n <= 0:
        return None
    chain = chain[:n]
    d_para = d_para[:n]
    dz = dz[:n]
    try:
        s = float(group.get("start", group.get("start_chainage", np.nan)))
        e = float(group.get("end", group.get("end_chainage", np.nan)))
    except Exception:
        return None
    if not (np.isfinite(s) and np.isfinite(e)):
        return None
    if e < s:
        s, e = e, s
    mask = (chain >= float(s)) & (chain <= float(e)) & np.isfinite(d_para) & np.isfinite(dz)
    if int(np.count_nonzero(mask)) <= 0:
        return None
    theta = np.degrees(np.arctan2(dz[mask], d_para[mask]))
    theta = theta[np.isfinite(theta)]
    if theta.size <= 0:
        return None
    med = float(np.median(theta))
    return med if np.isfinite(med) else None


def median_displacement_slope_for_group(group: dict, prof: Optional[dict]) -> Optional[float]:
    chain = np.asarray((prof or {}).get("chain", []), dtype=float)
    d_para = np.asarray((prof or {}).get("d_para", []), dtype=float)
    dz = np.asarray((prof or {}).get("dz", []), dtype=float)
    n = int(min(chain.size, d_para.size, dz.size))
    if n <= 0:
        return None
    chain = chain[:n]
    d_para = d_para[:n]
    dz = dz[:n]
    try:
        s = float(group.get("start", group.get("start_chainage", np.nan)))
        e = float(group.get("end", group.get("end_chainage", np.nan)))
    except Exception:
        return None
    if not (np.isfinite(s) and np.isfinite(e)):
        return None
    if e < s:
        s, e = e, s
    mask = (chain >= float(s)) & (chain <= float(e)) & np.isfinite(d_para) & np.isfinite(dz) & (np.abs(d_para) > 1e-9)
    if int(np.count_nonzero(mask)) <= 0:
        return None
    slopes = dz[mask] / d_para[mask]
    slopes = slopes[np.isfinite(slopes)]
    if slopes.size <= 0:
        return None
    med = float(np.median(slopes))
    return med if np.isfinite(med) else None


def _build_segment_record(
    p0: Tuple[float, float],
    p3: Tuple[float, float],
    toe: Tuple[float, float],
    theta_deg: float,
    *,
    segment_index: int,
    group_id: str,
    theta_source: str,
    handle_ratio: float = SEGMENT_HANDLE_RATIO,
) -> Optional[Dict[str, Any]]:
    x0, z0 = map(float, p0)
    x3, z3 = map(float, p3)
    xt, zt = map(float, toe)
    if not (np.isfinite(x0) and np.isfinite(z0) and np.isfinite(x3) and np.isfinite(z3) and x3 > x0):
        return None

    handle_len = float(max(1e-6, (x3 - x0) * float(handle_ratio)))
    slope = float(np.tan(np.deg2rad(float(theta_deg))))
    if not np.isfinite(slope):
        return None
    p1 = (x0 + handle_len, z0 + slope * handle_len)

    out_dx = float(xt - x3)
    out_dz = float(zt - z3)
    out_norm = float(np.hypot(out_dx, out_dz))
    if not np.isfinite(out_norm) or out_norm <= 1e-9:
        out_dx = float(x3 - x0)
        out_dz = float(z3 - z0)
        out_norm = float(np.hypot(out_dx, out_dz))
    if not np.isfinite(out_norm) or out_norm <= 1e-9:
        out_dx = 1.0
        out_dz = 0.0
        out_norm = 1.0
    ux = out_dx / out_norm
    uz = out_dz / out_norm
    p2 = (x3 - handle_len * ux, z3 - handle_len * uz)

    ctrl = [
        [float(x0), float(z0)],
        [float(p1[0]), float(p1[1])],
        [float(p2[0]), float(p2[1])],
        [float(x3), float(z3)],
    ]
    return {
        "segment_index": int(segment_index),
        "group_id": str(group_id or ""),
        "theta_deg": float(theta_deg),
        "theta_source": str(theta_source or ""),
        "degree": 3,
        "weights": [1.0, 1.0, 1.0, 1.0],
        "control_points": ctrl,
        "point_semantics": "segment_controls",
    }


def build_segmented_slope_guided_params(
    *,
    prof: dict,
    groups: list,
    endpoints: Tuple[float, float, float, float],
    control_points: Optional[List[List[float]]] = None,
) -> Optional[Dict[str, Any]]:
    s0, z0, s1, z1 = map(float, endpoints)
    group_rows = _normalize_groups_with_meta(groups)
    if not group_rows:
        return _build_nurbs_params_from_control_points([float(s0), float(s1)], [float(z0), float(z1)])

    specs = slope_theta_specs_for_groups(prof, group_rows, start_chainage=float(s0))
    if not specs or len(specs) != len(group_rows):
        return None

    cp_chain = build_default_nurbs_chainage(float(s0), float(s1), group_rows)
    expected_count = len(group_rows) + 1
    if len(cp_chain) != expected_count:
        return None

    cp_elev: List[float]
    cps_in = np.asarray(control_points or [], dtype=float)
    if cps_in.ndim == 2 and cps_in.shape[0] >= 2 and cps_in.shape[1] >= 2:
        order = np.argsort(cps_in[:, 0])
        cps_in = cps_in[order]
        xs = np.asarray(cps_in[:, 0], dtype=float)
        zs = np.asarray(cps_in[:, 1], dtype=float)
        keep = np.isfinite(xs) & np.isfinite(zs)
        xs = xs[keep]
        zs = zs[keep]
        if xs.size >= 2:
            cp_elev = np.interp(np.asarray(cp_chain, dtype=float), xs, zs).astype(float).tolist()
            cp_elev[0] = float(z0)
            cp_elev[-1] = float(z1)
        else:
            cp_elev = []
    else:
        cp_elev = []

    if len(cp_elev) != expected_count:
        cp_elev = [float(z0)]
        for idx in range(max(0, len(group_rows) - 1)):
            prev_x = float(cp_chain[idx])
            next_x = float(cp_chain[idx + 1])
            prev_z = float(cp_elev[-1])
            theta_deg = float(specs[idx]["theta_deg"])
            slope = float(np.tan(np.deg2rad(theta_deg)))
            if not np.isfinite(slope) or not np.isfinite(next_x) or next_x <= prev_x:
                return None
            cp_elev.append(float(prev_z + slope * (next_x - prev_x)))
        cp_elev.append(float(z1))

    cp_visible = [[float(x), float(z)] for x, z in zip(cp_chain, cp_elev)]
    toe = (float(s1), float(z1))
    segments: List[Dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        if idx + 1 >= len(cp_visible):
            return None
        seg = _build_segment_record(
            (float(cp_visible[idx][0]), float(cp_visible[idx][1])),
            (float(cp_visible[idx + 1][0]), float(cp_visible[idx + 1][1])),
            toe,
            float(spec["theta_deg"]),
            segment_index=int(spec["segment_index"]),
            group_id=str(spec["group_id"]),
            theta_source=str(spec["theta_source"]),
        )
        if seg is None:
            return None
        segments.append(seg)

    return {
        "degree": 3,
        "control_points": cp_visible,
        "weights": [1.0] * len(cp_visible),
        "point_semantics": "intersections",
        "segments": segments,
    }


def _build_default_nurbs_params_bezier_like(
    *,
    prof: dict,
    groups: list,
    base_curve: dict,
    endpoints: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    s0, z0, s1, z1 = endpoints
    cp_chain = build_default_nurbs_chainage(float(s0), float(s1), groups)

    xb = np.asarray((base_curve or {}).get("chain", []), dtype=float)
    zb = np.asarray((base_curve or {}).get("elev", []), dtype=float)
    mb = np.isfinite(xb) & np.isfinite(zb)
    xb = xb[mb]
    zb = zb[mb]
    if xb.size >= 2:
        cp_elev = np.interp(np.asarray(cp_chain, dtype=float), xb, zb).tolist()
    else:
        chain = np.asarray(prof.get("chain", []), dtype=float)
        elev = np.asarray(prof.get("elev_s", []), dtype=float)
        m = np.isfinite(chain) & np.isfinite(elev)
        chain = chain[m]
        elev = elev[m]
        if chain.size >= 2:
            cp_elev = np.interp(np.asarray(cp_chain, dtype=float), chain, elev).tolist()
        else:
            cp_elev = np.linspace(z0, z1, len(cp_chain)).tolist()

    cp_elev = _clamp_interior_control_points_below_ground(cp_chain, cp_elev, prof, clearance=0.35)
    cp_elev[0] = z0
    cp_elev[-1] = z1
    return _build_nurbs_params_from_control_points(cp_chain, cp_elev)


def _build_default_nurbs_params_slope_guided(
    *,
    prof: dict,
    groups: list,
    base_curve: dict,
    endpoints: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    s0, z0, s1, z1 = map(float, endpoints)
    params = build_segmented_slope_guided_params(
        prof=prof,
        groups=groups,
        endpoints=(float(s0), float(z0), float(s1), float(z1)),
        control_points=None,
    )
    if params:
        return params
    return _build_default_nurbs_params_bezier_like(prof=prof, groups=groups, base_curve=base_curve, endpoints=endpoints)


def build_default_nurbs_params(
    *,
    prof: dict,
    groups: list,
    base_curve: dict,
    endpoints: Tuple[float, float, float, float],
    nurbs_seed_method: Optional[str] = None,
) -> Dict[str, Any]:
    seed_method = normalize_nurbs_seed_method(nurbs_seed_method)
    if seed_method == "slope_guided":
        return _build_default_nurbs_params_slope_guided(
            prof=prof,
            groups=groups,
            base_curve=base_curve,
            endpoints=endpoints,
        )
    return _build_default_nurbs_params_bezier_like(
        prof=prof,
        groups=groups,
        base_curve=base_curve,
        endpoints=endpoints,
    )


def reconcile_nurbs_params_with_groups(
    *,
    prof: dict,
    groups: list,
    base_curve: dict,
    params: Optional[Dict[str, Any]],
    endpoints: Tuple[float, float, float, float],
    nurbs_seed_method: Optional[str] = None,
) -> Dict[str, Any]:
    seed_method = normalize_nurbs_seed_method(nurbs_seed_method)
    if seed_method == "slope_guided":
        s0, z0, s1, z1 = map(float, endpoints)
        cps_in = np.asarray(((params or {}) if isinstance(params, dict) else {}).get("control_points", []), dtype=float)
        rebuilt = build_segmented_slope_guided_params(
            prof=prof,
            groups=groups,
            endpoints=(float(s0), float(z0), float(s1), float(z1)),
            control_points=cps_in.tolist() if cps_in.ndim == 2 and cps_in.shape[0] >= 2 else None,
        )
        if rebuilt:
            return rebuilt
        return _build_default_nurbs_params_bezier_like(prof=prof, groups=groups, base_curve=base_curve, endpoints=endpoints)

    if not params:
        return build_default_nurbs_params(
            prof=prof,
            groups=groups,
            base_curve=base_curve,
            endpoints=endpoints,
            nurbs_seed_method=nurbs_seed_method,
        )

    s0, z0, s1, z1 = map(float, endpoints)
    expected_chain = np.asarray(build_default_nurbs_chainage(float(s0), float(s1), groups), dtype=float)
    if expected_chain.ndim != 1 or expected_chain.size < 2:
        return build_default_nurbs_params(
            prof=prof,
            groups=groups,
            base_curve=base_curve,
            endpoints=endpoints,
            nurbs_seed_method=nurbs_seed_method,
        )

    cps = np.asarray((params or {}).get("control_points", []), dtype=float)
    ws = np.asarray((params or {}).get("weights", []), dtype=float)
    deg = int((params or {}).get("degree", 3))
    if cps.ndim != 2 or cps.shape[0] < 2:
        return build_default_nurbs_params(
            prof=prof,
            groups=groups,
            base_curve=base_curve,
            endpoints=endpoints,
            nurbs_seed_method=nurbs_seed_method,
        )

    if ws.ndim != 1 or ws.size != cps.shape[0]:
        ws = np.ones(cps.shape[0], dtype=float)

    order = np.argsort(cps[:, 0])
    cps = cps[order]
    ws = ws[order]
    chain_now = np.asarray(cps[:, 0], dtype=float)
    elev_now = np.asarray(cps[:, 1], dtype=float)
    tol = 1e-6
    needs_rebuild = False
    if chain_now.size != expected_chain.size:
        needs_rebuild = True
    elif np.any(~np.isfinite(chain_now)) or np.any(~np.isfinite(elev_now)):
        needs_rebuild = True
    elif np.any(np.diff(chain_now) <= tol):
        needs_rebuild = True
    elif (float(chain_now[0]) < float(s0) - tol) or (float(chain_now[-1]) > float(s1) + tol):
        needs_rebuild = True

    if needs_rebuild:
        elev_interp = np.interp(expected_chain, chain_now, elev_now)
        weight_interp = np.interp(expected_chain, chain_now, ws)
        elev_interp[0] = float(z0)
        elev_interp[-1] = float(z1)
        weight_interp = np.where(np.isfinite(weight_interp) & (weight_interp > 0.0), weight_interp, 1.0)
        return {
            "degree": int(max(1, min(deg, expected_chain.size - 1))),
            "control_points": np.vstack([expected_chain, elev_interp]).T.tolist(),
            "weights": weight_interp.tolist(),
        }

    cps[0, 0], cps[0, 1] = float(s0), float(z0)
    cps[-1, 0], cps[-1, 1] = float(s1), float(z1)
    return {
        "degree": int(max(1, min(deg, cps.shape[0] - 1))),
        "control_points": cps.tolist(),
        "weights": ws.tolist(),
    }


def clamp_curve_below_ground(
    curve: Optional[Dict[str, np.ndarray]],
    *,
    prof: Optional[dict],
    clearance: float = 0.3,
    keep_endpoints: bool = True,
) -> Optional[Dict[str, np.ndarray]]:
    if not curve:
        return curve
    pf = prof if isinstance(prof, dict) else {}
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

    gch = np.asarray((pf or {}).get("chain", []), dtype=float)
    gz = np.asarray((pf or {}).get("elev_s", []), dtype=float)
    mg = np.isfinite(gch) & np.isfinite(gz)
    gch = gch[mg]
    gz = gz[mg]
    if gch.size < 2:
        return {"chain": ch, "elev": zz}
    go = np.argsort(gch)
    gch = gch[go]
    gz = gz[go]

    try:
        clear_m = float(clearance)
    except Exception:
        clear_m = 0.3
    clear_m = max(0.0, clear_m)
    g_at = np.interp(ch, gch, gz)
    zz2 = zz.copy()
    if keep_endpoints and zz2.size >= 3:
        zz2[1:-1] = np.minimum(zz2[1:-1], g_at[1:-1] - clear_m)
    else:
        zz2[:] = np.minimum(zz2, g_at - clear_m)
    return {"chain": ch, "elev": zz2}
