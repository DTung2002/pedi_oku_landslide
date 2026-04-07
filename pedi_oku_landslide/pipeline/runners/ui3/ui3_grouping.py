from typing import Any, Dict, List, Tuple

import numpy as np

WORKFLOW_GROUP_MIN_LEN_M = 0.0


def _mk_boundary(x: float, reason: str, score: float, fixed: bool = False) -> Dict[str, Any]:
    return {"x": float(x), "reasons": [str(reason)], "score": float(score), "fixed": bool(fixed)}


def _has_curvature_reason(c: Dict[str, Any], curvature_reason_prefix: str = "curvature_gt_") -> bool:
    return any(str(r).startswith(curvature_reason_prefix) for r in list(c.get("reasons", [])))


def _merge_close_boundaries(cands: List[Dict[str, Any]], tol_m: float = 1e-6) -> List[Dict[str, Any]]:
    if not cands:
        return []
    cands = sorted(cands, key=lambda t: float(t["x"]))
    out: List[Dict[str, Any]] = []
    for c in cands:
        if not out:
            out.append(dict(c))
            continue
        if abs(float(c["x"]) - float(out[-1]["x"])) <= float(tol_m):
            out[-1]["fixed"] = bool(out[-1]["fixed"]) or bool(c["fixed"])
            out[-1]["reasons"] = sorted(set(list(out[-1]["reasons"]) + list(c["reasons"])))
            if float(c["score"]) > float(out[-1]["score"]):
                out[-1]["x"] = float(c["x"])
                out[-1]["score"] = float(c["score"])
                if "curvature_value" in c:
                    out[-1]["curvature_value"] = float(c["curvature_value"])
        else:
            out.append(dict(c))
    for cur in out:
        if not _has_curvature_reason(cur):
            cur.pop("curvature_value", None)
    return out


def _snap_slip_span_end_to_rdp_node(smin: float, smax: float, xs: np.ndarray, snap_tol_m: float) -> float:
    xs = np.asarray(xs, dtype=float)
    tol = max(0.0, float(snap_tol_m))
    if tol <= 0.0 or xs.ndim != 1 or xs.size <= 0:
        return float(smax)

    keep = np.isfinite(xs) & (xs >= float(smin)) & (xs <= float(smax))
    if int(np.count_nonzero(keep)) <= 0:
        return float(smax)

    snapped_end = float(np.nanmax(xs[keep]))
    if (float(smax) - snapped_end) <= tol:
        return snapped_end
    return float(smax)


def _prune_first_20m_descending_curvature_pair(
    cands: List[Dict[str, Any]],
    *,
    window_m: float = 20.0,
    curvature_reason_prefix: str = "curvature_gt_",
    slip_start_reason: str = "slip_span_start",
) -> List[Dict[str, Any]]:
    if not cands:
        return []
    out = [dict(c) for c in sorted(cands, key=lambda t: float(t.get("x", 0.0)))]
    win = max(0.0, float(window_m))
    if win <= 0.0:
        return out

    slip_start_x = None
    for c in out:
        reasons = [str(r) for r in list(c.get("reasons", []))]
        if any(r == slip_start_reason for r in reasons):
            try:
                slip_start_x = float(c.get("x", np.nan))
            except Exception:
                slip_start_x = None
            break
    if slip_start_x is None or not np.isfinite(slip_start_x):
        return out

    eligible_idxs: List[int] = []
    for idx, c in enumerate(out):
        if not _has_curvature_reason(c, curvature_reason_prefix):
            continue
        try:
            x = float(c.get("x", np.nan))
            kval = float(c.get("curvature_value", np.nan))
        except Exception:
            continue
        if not (np.isfinite(x) and np.isfinite(kval)):
            continue
        if not (float(slip_start_x) < x <= (float(slip_start_x) + win)):
            continue
        eligible_idxs.append(idx)

    for pos in range(len(eligible_idxs) - 1):
        left_idx = eligible_idxs[pos]
        right_idx = eligible_idxs[pos + 1]
        left_k = float(out[left_idx].get("curvature_value", np.nan))
        right_k = float(out[right_idx].get("curvature_value", np.nan))
        if not (np.isfinite(left_k) and np.isfinite(right_k)):
            continue
        if left_k >= right_k:
            continue
        for idx in sorted((left_idx, right_idx), reverse=True):
            del out[idx]
        return _merge_close_boundaries(out, tol_m=1e-6)
    return out


def _prune_vector_zero_boundaries(
    cands: List[Dict[str, Any]],
    *,
    first_curvature_gap_m: float = 2.0,
    repeat_vector_gap_m: float = 10.0,
    curvature_reason_prefix: str = "curvature_gt_",
    vector_reason: str = "vector_angle_zero_deg",
) -> List[Dict[str, Any]]:
    _ = first_curvature_gap_m, repeat_vector_gap_m, curvature_reason_prefix
    if not cands:
        return []
    out = [dict(c) for c in sorted(cands, key=lambda t: float(t.get("x", 0.0)))]

    def _has_vector_reason(c: Dict[str, Any]) -> bool:
        return any(str(r) == vector_reason for r in list(c.get("reasons", [])))

    def _drop_vector_reason(idx: int) -> None:
        cur = dict(out[idx])
        reasons = [str(r) for r in list(cur.get("reasons", [])) if str(r) != vector_reason]
        if reasons or bool(cur.get("fixed", False)):
            cur["reasons"] = reasons
            out[idx] = cur
        else:
            del out[idx]

    first_kept_vector_x = None
    i = 0
    while i < len(out):
        if not _has_vector_reason(out[i]):
            i += 1
            continue
        x = float(out[i].get("x", np.nan))
        if not np.isfinite(x):
            _drop_vector_reason(i)
            continue
        if first_kept_vector_x is None:
            first_kept_vector_x = x
            i += 1
            continue
        _drop_vector_reason(i)

    return _merge_close_boundaries(out, tol_m=1e-6)


def _normalized_vector_angle_deg(theta_deg: np.ndarray) -> np.ndarray:
    theta_deg = np.asarray(theta_deg, dtype=float)
    return ((theta_deg + 90.0) % 180.0) - 90.0


def _vector_horizontal_boundaries(chain: np.ndarray, theta_deg: np.ndarray, smin: float, smax: float, zero_tol_deg: float = 1e-6) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    chain = np.asarray(chain, dtype=float)
    theta_deg = np.asarray(theta_deg, dtype=float)
    if chain.ndim != 1 or theta_deg.ndim != 1 or chain.size != theta_deg.size or chain.size < 1:
        return out

    theta_norm = _normalized_vector_angle_deg(theta_deg)
    keep = np.isfinite(chain) & np.isfinite(theta_norm) & (chain >= float(smin)) & (chain <= float(smax))
    if int(np.count_nonzero(keep)) <= 0:
        return out

    c = np.asarray(chain[keep], dtype=float)
    t = np.asarray(theta_norm[keep], dtype=float)
    order = np.argsort(c)
    c = c[order]
    t = t[order]

    for i in range(c.size):
        if abs(float(t[i])) <= float(zero_tol_deg):
            out.append(_mk_boundary(float(c[i]), "vector_angle_zero_deg", score=0.0, fixed=False))

    for i in range(c.size - 1):
        c0 = float(c[i])
        c1 = float(c[i + 1])
        t0 = float(t[i])
        t1 = float(t[i + 1])
        if not (np.isfinite(c0) and np.isfinite(c1) and np.isfinite(t0) and np.isfinite(t1)):
            continue
        if c1 <= c0 or abs(t0) <= float(zero_tol_deg) or abs(t1) <= float(zero_tol_deg):
            continue
        if (t0 * t1) >= 0.0:
            continue
        den = t1 - t0
        if abs(den) <= 1e-12:
            continue
        frac = float(np.clip(-t0 / den, 0.0, 1.0))
        x_cross = c0 + frac * (c1 - c0)
        if smin <= x_cross <= smax:
            out.append(_mk_boundary(x_cross, "vector_angle_zero_deg", score=0.0, fixed=False))
    return _merge_close_boundaries(out, tol_m=1e-6)


def auto_group_profile_by_criteria(
    prof: Dict[str, Any],
    rdp_eps_m: float = 0.5,
    curvature_thr_abs: float = 0.02,
    smooth_radius_m: float = 0.0,
    include_curvature_threshold: bool = True,
    include_vector_angle_zero: bool = True,
) -> List[Dict[str, Any]]:
    from pedi_oku_landslide.pipeline.runners.ui3_backend import extract_curvature_rdp_nodes

    chain = np.asarray(prof.get("chain", []), dtype=float)
    slip_mask = np.asarray(prof.get("slip_mask", [])) if prof.get("slip_mask", None) is not None else None
    if slip_mask is not None and slip_mask.shape == chain.shape:
        finite_mask = np.isfinite(chain) & (slip_mask == True)
        if int(np.count_nonzero(finite_mask)) >= 2:
            smin = float(np.nanmin(chain[finite_mask]))
            smax = float(np.nanmax(chain[finite_mask]))
        elif "slip_span" in prof and prof["slip_span"]:
            smin, smax = map(float, prof["slip_span"])
        else:
            elev_raw = np.asarray(prof.get("elev_orig", []), dtype=float)
            elev_fallback = np.asarray(prof.get("elev_s", []), dtype=float)
            if elev_raw.ndim != 1 or elev_raw.size != chain.size:
                elev_raw = elev_fallback
            finite = np.isfinite(chain) & np.isfinite(elev_raw)
            if int(np.count_nonzero(finite)) < 2:
                return []
            smin = float(np.nanmin(chain[finite]))
            smax = float(np.nanmax(chain[finite]))
    elif "slip_span" in prof and prof["slip_span"]:
        smin, smax = map(float, prof["slip_span"])
    else:
        elev_raw = np.asarray(prof.get("elev_orig", []), dtype=float)
        elev_fallback = np.asarray(prof.get("elev_s", []), dtype=float)
        if elev_raw.ndim != 1 or elev_raw.size != chain.size:
            elev_raw = elev_fallback
        finite = np.isfinite(chain) & np.isfinite(elev_raw)
        if int(np.count_nonzero(finite)) < 2:
            return []
        smin = float(np.nanmin(chain[finite]))
        smax = float(np.nanmax(chain[finite]))
    if smax < smin:
        smin, smax = smax, smin
    if smax <= smin:
        return []

    nodes = extract_curvature_rdp_nodes(
        prof,
        rdp_eps_m=float(rdp_eps_m),
        smooth_radius_m=float(smooth_radius_m),
        restrict_to_slip_span=False,
    )
    xs = np.asarray(nodes.get("chain", []), dtype=float)
    ks = np.asarray(nodes.get("curvature", []), dtype=float)
    smax_effective = _snap_slip_span_end_to_rdp_node(float(smin), float(smax), xs, snap_tol_m=float(rdp_eps_m))

    boundaries_meta: List[Dict[str, Any]] = [
        _mk_boundary(float(smin), "slip_span_start", score=0.0, fixed=True),
        _mk_boundary(float(smax_effective), "slip_span_end", score=0.0, fixed=True),
    ]

    if bool(include_curvature_threshold) and xs.size >= 3 and ks.size == xs.size:
        for i in range(1, xs.size - 1):
            x = float(xs[i])
            k = float(ks[i])
            if not (np.isfinite(x) and np.isfinite(k)):
                continue
            if not (smin < x < smax_effective):
                continue
            if abs(k) > float(curvature_thr_abs):
                curv_boundary = _mk_boundary(x, f"curvature_gt_{float(curvature_thr_abs):.2f}", score=abs(k), fixed=False)
                curv_boundary["curvature_value"] = float(k)
                boundaries_meta.append(curv_boundary)

    if bool(include_vector_angle_zero):
        theta = np.asarray(prof.get("theta", []), dtype=float)
        chain = np.asarray(prof.get("chain", []), dtype=float)
        boundaries_meta.extend(_vector_horizontal_boundaries(chain, theta, float(smin), float(smax_effective)))

    boundaries_meta = _merge_close_boundaries(boundaries_meta, tol_m=1e-6)
    if bool(include_curvature_threshold):
        boundaries_meta = _prune_first_20m_descending_curvature_pair(boundaries_meta, window_m=20.0)
    if bool(include_vector_angle_zero):
        boundaries_meta = _prune_vector_zero_boundaries(boundaries_meta, first_curvature_gap_m=2.0, repeat_vector_gap_m=10.0)
    boundaries_meta = [b for b in boundaries_meta if np.isfinite(float(b.get("x", np.nan)))]
    boundaries_meta.sort(key=lambda t: float(t["x"]))
    if len(boundaries_meta) < 2:
        return [{
            "id": "G1",
            "start": float(smin),
            "end": float(smax_effective),
            "color": "#1f77b4",
            "start_reason": "slip_span_start",
            "end_reason": "slip_span_end",
        }]

    uniq_boundaries: List[Dict[str, Any]] = []
    for b in boundaries_meta:
        if not uniq_boundaries or abs(float(b["x"]) - float(uniq_boundaries[-1]["x"])) > 1e-6:
            uniq_boundaries.append(dict(b))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    groups: List[Dict[str, Any]] = []
    for left_b, right_b in zip(uniq_boundaries[:-1], uniq_boundaries[1:]):
        s = float(left_b["x"])
        e = float(right_b["x"])
        if e <= s:
            continue
        idx = len(groups) + 1
        groups.append({
            "id": f"G{idx}",
            "start": float(s),
            "end": float(e),
            "color": colors[(idx - 1) % len(colors)],
            "start_reason": "+".join(left_b.get("reasons", [])) or "boundary",
            "end_reason": "+".join(right_b.get("reasons", [])) or "boundary",
        })
    return _renumber_groups_visual_order(groups)


def clamp_groups_to_slip(prof: Dict[str, Any], groups: List[Dict[str, Any]], min_len: float = WORKFLOW_GROUP_MIN_LEN_M) -> List[Dict[str, Any]]:
    chain = np.asarray(prof.get("chain", []), dtype=float)
    elevs = np.asarray(prof.get("elev_s", []), dtype=float)
    nodes = extract_curvature_rdp_nodes(
        prof,
        rdp_eps_m=0.5,
        smooth_radius_m=0.0,
        restrict_to_slip_span=False,
    )
    xs = np.asarray(nodes.get("chain", []), dtype=float)
    raw_span = None
    slip_mask = np.asarray(prof.get("slip_mask", [])) if prof.get("slip_mask", None) is not None else None
    if slip_mask is not None and slip_mask.shape == chain.shape:
        keep = np.isfinite(chain) & (slip_mask == True)
        if int(np.count_nonzero(keep)) > 0:
            raw_span = (float(np.nanmin(chain[keep])), float(np.nanmax(chain[keep])))
    if raw_span is None and "slip_span" in prof and prof["slip_span"]:
        raw_span = tuple(map(float, prof["slip_span"]))
    if raw_span is not None:
        smin, smax = raw_span
        smax = _snap_slip_span_end_to_rdp_node(float(smin), float(smax), xs, snap_tol_m=0.5)
    else:
        keep = np.isfinite(chain) & np.isfinite(elevs)
        if int(np.count_nonzero(keep)) <= 0:
            return []
        smin = float(np.nanmin(chain[keep]))
        smax = float(np.nanmax(chain[keep]))
    if smax < smin:
        smin, smax = smax, smin

    out: List[Dict[str, Any]] = []
    for g in (groups or []):
        try:
            s = float(g.get("start", g.get("start_chainage", 0.0)))
            e = float(g.get("end", g.get("end_chainage", 0.0)))
        except Exception:
            continue
        if e < s:
            s, e = e, s
        s = max(s, smin)
        e = min(e, smax)
        if (e - s) >= float(min_len):
            out.append({
                "id": str(g.get("id", f"G{len(out) + 1}")),
                "start": float(s),
                "end": float(e),
                "color": g.get("color", None),
                "start_reason": g.get("start_reason", ""),
                "end_reason": g.get("end_reason", ""),
            })
    return _renumber_groups_visual_order(out)


def _renumber_groups_visual_order(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not groups:
        return []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    ordered: List[Dict[str, Any]] = []
    sortable: List[Tuple[float, float, Dict[str, Any]]] = []
    saw_explicit_color = False
    legacy_default_palette = True

    def _group_index_from_id(gid: str) -> int | None:
        gid = str(gid or "").strip().upper()
        if len(gid) >= 2 and gid.startswith("G") and gid[1:].isdigit():
            idx = int(gid[1:])
            return idx if idx > 0 else None
        return None

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
        gg = dict(g or {})
        gg["start"] = float(s)
        gg["end"] = float(e)
        orig_id = str(gg.get("id", gg.get("group_id", "")) or "").strip()
        gg["_orig_group_id"] = orig_id
        color = str(gg.get("color", "") or "").strip().lower()
        if color:
            saw_explicit_color = True
            orig_idx = _group_index_from_id(orig_id)
            if orig_idx is None or color != colors[(orig_idx - 1) % len(colors)].lower():
                legacy_default_palette = False
        sortable.append((float(s), float(e), gg))
    sortable.sort(key=lambda t: (t[0], t[1]))
    reassign_legacy_default_palette = saw_explicit_color and legacy_default_palette
    for idx, (_, _, gg) in enumerate(sortable, start=1):
        gg["id"] = f"G{idx}"
        color = str(gg.get("color", "") or "").strip()
        if reassign_legacy_default_palette or not color:
            color = colors[(idx - 1) % len(colors)]
        gg["color"] = color
        gg.pop("_orig_group_id", None)
        ordered.append(gg)
    return ordered
