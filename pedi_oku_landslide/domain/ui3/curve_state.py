from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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


def build_default_nurbs_params(
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

    gch = np.asarray(prof.get("chain", []), dtype=float)
    gz = np.asarray(prof.get("elev_s", []), dtype=float)
    mg = np.isfinite(gch) & np.isfinite(gz)
    gch = gch[mg]
    gz = gz[mg]
    clearance = 0.35
    if gch.size >= 2 and len(cp_elev) >= 3:
        g_at_cp = np.interp(np.asarray(cp_chain, dtype=float), gch, gz)
        cp_elev_arr = np.asarray(cp_elev, dtype=float)
        cp_elev_arr[1:-1] = np.minimum(cp_elev_arr[1:-1], g_at_cp[1:-1] - clearance)
        cp_elev = cp_elev_arr.tolist()

    cp_elev[0] = z0
    cp_elev[-1] = z1
    cps = [[float(s), float(z)] for s, z in zip(cp_chain, cp_elev)]
    w = [1.0] * len(cps)
    deg = min(3, max(1, len(cps) - 1))
    return {"degree": int(deg), "control_points": cps, "weights": w}


def reconcile_nurbs_params_with_groups(
    *,
    prof: dict,
    groups: list,
    base_curve: dict,
    params: Optional[Dict[str, Any]],
    endpoints: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    if not params:
        return build_default_nurbs_params(prof=prof, groups=groups, base_curve=base_curve, endpoints=endpoints)

    s0, z0, s1, z1 = map(float, endpoints)
    expected_chain = np.asarray(build_default_nurbs_chainage(float(s0), float(s1), groups), dtype=float)
    if expected_chain.ndim != 1 or expected_chain.size < 2:
        return build_default_nurbs_params(prof=prof, groups=groups, base_curve=base_curve, endpoints=endpoints)

    cps = np.asarray((params or {}).get("control_points", []), dtype=float)
    ws = np.asarray((params or {}).get("weights", []), dtype=float)
    deg = int((params or {}).get("degree", 3))
    if cps.ndim != 2 or cps.shape[0] < 2:
        return build_default_nurbs_params(prof=prof, groups=groups, base_curve=base_curve, endpoints=endpoints)

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
