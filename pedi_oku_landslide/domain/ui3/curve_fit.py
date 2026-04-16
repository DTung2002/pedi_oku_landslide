from typing import Optional

import numpy as np


def estimate_slip_curve(
    prof: dict,
    groups: list,
    ds: float = 0.2,
    smooth_factor: float = 0.1,
    depth_gain: float = 3.0,
    min_depth: float = 1.0,
) -> dict:
    _sg = None

    chain = np.asarray(prof.get("chain"), float)
    elevg = np.asarray(prof.get("elev_s"), float)
    dz_raw = prof.get("dz", None)
    if dz_raw is None:
        dz = np.full_like(chain, np.nan, dtype=float)
    else:
        dz = np.asarray(dz_raw, float)
        if dz.shape != chain.shape:
            dz = np.full_like(chain, np.nan, dtype=float)

    ok = np.isfinite(chain) & np.isfinite(elevg)
    chain = chain[ok]
    elevg = elevg[ok]
    dz = dz[ok]
    if chain.size < 3 or not groups:
        return {"chain": [], "elev": [], "depth": []}

    s_a = float(min(g["start"] for g in groups))
    s_b = float(max(g["end"] for g in groups))
    if s_b <= s_a:
        return {"chain": [], "elev": [], "depth": []}

    s_new = np.arange(s_a, s_b + ds * 0.5, ds)
    z_g = np.interp(s_new, chain, elevg)
    mslip = (chain >= s_a) & (chain <= s_b) & np.isfinite(dz)
    if np.any(mslip):
        dz_90 = float(np.nanpercentile(np.abs(dz[mslip]), 90))
    else:
        dz_90 = 0.0
    depth = max(float(min_depth), dz_90 * float(depth_gain))

    u = (s_new - s_a) / (s_b - s_a)
    u = np.clip(u, 0.0, 1.0)
    w = 4.0 * u * (1.0 - u)
    z_target = z_g - depth * w

    if (smooth_factor > 0) and (_sg is not None) and (s_new.size >= 7):
        win = int(max(7, int(round(s_new.size * smooth_factor)) | 1))
        try:
            z_target = _sg(z_target, win, 2, mode="interp")
        except Exception:
            pass

    depth_arr = np.maximum(z_g - z_target, 0.0)
    return {"chain": s_new, "elev": z_target, "depth": depth_arr}


def fit_bezier_smooth_curve(chain, elevg, target_s, target_z, c0=0.30, c1=0.30, clearance=0.12):
    target_s = np.asarray(target_s, float)
    target_z = np.asarray(target_z, float)
    ok = np.isfinite(target_s) & np.isfinite(target_z)
    target_s, target_z = target_s[ok], target_z[ok]
    if target_s.size < 4:
        return {"chain": target_s, "elev": target_z}

    s0, s1 = float(target_s[0]), float(target_s[-1])
    if not np.isfinite(s0) or not np.isfinite(s1) or s1 <= s0:
        return {"chain": target_s, "elev": target_z}
    length = s1 - s0

    z0 = float(np.interp(s0, chain, elevg))
    z3 = float(np.interp(s1, chain, elevg))
    p0x, p3x = s0, s1
    p1x, p2x = s0 + c0 * length, s1 - c1 * length

    u = (target_s - s0) / length
    u = np.clip(u, 0.0, 1.0)
    b0 = (1 - u) ** 3
    b1 = 3 * (1 - u) ** 2 * u
    b2 = 3 * (1 - u) * u ** 2
    b3 = u ** 3
    a = np.vstack([b1, b2]).T
    rhs = target_z - (b0 * z0 + b3 * z3)

    try:
        sol, *_ = np.linalg.lstsq(a, rhs, rcond=None)
        z1, z2 = float(sol[0]), float(sol[1])
    except np.linalg.LinAlgError:
        zmed = float(np.nanmedian(target_z))
        z1 = z2 = zmed

    uu = np.linspace(0, 1, max(50, int(length * 5)))
    c0v = (1 - uu) ** 3
    c1v = 3 * (1 - uu) ** 2 * uu
    c2v = 3 * (1 - uu) * uu ** 2
    c3v = uu ** 3
    s_bez = c0v * p0x + c1v * p1x + c2v * p2x + c3v * p3x
    z_bez = c0v * z0 + c1v * z1 + c2v * z2 + c3v * z3

    zg = np.interp(s_bez, chain, elevg)
    if z_bez.size > 2:
        z_bez[1:-1] = np.minimum(z_bez[1:-1], zg[1:-1] - float(clearance))
    z_bez[0] = zg[0]
    z_bez[-1] = zg[-1]
    return {"chain": s_bez, "elev": z_bez}


def _make_open_uniform_knot(n_ctrl: int, degree: int) -> np.ndarray:
    m = int(n_ctrl) + int(degree) + 1
    kv = np.zeros(m, dtype=float)
    kv[: degree + 1] = 0.0
    kv[-(degree + 1):] = 1.0
    n_internal = m - 2 * (degree + 1)
    if n_internal > 0:
        kv[degree + 1: degree + 1 + n_internal] = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
    return kv


def _bspline_basis_all(u: float, degree: int, knot: np.ndarray, n_ctrl: int) -> np.ndarray:
    n = np.zeros(n_ctrl, dtype=float)
    for i in range(n_ctrl):
        if (knot[i] <= u < knot[i + 1]) or (u == 1.0 and knot[i] <= u <= knot[i + 1] and knot[i + 1] == 1.0):
            n[i] = 1.0

    for p in range(1, degree + 1):
        np_next = np.zeros(n_ctrl, dtype=float)
        for i in range(n_ctrl):
            left = 0.0
            right = 0.0
            left_den = knot[i + p] - knot[i]
            right_den = knot[i + p + 1] - knot[i + 1]
            if left_den != 0.0:
                left = (u - knot[i]) / left_den * n[i]
            if right_den != 0.0 and (i + 1) < n_ctrl:
                right = (knot[i + p + 1] - u) / right_den * n[i + 1]
            np_next[i] = left + right
        n = np_next
    return n


def _eval_nurbs_curve(
    ctrl_pts: np.ndarray,
    weights: np.ndarray,
    degree: int,
    n_samples: int,
    knot: Optional[np.ndarray] = None,
) -> np.ndarray:
    n_ctrl = int(ctrl_pts.shape[0])
    if knot is None:
        knot = _make_open_uniform_knot(n_ctrl, degree)

    us = np.linspace(0.0, 1.0, int(max(8, n_samples)))
    curve = np.zeros((us.size, 2), dtype=float)
    for j, u in enumerate(us):
        n = _bspline_basis_all(float(u), degree, knot, n_ctrl)
        wn = weights * n
        denom = float(np.sum(wn))
        if denom == 0.0:
            curve[j, :] = np.nan
        else:
            curve[j, :] = (wn @ ctrl_pts) / denom
    return curve


def evaluate_nurbs_curve(chain_ctrl, elev_ctrl, weights=None, degree: int = 3, n_samples: int = 300) -> dict:
    ch = np.asarray(chain_ctrl, dtype=float)
    zz = np.asarray(elev_ctrl, dtype=float)
    if ch.ndim != 1 or zz.ndim != 1 or ch.size != zz.size or ch.size < 2:
        return {"chain": np.array([], dtype=float), "elev": np.array([], dtype=float)}
    m = np.isfinite(ch) & np.isfinite(zz)
    ch = ch[m]
    zz = zz[m]
    if ch.size < 2:
        return {"chain": np.array([], dtype=float), "elev": np.array([], dtype=float)}
    order = np.argsort(ch)
    ch = ch[order]
    zz = zz[order]

    if weights is None:
        ww = np.ones(ch.size, dtype=float)
    else:
        ww = np.asarray(weights, dtype=float)
        if ww.ndim != 1 or ww.size != ch.size:
            ww = np.ones(ch.size, dtype=float)
        ww = np.where(np.isfinite(ww) & (ww > 0), ww, 1.0)

    n_ctrl = int(ch.size)
    deg = int(max(1, min(int(degree), n_ctrl - 1)))
    curve = _eval_nurbs_curve(np.vstack([ch, zz]).T, ww, degree=deg, n_samples=int(max(8, n_samples)))
    sx = curve[:, 0]
    sz = curve[:, 1]
    fin = np.isfinite(sx) & np.isfinite(sz)
    sx = sx[fin]
    sz = sz[fin]
    if sx.size < 2:
        return {"chain": np.array([], dtype=float), "elev": np.array([], dtype=float)}
    order2 = np.argsort(sx)
    return {"chain": sx[order2], "elev": sz[order2]}


def evaluate_piecewise_cubic_segments(segments, n_samples: int = 300) -> dict:
    segs = list(segments or [])
    if not segs:
        return {"chain": np.array([], dtype=float), "elev": np.array([], dtype=float)}

    widths = []
    valid_segments = []
    for seg in segs:
        cps = np.asarray((seg or {}).get("control_points", []), dtype=float)
        if cps.ndim != 2 or cps.shape[0] < 4:
            continue
        dx = float(abs(cps[-1, 0] - cps[0, 0]))
        if not np.isfinite(dx):
            dx = 0.0
        widths.append(max(dx, 1e-6))
        valid_segments.append(seg)
    if not valid_segments:
        return {"chain": np.array([], dtype=float), "elev": np.array([], dtype=float)}

    total_width = float(sum(widths))
    chain_parts = []
    elev_parts = []
    for idx, (seg, width) in enumerate(zip(valid_segments, widths)):
        cps = np.asarray((seg or {}).get("control_points", []), dtype=float)
        ww = np.asarray((seg or {}).get("weights", []), dtype=float)
        seg_degree = int((seg or {}).get("degree", 3))
        seg_samples = int(max(16, round(float(n_samples) * (width / total_width))))
        out = evaluate_nurbs_curve(
            chain_ctrl=cps[:, 0],
            elev_ctrl=cps[:, 1],
            weights=ww if ww.ndim == 1 and ww.size == cps.shape[0] else None,
            degree=seg_degree,
            n_samples=seg_samples,
        )
        sx = np.asarray(out.get("chain", []), dtype=float)
        sz = np.asarray(out.get("elev", []), dtype=float)
        keep = np.isfinite(sx) & np.isfinite(sz)
        sx = sx[keep]
        sz = sz[keep]
        if sx.size < 2:
            continue
        if idx > 0:
            sx = sx[1:]
            sz = sz[1:]
        if sx.size <= 0:
            continue
        chain_parts.append(sx)
        elev_parts.append(sz)

    if not chain_parts:
        return {"chain": np.array([], dtype=float), "elev": np.array([], dtype=float)}
    return {
        "chain": np.concatenate(chain_parts),
        "elev": np.concatenate(elev_parts),
    }
