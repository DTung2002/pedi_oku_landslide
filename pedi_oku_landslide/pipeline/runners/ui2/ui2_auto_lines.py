from typing import Dict, List, Optional, Tuple

import numpy as np
from rasterio.transform import Affine, xy as rio_xy
from shapely.geometry import LineString


def _unit(v: np.ndarray, eps: float = 1e-9):
    n = float(np.hypot(v[0], v[1]))
    if n < eps:
        return np.array([1.0, 0.0], dtype=float), n
    return (v / n), n


def _centroid_from_mask(mask: np.ndarray, transform: Affine) -> Optional[Tuple[float, float]]:
    rr, cc = np.nonzero(mask)
    if len(rr) == 0:
        return None
    r0 = rr.mean()
    c0 = cc.mean()
    x0, y0 = rio_xy(transform, r0, c0, offset="center")
    return float(x0), float(y0)


def _pca_dir_from_mask(mask: np.ndarray, transform: Affine) -> np.ndarray:
    rr, cc = np.nonzero(mask)
    if len(rr) < 2:
        return np.array([1.0, 0.0], dtype=float)
    xs, ys = rio_xy(transform, rr, cc, offset="center")
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    X = np.vstack([xs - xs.mean(), ys - ys.mean()]).T
    C = np.cov(X.T)
    vals, vecs = np.linalg.eigh(C)
    d = vecs[:, np.argmax(vals)]
    u, _ = _unit(d)
    return u


def _build_line(center_xy: Tuple[float, float], u_dir: np.ndarray, length_m: float) -> LineString:
    cx, cy = center_xy
    half = 0.5 * float(length_m)
    dx, dy = u_dir * half
    return LineString([(cx - dx, cy - dy), (cx + dx, cy + dy)])


def _bbox_length_from_mask(
    mask: np.ndarray,
    transform: Affine,
    scale: float = 1.0,
    min_length: float = 0.0,
) -> float:
    rr, cc = np.nonzero(mask)
    if len(rr) == 0:
        return 0.0
    xs, ys = rio_xy(transform, rr, cc, offset="center")
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    L = max(xs.max() - xs.min(), ys.max() - ys.min()) * float(scale)
    if not np.isfinite(L) or L <= 0.0:
        return 0.0
    return max(L, float(min_length))


def generate_auto_lines_from_arrays(
    dx: np.ndarray,
    dy: np.ndarray,
    mask: np.ndarray,
    transform: Affine,
    main_num_even: int,
    main_offset_m: float,
    cross_num_even: int,
    cross_offset_m: float,
    base_length_m: Optional[float] = None,
    min_mag_thresh: float = 1e-4,
) -> Dict[str, List[Dict]]:
    m_mask = mask > 0
    if not np.any(m_mask):
        return {"main": [], "cross": [], "debug": {}}
    center = _centroid_from_mask(m_mask, transform)
    if center is None:
        return {"main": [], "cross": [], "debug": {}}
    m_valid = m_mask & np.isfinite(dx) & np.isfinite(dy)
    if not np.any(m_valid):
        v_map = np.array([1.0, 0.0], dtype=float)
        u_main, mag_mean = _unit(v_map)
        mag_mean = 0.0
    else:
        v_pix = np.array([np.nanmean(dx[m_valid]), np.nanmean(dy[m_valid])], dtype=float)
        a, b, _, d, e, _ = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        v_map = np.array([a * v_pix[0] + b * v_pix[1], d * v_pix[0] + e * v_pix[1]], dtype=float)
        u_main, mag_mean = _unit(v_map)
    if mag_mean < float(min_mag_thresh):
        u_main = _pca_dir_from_mask(m_mask, transform)
        u_main, _ = _unit(u_main)
    u_norm = np.array([-u_main[1], u_main[0]], dtype=float)
    if base_length_m is not None and base_length_m > 0:
        L = float(base_length_m)
    else:
        L = _bbox_length_from_mask(m_mask, transform, scale=1.0, min_length=0.0)
    if not np.isfinite(L) or L <= 0.0:
        return {"main": [], "cross": [], "debug": {}}

    feats_main: List[Dict] = []
    feats_cross: List[Dict] = []
    main1 = _build_line(center, u_main, L)
    cross1 = _build_line(center, u_norm, L)
    ang_main = float(np.degrees(np.arctan2(u_main[1], u_main[0])))
    ang_cross = float(np.degrees(np.arctan2(u_norm[1], u_norm[0])))
    feats_main.append({"name": "ML1", "type": "main", "offset_m": 0.0, "angle_deg": ang_main, "geom": main1})
    feats_cross.append({"name": "CL1", "type": "cross", "offset_m": 0.0, "angle_deg": ang_cross, "geom": cross1})
    idx = 2
    for k in range(1, max(0, int(main_num_even)) // 2 + 1):
        for sgn in (+1, -1):
            off = sgn * float(main_offset_m) * k
            cx = center[0] + off * u_norm[0]
            cy = center[1] + off * u_norm[1]
            feats_main.append({
                "name": f"ML{idx}",
                "type": "main",
                "offset_m": off,
                "angle_deg": ang_main,
                "geom": _build_line((cx, cy), u_main, L),
            })
            idx += 1
    idx = 2
    for k in range(1, max(0, int(cross_num_even)) // 2 + 1):
        for sgn in (+1, -1):
            off = sgn * float(cross_offset_m) * k
            cx = center[0] + off * u_main[0]
            cy = center[1] + off * u_main[1]
            feats_cross.append({
                "name": f"CL{idx}",
                "type": "cross",
                "offset_m": off,
                "angle_deg": ang_cross,
                "geom": _build_line((cx, cy), u_norm, L),
            })
            idx += 1
    return {"main": feats_main, "cross": feats_cross, "debug": {"mag_mean": float(mag_mean), "u_main": u_main.tolist(), "ang_main_deg": ang_main}}
