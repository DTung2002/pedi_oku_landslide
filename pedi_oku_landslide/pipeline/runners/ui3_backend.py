# ui3_backend.py
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from shapely.geometry import LineString

matplotlib.use("Agg")  # headless
from pedi_oku_landslide.core.paths import OUTPUT_ROOT

CURVATURE_THRESHOLD_PLOT_ABS = 0.02

# Keep all UI3 outputs under the writable app output directory
def _out(*parts: str) -> str:
    return os.path.join(OUTPUT_ROOT, *parts)


#  PATH HELPERS
def auto_paths() -> dict:
    def pick_first_exists(cands):
        for p in cands:
            if p and os.path.exists(p):
                return p
        return cands[0] if cands else ""

    js = {}
    for cand in [
        _out("ui_shared_data.json"),
        _out("UI1", "ui_shared_data.json"),
    ]:
        if os.path.exists(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    js.update(json.load(f))
            except Exception:
                pass

    dem = pick_first_exists([
        _out("UI1", "before_asc_smooth.tif"),
        _out("UI1", "step1_crop", "before_ground.asc"),
        _out("UI1", "step1_crop", "before_ground.tif"),
        js.get("dem_ground_path", ""),
    ])

    dem_orig = pick_first_exists([
        _out("UI1", "step1_crop", "before_ground.asc"),
        _out("UI1", "step1_crop", "before_ground.tif"),
        js.get("dem_ground_path", ""),
    ])

    dx = pick_first_exists([
        _out("UI1", "dX.asc"),
        _out("UI1", "step2_sad", "dX.asc"),
        js.get("dx_path", ""),
    ])

    dy = pick_first_exists([
        _out("UI1", "dY.asc"),
        _out("UI1", "step2_sad", "dY.asc"),
        js.get("dy_path", ""),
    ])

    dz = pick_first_exists([
        _out("UI1", "dZ.asc"),
        _out("UI1", "step7_slipzone", "dZ_slipzone.asc"),
        _out("UI1", "step5_dz", "dZ.asc"),
        js.get("dz_path", ""),
    ])

    lines = pick_first_exists([
        _out("UI2", "step2_selected_lines", "selected_lines.gpkg"),
        js.get("lines_path", ""),
    ])

    slip = pick_first_exists([
        _out("UI1", "slip_zone.asc"),
        _out("UI1", "step7_slipzone", "slip_zone.asc"),
        js.get("slip_path", ""),
    ])

    return {"dem": dem, "dem_orig": dem_orig, "dx": dx, "dy": dy, "dz": dz, "lines": lines, "slip": slip}


# IO/GEOM HELPERS
def _open_raster(path: str):
    ds = rasterio.open(path)
    arr = ds.read(1).astype("float32")
    return ds, arr

def _rdp_polyline(points, eps):
    if len(points) <= 2:
        return points
    x1,y1 = points[0]
    x2,y2 = points[-1]
    dx, dy = x2-x1, y2-y1
    L2 = dx*dx + dy*dy
    idx, dmax = 0, -1.0
    for i,(x0,y0) in enumerate(points[1:-1], start=1):
        if L2 == 0:
            d = math.hypot(x0-x1, y0-y1)
        else:
            d = abs(dy*x0 - dx*y0 + x2*y1 - y2*x1)/math.sqrt(L2)
        if d > dmax:
            idx, dmax = i, d
    if dmax > eps:
        left  = _rdp_polyline(points[:idx+1], eps)
        right = _rdp_polyline(points[idx:], eps)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]
def _curvature_series(x, y):
    n = len(x)
    k = [0.0] * n
    for i in range(1, n-1):
        x1, y1 = x[i-1], y[i-1]
        x2, y2 = x[i],   y[i]
        x3, y3 = x[i+1], y[i+1]
        a = math.hypot(x2-x3, y2-y3)
        b = math.hypot(x3-x1, y3-y1)
        c = math.hypot(x1-x2, y1-y2)
        s = 0.5*(a+b+c)
        area2 = max(s*(s-a)*(s-b)*(s-c), 0.0)
        if area2 <= 0:
            kk = 0.0
        else:
            R = (a*b*c)/(4.0*math.sqrt(area2))
            kk = 0.0 if R == 0 else 1.0/R
            cross = (x2-x1)*(y3-y2) - (y2-y1)*(x3-x2)
            if cross < 0:
                kk = -kk
        k[i] = kk
    return k

#Hàm tính Curvature từ các điểm RDP (chính)
def _curvature_points_from_rdp(points):
    """
    Compute curvature k = 1/R for each point in an RDP polyline.
    points: list of (x, y) tuples.
    Returns list of k with same length; endpoints are 0.0.
    """
    if not points or len(points) < 3:
        return [0.0] * (len(points) if points else 0)
    k = [0.0] * len(points)
    for i in range(1, len(points) - 1):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        x3, y3 = points[i + 1]
        a = math.hypot(x2 - x3, y2 - y3)
        b = math.hypot(x3 - x1, y3 - y1)
        c = math.hypot(x1 - x2, y1 - y2)
        if a == 0 or b == 0 or c == 0:
            k[i] = 0.0
            continue
        s = 0.5 * (a + b + c)
        area2 = max(s * (s - a) * (s - b) * (s - c), 0.0)
        if area2 <= 0:
            k[i] = 0.0
            continue
        R = (a * b * c) / (4.0 * math.sqrt(area2))
        if R == 0:
            k[i] = 0.0
        else:
            curv = 1.0 / R
            cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
            k[i] = -curv if cross < 0 else curv
    return k


def _mean_filter_profile_by_chain(chain: np.ndarray, elev: np.ndarray, radius_m: float) -> np.ndarray:
    chain = np.asarray(chain, dtype=float)
    elev = np.asarray(elev, dtype=float)
    out = np.full(elev.shape, np.nan, dtype=float)
    if chain.ndim != 1 or elev.ndim != 1 or chain.size != elev.size:
        return out
    finite = np.isfinite(chain) & np.isfinite(elev)
    if int(np.count_nonzero(finite)) <= 0:
        return out
    c = chain[finite]
    z = elev[finite]
    radius = max(0.0, float(radius_m))
    vals = np.full(z.shape, np.nan, dtype=float)
    for i in range(c.size):
        mask = np.abs(c - c[i]) <= radius
        if int(np.count_nonzero(mask)) > 0:
            vals[i] = float(np.mean(z[mask]))
    out[finite] = vals
    return out


def _interp_series_at_x(xs: np.ndarray, ys: np.ndarray, xq: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    xq = np.asarray(xq, dtype=float)
    out = np.full(xq.shape, np.nan, dtype=float)
    if xs.ndim != 1 or ys.ndim != 1 or xs.size != ys.size or xs.size < 2:
        return out
    keep = np.isfinite(xs) & np.isfinite(ys)
    if int(np.count_nonzero(keep)) < 2:
        return out
    xs_u, idx = np.unique(xs[keep], return_index=True)
    ys_u = ys[keep][idx]
    if xs_u.size < 2:
        return out
    finite_q = np.isfinite(xq)
    if int(np.count_nonzero(finite_q)) > 0:
        out[finite_q] = np.interp(xq[finite_q], xs_u, ys_u)
    return out


def _curvature_from_theta_series(chain: np.ndarray, theta_deg: np.ndarray, smooth_radius_m: float = 0.0) -> np.ndarray:
    chain = np.asarray(chain, dtype=float)
    theta_deg = np.asarray(theta_deg, dtype=float)
    out = np.full(theta_deg.shape, np.nan, dtype=float)
    if chain.ndim != 1 or theta_deg.ndim != 1 or chain.size != theta_deg.size:
        return out
    finite = np.isfinite(chain) & np.isfinite(theta_deg)
    if int(np.count_nonzero(finite)) < 3:
        return out
    c = np.asarray(chain[finite], dtype=float)
    theta_rad = np.radians(np.asarray(theta_deg[finite], dtype=float))
    if float(smooth_radius_m) > 0.0:
        theta_rad = _mean_filter_profile_by_chain(c, theta_rad, float(smooth_radius_m))
    finite_theta = np.isfinite(c) & np.isfinite(theta_rad)
    if int(np.count_nonzero(finite_theta)) < 3:
        return out
    c = c[finite_theta]
    theta_rad = theta_rad[finite_theta]
    k = np.gradient(theta_rad, c)
    vals = np.full(chain[finite].shape, np.nan, dtype=float)
    vals[finite_theta] = np.asarray(k, dtype=float)
    out[finite] = vals
    return out


def _profile_elevation_for_curvature(prof: Dict[str, np.ndarray]) -> np.ndarray:
    chain = np.asarray(prof.get("chain", []), dtype=float)
    elev = np.asarray(prof.get("elev", []), dtype=float)
    elev_s = np.asarray(prof.get("elev_s", []), dtype=float)
    elev_orig = np.asarray(prof.get("elev_orig", []), dtype=float)
    src = str(prof.get("profile_dem_source", "") or "").strip().lower()

    def _valid(arr: np.ndarray) -> bool:
        return arr.ndim == 1 and arr.size == chain.size

    if src == "raw":
        for arr in (elev, elev_s, elev_orig):
            if _valid(arr):
                return arr
    elif src == "smooth":
        for arr in (elev_s, elev, elev_orig):
            if _valid(arr):
                return arr
    else:
        for arr in (elev_s, elev, elev_orig):
            if _valid(arr):
                return arr
    return np.array([], dtype=float)


def _clip_polyline_to_span(xs: np.ndarray, ys: np.ndarray, smin: float, smax: float) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.ndim != 1 or ys.ndim != 1 or xs.size != ys.size or xs.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    if not (np.isfinite(smin) and np.isfinite(smax)):
        return np.array([], dtype=float), np.array([], dtype=float)
    if smax < smin:
        smin, smax = smax, smin

    out_x: List[float] = []
    out_y: List[float] = []
    for i in range(xs.size - 1):
        x0 = float(xs[i]); x1 = float(xs[i + 1])
        y0 = float(ys[i]); y1 = float(ys[i + 1])
        if not (np.isfinite(x0) and np.isfinite(x1) and np.isfinite(y0) and np.isfinite(y1)):
            continue
        if x1 == x0:
            continue
        seg_lo = max(min(x0, x1), float(smin))
        seg_hi = min(max(x0, x1), float(smax))
        if seg_hi < seg_lo:
            continue

        def _interp(xq: float) -> float:
            t = (float(xq) - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))

        xa = float(seg_lo)
        xb = float(seg_hi)
        ya = _interp(xa)
        yb = _interp(xb)
        if not out_x or abs(out_x[-1] - xa) > 1e-9:
            out_x.append(xa)
            out_y.append(ya)
        out_x.append(xb)
        out_y.append(yb)

    if len(out_x) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.asarray(out_x, dtype=float), np.asarray(out_y, dtype=float)


def extract_curvature_rdp_nodes(
    prof: Dict[str, np.ndarray],
    *,
    rdp_eps_m: float = 0.5,
    smooth_radius_m: float = 2.0,
    restrict_to_slip_span: bool = True,
) -> Dict[str, np.ndarray]:
    chain = np.asarray(prof.get("chain", []), dtype=float)
    elev_curve = _profile_elevation_for_curvature(prof)
    empty = {
        "chain": np.array([], dtype=float),
        "elev": np.array([], dtype=float),
        "curvature": np.array([], dtype=float),
        "smin": np.array([], dtype=float),
        "smax": np.array([], dtype=float),
    }
    if chain.ndim != 1 or chain.size < 3:
        return empty
    if elev_curve.ndim != 1 or elev_curve.size != chain.size:
        return empty

    finite = np.isfinite(chain) & np.isfinite(elev_curve)
    finite0 = finite
    if int(np.count_nonzero(finite0)) < 2:
        return empty
    full_min = float(np.nanmin(chain[finite0]))
    full_max = float(np.nanmax(chain[finite0]))
    if prof.get("slip_span"):
        try:
            smin, smax = map(float, prof.get("slip_span"))
            if smax < smin:
                smin, smax = smax, smin
        except Exception:
            smin, smax = full_min, full_max
    else:
        smin, smax = full_min, full_max

    if int(np.count_nonzero(finite)) < 3:
        return empty

    chain_w = np.asarray(chain[finite], dtype=float)
    elev_w = np.asarray(elev_curve[finite], dtype=float)
    order = np.argsort(chain_w)
    chain_w = chain_w[order]
    elev_w = elev_w[order]
    elev_sm = _mean_filter_profile_by_chain(chain_w, elev_w, float(smooth_radius_m))
    finite_sm = np.isfinite(chain_w) & np.isfinite(elev_sm)
    if int(np.count_nonzero(finite_sm)) < 3:
        return empty

    pts = list(zip(chain_w[finite_sm].tolist(), elev_sm[finite_sm].tolist()))
    rdp_pts = _rdp_polyline(pts, float(rdp_eps_m))
    if len(rdp_pts) < 2:
        return empty

    k_x = np.asarray([p[0] for p in rdp_pts], dtype=float)
    k_z = np.asarray([p[1] for p in rdp_pts], dtype=float)
    k_vals = np.asarray(_curvature_points_from_rdp(rdp_pts), dtype=float)
    if restrict_to_slip_span:
        keep = np.isfinite(k_x) & (k_x >= float(smin)) & (k_x <= float(smax))
        k_x = k_x[keep]
        k_z = k_z[keep]
        k_vals = k_vals[keep]
    return {
        "chain": k_x,
        "elev": k_z,
        "curvature": k_vals,
        "smin": np.asarray([float(smin)], dtype=float),
        "smax": np.asarray([float(smax)], dtype=float),
    }


def _curvature_plot_series(
    prof: Dict[str, np.ndarray],
    *,
    rdp_eps_m: float = 0.5,
    smooth_radius_m: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    nodes = extract_curvature_rdp_nodes(
        prof,
        rdp_eps_m=float(rdp_eps_m),
        smooth_radius_m=float(smooth_radius_m),
        # Always compute on the full smoothed ground profile first, then crop
        # only for display so the visible endpoints are not re-derived locally.
        restrict_to_slip_span=False,
    )
    xs = np.asarray(nodes.get("chain", []), dtype=float)
    ys = np.asarray(nodes.get("curvature", []), dtype=float)
    if xs.size < 2 or ys.size != xs.size:
        return np.array([], dtype=float), np.array([], dtype=float)
    if prof.get("slip_span"):
        try:
            smin, smax = map(float, prof.get("slip_span"))
            return _clip_polyline_to_span(xs, ys, float(smin), float(smax))
        except Exception:
            return xs, ys
    return xs, ys


def _group_boundary_style(reasons: List[str]) -> Dict[str, Any]:
    rs = [str(r).strip().lower() for r in (reasons or []) if str(r).strip()]
    if any(r in ("slip_span_start", "slip_span_end") for r in rs):
        return {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.3, "zorder": 10}
    if any(r.startswith("curvature_gt_") for r in rs):
        return {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.3, "zorder": 10}
    if any("vector_angle_zero_deg" == r for r in rs):
        return {"color": "#d62728", "linestyle": "-", "linewidth": 1.3, "zorder": 10}
    return {"color": "#555555", "linestyle": (0, (4, 4)), "linewidth": 0.9, "zorder": 10}

def _infer_slip_curve_points(prof, group_ranges, eps_rdp=0.5, k_thr=0.0):
    ch = prof.get("chain")
    if ch is None:
        ch = prof.get("chainage_m")

    gz = prof.get("ground_z_smooth")
    if gz is None:
        gz = prof.get("ground_z")
    if gz is None:
        gz = prof.get("z")

    if ch is None or gz is None:
        return []
    if len(ch) < 4:
        return []

    ch = np.asarray(ch, dtype=float)
    gz = np.asarray(gz, dtype=float)

    pts = list(zip(map(float, ch), map(float, gz)))
    simp = _rdp_polyline(pts, eps_rdp)

    k = _curvature_series(list(map(float,ch)), list(map(float,gz)))

    bounds = []
    if group_ranges:
        for g in group_ranges:
            s = float(g.get("start", g.get("start_chainage", ch[0])))
            e = float(g.get("end",   g.get("end_chainage",   ch[-1])))
            if e < s: s,e = e,s
            bounds.append((s,e))
        bounds.sort()

    cps = []
    if bounds:
        cps.append((bounds[0][0], float(np.interp(bounds[0][0], ch, gz))))
        for (s,e) in bounds:
            i0 = max(0, int(np.searchsorted(ch, s))-1)
            i1 = min(len(ch)-1, int(np.searchsorted(ch, e)))
            if i1 <= i0:
                continue
            window = list(range(i0, i1+1))
            idx = max(window, key=lambda i:(abs(k[i]), -k[i]))
            cps.append((float(ch[idx]), float(gz[idx])))
        cps.append((bounds[-1][1], float(np.interp(bounds[-1][1], ch, gz))))
    else:
        cps = simp  # fallback

    xs, ys = zip(*sorted(set(cps)))
    if len(xs) < 3:
        return list(zip(xs,ys))
    try:
        from scipy.interpolate import splrep, splev
        tck = splrep(xs, ys, s=0)
        xs_new = np.linspace(xs[0], xs[-1], 200)
        ys_new = splev(xs_new, tck)
        return list(zip(xs_new.tolist(), ys_new.tolist()))
    except Exception:
        return list(zip(xs,ys))


# SAMPLING CORE
def _densify(line: LineString, step_m: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if line is None:
        return np.array([]), np.array([]), np.array([])

    if getattr(line, "is_empty", False):
        return np.array([]), np.array([]), np.array([])

    try:
        L = float(line.length)
    except Exception:
        return np.array([]), np.array([]), np.array([])

    if not np.isfinite(L) or L <= 0:
        return np.array([]), np.array([]), np.array([])

    n = max(2, int(np.ceil(L / step_m)) + 1)
    s = np.linspace(0.0, L, n)
    xs = np.empty(n); ys = np.empty(n)
    for i, d in enumerate(s):
        p = line.interpolate(d)
        xs[i], ys[i] = p.x, p.y
    return xs, ys, s


def _sample(ds: rasterio.io.DatasetReader, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    vals = list(ds.sample(list(zip(xs.tolist(), ys.tolist())), indexes=1))
    out = np.array([v[0] if len(v) else np.nan for v in vals], dtype="float32")
    # Normalize nodata -> NaN to avoid scaling issues in UI3 plots
    nodata = ds.nodata
    if nodata is not None:
        if np.isnan(nodata):
            out[~np.isfinite(out)] = np.nan
        else:
            out[np.isclose(out, float(nodata))] = np.nan
    return out


# PROFILE PIPELINE
def compute_profile(
    dem_path: str,
    dx_path: str,
    dy_path: str,
    dz_path: str,
    line_geom: LineString,
    step_m: float = 0.2,
    smooth_win: int = 11,
    smooth_poly: int = 2,
    slip_mask_path: Optional[str] = None,
    slip_only: bool = True,
    dem_orig_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    dem_ds, _ = _open_raster(dem_path)
    dx_ds,  _ = _open_raster(dx_path)
    dy_ds,  _ = _open_raster(dy_path)
    dz_ds,  _ = _open_raster(dz_path)

    xs, ys, chain = _densify(line_geom, step_m)
    if xs.size == 0:
        for d in (dem_ds, dx_ds, dy_ds, dz_ds): d.close()
        return {}

    elev = _sample(dem_ds, xs, ys)
    dx   = _sample(dx_ds,  xs, ys)
    dy   = _sample(dy_ds,  xs, ys)
    dz   = _sample(dz_ds,  xs, ys)

    elev_orig = None
    if dem_orig_path and os.path.exists(dem_orig_path):
        dem_orig_ds, _ = _open_raster(dem_orig_path)
        elev_orig = _sample(dem_orig_ds, xs, ys)
        dem_orig_ds.close()

    # Convert pixel displacement to meters; flip Y to north-up.
    px_m = abs(float(dx_ds.transform.a))
    py_m = abs(float(dx_ds.transform.e))
    dx = dx * px_m
    dy = -dy * py_m

    tang = np.zeros((xs.size, 2), dtype=float)
    tang[1:-1, 0] = xs[2:] - xs[:-2]
    tang[1:-1, 1] = ys[2:] - ys[:-2]
    tang[0, 0] = xs[1] - xs[0];   tang[0, 1] = ys[1] - ys[0]
    tang[-1,0]= xs[-1]-xs[-2];    tang[-1,1]= ys[-1]-ys[-2]
    nrm = np.hypot(tang[:,0], tang[:,1]); nrm[nrm==0] = 1.0
    tang[:,0] /= nrm; tang[:,1] /= nrm

    d_para = dx * tang[:,0] + dy * tang[:,1]
    theta_deg = np.degrees(np.arctan2(dz, d_para))

    mask_bool = None
    if slip_mask_path and os.path.exists(slip_mask_path):
        slip_ds, _ = _open_raster(slip_mask_path)
        slip_val = _sample(slip_ds, xs, ys)
        slip_ds.close()
        mask_bool = np.isfinite(slip_val) & (slip_val > 0)

        if slip_only:
            keep = mask_bool
            elev[~keep] = np.nan
            dx[~keep]   = np.nan
            dy[~keep]   = np.nan
            dz[~keep]   = np.nan
            d_para[~keep] = np.nan
            theta_deg[~keep] = np.nan
            if elev_orig is not None:
                elev_orig[~keep] = np.nan

    # smooth ground (DISABLED SAVITZKY-GOLAY, use UI1 smoothing instead)
    elev_s = elev.copy()

    for d in (dem_ds, dx_ds, dy_ds, dz_ds): d.close()
    if mask_bool is not None and slip_only:
        keep = np.isfinite(chain) & (mask_bool == True) & np.isfinite(elev_s)
    else:
        keep = np.isfinite(chain) & np.isfinite(elev_s)

    if np.any(keep):
        slip_span = (float(np.nanmin(chain[keep])), float(np.nanmax(chain[keep])))
    else:
        slip_span = (float(chain[0]), float(chain[-1]))

    return {
        "chain": chain,
        "x": xs, "y": ys,
        "elev": elev, "elev_s": elev_s,
        "dx": dx, "dy": dy, "dz": dz,
        "d_para": d_para,
        "theta": theta_deg,
        "slip_mask": mask_bool,
        "slip_span": slip_span,
        "elev_orig": elev_orig,
    }

# RENDER / EXPORT
def render_profile_png(
    prof: Dict[str, np.ndarray],
    out_png: str,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    vec_scale: float = 0.1,
    vec_width: float = 0.0015,
    head_len: float = 7.0,
    head_w: float = 5.0,
    highlight_theta: Optional[float] = None,
    group_ranges: Optional[List[dict]] = None,
    draw_curve: bool = False,
    save_curve_json: bool = False,
    overlay_curves: Optional[list] = None, # [(chain,elev) | (chain,elev,color,label)]
    figsize: Tuple[float, float] = (18, 10),
    dpi: int = 220,
    base_font: int = 20,
    label_font: int = 20,
    tick_font: int = 20,
    legend_font: int = 20,
    ground_lw: float = 2.2,
    ungrouped_color: str = "#bbbbbb",
    curvature_series: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    curvature_rdp_eps_m: float = 0.5,
    curvature_smooth_radius_m: float = 0.0,
) -> Tuple[str, Optional[str]]:

    if not prof:
        return "Empty profile", None

    # UI3 global font scale: shrink all legend/axis/tick text to 80%
    font_scale = 0.8
    base_font = max(1, int(round(base_font * font_scale)))
    label_font = max(1, int(round(label_font * font_scale)))
    tick_font = max(1, int(round(tick_font * font_scale)))
    legend_font = max(1, int(round(legend_font * font_scale)))
    # Additional 10% reduction for legend fonts.
    legend_font = max(1, int(round(legend_font * 0.9)))
    # Additional 10% reduction for requested axis labels/tick labels.
    axis_label_font = max(1, int(round(label_font * 0.9)))
    axis_tick_font = max(1, int(round(tick_font * 0.9)))

    x_user_min, x_user_max = x_min, x_max
    y_user_min, y_user_max = y_min, y_max
    curvature_plot_scale = -50.0
    curvature_plot_label = "Curvature plot (-50×k)"
    curvature_threshold_plot = abs(float(curvature_plot_scale)) * float(CURVATURE_THRESHOLD_PLOT_ABS)

    chain = prof["chain"]; elev_s = prof["elev_s"]
    d_para = prof["d_para"]; dz = prof["dz"]; theta = prof["theta"]
    profile_src = str(prof.get("profile_dem_source", "") or "").strip().lower()
    ground_label = "Ground"
    if profile_src == "raw":
        ground_label = "Ground (raw DEM)"
    elif profile_src == "smooth":
        ground_label = "Ground (smoothed DEM)"

    if (x_min is None) or (x_max is None):
        if "slip_span" in prof and prof["slip_span"]:
            smin, smax = prof["slip_span"]
            if x_min is None:
                x_min = float(smin)
            if x_max is None:
                x_max = float(smax)
        else:
            finite_xy = np.isfinite(chain) & np.isfinite(elev_s)
            if finite_xy.any():
                ch_min = float(np.nanmin(chain[finite_xy]))
                ch_max = float(np.nanmax(chain[finite_xy]))
                if x_min is None:
                    x_min = ch_min
                if x_max is None:
                    x_max = ch_max

    # Y-limits: theo min/max elevation + padding 2%
    if (y_min is None) or (y_max is None):
        finite_elev = np.isfinite(elev_s)
        if finite_elev.any():
            z_min = float(np.nanmin(elev_s[finite_elev]))
            z_max = float(np.nanmax(elev_s[finite_elev]))
            span = max(z_max - z_min, 0.5)
            pad = 0.02 * span
            if y_min is None:
                y_min = z_min - pad
            if y_max is None:
                y_max = z_max + pad

    def _set_chainage_xlim(ax_obj, left_val: float, right_val: float) -> None:
        lo = float(min(left_val, right_val))
        hi = float(max(left_val, right_val))
        ax_obj.set_xlim(lo, hi)

    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size': base_font}):
        # 2 rows: [vectors ; gradient/curvature]
        fig = plt.figure(figsize=figsize if figsize else (18, 12), dpi=dpi)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.0, 1.45], hspace=0.35)
        ax = fig.add_subplot(gs[0, 0])  # vectors/profile
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax)  # gradient

        ax.plot(chain, elev_s, "k-", lw=ground_lw, label=ground_label)


        finite_prof = np.isfinite(chain) & np.isfinite(elev_s) & np.isfinite(d_para) & np.isfinite(dz)
        slip_mask_arr = None
        try:
            sm = prof.get("slip_mask", None)
            if sm is not None:
                sm = np.asarray(sm)
                if sm.shape == chain.shape:
                    # Only True values are considered "inside mask"; all others default to ungrouped.
                    slip_mask_arr = (sm == True)
        except Exception:
            slip_mask_arr = None
        if slip_mask_arr is not None:
            finite_span = np.isfinite(chain) & np.isfinite(elev_s) & slip_mask_arr
            if finite_span.any():
                plot_span_min = float(np.nanmin(chain[finite_span]))
                plot_span_max = float(np.nanmax(chain[finite_span]))
            else:
                plot_span_min = None
                plot_span_max = None
        else:
            plot_span_min = None
            plot_span_max = None
        if (plot_span_min is None) or (plot_span_max is None):
            if "slip_span" in prof and prof["slip_span"]:
                plot_span_min, plot_span_max = map(float, prof["slip_span"])
                if plot_span_max < plot_span_min:
                    plot_span_min, plot_span_max = plot_span_max, plot_span_min
            else:
                finite_span = np.isfinite(chain) & np.isfinite(elev_s)
                if finite_span.any():
                    plot_span_min = float(np.nanmin(chain[finite_span]))
                    plot_span_max = float(np.nanmax(chain[finite_span]))
                else:
                    plot_span_min = None
                    plot_span_max = None
        prof_curv = dict(prof)
        if (plot_span_min is not None) and (plot_span_max is not None):
            prof_curv["slip_span"] = (float(plot_span_min), float(plot_span_max))
        else:
            prof_curv["slip_span"] = None

        def _resolve_curvature_series() -> Tuple[np.ndarray, np.ndarray]:
            if curvature_series is not None:
                try:
                    kx = np.asarray(curvature_series[0], dtype=float)
                    kv = np.asarray(curvature_series[1], dtype=float)
                    keep = np.isfinite(kx) & np.isfinite(kv)
                    kx = kx[keep]
                    kv = kv[keep]
                    if kx.size >= 2 and kv.size == kx.size:
                        order = np.argsort(kx)
                        kx = kx[order]
                        kv = kv[order]
                        if (plot_span_min is not None) and (plot_span_max is not None):
                            kx, kv = _clip_polyline_to_span(kx, kv, float(plot_span_min), float(plot_span_max))
                        if kx.size >= 2 and kv.size == kx.size:
                            return kx, kv
                except Exception:
                    pass
            return _curvature_plot_series(
                prof_curv,
                rdp_eps_m=float(curvature_rdp_eps_m),
                smooth_radius_m=float(curvature_smooth_radius_m),
            )
        if group_ranges:
            if "slip_span" in prof and prof["slip_span"]:
                smin, smax = prof["slip_span"]
            else:
                smin = float(np.nanmin(prof["chain"][finite_prof]))
                smax = float(np.nanmax(prof["chain"][finite_prof]))

            cmap = plt.get_cmap("tab10")
            prepared = []
            for gi, gr in enumerate(group_ranges):
                gid = gr.get("id", f"G{gi + 1}")
                s = float(gr.get("start", gr.get("start_chainage", 0.0)))
                e = float(gr.get("end",   gr.get("end_chainage",   0.0)))
                if e < s: s, e = e, s
                s = max(s, smin); e = min(e, smax)
                if e <= s:   
                    continue
                color = gr.get("color", None) or mcolors.to_hex(cmap(gi % 10))
                prepared.append((gi, gid, s, e, color))


            gidx = np.full(chain.shape, -1, dtype=int)
            for gi, gid, s, e, color in prepared:
                m = (chain >= s) & (chain <= e)
                if slip_mask_arr is not None:
                    m = m & slip_mask_arr
                gidx[m] = gi

            for gi, gid, s, e, color in prepared:
                m = finite_prof & (gidx == gi)
                if np.any(m):
                    ax.quiver(chain[m], elev_s[m], d_para[m], dz[m],
                              angles="xy", scale_units="xy", scale=vec_scale,
                              width=vec_width, color=color,
                              headlength=head_len, headwidth=head_w)
                    ax.plot([], [], color=color, lw=3, label="_nolegend_")

            order = np.argsort(chain)
            chain_s = chain[order]
            d_para_s = d_para[order]
            dz_s = dz[order]

            finite_s = np.isfinite(chain_s) & np.isfinite(d_para_s) & np.isfinite(dz_s)
            if (plot_span_min is not None) and (plot_span_max is not None):
                finite_s = finite_s & (chain_s >= float(plot_span_min)) & (chain_s <= float(plot_span_max))
            if finite_s.sum() >= 2:
                ch = chain_s[finite_s]
                # Vector slope angle (deg), normalized to [-90, 90].
                gradient_deg = np.degrees(np.arctan2(dz_s[finite_s], d_para_s[finite_s]))
                gradient_deg = ((gradient_deg + 90.0) % 180.0) - 90.0
                # savgol_filter disabled

                ax2.plot(ch, gradient_deg, lw=2.2, color="#2ca02c", zorder=5, label="Gradient")

            ax2.axhline(0.0, color="0.5", lw=1.0, zorder=1)
            ax2.set_ylabel("Gradient (deg)", fontsize=axis_label_font)
            ax2r = ax2.twinx()
            ax2r.set_ylabel(curvature_plot_label, fontsize=axis_label_font)

            # Curvature from the smoothed ground profile, sampled on RDP nodes.
            k_curve = None
            try:
                k_x, k_vals = _resolve_curvature_series()
                if k_x.size >= 3 and k_vals.size == k_x.size:
                    k_curve = np.asarray(k_vals, dtype=float)
                    k_curve_plot = float(curvature_plot_scale) * k_curve
                    ax2r.plot(k_x, k_curve_plot, lw=1.8, color="#222222",
                              marker="o", markersize=4, zorder=6, label="Curvature")
            except Exception:
                pass
            ax2r.axhline(float(curvature_threshold_plot), color="#cc3333", lw=1.1,
                         linestyle=(0, (4, 3)), zorder=4, label="_nolegend_")
            ax2r.axhline(-float(curvature_threshold_plot), color="#cc3333", lw=1.1,
                         linestyle=(0, (4, 3)), zorder=4, label="_nolegend_")
            if k_curve is not None and np.any(np.isfinite(k_curve)):
                qk = np.nanpercentile(np.abs(float(curvature_plot_scale) * k_curve), 98)
                qk = max(float(qk), float(curvature_threshold_plot))
                if np.isfinite(qk) and qk > 0:
                    ax2r.set_ylim(-1.2 * qk, 1.2 * qk)
            else:
                ax2r.set_ylim(-1.2 * float(curvature_threshold_plot), 1.2 * float(curvature_threshold_plot))

            ax2r.grid(False)
            ax2.grid(ls="--", lw=0.8, alpha=0.35)

            try:
                if finite_s.sum() >= 2:
                    q = np.nanpercentile(np.abs(gradient_deg), 98)
                    if np.isfinite(q) and q > 0:
                        ax2.set_ylim(-1.2 * q, 1.2 * q)
            except Exception:
                pass
            handles = ax2.get_lines() + ax2r.get_lines()
            if handles:
                keep = [(h, h.get_label()) for h in handles if not h.get_label().startswith("_")]
                if keep:
                    h_keep, l_keep = zip(*keep)
                    ax2.legend(list(h_keep), list(l_keep), loc="upper left",
                               bbox_to_anchor=(0.0, -0.18),
                               fontsize=legend_font, frameon=False, ncol=2)

            m = finite_prof & (gidx == -1)
            if np.any(m):
                ug_color = ungrouped_color or "#bbbbbb"
                ax.quiver(chain[m], elev_s[m], d_para[m], dz[m],
                          angles="xy", scale_units="xy", scale=vec_scale,
                          width=vec_width, color=ug_color, alpha=0.9,
                          headlength=head_len, headwidth=head_w)
                ax.plot([], [], color=ug_color, lw=3, label="_nolegend_")
        else:
            ax.quiver(
                chain, elev_s, d_para, dz,
                angles="xy", scale_units="xy", scale=vec_scale,
                width=vec_width, color="tab:red",
                headlength=head_len, headwidth=head_w
            )
            finite_th = np.isfinite(chain) & np.isfinite(d_para) & np.isfinite(dz)
            if (plot_span_min is not None) and (plot_span_max is not None):
                finite_th = finite_th & (chain >= float(plot_span_min)) & (chain <= float(plot_span_max))
            if finite_th.sum() >= 2:
                chain_f = chain[finite_th]
                # Vector slope angle (deg), normalized to [-90, 90].
                gradient_deg = np.degrees(np.arctan2(dz[finite_th], d_para[finite_th]))
                gradient_deg = ((gradient_deg + 90.0) % 180.0) - 90.0
                # savgol_filter disabled
                ax2.plot(chain_f, gradient_deg, color="#2ca02c", lw=2.4, zorder=5, label="Gradient")

            ax2.axhline(0.0, color="0.5", lw=1.0, zorder=1)
            ax2.set_xlabel("Chainage (m)")
            ax2.set_ylabel("Gradient (deg)")
            ax2r = ax2.twinx()
            ax2r.set_ylabel(curvature_plot_label)

            # Curvature from the smoothed ground profile, sampled on RDP nodes.
            k_curve = None
            try:
                k_x, k_vals = _resolve_curvature_series()
                if k_x.size >= 3 and k_vals.size == k_x.size:
                    k_curve = np.asarray(k_vals, dtype=float)
                    k_curve_plot = float(curvature_plot_scale) * k_curve
                    ax2r.plot(k_x, k_curve_plot, color="#222222", lw=2.0,
                              marker="o", markersize=4, zorder=6, label="Curvature")
            except Exception:
                pass
            ax2r.axhline(float(curvature_threshold_plot), color="#cc3333", lw=1.1,
                         linestyle=(0, (4, 3)), zorder=4, label="_nolegend_")
            ax2r.axhline(-float(curvature_threshold_plot), color="#cc3333", lw=1.1,
                         linestyle=(0, (4, 3)), zorder=4, label="_nolegend_")
            if k_curve is not None and np.any(np.isfinite(k_curve)):
                qk = np.nanpercentile(np.abs(float(curvature_plot_scale) * k_curve), 98)
                qk = max(float(qk), float(curvature_threshold_plot))
                if np.isfinite(qk) and qk > 0:
                    ax2r.set_ylim(-1.2 * qk, 1.2 * qk)
            else:
                ax2r.set_ylim(-1.2 * float(curvature_threshold_plot), 1.2 * float(curvature_threshold_plot))

            ax2r.grid(False)
            ax2.grid(True, linestyle="--", alpha=0.4)

            try:
                if finite_th.sum() >= 2:
                    q = np.nanpercentile(np.abs(gradient_deg), 98)
                    if np.isfinite(q) and q > 0:
                        pad = 1.2 * q
                        ax2.set_ylim(-pad, pad)
            except Exception:
                pass
            handles = ax2.get_lines() + ax2r.get_lines()
            if handles:
                keep = [(h, h.get_label()) for h in handles if not h.get_label().startswith("_")]
                if keep:
                    h_keep, l_keep = zip(*keep)
                    ax2.legend(list(h_keep), list(l_keep), loc="upper left",
                               bbox_to_anchor=(0.0, -0.18),
                               fontsize=legend_font, frameon=False, ncol=2)

            if x_min is not None and x_max is not None and (abs(float(x_max) - float(x_min)) > 1e-12):
                _set_chainage_xlim(ax2, x_min, x_max)

            if group_ranges:
                for gi, gr in enumerate(group_ranges):
                    s = float(gr.get("start", gr.get("start_chainage", 0.0)))
                    e = float(gr.get("end", gr.get("end_chainage", 0.0)))
                    if e < s: s, e = e, s
                    color = gr.get("color", None) or plt.get_cmap("tab10")(gi % 10)
                    ax2.axvspan(s, e, color=color, alpha=0.08, zorder=0)
        # --- vertical guides (apply for BOTH branches) ---
        if group_ranges:
            try:
                xmin, xmax = ax.get_xlim()
            except Exception:
                xmin, xmax = None, None
            clip_lo = clip_hi = None
            if xmin is not None and xmax is not None:
                clip_lo = float(min(xmin, xmax))
                clip_hi = float(max(xmin, xmax))

            bounds_meta: Dict[float, Dict[str, Any]] = {}
            for g in group_ranges:
                try:
                    s = float(g.get("start", g.get("start_chainage", 0.0)))
                    e = float(g.get("end", g.get("end_chainage", 0.0)))
                except Exception:
                    continue
                if e < s:
                    s, e = e, s
                if (clip_lo is not None) and (clip_hi is not None):
                    s = max(clip_lo, min(clip_hi, s))
                    e = max(clip_lo, min(clip_hi, e))
                for x, reason in (
                    (s, str(g.get("start_reason", "") or "").strip()),
                    (e, str(g.get("end_reason", "") or "").strip()),
                ):
                    key = round(float(x), 9)
                    meta = bounds_meta.setdefault(key, {"x": float(x), "reasons": []})
                    if reason:
                        meta["reasons"].append(reason)

            bounds_items = sorted(bounds_meta.values(), key=lambda t: float(t["x"]))
            bounds = [float(it["x"]) for it in bounds_items]
            if bounds:
                for it in bounds_items:
                    x = float(it["x"])
                    vkw = _group_boundary_style(list(it.get("reasons", [])))
                    ax.axvline(x, **vkw)
                    ax2.axvline(x, **vkw)
                # Number group intervals between adjacent boundaries.
                try:
                    import matplotlib.transforms as mtransforms
                    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
                    label_fs = max(8, int(round(tick_font * 0.9)))
                    label_items = []
                    for gi, gr in enumerate(group_ranges):
                        try:
                            s = float(gr.get("start", gr.get("start_chainage", np.nan)))
                            e = float(gr.get("end", gr.get("end_chainage", np.nan)))
                        except Exception:
                            continue
                        if not (np.isfinite(s) and np.isfinite(e)):
                            continue
                        if e < s:
                            s, e = e, s
                        if (clip_lo is not None) and (clip_hi is not None):
                            s = max(clip_lo, min(clip_hi, s))
                            e = max(clip_lo, min(clip_hi, e))
                        if not (np.isfinite(s) and np.isfinite(e)):
                            continue
                        if e <= s:
                            continue
                        xm = 0.5 * (s + e)
                        label_items.append((float(xm), float(s), float(e), int(gi)))
                    label_items.sort(key=lambda t: t[0])
                    for idx, (xm, _s, _e, _gi) in enumerate(label_items, start=1):
                        ax.text(
                            float(xm), 0.995, str(idx),
                            transform=trans,
                            ha="center", va="top",
                            fontsize=label_fs,
                            color="#333333",
                            zorder=60,
                        )
                except Exception:
                    pass
        # --- draw slip curve if requested ---
        # --- draw slip curve if requested (make it pop on top of vectors) ---
        # --- draw slip curve if requested (robust overlay) ---
        if draw_curve:
            try:
                curve_pts = _infer_slip_curve_points(prof, group_ranges, eps_rdp=0.5)
            except Exception:
                curve_pts = []

            if curve_pts:
                cx, cz = zip(*curve_pts)
                cx = np.asarray(cx, dtype=float)
                cz = np.asarray(cz, dtype=float)

                m = np.isfinite(cx) & np.isfinite(cz)
                cx = cx[m];
                cz = cz[m]

                try:
                    xmin, xmax = ax.get_xlim()
                    xmask = (cx >= xmin) & (cx <= xmax)
                    cx = cx[xmask];
                    cz = cz[xmask]
                except Exception:
                    pass
                try:
                    ymin, ymax = ax.get_ylim()
                    ymask = (cz >= ymin) & (cz <= ymax)
                    cx = cx[ymask];
                    cz = cz[ymask]
                except Exception:
                    pass

                if cx.size > 1:
                    ln, = ax.plot(
                        cx, cz,
                        color="#bf00ff", linewidth=3.0, zorder=50, label="Slip curve"
                    )
                    try:
                        import matplotlib.patheffects as pe
                        ln.set_path_effects([pe.Stroke(linewidth=5, foreground="white"), pe.Normal()])
                    except Exception:
                        pass

                if save_curve_json and out_png:
                    cjson = out_png.rsplit(".", 1)[0] + "_curve.json"
                    try:
                        import json as _json
                        with open(cjson, "w", encoding="utf-8") as f:
                            _json.dump({"curve": [{"s": float(x), "z": float(y)} for x, y in curve_pts]}, f,
                                       ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                        
                if save_curve_json and out_png and "elev_s" in prof and prof["elev_s"] is not None:
                    gjson = out_png.rsplit(".", 1)[0] + "_ground_smoothed.json"
                    try:
                        import json as _json
                        fin = np.isfinite(prof["chain"]) & np.isfinite(prof["elev_s"])
                        ch_fin = prof["chain"][fin]
                        el_fin = prof["elev_s"][fin]
                        with open(gjson, "w", encoding="utf-8") as f:
                            _json.dump({"ground_smoothed": [{"s": float(s), "z": float(z)} for s, z in zip(ch_fin, el_fin)]}, f,
                                       ensure_ascii=False, indent=2)
                    except Exception:
                        pass
            else:
                pass

        if (highlight_theta is not None) and (float(highlight_theta) > 0):
            thr = float(highlight_theta)
            m = np.isfinite(theta) & (theta >= thr) & np.isfinite(dz)
            if np.any(m):
                ax.plot(chain[m], elev_s[m], "o", color="limegreen", ms=5,
                        label=f"θ ΓëÑ {thr:.1f}┬░ & dz>0")

        ax2.set_xlabel("Chainage (m)", fontsize=axis_label_font)
        ax.set_ylabel("Elevation (m)", fontsize=axis_label_font)
        ax2.set_ylabel("Gradient (deg)", fontsize=axis_label_font)
        try:
            ax2r.set_ylabel("Curvature (1/m)", fontsize=axis_label_font)
        except Exception:
            pass
        ax.tick_params(axis="both", labelsize=axis_tick_font)
        ax2.tick_params(axis="both", labelsize=axis_tick_font)
        try:
            ax2r.tick_params(axis="both", labelsize=axis_tick_font)
        except Exception:
            pass
        ax.grid(ls="--", lw=0.8, alpha=0.35)
        ax.margins(x=0.02)

        if overlay_curves:
            user_x_fixed = (x_user_min is not None) and (x_user_max is not None) and (x_user_max > x_user_min)
            user_y_fixed = (y_user_min is not None) and (y_user_max is not None) and (y_user_max > y_user_min)
            try:
                xs_list: List[np.ndarray] = []
                ys_list: List[np.ndarray] = []

                # profile ground
                finite_xy = np.isfinite(prof["chain"]) & np.isfinite(prof["elev_s"])
                if np.any(finite_xy):
                    xs_list.append(np.asarray(prof["chain"][finite_xy], float))
                    ys_list.append(np.asarray(prof["elev_s"][finite_xy], float))

                for item in overlay_curves:
                    if len(item) == 2:
                        ch, zz = item
                    elif len(item) == 3:
                        ch, zz, _ = item
                    else:
                        ch, zz, _, _ = item

                    ch = np.asarray(ch, float)
                    zz = np.asarray(zz, float)
                    m = np.isfinite(ch) & np.isfinite(zz)
                    if not np.any(m):
                        continue

                    xs_list.append(ch[m])
                    ys_list.append(zz[m])

                if xs_list and ys_list:
                    xs_all = np.concatenate(xs_list)
                    ys_all = np.concatenate(ys_list)
                    x0, x1 = float(xs_all.min()), float(xs_all.max())
                    y0, y1 = float(ys_all.min()), float(ys_all.max())

                    dx = (x1 - x0) or 1.0
                    dy = (y1 - y0) or 1.0
                    xpad = 0.02 * dx
                    ypad = 0.05 * dy

                    if not user_x_fixed:
                        _set_chainage_xlim(ax, x0 - xpad, x1 + xpad)
                        _set_chainage_xlim(ax2, x0 - xpad, x1 + xpad)
                    if not user_y_fixed:
                        ax.set_ylim(y0 - ypad, y1 + ypad)
            except Exception:
                pass

        user_y_fixed = (y_user_min is not None) and (y_user_max is not None) and (y_user_max > y_user_min)
        user_x_fixed = (
            (x_user_min is not None)
            and (x_user_max is not None)
            and (abs(float(x_user_max) - float(x_user_min)) > 1e-12)
        )
        if user_y_fixed:
            ax.set_ylim(float(y_user_min), float(y_user_max))

        if user_x_fixed:
            _set_chainage_xlim(ax, float(x_user_min), float(x_user_max))
            _set_chainage_xlim(ax2, float(x_user_min), float(x_user_max))
        # Force chainage major ticks every 10 m on both panels.
        try:
            from matplotlib.ticker import MultipleLocator
            ax.xaxis.set_major_locator(MultipleLocator(20.0))
            ax2.xaxis.set_major_locator(MultipleLocator(20.0))
        except Exception:
            pass
        # --- OVERLAY SLIP CURVES
        if overlay_curves:
            for item in overlay_curves:
                if len(item) == 2:
                    ch, zz = item;
                    color, label = "#bf00ff", "Slip curve"
                elif len(item) == 3:
                    ch, zz, color = item;
                    label = "Slip curve"
                else:
                    ch, zz, color, label = item

                ch = np.asarray(ch, float)
                zz = np.asarray(zz, float)
                m = np.isfinite(ch) & np.isfinite(zz)
                if not np.any(m):
                    continue

                if np.any(m):
                    ln, = ax.plot(ch[m], zz[m], color=color, lw=3.0, zorder=50, label=label)
                    try:
                        import matplotlib.patheffects as pe
                        ln.set_path_effects([pe.Stroke(linewidth=5, foreground="white"), pe.Normal()])
                    except Exception:
                        pass
        # -------------------------------------

        ax.legend(loc="best", fontsize=legend_font, frameon=False)
        os.makedirs(os.path.dirname(out_png), exist_ok=True)

        # === WRITE AXES METADATA (TOP/BOT) with tight-crop correction ===
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # get_window_extent is already in display pixels
        bbox_top = ax.get_window_extent(renderer=renderer)
        x_top, y_top, w_top, h_top = map(float, bbox_top.bounds)
        x_min_top, x_max_top = ax.get_xlim()

        bbox_bot = ax2.get_window_extent(renderer=renderer)
        x_bot, y_bot, w_bot, h_bot = map(float, bbox_bot.bounds)
        x_min_bot, x_max_bot = ax2.get_xlim()


        pad_inches = 0.15
        tight = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
        pad_px = float(pad_inches) * float(fig.dpi)
        crop_x0 = float(tight.x0) - pad_px
        crop_y0 = float(tight.y0) - pad_px

        top_left_px = x_top - crop_x0
        bot_left_px = x_bot - crop_x0
        # Convert Matplotlib display-y (origin at bottom) to saved image y (origin at top).
        crop_h = float(tight.height) + (2.0 * pad_px)
        top_top_px = crop_h - ((y_top - crop_y0) + h_top)
        bot_top_px = crop_h - ((y_bot - crop_y0) + h_bot)

        meta = {
            "top": {
                "x_min": float(x_min_top), "x_max": float(x_max_top),
                "y_min": float(min(ax.get_ylim())), "y_max": float(max(ax.get_ylim())),
                "left_px": float(top_left_px), "top_px": float(top_top_px),
                "width_px": float(w_top), "height_px": float(h_top)
            },
            "bot": {
                "x_min": float(x_min_bot), "x_max": float(x_max_bot),
                "y_min": float(min(ax2.get_ylim())), "y_max": float(max(ax2.get_ylim())),
                "left_px": float(bot_left_px), "top_px": float(bot_top_px),
                "width_px": float(w_bot), "height_px": float(h_bot)
            }
        }

        meta_path = out_png.rsplit(".", 1)[0] + ".json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                import json as _json
                _json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)

        plt.close(fig)
        return f"Saved {out_png}", out_png


# === SLIP CURVE FITTING & RENDER (place at END of file) ===

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
    dz    = dz[ok]

    if chain.size < 3 or not groups:
        return {"chain": [], "elev": [], "depth": []}

    # --- span theo group ---
    sA = float(min(g["start"] for g in groups))
    sB = float(max(g["end"]   for g in groups))
    if sB <= sA:
        return {"chain": [], "elev": [], "depth": []}

    s_new = np.arange(sA, sB + ds * 0.5, ds)
    z_g   = np.interp(s_new, chain, elevg)

    mslip = (chain >= sA) & (chain <= sB) & np.isfinite(dz)
    if np.any(mslip):
        Dz = float(np.nanpercentile(np.abs(dz[mslip]), 90))
    else:
        Dz = 0.0
    D = max(float(min_depth), Dz * float(depth_gain))

    u = (s_new - sA) / (sB - sA)
    u = np.clip(u, 0.0, 1.0)
    w = 4.0 * u * (1.0 - u)

    z_target = z_g - D * w

    if (smooth_factor > 0) and (_sg is not None) and (s_new.size >= 7):
        win = int(max(7, int(round(s_new.size * smooth_factor)) | 1))
        try:
            z_target = _sg(z_target, win, 2, mode="interp")
        except Exception:
            pass

    depth = np.maximum(z_g - z_target, 0.0)
    return {"chain": s_new, "elev": z_target, "depth": depth}


def fit_bezier_smooth_curve(chain, elevg, target_s, target_z,
                            c0=0.30, c1=0.30, clearance=0.12):

    target_s = np.asarray(target_s, float)
    target_z = np.asarray(target_z, float)

    ok = np.isfinite(target_s) & np.isfinite(target_z)
    target_s, target_z = target_s[ok], target_z[ok]
    if target_s.size < 4:
        return {"chain": target_s, "elev": target_z}

    s0, s1 = float(target_s[0]), float(target_s[-1])
    if not np.isfinite(s0) or not np.isfinite(s1) or s1 <= s0:
        return {"chain": target_s, "elev": target_z}
    L = s1 - s0

    z0 = float(np.interp(s0, chain, elevg))
    z3 = float(np.interp(s1, chain, elevg))

    P0x, P3x = s0, s1
    P1x, P2x = s0 + c0 * L, s1 - c1 * L

    # Ma trß║¡n B├⌐zier cho z1,z2
    u = (target_s - s0) / L
    u = np.clip(u, 0.0, 1.0)
    B0 = (1-u)**3; B1 = 3*(1-u)**2*u; B2 = 3*(1-u)*u**2; B3 = u**3
    A   = np.vstack([B1, B2]).T
    rhs = target_z - (B0*z0 + B3*z3)

    try:
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        z1, z2 = float(sol[0]), float(sol[1])
    except np.linalg.LinAlgError:
        zmed = float(np.nanmedian(target_z))
        z1 = z2 = zmed
        print("[fit_bezier_smooth_curve] Warning: fallback median due to SVD failure")

    uu = np.linspace(0, 1, max(50, int(L * 5)))
    C0 = (1-uu)**3; C1 = 3*(1-uu)**2*uu; C2 = 3*(1-uu)*uu**2; C3 = uu**3
    s_bez = C0*P0x + C1*P1x + C2*P2x + C3*P3x
    z_bez = C0*z0  + C1*z1  + C2*z2  + C3*z3

    zg = np.interp(s_bez, chain, elevg)
    if z_bez.size > 2:
        z_bez[1:-1] = np.minimum(z_bez[1:-1], zg[1:-1] - float(clearance))
    z_bez[0]  = zg[0]
    z_bez[-1] = zg[-1]

    return {"chain": s_bez, "elev": z_bez}

def _make_open_uniform_knot(n_ctrl: int, degree: int) -> np.ndarray:
    """Open-uniform clamped knot vector for B-spline/NURBS."""
    m = int(n_ctrl) + int(degree) + 1
    kv = np.zeros(m, dtype=float)
    kv[: degree + 1] = 0.0
    kv[-(degree + 1):] = 1.0
    n_internal = m - 2 * (degree + 1)
    if n_internal > 0:
        kv[degree + 1: degree + 1 + n_internal] = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
    return kv


def _bspline_basis_all(u: float, degree: int, knot: np.ndarray, n_ctrl: int) -> np.ndarray:
    """Compute all non-rational B-spline basis functions N_i,p(u)."""
    N = np.zeros(n_ctrl, dtype=float)
    for i in range(n_ctrl):
        if (knot[i] <= u < knot[i + 1]) or (
            u == 1.0 and knot[i] <= u <= knot[i + 1] and knot[i + 1] == 1.0
        ):
            N[i] = 1.0

    for p in range(1, degree + 1):
        Np = np.zeros(n_ctrl, dtype=float)
        for i in range(n_ctrl):
            left = 0.0
            right = 0.0
            left_den = knot[i + p] - knot[i]
            right_den = knot[i + p + 1] - knot[i + 1]
            if left_den != 0.0:
                left = (u - knot[i]) / left_den * N[i]
            if right_den != 0.0 and (i + 1) < n_ctrl:
                right = (knot[i + p + 1] - u) / right_den * N[i + 1]
            Np[i] = left + right
        N = Np
    return N


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
        N = _bspline_basis_all(float(u), degree, knot, n_ctrl)
        wN = weights * N
        denom = float(np.sum(wN))
        if denom == 0.0:
            curve[j, :] = np.nan
        else:
            curve[j, :] = (wN @ ctrl_pts) / denom
    return curve


def evaluate_nurbs_curve(
    chain_ctrl,
    elev_ctrl,
    weights=None,
    degree: int = 3,
    n_samples: int = 300,
) -> dict:
    """Evaluate a global NURBS curve from control points in chain-elevation space."""
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
    curve = _eval_nurbs_curve(
        np.vstack([ch, zz]).T,
        ww,
        degree=deg,
        n_samples=int(max(8, n_samples)),
    )
    sx = curve[:, 0]
    sz = curve[:, 1]
    fin = np.isfinite(sx) & np.isfinite(sz)
    sx = sx[fin]
    sz = sz[fin]
    if sx.size < 2:
        return {"chain": np.array([], dtype=float), "elev": np.array([], dtype=float)}
    o2 = np.argsort(sx)
    return {"chain": sx[o2], "elev": sz[o2]}


from pedi_oku_landslide.pipeline.runners.ui3.ui3_curve_fit import (
    estimate_slip_curve as _estimate_slip_curve_impl,
    evaluate_nurbs_curve as _evaluate_nurbs_curve_impl,
    fit_bezier_smooth_curve as _fit_bezier_smooth_curve_impl,
)
from pedi_oku_landslide.pipeline.runners.ui3.ui3_grouping import (
    WORKFLOW_GROUP_MIN_LEN_M,
    _renumber_groups_visual_order as _normalize_groups_impl,
    auto_group_profile_by_criteria as _auto_group_profile_by_criteria_impl,
    clamp_groups_to_slip as _clamp_groups_to_slip_impl,
)
from pedi_oku_landslide.pipeline.runners.ui3.ui3_paths import auto_paths as _auto_paths_impl
from pedi_oku_landslide.pipeline.runners.ui3.ui3_profile_math import compute_profile as _compute_profile_impl
from pedi_oku_landslide.pipeline.runners.ui3.ui3_storage import (
    build_gdf_from_sections_csv,
    ensure_sections_csv_current,
    load_json,
    save_json,
)


def auto_paths() -> dict:
    return _auto_paths_impl()


def compute_profile(
    dem_path: str,
    dx_path: str,
    dy_path: str,
    dz_path: str,
    line_geom: LineString,
    step_m: float = 0.2,
    smooth_win: int = 11,
    smooth_poly: int = 2,
    slip_mask_path: Optional[str] = None,
    slip_only: bool = True,
    dem_orig_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    return _compute_profile_impl(
        dem_path,
        dx_path,
        dy_path,
        dz_path,
        line_geom,
        step_m=step_m,
        smooth_win=smooth_win,
        smooth_poly=smooth_poly,
        slip_mask_path=slip_mask_path,
        slip_only=slip_only,
        dem_orig_path=dem_orig_path,
    )


def estimate_slip_curve(
    prof: dict,
    groups: list,
    ds: float = 0.2,
    smooth_factor: float = 0.1,
    depth_gain: float = 3.0,
    min_depth: float = 1.0,
) -> dict:
    return _estimate_slip_curve_impl(
        prof,
        groups,
        ds=ds,
        smooth_factor=smooth_factor,
        depth_gain=depth_gain,
        min_depth=min_depth,
    )


def fit_bezier_smooth_curve(chain, elevg, target_s, target_z, c0=0.30, c1=0.30, clearance=0.12):
    return _fit_bezier_smooth_curve_impl(
        chain,
        elevg,
        target_s,
        target_z,
        c0=c0,
        c1=c1,
        clearance=clearance,
    )


def evaluate_nurbs_curve(chain_ctrl, elev_ctrl, weights=None, degree: int = 3, n_samples: int = 300) -> dict:
    return _evaluate_nurbs_curve_impl(
        chain_ctrl=chain_ctrl,
        elev_ctrl=elev_ctrl,
        weights=weights,
        degree=degree,
        n_samples=n_samples,
    )


def auto_group_profile_by_criteria(
    prof: Dict[str, Any],
    rdp_eps_m: float = 0.5,
    curvature_thr_abs: float = 0.02,
    smooth_radius_m: float = 0.0,
    include_curvature_threshold: bool = True,
    include_vector_angle_zero: bool = True,
) -> List[Dict[str, Any]]:
    return _auto_group_profile_by_criteria_impl(
        prof,
        rdp_eps_m=rdp_eps_m,
        curvature_thr_abs=curvature_thr_abs,
        smooth_radius_m=smooth_radius_m,
        include_curvature_threshold=include_curvature_threshold,
        include_vector_angle_zero=include_vector_angle_zero,
    )


def clamp_groups_to_slip(prof: Dict[str, Any], groups: List[Dict[str, Any]], min_len: float = WORKFLOW_GROUP_MIN_LEN_M) -> List[Dict[str, Any]]:
    return _clamp_groups_to_slip_impl(prof, groups, min_len=min_len)


class UI3BackendService:
    def __init__(self, *, base_dir: str = "") -> None:
        self._ctx: Dict[str, str] = {"project": "", "run_label": "", "run_dir": "", "base_dir": base_dir}
        self._inputs: Dict[str, Any] = {}

    def set_context(self, project: str, run_label: str, run_dir: str, base_dir: str = "") -> Dict[str, Any]:
        if base_dir:
            self._ctx["base_dir"] = str(base_dir)
        self._ctx.update({"project": project, "run_label": run_label, "run_dir": run_dir})
        self._inputs = self.load_inputs()
        return dict(self._inputs)

    def load_inputs(self) -> Dict[str, Any]:
        run_dir = str(self._ctx.get("run_dir", "") or "")
        base_dir = str(self._ctx.get("base_dir", "") or "")
        shared_jsons = [
            os.path.join(run_dir, "ui_shared_data.json"),
            os.path.join(base_dir, "output", "ui_shared_data.json"),
            os.path.join(base_dir, "output", "UI1", "ui_shared_data.json"),
        ]
        js: Dict[str, Any] = {}
        for path in shared_jsons:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        js.update(json.load(f))
                except Exception:
                    pass

        meta_inputs: Dict[str, Any] = {}
        meta_processed: Dict[str, Any] = {}
        meta_path = os.path.join(run_dir, "ingest_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}
                meta_inputs = meta.get("inputs") or {}
                meta_processed = meta.get("processed") or {}
            except Exception:
                pass

        ap = auto_paths()

        def _pick_first(*cands: str) -> str:
            for path in cands:
                if path and os.path.exists(path):
                    return path
            return ""

        dem_path_smooth = _pick_first(
            os.path.join(run_dir, "ui1", "after_dem_smooth.tif"),
            meta_inputs.get("after_dem") or "",
            meta_inputs.get("before_dem") or "",
            js.get("dem_ground_path") or "",
            ap.get("dem", ""),
            meta_inputs.get("before_asc") or "",
            os.path.join(run_dir, "input", "after_dem.tif"),
            os.path.join(run_dir, "input", "before_dem.tif"),
            os.path.join(run_dir, "input", "before.asc"),
            meta_processed.get("dem_cropped") or "",
        )
        dem_path_raw = _pick_first(
            meta_inputs.get("after_dem") or "",
            meta_inputs.get("before_dem") or "",
            os.path.join(run_dir, "input", "after_dem.tif"),
            os.path.join(run_dir, "input", "before_dem.tif"),
            js.get("dem_ground_path") or "",
            ap.get("dem_orig", ""),
            ap.get("dem", ""),
        )
        slip_path = js.get("slip_path") or ap.get("slip", "")
        if meta_processed.get("slip_mask"):
            slip_path = meta_processed.get("slip_mask")
        if not slip_path:
            slip_path = os.path.join(run_dir, "ui1", "landslide_mask.tif")
        if slip_path and (not os.path.exists(slip_path)):
            alt = slip_path.replace(".asc", ".tif")
            if os.path.exists(alt):
                slip_path = alt

        return {
            "shared_json": js,
            "meta_inputs": meta_inputs,
            "meta_processed": meta_processed,
            "auto_paths": ap,
            "dem_path_smooth": dem_path_smooth,
            "dem_path_raw": dem_path_raw,
            "dx_path": _pick_first(js.get("dx_path") or "", ap.get("dx", ""), os.path.join(run_dir, "ui1", "dx.tif")),
            "dy_path": _pick_first(js.get("dy_path") or "", ap.get("dy", ""), os.path.join(run_dir, "ui1", "dy.tif")),
            "dz_path": _pick_first(js.get("dz_path") or "", ap.get("dz", ""), os.path.join(run_dir, "ui1", "dz.tif")),
            "lines_path": "",
            "slip_path": slip_path,
        }

    def load_lines(self, csv_path: str, dem_path: str) -> Dict[str, Any]:
        migrated = ensure_sections_csv_current(csv_path, run_dir=str(self._ctx.get("run_dir", "") or ""))
        gdf = build_gdf_from_sections_csv(csv_path, dem_path)
        return {"migrated": bool(migrated), "gdf": gdf}

    def compute_profile_for_line(
        self,
        line_id: str,
        geom: LineString,
        profile_source: str,
        step_m: float,
        slip_only: bool,
        *,
        dem_path: str = "",
        dem_orig_path: str = "",
        dx_path: str = "",
        dy_path: str = "",
        dz_path: str = "",
        slip_mask_path: str = "",
    ) -> Optional[Dict[str, Any]]:
        inputs = self._inputs or self.load_inputs()
        dem = dem_path or inputs.get("dem_path_smooth") or inputs.get("dem_path_raw") or ""
        if profile_source == "raw":
            dem = dem_path or inputs.get("dem_path_raw") or dem
        dem_orig = dem_orig_path or inputs.get("dem_path_raw") or dem
        if not dem:
            return None
        prof = compute_profile(
            dem,
            dx_path or inputs.get("dx_path", ""),
            dy_path or inputs.get("dy_path", ""),
            dz_path or inputs.get("dz_path", ""),
            geom,
            step_m=step_m,
            smooth_win=11,
            smooth_poly=2,
            slip_mask_path=slip_mask_path or inputs.get("slip_path", ""),
            slip_only=bool(slip_only),
            dem_orig_path=dem_orig,
        )
        if not prof:
            return None
        prof["profile_dem_source"] = profile_source
        prof["profile_dem_path"] = dem
        prof["line_id"] = line_id
        return prof

    def load_groups(self, path: str) -> Any:
        return load_json(path, default=None)

    def save_groups(self, path: str, groups: Any) -> str:
        return save_json(path, groups)

    def auto_group(
        self,
        line_id: str,
        geom: Optional[LineString],
        grouping_settings: Dict[str, Any],
        profile_source: str,
        step_m: float,
        *,
        prof: Optional[Dict[str, Any]] = None,
        min_len: float = WORKFLOW_GROUP_MIN_LEN_M,
    ) -> Dict[str, Any]:
        if prof is None and geom is not None:
            prof = self.compute_profile_for_line(line_id, geom, profile_source, step_m, slip_only=False)
        if not prof:
            return {"profile": None, "groups": []}
        groups = auto_group_profile_by_criteria(prof, **dict(grouping_settings or {}))
        groups = clamp_groups_to_slip(prof, groups, min_len=min_len)
        return {"profile": prof, "groups": groups}

    def auto_group_profile(self, profile: Dict[str, Any], grouping_settings: Dict[str, Any], min_len: float = WORKFLOW_GROUP_MIN_LEN_M) -> List[Dict[str, Any]]:
        groups = auto_group_profile_by_criteria(profile, **dict(grouping_settings or {}))
        return clamp_groups_to_slip(profile, groups, min_len=min_len)

    def clamp_groups(self, profile: Dict[str, Any], groups: List[Dict[str, Any]], min_len: float = WORKFLOW_GROUP_MIN_LEN_M) -> List[Dict[str, Any]]:
        return clamp_groups_to_slip(profile, groups, min_len=min_len)

    def normalize_groups(self, groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return _normalize_groups_impl(groups)

    def build_curve_seed(self, profile: Dict[str, Any], groups: List[Dict[str, Any]], curve_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        curve_settings = dict(curve_settings or {})
        return estimate_slip_curve(
            profile,
            groups,
            ds=float(curve_settings.get("ds", 0.2)),
            smooth_factor=float(curve_settings.get("smooth_factor", 0.1)),
            depth_gain=float(curve_settings.get("depth_gain", 3.0)),
            min_depth=float(curve_settings.get("min_depth", 1.0)),
        )

    def fit_bezier_curve_seed(self, chain, elevg, target_s, target_z, curve_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        curve_settings = dict(curve_settings or {})
        return fit_bezier_smooth_curve(
            chain,
            elevg,
            target_s,
            target_z,
            c0=float(curve_settings.get("c0", 0.30)),
            c1=float(curve_settings.get("c1", 0.30)),
            clearance=float(curve_settings.get("clearance", 0.12)),
        )

    def evaluate_nurbs(self, params_or_ctrl_points, elev_ctrl=None, weights=None, degree: int = 3, n_samples: int = 300) -> Dict[str, Any]:
        if isinstance(params_or_ctrl_points, dict):
            params = dict(params_or_ctrl_points)
            cps = np.asarray(params.get("control_points", []), dtype=float)
            if cps.ndim != 2 or cps.shape[0] < 2:
                return {"chain": np.array([], dtype=float), "elev": np.array([], dtype=float)}
            return evaluate_nurbs_curve(
                chain_ctrl=cps[:, 0],
                elev_ctrl=cps[:, 1],
                weights=params.get("weights", weights),
                degree=int(params.get("degree", degree)),
                n_samples=int(params.get("n_samples", n_samples)),
            )
        return evaluate_nurbs_curve(
            chain_ctrl=params_or_ctrl_points,
            elev_ctrl=elev_ctrl,
            weights=weights,
            degree=degree,
            n_samples=n_samples,
        )

    def render_preview(self, profile: Dict[str, Any], render_settings: Dict[str, Any], groups=None, overlay_curves=None) -> Dict[str, Any]:
        render_settings = dict(render_settings or {})
        msg, path = render_profile_png(
            profile,
            render_settings["out_png"],
            y_min=render_settings.get("y_min"),
            y_max=render_settings.get("y_max"),
            x_min=render_settings.get("x_min"),
            x_max=render_settings.get("x_max"),
            vec_scale=float(render_settings.get("vec_scale", 0.1)),
            vec_width=float(render_settings.get("vec_width", 0.0015)),
            head_len=float(render_settings.get("head_len", 7.0)),
            head_w=float(render_settings.get("head_w", 5.0)),
            highlight_theta=render_settings.get("highlight_theta"),
            group_ranges=groups if groups else None,
            draw_curve=bool(render_settings.get("draw_curve", False)),
            save_curve_json=bool(render_settings.get("save_curve_json", False)),
            overlay_curves=overlay_curves,
            figsize=tuple(render_settings.get("figsize", (18, 10))),
            dpi=int(render_settings.get("dpi", 220)),
            base_font=int(render_settings.get("base_font", 20)),
            label_font=int(render_settings.get("label_font", 20)),
            tick_font=int(render_settings.get("tick_font", 20)),
            legend_font=int(render_settings.get("legend_font", 20)),
            ground_lw=float(render_settings.get("ground_lw", 2.2)),
            ungrouped_color=str(render_settings.get("ungrouped_color", "#bbbbbb")),
            curvature_series=render_settings.get("curvature_series"),
            curvature_rdp_eps_m=float(render_settings.get("curvature_rdp_eps_m", 0.5)),
            curvature_smooth_radius_m=float(render_settings.get("curvature_smooth_radius_m", 0.0)),
        )
        return {"message": msg, "path": path}

    def extract_curvature_nodes(self, profile: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return extract_curvature_rdp_nodes(profile, **kwargs)

    def save_nurbs_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        saved: Dict[str, Any] = {}
        for key, spec in dict(outputs or {}).items():
            if not isinstance(spec, dict):
                continue
            path = str(spec.get("path", "") or "")
            if not path:
                continue
            saved[key] = save_json(path, spec.get("payload"))
        return saved

    def load_saved_ui3_state(self, paths: Dict[str, str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, path in dict(paths or {}).items():
            out[key] = load_json(path, default=None)
        return out

    def sync_anchor_updates(self, line_id: str, curve_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"line_id": line_id, "curve": curve_data, "updated": 0}

    def export_vectors_and_ground(self, line_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        chain = np.asarray(profile.get("chain", []), dtype=float)
        elev = np.asarray(profile.get("elev_s", []), dtype=float)
        d_para = np.asarray(profile.get("d_para", []), dtype=float)
        dz = np.asarray(profile.get("dz", []), dtype=float)
        vectors = []
        for s, z, dp, dzv in zip(chain.tolist(), elev.tolist(), d_para.tolist(), dz.tolist()):
            if np.isfinite(s) and np.isfinite(z) and np.isfinite(dp) and np.isfinite(dzv):
                vectors.append({"s": float(s), "z": float(z), "d_para": float(dp), "dz": float(dzv)})
        ground = []
        for s, z in zip(chain.tolist(), elev.tolist()):
            if np.isfinite(s) and np.isfinite(z):
                ground.append({"s": float(s), "z": float(z)})
        return {"line_id": line_id, "vectors": vectors, "ground": ground}


