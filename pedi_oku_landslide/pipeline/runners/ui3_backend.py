# ui3_backend.py
import json
import os
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import LineString
try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None
import matplotlib
matplotlib.use("Agg")  # headless
import pandas as pd
from pedi_oku_landslide.core.paths import OUTPUT_ROOT

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
        _out("UI1", "step1_crop", "before_ground.asc"),
        _out("UI1", "step1_crop", "before_ground.tif"),
        js.get("dem_ground_path", ""),
        _out("UI1", "step1_crop", "before_crop.asc"),
    ])

    dx = pick_first_exists([
        _out("UI1", "step2_sad", "dX.asc"),
        js.get("dx_path", ""),
    ])

    dy = pick_first_exists([
        _out("UI1", "step2_sad", "dY.asc"),
        js.get("dy_path", ""),
    ])

    dz = pick_first_exists([
        _out("UI1", "step7_slipzone", "dZ_slipzone.asc"),
        _out("UI1", "step5_dz", "dZ.asc"),
        js.get("dz_path", ""),
    ])

    lines = pick_first_exists([
        _out("UI2", "step2_selected_lines", "selected_lines.gpkg"),
        js.get("lines_path", ""),
    ])

    slip = pick_first_exists([
        _out("UI1", "step7_slipzone", "slip_zone.asc"),
        js.get("slip_path", ""),
    ])

    return {"dem": dem, "dx": dx, "dy": dy, "dz": dz, "lines": lines, "slip": slip}


# IO/GEOM HELPERS
def _open_raster(path: str):
    ds = rasterio.open(path)
    arr = ds.read(1).astype("float32")
    return ds, arr

from shapely.geometry import LineString

def _ensure_lines_crs(lines_path: str, dst_crs) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(lines_path)

    if gdf.crs is None:
        try:
            gdf = gdf.set_crs(dst_crs)
        except Exception:
            pass
    elif str(gdf.crs) != str(dst_crs):
        try:
            gdf = gdf.to_crs(dst_crs)
        except Exception:
            pass

    return gdf


def list_lines(lines_path: str, dem_path: str) -> Tuple[List[str], gpd.GeoDataFrame, Dict[str, Any]]:
    dem_ds, _ = _open_raster(dem_path)
    dem_crs = dem_ds.crs
    dem_ds.close()

    if dem_crs is None:
        from rasterio.crs import CRS
        dem_crs = CRS.from_epsg(6677)

    gdf = _ensure_lines_crs(lines_path, dem_crs)

    labels: List[str] = []
    clean_geoms: List[LineString] = []
    keep_idx: List[int] = []

    for i, geom in enumerate(gdf.geometry):
        base = None
        if "name" in gdf.columns and isinstance(gdf.at[i, "name"], str):
            base = gdf.at[i, "name"]
        if not base:
            base = f"Line {i+1}"

        if geom is None or getattr(geom, "is_empty", False):
            labels.append(base)
            clean_geoms.append(geom)
            keep_idx.append(i)
            continue

        try:
            coords = np.asarray(geom.coords, dtype="float64")
            if not np.all(np.isfinite(coords)):
                labels.append(base)
                clean_geoms.append(geom)
                keep_idx.append(i)
                continue
        except Exception:
            labels.append(base)
            clean_geoms.append(geom)
            keep_idx.append(i)
            continue
        try:
            L = float(geom.length)
        except Exception:
            L = float("nan")

        if not np.isfinite(L) or L <= 0:
            labels.append(base)
        else:
            labels.append(f"{base}  ({L:.1f} m)")

        clean_geoms.append(geom)
        keep_idx.append(i)

    if keep_idx:
        gdf = gdf.iloc[keep_idx].copy()
        gdf.reset_index(drop=True, inplace=True)
        gdf = gdf.set_geometry(clean_geoms)
    else:
        gdf = gdf.iloc[0:0].copy()

    gdf["__label__"] = labels
    return labels, gdf, {"crs": dem_crs}


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
import math

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

    # smooth ground
    if xs.size >= smooth_win and smooth_win % 2 == 1:
        try:
            elev_s = savgol_filter(elev, smooth_win, smooth_poly, mode="interp")
        except Exception:
            elev_s = elev.copy()
    else:
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
    }

def clamp_groups_to_slip(prof: dict, groups: list, min_len: float = 0.20) -> list:
    chain = np.asarray(prof.get("chain"), float)
    elevs = np.asarray(prof.get("elev_s"), float)

    if "slip_span" in prof and prof["slip_span"]:
        smin, smax = prof["slip_span"]
    else:
        keep = np.isfinite(chain) & np.isfinite(elevs)
        if not np.any(keep):
            return []
        smin, smax = float(np.nanmin(chain[keep])), float(np.nanmax(chain[keep]))

    out = []
    for g in (groups or []):
        s = float(g.get("start", g.get("start_chainage", 0.0)))
        e = float(g.get("end",   g.get("end_chainage",   0.0)))
        if e < s: s, e = e, s
        # clamp
        s = max(s, smin)
        e = min(e, smax)
        if (e - s) >= min_len:
            out.append({
                "id": str(g.get("id", f"G{len(out)+1}")),
                "start": s, "end": e,
                "color": g.get("color", None)
            })
    return out

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

    def _dump_ground_json(out_png_path: str, prof_dict: Dict[str, np.ndarray]) -> None:
        try:
            xs = np.asarray(prof_dict.get("x", []), dtype=float)
            ys = np.asarray(prof_dict.get("y", []), dtype=float)
            z_raw = np.asarray(prof_dict.get("elev", []), dtype=float)
            z_smooth = np.asarray(prof_dict.get("elev_s", []), dtype=float)
            n = min(xs.size, ys.size, z_raw.size, z_smooth.size)
            if n == 0:
                return

            ui3_dir = os.path.dirname(os.path.dirname(out_png_path))
            base = os.path.splitext(os.path.basename(out_png_path))[0]
            out_json = os.path.join(ui3_dir, f"{base}_ground.json")

            rows = []
            for i in range(n):
                xr = float(xs[i]) if np.isfinite(xs[i]) else None
                yr = float(ys[i]) if np.isfinite(ys[i]) else None
                zr = float(z_raw[i]) if np.isfinite(z_raw[i]) else None
                zs = float(z_smooth[i]) if np.isfinite(z_smooth[i]) else None
                rows.append({
                    "index": i,
                    "x": xr,
                    "y": yr,
                    "z_raw": zr,
                    "z_smooth": zs,
                })

            payload = {
                "count": n,
                "data": rows,
            }
            os.makedirs(ui3_dir, exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    x_user_min, x_user_max = x_min, x_max
    y_user_min, y_user_max = y_min, y_max

    chain = prof["chain"]; elev_s = prof["elev_s"]
    d_para = prof["d_para"]; dz = prof["dz"]; theta = prof["theta"]

    if out_png:
        _dump_ground_json(out_png, prof)

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

    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size': base_font}):
        # 2 h├áng: [vectors ; theta-rate]
        fig = plt.figure(figsize=figsize if figsize else (18, 12), dpi=dpi)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.35)
        ax = fig.add_subplot(gs[0, 0])  # vectors/profile
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax)  # theta-rate

        ax.plot(chain, elev_s, "k-", lw=ground_lw, label="Ground (smoothed)")


        finite = np.isfinite(chain) & np.isfinite(elev_s) & np.isfinite(d_para) & np.isfinite(dz)
        if group_ranges:
            if "slip_span" in prof and prof["slip_span"]:
                smin, smax = prof["slip_span"]
            else:
                finite = np.isfinite(prof["chain"]) & np.isfinite(prof["elev_s"])
                smin = float(np.nanmin(prof["chain"][finite]))
                smax = float(np.nanmax(prof["chain"][finite]))

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
                gidx[m] = gi

            for gi, gid, s, e, color in prepared:
                m = finite & (gidx == gi)
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

            finite = np.isfinite(chain_s) & np.isfinite(d_para_s) & np.isfinite(dz_s)
            if finite.sum() >= 2:
                ch = chain_s[finite]
                theta_unw = np.unwrap(np.arctan2(dz_s[finite], d_para_s[finite]))  # rad
                dtheta_ds = np.gradient(theta_unw, ch)  # rad/m
                theta_rate = np.degrees(dtheta_ds)  # deg/m

                if savgol_filter is not None and theta_rate.size >= 9:
                    try:
                        theta_rate = savgol_filter(theta_rate, 9, 2, mode="interp")
                    except Exception:
                        pass

                ax2.plot(ch, theta_rate, lw=2.2, color="#2ca02c", zorder=5, label="θ-rate")

            ax2.axhline(0.0, color="0.5", lw=1.0, zorder=1)
            ax2.set_ylabel("θ rate (deg/m)", fontsize=axis_label_font)
            ax2r = ax2.twinx()
            ax2r.set_ylabel("Curvature (1/m)", fontsize=axis_label_font)

            # Curvature from RDP points on the profile (signed k = 1/R).
            k_curve = None
            try:
                elev_s_s = elev_s[order]
                finite_curve = np.isfinite(chain_s) & np.isfinite(elev_s_s)
                if finite_curve.sum() >= 3:
                    pts = list(zip(chain_s[finite_curve].tolist(),
                                   elev_s_s[finite_curve].tolist()))
                    rdp_pts = _rdp_polyline(pts, 0.5)
                    if len(rdp_pts) >= 3:
                        k_vals = _curvature_points_from_rdp(rdp_pts)
                        k_curve = np.asarray(k_vals, dtype=float)
                        k_x = [p[0] for p in rdp_pts]
                        ax2r.plot(k_x, k_vals, lw=1.8, color="#1f77b4",
                                  marker="o", markersize=4, zorder=6, label="Curvature")
            except Exception:
                pass
            if k_curve is not None and np.any(np.isfinite(k_curve)):
                qk = np.nanpercentile(np.abs(k_curve), 98)
                if np.isfinite(qk) and qk > 0:
                    ax2r.set_ylim(-1.2 * qk, 1.2 * qk)

            ax2r.grid(False)
            ax2.grid(ls="--", lw=0.8, alpha=0.35)

            try:
                if finite.sum() >= 2:
                    q = np.nanpercentile(np.abs(theta_rate), 98)
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

            m = finite & (gidx == -1)
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
            if finite_th.sum() >= 2:
                chain_f = chain[finite_th]
                theta_unw = np.unwrap(np.arctan2(dz[finite_th], d_para[finite_th]))  # rad
                dtheta_ds = np.gradient(theta_unw, chain_f)  # rad/m
                theta_rate = np.degrees(dtheta_ds)  # deg/m
                if savgol_filter is not None and theta_rate.size >= 9:
                    try:
                        theta_rate = savgol_filter(theta_rate, 9, 2, mode="interp")
                    except Exception:
                        pass
                ax2.plot(chain_f, theta_rate, color="#2ca02c", lw=2.4, zorder=5, label="θ-rate")

            ax2.axhline(0.0, color="0.5", lw=1.0, zorder=1)
            ax2.set_xlabel("Chainage (m)")
            ax2.set_ylabel("θ rate (deg/m)")
            ax2r = ax2.twinx()
            ax2r.set_ylabel("Curvature (1/m)")

            # Curvature from RDP points on the profile (signed k = 1/R).
            k_curve = None
            try:
                finite_curve = np.isfinite(chain) & np.isfinite(elev_s)
                if finite_curve.sum() >= 3:
                    pts = list(zip(chain[finite_curve].tolist(),
                                   elev_s[finite_curve].tolist()))
                    rdp_pts = _rdp_polyline(pts, 0.5)
                    if len(rdp_pts) >= 3:
                        k_vals = _curvature_points_from_rdp(rdp_pts)
                        k_curve = np.asarray(k_vals, dtype=float)
                        k_x = [p[0] for p in rdp_pts]
                        ax2r.plot(k_x, k_vals, color="#1f77b4", lw=2.0,
                                  marker="o", markersize=4, zorder=6, label="Curvature")
            except Exception:
                pass
            if k_curve is not None and np.any(np.isfinite(k_curve)):
                qk = np.nanpercentile(np.abs(k_curve), 98)
                if np.isfinite(qk) and qk > 0:
                    ax2r.set_ylim(-1.2 * qk, 1.2 * qk)

            ax2r.grid(False)
            ax2.grid(True, linestyle="--", alpha=0.4)

            try:
                if finite_th.sum() >= 2:
                    q = np.nanpercentile(np.abs(theta_rate), 98)
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

            if x_min is not None and x_max is not None and (x_max > x_min):
                ax2.set_xlim(x_min, x_max)

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

            bounds_set = set()
            for g in group_ranges:
                try:
                    s = float(g.get("start", g.get("start_chainage", 0.0)))
                    e = float(g.get("end", g.get("end_chainage", 0.0)))
                except Exception:
                    continue
                if e < s:
                    s, e = e, s
                if (xmin is not None) and (xmax is not None):
                    s = max(xmin, min(xmax, s))
                    e = max(xmin, min(xmax, e))
                bounds_set.add(s);
                bounds_set.add(e)

            bounds = sorted(bounds_set)
            if bounds:
                vkw = dict(color="#555555", linestyle=(0, (4, 4)), linewidth=0.9, zorder=10)
                for x in bounds:
                    ax.axvline(x, **vkw)
                    ax2.axvline(x, **vkw)
                # Number group intervals between adjacent dashed boundaries.
                try:
                    import matplotlib.transforms as mtransforms
                    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
                    label_fs = max(8, int(round(tick_font * 0.9)))
                    idx = 1
                    for x0, x1 in zip(bounds[:-1], bounds[1:]):
                        if not (np.isfinite(x0) and np.isfinite(x1)):
                            continue
                        if x1 <= x0:
                            continue
                        xm = 0.5 * (x0 + x1)
                        ax.text(
                            float(xm), 0.995, str(idx),
                            transform=trans,
                            ha="center", va="top",
                            fontsize=label_fs,
                            color="#333333",
                            zorder=60,
                        )
                        idx += 1
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
        ax2.set_ylabel("θ rate (deg/m)", fontsize=axis_label_font)
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
                        ax.set_xlim(x0 - xpad, x1 + xpad)
                        ax2.set_xlim(x0 - xpad, x1 + xpad)
                    if not user_y_fixed:
                        ax.set_ylim(y0 - ypad, y1 + ypad)
            except Exception:
                pass

        user_y_fixed = (y_user_min is not None) and (y_user_max is not None) and (y_user_max > y_user_min)
        user_x_fixed = (x_user_min is not None) and (x_user_max is not None) and (x_user_max > x_user_min)
        if user_y_fixed:
            ax.set_ylim(float(y_user_min), float(y_user_max))

        if user_x_fixed:
            ax.set_xlim(float(x_user_min), float(x_user_max))
            ax2.set_xlim(float(x_user_min), float(x_user_max))
        # Force chainage major ticks every 10 m on both panels.
        try:
            from matplotlib.ticker import MultipleLocator
            ax.xaxis.set_major_locator(MultipleLocator(10.0))
            ax2.xaxis.set_major_locator(MultipleLocator(10.0))
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


def export_csv(prof: Dict[str, np.ndarray], out_csv: Optional[str]) -> Tuple[str, Optional[str]]:
    if not prof:
        return "Empty profile", None

    if not out_csv:
        from datetime import datetime
        out_csv = os.path.join(UI3_EXPORTS, f"profile_{datetime.now():%Y%m%d_%H%M%S}.csv")

    df = pd.DataFrame({
        "chain_m":  prof["chain"],
        "elev_s_m": prof["elev_s"],
        "d_para_m": prof["d_para"],
        "dz_m":     prof["dz"],
        "theta_deg":prof["theta"],
    })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.6f")
    return f"[Γ£ô] Saved {out_csv}", out_csv


# === [UI3] Auto-group helpers (RDP + Curvature + θ-rate) ===
import matplotlib.colors as mcolors

def _rdp_indices(points: np.ndarray, eps: float) -> np.ndarray:
    if points.shape[0] <= 2:
        return np.arange(points.shape[0])
    a, b = points[0], points[-1]
    ab = b - a
    ab2 = float((ab**2).sum())
    if ab2 == 0:
        dists = np.hypot(*(points - a).T)
    else:
        t = np.clip(((points - a) @ ab) / ab2, 0, 1)
        proj = a + np.outer(t, ab)
        dists = np.hypot(*(points - proj).T)
    idx = int(np.argmax(dists))
    dmax = float(dists[idx])
    if dmax > eps:
        left  = _rdp_indices(points[:idx+1], eps)
        right = _rdp_indices(points[idx:],   eps)
        return np.concatenate([left[:-1], right + idx])
    else:
        return np.array([0, points.shape[0]-1])

def _curvature_3pt(chain: np.ndarray, elev: np.ndarray) -> np.ndarray:
    n = chain.size
    K = np.full(n, np.nan, dtype=float)
    def tri_curv(i0, i1, i2):
        x1, y1 = chain[i0], elev[i0]
        x2, y2 = chain[i1], elev[i1]
        x3, y3 = chain[i2], elev[i2]
        a = np.hypot(x2-x1, y2-y1)
        b = np.hypot(x3-x2, y3-y2)
        c = np.hypot(x3-x1, y3-y1)
        if a*b*c == 0:
            return 0.0
        S = 0.5 * abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
        sign = np.sign((x2-x1)*(y3-y2) - (y2-y1)*(x3-x2))
        return float(sign * (4.0*S)/(a*b*c))
    if n >= 3:
        K[1:-1] = [tri_curv(i-1, i, i+1) for i in range(1, n-1)]
        K[0] = K[1]; K[-1] = K[-2]
    return K

def _theta_rate_deg_per_m(chain: np.ndarray, d_para: np.ndarray, dz: np.ndarray) -> np.ndarray:
    mask = np.isfinite(chain) & np.isfinite(d_para) & np.isfinite(dz)
    out  = np.full(chain.shape, np.nan)
    if mask.sum() < 3:
        return out
    ch = chain[mask]
    th = np.unwrap(np.arctan2(dz[mask], d_para[mask]))   # rad
    dth_ds = np.gradient(th, ch)                          # rad/m
    out[mask] = np.degrees(dth_ds)                        # deg/m
    return out

def rdp_indices_from_profile(prof: dict, rdp_eps_m: float = 0.5) -> List[int]:
    """
    Return indices kept by the numpy-based RDP (_rdp) on the smoothed profile.
    Indices are relative to the filtered (finite) arrays used by auto_group_profile.
    """
    chain = np.asarray(prof.get("chain"))
    elev  = np.asarray(prof.get("elev_s"))
    if chain.ndim != 1 or elev.ndim != 1:
        return []
    finite = np.isfinite(chain) & np.isfinite(elev)
    if finite.sum() < 2:
        return []
    chain_f = chain[finite]
    elev_f  = elev[finite]
    pts = np.vstack([chain_f, elev_f]).T
    keep_idx = _rdp_indices(pts, rdp_eps_m)
    keep_idx.sort()
    return keep_idx.tolist()


def rdp_points_from_profile(prof: dict, rdp_eps_m: float = 0.5) -> List[List[float]]:
    """
    Return simplified points [(chain, elev), ...] from the polyline RDP on smoothed profile.
    """
    chain = np.asarray(prof.get("chain"))
    elev  = np.asarray(prof.get("elev_s"))
    if chain.ndim != 1 or elev.ndim != 1:
        return []
    finite = np.isfinite(chain) & np.isfinite(elev)
    if finite.sum() < 2:
        return []
    chain_f = chain[finite]
    elev_f  = elev[finite]
    pts = list(zip(map(float, chain_f), map(float, elev_f)))
    simp = _rdp_polyline(pts, rdp_eps_m)
    return [[float(x), float(y)] for x, y in simp]

def _cluster_chain_marks(chain_marks: np.ndarray, gap_m: float = 0.5) -> List[float]:
    vals = np.asarray(chain_marks, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return []
    vals.sort()
    clusters: List[List[float]] = [[float(vals[0])]]
    for v in vals[1:]:
        if (float(v) - clusters[-1][-1]) <= gap_m:
            clusters[-1].append(float(v))
        else:
            clusters.append([float(v)])
    return [float(np.mean(c)) for c in clusters if c]

def _robust_zscore_abs(vals: np.ndarray) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    out = np.zeros(arr.shape, dtype=float)
    m = np.isfinite(arr)
    if int(np.count_nonzero(m)) < 3:
        return out
    v = np.abs(arr[m])
    med = float(np.nanmedian(v))
    mad = float(np.nanmedian(np.abs(v - med)))
    if mad > 1e-9:
        scale = 1.4826 * mad
    else:
        q75, q25 = np.nanpercentile(v, [75, 25])
        iqr = float(q75 - q25)
        if iqr > 1e-9:
            scale = iqr / 1.349
        else:
            scale = float(np.nanstd(v))
    if (not np.isfinite(scale)) or scale <= 1e-9:
        return out
    out[m] = (v - med) / scale
    return out

def _smooth_series(vals: np.ndarray, win: int, poly: int = 2) -> np.ndarray:
    arr = np.asarray(vals, dtype=float).copy()
    n = int(arr.size)
    if n < 3:
        return arr
    m = np.isfinite(arr)
    if int(np.count_nonzero(m)) < 2:
        return arr
    if not np.all(m):
        idx = np.flatnonzero(m)
        if idx.size == 1:
            arr[:] = arr[idx[0]]
        else:
            arr[~m] = np.interp(np.flatnonzero(~m), idx, arr[idx])

    w = int(max(5, win))
    if (w % 2) == 0:
        w += 1
    if w > n:
        w = n if (n % 2 == 1) else (n - 1)
    if w < 3 or savgol_filter is None:
        return arr

    p = int(max(1, min(poly, w - 1)))
    try:
        return savgol_filter(arr, w, p, mode="interp")
    except Exception:
        return arr

def auto_group_profile_adaptive_hybrid(
    prof: dict,
    min_len_m: float = 2.0,
    min_boundary_gap_m: float = 1.5,
    score_quantile: float = 85.0,
    merge_angle_tol_deg: float = 15.0,
    smooth_span_m: float = 2.0,
) -> list:
    """
    Adaptive hybrid grouping:
    - Build change score from theta jump, curvature, and theta-rate.
    - Pick local maxima above a profile-adaptive quantile threshold.
    - Apply spacing suppression between neighboring boundaries.
    - Merge adjacent segments with similar vector direction.
    """
    chain = np.asarray(prof.get("chain"), dtype=float)
    elev_s = np.asarray(prof.get("elev_s"), dtype=float)
    dpa = np.asarray(prof.get("d_para"), dtype=float)
    dzz = np.asarray(prof.get("dz"), dtype=float)
    if chain.ndim != 1 or elev_s.ndim != 1 or dpa.ndim != 1 or dzz.ndim != 1:
        return []
    if chain.size < 5:
        return []

    if "slip_span" in prof and prof["slip_span"]:
        smin, smax = map(float, prof["slip_span"])
    else:
        finite_all = np.isfinite(chain) & np.isfinite(elev_s)
        if int(np.count_nonzero(finite_all)) < 2:
            return []
        smin = float(np.nanmin(chain[finite_all]))
        smax = float(np.nanmax(chain[finite_all]))
    if smax < smin:
        smin, smax = smax, smin
    if smax <= smin:
        return []

    mask = (
        np.isfinite(chain)
        & np.isfinite(elev_s)
        & np.isfinite(dpa)
        & np.isfinite(dzz)
        & (chain >= smin)
        & (chain <= smax)
    )
    if int(np.count_nonzero(mask)) < 5:
        if (smax - smin) >= float(min_len_m):
            return [{"id": "G1", "start": float(smin), "end": float(smax), "color": "#1f77b4"}]
        return []

    ch = chain[mask]
    zz = elev_s[mask]
    dx = dpa[mask]
    dz = dzz[mask]
    order = np.argsort(ch)
    ch = ch[order]
    zz = zz[order]
    dx = dx[order]
    dz = dz[order]

    ch_u, uniq_idx = np.unique(ch, return_index=True)
    zz_u = zz[uniq_idx]
    dx_u = dx[uniq_idx]
    dz_u = dz[uniq_idx]
    if ch_u.size < 5:
        if (smax - smin) >= float(min_len_m):
            return [{"id": "G1", "start": float(smin), "end": float(smax), "color": "#1f77b4"}]
        return []

    ds = np.diff(ch_u)
    ds = ds[np.isfinite(ds) & (ds > 1e-9)]
    if ds.size == 0:
        if (smax - smin) >= float(min_len_m):
            return [{"id": "G1", "start": float(smin), "end": float(smax), "color": "#1f77b4"}]
        return []
    med_ds = float(np.nanmedian(ds))
    w_est = int(round(float(max(0.5, smooth_span_m)) / med_ds))
    if (w_est % 2) == 0:
        w_est += 1
    w_est = int(max(5, min(w_est, 31)))

    zz_sm = _smooth_series(zz_u, w_est, poly=2)
    theta_deg = np.degrees(np.unwrap(np.arctan2(dz_u, dx_u)))
    theta_sm = _smooth_series(theta_deg, w_est, poly=2)
    kappa = np.asarray(_curvature_3pt(ch_u, zz_sm), dtype=float)
    theta_rate = _theta_rate_deg_per_m(ch_u, dx_u, dz_u)
    theta_rate_sm = _smooth_series(theta_rate, w_est, poly=2)

    jump = np.zeros(ch_u.shape, dtype=float)
    if theta_sm.size >= 2:
        dth = np.abs((theta_sm[1:] - theta_sm[:-1] + 180.0) % 360.0 - 180.0)
        jump[1:] = dth

    z_jump = _robust_zscore_abs(jump)
    z_k = _robust_zscore_abs(kappa)
    z_rate = _robust_zscore_abs(theta_rate_sm)
    score = 0.45 * z_jump + 0.35 * z_k + 0.20 * z_rate

    ms = np.isfinite(score)
    if int(np.count_nonzero(ms)) < 3:
        if (smax - smin) >= float(min_len_m):
            return [{"id": "G1", "start": float(smin), "end": float(smax), "color": "#1f77b4"}]
        return []
    q = float(np.clip(score_quantile, 50.0, 99.5))
    thr = float(np.nanpercentile(score[ms], q))

    peak_idx = []
    for i in range(1, ch_u.size - 1):
        if not np.isfinite(score[i]):
            continue
        if score[i] < thr:
            continue
        if (score[i] >= score[i - 1]) and (score[i] > score[i + 1]):
            peak_idx.append(i)

    candidates = []
    for i in peak_idx:
        x = float(ch_u[i])
        if (x <= (smin + 1e-6)) or (x >= (smax - 1e-6)):
            continue
        candidates.append((x, float(score[i])))
    candidates.sort(key=lambda t: t[1], reverse=True)

    selected = []
    min_gap = float(max(0.0, min_boundary_gap_m))
    for x, sc in candidates:
        if all(abs(x - sx) >= min_gap for sx, _ in selected):
            selected.append((x, sc))
    selected.sort(key=lambda t: t[0])

    boundaries = [float(smin)] + [x for x, _ in selected] + [float(smax)]
    uniq = [boundaries[0]]
    for b in boundaries[1:]:
        if (float(b) - float(uniq[-1])) > 1e-6:
            uniq.append(float(b))
    if len(uniq) < 2:
        return []

    ranges = []
    for s, e in zip(uniq[:-1], uniq[1:]):
        if e > s:
            ranges.append({"start": float(s), "end": float(e)})
    if not ranges:
        return []

    merged = _merge_by_vector_direction(
        ch_u, dx_u, dz_u, ranges, angle_tol_deg=float(max(1.0, merge_angle_tol_deg))
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    out = []
    for rg in merged:
        s = max(float(rg["start"]), float(smin))
        e = min(float(rg["end"]), float(smax))
        if (e - s) < float(min_len_m):
            continue
        i = len(out) + 1
        out.append({
            "id": f"G{i}",
            "start": float(s),
            "end": float(e),
            "color": colors[(i - 1) % len(colors)],
        })

    if not out and (smax - smin) >= float(min_len_m):
        out = [{
            "id": "G1",
            "start": float(smin),
            "end": float(smax),
            "color": colors[0],
        }]
    return out

def auto_group_profile_by_criteria(
    prof: dict,
    rdp_eps_m: float = 0.5,
    curvature_thr_abs: float = 2.0,
    horizontal_angle_tol_deg: float = 5.0,
    adjacent_angle_split_deg: float = 10.0,
    horizontal_cluster_gap_m: float = 0.5,
    min_len_m: float = 0.20,
) -> list:
    """
    New standalone grouping rule:
    1) Start/end of slip block.
    2) Points where |curvature| > curvature_thr_abs on RDP-simplified smoothed profile.
    3) Points where vector is horizontal within +/- horizontal_angle_tol_deg.
    4) Split between adjacent vectors when |delta angle| > adjacent_angle_split_deg.
    """
    chain = np.asarray(prof.get("chain"), dtype=float)
    elev_s = np.asarray(prof.get("elev_s"), dtype=float)
    dpa = np.asarray(prof.get("d_para"), dtype=float)
    dzz = np.asarray(prof.get("dz"), dtype=float)
    if chain.ndim != 1 or elev_s.ndim != 1 or dpa.ndim != 1 or dzz.ndim != 1:
        return []
    if chain.size < 2:
        return []

    if "slip_span" in prof and prof["slip_span"]:
        smin, smax = map(float, prof["slip_span"])
    else:
        finite_all = np.isfinite(chain) & np.isfinite(elev_s)
        if finite_all.sum() < 2:
            return []
        smin = float(np.nanmin(chain[finite_all]))
        smax = float(np.nanmax(chain[finite_all]))
    if smax < smin:
        smin, smax = smax, smin
    if smax <= smin:
        return []

    boundaries = [smin, smax]

    finite_curve = np.isfinite(chain) & np.isfinite(elev_s) & (chain >= smin) & (chain <= smax)
    if finite_curve.sum() >= 3:
        pts = list(zip(chain[finite_curve].tolist(), elev_s[finite_curve].tolist()))
        rdp_pts = _rdp_polyline(pts, rdp_eps_m)
        if len(rdp_pts) >= 3:
            k_vals = _curvature_points_from_rdp(rdp_pts)
            for (xv, _), kv in zip(rdp_pts, k_vals):
                if np.isfinite(kv) and abs(float(kv)) > float(curvature_thr_abs):
                    boundaries.append(float(xv))

    finite_vec = np.isfinite(chain) & np.isfinite(dpa) & np.isfinite(dzz) & (chain >= smin) & (chain <= smax)
    if finite_vec.any():
        chain_v = chain[finite_vec]
        dpa_v = dpa[finite_vec]
        dzz_v = dzz[finite_vec]
        ang = np.degrees(np.arctan2(dzz_v, dpa_v))  # [-180, 180]
        ang_abs = np.abs(ang)
        # Distance to nearest horizontal direction (0 deg or 180 deg).
        dist_to_horizontal = np.minimum(ang_abs, np.abs(180.0 - ang_abs))
        horiz = np.isfinite(dist_to_horizontal) & (dist_to_horizontal <= float(horizontal_angle_tol_deg))
        if np.any(horiz):
            chain_h = chain_v[horiz]
            boundaries.extend(_cluster_chain_marks(chain_h, gap_m=horizontal_cluster_gap_m))

        # Adjacent-vector split rule: split when angle jump is larger than threshold.
        if chain_v.size >= 2:
            a1 = ang[:-1]
            a2 = ang[1:]
            jump = np.abs((a2 - a1 + 180.0) % 360.0 - 180.0)
            split_mask = np.isfinite(jump) & (jump > float(adjacent_angle_split_deg))
            if np.any(split_mask):
                # Split at the exact position of vector i+1.
                split_points = chain_v[1:][split_mask]
                boundaries.extend(split_points.tolist())

    boundaries = [b for b in boundaries if np.isfinite(b) and (smin <= b <= smax)]
    if not boundaries:
        return []
    boundaries = sorted(boundaries)
    uniq = [boundaries[0]]
    for b in boundaries[1:]:
        if abs(b - uniq[-1]) > 1e-6:
            uniq.append(b)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    groups = []
    for s, e in zip(uniq[:-1], uniq[1:]):
        if (e - s) < float(min_len_m):
            continue
        i = len(groups) + 1
        groups.append({
            "id": f"G{i}",
            "start": float(s),
            "end": float(e),
            "color": colors[(i - 1) % len(colors)],
        })
    if not groups and (smax - smin) >= float(min_len_m):
        groups = [{
            "id": "G1",
            "start": float(smin),
            "end": float(smax),
            "color": colors[0],
        }]
    return groups


def auto_group_profile_by_theta_anchor(
    prof: dict,
    angle_threshold_deg: float = 20.0,
    min_len_m: float = 2,
) -> list:
    """
    Split groups by scanning vectors from left to right:
    - Keep the first vector angle of the current group as anchor.
    - When |theta(i) - theta(anchor)| > threshold, close current group and start a new one.
    """
    chain = np.asarray(prof.get("chain"), dtype=float)
    theta = np.asarray(prof.get("theta"), dtype=float)
    if chain.ndim != 1 or theta.ndim != 1:
        return []
    n = int(min(chain.size, theta.size))
    if n < 2:
        return []
    chain = chain[:n]
    theta = theta[:n]

    if "slip_span" in prof and prof["slip_span"]:
        smin, smax = map(float, prof["slip_span"])
    else:
        finite_all = np.isfinite(chain) & np.isfinite(theta)
        if int(np.count_nonzero(finite_all)) < 2:
            return []
        smin = float(np.nanmin(chain[finite_all]))
        smax = float(np.nanmax(chain[finite_all]))
    if smax < smin:
        smin, smax = smax, smin
    if smax <= smin:
        return []

    mask = np.isfinite(chain) & np.isfinite(theta) & (chain >= smin) & (chain <= smax)
    if int(np.count_nonzero(mask)) < 2:
        return []

    chain_v = chain[mask]
    theta_v = theta[mask]
    order = np.argsort(chain_v)
    chain_v = chain_v[order]
    theta_v = theta_v[order]
    if chain_v.size < 2:
        return []

    boundaries = [float(chain_v[0])]
    anchor_theta = float(theta_v[0])
    thr = float(max(0.0, angle_threshold_deg))

    for i in range(1, chain_v.size):
        d = abs(((float(theta_v[i]) - anchor_theta + 180.0) % 360.0) - 180.0)
        if d > thr:
            b = 0.5 * (float(chain_v[i - 1]) + float(chain_v[i]))
            if b > boundaries[-1] + 1e-9:
                boundaries.append(float(b))
            anchor_theta = float(theta_v[i])

    end_b = float(chain_v[-1])
    if end_b > boundaries[-1] + 1e-9:
        boundaries.append(end_b)
    if len(boundaries) < 2:
        return []

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    groups = []
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        s = float(max(s, smin))
        e = float(min(e, smax))
        if (e - s) < float(min_len_m):
            continue
        i = len(groups) + 1
        groups.append({
            "id": f"G{i}",
            "start": s,
            "end": e,
            "color": colors[(i - 1) % len(colors)],
        })

    if not groups and (smax - smin) >= float(min_len_m):
        groups = [{
            "id": "G1",
            "start": float(smin),
            "end": float(smax),
            "color": colors[0],
        }]
    return groups

def _mean_angle_deg(d_para_seg, dz_seg):
    V = np.vstack([d_para_seg, dz_seg]).T
    V = V[np.all(np.isfinite(V), axis=1)]
    if V.size == 0:
        return None
    ang = np.degrees(np.arctan2(V[:,1], V[:,0]))
    ux = np.cos(np.radians(ang)).mean()
    uy = np.sin(np.radians(ang)).mean()
    return float(np.degrees(np.arctan2(uy, ux)))

def _merge_by_vector_direction(chain: np.ndarray,
                               d_para: np.ndarray, dz: np.ndarray,
                               ranges: list, angle_tol_deg: float=20.0) -> list:
    out = []
    prev = None
    for rg in ranges:
        m = (chain >= rg["start"]) & (chain <= rg["end"])
        ang = _mean_angle_deg(d_para[m], dz[m])
        if prev is None:
            prev = dict(rg); prev["_ang"] = ang
        else:
            ok = (ang is not None and prev.get("_ang") is not None)
            if ok:
                diff = abs((ang - prev["_ang"] + 180) % 360 - 180)
            if ok and diff <= angle_tol_deg:
                prev["end"] = rg["end"]
                m2 = (chain >= prev["start"]) & (chain <= prev["end"])
                prev["_ang"] = _mean_angle_deg(d_para[m2], dz[m2])
            else:
                out.append({k:v for k,v in prev.items() if not k.startswith("_")})
                prev = dict(rg); prev["_ang"] = ang
    if prev is not None:
        out.append({k:v for k,v in prev.items() if not k.startswith("_")})
    return out

def auto_group_profile(prof: dict,
                       rdp_eps_m: float = 0.5,
                       K_min_abs: float = 0.02,
                       theta_rate_min: float = 5.0,
                       min_len_m: float = 3.0) -> list:

    chain = np.asarray(prof.get("chain"))
    elev  = np.asarray(prof.get("elev_s"))
    dpa   = np.asarray(prof.get("d_para"))
    dzz   = np.asarray(prof.get("dz"))

    assert chain.ndim == elev.ndim == dpa.ndim == dzz.ndim == 1, "Profile arrays must be 1D"
    assert chain.size >= 5, "Profile too short"

    finite = np.isfinite(chain) & np.isfinite(elev)
    if finite.sum() < 3:
        return []
    chain_f = chain[finite]
    elev_f  = elev[finite]
    dpa_f   = dpa[finite]
    dzz_f   = dzz[finite]

    pts = np.vstack([chain_f, elev_f]).T
    keep_idx = _rdp_indices(pts, rdp_eps_m); keep_idx.sort()

    cand = []
    for i in range(len(keep_idx)-1):
        i0, i1 = int(keep_idx[i]), int(keep_idx[i+1])
        s, e = float(chain_f[i0]), float(chain_f[i1])
        if (e - s) < min_len_m:
            continue
        m = (chain_f >= s) & (chain_f <= e)

        K = _curvature_3pt(chain_f[m], elev_f[m])
        K_mean = float(np.nanmean(np.abs(K))) if np.any(np.isfinite(K)) else 0.0

        th_rate = _theta_rate_deg_per_m(chain_f[m], dpa_f[m], dzz_f[m])
        TR_q95  = float(np.nanpercentile(np.abs(th_rate), 95)) if np.any(np.isfinite(th_rate)) else 0.0

        if (K_mean >= K_min_abs) or (TR_q95 >= theta_rate_min):
            cand.append({"start": s, "end": e})

    merged = _merge_by_vector_direction(chain_f, dpa_f, dzz_f, cand, angle_tol_deg=20.0)

    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
              "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    out = []

    if "slip_span" in prof and prof["slip_span"]:
        smin, smax = prof["slip_span"]
    else:
        keep_all = np.isfinite(chain) & np.isfinite(elev)
        smin = float(np.nanmin(chain[keep_all]));
        smax = float(np.nanmax(chain[keep_all]))

    for i, rg in enumerate(merged, 1):
        s = max(float(rg["start"]), smin)
        e = min(float(rg["end"]), smax)
        if e <= s:
            continue
        out.append({
            "id": f"G{i}",
            "start": s,
            "end": e,
            "color": colors[(i - 1) % len(colors)]
        })

    return out

# === SLIP CURVE FITTING & RENDER (place at END of file) ===

def estimate_slip_curve(
    prof: dict,
    groups: list,
    ds: float = 0.2,
    smooth_factor: float = 0.1,
    depth_gain: float = 3.0,
    min_depth: float = 1.0,
) -> dict:

    try:
        from scipy.signal import savgol_filter as _sg
    except Exception:
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


def _median_theta_deg(chain: np.ndarray, theta: np.ndarray, s0: float, s1: float) -> Optional[float]:
    lo, hi = (s0, s1) if s0 <= s1 else (s1, s0)
    m = np.isfinite(chain) & np.isfinite(theta) & (chain >= lo) & (chain <= hi)
    if int(np.count_nonzero(m)) == 0:
        return None
    return float(np.nanmedian(theta[m]))


def fit_nurbs_segmented_curve(
    prof: dict,
    groups: list,
    target_s,
    target_z,
    degree: int = 3,
    n_ctrl: int = 4,
    samples_per_meter: float = 6.0,
) -> dict:
    """
    Piecewise NURBS (C0-continuous) from slip crest -> toe by group boundaries.
    For each segment, use median vector angle of the corresponding group.
    """
    chain = np.asarray(prof.get("chain", []), dtype=float)
    elevg = np.asarray(prof.get("elev_s", []), dtype=float)
    theta = np.asarray(prof.get("theta", []), dtype=float)
    if chain.ndim != 1 or elevg.ndim != 1 or theta.ndim != 1:
        return {"chain": [], "elev": []}
    if chain.size < 3:
        return {"chain": [], "elev": []}

    ok = np.isfinite(chain) & np.isfinite(elevg)
    if int(np.count_nonzero(ok)) < 3:
        return {"chain": [], "elev": []}
    chain = chain[ok]
    elevg = elevg[ok]
    theta = theta[ok]

    if "slip_span" in prof and prof["slip_span"]:
        smin, smax = map(float, prof["slip_span"])
    else:
        smin = float(np.nanmin(chain))
        smax = float(np.nanmax(chain))
    if smax <= smin:
        return {"chain": [], "elev": []}

    # Normalize and sort groups; keep only segments inside slip span.
    norm_groups = []
    for g in (groups or []):
        try:
            gs = float(g.get("start", g.get("start_chainage", np.nan)))
            ge = float(g.get("end", g.get("end_chainage", np.nan)))
        except Exception:
            continue
        if not np.isfinite(gs) or not np.isfinite(ge):
            continue
        if ge < gs:
            gs, ge = ge, gs
        gs = max(gs, smin)
        ge = min(ge, smax)
        if ge > gs:
            norm_groups.append((gs, ge))
    if not norm_groups:
        return {"chain": [], "elev": []}
    norm_groups.sort(key=lambda x: (x[0], x[1]))

    target_s = np.asarray(target_s, dtype=float)
    target_z = np.asarray(target_z, dtype=float)
    target_ok = np.isfinite(target_s) & np.isfinite(target_z)
    target_s = target_s[target_ok]
    target_z = target_z[target_ok]
    has_target = target_s.size >= 2

    degree = int(max(1, degree))
    n_ctrl = int(max(4, n_ctrl))
    if n_ctrl != 4:
        n_ctrl = 4  # current implementation uses fixed 4 control points per group segment
    degree = min(degree, n_ctrl - 1)

    # Lock start/end of segmented control points to first/last group boundaries.
    global_start = float(norm_groups[0][0])
    global_end = float(norm_groups[-1][1])

    # Start point at Group 1 start (instead of slip crest).
    cur_s = float(global_start)
    cur_z = float(np.interp(cur_s, target_s, target_z)) if has_target else float(np.interp(cur_s, chain, elevg))

    out_s = []
    out_z = []
    last_theta = _median_theta_deg(chain, theta, smin, smax)
    if last_theta is None:
        last_theta = 0.0

    for idx, (gs, ge) in enumerate(norm_groups):
        seg_s = float(max(cur_s, gs))
        seg_e = float(max(seg_s, ge))
        if idx == (len(norm_groups) - 1):
            seg_e = float(max(seg_s, global_end))
        if seg_e <= seg_s:
            continue

        theta_med = _median_theta_deg(chain, theta, gs, ge)
        if theta_med is None:
            theta_med = last_theta
        last_theta = theta_med

        theta_clip = float(np.clip(theta_med, -85.0, 85.0))
        slope = float(np.tan(np.radians(theta_clip)))
        L = float(seg_e - seg_s)

        # Segment anchor at right boundary; if target is available, anchor to it.
        z_end_anchor = (
            float(np.interp(seg_e, target_s, target_z))
            if has_target
            else float(cur_z + slope * L)
        )

        cp_x = np.array([seg_s, seg_s + (L / 3.0), seg_s + (2.0 * L / 3.0), seg_e], dtype=float)
        cp_z = np.array(
            [
                cur_z,
                cur_z + slope * (L / 3.0),
                z_end_anchor - slope * (L / 3.0),
                z_end_anchor,
            ],
            dtype=float,
        )
        ctrl = np.vstack([cp_x, cp_z]).T
        weights = np.ones(ctrl.shape[0], dtype=float)
        n_samples = int(max(24, math.ceil(L * float(max(samples_per_meter, 2.0)))))

        try:
            seg_curve = _eval_nurbs_curve(ctrl, weights, degree=degree, n_samples=n_samples)
            sx = seg_curve[:, 0]
            sz = seg_curve[:, 1]
            fin = np.isfinite(sx) & np.isfinite(sz)
            sx = sx[fin]
            sz = sz[fin]
            if sx.size < 2:
                raise ValueError("invalid NURBS segment")
        except Exception:
            sx = np.linspace(seg_s, seg_e, n_samples)
            sz = np.linspace(cur_z, z_end_anchor, n_samples)

        # Keep direction along chainage.
        order = np.argsort(sx)
        sx = sx[order]
        sz = sz[order]

        if not out_s:
            out_s.extend(sx.tolist())
            out_z.extend(sz.tolist())
        else:
            out_s.extend(sx[1:].tolist())
            out_z.extend(sz[1:].tolist())

        # Group boundary intersection for next segment (progressive chainage).
        cur_s = float(seg_e)
        cur_z = float(np.interp(cur_s, sx, sz))

    if len(out_s) < 2:
        return {"chain": [], "elev": []}

    return {"chain": np.asarray(out_s, dtype=float), "elev": np.asarray(out_z, dtype=float)}
