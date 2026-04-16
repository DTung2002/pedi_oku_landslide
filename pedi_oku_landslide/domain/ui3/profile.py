import os
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
from shapely.geometry import LineString




def _open_raster(path: str):
    ds = rasterio.open(path)
    arr = ds.read(1).astype("float32")
    return ds, arr


def _densify(line: LineString, step_m: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if line is None:
        return np.array([]), np.array([]), np.array([])
    if getattr(line, "is_empty", False):
        return np.array([]), np.array([]), np.array([])
    try:
        length = float(line.length)
    except Exception:
        return np.array([]), np.array([]), np.array([])
    if not np.isfinite(length) or length <= 0:
        return np.array([]), np.array([]), np.array([])

    num_steps = int(np.floor(length / step_m))
    s_list = []

    for i in range(num_steps + 1):
        s_list.append(i * step_m)

    # Đảm bảo điểm cuối cùng được nối vào khớp với chiều dài tổng
    if abs(s_list[-1] - length) > 1e-9:
        s_list.append(length)

    s = np.array(s_list)
    xs = np.empty(len(s))
    ys = np.empty(len(s))
    for i, d in enumerate(s):
        p = line.interpolate(d)
        xs[i], ys[i] = p.x, p.y

    return xs, ys, s


def _sample(ds: rasterio.io.DatasetReader, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    vals = list(ds.sample(list(zip(xs.tolist(), ys.tolist())), indexes=1))
    out = np.array([v[0] if len(v) else np.nan for v in vals], dtype="float32")
    nodata = ds.nodata
    if nodata is not None:
        if np.isnan(nodata):
            out[~np.isfinite(out)] = np.nan
        else:
            out[np.isclose(out, float(nodata))] = np.nan
    return out


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
    dx_ds, _ = _open_raster(dx_path)
    dy_ds, _ = _open_raster(dy_path)
    dz_ds, _ = _open_raster(dz_path)

    xs, ys, chain = _densify(line_geom, step_m)
    if xs.size == 0:
        for ds in (dem_ds, dx_ds, dy_ds, dz_ds):
            ds.close()
        return {}

    elev = _sample(dem_ds, xs, ys)
    dx = _sample(dx_ds, xs, ys)
    dy = _sample(dy_ds, xs, ys)
    dz = _sample(dz_ds, xs, ys)

    elev_orig = None
    if dem_orig_path and os.path.exists(dem_orig_path):
        dem_orig_ds, _ = _open_raster(dem_orig_path)
        elev_orig = _sample(dem_orig_ds, xs, ys)
        dem_orig_ds.close()

    px_m = abs(float(dx_ds.transform.a))
    py_m = abs(float(dx_ds.transform.e))
    dx = dx * px_m
    dy = -dy * py_m

    tang = np.zeros((xs.size, 2), dtype=float)
    tang[1:-1, 0] = xs[2:] - xs[:-2]
    tang[1:-1, 1] = ys[2:] - ys[:-2]
    tang[0, 0] = xs[1] - xs[0]
    tang[0, 1] = ys[1] - ys[0]
    tang[-1, 0] = xs[-1] - xs[-2]
    tang[-1, 1] = ys[-1] - ys[-2]
    nrm = np.hypot(tang[:, 0], tang[:, 1])
    nrm[nrm == 0] = 1.0
    tang[:, 0] /= nrm
    tang[:, 1] /= nrm

    d_para = dx * tang[:, 0] + dy * tang[:, 1]
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
            dx[~keep] = np.nan
            dy[~keep] = np.nan
            dz[~keep] = np.nan
            d_para[~keep] = np.nan
            theta_deg[~keep] = np.nan
            if elev_orig is not None:
                elev_orig[~keep] = np.nan

    elev_s = elev.copy()
    for ds in (dem_ds, dx_ds, dy_ds, dz_ds):
        ds.close()

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
        "x": xs,
        "y": ys,
        "elev": elev,
        "elev_s": elev_s,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "d_para": d_para,
        "theta": theta_deg,
        "slip_mask": mask_bool,
        "slip_span": slip_span,
        "elev_orig": elev_orig,
    }
