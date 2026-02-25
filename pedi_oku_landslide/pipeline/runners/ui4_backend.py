from __future__ import annotations

import glob
import json
import math
import os
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore[assignment]

try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling
    from rasterio.windows import Window, transform as window_transform
except Exception:
    rasterio = None  # type: ignore[assignment]
    from_origin = None  # type: ignore[assignment]
    reproject = None  # type: ignore[assignment]
    Resampling = None  # type: ignore[assignment]
    Window = None  # type: ignore[assignment]
    window_transform = None  # type: ignore[assignment]

try:
    from scipy.linalg import lu_factor, lu_solve
    from scipy.optimize import curve_fit
    from scipy.spatial.distance import cdist
except Exception:
    lu_factor = None  # type: ignore[assignment]
    lu_solve = None  # type: ignore[assignment]
    curve_fit = None  # type: ignore[assignment]
    cdist = None  # type: ignore[assignment]

try:
    from shapely.geometry import MultiPoint, Point
    from shapely.prepared import prep
except Exception:
    MultiPoint = None  # type: ignore[assignment]
    Point = None  # type: ignore[assignment]
    prep = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore[assignment]


DEFAULT_UI4_PARAMS: Dict[str, Any] = {
    "chainage_step_m": 1.0,       # decimate along each curve
    "grid_res_m": 0.5,            # kriging grid resolution
    "buffer_m": 5.0,              # convex hull buffer
    "nodata_out": -9999.0,
    "variogram_pairs": 20000,
    "variogram_bins": 20,
    "variogram_min_pairs_per_bin": 50,
    "variogram_percentile_max_h": 95.0,
    "random_seed": 0,
    "predict_chunk_size": 2000,
    "duplicate_round_decimals": 3,  # merge near-duplicate points after depth sampling
}

DEFAULT_UI4_CONTOUR_PARAMS: Dict[str, Any] = {
    "surface_interval_m": 5.0,
    "depth_interval_m": 2.0,
    "figsize": (10, 8),
    "dpi": 180,
    "label_fontsize": 8,
    "linewidth": 0.8,
}


def _require_ui4_runtime_deps() -> None:
    missing = []
    if pd is None:
        missing.append("pandas")
    if any(x is None for x in (rasterio, from_origin, reproject, Resampling, Window, window_transform)):
        missing.append("rasterio")
    if any(x is None for x in (lu_factor, lu_solve, curve_fit, cdist)):
        missing.append("scipy")
    if any(x is None for x in (MultiPoint, Point, prep)):
        missing.append("shapely")
    if missing:
        raise RuntimeError(
            "UI4 backend dependencies are missing in this Python environment: "
            + ", ".join(missing)
        )


def _require_contour_deps() -> None:
    missing = []
    if any(x is None for x in (rasterio, reproject, Resampling, Window, window_transform)):
        missing.append("rasterio")
    if plt is None:
        missing.append("matplotlib")
    if missing:
        raise RuntimeError(
            "UI4 contour dependencies are missing in this Python environment: "
            + ", ".join(missing)
        )


def _log(log_fn: Optional[Callable[[str], None]], msg: str) -> None:
    if log_fn is None:
        return
    try:
        log_fn(str(msg))
    except Exception:
        pass


def _read_json_if_exists(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _pick_existing(candidates: Iterable[str]) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return ""


def _format_res_tag(grid_res_m: float) -> str:
    txt = f"{float(grid_res_m):g}".replace(".", "p")
    return f"{txt}m"


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def _find_ui4_mask_tif(run_dir: str) -> str:
    ui1_dir = os.path.join(run_dir, "ui1")
    candidates = [
        os.path.join(ui1_dir, "landslide_mask.tif"),
        os.path.join(ui1_dir, "detect_mask.tif"),
        os.path.join(ui1_dir, "mask.tif"),
        os.path.join(ui1_dir, "mask_binary.tif"),
    ]
    return _pick_existing(candidates)


def _finite_raster_stats(arr: np.ndarray) -> Dict[str, Any]:
    finite = np.isfinite(arr)
    n = int(np.count_nonzero(finite))
    if n == 0:
        return {"valid_pixels": 0, "min": None, "max": None}
    vals = arr[finite]
    return {
        "valid_pixels": n,
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
    }


def apply_mask_to_raster(
    src_tif: str,
    mask_tif: str,
    out_tif: str,
    *,
    out_nodata: float = -9999.0,
    crop_to_mask_bbox: bool = True,
) -> Dict[str, Any]:
    """
    Reproject/resample `mask_tif` to `src_tif` grid and set pixels outside mask to NoData.
    """
    _require_ui4_runtime_deps()
    src_tif = os.path.abspath(str(src_tif or ""))
    mask_tif = os.path.abspath(str(mask_tif or ""))
    out_tif = os.path.abspath(str(out_tif or ""))
    if not src_tif or not os.path.exists(src_tif):
        return {"ok": False, "error": f"Source raster not found: {src_tif}"}
    if not mask_tif or not os.path.exists(mask_tif):
        return {"ok": False, "error": f"Mask raster not found: {mask_tif}"}

    with rasterio.open(src_tif) as src:
        src_arr = src.read(1).astype(float)
        src_profile = src.profile.copy()
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata
        if src_nodata is not None:
            if np.isnan(src_nodata):
                src_arr[~np.isfinite(src_arr)] = np.nan
            else:
                src_arr[np.isclose(src_arr, float(src_nodata))] = np.nan

        with rasterio.open(mask_tif) as msk:
            mask_src = msk.read(1)
            mask_dst = np.zeros(src_arr.shape, dtype=np.uint8)
            reproject(
                source=mask_src,
                destination=mask_dst,
                src_transform=msk.transform,
                src_crs=msk.crs,
                dst_transform=src_transform,
                dst_crs=src_crs,
                resampling=Resampling.nearest,
            )

    mask_bool = np.isfinite(mask_dst) & (mask_dst > 0)
    if not np.any(mask_bool):
        return {"ok": False, "error": f"Mask has no positive pixels after reprojection: {mask_tif}"}

    out_arr = np.where(mask_bool, src_arr, float(out_nodata)).astype(np.float32)
    out_profile = src_profile.copy()
    out_profile.update(nodata=float(out_nodata), dtype="float32", count=1, compress="deflate")

    if crop_to_mask_bbox:
        rows, cols = np.where(mask_bool)
        r0 = int(rows.min())
        r1 = int(rows.max()) + 1
        c0 = int(cols.min())
        c1 = int(cols.max()) + 1
        win = Window(col_off=c0, row_off=r0, width=(c1 - c0), height=(r1 - r0))
        out_arr = out_arr[r0:r1, c0:c1]
        out_profile.update(
            height=int(out_arr.shape[0]),
            width=int(out_arr.shape[1]),
            transform=window_transform(win, src_transform),
        )

    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    with rasterio.open(out_tif, "w", **out_profile) as dst:
        dst.write(out_arr, 1)

    stats_arr = out_arr.astype(float)
    stats_arr[np.isclose(stats_arr, float(out_nodata))] = np.nan
    return {
        "ok": True,
        "src_tif": src_tif,
        "mask_tif": mask_tif,
        "out_tif": out_tif,
        "cropped": bool(crop_to_mask_bbox),
        "shape": {"ny": int(out_arr.shape[0]), "nx": int(out_arr.shape[1])},
        "stats": _finite_raster_stats(stats_arr),
    }


def collect_ui4_run_inputs(run_dir: str) -> Dict[str, Any]:
    """
    Discover run-scoped UI4 inputs for a single run directory.
    This powers UI4 tab readiness checks and serves as the default resolver
    for run-scoped UI4 backend execution.
    """
    run_dir = os.path.abspath(str(run_dir or "").strip())
    if not run_dir:
        return {"ok": False, "error": "Missing run_dir", "run_dir": ""}

    input_dir = os.path.join(run_dir, "input")
    ui2_dir = os.path.join(run_dir, "ui2")
    ui3_dir = os.path.join(run_dir, "ui3")
    curve_dir = os.path.join(ui3_dir, "curve")
    mask_tif = _find_ui4_mask_tif(run_dir)

    dem_tif_candidates = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    # Prefer DEM-like filenames when multiple TIFFs exist in input/.
    dem_preferred = [
        p for p in dem_tif_candidates
        if any(tag in os.path.basename(p).lower() for tag in ("ground", "dem"))
    ]
    dem_path = _pick_existing(dem_preferred or dem_tif_candidates)

    intersections_json = _pick_existing([os.path.join(ui2_dir, "intersections_main_cross.json")])
    anchors_xyz_json = _pick_existing([os.path.join(ui3_dir, "anchors_xyz.json")])

    nurbs_name_re = re.compile(
        r"^nurbs_(?:CL|ML)\d+__\([^)]+_m\)\.json$",
        re.IGNORECASE,
    )
    nurbs_curve_jsons = sorted(
        p for p in glob.glob(os.path.join(curve_dir, "nurbs_*.json"))
        if bool(nurbs_name_re.match(os.path.basename(p)))
    )
    group_jsons = sorted(glob.glob(os.path.join(curve_dir, "group_*.json")))
    nurbs_info_jsons = sorted(glob.glob(os.path.join(curve_dir, "nurbs_info_*.json")))

    counts = {
        "nurbs_curves": len(nurbs_curve_jsons),
        "groups": len(group_jsons),
        "nurbs_info": len(nurbs_info_jsons),
    }
    ready_checks = {
        "run_dir_exists": os.path.isdir(run_dir),
        "input_dir_exists": os.path.isdir(input_dir),
        "dem_exists": bool(dem_path and os.path.exists(dem_path) and str(dem_path).lower().endswith(".tif")),
        "curve_dir_exists": os.path.isdir(curve_dir),
        "nurbs_curves_exist": counts["nurbs_curves"] > 0,
        "intersections_exists": bool(intersections_json),
        "anchors_xyz_exists": bool(anchors_xyz_json),
    }

    missing_required = []
    if not ready_checks["dem_exists"]:
        missing_required.append("DEM (.tif in input/)")
    if not ready_checks["nurbs_curves_exist"]:
        missing_required.append("NURBS curves (nurbs_CLn__/nurbs_MLn__ in ui3/curve)")

    return {
        "ok": True,
        "run_dir": run_dir,
        "paths": {
            "input_dir": os.path.abspath(input_dir),
            "dem": dem_path,
            "dem_tif_candidates": [os.path.abspath(p) for p in dem_tif_candidates],
            "mask_tif": mask_tif,
            "intersections_main_cross_json": intersections_json,
            "anchors_xyz_json": anchors_xyz_json,
            "ui3_curve_dir": os.path.abspath(curve_dir),
            "nurbs_curve_jsons": [os.path.abspath(p) for p in nurbs_curve_jsons],
            "group_jsons": [os.path.abspath(p) for p in group_jsons],
            "nurbs_info_jsons": [os.path.abspath(p) for p in nurbs_info_jsons],
        },
        "counts": counts,
        "ready_checks": ready_checks,
        "ready_for_ui4": len(missing_required) == 0,
        "missing_required": missing_required,
    }


def decimate_by_chainage(df: pd.DataFrame, ds: float = 1.0) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.sort_values("chainage_m").copy()
    keep_idx = []
    last = -1e18
    for i, r in out.iterrows():
        ch = _safe_float(r.get("chainage_m"))
        if not np.isfinite(ch):
            continue
        if ch - last >= float(ds):
            keep_idx.append(i)
            last = ch
    if not keep_idx:
        return out.iloc[0:0].copy()
    return out.loc[keep_idx].copy()


def exp_variogram(h: np.ndarray, nugget: float, sill: float, rang: float) -> np.ndarray:
    rang = np.maximum(rang, 1e-6)
    return nugget + sill * (1.0 - np.exp(-h / rang))


def _curve_json_to_df(path: str) -> pd.DataFrame:
    _require_ui4_runtime_deps()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = data.get("points", [])
    if not isinstance(pts, list):
        return pd.DataFrame(columns=["line_id", "index", "chainage_m", "x", "y", "z", "source_json"])

    rows: List[Dict[str, Any]] = []
    line_id = str(data.get("line_id") or os.path.splitext(os.path.basename(path))[0])
    for k, p in enumerate(pts):
        if not isinstance(p, dict):
            continue
        ch = p.get("chainage_m", p.get("chain"))
        z = p.get("z", p.get("elev_m", p.get("elev")))
        rows.append(
            {
                "line_id": line_id,
                "index": p.get("index", k),
                "chainage_m": _safe_float(ch),
                "x": _safe_float(p.get("x")),
                "y": _safe_float(p.get("y")),
                "z": _safe_float(z),
                "source_json": os.path.abspath(path),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["line_id", "index", "chainage_m", "x", "y", "z", "source_json"])

    df = pd.DataFrame(rows)
    for col in ("chainage_m", "x", "y", "z"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[np.isfinite(df["x"]) & np.isfinite(df["y"]) & np.isfinite(df["z"])].copy()
    if "chainage_m" not in df or df["chainage_m"].isna().all():
        # Fallback: build chainage from cumulative XY distance if missing.
        xy = df[["x", "y"]].to_numpy(dtype=float)
        if len(xy) > 0:
            d = np.zeros(len(xy), dtype=float)
            if len(xy) > 1:
                d[1:] = np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))
            df["chainage_m"] = d
    return df[["line_id", "index", "chainage_m", "x", "y", "z", "source_json"]].copy()


def load_ui4_curve_points(
    json_paths: List[str],
    chainage_step_m: float = 1.0,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _require_ui4_runtime_deps()
    dfs: List[pd.DataFrame] = []
    for p in json_paths:
        try:
            df = _curve_json_to_df(p)
        except Exception as e:
            _log(log_fn, f"[UI4] Skip invalid curve JSON: {p} ({e})")
            continue
        if df.empty:
            _log(log_fn, f"[UI4] Empty curve JSON: {p}")
            continue
        dfs.append(df)

    if not dfs:
        empty = pd.DataFrame(columns=["line_id", "index", "chainage_m", "x", "y", "z", "source_json"])
        return empty.copy(), empty.copy()

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df[
        np.isfinite(all_df["x"]) &
        np.isfinite(all_df["y"]) &
        np.isfinite(all_df["z"]) &
        np.isfinite(all_df["chainage_m"])
    ].copy()

    dec_list: List[pd.DataFrame] = []
    for lid, g in all_df.groupby("line_id", sort=True):
        dec = decimate_by_chainage(g, ds=float(chainage_step_m))
        if dec.empty and not g.empty:
            dec = g.sort_values("chainage_m").iloc[[0]].copy()
        dec_list.append(dec)
        _log(log_fn, f"[UI4] Curve {lid}: raw={len(g)} decimated={len(dec)}")

    dec_df = pd.concat(dec_list, ignore_index=True) if dec_list else all_df.iloc[0:0].copy()
    return all_df, dec_df


def _sample_dem_and_compute_depth(
    dem_path: str,
    dec_df: pd.DataFrame,
    duplicate_round_decimals: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if dec_df.empty:
        return dec_df.copy(), {"n_input": 0, "n_valid_dem": 0, "n_after_dedupe": 0}

    work = dec_df.copy()
    with rasterio.open(dem_path) as src:
        samples = np.array([v[0] for v in src.sample(list(zip(work["x"], work["y"])))], dtype=float)
        nodata = src.nodata
        if nodata is not None:
            if np.isnan(nodata):
                samples[~np.isfinite(samples)] = np.nan
            else:
                samples[np.isclose(samples, float(nodata))] = np.nan
        work["z_dem"] = samples

    work = work[np.isfinite(work["z_dem"])].copy()
    # Keep raw depth (can be negative) for variogram/kriging stability and structure.
    # Final raster depth is clipped to non-negative after prediction.
    work["depth"] = (work["z_dem"] - work["z"])

    # Merge near-duplicate XY after depth computation to avoid singular kriging matrix.
    decs = int(max(0, int(duplicate_round_decimals)))
    work["_xk"] = np.round(work["x"].to_numpy(dtype=float), decs)
    work["_yk"] = np.round(work["y"].to_numpy(dtype=float), decs)
    dedup = (
        work.groupby(["_xk", "_yk"], as_index=False)
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            z=("z", "mean"),
            z_dem=("z_dem", "mean"),
            depth=("depth", "mean"),
            chainage_m=("chainage_m", "mean"),
            line_id=("line_id", "first"),
        )
    )
    dedup.drop(columns=["_xk", "_yk"], inplace=True, errors="ignore")

    stats = {
        "n_input": int(len(dec_df)),
        "n_valid_dem": int(len(work)),
        "n_after_dedupe": int(len(dedup)),
    }
    return dedup, stats


def _fit_exponential_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    pairs: int,
    bins_count: int,
    min_pairs_per_bin: int,
    percentile_max_h: float,
    random_seed: int,
) -> Dict[str, Any]:
    n = int(values.size)
    if n < 3:
        raise ValueError("Need at least 3 points for variogram fitting.")

    rng = np.random.default_rng(int(random_seed))
    pairs = int(max(1000, pairs))
    i = rng.integers(0, n, size=pairs)
    j = rng.integers(0, n, size=pairs)
    m = i != j
    i = i[m]
    j = j[m]
    if i.size < 10:
        raise ValueError("Not enough point pairs for variogram.")

    h = np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
    gamma = 0.5 * (values[i] - values[j]) ** 2

    h = h[np.isfinite(h)]
    gamma = gamma[np.isfinite(gamma)]
    if h.size < 10 or gamma.size < 10:
        raise ValueError("Invalid pair distances/semivariance values.")

    hmax = float(np.percentile(h, float(percentile_max_h)))
    if not np.isfinite(hmax) or hmax <= 0:
        hmax = float(np.nanmax(h)) if np.isfinite(np.nanmax(h)) else 1.0
    bins_count = int(max(5, bins_count))
    bins = np.linspace(0.0, hmax, bins_count)

    bin_centers = []
    gamma_means = []
    counts = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mm = (h >= b0) & (h < b1)
        c = int(np.count_nonzero(mm))
        if c >= int(min_pairs_per_bin):
            bin_centers.append(0.5 * (b0 + b1))
            gamma_means.append(float(np.mean(gamma[mm])))
            counts.append(c)

    bin_centers_arr = np.asarray(bin_centers, dtype=float)
    gamma_means_arr = np.asarray(gamma_means, dtype=float)

    vvar = float(np.nanvar(values))
    if not np.isfinite(vvar) or vvar <= 0:
        vvar = max(float(np.nanmean(np.abs(values))) ** 2, 1e-6)

    # Fallback when fit data is too sparse.
    if bin_centers_arr.size < 3:
        span = float(np.nanmax(h) - np.nanmin(h)) if h.size else 1.0
        rang = max(1e-3, span / 3.0)
        params = np.array([0.05 * vvar, 0.95 * vvar, rang], dtype=float)
        return {
            "params": params,
            "nugget": float(params[0]),
            "sill": float(params[1]),
            "range": float(params[2]),
            "bin_centers": bin_centers_arr,
            "gamma_means": gamma_means_arr,
            "bin_counts": np.asarray(counts, dtype=int),
            "fit_method": "fallback_sparse",
        }

    p0 = [0.05 * vvar, 0.95 * vvar, max(1e-3, float(np.max(bin_centers_arr)) / 3.0)]
    try:
        params, _ = curve_fit(
            exp_variogram,
            bin_centers_arr,
            gamma_means_arr,
            p0=p0,
            bounds=([0.0, 0.0, 1e-3], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
        fit_method = "curve_fit"
    except Exception:
        params = np.asarray(p0, dtype=float)
        fit_method = "fallback_p0"

    params = np.asarray(params, dtype=float)
    return {
        "params": params,
        "nugget": float(params[0]),
        "sill": float(params[1]),
        "range": float(params[2]),
        "bin_centers": bin_centers_arr,
        "gamma_means": gamma_means_arr,
        "bin_counts": np.asarray(counts, dtype=int),
        "fit_method": fit_method,
    }


def _build_ok_solver(coords: np.ndarray, values: np.ndarray, params: np.ndarray) -> Dict[str, Any]:
    n = int(values.size)
    if n < 2:
        raise ValueError("Need at least 2 points for ordinary kriging.")

    D = cdist(coords, coords)
    Gamma = exp_variogram(D, *params)
    np.fill_diagonal(Gamma, 0.0)

    K = np.empty((n + 1, n + 1), dtype=float)
    K[:n, :n] = Gamma
    K[:n, n] = 1.0
    K[n, :n] = 1.0
    K[n, n] = 0.0

    lu, piv = lu_factor(K)
    return {
        "coords": coords,
        "values": values,
        "params": np.asarray(params, dtype=float),
        "n": n,
        "lu": lu,
        "piv": piv,
    }


def _ok_predict(solver: Dict[str, Any], points_xy: np.ndarray, chunk: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    coords = solver["coords"]
    values = solver["values"]
    params = solver["params"]
    n = int(solver["n"])
    lu = solver["lu"]
    piv = solver["piv"]

    points_xy = np.asarray(points_xy, dtype=float)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must be shape (m, 2)")
    m = int(points_xy.shape[0])
    preds = np.empty(m, dtype=float)
    vars_ = np.empty(m, dtype=float)
    chunk = int(max(1, chunk))

    for s in range(0, m, chunk):
        e = min(m, s + chunk)
        P = points_xy[s:e]
        d = cdist(coords, P)                   # (n, chunk)
        g0 = exp_variogram(d, *params)         # (n, chunk)
        rhs = np.vstack([g0, np.ones((1, e - s), dtype=float)])  # (n+1, chunk)
        sol = lu_solve((lu, piv), rhs)
        w = sol[:n, :]
        mu = sol[n, :]
        preds[s:e] = w.T @ values
        vars_[s:e] = np.sum(w * g0, axis=0) + mu
    return preds, vars_


def _write_tif(path: str, arr: np.ndarray, *, transform, crs, nodata_out: float) -> None:
    ny, nx = arr.shape
    profile = {
        "driver": "GTiff",
        "height": int(ny),
        "width": int(nx),
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "crs": crs,
        "nodata": float(nodata_out),
        "compress": "deflate",
    }
    arr2 = np.where(np.isfinite(arr), arr, float(nodata_out)).astype("float32")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr2, 1)


def _read_raster_for_contour(tif_path: str) -> Tuple[np.ndarray, Any, Any]:
    _require_contour_deps()
    with rasterio.open(tif_path) as src:
        z = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            if np.isnan(nodata):
                z[~np.isfinite(z)] = np.nan
            else:
                z[np.isclose(z, float(nodata))] = np.nan
        transform = src.transform
        bounds = src.bounds
    return z, transform, bounds


def _grid_xy_from_transform(transform: Any, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    ny, nx = map(int, shape)
    xs = transform.c + (np.arange(nx) + 0.5) * transform.a
    ys = transform.f + (np.arange(ny) + 0.5) * transform.e
    return np.meshgrid(xs, ys)


def _contour_levels_from_interval(z: np.ndarray, interval: float) -> np.ndarray:
    return _contour_levels_from_interval_range(z, interval=interval)


def _contour_levels_from_interval_range(
    z: np.ndarray,
    *,
    interval: float,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> np.ndarray:
    interval = float(interval)
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("Contour interval must be > 0")
    if not np.any(np.isfinite(z)):
        return np.array([], dtype=float)
    data_min = float(np.nanmin(z))
    data_max = float(np.nanmax(z))
    if not (np.isfinite(data_min) and np.isfinite(data_max)):
        return np.array([], dtype=float)

    use_custom_range = (z_min is not None) or (z_max is not None)
    zmin = data_min if z_min is None else float(z_min)
    zmax = data_max if z_max is None else float(z_max)
    if not (np.isfinite(zmin) and np.isfinite(zmax)):
        return np.array([], dtype=float)
    if zmax < zmin:
        zmin, zmax = zmax, zmin
    if math.isclose(zmin, zmax, rel_tol=0.0, abs_tol=1e-12):
        return np.array([zmin], dtype=float)

    if use_custom_range:
        start = zmin
        stop = zmax
    else:
        start = math.floor(zmin / interval) * interval
        stop = math.ceil(zmax / interval) * interval
    levels = np.arange(start, stop + interval, interval, dtype=float)
    return levels[np.isfinite(levels)]


def render_contours_png_from_raster(
    tif_path: str,
    out_png: str,
    *,
    interval: float = 2.0,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    title: Optional[str] = None,
    cmap: Optional[str] = None,
    draw_raster: bool = False,
    alpha_raster: float = 0.65,
    linewidth: float = DEFAULT_UI4_CONTOUR_PARAMS["linewidth"],
    label_contours: bool = True,
    label_fontsize: int = DEFAULT_UI4_CONTOUR_PARAMS["label_fontsize"],
    contour_color: Optional[str] = None,
    figsize: Tuple[float, float] = DEFAULT_UI4_CONTOUR_PARAMS["figsize"],
    dpi: int = DEFAULT_UI4_CONTOUR_PARAMS["dpi"],
) -> Dict[str, Any]:
    """
    Render contour lines from a GeoTIFF and save to PNG.
    Ported/adapted from local `contours.py` (interactive) into backend-safe export.
    """
    _require_contour_deps()
    tif_path = os.path.abspath(str(tif_path or ""))
    out_png = os.path.abspath(str(out_png or ""))
    if not tif_path or not os.path.exists(tif_path):
        return {"ok": False, "error": f"Raster not found: {tif_path}"}

    z, transform, bounds = _read_raster_for_contour(tif_path)
    if not np.any(np.isfinite(z)):
        return {"ok": False, "error": "No valid contour levels"}

    interval_val = float(interval)
    if not np.isfinite(interval_val) or interval_val <= 0:
        return {"ok": False, "error": "Contour interval must be > 0"}

    # Match the standalone contour script logic: build X/Y directly from raster transform.
    ny, nx = z.shape
    xs = transform.c + (np.arange(nx) + 0.5) * transform.a
    ys = transform.f + (np.arange(ny) + 0.5) * transform.e  # transform.e is usually negative
    X, Y = np.meshgrid(xs, ys)

    data_zmin = float(np.nanmin(z))
    data_zmax = float(np.nanmax(z))
    use_auto_min = z_min is None
    use_auto_max = z_max is None
    level_zmin = data_zmin if use_auto_min else float(z_min)
    level_zmax = data_zmax if use_auto_max else float(z_max)
    if not (np.isfinite(level_zmin) and np.isfinite(level_zmax)):
        return {"ok": False, "error": "Invalid contour z-range."}
    if (z_min is not None) and (z_max is not None) and (float(z_max) <= float(z_min)):
        return {"ok": False, "error": "Manual contour range invalid: z_max must be > z_min."}
    if level_zmax < level_zmin:
        level_zmin, level_zmax = level_zmax, level_zmin

    if math.isclose(level_zmin, level_zmax, rel_tol=0.0, abs_tol=1e-12):
        levels = np.array([level_zmin], dtype=float)
    else:
        level_start = (
            np.floor(level_zmin / interval_val) * interval_val if use_auto_min else level_zmin
        )
        level_stop = (
            np.ceil(level_zmax / interval_val) * interval_val if use_auto_max else level_zmax
        )
        levels = np.arange(level_start, level_stop + interval_val, interval_val, dtype=float)
        levels = levels[np.isfinite(levels)]
    if levels.size < 2:
        return {"ok": False, "error": "No valid contour levels"}

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig = plt.figure(figsize=figsize, dpi=int(dpi))
    ax = fig.add_subplot(111)
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    z_plot = np.ma.masked_invalid(z)
    img = None
    if draw_raster:
        imshow_kwargs = {
            "extent": extent,
            "origin": "upper",
            "alpha": float(alpha_raster),
            "cmap": cmap,
        }
        if z_min is not None:
            imshow_kwargs["vmin"] = float(z_min)
        if z_max is not None:
            imshow_kwargs["vmax"] = float(z_max)
        img = ax.imshow(z_plot, **imshow_kwargs)
        fig.colorbar(img, ax=ax, shrink=0.85)

    contour_kwargs: Dict[str, Any] = {"levels": levels, "linewidths": float(linewidth)}
    if contour_color:
        contour_kwargs["colors"] = contour_color
    cs = ax.contour(X, Y, z_plot, **contour_kwargs)
    if label_contours:
        try:
            ax.clabel(cs, inline=True, fontsize=int(label_fontsize))
        except Exception:
            pass

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title or f"Contours: {os.path.basename(tif_path)} (interval={interval_val:g} m)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    return {
        "ok": True,
        "tif_path": tif_path,
        "png_path": out_png,
        "interval": interval_val,
        "z_min": (float(z_min) if z_min is not None else None),
        "z_max": (float(z_max) if z_max is not None else None),
        "levels_count": int(levels.size),
        "levels_min": float(levels.min()) if levels.size else None,
        "levels_max": float(levels.max()) if levels.size else None,
        "shape": {"ny": int(z.shape[0]), "nx": int(z.shape[1])},
    }


def render_ui4_contours_for_run(
    run_dir: str,
    *,
    out_subdir: str = "ui4",
    surface_interval_m: float = DEFAULT_UI4_CONTOUR_PARAMS["surface_interval_m"],
    depth_interval_m: float = DEFAULT_UI4_CONTOUR_PARAMS["depth_interval_m"],
    surface_z_min: Optional[float] = None,
    surface_z_max: Optional[float] = None,
    depth_z_min: Optional[float] = None,
    depth_z_max: Optional[float] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Generate contour PNGs from UI4 rasters under <run_dir>/ui4.
    Expected rasters are produced by `run_ui4_kriging_for_run(...)`.
    """
    _require_contour_deps()
    run_dir = os.path.abspath(str(run_dir or ""))
    ui4_dir = os.path.join(run_dir, str(out_subdir))
    shared = _read_json_if_exists(os.path.join(ui4_dir, "ui_shared_data.json"))

    surface_tif = _pick_existing([
        shared.get("ui4_surface_masked_tif", ""),
        os.path.join(ui4_dir, "slip_surface_kriging_masked.tif"),
        shared.get("ui4_surface_tif", ""),
        *glob.glob(os.path.join(ui4_dir, "slip_surface_kriging_*_masked.tif")),
        *glob.glob(os.path.join(ui4_dir, "slip_surface_kriging_*.tif")),
    ])
    depth_tif = _pick_existing([
        shared.get("ui4_depth_masked_tif", ""),
        os.path.join(ui4_dir, "slip_depth_kriging_masked.tif"),
        shared.get("ui4_depth_tif", ""),
        *glob.glob(os.path.join(ui4_dir, "slip_depth_kriging_*_masked.tif")),
        *glob.glob(os.path.join(ui4_dir, "slip_depth_kriging_*.tif")),
    ])

    if not surface_tif and not depth_tif:
        return {"ok": False, "error": f"No UI4 kriging rasters found in {ui4_dir}"}

    preview_dir = os.path.join(ui4_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)

    outputs: Dict[str, Any] = {"ok": True, "ui4_dir": ui4_dir, "preview_dir": preview_dir, "items": {}}

    if surface_tif:
        out_png = os.path.join(preview_dir, "contours_surface.png")
        res = render_contours_png_from_raster(
            surface_tif,
            out_png,
            interval=float(surface_interval_m),
            z_min=surface_z_min,
            z_max=surface_z_max,
            title=None,
            draw_raster=False,
            cmap="terrain",
        )
        outputs["items"]["surface"] = res
        _log(log_fn, f"[UI4] Surface contour: {res.get('png_path') if res.get('ok') else res.get('error')}")

    if depth_tif:
        out_png = os.path.join(preview_dir, "contours_depth.png")
        res = render_contours_png_from_raster(
            depth_tif,
            out_png,
            interval=float(depth_interval_m),
            z_min=depth_z_min,
            z_max=depth_z_max,
            title=None,
            draw_raster=False,
            cmap="viridis",
        )
        outputs["items"]["depth"] = res
        _log(log_fn, f"[UI4] Depth contour: {res.get('png_path') if res.get('ok') else res.get('error')}")

    summary_path = os.path.join(preview_dir, "ui4_contours_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    outputs["summary_json"] = summary_path
    return outputs


def run_ui4_kriging_from_paths(
    *,
    dem_path: str,
    curve_json_paths: List[str],
    out_dir: str,
    mask_tif_path: Optional[str] = None,
    chainage_step_m: float = DEFAULT_UI4_PARAMS["chainage_step_m"],
    grid_res_m: float = DEFAULT_UI4_PARAMS["grid_res_m"],
    buffer_m: float = DEFAULT_UI4_PARAMS["buffer_m"],
    nodata_out: float = DEFAULT_UI4_PARAMS["nodata_out"],
    variogram_pairs: int = DEFAULT_UI4_PARAMS["variogram_pairs"],
    variogram_bins: int = DEFAULT_UI4_PARAMS["variogram_bins"],
    variogram_min_pairs_per_bin: int = DEFAULT_UI4_PARAMS["variogram_min_pairs_per_bin"],
    variogram_percentile_max_h: float = DEFAULT_UI4_PARAMS["variogram_percentile_max_h"],
    random_seed: int = DEFAULT_UI4_PARAMS["random_seed"],
    predict_chunk_size: int = DEFAULT_UI4_PARAMS["predict_chunk_size"],
    duplicate_round_decimals: int = DEFAULT_UI4_PARAMS["duplicate_round_decimals"],
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    _require_ui4_runtime_deps()
    t0 = time.time()
    dem_path = os.path.abspath(str(dem_path or ""))
    out_dir = os.path.abspath(str(out_dir or ""))
    curve_json_paths = [os.path.abspath(str(p)) for p in (curve_json_paths or []) if str(p).strip()]
    mask_tif_path = os.path.abspath(str(mask_tif_path or "")) if str(mask_tif_path or "").strip() else ""

    if not dem_path or not os.path.exists(dem_path):
        return {"ok": False, "error": f"DEM not found: {dem_path}", "paths": {}}
    if not curve_json_paths:
        return {"ok": False, "error": "No curve JSON inputs.", "paths": {}}
    if mask_tif_path and not os.path.exists(mask_tif_path):
        _log(log_fn, f"[UI4] Mask not found (optional): {mask_tif_path}")
        mask_tif_path = ""

    os.makedirs(out_dir, exist_ok=True)
    _log(log_fn, f"[UI4] DEM: {dem_path}")
    _log(log_fn, f"[UI4] Curve JSONs: {len(curve_json_paths)}")

    all_df, dec_df = load_ui4_curve_points(curve_json_paths, chainage_step_m=chainage_step_m, log_fn=log_fn)
    if dec_df.empty:
        return {"ok": False, "error": "No valid curve points after parsing/decimation.", "paths": {}}

    _log(log_fn, f"[UI4] Total curve points: raw={len(all_df)} decimated={len(dec_df)}")
    krig_df, dem_sample_stats = _sample_dem_and_compute_depth(
        dem_path,
        dec_df,
        duplicate_round_decimals=int(duplicate_round_decimals),
    )
    if krig_df.empty or len(krig_df) < 3:
        return {
            "ok": False,
            "error": "Not enough valid kriging points after DEM sampling/deduplication.",
            "stats": dem_sample_stats,
            "paths": {},
        }

    coords = krig_df[["x", "y"]].to_numpy(dtype=float)
    values = krig_df["depth"].to_numpy(dtype=float)
    n = int(values.size)
    _log(log_fn, f"[UI4] Kriging points: n={n}")

    variogram = _fit_exponential_variogram(
        coords,
        values,
        pairs=int(variogram_pairs),
        bins_count=int(variogram_bins),
        min_pairs_per_bin=int(variogram_min_pairs_per_bin),
        percentile_max_h=float(variogram_percentile_max_h),
        random_seed=int(random_seed),
    )
    params = np.asarray(variogram["params"], dtype=float)
    _log(
        log_fn,
        "[UI4] Variogram params: "
        f"nugget={params[0]:.6g}, sill={params[1]:.6g}, range={params[2]:.6g} "
        f"({variogram.get('fit_method')})",
    )

    solver = _build_ok_solver(coords, values, params)

    # Build grid over convex hull (+ buffer), predict inside hull
    hull = MultiPoint([Point(float(x), float(y)) for x, y in coords]).convex_hull.buffer(float(buffer_m))
    minx, miny, maxx, maxy = map(float, hull.bounds)

    xs = np.arange(minx, maxx + float(grid_res_m), float(grid_res_m))
    ys = np.arange(miny, maxy + float(grid_res_m), float(grid_res_m))
    nx = int(len(xs))
    ny = int(len(ys))
    if nx < 2 or ny < 2:
        return {"ok": False, "error": "UI4 interpolation grid is too small.", "paths": {}}

    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    ph = prep(hull)
    mask = np.array([ph.covers(Point(float(p[0]), float(p[1]))) for p in grid_points], dtype=bool)
    inside_pts = grid_points[mask]
    if inside_pts.size == 0:
        return {"ok": False, "error": "No grid points inside hull.", "paths": {}}

    _log(log_fn, f"[UI4] Grid size: {nx}x{ny} ({nx*ny:,} cells), inside={inside_pts.shape[0]:,}")

    pred_depth, pred_var = _ok_predict(solver, inside_pts, chunk=int(predict_chunk_size))
    pred_depth = np.clip(pred_depth, 0.0, None)
    pred_var = np.clip(pred_var, 0.0, None)

    with rasterio.open(dem_path) as src:
        dem_inside = np.array([v[0] for v in src.sample([tuple(p) for p in inside_pts])], dtype=float)
        dem_nodata = src.nodata
        dem_crs = src.crs
        if dem_nodata is not None:
            if np.isnan(dem_nodata):
                dem_inside[~np.isfinite(dem_inside)] = np.nan
            else:
                dem_inside[np.isclose(dem_inside, float(dem_nodata))] = np.nan

    valid_inside = np.isfinite(dem_inside)
    slip_z = np.full_like(dem_inside, np.nan, dtype=float)
    slip_z[valid_inside] = dem_inside[valid_inside] - pred_depth[valid_inside]
    pred_depth[~valid_inside] = np.nan
    pred_var[~valid_inside] = np.nan

    Z = np.full(nx * ny, np.nan, dtype=float)
    Z_depth = np.full(nx * ny, np.nan, dtype=float)
    Z_var = np.full(nx * ny, np.nan, dtype=float)

    Z[mask] = slip_z
    Z_depth[mask] = pred_depth
    Z_var[mask] = pred_var

    Z = Z.reshape((ny, nx))
    Z_depth = Z_depth.reshape((ny, nx))
    Z_var = Z_var.reshape((ny, nx))

    # flip vertically for GeoTIFF (row0 = maxy)
    Z_top = np.flipud(Z)
    Z_depth_top = np.flipud(Z_depth)
    Z_var_top = np.flipud(Z_var)

    transform_grid = from_origin(minx, maxy, float(grid_res_m), float(grid_res_m))
    res_tag = _format_res_tag(grid_res_m)

    surface_tif = os.path.join(out_dir, f"slip_surface_kriging_{res_tag}.tif")
    depth_tif = os.path.join(out_dir, f"slip_depth_kriging_{res_tag}.tif")
    var_tif = os.path.join(out_dir, f"slip_depth_kriging_variance_{res_tag}.tif")

    _write_tif(surface_tif, Z_top, transform=transform_grid, crs=dem_crs, nodata_out=float(nodata_out))
    _write_tif(depth_tif, Z_depth_top, transform=transform_grid, crs=dem_crs, nodata_out=float(nodata_out))
    _write_tif(var_tif, Z_var_top, transform=transform_grid, crs=dem_crs, nodata_out=float(nodata_out))

    surface_masked_tif = ""
    depth_masked_tif = ""
    var_masked_tif = ""
    mask_outputs: Dict[str, Any] = {}
    if mask_tif_path:
        surface_masked_tif = os.path.join(out_dir, "slip_surface_kriging_masked.tif")
        depth_masked_tif = os.path.join(out_dir, "slip_depth_kriging_masked.tif")
        var_masked_tif = os.path.join(out_dir, "slip_depth_kriging_variance_masked.tif")
        mask_outputs["surface"] = apply_mask_to_raster(
            surface_tif, mask_tif_path, surface_masked_tif, out_nodata=float(nodata_out), crop_to_mask_bbox=True
        )
        mask_outputs["depth"] = apply_mask_to_raster(
            depth_tif, mask_tif_path, depth_masked_tif, out_nodata=float(nodata_out), crop_to_mask_bbox=True
        )
        mask_outputs["variance"] = apply_mask_to_raster(
            var_tif, mask_tif_path, var_masked_tif, out_nodata=float(nodata_out), crop_to_mask_bbox=True
        )
        for k, v in mask_outputs.items():
            if not v.get("ok", False):
                _log(log_fn, f"[UI4] Masked {k} raster failed: {v.get('error')}")
            else:
                _log(log_fn, f"[UI4] Masked {k} raster: {v.get('out_tif')}")

    raster_stats = {
        "surface": _finite_raster_stats(Z_top),
        "depth": _finite_raster_stats(Z_depth_top),
        "variance": _finite_raster_stats(Z_var_top),
    }
    if mask_outputs.get("surface", {}).get("ok"):
        raster_stats["surface_masked"] = dict(mask_outputs["surface"].get("stats", {}))
    if mask_outputs.get("depth", {}).get("ok"):
        raster_stats["depth_masked"] = dict(mask_outputs["depth"].get("stats", {}))
    if mask_outputs.get("variance", {}).get("ok"):
        raster_stats["variance_masked"] = dict(mask_outputs["variance"].get("stats", {}))

    summary = {
        "ok": True,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {
            "dem_path": dem_path,
            "curve_json_paths": curve_json_paths,
            "curve_json_count": len(curve_json_paths),
            "mask_tif_path": (mask_tif_path or None),
        },
        "params": {
            "chainage_step_m": float(chainage_step_m),
            "grid_res_m": float(grid_res_m),
            "buffer_m": float(buffer_m),
            "nodata_out": float(nodata_out),
            "variogram_pairs": int(variogram_pairs),
            "variogram_bins": int(variogram_bins),
            "variogram_min_pairs_per_bin": int(variogram_min_pairs_per_bin),
            "variogram_percentile_max_h": float(variogram_percentile_max_h),
            "random_seed": int(random_seed),
            "predict_chunk_size": int(predict_chunk_size),
            "duplicate_round_decimals": int(duplicate_round_decimals),
        },
        "stats": {
            "curve_points_raw": int(len(all_df)),
            "curve_points_decimated": int(len(dec_df)),
            **dem_sample_stats,
            "kriging_points_n": int(n),
            "grid_nx": int(nx),
            "grid_ny": int(ny),
            "grid_total_cells": int(nx * ny),
            "grid_inside_cells": int(np.count_nonzero(mask)),
            "valid_pred_cells": int(np.count_nonzero(np.isfinite(Z_depth_top))),
            "surface_min_m": float(np.nanmin(Z_top)) if np.any(np.isfinite(Z_top)) else None,
            "surface_max_m": float(np.nanmax(Z_top)) if np.any(np.isfinite(Z_top)) else None,
            "depth_min_m": float(np.nanmin(Z_depth_top)) if np.any(np.isfinite(Z_depth_top)) else None,
            "depth_max_m": float(np.nanmax(Z_depth_top)) if np.any(np.isfinite(Z_depth_top)) else None,
            "variance_max": float(np.nanmax(Z_var_top)) if np.any(np.isfinite(Z_var_top)) else None,
            "runtime_sec": round(float(time.time() - t0), 3),
        },
        "raster_stats": raster_stats,
        "variogram": {
            "fit_method": variogram.get("fit_method"),
            "nugget": float(variogram["nugget"]),
            "sill": float(variogram["sill"]),
            "range": float(variogram["range"]),
            "bin_count_used": int(len(variogram.get("bin_centers", []))),
            "bin_centers": np.asarray(variogram.get("bin_centers", []), dtype=float).tolist(),
            "gamma_means": np.asarray(variogram.get("gamma_means", []), dtype=float).tolist(),
            "bin_counts": np.asarray(variogram.get("bin_counts", []), dtype=int).tolist(),
        },
        "grid": {
            "bounds": {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy},
            "crs": (str(dem_crs) if dem_crs is not None else None),
            "resolution_m": float(grid_res_m),
        },
        "outputs": {
            "ui4_dir": out_dir,
            "slip_surface_tif": surface_tif,
            "slip_depth_tif": depth_tif,
            "slip_depth_variance_tif": var_tif,
            "slip_surface_masked_tif": (surface_masked_tif if mask_outputs.get("surface", {}).get("ok") else None),
            "slip_depth_masked_tif": (depth_masked_tif if mask_outputs.get("depth", {}).get("ok") else None),
            "slip_depth_variance_masked_tif": (var_masked_tif if mask_outputs.get("variance", {}).get("ok") else None),
        },
    }

    summary_path = os.path.join(out_dir, "ui4_kriging_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["outputs"]["summary_json"] = summary_path

    # Lightweight shared-data file for UI4 tab / future chaining.
    shared_ui4_path = os.path.join(out_dir, "ui_shared_data.json")
    with open(shared_ui4_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ui4_surface_tif": surface_tif,
                "ui4_depth_tif": depth_tif,
                "ui4_variance_tif": var_tif,
                "ui4_surface_masked_tif": (surface_masked_tif if mask_outputs.get("surface", {}).get("ok") else ""),
                "ui4_depth_masked_tif": (depth_masked_tif if mask_outputs.get("depth", {}).get("ok") else ""),
                "ui4_variance_masked_tif": (var_masked_tif if mask_outputs.get("variance", {}).get("ok") else ""),
                "ui4_mask_tif": (mask_tif_path or ""),
                "ui4_summary_json": summary_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    summary["outputs"]["ui4_shared_data_json"] = shared_ui4_path

    _log(log_fn, f"[UI4] Wrote surface: {surface_tif}")
    _log(log_fn, f"[UI4] Wrote depth: {depth_tif}")
    _log(log_fn, f"[UI4] Wrote variance: {var_tif}")
    _log(log_fn, f"[UI4] Done in {summary['stats']['runtime_sec']} s")
    return summary


def run_ui4_kriging_for_run(
    run_dir: str,
    *,
    out_subdir: str = "ui4",
    log_fn: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run UI4 kriging for a run directory by auto-discovering DEM + UI3 NURBS curves.

    Example:
        run_ui4_kriging_for_run("output/Bo1/20260224_144614_5")
    """
    info = collect_ui4_run_inputs(run_dir)
    if not info.get("ok", False):
        return {"ok": False, "error": info.get("error", "Input discovery failed")}
    if not info.get("ready_for_ui4", False):
        return {
            "ok": False,
            "error": "UI4 inputs not ready",
            "missing_required": info.get("missing_required", []),
            "discovery": info,
        }

    dem_path = info["paths"]["dem"]
    curve_json_paths = info["paths"]["nurbs_curve_jsons"]
    mask_tif_path = info.get("paths", {}).get("mask_tif") or ""
    out_dir = os.path.join(os.path.abspath(str(run_dir)), str(out_subdir))

    _log(log_fn, f"[UI4] Running kriging for run_dir={os.path.abspath(str(run_dir))}")
    result = run_ui4_kriging_from_paths(
        dem_path=dem_path,
        curve_json_paths=curve_json_paths,
        out_dir=out_dir,
        mask_tif_path=mask_tif_path,
        log_fn=log_fn,
        **kwargs,
    )
    if isinstance(result, dict):
        result.setdefault("discovery", info)
    return result
