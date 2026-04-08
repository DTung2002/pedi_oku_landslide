"""UI4 constants, default parameters, and shared utility functions."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import numpy as np


DEFAULT_UI4_PARAMS: Dict[str, Any] = {
    "chainage_step_m": 0.2,       # decimate along each curve
    "grid_res_m": 0.2,            # kriging grid resolution
    "buffer_m": 5.0,              # convex hull buffer
    "nodata_out": -9999.0,
    "use_pykrige": True,
    "variogram_model": "spherical",
    "smooth_sigma": 2.5,          # smoothing distance in meters
    "variogram_pairs": 20000,
    "variogram_bins": 20,
    "variogram_min_pairs_per_bin": 50,
    "variogram_percentile_max_h": 95.0,
    "random_seed": 0,
    "predict_chunk_size": 2000,
    "duplicate_round_decimals": 3,  # merge near-duplicate points after depth sampling
}

DEFAULT_UI4_CONTOUR_PARAMS: Dict[str, Any] = {
    "surface_interval_m": 1.0,
    "depth_interval_m": 1.0,
    "surface_smoothing_m": 2.5,
    "depth_smoothing_m": 2.0,
    "boundary_simplify_tolerance_m": 1.0,
    "major_interval_factor": 5.0,
    "figsize": (10, 8),
    "dpi": 200,
    "label_fontsize": 8,
    "linewidth": 1.0,
    "dem_overlay_linewidth": 0.8,
    "dem_overlay_color": "#dddddd",
    "slip_label_step_m": 5.0,
    "dem_label_step_m": 10.0,
    "panel_padding_m": 20.0,
}


# ── Lazy runtime-dependency checks ──────────────────────────────────────────

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore[assignment]

try:
    import rasterio
    from rasterio.features import shapes as rio_shapes
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling
    from rasterio.windows import Window, transform as window_transform
except Exception:
    rasterio = None  # type: ignore[assignment]
    rio_shapes = None  # type: ignore[assignment]
    from_origin = None  # type: ignore[assignment]
    reproject = None  # type: ignore[assignment]
    Resampling = None  # type: ignore[assignment]
    Window = None  # type: ignore[assignment]
    window_transform = None  # type: ignore[assignment]

try:
    from scipy.linalg import lu_factor, lu_solve
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import curve_fit
    from scipy.spatial.distance import cdist
except Exception:
    lu_factor = None  # type: ignore[assignment]
    lu_solve = None  # type: ignore[assignment]
    gaussian_filter = None  # type: ignore[assignment]
    curve_fit = None  # type: ignore[assignment]
    cdist = None  # type: ignore[assignment]

try:
    from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
    from shapely.ops import unary_union
    from shapely.prepared import prep
except Exception:
    MultiPoint = None  # type: ignore[assignment]
    MultiPolygon = None  # type: ignore[assignment]
    Point = None  # type: ignore[assignment]
    Polygon = None  # type: ignore[assignment]
    unary_union = None  # type: ignore[assignment]
    prep = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore[assignment]

try:
    from pykrige.ok import OrdinaryKriging as PyKrigeOrdinaryKriging
except Exception:
    PyKrigeOrdinaryKriging = None  # type: ignore[assignment]

try:
    import ezdxf
except Exception:
    ezdxf = None  # type: ignore[assignment]


# ── Shared utility functions ────────────────────────────────────────────────

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
    import json
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _pick_existing(candidates) -> str:
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


def _gaussian_smooth_nan(arr: np.ndarray, sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if gaussian_filter is None or not np.isfinite(sigma) or sigma <= 0:
        return np.asarray(arr, dtype=float)

    work = np.asarray(arr, dtype=float)
    if work.ndim != 2 or not np.any(np.isfinite(work)):
        return work.copy()
    nan_mask = ~np.isfinite(work)
    fill_value = float(np.nanmedian(work[np.isfinite(work)]))
    tmp = work.copy()
    tmp[nan_mask] = fill_value
    tmp = gaussian_filter(tmp, sigma=sigma)
    tmp[nan_mask] = np.nan
    return tmp


def _sigma_pixels_from_meters(smoothing_meters: float, grid_res_m: float) -> float:
    sm = float(smoothing_meters)
    gr = float(grid_res_m)
    if not np.isfinite(sm) or not np.isfinite(gr) or sm <= 0 or gr <= 0:
        return 0.0
    return float(sm / gr)


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
