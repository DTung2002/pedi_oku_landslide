from __future__ import annotations

import csv
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


def _find_ui4_mask_tif(run_dir: str) -> str:
    ui1_dir = os.path.join(run_dir, "ui1")
    candidates = [
        os.path.join(ui1_dir, "landslide_mask.tif"),
        os.path.join(ui1_dir, "detect_mask.tif"),
        os.path.join(ui1_dir, "mask.tif"),
        os.path.join(ui1_dir, "mask_binary.tif"),
    ]
    return _pick_existing(candidates)


def _find_ui4_dxf_boundary(run_dir: str) -> str:
    """Discover a DXF boundary file in the run's input/ directory."""
    input_dir = os.path.join(run_dir, "input")
    candidates = [
        os.path.join(input_dir, "Boundary.dxf"),
        os.path.join(input_dir, "boundary.dxf"),
    ]
    found = _pick_existing(candidates)
    if found:
        return found
    # Fallback: first .dxf in input/
    dxf_files = sorted(glob.glob(os.path.join(input_dir, "*.dxf")))
    return os.path.abspath(dxf_files[0]) if dxf_files else ""


def read_boundary_polygon_from_dxf(
    dxf_path: str,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Optional[Any]:
    """
    Read boundary entities from a DXF file and return a shapely Polygon.

    Strategy:
      - Collect closed LWPOLYLINE / POLYLINE entities as polygons
      - Union them; keep the largest-area polygon as the main boundary
      - Returns None if ezdxf is not installed or no valid boundary found
    """
    if ezdxf is None:
        _log(log_fn, "[UI4] ezdxf not installed, cannot read DXF boundary")
        return None
    if Polygon is None or unary_union is None:
        _log(log_fn, "[UI4] shapely not available, cannot process DXF boundary")
        return None

    dxf_path = os.path.abspath(str(dxf_path or ""))
    if not dxf_path or not os.path.exists(dxf_path):
        return None

    try:
        doc = ezdxf.readfile(dxf_path)
    except Exception as e:
        _log(log_fn, f"[UI4] Failed to read DXF: {dxf_path} ({e})")
        return None

    msp = doc.modelspace()
    polys = []

    # LWPOLYLINE
    for e in msp.query("LWPOLYLINE"):
        pts = [(p[0], p[1]) for p in e.get_points("xy")]
        if len(pts) < 3:
            continue
        if e.closed:
            polys.append(Polygon(pts))

    # POLYLINE (2D/3D)
    for e in msp.query("POLYLINE"):
        pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
        if len(pts) < 3:
            continue
        if e.is_closed:
            polys.append(Polygon(pts))

    if not polys:
        _log(log_fn, f"[UI4] No closed polyline found in DXF: {dxf_path}")
        return None

    # Clean invalid polygons
    polys = [p.buffer(0) for p in polys if p.is_valid or p.buffer(0).is_valid]
    if not polys:
        _log(log_fn, f"[UI4] All DXF polygons are invalid: {dxf_path}")
        return None

    geom = unary_union(polys)
    if isinstance(geom, Polygon):
        _log(log_fn, f"[UI4] DXF boundary polygon loaded: area={geom.area:.2f}")
        return geom
    if MultiPolygon is not None and isinstance(geom, MultiPolygon):
        biggest = max(list(geom.geoms), key=lambda g: float(getattr(g, "area", 0.0)))
        _log(log_fn, f"[UI4] DXF boundary: multi-polygon, using largest (area={biggest.area:.2f})")
        return biggest

    _log(log_fn, f"[UI4] Unexpected DXF boundary geometry type: {type(geom)}")
    return None


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
    dxf_boundary = _find_ui4_dxf_boundary(run_dir)

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
            "dxf_boundary_path": dxf_boundary,
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

    valid_hg = np.isfinite(h) & np.isfinite(gamma)
    h = h[valid_hg]
    gamma = gamma[valid_hg]
    if h.size < 10:
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


def _ring_area_abs_xy(ring_xy: np.ndarray) -> float:
    if ring_xy.ndim != 2 or ring_xy.shape[0] < 3 or ring_xy.shape[1] < 2:
        return 0.0
    x = ring_xy[:, 0]
    y = ring_xy[:, 1]
    # Shoelace area; works for closed/unclosed ring.
    return float(abs(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))))


def _boundary_xy_from_mask_tif(mask_tif: str) -> Optional[np.ndarray]:
    mask_tif = os.path.abspath(str(mask_tif or ""))
    if not mask_tif or not os.path.exists(mask_tif) or rasterio is None:
        return None
    if rio_shapes is None:
        return None
    try:
        with rasterio.open(mask_tif) as src:
            arr = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                if np.isnan(nodata):
                    arr[~np.isfinite(arr)] = np.nan
                else:
                    arr[np.isclose(arr, float(nodata))] = np.nan
            mask_bool = np.isfinite(arr) & (arr > 0)
            if not np.any(mask_bool):
                return None

            best_xy = None
            best_area = 0.0
            for geom, val in rio_shapes(mask_bool.astype(np.uint8), mask=mask_bool, transform=src.transform):
                try:
                    if int(val) != 1:
                        continue
                    coords = (geom or {}).get("coordinates")
                    if not coords:
                        continue
                    ring = np.asarray(coords[0], dtype=float)
                    if ring.ndim != 2 or ring.shape[1] < 2 or ring.shape[0] < 3:
                        continue
                    area = _ring_area_abs_xy(ring[:, :2])
                    if area > best_area:
                        best_area = area
                        best_xy = ring[:, :2]
                except Exception:
                    continue
            return best_xy
    except Exception:
        return None


def _polygon_from_boundary_xy(boundary_xy: Optional[np.ndarray]) -> Optional[Any]:
    if boundary_xy is None or Polygon is None:
        return None
    try:
        ring = np.asarray(boundary_xy, dtype=float)
    except Exception:
        return None
    if ring.ndim != 2 or ring.shape[1] < 2 or ring.shape[0] < 3:
        return None
    if not np.allclose(ring[0, :2], ring[-1, :2]):
        ring = np.vstack([ring[:, :2], ring[0, :2]])
    else:
        ring = ring[:, :2]
    try:
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
    except Exception:
        return None
    if poly is None or poly.is_empty:
        return None
    if poly.geom_type == "MultiPolygon":
        geoms = list(getattr(poly, "geoms", []))
        if not geoms:
            return None
        poly = max(geoms, key=lambda g: float(getattr(g, "area", 0.0)))
    return poly


def _simplify_boundary_xy(boundary_xy: Optional[np.ndarray], tolerance_m: float) -> Optional[np.ndarray]:
    if boundary_xy is None:
        return None
    ring = np.asarray(boundary_xy, dtype=float)
    if ring.ndim != 2 or ring.shape[1] < 2 or ring.shape[0] < 3:
        return None
    if Polygon is None:
        return ring[:, :2]
    tol = _safe_float(tolerance_m)
    if not np.isfinite(tol) or tol <= 0:
        tol = 0.0
    try:
        poly = _polygon_from_boundary_xy(ring[:, :2])
        if poly is None:
            return ring[:, :2]
        if tol > 0:
            poly = poly.simplify(float(tol), preserve_topology=True)
        if poly is None or poly.is_empty:
            return ring[:, :2]
        return np.asarray(poly.exterior.coords, dtype=float)[:, :2]
    except Exception:
        return ring[:, :2]


def _line_label_short(line_id: str) -> str:
    txt = str(line_id or "").strip()
    if not txt:
        return ""
    return txt.split("__")[0]


def _load_section_profile_lines(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(os.path.abspath(str(run_dir or "")), "ui2", "sections.csv")
    if not os.path.exists(path):
        return []
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#e377c2", "#bcbd22", "#8c564b"]
    lines: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                x1 = _safe_float(row.get("x1"))
                y1 = _safe_float(row.get("y1"))
                x2 = _safe_float(row.get("x2"))
                y2 = _safe_float(row.get("y2"))
                if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
                    continue
                line_id = str(row.get("line_id") or row.get("name") or f"L{i + 1}").strip()
                lines.append(
                    {
                        "line_id": line_id,
                        "label": _line_label_short(line_id) or f"L{i + 1}",
                        "x": np.asarray([x1, x2], dtype=float),
                        "y": np.asarray([y1, y2], dtype=float),
                        "color": palette[i % len(palette)],
                    }
                )
    except Exception:
        return []
    return lines


def _load_curve_profile_lines(curve_json_paths: List[str]) -> List[Dict[str, Any]]:
    if not curve_json_paths:
        return []
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#e377c2", "#bcbd22", "#8c564b"]
    lines: List[Dict[str, Any]] = []
    for i, p in enumerate(curve_json_paths):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            pts = data.get("points") or []
            if not isinstance(pts, list):
                continue
            xs: List[float] = []
            ys: List[float] = []
            for pt in pts:
                if not isinstance(pt, dict):
                    continue
                x = _safe_float(pt.get("x"))
                y = _safe_float(pt.get("y"))
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(float(x))
                    ys.append(float(y))
            if len(xs) < 2:
                continue
            line_id = str(data.get("line_id") or os.path.splitext(os.path.basename(p))[0]).strip()
            lines.append(
                {
                    "line_id": line_id,
                    "label": _line_label_short(line_id) or f"L{i + 1}",
                    "x": np.asarray(xs, dtype=float),
                    "y": np.asarray(ys, dtype=float),
                    "color": palette[i % len(palette)],
                }
            )
        except Exception:
            continue
    return lines


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

    zmin = data_min if z_min is None else float(z_min)
    zmax = data_max if z_max is None else float(z_max)
    if not (np.isfinite(zmin) and np.isfinite(zmax)):
        return np.array([], dtype=float)
    if zmax < zmin:
        zmin, zmax = zmax, zmin
    if math.isclose(zmin, zmax, rel_tol=0.0, abs_tol=1e-12):
        return np.array([zmin], dtype=float)

    start = math.floor(zmin / interval) * interval
    stop = math.ceil(zmax / interval) * interval
    levels = np.arange(start, stop + interval, interval, dtype=float)
    return levels[np.isfinite(levels)]


def _label_levels_by_step(levels: np.ndarray, *, label_step: float, base_interval: float) -> np.ndarray:
    arr = np.asarray(levels, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr
    arr = np.unique(np.round(arr, 12))
    if arr.size == 0:
        return arr
    arr.sort()

    step = _safe_float(label_step)
    if not np.isfinite(step) or step <= 0:
        return arr

    tol = max(1e-6, abs(step) * 1e-6)
    mult = np.round(arr / step)
    picked = arr[np.isclose(arr, mult * step, rtol=0.0, atol=tol)]
    if picked.size > 0:
        return picked

    base = _safe_float(base_interval)
    if not np.isfinite(base) or base <= 0:
        return arr
    stride = max(1, int(round(step / base)))
    return arr[::stride]


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
    smooth_meters: float = DEFAULT_UI4_CONTOUR_PARAMS["surface_smoothing_m"],
    boundary_simplify_tolerance_m: float = DEFAULT_UI4_CONTOUR_PARAMS["boundary_simplify_tolerance_m"],
    major_interval_factor: float = DEFAULT_UI4_CONTOUR_PARAMS["major_interval_factor"],
    boundary_xy: Optional[np.ndarray] = None,
    dem_overlay_tif: Optional[str] = None,
    dem_overlay_interval: Optional[float] = None,
    dem_overlay_smooth_meters: Optional[float] = None,
    dem_overlay_color: str = str(DEFAULT_UI4_CONTOUR_PARAMS["dem_overlay_color"]),
    dem_overlay_linewidth: float = float(DEFAULT_UI4_CONTOUR_PARAMS["dem_overlay_linewidth"]),
    slip_label_step_m: float = float(DEFAULT_UI4_CONTOUR_PARAMS["slip_label_step_m"]),
    dem_label_step_m: float = float(DEFAULT_UI4_CONTOUR_PARAMS["dem_label_step_m"]),
    panel_padding_m: float = float(DEFAULT_UI4_CONTOUR_PARAMS["panel_padding_m"]),
    profile_lines: Optional[List[Dict[str, Any]]] = None,
    colorbar_label: Optional[str] = None,
    figsize: Tuple[float, float] = DEFAULT_UI4_CONTOUR_PARAMS["figsize"],
    dpi: int = DEFAULT_UI4_CONTOUR_PARAMS["dpi"],
) -> Dict[str, Any]:
    """
    Render contour map from a GeoTIFF and save to PNG.
    Style supports colored background + contour lines + optional overlays.
    """
    _require_contour_deps()
    tif_path = os.path.abspath(str(tif_path or ""))
    out_png = os.path.abspath(str(out_png or ""))
    if not tif_path or not os.path.exists(tif_path):
        return {"ok": False, "error": f"Raster not found: {tif_path}"}

    z, transform, _bounds = _read_raster_for_contour(tif_path)
    if not np.any(np.isfinite(z)):
        return {"ok": False, "error": "No valid contour levels"}

    interval_val = float(interval)
    if not np.isfinite(interval_val) or interval_val <= 0:
        return {"ok": False, "error": "Contour interval must be > 0"}

    # Build X/Y directly from raster transform.
    ny, nx = z.shape
    xs = transform.c + (np.arange(nx) + 0.5) * transform.a
    ys = transform.f + (np.arange(ny) + 0.5) * transform.e  # transform.e is usually negative
    X, Y = np.meshgrid(xs, ys)
    grid_res_m = abs(float(transform.a)) if np.isfinite(float(transform.a)) else 0.0
    if grid_res_m <= 0:
        grid_res_m = abs(float(transform.e)) if np.isfinite(float(transform.e)) else 0.0
    sigma_px = _sigma_pixels_from_meters(_safe_float(smooth_meters), grid_res_m)

    z_plot_raw = np.asarray(z, dtype=float)
    z_plot_smooth = _gaussian_smooth_nan(z_plot_raw, sigma_px) if sigma_px > 0 else z_plot_raw.copy()

    # Mask by boundary polygon (prevent contours outside boundary).
    if boundary_xy is not None:
        try:
            from matplotlib.path import Path
            bxy = _simplify_boundary_xy(boundary_xy, float(boundary_simplify_tolerance_m))
            if bxy is not None and bxy.ndim == 2 and bxy.shape[1] >= 2 and bxy.shape[0] >= 3:
                poly_path = Path(bxy[:, :2])
                inside = poly_path.contains_points(
                    np.c_[X.ravel(), Y.ravel()]
                ).reshape(z_plot_smooth.shape)
                z_plot_smooth[~inside] = np.nan
                boundary_xy = bxy  # keep simplified boundary for plotting
        except Exception:
            pass

    z_plot = np.ma.masked_invalid(z_plot_smooth)
    if not np.any(np.isfinite(z_plot_smooth)):
        return {"ok": False, "error": "No valid contour levels"}

    data_zmin = float(np.nanmin(z_plot_smooth))
    data_zmax = float(np.nanmax(z_plot_smooth))
    level_zmin = data_zmin if z_min is None else float(z_min)
    level_zmax = data_zmax if z_max is None else float(z_max)
    if not (np.isfinite(level_zmin) and np.isfinite(level_zmax)):
        return {"ok": False, "error": "Invalid contour z-range."}
    if (z_min is not None) and (z_max is not None) and (float(z_max) <= float(z_min)):
        return {"ok": False, "error": "Manual contour range invalid: z_max must be > z_min."}
    if level_zmax < level_zmin:
        level_zmin, level_zmax = level_zmax, level_zmin

    levels_minor = _contour_levels_from_interval_range(
        z_plot_smooth,
        interval=interval_val,
        z_min=level_zmin,
        z_max=level_zmax,
    )
    if levels_minor.size < 2:
        return {"ok": False, "error": "No valid contour levels"}
    # Keep API compatibility while rendering a single contour layer (no major/minor split).
    major_interval = float(interval_val)
    levels_major = levels_minor

    # Define panel viewport once (boundary bbox + padding, fallback to slip raster valid extent).
    view_bounds = None
    try:
        pad = _safe_float(panel_padding_m)
        if not np.isfinite(pad) or pad < 0:
            pad = 0.0
        bx = boundary_xy[:, 0] if (boundary_xy is not None and len(boundary_xy) >= 3) else None
        by = boundary_xy[:, 1] if (boundary_xy is not None and len(boundary_xy) >= 3) else None
        if bx is None or by is None:
            valid = np.isfinite(z_plot_smooth)
            if np.any(valid):
                xx = X[valid]
                yy = Y[valid]
                bx = np.array([np.nanmin(xx), np.nanmax(xx)], dtype=float)
                by = np.array([np.nanmin(yy), np.nanmax(yy)], dtype=float)
        if bx is not None and by is not None:
            xmin = float(np.nanmin(bx) - pad)
            xmax = float(np.nanmax(bx) + pad)
            ymin = float(np.nanmin(by) - pad)
            ymax = float(np.nanmax(by) + pad)
            if np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax) and xmax > xmin and ymax > ymin:
                view_bounds = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "padding_m": float(pad)}
    except Exception:
        view_bounds = None

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig = plt.figure(figsize=figsize, dpi=int(dpi))
    ax = fig.add_subplot(111)

    cf = None
    if draw_raster:
        cf = ax.contourf(X, Y, z_plot, levels=levels_minor, cmap=(cmap or "terrain"), alpha=float(alpha_raster))

    use_colored_lines = (contour_color is None) or (not str(contour_color).strip())
    contour_kwargs: Dict[str, Any] = {
        "levels": levels_minor,
        "linewidths": 1.0,
        "alpha": 0.9,
        "zorder": 4,
    }
    if use_colored_lines:
        cmap_name = str(cmap or "terrain")
        contour_kwargs["cmap"] = cmap_name
    else:
        line_color = str(contour_color).strip()
        contour_kwargs["colors"] = line_color

    cs = ax.contour(X, Y, z_plot, **contour_kwargs)
    if label_contours:
        try:
            slip_label_levels = _label_levels_by_step(
                levels_minor,
                label_step=float(slip_label_step_m),
                base_interval=float(interval_val),
            )
            if slip_label_levels.size > 0:
                ax.clabel(cs, levels=slip_label_levels, inline=True, fontsize=int(label_fontsize), fmt="%.1f")
        except Exception:
            pass

    dem_overlay_used = False
    dem_overlay_interval_used = None
    dem_overlay_smooth_used = None
    if dem_overlay_tif:
        dem_overlay_path = os.path.abspath(str(dem_overlay_tif or ""))
        if dem_overlay_path and os.path.exists(dem_overlay_path):
            try:
                dem_z, dem_transform, _ = _read_raster_for_contour(dem_overlay_path)
                if np.any(np.isfinite(dem_z)):
                    dem_X, dem_Y = _grid_xy_from_transform(dem_transform, dem_z.shape)
                    dem_grid_res_m = abs(float(dem_transform.a)) if np.isfinite(float(dem_transform.a)) else 0.0
                    if dem_grid_res_m <= 0:
                        dem_grid_res_m = abs(float(dem_transform.e)) if np.isfinite(float(dem_transform.e)) else 0.0
                    dem_smooth_m = (
                        _safe_float(dem_overlay_smooth_meters)
                        if dem_overlay_smooth_meters is not None
                        else _safe_float(smooth_meters)
                    )
                    dem_sigma_px = _sigma_pixels_from_meters(dem_smooth_m, dem_grid_res_m)
                    dem_smooth = _gaussian_smooth_nan(np.asarray(dem_z, dtype=float), dem_sigma_px) if dem_sigma_px > 0 else dem_z
                    # DEM overlay is not clipped by mask/boundary; limit by panel viewport only.
                    dem_X_plot = dem_X
                    dem_Y_plot = dem_Y
                    dem_Z_plot = np.asarray(dem_smooth, dtype=float)
                    if isinstance(view_bounds, dict):
                        x_lo = float(min(view_bounds["xmin"], view_bounds["xmax"]))
                        x_hi = float(max(view_bounds["xmin"], view_bounds["xmax"]))
                        y_lo = float(min(view_bounds["ymin"], view_bounds["ymax"]))
                        y_hi = float(max(view_bounds["ymin"], view_bounds["ymax"]))
                        xv = np.asarray(dem_X[0, :], dtype=float)
                        yv = np.asarray(dem_Y[:, 0], dtype=float)
                        col_idx = np.where(np.isfinite(xv) & (xv >= x_lo) & (xv <= x_hi))[0]
                        row_idx = np.where(np.isfinite(yv) & (yv >= y_lo) & (yv <= y_hi))[0]
                        if col_idx.size >= 2 and row_idx.size >= 2:
                            c0, c1 = int(col_idx.min()), int(col_idx.max())
                            r0, r1 = int(row_idx.min()), int(row_idx.max())
                            dem_X_plot = dem_X[r0:r1 + 1, c0:c1 + 1]
                            dem_Y_plot = dem_Y[r0:r1 + 1, c0:c1 + 1]
                            dem_Z_plot = dem_Z_plot[r0:r1 + 1, c0:c1 + 1]
                    dem_interval = _safe_float(dem_overlay_interval) if dem_overlay_interval is not None else interval_val
                    if not np.isfinite(dem_interval) or dem_interval <= 0:
                        dem_interval = interval_val
                    dem_levels = _contour_levels_from_interval_range(
                        dem_Z_plot,
                        interval=float(dem_interval),
                    )
                    if dem_levels.size >= 2 and np.any(np.isfinite(dem_Z_plot)):
                        dem_cs = ax.contour(
                            dem_X_plot,
                            dem_Y_plot,
                            np.ma.masked_invalid(dem_Z_plot),
                            levels=dem_levels,
                            colors=str(dem_overlay_color or "#dddddd"),
                            linewidths=max(0.1, float(dem_overlay_linewidth)),
                            alpha=0.65,
                            zorder=2,
                        )
                        try:
                            dem_label_levels = _label_levels_by_step(
                                dem_levels,
                                label_step=float(dem_label_step_m),
                                base_interval=float(dem_interval),
                            )
                            if dem_label_levels.size == 0:
                                dem_label_levels = dem_levels
                            ax.clabel(
                                dem_cs,
                                levels=dem_label_levels,
                                inline=True,
                                fontsize=max(6, int(label_fontsize) - 1),
                                fmt="%.1f",
                                colors="#b8b8b8",
                            )
                        except Exception:
                            pass
                        dem_overlay_used = True
                        dem_overlay_interval_used = float(dem_interval)
                        dem_overlay_smooth_used = float(dem_smooth_m)
            except Exception:
                dem_overlay_used = False

    # Optional overlays: keep boundary/profile lines visible above contours.
    if boundary_xy is not None:
        try:
            bxy = _simplify_boundary_xy(boundary_xy, float(boundary_simplify_tolerance_m))
            if bxy is not None and bxy.ndim == 2 and bxy.shape[1] >= 2 and bxy.shape[0] >= 2:
                ax.plot(
                    bxy[:, 0],
                    bxy[:, 1],
                    color="#1f77b4",
                    linewidth=1.2,
                    alpha=0.95,
                    zorder=6,
                )
        except Exception:
            pass

    if profile_lines:
        for i, line in enumerate(profile_lines):
            try:
                xs = np.asarray(line.get("x", []), dtype=float)
                ys = np.asarray(line.get("y", []), dtype=float)
                if xs.size < 2 or ys.size < 2:
                    continue
                c = str(line.get("color") or f"C{i % 10}")
                lbl = str(line.get("label") or line.get("line_id") or f"Profile {i + 1}")
                ax.plot(xs, ys, color=c, linewidth=1.3, alpha=0.9, zorder=5, label=lbl)
            except Exception:
                continue

    if cf is not None:
        cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
        cb_label = str(colorbar_label or "").strip()
        if not cb_label:
            name = os.path.basename(tif_path).lower()
            cb_label = "Độ sâu (m)" if "depth" in name else "Cao độ bề mặt (m)"
        cbar.set_label(cb_label)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        title
        or (
            "Bản đồ Địa hình Tổng hợp: Mặt trượt và Tự nhiên xung quanh\n"
            f"(Đường đồng mức cách nhau {interval_val:g}m)"
        )
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    if isinstance(view_bounds, dict):
        ax.set_xlim(float(view_bounds["xmin"]), float(view_bounds["xmax"]))
        ax.set_ylim(float(view_bounds["ymin"]), float(view_bounds["ymax"]))

    # Show legend only when overlays are present.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")
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
        "levels_count": int(levels_minor.size),
        "levels_min": float(levels_minor.min()) if levels_minor.size else None,
        "levels_max": float(levels_minor.max()) if levels_minor.size else None,
        "major_levels_count": int(levels_major.size),
        "major_interval": major_interval,
        "smooth_meters": float(_safe_float(smooth_meters)),
        "smooth_sigma_pixels": float(sigma_px),
        "dem_overlay": {
            "used": bool(dem_overlay_used),
            "tif_path": (os.path.abspath(str(dem_overlay_tif)) if dem_overlay_tif else ""),
            "interval": dem_overlay_interval_used,
            "smooth_meters": dem_overlay_smooth_used,
            "linewidth": float(dem_overlay_linewidth),
            "color": str(dem_overlay_color or "#dddddd"),
        },
        "view_bounds": view_bounds,
        "shape": {"ny": int(z.shape[0]), "nx": int(z.shape[1])},
    }


def render_ui4_contours_for_run(
    run_dir: str,
    *,
    out_subdir: str = "ui4",
    surface_interval_m: float = DEFAULT_UI4_CONTOUR_PARAMS["surface_interval_m"],
    depth_interval_m: float = DEFAULT_UI4_CONTOUR_PARAMS["depth_interval_m"],
    surface_smoothing_m: float = DEFAULT_UI4_CONTOUR_PARAMS["surface_smoothing_m"],
    depth_smoothing_m: float = DEFAULT_UI4_CONTOUR_PARAMS["depth_smoothing_m"],
    boundary_simplify_tolerance_m: float = DEFAULT_UI4_CONTOUR_PARAMS["boundary_simplify_tolerance_m"],
    major_interval_factor: float = DEFAULT_UI4_CONTOUR_PARAMS["major_interval_factor"],
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
    run_inputs = collect_ui4_run_inputs(run_dir)
    run_paths = run_inputs.get("paths", {}) if isinstance(run_inputs, dict) else {}

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

    # Overlays for style parity with reference map.
    mask_tif = _pick_existing(
        [
            shared.get("ui4_mask_tif", ""),
            run_paths.get("mask_tif", ""),
            _find_ui4_mask_tif(run_dir),
        ]
    )
    boundary_xy = _boundary_xy_from_mask_tif(mask_tif)
    dem_tif = _pick_existing([run_paths.get("dem", "")])

    if surface_tif:
        out_png = os.path.join(preview_dir, "contours_surface.png")
        res = render_contours_png_from_raster(
            surface_tif,
            out_png,
            interval=float(surface_interval_m),
            z_min=surface_z_min,
            z_max=surface_z_max,
            title="Slip Surface Contours",
            draw_raster=False,
            label_contours=True,
            linewidth=1.0,
            cmap="terrain",
            contour_color=None,
            smooth_meters=float(surface_smoothing_m),
            boundary_simplify_tolerance_m=float(boundary_simplify_tolerance_m),
            major_interval_factor=float(major_interval_factor),
            boundary_xy=boundary_xy,
            dem_overlay_tif=dem_tif,
            dem_overlay_interval=float(surface_interval_m),
            dem_overlay_smooth_meters=float(surface_smoothing_m),
            dem_overlay_color=str(DEFAULT_UI4_CONTOUR_PARAMS["dem_overlay_color"]),
            dem_overlay_linewidth=float(DEFAULT_UI4_CONTOUR_PARAMS["dem_overlay_linewidth"]),
            panel_padding_m=float(DEFAULT_UI4_CONTOUR_PARAMS["panel_padding_m"]),
            profile_lines=None,
            colorbar_label="Cao độ bề mặt (m)",
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
            draw_raster=False,
            title="Slip Depth Contours",
            label_contours=True,
            linewidth=1.0,
            cmap="viridis",
            contour_color=None,
            smooth_meters=float(depth_smoothing_m),
            boundary_simplify_tolerance_m=float(boundary_simplify_tolerance_m),
            major_interval_factor=float(major_interval_factor),
            boundary_xy=boundary_xy,
            dem_overlay_tif=dem_tif,
            dem_overlay_interval=float(surface_interval_m),
            dem_overlay_smooth_meters=float(surface_smoothing_m),
            dem_overlay_color=str(DEFAULT_UI4_CONTOUR_PARAMS["dem_overlay_color"]),
            dem_overlay_linewidth=float(DEFAULT_UI4_CONTOUR_PARAMS["dem_overlay_linewidth"]),
            panel_padding_m=float(DEFAULT_UI4_CONTOUR_PARAMS["panel_padding_m"]),
            profile_lines=None,
            colorbar_label="Độ sâu (m)",
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
    dxf_boundary_path: Optional[str] = None,
    chainage_step_m: float = DEFAULT_UI4_PARAMS["chainage_step_m"],
    grid_res_m: float = DEFAULT_UI4_PARAMS["grid_res_m"],
    buffer_m: float = DEFAULT_UI4_PARAMS["buffer_m"],
    nodata_out: float = DEFAULT_UI4_PARAMS["nodata_out"],
    use_pykrige: bool = DEFAULT_UI4_PARAMS["use_pykrige"],
    variogram_model: str = DEFAULT_UI4_PARAMS["variogram_model"],
    smooth_sigma: float = DEFAULT_UI4_PARAMS["smooth_sigma"],
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
    dxf_boundary_path = os.path.abspath(str(dxf_boundary_path or "")) if str(dxf_boundary_path or "").strip() else ""
    smooth_meters = _safe_float(smooth_sigma)
    if not np.isfinite(smooth_meters):
        smooth_meters = 0.0
    grid_res_safe = max(1e-6, abs(float(grid_res_m)))
    if grid_res_safe <= 0.3 and smooth_meters > 0 and smooth_meters < 2.0:
        _log(log_fn, f"[UI4] Overriding smooth_sigma {smooth_meters:.2f}m -> 2.5m (grid_res={grid_res_safe:.3f}m is fine, need stronger smoothing)")
        smooth_meters = 2.5
    smooth_sigma_px = _sigma_pixels_from_meters(smooth_meters, grid_res_safe)

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
    values_surface = krig_df["z"].to_numpy(dtype=float)
    n = int(values_surface.size)
    _log(log_fn, f"[UI4] Kriging points: n={n}")

    # Determine interpolation domain: DXF boundary > mask TIF > convex hull buffer.
    interp_poly = None
    interp_domain = "convex_hull_buffer"
    if dxf_boundary_path and os.path.exists(dxf_boundary_path):
        interp_poly = read_boundary_polygon_from_dxf(dxf_boundary_path, log_fn=log_fn)
        if interp_poly is not None:
            interp_domain = "dxf_boundary"
            _log(log_fn, f"[UI4] Using DXF boundary: {dxf_boundary_path}")
    if interp_poly is None and mask_tif_path:
        interp_poly = _polygon_from_boundary_xy(_boundary_xy_from_mask_tif(mask_tif_path))
        if interp_poly is not None:
            interp_domain = "mask_polygon"
    if interp_poly is None:
        interp_poly = MultiPoint([Point(float(x), float(y)) for x, y in coords]).convex_hull.buffer(float(buffer_m))
        interp_domain = "convex_hull_buffer"
    minx, miny, maxx, maxy = map(float, interp_poly.bounds)

    xs = np.arange(minx, maxx + float(grid_res_m), float(grid_res_m))
    ys = np.arange(miny, maxy + float(grid_res_m), float(grid_res_m))
    nx = int(len(xs))
    ny = int(len(ys))
    if nx < 2 or ny < 2:
        return {"ok": False, "error": "UI4 interpolation grid is too small.", "paths": {}}

    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Row-by-row polygon containment (much faster than point-by-point).
    ph = prep(interp_poly)
    mask_2d = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        row_pts = [Point(float(x), float(ys[j])) for x in xs]
        mask_2d[j, :] = [ph.contains(pt) for pt in row_pts]
    mask = mask_2d.ravel()
    if not np.any(mask):
        # Fallback: try covers() for edge cases.
        for j in range(ny):
            row_pts = [Point(float(x), float(ys[j])) for x in xs]
            mask_2d[j, :] = [ph.covers(pt) for pt in row_pts]
        mask = mask_2d.ravel()
    inside_pts = grid_points[mask]
    if inside_pts.size == 0:
        return {"ok": False, "error": "No grid points inside interpolation domain.", "paths": {}}

    _log(
        log_fn,
        f"[UI4] Grid size: {nx}x{ny} ({nx*ny:,} cells), inside={inside_pts.shape[0]:,}, domain={interp_domain}",
    )

    Z_surface = np.full((ny, nx), np.nan, dtype=float)
    Z_var = np.full((ny, nx), np.nan, dtype=float)
    variogram: Dict[str, Any] = {
        "fit_method": None,
        "nugget": None,
        "sill": None,
        "range": None,
        "bin_centers": np.asarray([], dtype=float),
        "gamma_means": np.asarray([], dtype=float),
        "bin_counts": np.asarray([], dtype=int),
    }

    use_pykrige_effective = bool(use_pykrige) and (PyKrigeOrdinaryKriging is not None)
    if use_pykrige_effective:
        try:
            model_name = str(variogram_model or "spherical").strip().lower() or "spherical"
            _log(log_fn, f"[UI4] Kriging engine: pykrige ({model_name})")
            ok = PyKrigeOrdinaryKriging(
                coords[:, 0],
                coords[:, 1],
                values_surface,
                variogram_model=model_name,
                verbose=False,
                enable_plotting=False,
            )
            zgrid, ss = ok.execute("grid", xs, ys)
            Z_surface = np.asarray(zgrid, dtype=float).reshape((ny, nx))
            Z_var = np.asarray(ss, dtype=float).reshape((ny, nx))
            variogram["fit_method"] = f"pykrige_{model_name}"
        except Exception as e:
            _log(log_fn, f"[UI4] pykrige failed, fallback to internal OK solver: {e}")
            use_pykrige_effective = False

    if not use_pykrige_effective:
        variogram = _fit_exponential_variogram(
            coords,
            values_surface,
            pairs=int(variogram_pairs),
            bins_count=int(variogram_bins),
            min_pairs_per_bin=int(variogram_min_pairs_per_bin),
            percentile_max_h=float(variogram_percentile_max_h),
            random_seed=int(random_seed),
        )
        params = np.asarray(variogram["params"], dtype=float)
        _log(
            log_fn,
            "[UI4] Kriging engine: internal OK (exponential variogram) | "
            f"nugget={params[0]:.6g}, sill={params[1]:.6g}, range={params[2]:.6g} "
            f"({variogram.get('fit_method')})",
        )
        solver = _build_ok_solver(coords, values_surface, params)
        pred_surface, pred_var = _ok_predict(solver, inside_pts, chunk=int(predict_chunk_size))
        pred_var = np.clip(pred_var, 0.0, None)
        flat_surface = np.full(nx * ny, np.nan, dtype=float)
        flat_var = np.full(nx * ny, np.nan, dtype=float)
        flat_surface[mask] = pred_surface
        flat_var[mask] = pred_var
        Z_surface = flat_surface.reshape((ny, nx))
        Z_var = flat_var.reshape((ny, nx))

    mask_2d = mask.reshape((ny, nx))
    Z_surface[~mask_2d] = np.nan
    Z_var[~mask_2d] = np.nan

    if smooth_sigma_px > 0:
        Z_surface = _gaussian_smooth_nan(Z_surface, smooth_sigma_px)
        Z_surface[~mask_2d] = np.nan

    with rasterio.open(dem_path) as src:
        dem_inside = np.array([v[0] for v in src.sample([tuple(p) for p in inside_pts])], dtype=float)
        dem_nodata = src.nodata
        dem_crs = src.crs
        if dem_nodata is not None:
            if np.isnan(dem_nodata):
                dem_inside[~np.isfinite(dem_inside)] = np.nan
            else:
                dem_inside[np.isclose(dem_inside, float(dem_nodata))] = np.nan

    Z_flat = np.full(nx * ny, np.nan, dtype=float)
    Z_depth_flat = np.full(nx * ny, np.nan, dtype=float)
    Z_var_flat = np.full(nx * ny, np.nan, dtype=float)
    surface_inside = Z_surface.ravel()[mask]
    var_inside = Z_var.ravel()[mask]
    valid_inside = np.isfinite(dem_inside) & np.isfinite(surface_inside)
    depth_inside = np.full_like(surface_inside, np.nan, dtype=float)
    depth_inside[valid_inside] = dem_inside[valid_inside] - surface_inside[valid_inside]
    depth_inside = np.where(np.isfinite(depth_inside), np.clip(depth_inside, 0.0, None), np.nan)

    Z_flat[mask] = np.where(valid_inside, surface_inside, np.nan)
    Z_depth_flat[mask] = depth_inside
    Z_var_flat[mask] = np.where(valid_inside, np.clip(var_inside, 0.0, None), np.nan)

    Z = Z_flat.reshape((ny, nx))
    Z_depth = Z_depth_flat.reshape((ny, nx))
    Z_var = Z_var_flat.reshape((ny, nx))

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
            "use_pykrige": bool(use_pykrige),
            "variogram_model": str(variogram_model),
            "smooth_sigma": float(smooth_meters),
            "smooth_sigma_pixels": float(smooth_sigma_px),
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
            "nugget": (
                float(variogram["nugget"]) if variogram.get("nugget") is not None else None
            ),
            "sill": (
                float(variogram["sill"]) if variogram.get("sill") is not None else None
            ),
            "range": (
                float(variogram["range"]) if variogram.get("range") is not None else None
            ),
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
    dxf_boundary_path = info.get("paths", {}).get("dxf_boundary_path") or ""
    out_dir = os.path.join(os.path.abspath(str(run_dir)), str(out_subdir))

    _log(log_fn, f"[UI4] Running kriging for run_dir={os.path.abspath(str(run_dir))}")
    result = run_ui4_kriging_from_paths(
        dem_path=dem_path,
        curve_json_paths=curve_json_paths,
        out_dir=out_dir,
        mask_tif_path=mask_tif_path,
        dxf_boundary_path=dxf_boundary_path,
        log_fn=log_fn,
        **kwargs,
    )
    if isinstance(result, dict):
        result.setdefault("discovery", info)
    return result
