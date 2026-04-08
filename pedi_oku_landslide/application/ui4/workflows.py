"""UI4 main kriging orchestration: run_ui4_kriging_from_paths and run_ui4_kriging_for_run."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from pedi_oku_landslide.domain.ui4.types import (
    DEFAULT_UI4_PARAMS,
    MultiPoint,
    Point,
    PyKrigeOrdinaryKriging,
    _finite_raster_stats,
    _format_res_tag,
    _gaussian_smooth_nan,
    _log,
    _require_ui4_runtime_deps,
    _safe_float,
    _sigma_pixels_from_meters,
    _write_tif,
    from_origin,
    prep,
    rasterio,
)
from pedi_oku_landslide.domain.ui4.boundary import (
    _boundary_xy_from_mask_tif,
    _polygon_from_boundary_xy,
    apply_mask_to_raster,
    read_boundary_polygon_from_dxf,
)
from pedi_oku_landslide.application.ui4.inputs import collect_ui4_run_inputs
from pedi_oku_landslide.domain.ui4.kriging import (
    build_ok_solver,
    fit_exponential_variogram,
    ok_predict,
)
from pedi_oku_landslide.domain.ui4.surface import (
    load_ui4_curve_points,
    sample_dem_and_compute_depth,
)


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
    krig_df, dem_sample_stats = sample_dem_and_compute_depth(
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

    # Row-by-row polygon containment.
    ph = prep(interp_poly)
    mask_2d = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        row_pts = [Point(float(x), float(ys[j])) for x in xs]
        mask_2d[j, :] = [ph.contains(pt) for pt in row_pts]
    mask = mask_2d.ravel()
    if not np.any(mask):
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
        variogram = fit_exponential_variogram(
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
        solver = build_ok_solver(coords, values_surface, params)
        pred_surface, pred_var = ok_predict(solver, inside_pts, chunk=int(predict_chunk_size))
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
