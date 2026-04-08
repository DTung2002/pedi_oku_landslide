"""UI4 contour generation and rendering."""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .types import (
    DEFAULT_UI4_CONTOUR_PARAMS,
    _gaussian_smooth_nan,
    _log,
    _pick_existing,
    _read_json_if_exists,
    _require_contour_deps,
    _safe_float,
    _sigma_pixels_from_meters,
    plt,
    rasterio,
)
from .boundary import _boundary_xy_from_mask_tif, _simplify_boundary_xy
from pedi_oku_landslide.application.ui4.inputs import _find_ui4_mask_tif, collect_ui4_run_inputs


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
    ys = transform.f + (np.arange(ny) + 0.5) * transform.e
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
    major_interval = float(interval_val)
    levels_major = levels_minor

    # Define panel viewport once.
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

    # Optional overlays: boundary/profile lines.
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
