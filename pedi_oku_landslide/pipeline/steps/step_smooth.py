# pedi_oku_landslide/pipeline/step_smooth.py
import os
import json
import numpy as np
import rasterio
from rasterio.transform import Affine
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt

from pedi_oku_landslide.services.session_store import AnalysisContext
from pedi_oku_landslide.core.analysis import smooth_mean
from pedi_oku_landslide.pipeline.ingest import resolve_run_input_path

def _radius_m_to_px(transform: Affine, radius_m: float) -> tuple[float, float]:
    xres = abs(float(transform.a))
    yres = abs(float(transform.e))
    if xres <= 0 or yres <= 0:
        raise ValueError("Invalid raster resolution: transform.a/transform.e must be non-zero.")
    return (float(radius_m) / yres, float(radius_m) / xres)  # (row, col)

def _save_preview_png(arr: np.ndarray, transform: Affine, out_png: str, title: str):
    # hillshade + axes + grid (giống ingest)
    med = float(np.nanmedian(arr)) if np.isfinite(np.nanmedian(arr)) else 0.0
    arr = np.where(np.isfinite(arr), arr, med)
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(arr, vert_exag=1.0)

    h, w = arr.shape
    x_min = transform.c
    x_max = x_min + transform.a * w
    y_max = transform.f
    y_min = y_max + transform.e * h
    extent = [x_min, x_max, y_min, y_max]

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(9, 9), dpi=120)
    plt.imshow(hs, cmap="gray", extent=extent, origin="upper")
    plt.title(title)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.9, color="red")
    plt.ticklabel_format(style="plain", useOffset=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def run_smooth(ctx: AnalysisContext, param_m: float = 2.0) -> dict:
    """
    Smooth BEFORE.asc, AFTER.asc, and AFTER DEM using Mean filter.
    Outputs:
      - GeoTIFFs: ui1/before_asc_smooth.tif, ui1/after_asc_smooth.tif
                  ui1/after_dem_smooth.tif
      - PNG previews: ui1/before_asc_smooth.png, ui1/after_asc_smooth.png,
                      ui1/after_dem_smooth.png
      - JSON: ui1/smooth_meta.json (method, param_m, radius_px)
    Returns dict with output paths.
    """
    # ---- read BEFORE/AFTER ASC + AFTER DEM
    before_path = resolve_run_input_path(ctx.run_dir, "before_asc")
    after_path = resolve_run_input_path(ctx.run_dir, "after_asc")
    after_dem_path = resolve_run_input_path(ctx.run_dir, "after_dem")
    if not (os.path.exists(before_path) and os.path.exists(after_path) and os.path.exists(after_dem_path)):
        raise FileNotFoundError("before.asc, after.asc, or after_dem.tif not found in run/input")

    with rasterio.open(before_path) as ds_b:
        b_arr = ds_b.read(1).astype("float32")
        b_meta = ds_b.meta.copy()
        b_transform = ds_b.transform
        b_crs = ds_b.crs

    with rasterio.open(after_path) as ds_a:
        a_arr = ds_a.read(1).astype("float32")
        a_meta = ds_a.meta.copy()
        a_transform = ds_a.transform
        a_crs = ds_a.crs

    with rasterio.open(after_dem_path) as ds_d:
        d_arr = ds_d.read(1).astype("float32")
        d_meta = ds_d.meta.copy()
        d_transform = ds_d.transform
        d_crs = ds_d.crs

    # optional: check CRS一致 (nếu khác, cảnh báo/lỗi)
    if (b_crs is not None) and (a_crs is not None) and (b_crs != a_crs):
        raise ValueError("BEFORE and AFTER have different CRS. Please reproject first.")
    if (a_crs is not None) and (d_crs is not None) and (a_crs != d_crs):
        raise ValueError("AFTER.asc and AFTER DEM.tif have different CRS. Please reproject first.")

    # ---- smooth (Mean-only)
    method = "Mean"
    b_radius_px = _radius_m_to_px(b_transform, param_m)
    a_radius_px = _radius_m_to_px(a_transform, param_m)
    d_radius_px = _radius_m_to_px(d_transform, param_m)
    b_sm = smooth_mean(b_arr, radius_px=b_radius_px)
    a_sm = smooth_mean(a_arr, radius_px=a_radius_px)
    d_sm = smooth_mean(d_arr, radius_px=d_radius_px)

    # ---- write GeoTIFFs
    out_b_tif = os.path.join(ctx.out_ui1, "before_asc_smooth.tif")
    out_a_tif = os.path.join(ctx.out_ui1, "after_asc_smooth.tif")
    out_d_tif = os.path.join(ctx.out_ui1, "after_dem_smooth.tif")
    os.makedirs(ctx.out_ui1, exist_ok=True)
    b_meta.update(dtype="float32", count=1, compress="lzw")
    a_meta.update(dtype="float32", count=1, compress="lzw")
    d_meta.update(dtype="float32", count=1, compress="lzw")

    with rasterio.open(out_b_tif, "w", **b_meta) as d:
        d.write(b_sm, 1)
    with rasterio.open(out_a_tif, "w", **a_meta) as d:
        d.write(a_sm, 1)
    with rasterio.open(out_d_tif, "w", **d_meta) as d:
        d.write(d_sm, 1)

    # ---- PNG previews
    out_b_png = os.path.join(ctx.out_ui1, "before_asc_smooth.png")
    out_a_png = os.path.join(ctx.out_ui1, "after_asc_smooth.png")
    out_d_png = os.path.join(ctx.out_ui1, "after_dem_smooth.png")
    _save_preview_png(
        b_sm,
        b_transform,
        out_b_png,
        f"before.asc (smooth {method} radius={param_m}m, px={b_radius_px[1]:.2f}x{b_radius_px[0]:.2f})",
    )
    _save_preview_png(
        a_sm,
        a_transform,
        out_a_png,
        f"after.asc (smooth {method} radius={param_m}m, px={a_radius_px[1]:.2f}x{a_radius_px[0]:.2f})",
    )
    _save_preview_png(
        d_sm,
        d_transform,
        out_d_png,
        f"after_dem.tif (smooth {method} radius={param_m}m, px={d_radius_px[1]:.2f}x{d_radius_px[0]:.2f})",
    )

    # ---- meta
    meta_path = os.path.join(ctx.out_ui1, "smooth_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": method,
                "param_m": float(param_m),
                "before_radius_px_rc": [float(b_radius_px[0]), float(b_radius_px[1])],
                "after_radius_px_rc": [float(a_radius_px[0]), float(a_radius_px[1])],
                "after_dem_radius_px_rc": [float(d_radius_px[0]), float(d_radius_px[1])],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "before_tif": out_b_tif.replace("\\", "/"),
        "after_tif":  out_a_tif.replace("\\", "/"),
        "after_dem_tif": out_d_tif.replace("\\", "/"),
        "before_png": out_b_png.replace("\\", "/"),
        "after_png":  out_a_png.replace("\\", "/"),
        "after_dem_png": out_d_png.replace("\\", "/"),
        "meta": meta_path.replace("\\", "/"),
    }
