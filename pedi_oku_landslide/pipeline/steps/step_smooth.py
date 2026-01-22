# pedi_oku_landslide/pipeline/step_smooth.py
import os
import json
import numpy as np
import rasterio
from rasterio.transform import Affine
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt

from pedi_oku_landslide.project.path_manager import AnalysisContext
from pedi_oku_landslide.core.analysis import smooth_gaussian

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

def run_smooth(ctx: AnalysisContext, sigma_px: float = 2.0) -> dict:
    """
    Smooth both BEFORE.asc and AFTER.asc using Gaussian filter (sigma in pixels).
    Outputs:
      - GeoTIFFs: ui1/before_asc_smooth.tif, ui1/after_asc_smooth.tif
      - PNG previews: ui1/before_asc_smooth.png, ui1/after_asc_smooth.png
      - JSON: ui1/smooth_meta.json (sigma)
    Returns dict with output paths.
    """
    # ---- read BEFORE.asc
    before_path = os.path.join(ctx.in_dir, "before.asc")
    after_path  = os.path.join(ctx.in_dir, "after.asc")
    if not (os.path.exists(before_path) and os.path.exists(after_path)):
        raise FileNotFoundError("before.asc or after.asc not found in run/input")

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

    # optional: check CRS一致 (nếu khác, cảnh báo/lỗi)
    if (b_crs is not None) and (a_crs is not None) and (b_crs != a_crs):
        raise ValueError("BEFORE and AFTER have different CRS. Please reproject first.")

    # ---- smooth
    b_sm = smooth_gaussian(b_arr, sigma_px=sigma_px)
    a_sm = smooth_gaussian(a_arr, sigma_px=sigma_px)

    # ---- write GeoTIFFs
    out_b_tif = os.path.join(ctx.out_ui1, "before_asc_smooth.tif")
    out_a_tif = os.path.join(ctx.out_ui1, "after_asc_smooth.tif")
    os.makedirs(ctx.out_ui1, exist_ok=True)
    b_meta.update(dtype="float32", count=1, compress="lzw")
    a_meta.update(dtype="float32", count=1, compress="lzw")

    with rasterio.open(out_b_tif, "w", **b_meta) as d:
        d.write(b_sm, 1)
    with rasterio.open(out_a_tif, "w", **a_meta) as d:
        d.write(a_sm, 1)

    # ---- PNG previews
    out_b_png = os.path.join(ctx.out_ui1, "before_asc_smooth.png")
    out_a_png = os.path.join(ctx.out_ui1, "after_asc_smooth.png")
    _save_preview_png(b_sm, b_transform, out_b_png, f"before.asc (smooth σ={sigma_px}px)")
    _save_preview_png(a_sm, a_transform, out_a_png, f"after.asc (smooth σ={sigma_px}px)")

    # ---- meta
    meta_path = os.path.join(ctx.out_ui1, "smooth_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"sigma_px": float(sigma_px)}, f, ensure_ascii=False, indent=2)

    return {
        "before_tif": out_b_tif.replace("\\", "/"),
        "after_tif":  out_a_tif.replace("\\", "/"),
        "before_png": out_b_png.replace("\\", "/"),
        "after_png":  out_a_png.replace("\\", "/"),
        "meta": meta_path.replace("\\", "/"),
    }
