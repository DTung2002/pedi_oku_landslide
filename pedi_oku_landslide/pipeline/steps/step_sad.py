# pedi_oku_landslide/pipeline/step_sad.py
from __future__ import annotations

import os
import json
from typing import Dict, Tuple, Optional

import numpy as np
import rasterio
from rasterio.transform import Affine
import matplotlib.pyplot as plt

try:
    import cv2  # OpenCV (optional for "opencv" method)
except Exception:
    cv2 = None

from pedi_oku_landslide.project.path_manager import AnalysisContext
from pedi_oku_landslide.pipeline.ingest import update_ingest_processed


# ------------------------- Low-level helpers -------------------------

def _read_raster(path: str) -> Tuple[np.ndarray, dict, Affine, Optional[object]]:
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype("float32")
        meta = ds.meta.copy()
        transform = ds.transform
        crs = ds.crs
    return arr, meta, transform, crs


def _pixel_size_from_transform(transform: Affine) -> Tuple[float, float]:
    # Typically transform.a = pixel width (xres), transform.e = negative pixel height
    return abs(float(transform.a)), abs(float(transform.e))


def _nan_to_median(a: np.ndarray) -> Tuple[np.ndarray, float]:
    med = float(np.nanmedian(a)) if np.isfinite(np.nanmedian(a)) else 0.0
    out = np.where(np.isfinite(a), a, med).astype("float32", copy=False)
    return out, med


def _save_png_diverging(arr: np.ndarray, transform: Affine, out_png: str,
                        title: str, unit: str, vlim: float | None = None) -> None:
    finite = np.isfinite(arr)
    if vlim is None and np.any(finite):
        p2, p98 = np.nanpercentile(arr[finite], [2, 98])
        vmax = float(max(abs(p2), abs(p98)))
        vlim = vmax if vmax > 0 else 1.0
    if vlim is None:
        vlim = 1.0

    h, w = arr.shape
    x_min = transform.c
    x_max = x_min + transform.a * w
    y_max = transform.f
    y_min = y_max + transform.e * h
    extent = [x_min, x_max, y_min, y_max]

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(9, 9), dpi=120)
    im = plt.imshow(arr, cmap="RdBu_r", vmin=-vlim, vmax=vlim, extent=extent, origin="upper")
    plt.title(title)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="k")
    plt.ticklabel_format(style="plain", useOffset=False)
    cbar = plt.colorbar(im, shrink=0.85)
    cbar.set_label(unit)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def _save_png_sequential(arr: np.ndarray, transform: Affine, out_png: str,
                         title: str, unit: str) -> None:
    h, w = arr.shape
    x_min = transform.c
    x_max = x_min + transform.a * w
    y_max = transform.f
    y_min = y_max + transform.e * h
    extent = [x_min, x_max, y_min, y_max]

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(9, 9), dpi=120)
    im = plt.imshow(arr, cmap="viridis", extent=extent, origin="upper")
    plt.title(title)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="k")
    plt.ticklabel_format(style="plain", useOffset=False)
    cbar = plt.colorbar(im, shrink=0.85)
    cbar.set_label(unit)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def _bilinear_sample(arr: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    print(np.isnan(yy).any())
    print(np.isinf(yy).any())

    """
    Bilinear interpolation on 2D array.
    - (yy, xx) are float indices in row/col coords.
    - Out-of-bounds -> NaN.
    """
    h, w = arr.shape
    x0 = np.floor(xx).astype(int); x1 = x0 + 1
    y0 = np.floor(yy).astype(int); y1 = y0 + 1

    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)

    out = np.full(yy.shape, np.nan, dtype="float32")
    if not np.any(valid):
        return out

    x = xx[valid]; y = yy[valid]
    x0v = x0[valid]; x1v = x1[valid]; y0v = y0[valid]; y1v = y1[valid]

    Ia = arr[y0v, x0v]
    Ib = arr[y0v, x1v]
    Ic = arr[y1v, x0v]
    Id = arr[y1v, x1v]

    wa = (x1v - x) * (y1v - y)
    wb = (x - x0v) * (y1v - y)
    wc = (x1v - x) * (y - y0v)
    wd = (x - x0v) * (y - y0v)

    vals = wa * Ia + wb * Ib + wc * Ic + wd * Id
    out[valid] = vals.astype("float32")
    return out


# -------------------- Displacement (dX, dY) estimation --------------------

def _sad_traditional(before: np.ndarray, after: np.ndarray,
                     patch_radius_px: int, search_radius_px: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Traditional SAD (Sum of Absolute Differences).
    For each pixel center, find (dx, dy) in [-R..R] minimizing sum|patch_before - patch_after|.

    NOTE: Very slow for large rasters. Use small windows / test-sized data.
    """
    H, W = before.shape
    dX = np.full((H, W), np.nan, dtype="float32")
    dY = np.full((H, W), np.nan, dtype="float32")

    # Valid area where patch fully fits
    pr = patch_radius_px
    sr = search_radius_px
    y_min = pr + sr
    y_max = H - pr - sr
    x_min = pr + sr
    x_max = W - pr - sr

    # Pre-fill NaN handling
    b, _ = _nan_to_median(before)
    a, _ = _nan_to_median(after)

    patch_size = 2 * pr + 1

    for y in range(y_min, y_max):
        # Extract before patch once per row
        for x in range(x_min, x_max):
            pb = b[y - pr:y + pr + 1, x - pr:x + pr + 1]
            best_cost = np.inf
            best_dx = 0
            best_dy = 0
            # Search window
            for dy in range(-sr, sr + 1):
                yy = y + dy
                for dx in range(-sr, sr + 1):
                    xx = x + dx
                    pa = a[yy - pr:yy + pr + 1, xx - pr:xx + pr + 1]
                    if pa.shape != (patch_size, patch_size):
                        continue
                    cost = np.sum(np.abs(pb - pa))
                    if cost < best_cost:
                        best_cost = cost
                        best_dx = dx
                        best_dy = dy
            dX[y, x] = best_dx
            dY[y, x] = best_dy

    return dX, dY


def _sad_opencv(before: np.ndarray, after: np.ndarray,
                patch_radius_px: int, search_radius_px: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    OpenCV-based search using matchTemplate (TM_SQDIFF) on local windows.
    Faster than pure Python. Requires cv2.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is not available. Please install opencv-python.")

    H, W = before.shape
    dX = np.full((H, W), np.nan, dtype="float32")
    dY = np.full((H, W), np.nan, dtype="float32")

    pr = patch_radius_px
    sr = search_radius_px
    y_min = pr + sr
    y_max = H - pr - sr
    x_min = pr + sr
    x_max = W - pr - sr

    b, _ = _nan_to_median(before)
    a, _ = _nan_to_median(after)

    # OpenCV expects float32 or uint8
    b32 = b.astype("float32", copy=False)
    a32 = a.astype("float32", copy=False)

    patch_size = 2 * pr + 1
    search_size = 2 * (pr + sr) + 1  # local window around center in AFTER

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            # Template from BEFORE
            tpl = b32[y - pr:y + pr + 1, x - pr:x + pr + 1]
            # Search region from AFTER
            y0 = y - (pr + sr)
            y1 = y + (pr + sr) + 1
            x0 = x - (pr + sr)
            x1 = x + (pr + sr) + 1
            roi = a32[y0:y1, x0:x1]
            if roi.shape[0] < search_size or roi.shape[1] < search_size:
                continue

            # TM_SQDIFF: best match is MIN location
            res = cv2.matchTemplate(roi, tpl, method=cv2.TM_SQDIFF)
            min_val, _, min_loc, _ = cv2.minMaxLoc(res)
            # min_loc is top-left in roi where template best matches
            # Convert to displacement relative to center (x, y)
            # In roi coords: template top-left = (min_loc.x, min_loc.y)
            # That corresponds to AFTER patch center at:
            # (x0 + min_loc.x + pr, y0 + min_loc.y + pr)
            x_after_center = x0 + min_loc[0] + pr
            y_after_center = y0 + min_loc[1] + pr
            dx = x_after_center - x
            dy = y_after_center - y

            dX[y, x] = dx
            dY[y, x] = dy

    return dX, dY


# -------------------- dZ from displacement --------------------

def _dz_from_displacement(before_pz: np.ndarray, after_pz: np.ndarray,
                          dX: np.ndarray, dY: np.ndarray) -> np.ndarray:
    """
    dZ = Z_after( row+dy, col+dx ) - Z_before( row, col )
    Interpolate after_pz at displaced coordinates using bilinear sampling.
    """
    H, W = before_pz.shape
    yy, xx = np.mgrid[0:H, 0:W].astype("float32")
    # sample AFTER at displaced coords
    yy2 = yy + dY
    xx2 = xx + dX
    Za = _bilinear_sample(after_pz, yy2, xx2)
    Zb = before_pz
    dz = np.where(np.isfinite(Za) & np.isfinite(Zb), Za - Zb, np.nan).astype("float32")
    return dz


# -------------------- Main entry --------------------

def run_sad(ctx: AnalysisContext,
            method: str = "opencv",
            patch_size_m: float = 20.0,
            search_radius_m: float = 2.0,
            use_smoothed: bool = True,
            vlim_dz: float | None = None) -> Dict:
    """
    Estimate displacement (dX, dY) between BEFORE.asc and AFTER.asc,
    then compute dZ from before_pz.asc / after_pz.asc using the displacement.

    Parameters
    ----------
    method : "opencv" | "traditional"
        - "opencv": cv2.matchTemplate (fast, recommended)
        - "traditional": pure Python SAD (correct but slow)
    patch_size_m : float
        Template window size (meters). (diameter)
    search_radius_m : float
        Max search radius around each pixel (meters).
    use_smoothed : bool
        If True and smoothed TIFFs exist (before_asc_smooth.tif/after_asc_smooth.tif),
        use them to estimate displacement; otherwise fall back to raw ASC.
    vlim_dz : float | None
        Color limit for dz PNG (symmetric). If None, auto percentile 2–98.

    Returns
    -------
    dict with paths: dx_tif, dy_tif, dz_tif, dx_png, dy_png, dz_png, meta
    """
    os.makedirs(ctx.out_ui1, exist_ok=True)

    # ---- choose source rasters for displacement (before.asc / after.asc, smoothed if available)
    b_smooth = os.path.join(ctx.out_ui1, "before_asc_smooth.tif")
    a_smooth = os.path.join(ctx.out_ui1, "after_asc_smooth.tif")
    if use_smoothed and os.path.exists(b_smooth) and os.path.exists(a_smooth):
        b_src = b_smooth
        a_src = a_smooth
        src_tag = "smoothed"
    else:
        b_src = os.path.join(ctx.in_dir, "before.asc")
        a_src = os.path.join(ctx.in_dir, "after.asc")
        src_tag = "raw"

    if not (os.path.exists(b_src) and os.path.exists(a_src)):
        raise FileNotFoundError("Cannot find BEFORE/AFTER rasters for displacement.")

    # ---- read rasters
    before, meta_b, transform_b, crs_b = _read_raster(b_src)
    after,  meta_a, transform_a, crs_a = _read_raster(a_src)

    # basic CRS check if present
    if (crs_b is not None) and (crs_a is not None) and (crs_b != crs_a):
        raise ValueError("BEFORE and AFTER CRS differ. Please reproject first.")

    # pixel size (meters)
    px_bx, px_by = _pixel_size_from_transform(transform_b)
    # assume square pixels; take mean as robust
    pixel_m = float((px_bx + px_by) / 2.0) if (px_bx > 0 and px_by > 0) else 1.0

    # window sizes in pixels
    patch_radius_px = max(1, int(round((patch_size_m / pixel_m) / 2.0)))
    search_radius_px = max(1, int(round(search_radius_m / pixel_m)))

    # ---- compute displacement
    if method.lower() == "opencv":
        dX, dY = _sad_opencv(before, after, patch_radius_px, search_radius_px)
    elif method.lower() in ("traditional", "sad", "original"):
        dX, dY = _sad_traditional(before, after, patch_radius_px, search_radius_px)
    else:
        raise ValueError("Unknown method. Use 'opencv' or 'traditional'.")

    # ---- save dX, dY GeoTIFFs
    dx_tif = os.path.join(ctx.out_ui1, "dx.tif")
    dy_tif = os.path.join(ctx.out_ui1, "dy.tif")
    m = meta_b.copy()
    m.update(dtype="float32", count=1, compress="lzw")
    with rasterio.open(dx_tif, "w", **m) as d:
        d.write(dX.astype("float32"), 1)
    with rasterio.open(dy_tif, "w", **m) as d:
        d.write(dY.astype("float32"), 1)

    # ---- save dX, dY PNG (in pixels)
    dx_png = os.path.join(ctx.out_ui1, "dx.png")
    dy_png = os.path.join(ctx.out_ui1, "dy.png")
    _save_png_diverging(dX, transform_b, dx_png, f"dX (pixels) [{src_tag}/{method}]", "pixels")
    _save_png_diverging(dY, transform_b, dy_png, f"dY (pixels) [{src_tag}/{method}]", "pixels")

    # ---- compute dZ using BEFORE/AFTER PZ rasters
    before_pz_path = os.path.join(ctx.in_dir, "before_pz.asc")
    after_pz_path  = os.path.join(ctx.in_dir, "after_pz.asc")
    if not (os.path.exists(before_pz_path) and os.path.exists(after_pz_path)):
        raise FileNotFoundError("before_pz.asc / after_pz.asc not found for dZ computation.")

    before_pz, meta_bz, transform_bz, crs_bz = _read_raster(before_pz_path)
    after_pz,  meta_az, transform_az, crs_az = _read_raster(after_pz_path)

    # check grid compatibility (same size & transform)
    if before_pz.shape != after_pz.shape:
        raise ValueError("before_pz and after_pz have different shapes.")
    if (transform_bz != transform_az):
        raise ValueError("before_pz and after_pz have different transforms (grids).")

    dz = _dz_from_displacement(before_pz, after_pz, dX, dY)

    # ---- save dZ GeoTIFF + PNG
    dz_tif = os.path.join(ctx.out_ui1, "dz.tif")
    mm = meta_bz.copy()
    mm.update(dtype="float32", count=1, compress="lzw")
    with rasterio.open(dz_tif, "w", **mm) as d:
        d.write(dz, 1)

    dz_png = os.path.join(ctx.out_ui1, "dz.png")
    _save_png_diverging(dz, transform_bz, dz_png, f"dZ (after_pz shifted - before_pz) [{method}]", "dZ", vlim=vlim_dz)

    # ---- write meta
    meta_json = os.path.join(ctx.out_ui1, "sad_meta.json")
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump({
            "method": method,
            "source": src_tag,
            "patch_size_m": float(patch_size_m),
            "search_radius_m": float(search_radius_m),
            "pixel_size_m": float(pixel_m),
            "patch_radius_px": int(patch_radius_px),
            "search_radius_px": int(search_radius_px),
            "notes": "dX/dY in pixel units; dZ computed by bilinear sampling of after_pz at displaced coords."
        }, f, ensure_ascii=False, indent=2)
    # ---- update ingest_meta.processed để UI khác (UI3, Detect, Section...)
    try:
        update_ingest_processed(
            ctx.run_dir,
            dx=dx_tif.replace("\\", "/"),
            dy=dy_tif.replace("\\", "/"),
            dz=dz_tif.replace("\\", "/"),
        )
    except Exception as e:
        print(f"[WARN] Failed to update ingest_meta from SAD: {e}")

    return {
        "dx_tif": dx_tif.replace("\\", "/"),
        "dy_tif": dy_tif.replace("\\", "/"),
        "dz_tif": dz_tif.replace("\\", "/"),
        "dx_png": dx_png.replace("\\", "/"),
        "dy_png": dy_png.replace("\\", "/"),
        "dz_png": dz_png.replace("\\", "/"),
        "meta": meta_json.replace("\\", "/"),
    }
