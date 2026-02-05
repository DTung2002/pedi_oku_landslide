# pedi_oku_landslide/pipeline/step_detect.py
from __future__ import annotations

import os
import json
import traceback
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from pedi_oku_landslide.project.path_manager import AnalysisContext
from pedi_oku_landslide.pipeline.ingest import update_ingest_processed


def _read(path: str):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype("float32")
        meta = ds.meta.copy()
        transform = ds.transform
        crs = ds.crs
    return arr, meta, transform, crs


def run_detect(
    ctx: AnalysisContext,
    method: str = "threshold",
    threshold_m: float = 0.8,
    k: int = 2,
) -> dict:
    """
    Detect landslide zones từ dX/dY.
    - method="threshold": dùng ngưỡng theo mét (magnitude_m >= threshold_m)
    - method="auto": KMeans trên magnitude (m)

    Xuất:
      - ui1/landslide_mask.tif (uint8, nodata=0)
      - ui1/landslide_overlay.png (overlay mask trên hillshade)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from rasterio.plot import plotting_extent
    from scipy.ndimage import uniform_filter

    dx_path = os.path.join(ctx.out_ui1, "dx.tif")
    dy_path = os.path.join(ctx.out_ui1, "dy.tif")
    if not (os.path.exists(dx_path) and os.path.exists(dy_path)):
        raise FileNotFoundError("dx.tif or dy.tif is missing. Please run SAD first.")

    # ---- đọc dX, dY + transform để suy ra pixel size (m)
    with rasterio.open(dx_path) as dx_ds:
        dX = dx_ds.read(1).astype("float32")
        meta_dx = dx_ds.meta.copy()
        transform = dx_ds.transform
        crs = dx_ds.crs
        H, W = dX.shape
        px_m = abs(float(transform.a))
        py_m = abs(float(transform.e))
        pix_m = (px_m + py_m) / 2.0 if (px_m > 0 and py_m > 0) else 1.0

    with rasterio.open(dy_path) as dy_ds:
        dY = dy_ds.read(1).astype("float32")

    # ---- magnitude theo mét
    mag_m = np.sqrt((dX * px_m) ** 2 + (dY * py_m) ** 2).astype("float32")
    mag_m[~np.isfinite(mag_m)] = 0.0

    # ---- tạo mask
    if method == "threshold":
        thr = float(threshold_m)
        mask = (mag_m >= thr).astype("uint8")
    else:
        # auto clustering (KMeans) trên magnitude_m
        flat = mag_m.reshape(-1, 1)
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(flat)
        lab_img = labels.reshape(mag_m.shape)
        means = [float(np.mean(mag_m[lab_img == i])) if np.any(lab_img == i) else -1.0 for i in range(k)]
        slide_idx = int(np.argmax(means))
        mask = (lab_img == slide_idx).astype("uint8")

    # ---- ghi mask.tif (uint8, nodata=0) - KHÔNG dùng -9999 cho uint8
    mask_tif = os.path.join(ctx.out_ui1, "landslide_mask.tif")
    os.makedirs(ctx.out_ui1, exist_ok=True)
    for k_ in ("dtype", "nodata"):
        meta_dx.pop(k_, None)
    meta_dx.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="lzw")
    with rasterio.open(mask_tif, "w", **meta_dx) as dst:
        dst.write(mask, 1)

    # ---- crop DEM with mask for UI3 (optional)
    dem_cropped = None
    try:
        meta_path = os.path.join(ctx.run_dir, "ingest_meta.json")
        dem_path = None
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
            inputs = meta.get("inputs") or {}
            dem_path = inputs.get("before_dem")

        if not dem_path or not os.path.exists(dem_path):
            # fallback: look for GeoTIFF in input dir
            for name in os.listdir(ctx.in_dir):
                if name.lower().endswith((".tif", ".tiff")):
                    dem_path = os.path.join(ctx.in_dir, name)
                    break
        if not dem_path or not os.path.exists(dem_path):
            raise FileNotFoundError("DEM GeoTIFF not found for cropping.")

        if dem_path and os.path.exists(dem_path):
            with rasterio.open(dem_path) as dem_ds:
                dem_arr = dem_ds.read(1)
                dem_meta = dem_ds.meta.copy()
                dem_transform = dem_ds.transform
                dem_crs = dem_ds.crs
                dem_height = dem_ds.height
                dem_width = dem_ds.width

            with rasterio.open(mask_tif) as msk_ds:
                mask_src = msk_ds.read(1)
                mask_transform = msk_ds.transform
                mask_crs = msk_ds.crs

            needs_reproject = (mask_transform != dem_transform) or \
                (mask_src.shape[0] != dem_height) or (mask_src.shape[1] != dem_width) or \
                (mask_crs != dem_crs)

            if needs_reproject:
                # If CRS missing, assume same CRS for alignment.
                src_crs = mask_crs or dem_crs or "EPSG:3857"
                dst_crs = dem_crs or mask_crs or "EPSG:3857"
                mask_re = np.zeros((dem_height, dem_width), dtype=mask_src.dtype)
                reproject(
                    source=mask_src,
                    destination=mask_re,
                    src_transform=mask_transform,
                    src_crs=src_crs,
                    dst_transform=dem_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )
                mask_use = mask_re
            else:
                mask_use = mask_src

            nodata = dem_meta.get("nodata")
            if nodata is None:
                nodata = -9999.0
            dem_crop_arr = np.where(mask_use > 0, dem_arr, nodata)

            dem_meta.update(dtype=dem_crop_arr.dtype, nodata=nodata, compress="lzw")
            dem_cropped = os.path.join(ctx.out_ui1, "dem_cropped.tif")
            with rasterio.open(dem_cropped, "w", **dem_meta) as dst:
                dst.write(dem_crop_arr, 1)
    except Exception as e:
        print(f"[WARN] Failed to crop DEM with mask: {e}")
        try:
            log_path = os.path.join(ctx.out_ui1, "dem_cropped_error.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(traceback.format_exc())
                f.write("\n")
        except Exception:
            pass

    # ---- nền hillshade từ BEFORE.asc (tránh masked.filled(np.nan) gây int32 lỗi)
    before_asc = os.path.join(ctx.in_dir, "before.asc")
    if os.path.exists(before_asc):
        with rasterio.open(before_asc) as dem_src:
            dem = dem_src.read(1).astype("float32")   # <-- đọc thường, ép float32
            nd = dem_src.nodata
            if nd is not None:
                dem[dem == nd] = np.nan
            # thay NaN bằng median an toàn để hillshade
            med = float(np.nanmedian(dem)) if np.any(np.isfinite(dem)) else 0.0
            dem = np.where(np.isfinite(dem), dem, med).astype("float32")
            dem_s = uniform_filter(dem, size=11, mode="nearest")
            ls = LightSource(azdeg=315, altdeg=45)
            hill = ls.hillshade(dem_s, vert_exag=1.0)
            base_extent = plotting_extent(dem_src)
            base_img = hill
    else:
        # fallback: dùng extent từ dX transform
        x_min = transform.c
        x_max = x_min + transform.a * W
        y_max = transform.f
        y_min = y_max + transform.e * H
        base_extent = [x_min, x_max, y_min, y_max]
        base_img = np.zeros((H, W), dtype="float32")

    # ---- vẽ overlay: hillshade nền + HEATMAP magnitude (chỉ trên vùng mask), không dùng opacity
    mask_png = os.path.join(ctx.out_ui1, "landslide_overlay.png")
    plt.figure(figsize=(8, 8), dpi=140)

    # nền địa hình
    plt.imshow(base_img, cmap="gray", extent=base_extent, origin="upper")

    # heatmap chỉ trong vùng mask (không phủ chỗ khác)
    mag_show = np.ma.masked_where(mask == 0, mag_m)  # chỉ vẽ nơi mask==1
    h = plt.imshow(
        mag_show, cmap="turbo", extent=base_extent, origin="upper",
        interpolation="nearest"  # giữ chi tiết
    )

    # viền vùng mask cho rõ ranh giới
    try:
        plt.contour(
            mask, levels=[0.5], colors="black", linewidths=1.0,
            extent=base_extent, origin="upper"
        )
    except Exception:
        pass

    cbar = plt.colorbar(h, fraction=0.046, pad=0.04)
    cbar.set_label("Displacement magnitude (m)")

    plt.title("Detected Landslide Zone — heatmap over hillshade")
    plt.xlabel("X (m)");
    plt.ylabel("Y (m)")
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(mask_png, dpi=260)
    plt.close()
    # ---- update ingest_meta.processed.slip_mask để UI3 dùng
    try:
        update_ingest_processed(
            ctx.run_dir,
            slip_mask=mask_tif.replace("\\", "/"),
            dem_cropped=(dem_cropped.replace("\\", "/") if dem_cropped else None),
        )
    except Exception as e:
        print(f"[WARN] Failed to update ingest_meta from Detect: {e}")

    return {
        "mask_tif": mask_tif.replace("\\", "/"),
        "mask_png": mask_png.replace("\\", "/"),
        "threshold_m": float(threshold_m),
        "pixel_size_m": float(pix_m),
    }



def render_vectors(
    ctx: AnalysisContext,
    step: int = 25,
    scale: float = 0.1,
    vector_color: str = "blue",
    vector_width: float = 0.01,
    min_m: float = 0.05,
    max_m: float = 2.0,
) -> dict:
    """
    Vẽ vector dX/dY (đơn vị mét) trên hillshade DEM.
    - Chỉ vẽ bên trong landslide_mask.tif (nếu có).
    - ĐẢO dấu trục Y để phù hợp hệ toạ độ (hướng tăng dương lên Bắc):
        U =  dX * px_m
        V = -dY * py_m
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from rasterio.plot import plotting_extent
    from scipy.ndimage import uniform_filter

    dx_path = os.path.join(ctx.out_ui1, "dx.tif")
    dy_path = os.path.join(ctx.out_ui1, "dy.tif")
    if not (os.path.exists(dx_path) and os.path.exists(dy_path)):
        raise FileNotFoundError("dx.tif or dy.tif is missing. Please run SAD first.")

    # --- đọc dX/dY và suy pixel size (m)
    with rasterio.open(dx_path) as dx_ds:
        dX = dx_ds.read(1).astype("float32")
        transform = dx_ds.transform
        H, W = dX.shape
        px_m = abs(float(transform.a))
        py_m = abs(float(transform.e))

    with rasterio.open(dy_path) as dy_ds:
        dY = dy_ds.read(1).astype("float32")

    # --- DEM hillshade nền từ before.asc
    before_asc = os.path.join(ctx.in_dir, "before.asc")
    with rasterio.open(before_asc) as dem_src:
        dem = dem_src.read(1).astype("float32")
        nd = dem_src.nodata
        if nd is not None:
            dem[dem == nd] = np.nan
        med = float(np.nanmedian(dem)) if np.any(np.isfinite(dem)) else 0.0
        dem = np.where(np.isfinite(dem), dem, med).astype("float32")
        dem_s = uniform_filter(dem, size=11, mode="nearest")
        ls = LightSource(azdeg=315, altdeg=45)
        hill = ls.hillshade(dem_s, vert_exag=1.0)
        extent = plotting_extent(dem_src)

    # --- mask landslide (nếu có)
    mask_path = os.path.join(ctx.out_ui1, "landslide_mask.tif")
    if os.path.exists(mask_path):
        with rasterio.open(mask_path) as msk_ds:
            m = msk_ds.read(1)
            in_zone = (m > 0)
    else:
        in_zone = np.ones_like(dX, dtype=bool)

    # --- chọn mẫu theo step + lọc độ lớn theo mét
    mag_m = np.sqrt((dX * px_m) ** 2 + (dY * py_m) ** 2).astype("float32")
    sample = np.zeros_like(mag_m, dtype=bool)
    sample[::max(1, int(step)), ::max(1, int(step))] = True

    ok = sample & in_zone & np.isfinite(mag_m) & (mag_m >= float(min_m)) & (mag_m <= float(max_m))

    rows, cols = np.where(ok)
    if rows.size == 0:
        raise RuntimeError("No vectors after filtering; run Detect first or relax filters.")

    # --- world coords ở tâm pixel
    Xw = transform.c + cols * transform.a + transform.a / 2.0
    Yw = transform.f + rows * transform.e + transform.e / 2.0

    # --- vector theo mét (ĐẢO dấu trục Y)
    U = dX[rows, cols] * px_m
    V = -dY[rows, cols] * py_m  # đảo dấu -> hướng Nam/Đông-Nam nếu dY dương (pixel xuống)

    out_png = os.path.join(ctx.out_ui1, "vectors_overlay.png")
    plt.figure(figsize=(11, 8), dpi=140)
    plt.imshow(hill, cmap="gray", extent=extent, origin="upper")
    plt.quiver(
        Xw, Yw, U, V,
        color=vector_color, angles="xy",
        scale_units="xy", scale=(1.0 / max(scale, 1e-6)),
        width=float(vector_width), headwidth=6, headlength=7
    )
    plt.title(f"Displacement Vectors (step={step}, scale={scale}, {min_m}–{max_m} m) — masked")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=260)
    plt.close()
    return {"vectors_png": out_png.replace("\\", "/")}



