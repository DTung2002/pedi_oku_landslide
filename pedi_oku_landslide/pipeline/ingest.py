# pedi_oku_landslide/pipeline/ingest.py
import os
import json
import shutil
from typing import Dict
from typing import Optional
from pyproj import CRS
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from pedi_oku_landslide.config.settings import load_config
from pedi_oku_landslide.core.gis_utils import hillshade
from pedi_oku_landslide.project.path_manager import AnalysisContext
def update_ingest_processed(run_dir: str, **kwargs):
    """
    Cập nhật các field trong ingest_meta.json["processed"].

    Ví dụ:
        update_ingest_processed(run_dir,
                                dx=dx_tif,
                                dy=dy_tif,
                                dz=dz_tif)

    Nếu ingest_meta.json chưa có 'processed' thì tự tạo.
    """
    meta_path = os.path.join(run_dir, "ingest_meta.json")
    if not os.path.exists(meta_path):
        print(f"[WARN] ingest_meta.json not found: {meta_path}")
        return

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read ingest_meta.json: {e}")
        return

    processed = data.get("processed") or {}
    processed.update(kwargs)
    data["processed"] = processed

    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("[OK] Updated ingest_meta.processed:", kwargs)
    except Exception as e:
        print(f"[WARN] Failed to write ingest_meta.json: {e}")

def _copy(src: str, dst_dir: str) -> str:
    """Copy a file to dst_dir (creating it if needed) and return the new path."""
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    return dst

def _load_crs_from_sibling_prj(path: str) -> Optional[CRS]:
    """
    Nếu tồn tại <path>.prj cạnh file raster (ví dụ ASC), đọc WKT và trả về CRS.
    Trả về None nếu không tìm thấy hoặc đọc lỗi.
    """
    prj_path = os.path.splitext(path)[0] + ".prj"
    if not os.path.exists(prj_path):
        return None
    try:
        with open(prj_path, "r", encoding="utf-8") as f:
            wkt = f.read().strip()
        return CRS.from_wkt(wkt) if wkt else None
    except Exception:
        return None


def _save_hillshade_png_from_raster(src_path: str, out_png: str, fallback_epsg: str | None = None) -> dict:
    """
    Render hillshade PNG lớn, vẽ trục + lưới. Xử lý 3 tình huống CRS:
      - CRS có sẵn trong raster
      - Không có -> thử đọc <basename>.prj
      - Vẫn không có -> dùng fallback_epsg (nếu truyền), ngược lại 'unknown'

    Returns: metadata gồm crs, extent, bounds, width/height, và 'crs_source'
    """
    import rasterio
    from pyproj import CRS

    with rasterio.open(src_path) as ds:
        arr = ds.read(1).astype("float32")
        transform = ds.transform
        crs_ds = ds.crs
        bounds = ds.bounds
        width, height = ds.width, ds.height

    # 1) Resolve CRS
    crs_source = "embedded"
    crs_obj: CRS | None = None

    if crs_ds:
        # CRS có sẵn trong raster
        crs_obj = CRS.from_user_input(crs_ds)
        crs_source = "embedded"
    else:
        crs_from_prj = _load_crs_from_sibling_prj(src_path)
        if crs_from_prj:
            crs_obj = crs_from_prj
            crs_source = "sibling_prj"
        elif fallback_epsg:
            try:
                crs_obj = CRS.from_user_input(fallback_epsg)
                crs_source = "fallback"
            except Exception:
                crs_obj = None
                crs_source = "unknown"
        else:
            crs_obj = None
            crs_source = "unknown"

    # 2) Làm sạch dữ liệu
    med = float(np.nanmedian(arr)) if np.isfinite(np.nanmedian(arr)) else 0.0
    arr = np.where(np.isfinite(arr), arr, med)

    # 3) Hillshade
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(arr, vert_exag=1.0)  # float [0..1]

    # 4) Extent theo affine transform
    x_min = transform.c
    x_max = x_min + transform.a * width
    y_max = transform.f
    y_min = y_max + transform.e * height
    extent = [x_min, x_max, y_min, y_max]

    # 5) Vẽ figure lớn + lưới + tiêu đề CRS
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(9.5, 9.5), dpi=120)
    plt.imshow(hs, cmap="gray", extent=extent, origin="upper")

    if crs_obj:
        crs_str = crs_obj.to_string()
        title_crs = f"CRS: {crs_str} ({crs_source})"
        x_label = "X (map units)"
        y_label = "Y (map units)"
    else:
        title_crs = "CRS: unknown"
        x_label = "X (grid units)"
        y_label = "Y (grid units)"

    plt.title(f"{os.path.basename(src_path)} — {title_crs}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.9, color="red")
    plt.ticklabel_format(style="plain", useOffset=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

    return {
        "crs": (crs_obj.to_string() if crs_obj else None),
        "crs_source": crs_source,  # 'embedded' | 'sibling_prj' | 'fallback' | 'unknown'
        "extent": extent,
        "bounds": (bounds.left, bounds.bottom, bounds.right, bounds.top),
        "width": int(width),
        "height": int(height),
    }


def run_ingest(ctx: AnalysisContext, files: Dict[str, str]) -> Dict:
    """
    Ingest 5 inputs into a new run folder, prepare previews, and write metadata.

    Required keys in `files`:
      - before_dem : GeoTIFF (.tif)
      - before_asc : ASCII grid (.asc)
      - after_asc  : ASCII grid (.asc)
      - before_pz  : ASCII grid (.asc)
      - after_pz   : ASCII grid (.asc)
    """
    import os, json
    import rasterio
    from  pedi_oku_landslide.core.gis_utils import hillshade
    # load fallback EPSG from config (optional)
    try:
        from config.settings import load_config
        cfg = load_config(ctx.base_dir)
        fallback_epsg = getattr(getattr(cfg, "crs", None), "fallback_epsg", None)
    except Exception:
        fallback_epsg = None  # if config not present, just skip fallback

    # 1) Validate inputs
    required = ["before_dem", "before_asc", "after_asc", "before_pz", "after_pz"]
    missing = [k for k in required if not files.get(k)]
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")

    # 2) Copy inputs into run/input
    copied = {k: _copy(files[k], ctx.in_dir) for k in required}

    # 3) Hillshade from BEFORE DEM (GeoTIFF) to GeoTIFF preview in ui1
    with rasterio.open(copied["before_dem"]) as ds:
        dem = ds.read(1).astype("float32")
        meta = ds.meta.copy()

    hs_dem = hillshade(dem).astype("float32")
    meta.update(dtype="float32", count=1, compress="lzw")
    os.makedirs(ctx.out_ui1, exist_ok=True)
    hs_dem_tif = os.path.join(ctx.out_ui1, "before_dem_hillshade.tif")
    with rasterio.open(hs_dem_tif, "w", **meta) as ds:
        ds.write(hs_dem, 1)

    # 4) Quick PNG previews from BEFORE/AFTER ASC (with CRS/grid & axes)
    hs_before_png = os.path.join(ctx.out_ui1, "before_asc_hillshade.png")
    hs_after_png  = os.path.join(ctx.out_ui1, "after_asc_hillshade.png")

    meta_before = _save_hillshade_png_from_raster(
        copied["before_asc"], hs_before_png, fallback_epsg=fallback_epsg
    )
    meta_after  = _save_hillshade_png_from_raster(
        copied["after_asc"],  hs_after_png,  fallback_epsg=fallback_epsg
    )
    info = {
        "project_id": ctx.project_id,
        "run_id": ctx.run_id,
        "run_dir": ctx.run_dir.replace("\\", "/"),
        "input_dir": ctx.in_dir.replace("\\", "/"),
        "outputs": {
            "ui1": ctx.out_ui1.replace("\\", "/"),
            "ui2": ctx.out_ui2.replace("\\", "/"),
            "ui3": ctx.out_ui3.replace("\\", "/"),
        },
        "inputs": {k: v.replace("\\", "/") for k, v in copied.items()},

        "processed": {
            "dem_cropped": None,   # sẽ được step crop DEM / UI1 cập nhật sau
            "dx": None,            # step_sad sẽ cập nhật
            "dy": None,
            "dz": None,
            "slip_mask": None,     # step_detect sẽ cập nhật
            "lines": None          # UI2 (section) sẽ cập nhật
        },

        "preview": {
            "before_dem_hillshade_tif": hs_dem_tif.replace("\\", "/"),
            "before_asc_hillshade_png": hs_before_png.replace("\\", "/"),
            "after_asc_hillshade_png":  hs_after_png.replace("\\", "/"),
        },
        "preview_meta": {
            "before_asc": meta_before,
            "after_asc":  meta_after,
        },
        "note": "Later we can derive ASC files from DEMs and only ask for 2 DEM inputs."
    }

    with open(os.path.join(ctx.run_dir, "ingest_meta.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    return info

