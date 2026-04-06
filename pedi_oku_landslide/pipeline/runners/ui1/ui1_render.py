import json
import os

import numpy as np
import rasterio

from .ui1_ingest import resolve_run_input_path
from .ui1_types import AnalysisContext


def export_vectors_json(
    ctx: AnalysisContext,
    *,
    step: int,
    scale: float,
    vector_color: str,
    vector_width: float,
    vector_opacity: float,
    min_m: float = 0.05,
    max_m: float = 2.0,
) -> str:
    dx_path = os.path.join(ctx.out_ui1, "dx.tif")
    dy_path = os.path.join(ctx.out_ui1, "dy.tif")
    dem_path = resolve_run_input_path(ctx.run_dir, "after_asc")
    mask_path = os.path.join(ctx.out_ui1, "landslide_mask.tif")

    if not (os.path.exists(dx_path) and os.path.exists(dy_path)):
        raise FileNotFoundError("dx.tif or dy.tif is missing.")
    if not os.path.exists(dem_path):
        raise FileNotFoundError("after.asc is missing.")

    with rasterio.open(dx_path) as dx_ds:
        d_x = dx_ds.read(1).astype("float32")
        transform = dx_ds.transform
        px_m = abs(float(transform.a))
        py_m = abs(float(transform.e))

    with rasterio.open(dy_path) as dy_ds:
        d_y = dy_ds.read(1).astype("float32")

    with rasterio.open(dem_path) as dem_ds:
        dem = dem_ds.read(1).astype("float32")
        nodata = dem_ds.nodata
        if nodata is not None:
            dem[dem == nodata] = np.nan

    if os.path.exists(mask_path):
        with rasterio.open(mask_path) as mask_ds:
            in_zone = mask_ds.read(1) > 0
    else:
        in_zone = np.ones_like(d_x, dtype=bool)

    magnitude_m = np.sqrt((d_x * px_m) ** 2 + (d_y * py_m) ** 2).astype("float32")
    sample = np.zeros_like(magnitude_m, dtype=bool)
    sample[:: max(1, int(step)), :: max(1, int(step))] = True
    ok = sample & in_zone & np.isfinite(magnitude_m) & (magnitude_m >= float(min_m)) & (magnitude_m <= float(max_m))

    rows, cols = np.where(ok)
    out_dir = os.path.join(ctx.out_ui1, "vector")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "vectors.json")

    x_world = transform.c + cols * transform.a + transform.a / 2.0
    y_world = transform.f + rows * transform.e + transform.e / 2.0

    dx_px = d_x[rows, cols]
    dy_px = d_y[rows, cols]
    dx_m = dx_px * px_m
    dy_m = dy_px * py_m
    z_vals = dem[rows, cols]
    magnitude_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
    direction_deg = np.degrees(np.arctan2(dy_m, dx_m))

    vectors = []
    for i in range(len(rows)):
        vectors.append(
            {
                "row": int(rows[i]),
                "col": int(cols[i]),
                "x": float(x_world[i]),
                "y": float(y_world[i]),
                "z": (None if not np.isfinite(z_vals[i]) else float(z_vals[i])),
                "direction_deg": (None if not np.isfinite(direction_deg[i]) else float(direction_deg[i])),
                "magnitude_m": (None if not np.isfinite(magnitude_m[i]) else float(magnitude_m[i])),
                "dx_px": (None if not np.isfinite(dx_px[i]) else float(dx_px[i])),
                "dy_px": (None if not np.isfinite(dy_px[i]) else float(dy_px[i])),
                "dx_m": (None if not np.isfinite(dx_m[i]) else float(dx_m[i])),
                "dy_m": (None if not np.isfinite(dy_m[i]) else float(dy_m[i])),
            }
        )

    payload = {
        "project_id": ctx.project_id,
        "run_id": ctx.run_id,
        "count": len(vectors),
        "step": int(step),
        "scale": float(scale),
        "vector_color": str(vector_color),
        "vector_width": float(vector_width),
        "vector_opacity": float(vector_opacity),
        "filters": {
            "min_m": float(min_m),
            "max_m": float(max_m),
        },
        "sources": {
            "dx_tif": dx_path.replace("\\", "/"),
            "dy_tif": dy_path.replace("\\", "/"),
            "dem_after_asc": dem_path.replace("\\", "/"),
            "mask_tif": (mask_path.replace("\\", "/") if os.path.exists(mask_path) else None),
        },
        "vectors": vectors,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_json.replace("\\", "/")


__all__ = ["export_vectors_json"]
