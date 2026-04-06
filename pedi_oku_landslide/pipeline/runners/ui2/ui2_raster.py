import os
from typing import Dict

import numpy as np
import rasterio
from rasterio.enums import Resampling

from .ui2_paths import file_exists, find_mask_tif, resolve_after_tif, validate_run_inputs
from .ui2_visualization import hillshade, rgba_from_scalar


def validate_context(run_dir: str) -> Dict[str, object]:
    ui1_dir = os.path.join(run_dir, "ui1")
    ui2_dir = os.path.join(run_dir, "ui2")
    missing, after_tif, mask_tif = validate_run_inputs(run_dir)
    return {
        "ui1_dir": ui1_dir,
        "ui2_dir": ui2_dir,
        "after_tif": after_tif,
        "mask_tif": mask_tif,
        "missing": missing,
        "ready": os.path.isdir(ui1_dir) and not missing,
    }


def load_layers(run_dir: str, vector_settings: Dict[str, object]) -> Dict[str, object]:
    ctx = validate_context(run_dir)
    if not ctx["ready"]:
        raise FileNotFoundError(", ".join(str(p) for p in ctx["missing"]))
    ui1_dir = str(ctx["ui1_dir"])
    dz_tif = os.path.join(ui1_dir, "dz.tif")
    after_tif = str(ctx["after_tif"])
    mask_tif = ctx["mask_tif"] or find_mask_tif(ui1_dir)

    with rasterio.open(dz_tif) as ds:
        dz = ds.read(1).astype("float32")
        transform = ds.transform
        inv_transform = ~transform
        width, height = ds.width, ds.height

    with rasterio.open(after_tif) as ds:
        if (ds.width, ds.height) != (width, height) or ds.transform != transform:
            after = ds.read(1, out_shape=(height, width), resampling=Resampling.bilinear).astype("float32")
        else:
            after = ds.read(1).astype("float32")

    def _read_align(name: str, *, nearest: bool = False):
        path = os.path.join(ui1_dir, name)
        if not file_exists(path):
            return None
        with rasterio.open(path) as ds:
            if (ds.width, ds.height) != (width, height) or ds.transform != transform:
                resampling = Resampling.nearest if nearest else Resampling.bilinear
                return ds.read(1, out_shape=(height, width), resampling=resampling).astype("float32")
            return ds.read(1).astype("float32")

    dx = _read_align("dx.tif")
    dy = _read_align("dy.tif")
    if mask_tif and file_exists(str(mask_tif)):
        with rasterio.open(str(mask_tif)) as ds:
            if (ds.width, ds.height) != (width, height) or ds.transform != transform:
                mask = ds.read(1, out_shape=(height, width), resampling=Resampling.nearest).astype("uint8")
            else:
                mask = ds.read(1).astype("uint8")
        mask = (mask > 0).astype("uint8")
    else:
        mask = np.ones_like(dz, dtype="uint8")

    cell = float(abs(transform.a))
    hs8 = hillshade(after, cell)
    if dx is not None and dy is not None:
        heat_scalar = np.hypot(dx, dy)
        heat_scalar[mask == 0] = np.nan
    else:
        heat_scalar = dz.copy()
        heat_scalar[mask == 0] = np.nan
    alpha = float(vector_settings.get("heat_alpha", 1.0))
    heat_rgba = rgba_from_scalar(heat_scalar, cm="turbo", alpha=alpha)
    return {
        "dz": dz,
        "dx": dx,
        "dy": dy,
        "mask": mask,
        "after": after,
        "transform": transform,
        "inv_transform": inv_transform,
        "width": width,
        "height": height,
        "dem_path": after_tif,
        "ui1_dir": ui1_dir,
        "hillshade": hs8,
        "heat_rgba": heat_rgba,
    }
