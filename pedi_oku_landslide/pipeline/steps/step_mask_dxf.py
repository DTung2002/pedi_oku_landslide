from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LightSource
from rasterio.features import rasterize
from rasterio.plot import plotting_extent
from rasterio.warp import Resampling, reproject

from pedi_oku_landslide.project.path_manager import AnalysisContext


def _parse_lwpolylines_from_dxf(
    dxf_path: str,
    layer_name: Optional[str] = None,
) -> Tuple[List[List[Tuple[float, float]]], Dict[str, Any]]:
    """
    Parse closed LWPOLYLINE entities from DXF.
    Returns polygon rings and parsing stats.
    """
    with open(dxf_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = [ln.rstrip("\r\n") for ln in f]

    pairs: List[Tuple[str, str]] = []
    i = 0
    while i + 1 < len(raw):
        pairs.append((raw[i].strip(), raw[i + 1].strip()))
        i += 2

    in_entities = False
    idx = 0
    rings: List[List[Tuple[float, float]]] = []
    stats: Dict[str, Any] = {
        "entities_total": 0,
        "lwpolyline_total": 0,
        "lwpolyline_kept": 0,
        "skipped_open": 0,
        "skipped_layer": 0,
        "layer_filter": (layer_name or ""),
        "layers_seen": {},
    }

    while idx < len(pairs):
        code, value = pairs[idx]

        if code == "0" and value == "SECTION":
            if idx + 1 < len(pairs):
                n_code, n_val = pairs[idx + 1]
                if n_code == "2" and n_val == "ENTITIES":
                    in_entities = True
                    idx += 2
                    continue
            idx += 1
            continue

        if in_entities and code == "0" and value == "ENDSEC":
            break

        if not in_entities:
            idx += 1
            continue

        if code != "0":
            idx += 1
            continue

        stats["entities_total"] += 1
        entity_type = value
        idx += 1

        if entity_type != "LWPOLYLINE":
            while idx < len(pairs) and pairs[idx][0] != "0":
                idx += 1
            continue

        stats["lwpolyline_total"] += 1
        layer = ""
        flags = 0
        points: List[Tuple[float, float]] = []
        pending_x: Optional[float] = None

        while idx < len(pairs) and pairs[idx][0] != "0":
            c, v = pairs[idx]
            if c == "8":
                layer = v
            elif c == "70":
                try:
                    flags = int(float(v))
                except Exception:
                    flags = 0
            elif c == "10":
                try:
                    pending_x = float(v)
                except Exception:
                    pending_x = None
            elif c == "20" and pending_x is not None:
                try:
                    points.append((pending_x, float(v)))
                except Exception:
                    pass
                pending_x = None
            idx += 1

        stats["layers_seen"][layer] = int(stats["layers_seen"].get(layer, 0)) + 1

        if layer_name and layer != layer_name:
            stats["skipped_layer"] += 1
            continue

        if len(points) < 3:
            continue

        is_closed = bool(flags & 1)
        if not is_closed:
            stats["skipped_open"] += 1
            continue

        if points[0] != points[-1]:
            points.append(points[0])
        if len(points) < 4:
            continue

        rings.append(points)
        stats["lwpolyline_kept"] += 1

    return rings, stats


def _save_mask_overlay(mask_tif: str, base_raster: str, out_png: str) -> str:
    with rasterio.open(base_raster) as base_ds:
        base_arr = base_ds.read(1).astype("float32")
        base_nd = base_ds.nodata
        base_tf = base_ds.transform
        base_crs = base_ds.crs
        base_extent = plotting_extent(base_ds)
        if base_nd is not None:
            if np.isnan(base_nd):
                base_arr[~np.isfinite(base_arr)] = np.nan
            else:
                base_arr[np.isclose(base_arr, float(base_nd))] = np.nan

    with rasterio.open(mask_tif) as mask_ds:
        m_src = mask_ds.read(1).astype("uint8")
        m_tf = mask_ds.transform
        m_crs = mask_ds.crs

    if (
        m_src.shape != base_arr.shape
        or m_tf != base_tf
        or (m_crs is not None and base_crs is not None and m_crs != base_crs)
    ):
        m_use = np.zeros(base_arr.shape, dtype=np.uint8)
        reproject(
            source=m_src,
            destination=m_use,
            src_transform=m_tf,
            src_crs=(m_crs or base_crs or "EPSG:3857"),
            dst_transform=base_tf,
            dst_crs=(base_crs or m_crs or "EPSG:3857"),
            resampling=Resampling.nearest,
        )
    else:
        m_use = m_src

    med = float(np.nanmedian(base_arr)) if np.any(np.isfinite(base_arr)) else 0.0
    fill = np.where(np.isfinite(base_arr), base_arr, med).astype("float32")
    hill = LightSource(azdeg=315, altdeg=45).hillshade(fill, vert_exag=1.0)
    m_show = np.ma.masked_where(m_use <= 0, m_use)

    plt.figure(figsize=(10, 8), dpi=150)
    plt.imshow(hill, cmap="gray", extent=base_extent, origin="upper")
    plt.imshow(m_show, cmap="autumn", extent=base_extent, origin="upper", alpha=0.45, interpolation="nearest")
    try:
        plt.contour(m_use, levels=[0.5], colors="red", linewidths=1.0, extent=base_extent, origin="upper")
    except Exception:
        pass
    plt.title("Manual Landslide Mask (DXF)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=240)
    plt.close()
    return out_png.replace("\\", "/")


def run_mask_from_dxf(
    ctx: AnalysisContext,
    dxf_path: str,
    *,
    layer_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build ui1/landslide_mask.tif from DXF closed LWPOLYLINE entities.
    The mask grid follows ui1/dx.tif to stay compatible with downstream UI1/UI2/UI3.
    """
    dxf_path = os.path.abspath(str(dxf_path or "").strip())
    if not dxf_path or not os.path.exists(dxf_path):
        raise FileNotFoundError(f"DXF not found: {dxf_path}")

    dx_tif = os.path.join(ctx.out_ui1, "dx.tif")
    if not os.path.exists(dx_tif):
        raise FileNotFoundError("dx.tif is missing. Please run SAD before importing DXF mask.")

    rings, stats = _parse_lwpolylines_from_dxf(dxf_path, layer_name=layer_name)
    if not rings:
        raise RuntimeError(
            "No closed LWPOLYLINE found in DXF for mask creation. "
            f"Stats: kept={stats.get('lwpolyline_kept', 0)}, total={stats.get('lwpolyline_total', 0)}"
        )

    with rasterio.open(dx_tif) as ds:
        dx_meta = ds.meta.copy()
        transform = ds.transform
        out_shape = (ds.height, ds.width)
        crs = ds.crs

    geoms = [{"type": "Polygon", "coordinates": [ring]} for ring in rings]
    mask = rasterize(
        geoms,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8",
    )

    mask_tif = os.path.join(ctx.out_ui1, "landslide_mask.tif")
    os.makedirs(ctx.out_ui1, exist_ok=True)
    for k in ("dtype", "nodata"):
        dx_meta.pop(k, None)
    dx_meta.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="lzw")
    with rasterio.open(mask_tif, "w", **dx_meta) as dst:
        dst.write(mask.astype("uint8"), 1)

    overlay_png = os.path.join(ctx.out_ui1, "landslide_overlay.png")
    base_for_overlay = os.path.join(ctx.in_dir, "before.asc")
    if not os.path.exists(base_for_overlay):
        base_for_overlay = dx_tif
    overlay_png = _save_mask_overlay(mask_tif, base_for_overlay, overlay_png)

    meta_json = os.path.join(ctx.out_ui1, "mask_from_dxf_meta.json")
    meta_payload = {
        "dxf_path": dxf_path.replace("\\", "/"),
        "layer_name": (layer_name or ""),
        "mask_tif": mask_tif.replace("\\", "/"),
        "overlay_png": overlay_png,
        "grid_reference": dx_tif.replace("\\", "/"),
        "grid_crs": (str(crs) if crs is not None else None),
        "grid_shape": {"height": int(out_shape[0]), "width": int(out_shape[1])},
        "stats": {
            **stats,
            "mask_pixels_positive": int(np.count_nonzero(mask > 0)),
        },
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, ensure_ascii=False, indent=2)

    try:
        from pedi_oku_landslide.pipeline.ingest import update_ingest_processed
        update_ingest_processed(ctx.run_dir, slip_mask=mask_tif.replace("\\", "/"))
    except Exception as e:
        print(f"[WARN] Failed to update ingest_meta from DXF mask import: {e}")

    return {
        "mask_tif": mask_tif.replace("\\", "/"),
        "mask_png": overlay_png,
        "meta_json": meta_json.replace("\\", "/"),
        "dxf_path": dxf_path.replace("\\", "/"),
        "polygon_count": int(len(rings)),
        "mask_pixels_positive": int(np.count_nonzero(mask > 0)),
    }
