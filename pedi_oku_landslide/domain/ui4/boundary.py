"""UI4 DXF boundary reading and raster mask logic."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .types import (
    Polygon,
    MultiPolygon,
    _log,
    _pick_existing,
    _safe_float,
    ezdxf,
    rasterio,
    rio_shapes,
    unary_union,
)


def read_boundary_polygon_from_dxf(
    dxf_path: str,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Optional[Any]:
    """
    Read boundary entities from a DXF file and return a shapely Polygon.

    Strategy:
      - Collect closed LWPOLYLINE / POLYLINE entities as polygons
      - Union them; keep the largest-area polygon as the main boundary
      - Returns None if ezdxf is not installed or no valid boundary found
    """
    if ezdxf is None:
        _log(log_fn, "[UI4] ezdxf not installed, cannot read DXF boundary")
        return None
    if Polygon is None or unary_union is None:
        _log(log_fn, "[UI4] shapely not available, cannot process DXF boundary")
        return None

    dxf_path = os.path.abspath(str(dxf_path or ""))
    if not dxf_path or not os.path.exists(dxf_path):
        return None

    try:
        doc = ezdxf.readfile(dxf_path)
    except Exception as e:
        _log(log_fn, f"[UI4] Failed to read DXF: {dxf_path} ({e})")
        return None

    msp = doc.modelspace()
    polys = []

    # LWPOLYLINE
    for e in msp.query("LWPOLYLINE"):
        pts = [(p[0], p[1]) for p in e.get_points("xy")]
        if len(pts) < 3:
            continue
        if e.closed:
            polys.append(Polygon(pts))

    # POLYLINE (2D/3D)
    for e in msp.query("POLYLINE"):
        pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
        if len(pts) < 3:
            continue
        if e.is_closed:
            polys.append(Polygon(pts))

    if not polys:
        _log(log_fn, f"[UI4] No closed polyline found in DXF: {dxf_path}")
        return None

    # Clean invalid polygons
    polys = [p.buffer(0) for p in polys if p.is_valid or p.buffer(0).is_valid]
    if not polys:
        _log(log_fn, f"[UI4] All DXF polygons are invalid: {dxf_path}")
        return None

    geom = unary_union(polys)
    if isinstance(geom, Polygon):
        _log(log_fn, f"[UI4] DXF boundary polygon loaded: area={geom.area:.2f}")
        return geom
    if MultiPolygon is not None and isinstance(geom, MultiPolygon):
        biggest = max(list(geom.geoms), key=lambda g: float(getattr(g, "area", 0.0)))
        _log(log_fn, f"[UI4] DXF boundary: multi-polygon, using largest (area={biggest.area:.2f})")
        return biggest

    _log(log_fn, f"[UI4] Unexpected DXF boundary geometry type: {type(geom)}")
    return None


def apply_mask_to_raster(
    src_tif: str,
    mask_tif: str,
    out_tif: str,
    *,
    out_nodata: float = -9999.0,
    crop_to_mask_bbox: bool = True,
) -> Dict[str, Any]:
    """
    Reproject/resample `mask_tif` to `src_tif` grid and set pixels outside mask to NoData.
    """
    from .types import (
        _require_ui4_runtime_deps,
        _finite_raster_stats,
        Resampling,
        Window,
        window_transform,
    )

    _require_ui4_runtime_deps()
    src_tif = os.path.abspath(str(src_tif or ""))
    mask_tif = os.path.abspath(str(mask_tif or ""))
    out_tif = os.path.abspath(str(out_tif or ""))
    if not src_tif or not os.path.exists(src_tif):
        return {"ok": False, "error": f"Source raster not found: {src_tif}"}
    if not mask_tif or not os.path.exists(mask_tif):
        return {"ok": False, "error": f"Mask raster not found: {mask_tif}"}

    from .types import reproject as _reproject

    with rasterio.open(src_tif) as src:
        src_arr = src.read(1).astype(float)
        src_profile = src.profile.copy()
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata
        if src_nodata is not None:
            if np.isnan(src_nodata):
                src_arr[~np.isfinite(src_arr)] = np.nan
            else:
                src_arr[np.isclose(src_arr, float(src_nodata))] = np.nan

        with rasterio.open(mask_tif) as msk:
            mask_src = msk.read(1)
            mask_dst = np.zeros(src_arr.shape, dtype=np.uint8)
            _reproject(
                source=mask_src,
                destination=mask_dst,
                src_transform=msk.transform,
                src_crs=msk.crs,
                dst_transform=src_transform,
                dst_crs=src_crs,
                resampling=Resampling.nearest,
            )

    mask_bool = np.isfinite(mask_dst) & (mask_dst > 0)
    if not np.any(mask_bool):
        return {"ok": False, "error": f"Mask has no positive pixels after reprojection: {mask_tif}"}

    out_arr = np.where(mask_bool, src_arr, float(out_nodata)).astype(np.float32)
    out_profile = src_profile.copy()
    out_profile.update(nodata=float(out_nodata), dtype="float32", count=1, compress="deflate")

    if crop_to_mask_bbox:
        rows, cols = np.where(mask_bool)
        r0 = int(rows.min())
        r1 = int(rows.max()) + 1
        c0 = int(cols.min())
        c1 = int(cols.max()) + 1
        win = Window(col_off=c0, row_off=r0, width=(c1 - c0), height=(r1 - r0))
        out_arr = out_arr[r0:r1, c0:c1]
        out_profile.update(
            height=int(out_arr.shape[0]),
            width=int(out_arr.shape[1]),
            transform=window_transform(win, src_transform),
        )

    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    with rasterio.open(out_tif, "w", **out_profile) as dst:
        dst.write(out_arr, 1)

    stats_arr = out_arr.astype(float)
    stats_arr[np.isclose(stats_arr, float(out_nodata))] = np.nan
    return {
        "ok": True,
        "src_tif": src_tif,
        "mask_tif": mask_tif,
        "out_tif": out_tif,
        "cropped": bool(crop_to_mask_bbox),
        "shape": {"ny": int(out_arr.shape[0]), "nx": int(out_arr.shape[1])},
        "stats": _finite_raster_stats(stats_arr),
    }


def _ring_area_abs_xy(ring_xy: np.ndarray) -> float:
    if ring_xy.ndim != 2 or ring_xy.shape[0] < 3 or ring_xy.shape[1] < 2:
        return 0.0
    x = ring_xy[:, 0]
    y = ring_xy[:, 1]
    # Shoelace area; works for closed/unclosed ring.
    return float(abs(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))))


def _boundary_xy_from_mask_tif(mask_tif: str) -> Optional[np.ndarray]:
    mask_tif = os.path.abspath(str(mask_tif or ""))
    if not mask_tif or not os.path.exists(mask_tif) or rasterio is None:
        return None
    if rio_shapes is None:
        return None
    try:
        with rasterio.open(mask_tif) as src:
            arr = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                if np.isnan(nodata):
                    arr[~np.isfinite(arr)] = np.nan
                else:
                    arr[np.isclose(arr, float(nodata))] = np.nan
            mask_bool = np.isfinite(arr) & (arr > 0)
            if not np.any(mask_bool):
                return None

            best_xy = None
            best_area = 0.0
            for geom, val in rio_shapes(mask_bool.astype(np.uint8), mask=mask_bool, transform=src.transform):
                try:
                    if int(val) != 1:
                        continue
                    coords = (geom or {}).get("coordinates")
                    if not coords:
                        continue
                    ring = np.asarray(coords[0], dtype=float)
                    if ring.ndim != 2 or ring.shape[1] < 2 or ring.shape[0] < 3:
                        continue
                    area = _ring_area_abs_xy(ring[:, :2])
                    if area > best_area:
                        best_area = area
                        best_xy = ring[:, :2]
                except Exception:
                    continue
            return best_xy
    except Exception:
        return None


def _polygon_from_boundary_xy(boundary_xy: Optional[np.ndarray]) -> Optional[Any]:
    if boundary_xy is None or Polygon is None:
        return None
    try:
        ring = np.asarray(boundary_xy, dtype=float)
    except Exception:
        return None
    if ring.ndim != 2 or ring.shape[1] < 2 or ring.shape[0] < 3:
        return None
    if not np.allclose(ring[0, :2], ring[-1, :2]):
        ring = np.vstack([ring[:, :2], ring[0, :2]])
    else:
        ring = ring[:, :2]
    try:
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
    except Exception:
        return None
    if poly is None or poly.is_empty:
        return None
    if poly.geom_type == "MultiPolygon":
        geoms = list(getattr(poly, "geoms", []))
        if not geoms:
            return None
        poly = max(geoms, key=lambda g: float(getattr(g, "area", 0.0)))
    return poly


def _simplify_boundary_xy(boundary_xy: Optional[np.ndarray], tolerance_m: float) -> Optional[np.ndarray]:
    if boundary_xy is None:
        return None
    ring = np.asarray(boundary_xy, dtype=float)
    if ring.ndim != 2 or ring.shape[1] < 2 or ring.shape[0] < 3:
        return None
    if Polygon is None:
        return ring[:, :2]
    tol = _safe_float(tolerance_m)
    if not np.isfinite(tol) or tol <= 0:
        tol = 0.0
    try:
        poly = _polygon_from_boundary_xy(ring[:, :2])
        if poly is None:
            return ring[:, :2]
        if tol > 0:
            poly = poly.simplify(float(tol), preserve_topology=True)
        if poly is None or poly.is_empty:
            return ring[:, :2]
        return np.asarray(poly.exterior.coords, dtype=float)[:, :2]
    except Exception:
        return ring[:, :2]
