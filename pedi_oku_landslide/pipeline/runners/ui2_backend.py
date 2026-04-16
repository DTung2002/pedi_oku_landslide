import json
import os
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from rasterio.crs import CRS
from rasterio.transform import xy as rio_xy
from shapely.geometry import LineString, MultiLineString
from skimage import measure

from .ui2.ui2_auto_lines import generate_auto_lines_from_arrays
from .ui2.ui2_intersections import build_main_cross_intersections, save_main_cross_intersections
from .ui2.ui2_paths import validate_run_inputs
from .ui2.ui2_polylines_storage import read_polylines_json, write_polylines_json
from .ui2.ui2_raster import load_layers as load_run_layers
from .ui2.ui2_raster import validate_context as validate_run_context
from .ui2.ui2_sections_storage import (
    SECTION_CHAINAGE_ORIGIN,
    SECTION_CSV_FIELDNAMES,
    SECTION_DIRECTION_VERSION,
    ensure_sections_csv_current,
    read_sections_csv_rows,
    write_sections_csv_rows,
)


DEFAULT_CRS = CRS.from_epsg(6678)


def default_paths(base_root: str = "output") -> Dict[str, str]:
    return {
        "dem_path": os.path.join(base_root, "UI1", "step1_crop", "before_crop.asc"),
        "dx_path": os.path.join(base_root, "UI1", "step2_sad", "dX.asc"),
        "dy_path": os.path.join(base_root, "UI1", "step2_sad", "dY.asc"),
        "slip_path": os.path.join(base_root, "UI1", "step7_slipzone", "slip_zone.asc"),
    }


def map_to_xy(transform, row: float, col: float):
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    x = c + a * col + b * row
    y = f + d * col + e * row
    return float(x), float(y)


def _read_raster_array(path: str) -> Tuple[np.ndarray, rasterio.Affine, Optional[object]]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        transform = src.transform
        crs = DEFAULT_CRS
    return arr, transform, crs


def generate_vector_overlay_image(
    dem_path: str,
    dx_path: str,
    dy_path: str,
    slip_mask_path: Optional[str] = None,
    stride: int = 20,
    scale: float = 1.0,
    vector_color: str = "red",
    output_dir: str = "output/UI2/step1_vector_overlay",
    output_name: str = "vector_overlay.png",
    show_grid: bool = True,
    grid_interval: float = 20.0,
):
    os.makedirs(output_dir, exist_ok=True)
    with rasterio.open(dem_path) as src_dem:
        dem = src_dem.read(1)
        transform = src_dem.transform
        crs = DEFAULT_CRS
        H, W = dem.shape
        bounds = src_dem.bounds
    with rasterio.open(dx_path) as src_dx:
        dx = src_dx.read(1)
    with rasterio.open(dy_path) as src_dy:
        dy = src_dy.read(1)

    rows = np.arange(0, H, stride)
    cols = np.arange(0, W, stride)
    C, R = np.meshgrid(cols, rows)
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    Xs_map = c + a * C + b * R
    Ys_map = f + d * C + e * R
    dXs_m = dx[::stride, ::stride].astype(float)
    dYs_m = dy[::stride, ::stride].astype(float)

    if slip_mask_path and os.path.exists(slip_mask_path):
        with rasterio.open(slip_mask_path) as src_m:
            mask_full = src_m.read(1)
            tm = src_m.transform
        m = mask_full[::stride, ::stride] > 0
        Xs_map = Xs_map[m]
        Ys_map = Ys_map[m]
        dXs_m = dXs_m[m]
        dYs_m = dYs_m[m]
    else:
        Xs_map = Xs_map.ravel()
        Ys_map = Ys_map.ravel()
        dXs_m = dXs_m.ravel()
        dYs_m = dYs_m.ravel()

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(dem, cmap="gray", origin="upper", extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if show_grid:
        xs = np.arange(np.ceil(bounds.left / grid_interval) * grid_interval, bounds.right + 1e-9, grid_interval)
        ys = np.arange(np.ceil(bounds.bottom / grid_interval) * grid_interval, bounds.top + 1e-9, grid_interval)
        for xg in xs:
            ax.plot([xg, xg], [bounds.bottom, bounds.top], ls="--", lw=0.6, color="white", alpha=0.6)
        for yg in ys:
            ax.plot([bounds.left, bounds.right], [yg, yg], ls="--", lw=0.6, color="white", alpha=0.6)
        fs = 8
        for xg in xs:
            ax.annotate(f"{xg:.0f}", (xg, bounds.bottom), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=fs, color="red", alpha=0.85)
        for yg in ys:
            ax.annotate(f"{yg:.0f}", (bounds.left, yg), xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=fs, color="red", alpha=0.85)

    if slip_mask_path and os.path.exists(slip_mask_path):
        mask_bin = (mask_full > 0).astype(np.uint8)
        contours = measure.find_contours(mask_bin, 0.5)
        a2, b2, c2, d2, e2, f2 = tm.a, tm.b, tm.c, tm.d, tm.e, tm.f
        for cnt in contours:
            rr = cnt[:, 0]
            cc = cnt[:, 1]
            xs = c2 + a2 * cc + b2 * rr
            ys = f2 + d2 * cc + e2 * rr
            ax.plot(xs, ys, color="cyan", linewidth=1.3)

    ax.quiver(Xs_map, Ys_map, dXs_m * scale, dYs_m * scale, color=vector_color, scale_units="xy", scale=1, angles="xy", pivot="middle", headwidth=3, headlength=5, headaxislength=3, width=0.004)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=8, colors="#333333", length=3)
    out_png = os.path.join(output_dir, output_name)
    FigureCanvas(fig).print_png(out_png)
    plt.close(fig)
    return f"[✓] Vector overlay saved to {out_png}", {"png_path": out_png, "transform": transform, "crs": crs}


def save_selected_lines_gpkg(
    lines: List[LineString],
    crs,
    output_path: str = "output/UI2/step2_selected_lines/selected_lines.gpkg",
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if crs is None:
        crs = DEFAULT_CRS
    elif isinstance(crs, str):
        try:
            crs = CRS.from_string(crs)
        except Exception:
            crs = DEFAULT_CRS
    gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)
    gdf.to_file(output_path, driver="GPKG")
    return f"[✓] Saved {len(lines)} lines → {output_path}", {"gpkg_path": output_path}


def read_vector_lines(file_path: str) -> Tuple[List[LineString], Optional[object]]:
    gdf = gpd.read_file(file_path)
    crs = gdf.crs
    lines: List[LineString] = []
    for geom in gdf.geometry:
        if isinstance(geom, MultiLineString):
            for g in geom.geoms:
                if isinstance(g, LineString) and len(g.coords) >= 2:
                    lines.append(LineString(g.coords))
        elif isinstance(geom, LineString) and len(geom.coords) >= 2:
            lines.append(LineString(geom.coords))
    return lines, crs


def generate_auto_lines_from_slipzone(
    dem_path: str,
    dx_path: str,
    dy_path: str,
    slip_mask_path: str,
    main_num_even: int,
    main_offset_m: float,
    cross_num_even: int,
    cross_offset_m: float,
    base_length_m: Optional[float] = None,
    min_mag_thresh: float = 1e-4,
) -> Dict[str, List[Dict]]:
    with rasterio.open(dem_path) as src:
        transform = src.transform
    with rasterio.open(dx_path) as src_dx:
        dx = src_dx.read(1).astype(float)
    with rasterio.open(dy_path) as src_dy:
        dy = src_dy.read(1).astype(float)
    with rasterio.open(slip_mask_path) as src_mask:
        mask = (src_mask.read(1) > 0)
    return generate_auto_lines_from_arrays(
        dx=dx,
        dy=dy,
        mask=mask,
        transform=transform,
        main_num_even=main_num_even,
        main_offset_m=main_offset_m,
        cross_num_even=cross_num_even,
        cross_offset_m=cross_offset_m,
        base_length_m=base_length_m,
        min_mag_thresh=min_mag_thresh,
    )


def write_shared_json(before_file: str, after_file: str, workspace: str = "output/UI2", **extras):
    os.makedirs(workspace, exist_ok=True)
    json_path = os.path.join(workspace, "ui_shared_data.json")
    data = {"before_file": before_file, "after_file": after_file, "workspace": workspace}
    for k, v in (extras or {}).items():
        if v is not None:
            data[k] = v
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"[✓] Shared data saved to {json_path}", {"json_path": json_path}


class UI2BackendService:
    def __init__(self) -> None:
        self._ctx: Dict[str, Any] = {}

    def set_context(self, project: str, run_label: str, run_dir: str, base_dir: str = "") -> Dict[str, Any]:
        validated = validate_run_context(run_dir)
        self._ctx = {
            "project": project or "",
            "run_label": run_label or "",
            "run_dir": run_dir or "",
            "base_dir": base_dir or "",
            **validated,
        }
        return dict(self._ctx)

    def validate_context(self, run_dir: str) -> Dict[str, Any]:
        return validate_run_context(run_dir)

    def load_layers(self, run_dir: str, vector_settings: Dict[str, Any]) -> Dict[str, Any]:
        return load_run_layers(run_dir, vector_settings)

    def load_sections(self, run_dir: str) -> Dict[str, Any]:
        csv_path = os.path.join(run_dir, "ui2", "sections.csv")
        if not os.path.isfile(csv_path):
            return {"rows": [], "migrated": False, "csv_path": csv_path}
        rows, migrated = ensure_sections_csv_current(csv_path, run_dir=run_dir)
        return {"rows": rows, "migrated": migrated, "csv_path": csv_path}

    def save_sections(self, run_dir: str, rows: List[Dict[str, Any]]) -> str:
        csv_path = os.path.join(run_dir, "ui2", "sections.csv")
        write_sections_csv_rows(csv_path, rows)
        return csv_path

    def load_polylines(self, run_dir: str) -> Dict[str, Any]:
        json_path = os.path.join(run_dir, "ui2", "polylines.json")
        if not os.path.isfile(json_path):
            return {"rows": [], "json_path": json_path}
        return {"rows": read_polylines_json(json_path), "json_path": json_path}

    def save_polylines(self, run_dir: str, rows: List[Dict[str, Any]]) -> str:
        json_path = os.path.join(run_dir, "ui2", "polylines.json")
        write_polylines_json(json_path, rows)
        return json_path

    def generate_auto_lines(self, dx, dy, mask, transform, params: Dict[str, Any]) -> Dict[str, Any]:
        return generate_auto_lines_from_arrays(
            dx=dx,
            dy=dy,
            mask=mask,
            transform=transform,
            main_num_even=int(params.get("main_num_even", 0)),
            main_offset_m=float(params.get("main_offset_m", 0.0)),
            cross_num_even=int(params.get("cross_num_even", 0)),
            cross_offset_m=float(params.get("cross_offset_m", 0.0)),
            base_length_m=params.get("base_length_m"),
            min_mag_thresh=float(params.get("min_mag_thresh", 1e-4)),
        )

    def build_main_cross_intersections(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return build_main_cross_intersections(rows)

    def save_main_cross_intersections(self, run_dir: str, rows: List[Dict[str, Any]]) -> Optional[str]:
        return save_main_cross_intersections(os.path.join(run_dir, "ui2"), rows)

    def read_sections_csv_rows(self, csv_path: str) -> List[Dict[str, Any]]:
        return read_sections_csv_rows(csv_path)

    def write_sections_csv_rows(self, csv_path: str, rows: List[Dict[str, Any]]) -> None:
        write_sections_csv_rows(csv_path, rows)

    def reset_context(self) -> None:
        self._ctx.clear()


__all__ = [
    "SECTION_CHAINAGE_ORIGIN",
    "SECTION_CSV_FIELDNAMES",
    "SECTION_DIRECTION_VERSION",
    "UI2BackendService",
    "default_paths",
    "generate_auto_lines_from_slipzone",
    "generate_vector_overlay_image",
    "map_to_xy",
    "read_vector_lines",
    "save_selected_lines_gpkg",
    "validate_run_inputs",
    "write_shared_json",
]
