import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from rasterio.crs import CRS
from skimage import measure


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
