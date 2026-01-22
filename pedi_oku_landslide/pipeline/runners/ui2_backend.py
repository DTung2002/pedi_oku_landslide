import os
import io
import json
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import rasterio
from rasterio.transform import xy
from rasterio.transform import xy as rio_xy
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from typing import List, Tuple, Optional, Dict
import numpy as np
import rasterio
from rasterio.transform import xy as rio_xy
from shapely.geometry import LineString
from typing import Dict, List, Tuple, Optional

def default_paths(base_root: str = "output") -> Dict[str, str]:
    """
    Standardized locations for UI1 outputs used by UI2.
    """
    return {
        "dem_path": os.path.join(base_root, "UI1", "step1_crop", "before_crop.asc"),
        "dx_path": os.path.join(base_root, "UI1", "step2_sad", "dX.asc"),
        "dy_path": os.path.join(base_root, "UI1", "step2_sad", "dY.asc"),
        "slip_path": os.path.join(base_root, "UI1", "step7_slipzone", "slip_zone.asc"),
    }


def map_to_xy(transform, row: float, col: float):
    # x = c + a*col + b*row ; y = f + d*col + e*row
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    x = c + a * col + b * row
    y = f + d * col + e * row
    return float(x), float(y)

def _read_raster_array(path: str) -> Tuple[np.ndarray, rasterio.Affine, Optional[object]]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        transform = src.transform
        crs = src.crs
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
    """
    Vẽ DEM + vector dịch chuyển (dx, dy) + viền slip zone (nếu có)
    Tất cả hiển thị trong hệ tọa độ map (CRS của DEM).
    """
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(dem_path) as src_dem:
        dem = src_dem.read(1)
        transform = src_dem.transform  
        crs = src_dem.crs
        H, W = dem.shape
        bounds = src_dem.bounds  

    with rasterio.open(dx_path) as src_dx:
        dx = src_dx.read(1)
    with rasterio.open(dy_path) as src_dy:
        dy = src_dy.read(1)

    rows = np.arange(0, H, stride)
    cols = np.arange(0, W, stride)
    C, R = np.meshgrid(cols, rows)  # C: col (x pix), R: row (y pix)

    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    Xs_map = c + a * C + b * R
    Ys_map = f + d * C + e * R

    dXs_m = dx[::stride, ::stride].astype(float)
    dYs_m = dy[::stride, ::stride].astype(float)

    if slip_mask_path and os.path.exists(slip_mask_path):
        with rasterio.open(slip_mask_path) as src_m:
            mask_full = src_m.read(1)
            tm = src_m.transform
        m = (mask_full[::stride, ::stride] > 0)

        Xs_map = Xs_map[m]
        Ys_map = Ys_map[m]
        dXs_m  = dXs_m[m]
        dYs_m  = dYs_m[m]
    else:
        Xs_map = Xs_map.ravel()
        Ys_map = Ys_map.ravel()
        dXs_m  = dXs_m.ravel()
        dYs_m  = dYs_m.ravel()

    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    ax.imshow(
        dem, cmap="gray", origin="upper",
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
    )

    # PNG full-bleed (không margin) + lưới & tick số vẽ thủ công bên trong ảnh
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if show_grid:
        xs = np.arange(np.ceil(bounds.left / grid_interval) * grid_interval,
                       bounds.right + 1e-9, grid_interval)
        ys = np.arange(np.ceil(bounds.bottom / grid_interval) * grid_interval,
                       bounds.top + 1e-9, grid_interval)
        # Lưới
        for xg in xs:
            ax.plot([xg, xg], [bounds.bottom, bounds.top],
                    ls="--", lw=0.6, color="white", alpha=0.6)
        for yg in ys:
            ax.plot([bounds.left, bounds.right], [yg, yg],
                    ls="--", lw=0.6, color="white", alpha=0.6)
        # Tick số (ghi ngay trong khung ảnh, không cần margin)
        fs = 8
        for xg in xs:
            ax.annotate(f"{xg:.0f}", (xg, bounds.bottom),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=fs, color='red', alpha=0.85)
        for yg in ys:
            ax.annotate(f"{yg:.0f}", (bounds.left, yg),
                        xytext=(3, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=fs, color='red', alpha=0.85)

    if slip_mask_path and os.path.exists(slip_mask_path):
        mask_bin = (mask_full > 0).astype(np.uint8)
        contours = measure.find_contours(mask_bin, 0.5)
        a2, b2, c2, d2, e2, f2 = tm.a, tm.b, tm.c, tm.d, tm.e, tm.f
        for cnt in contours:
            rr = cnt[:, 0]  # rows (float)
            cc = cnt[:, 1]  # cols (float)
            xs = c2 + a2 * cc + b2 * rr
            ys = f2 + d2 * cc + e2 * rr
            ax.plot(xs, ys, color="cyan", linewidth=1.3)

    ax.quiver(
        Xs_map, Ys_map, dXs_m * scale, dYs_m * scale,
        color=vector_color,
        scale_units="xy", scale=1,
        angles="xy", pivot="middle",
        headwidth=3, headlength=5, headaxislength=3, width=0.004
    )

    ax.set_aspect("equal")
    ax.tick_params(axis='both', labelsize=8, colors='#333333', length=3)

    out_png = os.path.join(output_dir, output_name)
    FigureCanvas(fig).print_png(out_png)
    plt.close(fig)

    return f"[✓] Vector overlay saved to {out_png}", {
        "png_path": out_png, "transform": transform, "crs": crs
    }

def _read_raster_array(path):
    """Đọc raster -> (array, transform, crs)"""
    with rasterio.open(path) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs
    return arr, transform, crs


def save_selected_lines_gpkg(
    lines: List[LineString],
    crs,
    output_path: str = "output/UI2/step2_selected_lines/selected_lines.gpkg",
):
    from rasterio.crs import CRS
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Chuẩn hóa CRS
    if crs is None:
        crs = CRS.from_epsg(6677)  # fallback, cùng logic với UI1 khi thiếu CRS
    elif isinstance(crs, str):
        try:
            crs = CRS.from_string(crs)
        except Exception:
            crs = CRS.from_epsg(6677)

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



def _unit(v: np.ndarray, eps=1e-9):
    n = float(np.hypot(v[0], v[1]))
    return (v / n) if n > eps else np.array([1.0, 0.0], dtype=float), n

def _centroid_from_mask(mask: np.ndarray, transform) -> Tuple[float, float]:
    rr, cc = np.nonzero(mask)
    if len(rr) == 0:
        return None
    r0, c0 = rr.mean(), cc.mean()
    x0, y0 = rio_xy(transform, r0, c0, offset="center")
    return float(x0), float(y0)

def _pca_dir_from_mask(mask: np.ndarray, transform) -> np.ndarray:
    rr, cc = np.nonzero(mask)
    if len(rr) < 2:
        return np.array([1.0, 0.0])
    xs, ys = rio_xy(transform, rr, cc, offset="center")
    X = np.vstack([np.array(xs) - np.mean(xs), np.array(ys) - np.mean(ys)]).T
    C = np.cov(X.T)
    vals, vecs = np.linalg.eigh(C)  # vecs[:,i] cột là eigenvector
    d = vecs[:, np.argmax(vals)]
    # hướng dọc “chiều dài” vùng – chuẩn hóa
    u, _ = _unit(d)
    return u

def _build_line(center_xy: Tuple[float, float], u_dir: np.ndarray, length: float) -> LineString:
    cx, cy = center_xy
    half = 0.5 * length
    dx, dy = u_dir * half
    return LineString([(cx - dx, cy - dy), (cx + dx, cy + dy)])

def _bbox_length_from_mask(mask: np.ndarray, transform, scale=1.2) -> float:
    rr, cc = np.nonzero(mask)
    if len(rr) == 0:
        return 100.0  # fallback
    xs, ys = rio_xy(transform, rr, cc, offset="center")
    L = max((max(xs) - min(xs)), (max(ys) - min(ys)))
    return float(max(L * scale, 50.0))

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
    min_mag_thresh: float = 1e-4
) -> Dict[str, List[Dict]]:
    """
    Trả về dict: {"main": [feat...], "cross": [feat...]}.
    Mỗi feat: {"name","type","offset_m","angle_deg","geom"}.
    """
    with rasterio.open(dem_path) as src:
        transform = src.transform

    dx = rasterio.open(dx_path).read(1).astype(float)
    dy = rasterio.open(dy_path).read(1).astype(float)
    mask = (rasterio.open(slip_mask_path).read(1) > 0)

    # centroid
    center = _centroid_from_mask(mask, transform)
    if center is None:
        return {"main": [], "cross": []}

    # vector trung bình trong slip zone
    m = mask & np.isfinite(dx) & np.isfinite(dy)
    if m.sum() == 0:
        v = np.array([1.0, 0.0])
    else:
        v = np.array([np.nanmean(dx[m]), np.nanmean(dy[m])], dtype=float)

    u_main, mag = _unit(v)
    if mag < min_mag_thresh:
        u_main = _pca_dir_from_mask(mask, transform)

    # pháp tuyến
    u_norm = np.array([-u_main[1], u_main[0]], dtype=float)

    # chiều dài cơ sở
    L = base_length_m if base_length_m and base_length_m > 0 else _bbox_length_from_mask(mask, transform)

    feats_main, feats_cross = [], []

    # #1
    main1 = _build_line(center, u_main, L)
    cross1 = _build_line(center, u_norm, L)

    ang_main = np.degrees(np.arctan2(u_main[1], u_main[0]))
    ang_cross = np.degrees(np.arctan2(u_norm[1], u_norm[0]))

    feats_main.append({"name": "ML-001", "type": "main", "offset_m": 0.0, "angle_deg": ang_main, "geom": main1})
    feats_cross.append({"name": "CL-001", "type": "cross", "offset_m": 0.0, "angle_deg": ang_cross, "geom": cross1})

    # bổ sung (đối xứng 2 bên)
    def _pairs(n_even: int):
        n_even = max(0, int(n_even))
        n_even -= n_even % 2
        for k in range(1, n_even // 2 + 1):
            yield k, +1
            yield k, -1

    # Main bổ sung: offset theo u_norm
    idx = 2
    for k in range(1, max(0, main_num_even) // 2 + 1):
        for sgn in (+1, -1):
            off = sgn * k * float(main_offset_m)
            cx = center[0] + off * u_norm[0]
            cy = center[1] + off * u_norm[1]
            geom = _build_line((cx, cy), u_main, L)
            feats_main.append({
                "name": f"ML-{idx:03d}",
                "type": "main",
                "offset_m": off,
                "angle_deg": ang_main,
                "geom": geom
            })
            idx += 1

    # Cross bổ sung: offset theo u_main
    idx = 2
    for k in range(1, max(0, cross_num_even) // 2 + 1):
        for sgn in (+1, -1):
            off = sgn * k * float(cross_offset_m)
            cx = center[0] + off * u_main[0]
            cy = center[1] + off * u_main[1]
            geom = _build_line((cx, cy), u_norm, L)
            feats_cross.append({
                "name": f"CL-{idx:03d}",
                "type": "cross",
                "offset_m": off,
                "angle_deg": ang_cross,
                "geom": geom
            })
            idx += 1

    return {"main": feats_main, "cross": feats_cross}

def write_shared_json(
    before_file: str,
    after_file: str,
    workspace: str = "output/UI2",
    **extras
):
    """
    Lưu ui_shared_data.json cho UI3.
    Giữ backward-compat: nếu không có extras vẫn OK.
    """
    import json, os
    os.makedirs(workspace, exist_ok=True)
    json_path = os.path.join(workspace, "ui_shared_data.json")
    data = {
        "before_file": before_file,
        "after_file": after_file,
        "workspace": workspace
    }
    # merge các key optional (bỏ None để file gọn gàng)
    for k, v in (extras or {}).items():
        if v is not None:
            data[k] = v
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"[✓] Shared data saved to {json_path}", {"json_path": json_path}
