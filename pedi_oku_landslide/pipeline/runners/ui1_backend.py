import os
import time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.plot import plotting_extent
from rasterio.transform import xy as rio_xy
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from tqdm import tqdm
from scipy.ndimage import uniform_filter, map_coordinates, label, binary_fill_holes
from skimage import measure

import geopandas as gpd
from shapely.geometry import Polygon, LineString

from PyQt5.QtWidgets import QComboBox, QGroupBox, QFormLayout
from scipy.ndimage import uniform_filter
from matplotlib.colors import LightSource

#TIỆN ÍCH CHUNG
def load_asc(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()
    return arr, profile

def save_asc(path, data, profile, nodata_val=np.nan, dtype="float32"):
    prof = profile.copy()
    prof.update(driver="AAIGrid", dtype=dtype, count=1, nodata=nodata_val)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data, 1)

def save_hillshade(dem, out_png, smooth_size, profile, title):
    dem_filled = np.where(np.isnan(dem), np.nanmedian(dem), dem)
    smooth = uniform_filter(dem_filled, size=smooth_size, mode='nearest')
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(smooth, vert_exag=1.0)

    transform = profile['transform']
    xres = transform[0]
    yres = -transform[4]
    xmin = transform[2]
    ymax = transform[5]
    xmax = xmin + dem.shape[1] * xres
    ymin = ymax - dem.shape[0] * yres
    extent = [xmin, xmax, ymin, ymax]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(hs, cmap='gray', extent=extent, origin='upper')
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def run_UI1_1_crop(before_path, after_path, before_pz_path, after_pz_path,
                   output_dir="output/UI1/step1_crop", nodata_val=np.nan, smooth_size=11):
    import os, numpy as np, rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.errors import RasterioIOError
    from scipy.ndimage import uniform_filter
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from matplotlib.colors import LightSource

    def load_and_clean(path):
        with rasterio.open(path) as src:
            data = src.read(1).astype(float)
            profile = src.profile
            if src.crs is None:
                profile['crs'] = rasterio.crs.CRS.from_epsg(6677)
            transform = src.transform
        data[data < -100] = np.nan
        return data, profile, transform

    def crop_to_ref(data, profile, ref_profile):
        dst = np.full((ref_profile['height'], ref_profile['width']), np.nan, dtype='float32')
        reproject(
            source=data,
            destination=dst,
            src_transform=profile['transform'],
            src_crs=profile['crs'],
            dst_transform=ref_profile['transform'],
            dst_crs=ref_profile['crs'],
            resampling=Resampling.nearest
        )
        return dst

    def save_asc(path, data, profile):
        profile_out = profile.copy()
        profile_out.update({'dtype': 'float32', 'count': 1, 'nodata': nodata_val})
        with rasterio.open(path, 'w', **profile_out) as dst:
            dst.write(data.astype('float32'), 1)

    def save_hillshade_with_grid(data, transform, out_path, title, grid_interval=20, figsize=(8, 8)):
        from matplotlib.colors import LightSource
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        import numpy as np
    
        # Hillshade tính từ dữ liệu
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(np.nan_to_num(data, nan=np.nanmedian(data)), vert_exag=1)
    
        # Tính extent
        x_min = transform[2]
        x_max = x_min + transform[0] * data.shape[1]
        y_max = transform[5]
        y_min = y_max + transform[4] * data.shape[0]
    
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(hillshade, cmap='gray', extent=[x_min, x_max, y_min, y_max])
        ax.set_title(title, fontsize=10)
    
        # Đặt giới hạn trục grid
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
        # Lưới tọa độ
        ax.xaxis.set_major_locator(MultipleLocator(grid_interval))
        ax.yaxis.set_major_locator(MultipleLocator(grid_interval))
        ax.grid(which='major', color='red', linewidth=0.5, linestyle='--')
    
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


    os.makedirs(output_dir, exist_ok=True)
    try:
        b, pb, tb = load_and_clean(before_path)
        a, pa, ta = load_and_clean(after_path)
        bpz, pbpz, tbpz = load_and_clean(before_pz_path)
        apz, papz, tapz = load_and_clean(after_pz_path)
    except RasterioIOError as e:
        return f"[✗] Failed to read DEMs: {e}", {}

    # chọn ref nhỏ nhất
    profiles = [pb, pa, pbpz, papz]
    areas = [(p['width']*abs(p['transform'][0]))*(p['height']*abs(p['transform'][4])) for p in profiles]
    ref_idx = np.argmin(areas)
    ref_profile = profiles[ref_idx]

    b_crop = crop_to_ref(b, pb, ref_profile)
    a_crop = crop_to_ref(a, pa, ref_profile)
    bpz_crop = crop_to_ref(bpz, pbpz, ref_profile)
    apz_crop = crop_to_ref(apz, papz, ref_profile)

    def smooth(d):
        f = np.where(np.isnan(d), np.nanmedian(d), d)
        s = uniform_filter(f, size=smooth_size, mode='nearest')
        s[np.isnan(d)] = np.nan
        return s

    b_smooth = smooth(b_crop)
    a_smooth = smooth(a_crop)

    out_b = os.path.join(output_dir, "before_crop.asc")
    out_a = os.path.join(output_dir, "after_crop.asc")
    out_bpz = os.path.join(output_dir, "before_pz_crop.asc")
    out_apz = os.path.join(output_dir, "after_pz_crop.asc")

    save_asc(out_b, b_crop, ref_profile)
    save_asc(out_a, a_crop, ref_profile)
    save_asc(out_bpz, bpz_crop, ref_profile)
    save_asc(out_apz, apz_crop, ref_profile)

    hill_b = os.path.join(output_dir, "hill_before.png")
    hill_a = os.path.join(output_dir, "hill_after.png")
    hill_bs = os.path.join(output_dir, "hill_before_smooth.png")
    hill_as = os.path.join(output_dir, "hill_after_smooth.png")

    save_hillshade_with_grid(b_crop, ref_profile['transform'], hill_b, "Before DEM – Hillshade")
    save_hillshade_with_grid(a_crop, ref_profile['transform'], hill_a, "After DEM – Hillshade")
    save_hillshade_with_grid(b_smooth, ref_profile['transform'], hill_bs, "Before DEM – Smoothed Hillshade")
    save_hillshade_with_grid(a_smooth, ref_profile['transform'], hill_as, "After DEM – Smoothed Hillshade")

    return "[✓] Cropped and smoothed DEMs saved.", {
        'before_crop': out_b,
        'after_crop': out_a,
        'before_pz_crop': out_bpz,
        'after_pz_crop': out_apz,
        'hill_before': hill_b,
        'hill_after': hill_a,
        'hill_before_smooth': hill_bs,
        'hill_after_smooth': hill_as
    }


# Bước 2: UI1_2_sad (OpenCV + affine mapping)
def run_UI1_2_sad(before_path, after_path, output_dir="output/UI1/step2_sad",
                  patch_size_m=20, search_radius_m=2, cellsize=0.2, stride_m=0.2):
    """
    OpenCV template-matching trên HILLSHADE (đã smooth), dùng TM_SQDIFF (SSD).
    Đầu ra dX,dY theo mét, dấu trục chuẩn: x sang phải +; y lên trên + (i tăng xuống => dy = -off_y_px*cellsize)
    """
    import os, time
    import numpy as np, cv2
    os.makedirs(output_dir, exist_ok=True)

    # 1) Đọc DEM ASCII + profile
    before, profile = load_asc(before_path)
    after,  _       = load_asc(after_path)

    # 2) Chuyển DEM -> hillshade (đã smooth)
    def _hillshade_from_dem(arr):
        med = np.nanmedian(arr)
        if not np.isfinite(med): med = 0.0
        arr = np.where(~np.isfinite(arr), med, arr).astype(np.float32)
        k = max(3, int(round(2.0 / max(cellsize, 1e-6))))  # ~2 m theo cellsize
        arr_s = uniform_filter(arr, size=k)
        ls = LightSource(azdeg=315, altdeg=45)
        hs = ls.hillshade(arr_s, vert_exag=1.0).astype(np.float32)
        return hs

    before_hs = _hillshade_from_dem(before)
    after_hs  = _hillshade_from_dem(after)

    # 3) Chuẩn hóa NaN/Inf OpenCV
    def _prep(a):
        med = np.nanmedian(a)
        if not np.isfinite(med): med = 0.0
        a = np.where(~np.isfinite(a), med, a).astype(np.float32)
        return np.ascontiguousarray(a)

    before_f = _prep(before_hs)
    after_f  = _prep(after_hs)
    H, W = before_f.shape

    # 4) Tham số pixel
    PATCH   = int(round(patch_size_m   / max(cellsize,1e-6)))
    R       = int(round(search_radius_m/ max(cellsize,1e-6)))
    STRIDE  = max(1, int(round(stride_m / max(cellsize,1e-6))))
    if PATCH <= 1 or R < 1:
        return f"[✗] Tham số patch/search quá nhỏ: PATCH={PATCH}, R={R}", {}

    half = PATCH // 2
    # Patch kích thước đầy đủ
    if PATCH % 2 == 0:
        PATCH += 1
        half = PATCH // 2

    # 5) Vô hiệu hóa OpenCL/đa luồng để tránh crash
    try: cv2.ocl.setUseOpenCL(False)
    except: pass
    try: cv2.setNumThreads(1)
    except: pass

    dX = np.full((H, W), np.nan, dtype=np.float32)
    dY = np.full((H, W), np.nan, dtype=np.float32)
    score_map = np.full((H, W), np.nan, dtype=np.float32)

    rows = range(half + R, H - half - R, STRIDE)
    cols = range(half + R, W - half - R, STRIDE)

    start = time.time()
    for i in rows:
        i0, i1 = i - half, i + half + 1
        for j in cols:
            j0, j1 = j - half, j + half + 1
            tpl = before_f[i0:i1, j0:j1]
            if tpl.shape != (PATCH, PATCH):
                continue

            si0, si1 = i0 - R, i1 + R
            sj0, sj1 = j0 - R, j1 + R
            if si0 < 0 or sj0 < 0 or si1 > H or sj1 > W:
                continue
            sea = after_f[si0:si1, sj0:sj1]
            if sea.shape[0] < PATCH or sea.shape[1] < PATCH:
                continue

            # OpenCV TM_SQDIFF (SSD): min là tốt nhất
            try:
                res = cv2.matchTemplate(sea, tpl, cv2.TM_SQDIFF)
            except Exception:
                # Fallback: chuẩn hóa về uint8 nếu cần
                sea8 = cv2.normalize(sea, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                tpl8 = cv2.normalize(tpl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                res  = cv2.matchTemplate(sea8, tpl8, cv2.TM_SQDIFF)

            min_val, _, min_loc, _ = cv2.minMaxLoc(res)
            # Offset tính từ góc patch trong cửa sổ tìm kiếm
            off_x_px = min_loc[0] - R
            off_y_px = min_loc[1] - R

            # 6) Quy đổi về mét
            dX[i, j] =  off_x_px * cellsize
            dY[i, j] = -off_y_px * cellsize
            score_map[i, j] = float(min_val)

    # 7) Lưu
    dx_path = os.path.join(output_dir, "dX.asc")
    dy_path = os.path.join(output_dir, "dY.asc")
    score_path = os.path.join(output_dir, "score_map.asc")
    save_asc(dx_path, dX, profile)
    save_asc(dy_path, dY, profile)
    save_asc(score_path, score_map, profile)
    return "[✓] SAD (OpenCV/SSD) matching hoàn tất.", {
        "dX_path": dx_path, "dY_path": dy_path, "score_path": score_path
    }

# ước 2 (bổ sung): UI1_2_sad_original NumPy thuần

def run_UI1_2_sad_original(
    before_path: str,
    after_path: str,
    output_dir: str = "UI1_workspace/step2_vector_sad_original",
    patch_size_m: float = 20.0,       # kích thước patch (m)
    search_radius_m: float = 2.0,     # bán kính tìm kiếm (m)
    cellsize: float | None = None,    # None -> lấy từ metadata
    stride_m: float = 1.0,            # bước lấy mẫu (m) - mặc định 1.0 m
    hillshade_smooth_m: float = 2.0,  # làm mượt DEM trước khi hillshade (m)
    azdeg: float = 315.0,
    altdeg: float = 45.0,
    nodata_value: float = -9999.0,
):
    """
    Tính vector dịch chuyển (dX, dY) bằng SAD trên hillshade (DEM đã smooth).
    - dX dương sang phải; dY dương hướng Bắc (trục ảnh y ngược).
    - Xuất: dX.asc, dY.asc (ESRI ASCII, đơn vị: mét).
    Trả về: (status: str, outputs: dict)
    """
    import os
    import numpy as np
    import rasterio
    from matplotlib.colors import LightSource
    from scipy.ndimage import uniform_filter
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    # -------- helpers --------
    def _load_band(path):
        with rasterio.open(path) as src:
            arr = src.read(1, masked=True).filled(np.nan).astype(np.float32)
            profile = src.profile.copy()
            resx, resy = src.res
        return arr, profile, float(resx), float(resy)

    def _save_esri_ascii(path, data, ref_profile, nodata):
        # Ghi ESRI ASCII Grid thuần văn bản (tránh lỗi driver AAIGrid)
        from affine import Affine
        transform = ref_profile.get("transform", None)
        if transform is None:
            raise ValueError("Missing transform in profile; cannot write ESRI ASCII.")
        if not isinstance(transform, Affine):
            transform = Affine(*transform)

        nrows, ncols = data.shape
        cellsize_x = float(transform.a)
        cellsize_y = float(-transform.e)
        cellsize = cellsize_x if abs(cellsize_x - cellsize_y) < 1e-9 else cellsize_x
        xllcorner = float(transform.c)
        yllcorner = float(transform.f + transform.e * nrows)  # y_min

        out = np.where(np.isfinite(data), data, float(nodata)).astype(np.float32)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"ncols         {ncols}\n")
            f.write(f"nrows         {nrows}\n")
            f.write(f"xllcorner     {xllcorner:.6f}\n")
            f.write(f"yllcorner     {yllcorner:.6f}\n")
            f.write(f"cellsize      {cellsize:.6f}\n")
            f.write(f"NODATA_value  {nodata}\n")
            np.savetxt(f, out, fmt="%.6f")

    def _hillshade_from_dem(dem, cellsize_m, smooth_m, az, alt):
        # fill NaN -> median để ổn định
        med = np.nanmedian(dem)
        if not np.isfinite(med): med = 0.0
        dem = np.where(np.isfinite(dem), dem, med).astype(np.float32, copy=False)
        # smooth ~ 2 m
        k = max(3, int(round(max(smooth_m, cellsize_m) / max(cellsize_m, 1e-6))))
        dem_s = uniform_filter(dem, size=k, mode="nearest").astype(np.float32, copy=False)
        # hillshade
        ls = LightSource(azdeg=az, altdeg=alt)
        hs = ls.hillshade(dem_s, vert_exag=1.0).astype(np.float32)
        # đảm bảo finite & contiguous
        med_hs = np.nanmedian(hs)
        if not np.isfinite(med_hs): med_hs = 0.0
        hs = np.where(np.isfinite(hs), hs, med_hs).astype(np.float32, copy=False)
        return np.ascontiguousarray(hs)

    # -------- load & prepare --------
    before_dem, profile, rx, ry = _load_band(before_path)
    after_dem,  _,      _,  _   = _load_band(after_path)
    if cellsize is None:
        cellsize = float(rx)

    before_hs = _hillshade_from_dem(before_dem, cellsize, hillshade_smooth_m, azdeg, altdeg)
    after_hs  = _hillshade_from_dem(after_dem,  cellsize, hillshade_smooth_m, azdeg, altdeg)

    H, W = before_hs.shape

    # -------- params (pixels) --------
    PATCH = int(round(patch_size_m / max(cellsize, 1e-6)))
    if PATCH < 3: PATCH = 3
    if PATCH % 2 == 0: PATCH += 1
    HALF = PATCH // 2

    R = int(round(search_radius_m / max(cellsize, 1e-6)))
    STRIDE = max(1, int(round(stride_m / max(cellsize, 1e-6))))

    # giữ patch vừa ảnh (phòng AOI rất nhỏ)
    max_patch_allowed = max(3, min(H, W) - 2*max(1, R) - 1)
    if PATCH > max_patch_allowed:
        PATCH = max_patch_allowed if max_patch_allowed % 2 == 1 else max_patch_allowed - 1
        HALF = max(1, PATCH // 2)

    # pad reflect cho ảnh sau
    pad = R + HALF
    pad_mode = "reflect" if pad < min(H, W) else "edge"
    after_pad = np.pad(after_hs, pad_width=pad, mode=pad_mode)

    # -------- outputs --------
    dX = np.full((H, W), np.nan, dtype=np.float32)
    dY = np.full((H, W), np.nan, dtype=np.float32)

    # -------- SAD loop --------
    rows = range(HALF, H - HALF, STRIDE)
    cols = range(HALF, W - HALF, STRIDE)

    for i in tqdm(rows, desc="SAD (Original)", leave=False):
        i0, i1 = i - HALF, i + HALF + 1
        for j in cols:
            j0, j1 = j - HALF, j + HALF + 1
            tpl = before_hs[i0:i1, j0:j1]

            best = None
            best_di = best_dj = 0

            for di in range(-R, R + 1):
                Ai = (i + di) + pad - HALF
                for dj in range(-R, R + 1):
                    Aj = (j + dj) + pad - HALF
                    cand = after_pad[Ai:Ai + PATCH, Aj:Aj + PATCH]
                    score = float(np.abs(cand - tpl).sum())  # SAD
                    if (best is None) or (score < best):
                        best = score
                        best_di, best_dj = di, dj

            dX[i, j] =  best_dj * cellsize
            dY[i, j] = -best_di * cellsize

    # -------- save --------
    dx_path = os.path.join(output_dir, "dX.asc")
    dy_path = os.path.join(output_dir, "dY.asc")
    _save_esri_ascii(dx_path, dX, profile, nodata_value)
    _save_esri_ascii(dy_path, dY, profile, nodata_value)

    return (
        f"[✓] SAD (Original) → {output_dir} | stride_px={STRIDE}, patch_px={PATCH}, search_px={R}",
        {"dX_path": dx_path, "dY_path": dy_path}
    )

# Bước 3: UI1_3_plot_vector
def run_UI1_3_plot_vector(
    dem_path, dx_path, dy_path, slip_zone_path,
    vector_width=0.0025, vector_scale=10, vector_color="red",
    output_path="output/UI1/step3_vector_overlay.png", step=10
):
    assert vector_color in ['magenta', 'blue', 'red', 'green', 'yellow', 'black'], "Invalid vector color"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def load_masked(path):
        with rasterio.open(path) as src:
            return src.read(1, masked=True).astype("float32")

    dx = load_masked(dx_path)
    dy = load_masked(dy_path)
    mask = load_masked(slip_zone_path)
    H, W = dx.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))

    if dem_path.lower().endswith(".asc"):
        dem, _ = load_asc(dem_path)
        smoothed = uniform_filter(dem, size=11, mode='nearest')
        ls = LightSource(azdeg=315, altdeg=45)
        dem_display = ls.hillshade(smoothed, vert_exag=1.0)
        cmap = 'gray'
    else:
        dem_display = plt.imread(dem_path)
        cmap = None

    valid = (~dx.mask) & (~dy.mask) & (mask == 1)
    sub = np.zeros_like(valid, dtype=bool)
    sub[::step, ::step] = True
    valid &= sub

    plt.figure(figsize=(10, 10))
    plt.imshow(dem_display, cmap=cmap, origin='upper')
    plt.quiver(xx[valid], yy[valid],
               dx.data[valid] * vector_scale,
               dy.data[valid] * vector_scale,
               color=vector_color, width=vector_width,
               scale_units='xy', scale=1)
    plt.title("Displacement Vectors (within Slip Zone)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return f"[✓] Vector overlay saved to {output_path}", output_path


#Bước 4: UI1_3_plot_vector_geotiff
def run_UI1_3_plot_vector_geotiff(
    dem_path, dx_path, dy_path, slip_zone_path,
    output_png="output/UI1/step4_vector_geotiff.png",
    vector_scale=45, vector_color="red", stride=20,
    vector_width=3, headwidth=6, headlength=5, headaxislength=3
):
    if not os.path.exists(dem_path):
        return f"[✗] DEM file not found: {dem_path}", None

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        transform = dem_src.transform
        extent = plotting_extent(dem_src)

    dx, _ = load_asc(dx_path)
    dy, _ = load_asc(dy_path)
    mask, _ = load_asc(slip_zone_path)

    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return "[✗] No vectors to plot (empty slip zone)", None
    
    sample_mask = np.zeros_like(mask, dtype=bool)
    sample_mask[::stride, ::stride] = True
    sel = sample_mask[rows, cols]      # <-- tạo một lần
    rows = rows[sel]
    cols = cols[sel]

    x_coords = transform[2] + cols * transform[0] + transform[0] / 2
    y_coords = transform[5] + rows * transform[4] + transform[4] / 2

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(dem, cmap="gray", extent=extent, origin="upper")
    ax.quiver(x_coords, y_coords,
              dx[rows, cols], dy[rows, cols],
              color=vector_color, angles='xy', scale=vector_scale,
              width=vector_width / 1000.0,
              headwidth=headwidth, headlength=headlength, headaxislength=headaxislength)
    ax.set_title("Displacement Vectors (within Slip Zone)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    return f"[✓] Vector overlay saved to {output_png}", output_png


#Bước 5: UI1_4_dz
def run_UI1_4_dz(before_path, after_path, dx_path, dy_path,
                 output_dir="output/UI1/step5_dz", cellsize=0.2):
    os.makedirs(output_dir, exist_ok=True)

    before, profile = load_asc(before_path)
    after, _ = load_asc(after_path)
    dX, _ = load_asc(dx_path)
    dY, _ = load_asc(dy_path)

    H, W = before.shape
    dZ = np.full((H, W), np.nan, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    jj_shift = xx + dX / cellsize
    ii_shift = yy + dY / cellsize

    valid_mask = (~np.isnan(dX) & ~np.isnan(dY) &
                  (ii_shift >= 0) & (ii_shift < H - 1) &
                  (jj_shift >= 0) & (jj_shift < W - 1))

    coords = [ii_shift[valid_mask], jj_shift[valid_mask]]
    after_interp = np.full((H, W), np.nan, dtype=np.float32)
    after_interp[valid_mask] = map_coordinates(after, coords, order=1, mode='nearest')
    dZ[valid_mask] = after_interp[valid_mask] - before[valid_mask]

    dz_path = os.path.join(output_dir, "dZ.asc")
    save_asc(dz_path, dZ, profile)
    return f"[✓] dZ computed and saved to {dz_path}", {"dZ_path": dz_path}


#Bước 6: UI1_4_dz_direct
def run_UI1_4_dz_direct(before_path, after_path, output_dir="output/UI1/step6_dz_direct"):
    os.makedirs(output_dir, exist_ok=True)
    before, profile = load_asc(before_path)
    after, _ = load_asc(after_path)
    dZ = after - before
    dZ[np.isnan(before) | np.isnan(after)] = np.nan
    dz_path = os.path.join(output_dir, "dZ.asc")
    save_asc(dz_path, dZ, profile)
    return f"[✓] Direct dZ saved to {dz_path}", {"dZ_path": dz_path}


#Bước 7: UI1_5_detect_slipzone
def run_UI1_5_detect_slipzone(dx_path, dy_path, dz_path, threshold_mm,
                              output_dir="output/UI1/step7_slipzone", pixel_size=0.2):
    os.makedirs(output_dir, exist_ok=True)
    threshold = threshold_mm / 1000.0
    dx, profile = load_asc(dx_path)
    dy, _ = load_asc(dy_path)
    dz, _ = load_asc(dz_path)

    dxy = np.sqrt(dx**2 + dy**2)
    mask = dxy >= threshold
    labeled, _ = label(mask)
    sizes = np.bincount(labeled.ravel()); sizes[0] = 0
    largest = sizes.argmax()
    slip_mask = (labeled == largest)
    slip_mask_filled = binary_fill_holes(slip_mask)

    slip_asc = np.where(slip_mask_filled, 1, 0).astype(np.uint8)
    slip_asc_path = os.path.join(output_dir, "slip_zone.asc")
    save_asc(slip_asc_path, slip_asc, profile, nodata_val=0, dtype="uint8")

    dz_masked = np.where(slip_mask_filled, dz, np.nan)
    dz_slip_path = os.path.join(output_dir, "dZ_slipzone.asc")
    save_asc(dz_slip_path, dz_masked, profile)

    slip_png_path = os.path.join(output_dir, "slip_mask.png")
    area = slip_mask_filled.sum() * pixel_size**2
    dxy[dxy > 10000] = np.nan
    vmin, vmax = np.nanpercentile(dxy, 5), np.nanpercentile(dxy, 95)

    plt.figure(figsize=(6, 6))
    plt.imshow(dxy, cmap="inferno", origin="upper", vmin=vmin, vmax=vmax)
    contours = measure.find_contours(slip_mask_filled.astype(float), 0.5)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], 'lime', lw=1.5)
    plt.title(f"Slip Zone (dXY ≥ {threshold:.2f} m)")
    plt.text(10, 20, f"Slip Area: {area:.1f} m²", color="cyan", fontsize=11)
    plt.colorbar(label="dXY (m)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(slip_png_path, dpi=300)
    plt.close()

    return f"[✓] Slip zone detected. Saved to {output_dir}", {
        "slip_zone_asc": slip_asc_path,
        "dZ_slipzone_asc": dz_slip_path,
        "slip_mask_png": slip_png_path
    }


#Bước 8: UI1_5_extract_boundary
def run_UI1_5_extract_boundary(slip_zone_path, output_path):
    try:
        with rasterio.open(slip_zone_path) as src:
            mask = src.read(1)
            transform, crs = src.transform, src.crs
        contours = measure.find_contours(mask, 0.5)
        polygons = [Polygon([rio_xy(transform, r, c) for r, c in contour]) for contour in contours]
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
        gdf.to_file(output_path, driver="GPKG")
        return "[✓] Exported slip zone boundary.", output_path
    except Exception as e:
        return f"[✗] Failed to extract slip boundary: {e}", None


#Bước 9: UI1_6_export_vector_gpkg
def run_UI1_6_export_vector_gpkg(dx_path, dy_path, output_path, pixel_size=0.2, stride=10):
    try:
        dx, profile = load_asc(dx_path)
        dy, _ = load_asc(dy_path)
        transform, crs = profile["transform"], profile["crs"]

        vectors = []
        rows, cols = dx.shape
        for r in range(0, rows, stride):
            for c in range(0, cols, stride):
                if np.isnan(dx[r, c]) or np.isnan(dy[r, c]):
                    continue
                x0, y0 = rio_xy(transform, r, c)
                x1, y1 = x0 + dx[r, c], y0 + dy[r, c]
                vectors.append(LineString([(x0, y0), (x1, y1)]))

        gdf = gpd.GeoDataFrame(geometry=vectors, crs=crs)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdf.to_file(output_path, driver="GPKG")
        return "[✓] Exported vector displacement GPKG.", output_path
    except Exception as e:
        return f"[✗] Failed to export vector GPKG: {e}", None

