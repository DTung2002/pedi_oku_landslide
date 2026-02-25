import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# =========================
# INPUTS
# =========================
dem_path   = r"2021-04-11_ground.tif"
slip_path  = r"slip_surface_kriging.tif"
depth_path = r"slip_depth_kriging.tif"
var_path   = r"slip_depth_kriging_variance.tif"

# Hiển thị contour (m)
slip_contour_step  = 5.0
depth_contour_step = 2.0

# Downsample khi vẽ 3D (để nhanh và nhẹ)
ds3d = 6  # lấy mỗi ds3d pixel một lần (tùy máy bạn tăng/giảm)

# =========================
# HELPERS
# =========================
def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        transform = src.transform
        bounds = src.bounds
        crs = src.crs
    return arr, transform, bounds, crs

def grid_xy(transform, shape):
    ny, nx = shape
    cols = np.arange(nx)
    rows = np.arange(ny)
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    X, Y = np.meshgrid(xs, ys)
    return X, Y

def levels_from_step(arr, step):
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    start = np.floor(vmin / step) * step
    stop  = np.ceil(vmax / step) * step
    return np.arange(start, stop + step, step)

def hillshade(Z, azimuth=315, altitude=45):
    # Hillshade đơn giản (không cần GDAL)
    # Z là mảng, giả định pixel square tương đối; chỉ để nhìn nổi khối
    Z = np.nan_to_num(Z, nan=np.nanmedian(Z))
    dy, dx = np.gradient(Z)
    slope = np.pi/2 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.deg2rad(azimuth)
    alt = np.deg2rad(altitude)
    hs = np.sin(alt)*np.sin(slope) + np.cos(alt)*np.cos(slope)*np.cos(az - aspect)
    return np.clip(hs, 0, 1)

# =========================
# READ DATA
# =========================
DEM, t_dem, b_dem, _ = read_raster(dem_path)
SLIP, t_s, b_s, _    = read_raster(slip_path)
DEPTH, t_d, b_d, _   = read_raster(depth_path)
VAR, t_v, b_v, _     = read_raster(var_path)

# Tạo lưới XY cho slip/depth (đã cùng hệ quy chiếu và extent với raster kriging)
X, Y = grid_xy(t_s, SLIP.shape)

# =========================
# 1) 2D MAP: DEM + slip surface + contours
# =========================
hs = hillshade(DEM)

plt.figure(figsize=(12, 9))
extent = (b_s.left, b_s.right, b_s.bottom, b_s.top)

# Nền hillshade từ DEM để có cảm giác địa hình
plt.imshow(hs, extent=extent, origin="upper", alpha=0.6)

# Tô slip surface (semi-transparent)
img = plt.imshow(SLIP, extent=extent, origin="upper", alpha=0.65)
plt.colorbar(img, label="Slip surface elevation (m)")

# Contour slip surface
slip_levels = levels_from_step(SLIP, slip_contour_step)
cs = plt.contour(X, Y, SLIP, levels=slip_levels, linewidths=0.8)
plt.clabel(cs, inline=True, fontsize=8, fmt="%.0f")

plt.gca().set_aspect("equal", adjustable="box")
plt.title(f"Slip surface (color) + contours (step={slip_contour_step} m) over DEM hillshade")
plt.xlabel("X"); plt.ylabel("Y")
plt.tight_layout()
plt.show()

# =========================
# 2) 2D MAP: depth + contours + optional variance mask
# =========================
plt.figure(figsize=(12, 9))

# Nếu muốn mask vùng variance quá cao, bạn có thể bật đoạn dưới:
# thr = np.nanpercentile(VAR, 85)  # ví dụ: bỏ 15% vùng bất định nhất
# DEPTH_plot = np.where(VAR <= thr, DEPTH, np.nan)
DEPTH_plot = DEPTH

img2 = plt.imshow(DEPTH_plot, extent=extent, origin="upper")
plt.colorbar(img2, label="Slip depth (m)")

depth_levels = levels_from_step(DEPTH_plot, depth_contour_step)
cs2 = plt.contour(X, Y, DEPTH_plot, levels=depth_levels, linewidths=0.8)
plt.clabel(cs2, inline=True, fontsize=8, fmt="%.0f")

plt.gca().set_aspect("equal", adjustable="box")
plt.title(f"Slip depth (color) + contours (step={depth_contour_step} m)")
plt.xlabel("X"); plt.ylabel("Y")
plt.tight_layout()
plt.show()

# =========================
# 3) 3D SURFACE: slip surface
# =========================
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Downsample để plot nhanh
SLIP3 = SLIP[::ds3d, ::ds3d]
X3 = X[::ds3d, ::ds3d]
Y3 = Y[::ds3d, ::ds3d]

# Bỏ NaN để plot sạch hơn (mask)
mask3 = np.isfinite(SLIP3)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

# Vì plot_surface không nhận mask rời rạc tốt, ta set NaN -> không vẽ
Z3 = np.where(mask3, SLIP3, np.nan)

surf = ax.plot_surface(X3, Y3, Z3, rstride=1, cstride=1, linewidth=0, antialiased=True)
fig.colorbar(surf, ax=ax, shrink=0.6, label="Slip surface elevation (m)")

ax.set_title("3D Slip Surface (downsampled)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z (m)")
plt.tight_layout()
plt.show()
