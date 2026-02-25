import numpy as np
import rasterio
import matplotlib.pyplot as plt

tif_path = "slip_surface_kriging.tif"  # hoặc slip_depth_kriging_0p5m.tif
interval = 2.0  # khoảng cách đường đồng mức (m). Ví dụ 0.5, 1, 2...

with rasterio.open(tif_path) as src:
    Z = src.read(1).astype(float)
    nodata = src.nodata
    Z[Z == nodata] = np.nan
    transform = src.transform

# Tạo lưới tọa độ X,Y từ transform
ny, nx = Z.shape
xs = transform.c + (np.arange(nx) + 0.5) * transform.a
ys = transform.f + (np.arange(ny) + 0.5) * transform.e  # e thường âm
X, Y = np.meshgrid(xs, ys)

# Chọn levels tự động theo interval
zmin = np.nanmin(Z)
zmax = np.nanmax(Z)
levels = np.arange(np.floor(zmin / interval) * interval,
                   np.ceil(zmax / interval) * interval + interval,
                   interval)

plt.figure(figsize=(10, 8))
cs = plt.contour(X, Y, Z, levels=levels)        # đường đồng mức
plt.clabel(cs, inline=True, fontsize=8)         # hiện nhãn cao độ
plt.gca().set_aspect("equal", adjustable="box")
plt.title(f"Contours: {tif_path} (interval={interval} m)")
plt.xlabel("X"); plt.ylabel("Y")
plt.tight_layout()
plt.show()