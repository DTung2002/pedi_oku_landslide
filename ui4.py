import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from shapely.geometry import Point, MultiPoint
from shapely.prepared import prep
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from scipy.linalg import lu_factor, lu_solve

# ----------------------------
# Inputs
# ----------------------------
dem_path = r"/mnt/data/2021-04-11_ground.tif"
json_paths = [
    r"/mnt/data/nurbs_CL1__(125.8_m).json",
    r"/mnt/data/nurbs_CL2__(125.8_m).json",
    r"/mnt/data/nurbs_CL3__(125.8_m).json",
    r"/mnt/data/nurbs_ML1__(125.8_m).json",
    r"/mnt/data/nurbs_ML2__(125.8_m).json",
    r"/mnt/data/nurbs_ML3__(125.8_m).json",
]

# Tuning params
chainage_step_m = 1.0   # decimate along each curve
grid_res_m      = 0.5   # kriging grid resolution
buffer_m        = 5.0   # buffer around convex hull of all points
nodata_out      = -9999.0

# ----------------------------
# Helpers
# ----------------------------
def decimate_by_chainage(df, ds=1.0):
    df = df.sort_values("chainage_m")
    keep_idx = []
    last = -1e18
    for i, r in df.iterrows():
        if r["chainage_m"] - last >= ds:
            keep_idx.append(i)
            last = r["chainage_m"]
    return df.loc[keep_idx]

def exp_variogram(h, nugget, sill, rang):
    rang = np.maximum(rang, 1e-6)
    return nugget + sill * (1.0 - np.exp(-h / rang))

# ----------------------------
# 1) Read & decimate slip curves
# ----------------------------
dfs = []
for p in json_paths:
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = data["points"]
    df = pd.DataFrame(pts)
    df["line_id"] = data.get("line_id", p)
    dfs.append(df[["line_id", "index", "chainage_m", "x", "y", "z"]])

all_df = pd.concat(dfs, ignore_index=True)

dec_list = []
for lid, g in all_df.groupby("line_id"):
    dec_list.append(decimate_by_chainage(g, ds=chainage_step_m))
dec_df = pd.concat(dec_list, ignore_index=True)

# ----------------------------
# 2) Sample DEM at curve points, compute depth
# ----------------------------
with rasterio.open(dem_path) as src:
    dem_vals = np.array([v[0] for v in src.sample(list(zip(dec_df["x"], dec_df["y"])))], dtype=float)
dec_df["z_dem"] = dem_vals
dec_df["depth"] = (dec_df["z_dem"] - dec_df["z"]).clip(lower=0.0)

coords = dec_df[["x", "y"]].to_numpy()
values = dec_df["depth"].to_numpy()
n = len(values)

# ----------------------------
# 3) Empirical variogram -> fit exponential
# ----------------------------
rng = np.random.default_rng(0)
pairs = 20000
i = rng.integers(0, n, size=pairs)
j = rng.integers(0, n, size=pairs)
m = i != j
i, j = i[m], j[m]

h = np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
gamma = 0.5 * (values[i] - values[j]) ** 2

bins = np.linspace(0, np.percentile(h, 95), 20)
bin_centers, gamma_means = [], []
for b0, b1 in zip(bins[:-1], bins[1:]):
    mm = (h >= b0) & (h < b1)
    if mm.sum() > 50:
        bin_centers.append(0.5 * (b0 + b1))
        gamma_means.append(np.mean(gamma[mm]))

bin_centers = np.array(bin_centers)
gamma_means = np.array(gamma_means)

vvar = np.var(values)
p0 = [0.05 * vvar, 0.95 * vvar, np.max(bin_centers) / 3]
params, _ = curve_fit(
    exp_variogram, bin_centers, gamma_means,
    p0=p0, bounds=([0, 0, 1e-3], [np.inf, np.inf, np.inf]),
    maxfev=20000
)
nugget, sill, rang = params
print("Variogram params:", params)

# ----------------------------
# 4) Ordinary Kriging system (LU factorization)
# ----------------------------
D = cdist(coords, coords)
Gamma = exp_variogram(D, *params)
np.fill_diagonal(Gamma, 0.0)

K = np.empty((n + 1, n + 1), float)
K[:n, :n] = Gamma
K[:n, n] = 1.0
K[n, :n] = 1.0
K[n, n] = 0.0

lu, piv = lu_factor(K)

def ok_predict(points_xy, chunk=2000):
    m = points_xy.shape[0]
    preds = np.empty(m, float)
    vars_ = np.empty(m, float)
    for s in range(0, m, chunk):
        e = min(m, s + chunk)
        P = points_xy[s:e]
        d = cdist(coords, P)                 # (n, chunk)
        g0 = exp_variogram(d, *params)       # (n, chunk)
        rhs = np.vstack([g0, np.ones((1, e - s))])  # (n+1, chunk)
        sol = lu_solve((lu, piv), rhs)
        w = sol[:n, :]
        mu = sol[n, :]
        preds[s:e] = w.T @ values
        vars_[s:e] = np.sum(w * g0, axis=0) + mu
    return preds, vars_

# ----------------------------
# 5) Build grid over convex hull (+ buffer), predict inside hull
# ----------------------------
hull = MultiPoint([Point(xy) for xy in coords]).convex_hull.buffer(buffer_m)
minx, miny, maxx, maxy = hull.bounds

xs = np.arange(minx, maxx + grid_res_m, grid_res_m)
ys = np.arange(miny, maxy + grid_res_m, grid_res_m)
nx, ny = len(xs), len(ys)

xx, yy = np.meshgrid(xs, ys)
grid_points = np.column_stack([xx.ravel(), yy.ravel()])

ph = prep(hull)
mask = np.array([ph.contains(Point(p)) for p in grid_points])
inside_pts = grid_points[mask]

pred_depth, pred_var = ok_predict(inside_pts, chunk=2000)
pred_depth = np.clip(pred_depth, 0.0, None)

# Sample DEM on grid points and compute slip elevation
with rasterio.open(dem_path) as src:
    dem_inside = np.array([v[0] for v in src.sample([tuple(p) for p in inside_pts])], float)

slip_z = dem_inside - pred_depth

# ----------------------------
# 6) Write outputs as GeoTIFF
# ----------------------------
Z = np.full(nx * ny, np.nan, float)
Z_depth = np.full(nx * ny, np.nan, float)
Z_var = np.full(nx * ny, np.nan, float)

Z[mask] = slip_z
Z_depth[mask] = pred_depth
Z_var[mask] = pred_var

Z = Z.reshape((ny, nx))
Z_depth = Z_depth.reshape((ny, nx))
Z_var = Z_var.reshape((ny, nx))

# flip vertically for GeoTIFF (row0 = maxy)
Z_top = np.flipud(Z)
Z_depth_top = np.flipud(Z_depth)
Z_var_top = np.flipud(Z_var)

transform_grid = from_origin(minx, maxy, grid_res_m, grid_res_m)

profile = {
    "driver": "GTiff",
    "height": ny,
    "width": nx,
    "count": 1,
    "dtype": "float32",
    "transform": transform_grid,
    "crs": None,
    "nodata": nodata_out,
    "compress": "deflate",
}

def write_tif(path, arr):
    arr2 = np.where(np.isfinite(arr), arr, nodata_out).astype("float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr2, 1)

write_tif("slip_surface_kriging.tif", Z_top)
write_tif("slip_depth_kriging.tif", Z_depth_top)
write_tif("slip_depth_kriging_variance.tif", Z_var_top)

print("Done.")