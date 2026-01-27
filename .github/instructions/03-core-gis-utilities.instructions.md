## What is the Problem
Cung cap tien ich xu ly raster/GIS co ban va quan ly duong dan (APP_ROOT/INTERNAL_ROOT/OUTPUT_ROOT) de cac module khac dung chung.

## Information Context
File chinh:
- pedi_oku_landslide/core/paths.py: tinh APP_ROOT, INTERNAL_ROOT theo dev/exe; tao OUTPUT_ROOT.
- pedi_oku_landslide/core/analysis.py: smooth_gaussian (gaussian_filter) giu NaN.
- pedi_oku_landslide/core/gis_utils.py: hillshade tu DEM.
- pedi_oku_landslide/core/io_raster.py: read_asc, write_tif, make_run_folder (legacy).
Thu vien chinh: numpy, scipy.ndimage, rasterio, matplotlib.colors.LightSource.
Quy uoc/luu y:
- OUTPUT_ROOT luon duoc tao trong APP_ROOT.
- smooth_gaussian giu NaN va ep dtype goc.
- hillshade tra ve uint8 0-255.
Vi du nho:
```
from pedi_oku_landslide.core.paths import OUTPUT_ROOT
out_dir = OUTPUT_ROOT
```

## Steps to Complete
1) Neu thay doi vi tri output/asset, cap nhat core/paths.py va kiem tra __main__.py.
2) Neu them ham xu ly raster, giu signature don gian va tra ve array/meta ro rang.
3) Dam bao ham moi khong tu ghi file neu khong can thiet; de pipeline/step quan ly IO.
4) Them test thu cong bang raster nho (asc/tif) de kiem tra NaN va CRS.
