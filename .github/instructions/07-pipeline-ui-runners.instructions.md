## What is the Problem
Backend xu ly cho cac UI (UI1/2/3): tao boundary, lines, profiles, curvature, grouping va xuat ket qua cho UI.

## Information Context
File chinh:
- pedi_oku_landslide/pipeline/runners/ui1_backend.py: xu ly sau ingest/SAD, tao hillshade/overlay, boundary, vector.
- pedi_oku_landslide/pipeline/runners/ui2_backend.py: xu ly slip mask de tao section lines (CSV/JSON) va geometry.
- pedi_oku_landslide/pipeline/runners/ui3_backend.py: xu ly profile/curvature, grouping, xuat ket qua UI3.
Thu vien chinh: numpy, rasterio, geopandas, shapely, skimage, scipy, matplotlib, pandas.
Quy uoc/luu y:
- Du lieu vao/ra gan voi AnalysisContext (ui1/ui2/ui3 folders).
- Nhieu ham su dung CRS fallback EPSG:6677 neu thieu CRS.
- UI2/3 phu thuoc outputs tu UI1 (dx/dy/mask) va sections.csv.
Vi du nho:
```
from pedi_oku_landslide.pipeline.runners.ui2_backend import generate_auto_lines_from_slipzone
outs = generate_auto_lines_from_slipzone(mask, transform)
```

## Steps to Complete
1) Map input/output tren run_dir (ingest_meta.json, dx/dy, mask, sections.csv).
2) Neu them output moi, cap nhat UI tab tuong ung va ingest_meta.processed neu can.
3) Kiem tra CRS/transform truoc khi ve line/profile; fallback EPSG chi la last-resort.
4) Test voi data nho: tao section lines tu mask va kiem tra file CSV/JSON.
5) Kiem tra UI3 output (curves/groups) duoc ghi duoi ui3/.
