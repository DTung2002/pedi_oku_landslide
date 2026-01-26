## What is the Problem
Nhap 5 input raster, tao preview hillshade, va khoi tao ingest_meta.json cho toan pipeline.

## Information Context
File chinh: pedi_oku_landslide/pipeline/ingest.py
Dau vao bat buoc:
- before_dem (GeoTIFF)
- before_asc, after_asc, before_pz, after_pz (ASC)
Dau ra:
- ui1/before_dem_hillshade.tif
- ui1/before_asc_hillshade.png, ui1/after_asc_hillshade.png
- ingest_meta.json (inputs, outputs, processed, preview_meta)
Thu vien chinh: rasterio, pyproj, numpy, matplotlib.
Quy uoc/luu y:
- Co co che CRS: embedded -> sibling .prj -> fallback_epsg (config).
- update_ingest_processed dung de cap nhat truong processed tu cac step sau.
- Duong dan trong ingest_meta.json duoc normalize thanh '/'.
Vi du nho:
```
info = run_ingest(ctx, files)
print(info["preview"]["before_asc_hillshade_png"])
```

## Steps to Complete
1) Xac nhan du 5 file dau vao va dung dinh dang.
2) Chay run_ingest, kiem tra ingest_meta.json duoc tao.
3) Neu them input moi, cap nhat required list va ingest_meta.json.
4) Neu thay doi logic CRS, cap nhat _load_crs_from_sibling_prj va fallback_epsg.
5) Kiem tra preview png co grid/axes va khong bi NaN.
