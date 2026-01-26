## What is the Problem
Xu ly chinh cua pipeline: smooth raster, tinh dX/dY/dZ (SAD), va detect landslide mask.

## Information Context
File chinh:
- pedi_oku_landslide/pipeline/steps/step_smooth.py: gaussian smoothing, tao tif/png + smooth_meta.json.
- pedi_oku_landslide/pipeline/steps/step_sad.py: SAD (opencv/traditional), dX/dY/dZ, tao tif/png + sad_meta.json.
- pedi_oku_landslide/pipeline/steps/step_detect.py: tao landslide_mask.tif, landslide_overlay.png, crop DEM.
Thu vien chinh: numpy, rasterio, matplotlib, scipy.ndimage, sklearn (KMeans), cv2 (optional).
Quy uoc/luu y:
- SAD uu tien method "opencv" neu cv2 co san.
- dX/dY luu theo pixel units; dZ tinh tu before_pz/after_pz.
- Detect can dx.tif/dy.tif; cap nhat ingest_meta.processed (dx, dy, dz, slip_mask, dem_cropped).
- Mask la uint8 nodata=0 (khong dung -9999).
Vi du nho:
```
out = run_sad(ctx, method="opencv", patch_size_m=20.0, search_radius_m=2.0)
```

## Steps to Complete
1) Dam bao ingest da tao input trong run_dir/input.
2) Neu can, chay step_smooth de tao raster smoothed.
3) Chay step_sad de tao dx/dy/dz va sad_meta.json; kiem tra CRS/transform phu hop.
4) Chay step_detect de tao mask/overlay va cap nhat ingest_meta.processed.
5) Kiem thu: mo cac PNG dau ra, kiem tra extent, grid, va thang mau.
