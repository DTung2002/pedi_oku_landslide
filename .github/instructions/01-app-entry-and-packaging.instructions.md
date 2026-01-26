## What is the Problem
Dinh nghia entrypoint chay app desktop (PyQt5) va dong goi PyInstaller, bao dam duong dan va bien moi truong GDAL/PROJ chuan trong ca dev va exe.

## Information Context
Framework/chinh: PyQt5 UI; PyInstaller spec file.
Thu vien chinh (khong thay file pin version): numpy, scipy, rasterio, pyproj, fiona, shapely, geopandas, sklearn, matplotlib, cv2 (opencv), skimage, pandas, tqdm.
Diem khoi dau:
- __main__.py (repo root): set GDAL/PROJ env neu frozen, chdir, goi run_ui.
- pedi_oku_landslide/__main__.py: logic tuong tu, import ui.main_window.
Dong goi:
- Landslide.spec: collect data/assets, gdal/proj data, dynamic libs cho cac thu vien GIS/ML.
Quy uoc/luu y:
- Khi frozen, bat buoc set GDAL_DATA va PROJ_LIB theo _MEIPASS.
- APP_ROOT va INTERNAL_ROOT lay tu core/paths.py; output luu duoi APP_ROOT/output.
- Khong thay requirements.txt/pyproject trong repo; neu can pin version thi tao file moi va cap nhat spec.
Vi du nho:
```
from pedi_oku_landslide.core.paths import APP_ROOT, INTERNAL_ROOT
from pedi_oku_landslide.ui.main_window import run_ui
run_ui(str(APP_ROOT), str(INTERNAL_ROOT))
```

## Steps to Complete
1) Xac dinh entrypoint can sua: __main__.py (root) hay pedi_oku_landslide/__main__.py.
2) Neu doi cau truc thu muc, cap nhat core/paths.py va duong dan assets trong UI.
3) Neu bo sung thu vien GIS/ML, cap nhat Landslide.spec (collect_data_files/dynamic_libs).
4) Kiem tra bien GDAL_DATA/PROJ_LIB trong che do frozen.
5) Chay app dev hoac build exe, kiem tra output tao duoi APP_ROOT/output.
