# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_dynamic_libs

# -------------------- Path Resolution (for standard or conda env) --------------------
def get_gdal_data():
    try:
        import rasterio
        return os.path.join(os.path.dirname(rasterio.__file__), 'gdal_data')
    except ImportError:
        return None

def get_proj_data():
    try:
        import pyproj
        return pyproj.datadir.get_data_dir()
    except ImportError:
        return None

# -------------------- Collections --------------------
datas = [
    ('pedi_oku_landslide/assets', 'pedi_oku_landslide/assets'),
    ('pedi_oku_landslide/config', 'pedi_oku_landslide/config'),
]

# Add GDAL/PROJ data if found
gdal_data = get_gdal_data()
if gdal_data and os.path.exists(gdal_data):
    datas.append((gdal_data, 'gdal-data'))

proj_data = get_proj_data()
if proj_data and os.path.exists(proj_data):
    datas.append((proj_data, 'proj-data'))

binaries = []
hiddenimports = [
    "scipy._cyutility",
    "scipy._lib._ccallback_c",
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "sklearn.utils._vector_sentinel",
]

# List of packages to collect all data/binaries/hiddenimports
packages_to_collect = [
    'PyQt5', 'cv2', 'numpy', 'scipy', 'rasterio', 'pyproj', 
    'geopandas', 'fiona', 'shapely', 'sklearn', 'matplotlib', 'torch'
]

for pkg in packages_to_collect:
    tmp_datas, tmp_binaries, tmp_hiddenimports = collect_all(pkg)
    datas += tmp_datas
    binaries += tmp_binaries
    hiddenimports += tmp_hiddenimports

# -------------------- Analysis --------------------
a = Analysis(
    ['main.py'],  # entry point in root
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pkg_resources', 'setuptools', 'jaraco'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Landslide',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Landslide',
)