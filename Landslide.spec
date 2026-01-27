# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

import os
CONDA_PREFIX = os.environ.get("CONDA_PREFIX")  # lấy đúng env đang activate
if not CONDA_PREFIX:
    raise SystemExit("Hãy activate đúng conda env rồi mới chạy pyinstaller với spec này.")

 
datas = [
    ('pedi_oku_landslide\\assets', 'pedi_oku_landslide\\assets'),
    ('pedi_oku_landslide\\config', 'pedi_oku_landslide\\config'),

    (os.path.join(CONDA_PREFIX, 'Library', 'share', 'gdal'), 'gdal-data'),
    (os.path.join(CONDA_PREFIX, 'Library', 'share', 'proj'), 'proj-data'),
    (os.path.join(CONDA_PREFIX, 'Library', 'lib', 'tcl8.6'), '_tcl_data'),
    (os.path.join(CONDA_PREFIX, 'Library', 'lib', 'tk8.6'), '_tk_data'),
]


binaries = []
hiddenimports = []
datas += collect_data_files('PyQt5')
binaries += collect_dynamic_libs('cv2')
binaries += collect_dynamic_libs('numpy')
binaries += collect_dynamic_libs('scipy')
binaries += collect_dynamic_libs('rasterio')
binaries += collect_dynamic_libs('pyproj')
binaries += collect_dynamic_libs('fiona')
binaries += collect_dynamic_libs('shapely')
binaries += collect_dynamic_libs('sklearn')
hiddenimports += collect_submodules('PyQt5')
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('scipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('rasterio')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyproj')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('geopandas')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('fiona')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('shapely')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sklearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('matplotlib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
binaries += collect_dynamic_libs('cv2')
tmp_ret = collect_all('scipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
 
 
a = Analysis(
    ['__main__.py'],
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
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='Landslide',
)
 
hiddenimports += [
    "scipy._cyutility",
    "scipy._lib._ccallback_c",
]