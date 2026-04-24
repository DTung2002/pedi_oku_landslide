# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_all

# -------------------- Path Resolution (for standard or conda env) --------------------
def collect_data_dir(src, dest):
    out = []
    if not os.path.isdir(src):
        return out
    for root, dirs, files in os.walk(src):
        dirs[:] = [
            d for d in dirs
            if d not in {"__pycache__", ".cache", ".tmp", "temp"}
            and not d.startswith(".")
        ]
        rel = os.path.relpath(root, src)
        dest_dir = dest if rel == "." else os.path.join(dest, rel)
        for name in files:
            lower = name.lower()
            if lower.endswith((".pyc", ".pyo")) or lower in {".ds_store", "thumbs.db"}:
                continue
            out.append((os.path.join(root, name), dest_dir))
    return out


def get_gdal_data():
    try:
        import rasterio
        return os.path.join(os.path.dirname(rasterio.__file__), 'gdal_data')
    except Exception:
        return None

def get_proj_data():
    try:
        import pyproj
        return pyproj.datadir.get_data_dir()
    except Exception:
        return None

# -------------------- Collections --------------------
datas = []
datas += collect_data_dir('pedi_oku_landslide/assets', 'pedi_oku_landslide/assets')
datas += collect_data_dir('pedi_oku_landslide/config', 'pedi_oku_landslide/config')

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
]

# List of packages to collect all data/binaries/hiddenimports
packages_to_collect = [
    'PyQt5', 'cv2', 'numpy', 'scipy', 'rasterio', 'pyproj', 
    'geopandas', 'fiona', 'shapely', 'sklearn', 'matplotlib'
]

def keep_runtime_submodule(name):
    blocked_parts = (
        ".tests",
        ".testing",
        ".conftest",
        ".benchmarks",
        ".examples",
        ".docs",
    )
    return not any(part in name for part in blocked_parts)


for pkg in packages_to_collect:
    tmp_datas, tmp_binaries, tmp_hiddenimports = collect_all(
        pkg,
        filter_submodules=keep_runtime_submodule,
        exclude_datas=["**/tests/**", "**/testing/**", "**/__pycache__/**"],
    )
    datas += tmp_datas
    binaries += tmp_binaries
    hiddenimports += tmp_hiddenimports

# OpenCV wheels/conda packages write absolute paths into cv2/config*.py
# (for example .../env/Lib/site-packages/cv2/python-3.11).  Those paths do
# not exist after copying the app to another machine, so cv2 imports its loader
# again and raises: "recursion is detected during loading of cv2".
# Replace those config files with frozen-app relative paths.
cv2_config_dir = os.path.join("build", "pyinstaller_cv2_config")
os.makedirs(cv2_config_dir, exist_ok=True)

cv2_config_py = os.path.join(cv2_config_dir, "config.py")
with open(cv2_config_py, "w", encoding="utf-8") as f:
    f.write(
        "import os\n"
        "BINARIES_PATHS = [os.path.dirname(LOADER_DIR)] + BINARIES_PATHS\n"
    )

cv2_config_ver = os.path.join(
    cv2_config_dir, f"config-{sys.version_info[0]}.{sys.version_info[1]}.py"
)
with open(cv2_config_ver, "w", encoding="utf-8") as f:
    f.write(
        "import os\n"
        f"PYTHON_EXTENSIONS_PATHS = [os.path.join(LOADER_DIR, 'python-{sys.version_info[0]}.{sys.version_info[1]}')] + PYTHON_EXTENSIONS_PATHS\n"
    )

datas = [
    item for item in datas
    if not (
        len(item) >= 2
        and os.path.normpath(item[1]) == "cv2"
        and os.path.basename(item[0]).startswith("config")
        and os.path.basename(item[0]).endswith(".py")
    )
]
datas += [
    (cv2_config_py, "cv2"),
    (cv2_config_ver, "cv2"),
]

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
    excludes=[
        'pkg_resources',
        'setuptools',
        'jaraco',
        'torch',
        'torchvision',
        'torchaudio',
        'tensorboard',
        'tensorflow',
        'dask',
        'pytest',
        'numba',
    ],
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
