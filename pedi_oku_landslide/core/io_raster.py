from typing import Tuple, Dict
import numpy as np
import rasterio

def read_asc(path: str) -> Tuple[np.ndarray, Dict]:
    """Đọc file .asc (AAIGrid) → (mảng, metadata)."""
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        meta = ds.meta.copy()
    return arr, meta

def write_tif(path: str, arr: np.ndarray, meta: Dict) -> None:
    """Ghi GeoTIFF 1 band (nếu cần dùng sau)."""
    m = meta.copy()
    m.update(dtype=arr.dtype, count=1, compress="lzw")
    with rasterio.open(path, "w", **m) as ds:
        ds.write(arr, 1)
import os
import datetime

def make_run_folder(base_dir="output") -> str:
    """
    Tạo thư mục con dạng output/run_YYYYMMDD_HHMMSS/
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(base_dir, f"run_{ts}")
    os.makedirs(folder, exist_ok=True)
    return folder
