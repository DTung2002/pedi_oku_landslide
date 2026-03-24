import os
import numpy as np
import rasterio
from pedi_oku_landslide.pipeline.runners.ui1_backend import save_asc, run_UI1_2_sad, run_UI1_2_sad_gpu

def create_dummy_asc(path, size=100):
    data = np.random.rand(size, size).astype(np.float32)
    profile = {
        'driver': 'AAIGrid',
        'dtype': 'float32',
        'nodata': -9999.0,
        'width': size,
        'height': size,
        'count': 1,
        'crs': None,
        'transform': rasterio.transform.from_origin(0, size, 0.2, 0.2)
    }
    save_asc(path, data, profile)
    return path

def verify():
    os.makedirs("tests/data", exist_ok=True)
    before_path = create_dummy_asc("tests/data/before.asc")
    after_path = create_dummy_asc("tests/data/after.asc")
    
    print("Running OpenCV SAD...")
    status_cv, out_cv = run_UI1_2_sad(before_path, after_path, output_dir="tests/out_cv", cellsize=0.2)
    print(status_cv)
    
    print("Running GPU SAD...")
    status_gpu, out_gpu = run_UI1_2_sad_gpu(before_path, after_path, output_dir="tests/out_gpu", cellsize=0.2)
    print(status_gpu)
    
    # Compare results
    with rasterio.open(out_cv["dX_path"]) as src:
        dx_cv = src.read(1)
    with rasterio.open(out_gpu["dX_path"]) as src:
        dx_gpu = src.read(1)
        
    # Note: OpenCV version uses SSD (TM_SQDIFF), while GPU version uses SAD (torch.abs).
    # They might not produce identical results, but they should be similar.
    # However, for verification of the *logic*, I should probably check if GPU version 
    # produces reasonable values.
    
    valid = ~np.isnan(dx_cv) & ~np.isnan(dx_gpu)
    if np.any(valid):
        diff = np.abs(dx_cv[valid] - dx_gpu[valid])
        print(f"Mean diff in dX: {np.mean(diff)}")
        print(f"Max diff in dX: {np.max(diff)}")
    else:
        print("No valid overlapping pixels to compare.")

if __name__ == "__main__":
    verify()
