import os
import json
import numpy as np
import rasterio
from pedi_oku_landslide.services.session_store import AnalysisContext
from pedi_oku_landslide.pipeline.steps.step_sad import run_sad

def create_dummy_asc(path, size=400, is_pz=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if is_pz:
        data = (np.arange(size*size) % 100).reshape((size, size)).astype(np.float32)
    else:
        # Before / After DEM
        data = np.random.rand(size, size).astype(np.float32) * 10
    
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
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data, 1)

def verify():
    base_dir = "tests/test_run"
    run_dir = os.path.join(base_dir, "output/project_xyz/run_1")
    ctx = AnalysisContext(
        project_id="project_xyz",
        run_id="run_1",
        base_dir=base_dir,
        project_dir=os.path.join(base_dir, "output/project_xyz"),
        run_dir=run_dir,
        in_dir=os.path.join(run_dir, "input"),
        out_ui1=os.path.join(run_dir, "ui1"),
        out_ui2=os.path.join(run_dir, "ui2"),
        out_ui3=os.path.join(run_dir, "ui3")
    )
    
    os.makedirs(ctx.in_dir, exist_ok=True)
    os.makedirs(ctx.out_ui1, exist_ok=True)
    
    # Create input files
    create_dummy_asc(os.path.join(ctx.in_dir, "before.asc"))
    create_dummy_asc(os.path.join(ctx.in_dir, "after.asc"))
    create_dummy_asc(os.path.join(ctx.in_dir, "before_pz.asc"), is_pz=True)
    create_dummy_asc(os.path.join(ctx.in_dir, "after_pz.asc"), is_pz=True)
    
    print("Running GPU SAD in Main Pipeline...")
    try:
        out_gpu = run_sad(ctx, method="gpu", patch_size_m=20.0, search_radius_m=2.0, use_smoothed=False)
        print("GPU SAD Completed Successfully:", out_gpu)
    except Exception as e:
        print("GPU SAD Failed:", e)

if __name__ == "__main__":
    verify()
