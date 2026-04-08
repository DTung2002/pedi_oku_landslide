"""Quick import verification for refactored modules."""
import sys
import traceback

errors = []

# Test 1: UI4 backend facade imports (what ui4_frontend.py uses)
try:
    from pedi_oku_landslide.pipeline.runners.ui4_backend import (
        collect_ui4_run_inputs,
        render_ui4_contours_for_run,
        run_ui4_kriging_for_run,
    )
    print("[OK] ui4_backend facade imports")
except Exception as e:
    errors.append(f"ui4_backend facade: {e}")
    traceback.print_exc()

# Test 2: UI4 sub-modules direct imports
try:
    from pedi_oku_landslide.pipeline.runners.ui4.ui4_types import DEFAULT_UI4_PARAMS, DEFAULT_UI4_CONTOUR_PARAMS
    from pedi_oku_landslide.pipeline.runners.ui4.ui4_inputs import collect_ui4_run_inputs
    from pedi_oku_landslide.pipeline.runners.ui4.ui4_boundary import read_boundary_polygon_from_dxf, apply_mask_to_raster
    from pedi_oku_landslide.pipeline.runners.ui4.ui4_kriging import exp_variogram, build_ok_solver, ok_predict
    from pedi_oku_landslide.pipeline.runners.ui4.ui4_surface import load_ui4_curve_points, sample_dem_and_compute_depth
    from pedi_oku_landslide.pipeline.runners.ui4.ui4_contour import render_contours_png_from_raster, render_ui4_contours_for_run
    from pedi_oku_landslide.pipeline.runners.ui4.ui4_render import run_ui4_kriging_from_paths, run_ui4_kriging_for_run
    print("[OK] ui4 sub-module imports")
except Exception as e:
    errors.append(f"ui4 sub-modules: {e}")
    traceback.print_exc()

# Test 3: Other backend modules still work
try:
    from pedi_oku_landslide.pipeline.runners.ui1_backend import UI1BackendService
    print("[OK] ui1_backend imports")
except Exception as e:
    errors.append(f"ui1_backend: {e}")
    traceback.print_exc()

try:
    from pedi_oku_landslide.pipeline.runners.ui2_backend import UI2BackendService
    print("[OK] ui2_backend imports")
except Exception as e:
    errors.append(f"ui2_backend: {e}")
    traceback.print_exc()

try:
    from pedi_oku_landslide.pipeline.runners.ui3_backend import UI3BackendService
    print("[OK] ui3_backend imports")
except Exception as e:
    errors.append(f"ui3_backend: {e}")
    traceback.print_exc()

# Test 4: Core modules
try:
    from pedi_oku_landslide.core.paths import APP_ROOT
    from pedi_oku_landslide.core.analysis import smooth_mean
    from pedi_oku_landslide.services.session_store import AnalysisContext
    print("[OK] core + services imports")
except Exception as e:
    errors.append(f"core/services: {e}")
    traceback.print_exc()

print()
if errors:
    print(f"FAILED: {len(errors)} error(s)")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL IMPORTS OK")
    sys.exit(0)
