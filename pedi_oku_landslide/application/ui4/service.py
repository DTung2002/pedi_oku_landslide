"""
UI4 backend facade — re-exports from decomposed ui4/ sub‐modules.

All public API is preserved; downstream code (ui4_frontend.py) can
continue importing from this file without changes.
"""
from __future__ import annotations

# ── Public API re-exports ───────────────────────────────────────────────────

from pedi_oku_landslide.domain.ui4.types import (  # noqa: F401
    DEFAULT_UI4_CONTOUR_PARAMS,
    DEFAULT_UI4_PARAMS,
)

from pedi_oku_landslide.application.ui4.inputs import (  # noqa: F401
    collect_ui4_run_inputs,
)

from pedi_oku_landslide.domain.ui4.boundary import (  # noqa: F401
    apply_mask_to_raster,
    read_boundary_polygon_from_dxf,
)

from pedi_oku_landslide.domain.ui4.kriging import (  # noqa: F401
    build_ok_solver,
    exp_variogram,
    fit_exponential_variogram,
    ok_predict,
)

from pedi_oku_landslide.domain.ui4.surface import (  # noqa: F401
    decimate_by_chainage,
    load_ui4_curve_points,
    sample_dem_and_compute_depth,
)

from pedi_oku_landslide.domain.ui4.contour import (  # noqa: F401
    render_contours_png_from_raster,
    render_ui4_contours_for_run,
)

from pedi_oku_landslide.application.ui4.state import (  # noqa: F401
    list_ui4_preview_pngs,
    load_ui4_summary_for_run,
    summary_range_for_kind,
)

from pedi_oku_landslide.application.ui4.workflows import (  # noqa: F401
    run_ui4_kriging_for_run,
    run_ui4_kriging_from_paths,
)
