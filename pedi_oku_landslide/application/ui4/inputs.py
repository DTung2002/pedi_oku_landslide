"""UI4 run-input discovery and readiness checks."""
from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, List

from pedi_oku_landslide.domain.ui4.types import _pick_existing


def _find_ui4_mask_tif(run_dir: str) -> str:
    ui1_dir = os.path.join(run_dir, "ui1")
    candidates = [
        os.path.join(ui1_dir, "landslide_mask.tif"),
        os.path.join(ui1_dir, "detect_mask.tif"),
        os.path.join(ui1_dir, "mask.tif"),
        os.path.join(ui1_dir, "mask_binary.tif"),
    ]
    return _pick_existing(candidates)


def _find_ui4_dxf_boundary(run_dir: str) -> str:
    """Discover a DXF boundary file in the run's input/ directory."""
    input_dir = os.path.join(run_dir, "input")
    candidates = [
        os.path.join(input_dir, "Boundary.dxf"),
        os.path.join(input_dir, "boundary.dxf"),
    ]
    found = _pick_existing(candidates)
    if found:
        return found
    # Fallback: first .dxf in input/
    dxf_files = sorted(glob.glob(os.path.join(input_dir, "*.dxf")))
    return os.path.abspath(dxf_files[0]) if dxf_files else ""


def collect_ui4_run_inputs(run_dir: str) -> Dict[str, Any]:
    """
    Discover run-scoped UI4 inputs for a single run directory.
    This powers UI4 tab readiness checks and serves as the default resolver
    for run-scoped UI4 backend execution.
    """
    run_dir = os.path.abspath(str(run_dir or "").strip())
    if not run_dir:
        return {"ok": False, "error": "Missing run_dir", "run_dir": ""}

    input_dir = os.path.join(run_dir, "input")
    ui2_dir = os.path.join(run_dir, "ui2")
    ui3_dir = os.path.join(run_dir, "ui3")
    curve_dir = os.path.join(ui3_dir, "curve")
    groups_dir = os.path.join(ui3_dir, "groups")
    mask_tif = _find_ui4_mask_tif(run_dir)
    dxf_boundary = _find_ui4_dxf_boundary(run_dir)

    dem_tif_candidates = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    dem_preferred = [
        os.path.join(input_dir, "after_dem.tif"),
        os.path.join(input_dir, "before_dem.tif"),
    ]
    dem_preferred.extend(
        p for p in dem_tif_candidates
        if any(tag in os.path.basename(p).lower() for tag in ("ground", "dem"))
    )
    dem_path = _pick_existing(dem_preferred or dem_tif_candidates)

    intersections_json = _pick_existing([os.path.join(ui2_dir, "intersections_main_cross.json")])
    anchors_json = _pick_existing(
        [
            os.path.join(ui3_dir, "anchors.json"),
            os.path.join(ui3_dir, "anchors_xyz.json"),
        ]
    )

    nurbs_name_re = re.compile(
        r"^nurbs_(?:CL|ML)\d+__\([^)]+_m\)\.json$",
        re.IGNORECASE,
    )
    nurbs_curve_jsons = sorted(
        p for p in glob.glob(os.path.join(curve_dir, "nurbs_*.json"))
        if bool(nurbs_name_re.match(os.path.basename(p)))
    )
    group_jsons = sorted(glob.glob(os.path.join(groups_dir, "*.json")))
    nurbs_info_jsons = sorted(glob.glob(os.path.join(curve_dir, "nurbs_info_*.json")))

    counts = {
        "nurbs_curves": len(nurbs_curve_jsons),
        "groups": len(group_jsons),
        "nurbs_info": len(nurbs_info_jsons),
    }
    ready_checks = {
        "run_dir_exists": os.path.isdir(run_dir),
        "input_dir_exists": os.path.isdir(input_dir),
        "dem_exists": bool(dem_path and os.path.exists(dem_path) and str(dem_path).lower().endswith(".tif")),
        "curve_dir_exists": os.path.isdir(curve_dir),
        "groups_dir_exists": os.path.isdir(groups_dir),
        "nurbs_curves_exist": counts["nurbs_curves"] > 0,
        "intersections_exists": bool(intersections_json),
        "anchors_exists": bool(anchors_json),
    }

    missing_required: List[str] = []
    if not ready_checks["dem_exists"]:
        missing_required.append("DEM (.tif in input/)")
    if not ready_checks["nurbs_curves_exist"]:
        missing_required.append("NURBS curves (nurbs_CLn__/nurbs_MLn__ in ui3/curve)")

    return {
        "ok": True,
        "run_dir": run_dir,
        "paths": {
            "input_dir": os.path.abspath(input_dir),
            "dem": dem_path,
            "dem_tif_candidates": [os.path.abspath(p) for p in dem_tif_candidates],
            "mask_tif": mask_tif,
            "dxf_boundary_path": dxf_boundary,
            "intersections_main_cross_json": intersections_json,
            "anchors_json": anchors_json,
            "ui3_curve_dir": os.path.abspath(curve_dir),
            "ui3_groups_dir": os.path.abspath(groups_dir),
            "nurbs_curve_jsons": [os.path.abspath(p) for p in nurbs_curve_jsons],
            "group_jsons": [os.path.abspath(p) for p in group_jsons],
            "nurbs_info_jsons": [os.path.abspath(p) for p in nurbs_info_jsons],
        },
        "counts": counts,
        "ready_checks": ready_checks,
        "ready_for_ui4": len(missing_required) == 0,
        "missing_required": missing_required,
    }
