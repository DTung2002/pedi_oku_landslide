from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString

from pedi_oku_landslide.domain.ui3.anchors import (
    load_json_items,
    save_json_items,
    update_anchors_for_saved_main_curve,
)
from pedi_oku_landslide.domain.ui3.curve_fit import (
    evaluate_nurbs_curve,
    evaluate_piecewise_cubic_segments,
    estimate_slip_curve,
    fit_bezier_smooth_curve,
)
from pedi_oku_landslide.domain.ui3.grouping import (
    WORKFLOW_GROUP_MIN_LEN_M,
    auto_group_profile_by_criteria,
    clamp_groups_to_slip,
    renumber_groups_visual_order,
)
from pedi_oku_landslide.domain.ui3.profile import compute_profile
from pedi_oku_landslide.infrastructure.rendering.ui3_render import render_profile_png
from pedi_oku_landslide.infrastructure.storage.ui3_storage import load_json, save_json


def compute_profile_for_line(
    *,
    line_id: str,
    geom: LineString,
    profile_source: str,
    step_m: float,
    slip_only: bool,
    inputs: Dict[str, Any],
    dem_path: str = "",
    dem_orig_path: str = "",
    dx_path: str = "",
    dy_path: str = "",
    dz_path: str = "",
    slip_mask_path: str = "",
) -> Optional[Dict[str, Any]]:
    dem = dem_path or inputs.get("dem_path_smooth") or inputs.get("dem_path_raw") or ""
    if profile_source == "raw":
        dem = dem_path or inputs.get("dem_path_raw") or dem
    dem_orig = dem_orig_path or inputs.get("dem_path_raw") or dem
    if not dem:
        return None

    prof = compute_profile(
        dem,
        dx_path or inputs.get("dx_path", ""),
        dy_path or inputs.get("dy_path", ""),
        dz_path or inputs.get("dz_path", ""),
        geom,
        step_m=step_m,
        smooth_win=11,
        smooth_poly=2,
        slip_mask_path=slip_mask_path or inputs.get("slip_path", ""),
        slip_only=bool(slip_only),
        dem_orig_path=dem_orig,
    )
    if not prof:
        return None
    prof["profile_dem_source"] = profile_source
    prof["profile_dem_path"] = dem
    prof["line_id"] = line_id
    return prof


def auto_group(
    *,
    line_id: str,
    geom: Optional[LineString],
    grouping_settings: Dict[str, Any],
    profile_source: str,
    step_m: float,
    inputs: Dict[str, Any],
    prof: Optional[Dict[str, Any]] = None,
    min_len: float = WORKFLOW_GROUP_MIN_LEN_M,
) -> Dict[str, Any]:
    if prof is None and geom is not None:
        prof = compute_profile_for_line(
            line_id=line_id,
            geom=geom,
            profile_source=profile_source,
            step_m=step_m,
            slip_only=False,
            inputs=inputs,
        )
    if not prof:
        return {"profile": None, "groups": []}
    groups = auto_group_profile_by_criteria(prof, **dict(grouping_settings or {}))
    groups = clamp_groups_to_slip(prof, groups, min_len=min_len)
    return {"profile": prof, "groups": groups}


def build_curve_seed(profile: Dict[str, Any], groups: List[Dict[str, Any]], curve_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    curve_settings = dict(curve_settings or {})
    return estimate_slip_curve(
        profile,
        groups,
        ds=float(curve_settings.get("ds", 0.2)),
        smooth_factor=float(curve_settings.get("smooth_factor", 0.1)),
        depth_gain=float(curve_settings.get("depth_gain", 3.0)),
        min_depth=float(curve_settings.get("min_depth", 1.0)),
    )


def fit_bezier_curve_seed(chain, elevg, target_s, target_z, curve_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    curve_settings = dict(curve_settings or {})
    return fit_bezier_smooth_curve(
        chain,
        elevg,
        target_s,
        target_z,
        c0=float(curve_settings.get("c0", 0.30)),
        c1=float(curve_settings.get("c1", 0.30)),
        clearance=float(curve_settings.get("clearance", 0.12)),
    )


def evaluate_nurbs(params_or_ctrl_points, elev_ctrl=None, weights=None, degree: int = 3, n_samples: int = 300) -> Dict[str, Any]:
    if isinstance(params_or_ctrl_points, dict):
        params = dict(params_or_ctrl_points)
        segments = params.get("segments", []) or []
        if isinstance(segments, list) and segments:
            return evaluate_piecewise_cubic_segments(
                segments=segments,
                n_samples=int(params.get("n_samples", n_samples)),
            )
        cps = np.asarray(params.get("control_points", []), dtype=float)
        if cps.ndim != 2 or cps.shape[0] < 2:
            return {"chain": np.array([], dtype=float), "elev": np.array([], dtype=float)}
        return evaluate_nurbs_curve(
            chain_ctrl=cps[:, 0],
            elev_ctrl=cps[:, 1],
            weights=params.get("weights", weights),
            degree=int(params.get("degree", degree)),
            n_samples=int(params.get("n_samples", n_samples)),
        )
    return evaluate_nurbs_curve(
        chain_ctrl=params_or_ctrl_points,
        elev_ctrl=elev_ctrl,
        weights=weights,
        degree=degree,
        n_samples=n_samples,
    )


def render_preview(*, profile: Dict[str, Any], render_settings: Dict[str, Any], groups=None, overlay_curves=None) -> Dict[str, Any]:
    render_settings = dict(render_settings or {})
    msg, path = render_profile_png(
        profile,
        render_settings["out_png"],
        y_min=render_settings.get("y_min"),
        y_max=render_settings.get("y_max"),
        x_min=render_settings.get("x_min"),
        x_max=render_settings.get("x_max"),
        vec_scale=float(render_settings.get("vec_scale", 0.1)),
        vec_width=float(render_settings.get("vec_width", 0.0015)),
        head_len=float(render_settings.get("head_len", 7.0)),
        head_w=float(render_settings.get("head_w", 5.0)),
        highlight_theta=render_settings.get("highlight_theta"),
        group_ranges=groups if groups else None,
        draw_curve=bool(render_settings.get("draw_curve", False)),
        save_curve_json=bool(render_settings.get("save_curve_json", False)),
        overlay_curves=overlay_curves,
        figsize=tuple(render_settings.get("figsize", (18, 10))),
        dpi=int(render_settings.get("dpi", 220)),
        base_font=int(render_settings.get("base_font", 20)),
        label_font=int(render_settings.get("label_font", 20)),
        tick_font=int(render_settings.get("tick_font", 20)),
        legend_font=int(render_settings.get("legend_font", 20)),
        ground_lw=float(render_settings.get("ground_lw", 2.2)),
        ungrouped_color=str(render_settings.get("ungrouped_color", "#bbbbbb")),
        curvature_series=render_settings.get("curvature_series"),
        curvature_rdp_eps_m=float(render_settings.get("curvature_rdp_eps_m", 0.5)),
        curvature_smooth_radius_m=float(render_settings.get("curvature_smooth_radius_m", 0.0)),
    )
    return {"message": msg, "path": path}


def save_nurbs_outputs(outputs: Dict[str, Any]) -> Dict[str, Any]:
    saved: Dict[str, Any] = {}
    for key, spec in dict(outputs or {}).items():
        if not isinstance(spec, dict):
            continue
        path = str(spec.get("path", "") or "")
        if not path:
            continue
        saved[key] = save_json(path, spec.get("payload"))
    return saved


def load_saved_ui3_state(paths: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, path in dict(paths or {}).items():
        out[key] = load_json(path, default=None)
    return out


def sync_anchor_updates(*, line_id: str, curve_data: Dict[str, Any], intersections_path: str, anchors_path: str) -> Dict[str, Any]:
    intersections = load_json_items(intersections_path)
    anchors = load_json_items(anchors_path)
    payload, updated = update_anchors_for_saved_main_curve(
        curve=curve_data,
        intersections=intersections,
        existing_anchors=anchors,
        main_line_id=line_id,
    )
    if updated > 0:
        save_json_items(anchors_path, payload)
    return {"line_id": line_id, "curve": curve_data, "updated": updated, "path": anchors_path if updated > 0 else ""}


def export_vectors_and_ground(*, line_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    chain = np.asarray(profile.get("chain", []), dtype=float)
    elev = np.asarray(profile.get("elev_s", []), dtype=float)
    d_para = np.asarray(profile.get("d_para", []), dtype=float)
    dz = np.asarray(profile.get("dz", []), dtype=float)
    vectors = []
    for s, z, dp, dzv in zip(chain.tolist(), elev.tolist(), d_para.tolist(), dz.tolist()):
        if np.isfinite(s) and np.isfinite(z) and np.isfinite(dp) and np.isfinite(dzv):
            vectors.append({"s": float(s), "z": float(z), "d_para": float(dp), "dz": float(dzv)})
    ground = []
    for s, z in zip(chain.tolist(), elev.tolist()):
        if np.isfinite(s) and np.isfinite(z):
            ground.append({"s": float(s), "z": float(z)})
    return {"line_id": line_id, "vectors": vectors, "ground": ground}


def normalize_groups(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return renumber_groups_visual_order(groups)
