# ui3_backend.py
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString
from pedi_oku_landslide.domain.ui3.curve_fit import (
    estimate_slip_curve as _estimate_slip_curve_impl,
    evaluate_nurbs_curve as _evaluate_nurbs_curve_impl,
    fit_nurbs_params_from_curve as _fit_nurbs_params_from_curve_impl,
    fit_bezier_smooth_curve as _fit_bezier_smooth_curve_impl,
)
from pedi_oku_landslide.domain.ui3.grouping import (
    WORKFLOW_GROUP_MIN_LEN_M,
    auto_group_profile_by_criteria as _auto_group_profile_by_criteria_impl,
    clamp_groups_to_slip as _clamp_groups_to_slip_impl,
    effective_profile_slip_span_range as _effective_profile_slip_span_range_impl,
    extract_curvature_rdp_nodes as _extract_curvature_rdp_nodes_impl,
    filter_rdp_nodes_to_slip_zone as _filter_rdp_nodes_to_slip_zone_impl,
)
from pedi_oku_landslide.infrastructure.storage.ui3_paths import UI3RunPaths
from pedi_oku_landslide.infrastructure.storage.ui3_storage import (
    build_gdf_from_sections_csv,
    ensure_sections_csv_current,
    save_json,
)
from pedi_oku_landslide.application.ui3.exports import (
    save_ground_csv_for_line as _save_ground_csv_for_line_impl,
    save_rdp_csv_for_line as _save_rdp_csv_for_line_impl,
    save_theta_csv_for_line as _save_theta_csv_for_line_impl,
    save_vectors_csv_for_line as _save_vectors_csv_for_line_impl,
)
from pedi_oku_landslide.application.ui3.group_state import (
    build_group_json_payload as _build_group_json_payload_impl,
    load_group_json_data as _load_group_json_data_impl,
    save_group_json as _save_group_json_impl,
    saved_curvature_series_for_line as _saved_curvature_series_for_line_impl,
)
from pedi_oku_landslide.application.ui3.inputs import discover_ui3_inputs
from pedi_oku_landslide.domain.ui3.anchors import (
    constrain_curve_to_points as _constrain_curve_to_points_impl,
    constrain_curve_to_cross_anchors as _constrain_curve_to_cross_anchors_impl,
    extend_endpoint_targets_with_points as _extend_endpoint_targets_with_points_impl,
    extend_endpoint_targets_with_cross_anchors as _extend_endpoint_targets_with_cross_anchors_impl,
    anchors_for_cross_line as _anchors_for_cross_line_impl,
    anchors_ready_for_cross_constraints as _anchors_ready_for_cross_constraints_impl,
    load_json_items as _load_anchor_json_items_impl,
    save_json_items as _save_anchor_json_items_impl,
)
from pedi_oku_landslide.domain.ui3.boring_holes import (
    build_boring_holes_payload as _build_boring_holes_payload_impl,
    load_boring_holes_payload as _load_boring_holes_payload_impl,
    project_boring_holes_to_line as _project_boring_holes_to_line_impl,
)
from pedi_oku_landslide.domain.ui3.curve_state import (
    build_default_nurbs_chainage as _build_default_nurbs_chainage_impl,
    build_default_nurbs_params as _build_default_nurbs_params_impl,
    clamp_curve_below_ground as _clamp_curve_below_ground_impl,
    grouped_vector_endpoints as _grouped_vector_endpoints_impl,
    profile_endpoints as _profile_endpoints_impl,
    reconcile_nurbs_params_with_groups as _reconcile_nurbs_params_with_groups_impl,
)
from pedi_oku_landslide.application.ui3.workflows import (
    auto_group as _auto_group_workflow_impl,
    build_global_forward_fit_spline_workflow as _build_global_forward_fit_spline_workflow_impl,
    build_curve_seed as _build_curve_seed_workflow_impl,
    compute_profile_for_line as _compute_profile_for_line_workflow_impl,
    evaluate_nurbs as _evaluate_nurbs_workflow_impl,
    export_vectors_and_ground as _export_vectors_and_ground_workflow_impl,
    fit_bezier_curve_seed as _fit_bezier_curve_seed_workflow_impl,
    load_theta_csv_group_angles_workflow as _load_theta_csv_group_angles_workflow_impl,
    load_saved_ui3_state as _load_saved_ui3_state_workflow_impl,
    normalize_groups as _normalize_groups_workflow_impl,
    render_preview as _render_preview_workflow_impl,
    save_global_fit_spline_outputs as _save_global_fit_spline_outputs_workflow_impl,
    save_nurbs_outputs as _save_nurbs_outputs_workflow_impl,
    sync_anchor_updates as _sync_anchor_updates_workflow_impl,
)



class UI3ProfileServiceMixin:
    def load_inputs(self) -> Dict[str, Any]:
        return discover_ui3_inputs(
            run_dir=str(self._ctx.get("run_dir", "") or ""),
            base_dir=str(self._ctx.get("base_dir", "") or ""),
        )

    def load_lines(self, csv_path: str, dem_path: str) -> Dict[str, Any]:
        migrated = ensure_sections_csv_current(csv_path, run_dir=str(self._ctx.get("run_dir", "") or ""))
        gdf = build_gdf_from_sections_csv(csv_path, dem_path)
        return {"migrated": bool(migrated), "gdf": gdf}

    def compute_profile_for_line(
        self,
        line_id: str,
        geom: LineString,
        profile_source: str,
        step_m: float,
        slip_only: bool,
        *,
        dem_path: str = "",
        dem_orig_path: str = "",
        dx_path: str = "",
        dy_path: str = "",
        dz_path: str = "",
        slip_mask_path: str = "",
    ) -> Optional[Dict[str, Any]]:
        return _compute_profile_for_line_workflow_impl(
            line_id=line_id,
            geom=geom,
            profile_source=profile_source,
            step_m=step_m,
            slip_only=slip_only,
            inputs=self._inputs or self.load_inputs(),
            dem_path=dem_path,
            dem_orig_path=dem_orig_path,
            dx_path=dx_path,
            dy_path=dy_path,
            dz_path=dz_path,
            slip_mask_path=slip_mask_path,
        )

    def render_preview(self, profile: Dict[str, Any], render_settings: Dict[str, Any], groups=None, overlay_curves=None) -> Dict[str, Any]:
        return _render_preview_workflow_impl(
            profile=profile,
            render_settings=render_settings,
            groups=groups,
            overlay_curves=overlay_curves,
        )

    def load_theta_csv_group_angles(self, *, csv_path: str, groups: Optional[List[dict]]) -> List[dict]:
        return _load_theta_csv_group_angles_workflow_impl(csv_path=csv_path, groups=groups)

    def extract_curvature_nodes(self, profile: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return _extract_curvature_rdp_nodes_impl(profile, **kwargs)

    def profile_slip_span_range(self, profile: Dict[str, Any], **kwargs) -> Optional[Tuple[float, float]]:
        return _effective_profile_slip_span_range_impl(profile, **kwargs)

    def filter_rdp_nodes_to_slip_zone(self, profile: Optional[Dict[str, Any]], chain, elev, curv, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _filter_rdp_nodes_to_slip_zone_impl(profile, chain, elev, curv, **kwargs)
