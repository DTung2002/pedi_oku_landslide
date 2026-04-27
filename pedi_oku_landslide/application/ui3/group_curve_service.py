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



class UI3GroupCurveServiceMixin:
    def auto_group(
        self,
        line_id: str,
        geom: Optional[LineString],
        grouping_settings: Dict[str, Any],
        profile_source: str,
        step_m: float,
        *,
        prof: Optional[Dict[str, Any]] = None,
        min_len: float = WORKFLOW_GROUP_MIN_LEN_M,
    ) -> Dict[str, Any]:
        return _auto_group_workflow_impl(
            line_id=line_id,
            geom=geom,
            grouping_settings=grouping_settings,
            profile_source=profile_source,
            step_m=step_m,
            inputs=self._inputs or self.load_inputs(),
            prof=prof,
            min_len=min_len,
        )

    def auto_group_profile(self, profile: Dict[str, Any], grouping_settings: Dict[str, Any], min_len: float = WORKFLOW_GROUP_MIN_LEN_M) -> List[Dict[str, Any]]:
        groups = _auto_group_profile_by_criteria_impl(profile, **dict(grouping_settings or {}))
        return _clamp_groups_to_slip_impl(profile, groups, min_len=min_len)

    def clamp_groups(self, profile: Dict[str, Any], groups: List[Dict[str, Any]], min_len: float = WORKFLOW_GROUP_MIN_LEN_M) -> List[Dict[str, Any]]:
        return _clamp_groups_to_slip_impl(profile, groups, min_len=min_len)

    def normalize_groups(self, groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return _normalize_groups_workflow_impl(groups)

    def build_curve_seed(self, profile: Dict[str, Any], groups: List[Dict[str, Any]], curve_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return _build_curve_seed_workflow_impl(profile, groups, curve_settings=curve_settings)

    def fit_bezier_curve_seed(self, chain, elevg, target_s, target_z, curve_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return _fit_bezier_curve_seed_workflow_impl(chain, elevg, target_s, target_z, curve_settings=curve_settings)

    def evaluate_nurbs(self, params_or_ctrl_points, elev_ctrl=None, weights=None, degree: int = 3, n_samples: int = 300) -> Dict[str, Any]:
        return _evaluate_nurbs_workflow_impl(
            params_or_ctrl_points,
            elev_ctrl=elev_ctrl,
            weights=weights,
            degree=degree,
            n_samples=n_samples,
        )

    def fit_nurbs_params_from_curve(
        self,
        curve: Dict[str, Any],
        groups: List[dict],
        *,
        degree: int = 3,
        min_control_points: int = 4,
    ) -> Dict[str, Any]:
        return _fit_nurbs_params_from_curve_impl(
            curve,
            groups,
            degree=degree,
            min_control_points=min_control_points,
        )

    def build_global_forward_fit_spline(
        self,
        *,
        profile: Optional[Dict[str, Any]] = None,
        prof: Optional[Dict[str, Any]] = None,
        groups: Optional[List[Dict[str, Any]]] = None,
        theta_rows: Optional[List[Dict[str, Any]]] = None,
        short_length_m: float = 0.1,
    ) -> Dict[str, Any]:
        profile_data = profile if profile is not None else prof
        if profile_data is None:
            raise ValueError("Profile is required for global fit spline generation.")
        return _build_global_forward_fit_spline_workflow_impl(
            profile=profile_data,
            groups=list(groups or []),
            theta_rows=list(theta_rows or []),
            short_length_m=short_length_m,
        )
