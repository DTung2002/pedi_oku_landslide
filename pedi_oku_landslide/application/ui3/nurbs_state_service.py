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



class UI3NurbsStateServiceMixin:
    def profile_endpoints(self, prof: dict, *, rdp_eps_m: float, smooth_radius_m: float) -> Optional[Tuple[float, float, float, float]]:
        return _profile_endpoints_impl(
            prof,
            profile_slip_span_range=self.profile_slip_span_range,
            rdp_eps_m=rdp_eps_m,
            smooth_radius_m=smooth_radius_m,
        )

    def grouped_vector_endpoints(self, prof: dict, groups: List[dict]) -> Optional[Tuple[float, float, float, float]]:
        return _grouped_vector_endpoints_impl(prof, groups)

    def build_default_nurbs_chainage(self, s0: float, s1: float, groups: List[dict]) -> List[float]:
        return _build_default_nurbs_chainage_impl(s0, s1, groups)

    def build_default_nurbs_params(
        self,
        *,
        prof: dict,
        groups: List[dict],
        base_curve: dict,
        endpoints: Tuple[float, float, float, float],
        nurbs_seed_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        return _build_default_nurbs_params_impl(
            prof=prof,
            groups=groups,
            base_curve=base_curve,
            endpoints=endpoints,
            nurbs_seed_method=nurbs_seed_method,
        )

    def reconcile_nurbs_params_with_groups(
        self,
        *,
        prof: dict,
        groups: List[dict],
        base_curve: dict,
        params: Optional[Dict[str, Any]],
        endpoints: Tuple[float, float, float, float],
        nurbs_seed_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        return _reconcile_nurbs_params_with_groups_impl(
            prof=prof,
            groups=groups,
            base_curve=base_curve,
            params=params,
            endpoints=endpoints,
            nurbs_seed_method=nurbs_seed_method,
        )

    def clamp_curve_below_ground(
        self,
        curve: Optional[Dict[str, np.ndarray]],
        *,
        prof: Optional[dict],
        clearance: float = 0.3,
        keep_endpoints: bool = True,
    ) -> Optional[Dict[str, np.ndarray]]:
        return _clamp_curve_below_ground_impl(
            curve,
            prof=prof,
            clearance=clearance,
            keep_endpoints=keep_endpoints,
        )
