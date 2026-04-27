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



class UI3StorageExportServiceMixin:
    def ui3_paths(self) -> UI3RunPaths:
        return UI3RunPaths(str(self._ctx.get("run_dir", "") or ""))

    def save_ground_csv(
        self,
        *,
        line_id: str,
        geom: LineString,
        step_m: float,
        profile_source: str,
        ground_export_step_m: float,
        ground_export_dem_path: str,
        dx_path: str,
        dy_path: str,
        dz_path: str,
        slip_path: str,
        out_csv: str,
    ) -> Optional[str]:
        return _save_ground_csv_for_line_impl(
            self,
            line_id=line_id,
            geom=geom,
            step_m=step_m,
            profile_source=profile_source,
            ground_export_step_m=ground_export_step_m,
            ground_export_dem_path=ground_export_dem_path,
            dx_path=dx_path,
            dy_path=dy_path,
            dz_path=dz_path,
            slip_path=slip_path,
            out_csv=out_csv,
        )

    def save_rdp_csv(self, *, line_id: str, profile: Dict[str, Any], rdp_eps_m: float, smooth_radius_m: float, out_csv: str) -> Optional[str]:
        return _save_rdp_csv_for_line_impl(
            line_id=line_id,
            prof=profile,
            rdp_eps_m=rdp_eps_m,
            smooth_radius_m=smooth_radius_m,
            out_csv=out_csv,
        )

    def save_theta_csv(self, *, line_id: str, profile: Dict[str, Any], groups: Optional[List[dict]], out_csv: str) -> Optional[str]:
        return _save_theta_csv_for_line_impl(
            line_id=line_id,
            prof=profile,
            groups=groups,
            out_csv=out_csv,
        )

    def save_vectors_csv(self, *, line_id: str, profile: Dict[str, Any], out_csv: str) -> Optional[str]:
        return _save_vectors_csv_for_line_impl(
            line_id=line_id,
            prof=profile,
            out_csv=out_csv,
        )

    def load_group_json_data(
        self,
        *,
        path: str,
        line_id: str,
        apply_settings: bool,
        apply_group_json_settings,
    ) -> Tuple[Dict[str, Any], List[dict], str]:
        return _load_group_json_data_impl(
            path,
            line_id=line_id,
            apply_settings=apply_settings,
            apply_group_json_settings=apply_group_json_settings,
        )

    def build_group_json_payload(
        self,
        *,
        line_label: str,
        groups: List[dict],
        prof: Optional[dict],
        chainage_origin: str,
        curve_method: str,
        nurbs_seed_method: Optional[str],
        profile_dem_source: str,
        profile_dem_path: str,
        grouping_params: Dict[str, Any],
        group_method: Optional[str],
    ) -> Dict[str, Any]:
        return _build_group_json_payload_impl(
            line_label=line_label,
            groups=groups,
            prof=prof,
            chainage_origin=chainage_origin,
            curve_method=curve_method,
            nurbs_seed_method=nurbs_seed_method,
            profile_dem_source=profile_dem_source,
            profile_dem_path=profile_dem_path,
            grouping_params=grouping_params,
            group_method=group_method,
            extract_curvature_nodes=self.extract_curvature_nodes,
            profile_slip_span_range=self.profile_slip_span_range,
        )

    def save_group_json(self, path: str, payload: Dict[str, Any]) -> str:
        return _save_group_json_impl(path, payload)

    def saved_curvature_series_for_line(
        self,
        *,
        path: str,
        current_groups: Optional[List[dict]],
        prof: Optional[dict],
        repair_saved_curvature_points,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return _saved_curvature_series_for_line_impl(
            path,
            current_groups=current_groups,
            prof=prof,
            repair_saved_curvature_points=repair_saved_curvature_points,
        )

    def save_nurbs_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        return _save_nurbs_outputs_workflow_impl(outputs)

    def save_global_fit_spline_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        return _save_global_fit_spline_outputs_workflow_impl(outputs)

    def load_saved_ui3_state(self, paths: Dict[str, str]) -> Dict[str, Any]:
        return _load_saved_ui3_state_workflow_impl(paths)

    def sync_anchor_updates(self, line_id: str, curve_data: Dict[str, Any]) -> Dict[str, Any]:
        paths = self.ui3_paths()
        return _sync_anchor_updates_workflow_impl(
            line_id=line_id,
            curve_data=curve_data,
            intersections_path=paths.ui2_intersections_json_path(),
            anchors_path=paths.anchors_json_path(),
        )

    def export_vectors_and_ground(self, line_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        return _export_vectors_and_ground_workflow_impl(line_id=line_id, profile=profile)
