# ui3_backend.py
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString
from pedi_oku_landslide.domain.ui3.curve_fit import (
    estimate_slip_curve as _estimate_slip_curve_impl,
    evaluate_nurbs_curve as _evaluate_nurbs_curve_impl,
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
    load_json,
    save_json,
)
from pedi_oku_landslide.application.ui3.exports import (
    save_ground_csv_for_line as _save_ground_csv_for_line_impl,
    save_rdp_csv_for_line as _save_rdp_csv_for_line_impl,
)
from pedi_oku_landslide.application.ui3.group_state import (
    build_group_json_payload as _build_group_json_payload_impl,
    load_group_json_data as _load_group_json_data_impl,
    save_group_json as _save_group_json_impl,
    saved_curvature_series_for_line as _saved_curvature_series_for_line_impl,
)
from pedi_oku_landslide.application.ui3.inputs import discover_ui3_inputs
from pedi_oku_landslide.domain.ui3.anchors import (
    constrain_curve_to_cross_anchors as _constrain_curve_to_cross_anchors_impl,
    extend_endpoint_targets_with_cross_anchors as _extend_endpoint_targets_with_cross_anchors_impl,
    anchors_for_cross_line as _anchors_for_cross_line_impl,
    anchors_ready_for_cross_constraints as _anchors_ready_for_cross_constraints_impl,
    load_json_items as _load_anchor_json_items_impl,
    save_json_items as _save_anchor_json_items_impl,
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
    build_curve_seed as _build_curve_seed_workflow_impl,
    compute_profile_for_line as _compute_profile_for_line_workflow_impl,
    evaluate_nurbs as _evaluate_nurbs_workflow_impl,
    export_vectors_and_ground as _export_vectors_and_ground_workflow_impl,
    fit_bezier_curve_seed as _fit_bezier_curve_seed_workflow_impl,
    load_saved_ui3_state as _load_saved_ui3_state_workflow_impl,
    normalize_groups as _normalize_groups_workflow_impl,
    render_preview as _render_preview_workflow_impl,
    save_nurbs_outputs as _save_nurbs_outputs_workflow_impl,
    sync_anchor_updates as _sync_anchor_updates_workflow_impl,
)


class UI3BackendService:
    def __init__(self, *, base_dir: str = "") -> None:
        self._ctx: Dict[str, str] = {"project": "", "run_label": "", "run_dir": "", "base_dir": base_dir}
        self._inputs: Dict[str, Any] = {}

    def set_context(self, project: str, run_label: str, run_dir: str, base_dir: str = "") -> Dict[str, Any]:
        if base_dir:
            self._ctx["base_dir"] = str(base_dir)
        self._ctx.update({"project": project, "run_label": run_label, "run_dir": run_dir})
        self._inputs = self.load_inputs()
        return dict(self._inputs)

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

    def load_groups(self, path: str) -> Any:
        return load_json(path, default=None)

    def save_groups(self, path: str, groups: Any) -> str:
        return save_json(path, groups)

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

    def render_preview(self, profile: Dict[str, Any], render_settings: Dict[str, Any], groups=None, overlay_curves=None) -> Dict[str, Any]:
        return _render_preview_workflow_impl(
            profile=profile,
            render_settings=render_settings,
            groups=groups,
            overlay_curves=overlay_curves,
        )

    def extract_curvature_nodes(self, profile: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return _extract_curvature_rdp_nodes_impl(profile, **kwargs)

    def profile_slip_span_range(self, profile: Dict[str, Any], **kwargs) -> Optional[Tuple[float, float]]:
        return _effective_profile_slip_span_range_impl(profile, **kwargs)

    def filter_rdp_nodes_to_slip_zone(self, profile: Optional[Dict[str, Any]], chain, elev, curv, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _filter_rdp_nodes_to_slip_zone_impl(profile, chain, elev, curv, **kwargs)

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

    def load_anchor_items(self, path: str) -> Dict[str, Any]:
        return _load_anchor_json_items_impl(path)

    def save_anchor_items(self, path: str, data: Dict[str, Any]) -> str:
        return _save_anchor_json_items_impl(path, data)

    def anchors_ready_for_cross_constraints(self, intersections: Dict[str, Any], anchors: Dict[str, Any]) -> bool:
        return _anchors_ready_for_cross_constraints_impl(intersections, anchors)

    def anchors_for_cross_line(self, intersections: Dict[str, Any], anchors: Dict[str, Any], cross_line_id: str, *, require_ready: bool = True) -> List[dict]:
        return _anchors_for_cross_line_impl(intersections, anchors, cross_line_id, require_ready=require_ready)

    def extend_endpoint_targets_with_cross_anchors(
        self,
        prof: dict,
        endpoints: Optional[Tuple[float, float, float, float]],
        anchors: List[dict],
    ) -> Optional[Tuple[float, float, float, float]]:
        return _extend_endpoint_targets_with_cross_anchors_impl(prof, endpoints, anchors)

    def constrain_curve_to_cross_anchors(self, curve: Optional[Dict[str, np.ndarray]], anchors: List[dict]) -> Optional[Dict[str, np.ndarray]]:
        return _constrain_curve_to_cross_anchors_impl(curve, anchors)

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
    ) -> Dict[str, Any]:
        return _build_default_nurbs_params_impl(
            prof=prof,
            groups=groups,
            base_curve=base_curve,
            endpoints=endpoints,
        )

    def reconcile_nurbs_params_with_groups(
        self,
        *,
        prof: dict,
        groups: List[dict],
        base_curve: dict,
        params: Optional[Dict[str, Any]],
        endpoints: Tuple[float, float, float, float],
    ) -> Dict[str, Any]:
        return _reconcile_nurbs_params_with_groups_impl(
            prof=prof,
            groups=groups,
            base_curve=base_curve,
            params=params,
            endpoints=endpoints,
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

    def save_nurbs_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        return _save_nurbs_outputs_workflow_impl(outputs)

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
