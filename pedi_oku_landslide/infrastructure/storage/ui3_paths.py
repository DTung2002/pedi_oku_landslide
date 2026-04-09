import json
import os
from dataclasses import dataclass
from typing import Dict

from pedi_oku_landslide.core.paths import OUTPUT_ROOT


def _out(*parts: str) -> str:
    return os.path.join(OUTPUT_ROOT, *parts)


def auto_paths() -> Dict[str, str]:
    def pick_first_exists(cands):
        for p in cands:
            if p and os.path.exists(p):
                return p
        return cands[0] if cands else ""

    js = {}
    for cand in [
        _out("ui_shared_data.json"),
        _out("UI1", "ui_shared_data.json"),
    ]:
        if os.path.exists(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    js.update(json.load(f))
            except Exception:
                pass

    dem = pick_first_exists([
        _out("UI1", "before_asc_smooth.tif"),
        _out("UI1", "step1_crop", "before_ground.asc"),
        _out("UI1", "step1_crop", "before_ground.tif"),
        js.get("dem_ground_path", ""),
    ])

    dem_orig = pick_first_exists([
        _out("UI1", "step1_crop", "before_ground.asc"),
        _out("UI1", "step1_crop", "before_ground.tif"),
        js.get("dem_ground_path", ""),
    ])

    dx = pick_first_exists([
        _out("UI1", "dX.asc"),
        _out("UI1", "step2_sad", "dX.asc"),
        js.get("dx_path", ""),
    ])

    dy = pick_first_exists([
        _out("UI1", "dY.asc"),
        _out("UI1", "step2_sad", "dY.asc"),
        js.get("dy_path", ""),
    ])

    dz = pick_first_exists([
        _out("UI1", "dZ.asc"),
        _out("UI1", "step7_slipzone", "dZ_slipzone.asc"),
        _out("UI1", "step5_dz", "dZ.asc"),
        js.get("dz_path", ""),
    ])

    lines = pick_first_exists([
        _out("UI2", "step2_selected_lines", "selected_lines.gpkg"),
        js.get("lines_path", ""),
    ])

    slip = pick_first_exists([
        _out("UI1", "slip_zone.asc"),
        _out("UI1", "step7_slipzone", "slip_zone.asc"),
        js.get("slip_path", ""),
    ])

    return {"dem": dem, "dem_orig": dem_orig, "dx": dx, "dy": dy, "dz": dz, "lines": lines, "slip": slip}


@dataclass(frozen=True)
class UI3RunPaths:
    run_dir: str

    def ui3_run_dir(self) -> str:
        if not self.run_dir:
            raise RuntimeError("[UI3] Run context is empty. Call set_context() first.")
        path = os.path.join(self.run_dir, "ui3")
        os.makedirs(path, exist_ok=True)
        return path

    def preview_dir(self) -> str:
        path = os.path.join(self.ui3_run_dir(), "preview")
        os.makedirs(path, exist_ok=True)
        return path

    def groups_dir(self) -> str:
        path = os.path.join(self.ui3_run_dir(), "groups")
        os.makedirs(path, exist_ok=True)
        return path

    def curve_dir(self) -> str:
        path = os.path.join(self.ui3_run_dir(), "curve")
        os.makedirs(path, exist_ok=True)
        return path

    def ground_dir(self) -> str:
        path = os.path.join(self.ui3_run_dir(), "ground")
        os.makedirs(path, exist_ok=True)
        return path

    def profile_png_path_for(self, line_id: str) -> str:
        return os.path.join(self.preview_dir(), f"profile_{line_id}.png")

    def nurbs_png_path_for(self, line_id: str) -> str:
        return os.path.join(self.preview_dir(), f"profile_{line_id}_nurbs.png")

    def nurbs_json_path_for(self, line_id: str) -> str:
        return os.path.join(self.preview_dir(), f"profile_{line_id}_nurbs.json")

    def groups_json_path_for(self, line_id: str) -> str:
        return os.path.join(self.groups_dir(), f"{line_id}.json")

    def ground_csv_path_for(self, line_id: str) -> str:
        return os.path.join(self.ground_dir(), f"{line_id}_ground.csv")

    def rdp_csv_path_for(self, line_id: str) -> str:
        return os.path.join(self.ground_dir(), f"{line_id}_RDP.csv")

    def curve_nurbs_info_json_path_for(self, line_id: str) -> str:
        return os.path.join(self.curve_dir(), f"nurbs_info_{line_id}.json")

    def boring_holes_dir(self) -> str:
        path = os.path.join(self.ui3_run_dir(), "boring_holes")
        os.makedirs(path, exist_ok=True)
        return path

    def boring_holes_json_path_for(self, line_id: str) -> str:
        return os.path.join(self.boring_holes_dir(), f"{line_id}_boring_holes.json")

    def anchors_json_path(self) -> str:
        return os.path.join(self.ui3_run_dir(), "anchors.json")

    def ui2_intersections_json_path(self) -> str:
        if not self.run_dir:
            raise RuntimeError("[UI3] Run context is empty. Call set_context() first.")
        return os.path.join(self.run_dir, "ui2", "intersections_main_cross.json")
