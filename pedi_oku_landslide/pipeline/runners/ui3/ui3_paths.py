import json
import os

from pedi_oku_landslide.core.paths import OUTPUT_ROOT


def _out(*parts: str) -> str:
    return os.path.join(OUTPUT_ROOT, *parts)


def auto_paths() -> dict:
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
