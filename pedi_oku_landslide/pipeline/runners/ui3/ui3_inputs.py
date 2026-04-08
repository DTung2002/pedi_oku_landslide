import json
import os
from typing import Any, Dict

from .ui3_paths import auto_paths


def _pick_first_existing(*cands: str) -> str:
    for path in cands:
        if path and os.path.exists(path):
            return path
    return ""


def discover_ui3_inputs(*, run_dir: str, base_dir: str) -> Dict[str, Any]:
    run_dir = str(run_dir or "")
    base_dir = str(base_dir or "")

    shared_jsons = [
        os.path.join(run_dir, "ui_shared_data.json"),
        os.path.join(base_dir, "output", "ui_shared_data.json"),
        os.path.join(base_dir, "output", "UI1", "ui_shared_data.json"),
    ]
    js: Dict[str, Any] = {}
    for path in shared_jsons:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    js.update(json.load(f))
            except Exception:
                pass

    meta_inputs: Dict[str, Any] = {}
    meta_processed: Dict[str, Any] = {}
    meta_path = os.path.join(run_dir, "ingest_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
            meta_inputs = meta.get("inputs") or {}
            meta_processed = meta.get("processed") or {}
        except Exception:
            pass

    ap = auto_paths()
    dem_path_smooth = _pick_first_existing(
        os.path.join(run_dir, "ui1", "after_dem_smooth.tif"),
        meta_inputs.get("after_dem") or "",
        meta_inputs.get("before_dem") or "",
        js.get("dem_ground_path") or "",
        ap.get("dem", ""),
        meta_inputs.get("before_asc") or "",
        os.path.join(run_dir, "input", "after_dem.tif"),
        os.path.join(run_dir, "input", "before_dem.tif"),
        os.path.join(run_dir, "input", "before.asc"),
        meta_processed.get("dem_cropped") or "",
    )
    dem_path_raw = _pick_first_existing(
        meta_inputs.get("after_dem") or "",
        meta_inputs.get("before_dem") or "",
        os.path.join(run_dir, "input", "after_dem.tif"),
        os.path.join(run_dir, "input", "before_dem.tif"),
        js.get("dem_ground_path") or "",
        ap.get("dem_orig", ""),
        ap.get("dem", ""),
    )
    slip_path = js.get("slip_path") or ap.get("slip", "")
    if meta_processed.get("slip_mask"):
        slip_path = meta_processed.get("slip_mask")
    if not slip_path:
        slip_path = os.path.join(run_dir, "ui1", "landslide_mask.tif")
    if slip_path and (not os.path.exists(slip_path)):
        alt = slip_path.replace(".asc", ".tif")
        if os.path.exists(alt):
            slip_path = alt

    return {
        "shared_json": js,
        "meta_inputs": meta_inputs,
        "meta_processed": meta_processed,
        "auto_paths": ap,
        "dem_path_smooth": dem_path_smooth,
        "dem_path_raw": dem_path_raw,
        "dx_path": _pick_first_existing(js.get("dx_path") or "", ap.get("dx", ""), os.path.join(run_dir, "ui1", "dx.tif")),
        "dy_path": _pick_first_existing(js.get("dy_path") or "", ap.get("dy", ""), os.path.join(run_dir, "ui1", "dy.tif")),
        "dz_path": _pick_first_existing(js.get("dz_path") or "", ap.get("dz", ""), os.path.join(run_dir, "ui1", "dz.tif")),
        "lines_path": "",
        "slip_path": slip_path,
    }
