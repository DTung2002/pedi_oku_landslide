"""UI4 lightweight state helpers for summary and preview discovery."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple


def read_ui4_json_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_ui4_summary_for_run(run_dir: str) -> Dict[str, Any]:
    run_dir = str(run_dir or "").strip()
    if not run_dir:
        return {}
    ui4_dir = os.path.join(run_dir, "ui4")
    shared_path = os.path.join(ui4_dir, "ui_shared_data.json")
    shared = read_ui4_json_file(shared_path) if os.path.exists(shared_path) else {}
    summary_path = str(shared.get("ui4_summary_json") or os.path.join(ui4_dir, "ui4_kriging_summary.json"))
    if not os.path.exists(summary_path):
        return {}
    return read_ui4_json_file(summary_path)


def summary_range_for_kind(summary: Dict[str, Any], kind: str) -> Optional[Tuple[float, float]]:
    if not isinstance(summary, dict):
        return None
    raster_stats = summary.get("raster_stats", {}) if isinstance(summary.get("raster_stats"), dict) else {}
    for key in (f"{kind}_masked", kind):
        rs = raster_stats.get(key, {})
        if not isinstance(rs, dict):
            continue
        zmin = rs.get("min")
        zmax = rs.get("max")
        try:
            zmin_f = float(zmin)
            zmax_f = float(zmax)
        except Exception:
            continue
        if zmax_f > zmin_f:
            return (zmin_f, zmax_f)

    stats = summary.get("stats", {}) if isinstance(summary.get("stats"), dict) else {}
    legacy_min = stats.get(f"{kind}_min_m")
    legacy_max = stats.get(f"{kind}_max_m")
    try:
        legacy_min_f = float(legacy_min)
        legacy_max_f = float(legacy_max)
    except Exception:
        return None
    if legacy_max_f > legacy_min_f:
        return (legacy_min_f, legacy_max_f)
    return None


def list_ui4_preview_pngs(run_dir: str) -> Dict[str, Any]:
    run_dir = str(run_dir or "").strip()
    if not run_dir:
        return {"ok": False, "error": "missing run context", "preview_dir": "", "pngs": []}
    preview_dir = os.path.join(run_dir, "ui4", "preview")
    if not os.path.isdir(preview_dir):
        return {"ok": False, "error": f"folder not found ({preview_dir})", "preview_dir": preview_dir, "pngs": []}

    pngs: List[str] = sorted(
        os.path.join(preview_dir, name)
        for name in os.listdir(preview_dir)
        if name.lower().endswith(".png")
    )
    return {"ok": True, "preview_dir": preview_dir, "pngs": pngs}
