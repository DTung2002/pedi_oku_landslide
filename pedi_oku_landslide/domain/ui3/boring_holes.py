import os
from typing import Any, Dict, List

import numpy as np
from shapely.geometry import LineString, Point

from pedi_oku_landslide.infrastructure.storage.ui3_storage import load_json

BORING_HOLES_VERSION = 1
BORING_HOLES_DEFAULT_TOLERANCE_M = 1.0
BORING_HOLES_CHAINAGE_TOL_M = 1e-6


def build_boring_holes_payload(items: List[dict], *, distance_tolerance_m: float = BORING_HOLES_DEFAULT_TOLERANCE_M) -> Dict[str, Any]:
    try:
        tol = float(distance_tolerance_m)
    except Exception:
        tol = BORING_HOLES_DEFAULT_TOLERANCE_M
    if not np.isfinite(tol) or tol <= 0.0:
        tol = BORING_HOLES_DEFAULT_TOLERANCE_M
    out_items: List[Dict[str, Any]] = []
    for rec in items or []:
        try:
            bh = str(rec.get("bh", rec.get("label", "")) or "").strip()
            x = float(rec.get("x"))
            y = float(rec.get("y"))
            z = float(rec.get("z"))
        except Exception:
            continue
        if not bh or not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue
        out_items.append({"bh": bh, "x": float(x), "y": float(y), "z": float(z)})
    return {
        "version": BORING_HOLES_VERSION,
        "distance_tolerance_m": float(tol),
        "items": out_items,
    }


def load_boring_holes_payload(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return build_boring_holes_payload([])
    data = load_json(path, default={})
    if not isinstance(data, dict):
        data = {}
    return build_boring_holes_payload(
        data.get("items", []),
        distance_tolerance_m=data.get("distance_tolerance_m", BORING_HOLES_DEFAULT_TOLERANCE_M),
    )


def project_boring_holes_to_line(
    line_geom: LineString,
    boring_holes: Dict[str, Any],
    *,
    distance_tolerance_m: float = BORING_HOLES_DEFAULT_TOLERANCE_M,
) -> Dict[str, Any]:
    try:
        tol = float(distance_tolerance_m)
    except Exception:
        tol = BORING_HOLES_DEFAULT_TOLERANCE_M
    if not np.isfinite(tol) or tol <= 0.0:
        tol = BORING_HOLES_DEFAULT_TOLERANCE_M

    if line_geom is None or getattr(line_geom, "is_empty", False):
        return {"distance_tolerance_m": float(tol), "items": [], "skipped": []}

    payload = build_boring_holes_payload(
        (boring_holes or {}).get("items", []),
        distance_tolerance_m=(boring_holes or {}).get("distance_tolerance_m", tol),
    )
    items = payload.get("items", []) or []
    accepted: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for rec in items:
        try:
            bh = str(rec.get("bh", "")).strip()
            x = float(rec.get("x"))
            y = float(rec.get("y"))
            z = float(rec.get("z"))
        except Exception:
            continue
        pt = Point(float(x), float(y))
        try:
            dist = float(line_geom.distance(pt))
            s_val = float(line_geom.project(pt))
            proj_pt = line_geom.interpolate(s_val)
        except Exception:
            skipped.append(f"Skipping boring hole '{bh}': projection failed.")
            continue
        if not (np.isfinite(dist) and np.isfinite(s_val)):
            skipped.append(f"Skipping boring hole '{bh}': invalid projection result.")
            continue
        if dist > tol:
            continue
        accepted.append(
            {
                "label": bh,
                "bh": bh,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "distance_to_line_m": float(dist),
                "s_on_line": float(s_val),
                "projected_x": float(proj_pt.x),
                "projected_y": float(proj_pt.y),
            }
        )

    accepted.sort(key=lambda d: (float(d.get("s_on_line", 0.0)), float(d.get("distance_to_line_m", np.inf))))
    deduped: List[Dict[str, Any]] = []
    for rec in accepted:
        if not deduped:
            deduped.append(rec)
            continue
        prev = deduped[-1]
        if np.isclose(float(prev.get("s_on_line", np.nan)), float(rec.get("s_on_line", np.nan)), rtol=0.0, atol=BORING_HOLES_CHAINAGE_TOL_M):
            prev_dist = float(prev.get("distance_to_line_m", np.inf))
            cur_dist = float(rec.get("distance_to_line_m", np.inf))
            if cur_dist < prev_dist:
                skipped.append(
                    f"Skipping boring hole '{prev.get('label', '')}': projected chainage overlaps '{rec.get('label', '')}'."
                )
                deduped[-1] = rec
            else:
                skipped.append(
                    f"Skipping boring hole '{rec.get('label', '')}': projected chainage overlaps '{prev.get('label', '')}'."
                )
            continue
        deduped.append(rec)

    return {
        "distance_tolerance_m": float(tol),
        "items": deduped,
        "skipped": skipped,
    }
