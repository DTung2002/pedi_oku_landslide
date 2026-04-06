import json
import os
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import LineString, Point


def normalize_line_role(line_role: str, line_id: str = "") -> str:
    role = (line_role or "").strip().lower()
    lid = (line_id or "").strip().lower()
    if role in ("main", "ml"):
        return "main"
    if role in ("cross", "aux", "cl"):
        return "cross"
    if lid.startswith("ml") or lid.startswith("main"):
        return "main"
    if lid.startswith("cl") or lid.startswith("cross"):
        return "cross"
    return ""


def parse_auto_line_id(line_id: str) -> Tuple[str, int]:
    txt = str(line_id or "").strip().upper().replace("-", "")
    if txt.startswith("ML") and txt[2:].isdigit():
        return "main", int(txt[2:])
    if txt.startswith("CL") and txt[2:].isdigit():
        return "cross", int(txt[2:])
    return "", 0


def line_order_key(line_id: str, fallback_idx: int) -> Tuple[int, str]:
    txt = str(line_id or "").strip()
    digits = "".join(ch for ch in txt if ch.isdigit())
    if digits:
        try:
            return int(digits), txt
        except Exception:
            pass
    return int(fallback_idx), txt


def pick_intersection_point(geom_a: LineString, geom_b: LineString) -> Tuple[Optional[Point], str]:
    try:
        inter = geom_a.intersection(geom_b)
    except Exception:
        return None, "error"
    if inter is None or getattr(inter, "is_empty", True):
        return None, "no_intersection"
    gtype = getattr(inter, "geom_type", "")
    if gtype == "Point":
        return inter, "ok"
    if gtype == "MultiPoint":
        pts = [p for p in getattr(inter, "geoms", []) if getattr(p, "geom_type", "") == "Point"]
        if pts:
            return pts[0], "ok"
        return None, "no_point"
    return None, gtype or "unsupported"


def build_main_cross_intersections(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mains = [rec for rec in records if rec.get("line_role") == "main"]
    crosses = [rec for rec in records if rec.get("line_role") == "cross"]
    mains.sort(key=lambda d: line_order_key(d.get("line_id", ""), d.get("row_index", 0)))
    crosses.sort(key=lambda d: line_order_key(d.get("line_id", ""), d.get("row_index", 0)))
    payload_items: List[Dict[str, Any]] = []
    for mi, mrec in enumerate(mains, start=1):
        for ci, crec in enumerate(crosses, start=1):
            pt, status = pick_intersection_point(mrec["geom"], crec["geom"])
            x = y = s_m = s_c = None
            if pt is not None:
                try:
                    x = float(pt.x)
                    y = float(pt.y)
                    s_m = float(mrec["geom"].project(pt))
                    s_c = float(crec["geom"].project(pt))
                except Exception:
                    x = y = s_m = s_c = None
                    status = "project_error"
            payload_items.append({
                "main_line_id": str(mrec["line_id"]),
                "cross_line_id": str(crec["line_id"]),
                "main_row_index": int(mrec.get("row_index", 0)),
                "cross_row_index": int(crec.get("row_index", 0)),
                "main_label_fixed": f"L{mi}",
                "main_order": int(mi),
                "cross_order": int(ci),
                "x": x,
                "y": y,
                "s_on_main": s_m,
                "s_on_cross": s_c,
                "status": status,
            })
    return payload_items


def save_main_cross_intersections(ui2_dir: str, records: List[Dict[str, Any]]) -> Optional[str]:
    items = build_main_cross_intersections(records)
    path = os.path.join(ui2_dir, "intersections_main_cross.json")
    payload = {
        "version": 1,
        "main_count": int(sum(1 for rec in records if rec.get("line_role") == "main")),
        "cross_count": int(sum(1 for rec in records if rec.get("line_role") == "cross")),
        "intersection_count": int(len(items)),
        "ok_count": int(sum(1 for it in items if it.get("status") in ("ok", "multi_point"))),
        "items": items,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
