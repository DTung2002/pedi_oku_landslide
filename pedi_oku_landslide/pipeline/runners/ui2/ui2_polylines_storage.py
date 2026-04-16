import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shapely.geometry import LineString


def _normalize_vertex(vertex: Sequence[Any]) -> Optional[Tuple[float, float]]:
    try:
        x = float(vertex[0])
        y = float(vertex[1])
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return float(x), float(y)


def _normalize_vertices(vertices: Sequence[Sequence[Any]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for vertex in list(vertices or []):
        norm = _normalize_vertex(vertex)
        if norm is None:
            continue
        if not out or abs(out[-1][0] - norm[0]) > 1e-9 or abs(out[-1][1] - norm[1]) > 1e-9:
            out.append(norm)
    return out


def polyline_length_m(vertices: Sequence[Sequence[Any]]) -> float:
    pts = _normalize_vertices(vertices)
    if len(pts) < 2:
        return 0.0
    try:
        return float(LineString(pts).length)
    except Exception:
        return 0.0


def canonical_polyline_record(
    idx: int,
    vertices: Sequence[Sequence[Any]],
    *,
    line_id: str = "",
    line_role: str = "",
) -> Optional[Dict[str, Any]]:
    pts = _normalize_vertices(vertices)
    if len(pts) < 2:
        return None
    return {
        "idx": int(idx),
        "section_type": "polyline",
        "line_id": str(line_id or "").strip(),
        "line_role": str(line_role or "").strip(),
        "vertex_count": int(len(pts)),
        "length_m": float(polyline_length_m(pts)),
        "vertices": [[float(x), float(y)] for x, y in pts],
    }


def read_polylines_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = payload.get("polylines", [])
    if not isinstance(payload, list):
        return []
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            continue
        canonical = canonical_polyline_record(
            int(row.get("idx", idx) or idx),
            row.get("vertices", []),
            line_id=str(row.get("line_id", "") or "").strip(),
            line_role=str(row.get("line_role", "") or "").strip(),
        )
        if canonical is not None:
            out.append(canonical)
    return out


def write_polylines_json(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        canonical = canonical_polyline_record(
            int(row.get("idx", idx) or idx),
            row.get("vertices", []),
            line_id=str(row.get("line_id", "") or "").strip(),
            line_role=str(row.get("line_role", "") or "").strip(),
        )
        if canonical is not None:
            payload.append(canonical)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
