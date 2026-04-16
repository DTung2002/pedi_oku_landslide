import csv
import os
from typing import Any, Dict, List, Tuple

SECTION_DIRECTION_VERSION = 3
SECTION_CHAINAGE_ORIGIN = "picked"
SECTION_CSV_FIELDNAMES = [
    "idx",
    "x1",
    "y1",
    "x2",
    "y2",
    "line_id",
    "line_role",
    "direction_version",
    "chainage_origin",
]


def reverse_section_points(p0: Tuple[float, float], p1: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))


def canonical_section_csv_row(
    idx: int,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    line_id: str = "",
    line_role: str = "",
) -> Dict[str, Any]:
    return {
        "idx": int(idx),
        "x1": float(p0[0]),
        "y1": float(p0[1]),
        "x2": float(p1[0]),
        "y2": float(p1[1]),
        "line_id": str(line_id or "").strip(),
        "line_role": str(line_role or "").strip(),
        "direction_version": int(SECTION_DIRECTION_VERSION),
        "chainage_origin": SECTION_CHAINAGE_ORIGIN,
    }


def delete_legacy_ui3_outputs(run_dir: str) -> None:
    if not run_dir:
        return
    for rel in (os.path.join("ui3", "curve"), os.path.join("ui3", "groups")):
        path = os.path.join(run_dir, rel)
        if not os.path.isdir(path):
            continue
        try:
            for name in os.listdir(path):
                file_path = os.path.join(path, name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception:
            continue


def read_sections_csv_rows(csv_path: str) -> List[Dict[str, Any]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_sections_csv_rows(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SECTION_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "idx": int(row.get("idx", 0) or 0),
                "x1": float(row.get("x1", 0.0) or 0.0),
                "y1": float(row.get("y1", 0.0) or 0.0),
                "x2": float(row.get("x2", 0.0) or 0.0),
                "y2": float(row.get("y2", 0.0) or 0.0),
                "line_id": str(row.get("line_id", "") or "").strip(),
                "line_role": str(row.get("line_role", "") or "").strip(),
                "direction_version": int(SECTION_DIRECTION_VERSION),
                "chainage_origin": SECTION_CHAINAGE_ORIGIN,
            })


def ensure_sections_csv_current(csv_path: str, *, run_dir: str) -> Tuple[List[Dict[str, Any]], bool]:
    rows = read_sections_csv_rows(csv_path)
    migrated = False
    canonical_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        try:
            x1 = float(row.get("x1"))
            y1 = float(row.get("y1"))
            x2 = float(row.get("x2"))
            y2 = float(row.get("y2"))
        except Exception:
            continue
        try:
            version = int(str(row.get("direction_version", "")).strip() or "0")
        except Exception:
            version = 0
        origin = str(row.get("chainage_origin", "") or "").strip().lower()
        is_current = (version >= SECTION_DIRECTION_VERSION) and (origin == SECTION_CHAINAGE_ORIGIN)
        if is_current:
            p0 = (x1, y1)
            p1 = (x2, y2)
        else:
            migrated = True
            p0, p1 = reverse_section_points((x1, y1), (x2, y2))
        canonical_rows.append(canonical_section_csv_row(
            int(row.get("idx") or i),
            p0,
            p1,
            line_id=str(row.get("line_id", row.get("name", "")) or "").strip(),
            line_role=str(row.get("line_role", row.get("role", "")) or "").strip(),
        ))
    if migrated:
        write_sections_csv_rows(csv_path, canonical_rows)
        delete_legacy_ui3_outputs(run_dir)
    return canonical_rows, migrated
