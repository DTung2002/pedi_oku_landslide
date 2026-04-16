import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import rasterio
from shapely.geometry import LineString

SECTION_DIRECTION_VERSION = 3
SECTION_CHAINAGE_ORIGIN = "picked"
DEFAULT_CRS = "EPSG:6678"
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


def canonical_section_csv_row(
    idx: int,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    *,
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


def delete_legacy_ui3_outputs_for_run(run_dir: str) -> None:
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


def ensure_sections_csv_current(csv_path: str, *, run_dir: str) -> bool:
    if not os.path.exists(csv_path):
        return False
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
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
            p0 = (x2, y2)
            p1 = (x1, y1)
        canonical_rows.append(
            canonical_section_csv_row(
                int(row.get("idx") or i),
                p0,
                p1,
                line_id=str(row.get("line_id", row.get("name", "")) or "").strip(),
                line_role=str(row.get("line_role", row.get("role", "")) or "").strip(),
            )
        )
    if migrated:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SECTION_CSV_FIELDNAMES)
            writer.writeheader()
            for row in canonical_rows:
                writer.writerow(row)
        delete_legacy_ui3_outputs_for_run(run_dir)
    return migrated


def build_gdf_from_sections_csv(csv_path: str, dem_path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"sections.csv not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row.get("idx") or 0)
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
            except Exception:
                continue
            line_id = str(row.get("line_id", row.get("name", "")) or "").strip()
            line_role = str(row.get("line_role", row.get("role", "")) or "").strip()
            rows.append((idx, x1, y1, x2, y2, line_id, line_role))

    if not rows:
        return gpd.GeoDataFrame(columns=["idx", "name", "line_id", "line_role", "length_m", "geometry"], geometry="geometry")

    crs = DEFAULT_CRS

    idxs, xs1, ys1, xs2, ys2, line_ids, line_roles = zip(*rows)
    geoms = [LineString([(x1, y1), (x2, y2)]) for (_, x1, y1, x2, y2, _, _) in rows]
    lengths = []
    for geom in geoms:
        try:
            lengths.append(float(geom.length))
        except Exception:
            lengths.append(float("nan"))

    names = [lid if str(lid).strip() else f"Line {i}" for i, lid in zip(idxs, line_ids)]
    return gpd.GeoDataFrame(
        {
            "idx": list(idxs),
            "name": names,
            "line_id": [str(v or "").strip() for v in line_ids],
            "line_role": [str(v or "").strip() for v in line_roles],
            "length_m": lengths,
        },
        geometry=geoms,
        crs=crs,
    )


def load_json(path: str, default: Optional[Any] = None) -> Any:
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
