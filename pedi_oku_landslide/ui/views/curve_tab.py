import math
import os
import json
import csv
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import geopandas as gpd
import rasterio
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap, QPixmapCache
from PyQt5.QtWidgets import (
    QAction,
    QAbstractSpinBox,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from shapely.geometry import LineString

from pedi_oku_landslide.pipeline.runners.ui3_backend import (
    auto_paths,
    compute_profile,
    estimate_slip_curve,
    evaluate_nurbs_curve,
    extract_curvature_rdp_nodes,
    fit_bezier_smooth_curve,
    render_profile_png,
)

WORKFLOW_GROUP_MIN_LEN_M = 0.0
WORKFLOW_GROUPING_PARAMS = {
    "rdp_eps_m": 0.5,
    "curvature_thr_abs": 0.02,
    "smooth_radius_m": 0.0,
    "include_curvature_threshold": True,
    "include_vector_angle_zero": True,
}
SECTION_DIRECTION_VERSION = 2
SECTION_CHAINAGE_ORIGIN = "right"
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


def _mk_boundary(x: float, reason: str, score: float, fixed: bool = False) -> Dict[str, Any]:
    return {"x": float(x), "reasons": [str(reason)], "score": float(score), "fixed": bool(fixed)}


def _has_curvature_reason(c: Dict[str, Any], curvature_reason_prefix: str = "curvature_gt_") -> bool:
    return any(str(r).startswith(curvature_reason_prefix) for r in list(c.get("reasons", [])))


def _merge_close_boundaries(cands: List[Dict[str, Any]], tol_m: float = 1e-6) -> List[Dict[str, Any]]:
    if not cands:
        return []
    cands = sorted(cands, key=lambda t: float(t["x"]))
    out: List[Dict[str, Any]] = []
    for c in cands:
        if not out:
            out.append(dict(c))
            continue
        if abs(float(c["x"]) - float(out[-1]["x"])) <= float(tol_m):
            out[-1]["fixed"] = bool(out[-1]["fixed"]) or bool(c["fixed"])
            out[-1]["reasons"] = sorted(set(list(out[-1]["reasons"]) + list(c["reasons"])))
            if float(c["score"]) > float(out[-1]["score"]):
                out[-1]["x"] = float(c["x"])
                out[-1]["score"] = float(c["score"])
                if "curvature_value" in c:
                    out[-1]["curvature_value"] = float(c["curvature_value"])
        else:
            out.append(dict(c))
    for cur in out:
        if not _has_curvature_reason(cur):
            cur.pop("curvature_value", None)
    return out


def _prune_first_20m_descending_curvature_pair(
    cands: List[Dict[str, Any]],
    *,
    window_m: float = 20.0,
    curvature_reason_prefix: str = "curvature_gt_",
    slip_start_reason: str = "slip_span_start",
) -> List[Dict[str, Any]]:
    if not cands:
        return []
    out = [dict(c) for c in sorted(cands, key=lambda t: float(t.get("x", 0.0)))]
    win = max(0.0, float(window_m))
    if win <= 0.0:
        return out

    slip_start_x: Optional[float] = None
    for c in out:
        reasons = [str(r) for r in list(c.get("reasons", []))]
        if any(r == slip_start_reason for r in reasons):
            try:
                slip_start_x = float(c.get("x", np.nan))
            except Exception:
                slip_start_x = None
            break
    if slip_start_x is None or not np.isfinite(slip_start_x):
        return out

    eligible_idxs: List[int] = []
    for idx, c in enumerate(out):
        if not _has_curvature_reason(c, curvature_reason_prefix):
            continue
        try:
            x = float(c.get("x", np.nan))
            kval = float(c.get("curvature_value", np.nan))
        except Exception:
            continue
        if not (np.isfinite(x) and np.isfinite(kval)):
            continue
        if not (float(slip_start_x) < x <= (float(slip_start_x) + win)):
            continue
        eligible_idxs.append(idx)

    for pos in range(len(eligible_idxs) - 1):
        left_idx = eligible_idxs[pos]
        right_idx = eligible_idxs[pos + 1]
        left_k = float(out[left_idx].get("curvature_value", np.nan))
        right_k = float(out[right_idx].get("curvature_value", np.nan))
        if not (np.isfinite(left_k) and np.isfinite(right_k)):
            continue
        if left_k <= right_k:
            continue

        for idx in sorted((left_idx, right_idx), reverse=True):
            del out[idx]
        return _merge_close_boundaries(out, tol_m=1e-6)
    return out


def _prune_vector_zero_boundaries(
    cands: List[Dict[str, Any]],
    *,
    first_curvature_gap_m: float = 2.0,
    repeat_vector_gap_m: float = 10.0,
    curvature_reason_prefix: str = "curvature_gt_",
    vector_reason: str = "vector_angle_zero_deg",
) -> List[Dict[str, Any]]:
    if not cands:
        return []
    out = [dict(c) for c in sorted(cands, key=lambda t: float(t.get("x", 0.0)))]

    def _has_vector_reason(c: Dict[str, Any]) -> bool:
        return any(str(r) == vector_reason for r in list(c.get("reasons", [])))

    def _drop_vector_reason(idx: int) -> None:
        cur = dict(out[idx])
        reasons = [str(r) for r in list(cur.get("reasons", [])) if str(r) != vector_reason]
        if reasons or bool(cur.get("fixed", False)):
            cur["reasons"] = reasons
            out[idx] = cur
        else:
            del out[idx]

    # Rule 1: the first surviving vector boundary must not be closer than 2 m
    # to the nearest preceding curvature boundary.
    while True:
        first_idx = next((i for i, c in enumerate(out) if _has_vector_reason(c)), None)
        if first_idx is None:
            break
        curv_idx = None
        for j in range(first_idx - 1, -1, -1):
            if _has_curvature_reason(out[j], curvature_reason_prefix):
                curv_idx = j
                break
        if curv_idx is None:
            break
        dist = float(out[first_idx]["x"]) - float(out[curv_idx]["x"])
        if dist < float(first_curvature_gap_m):
            _drop_vector_reason(first_idx)
            continue
        break

    # Rule 2: with UI3's reversed logic, keep only the first surviving vector
    # boundary seen from the right side; drop all later vector boundaries.
    first_kept_vector_x: Optional[float] = None
    i = len(out) - 1
    while i >= 0:
        if not _has_vector_reason(out[i]):
            i -= 1
            continue
        x = float(out[i]["x"])
        if first_kept_vector_x is None:
            first_kept_vector_x = x
            i -= 1
            continue
        _drop_vector_reason(i)
        i -= 1

    return _merge_close_boundaries(out, tol_m=1e-6)


def _normalized_vector_angle_deg(theta_deg: np.ndarray) -> np.ndarray:
    theta_deg = np.asarray(theta_deg, dtype=float)
    return ((theta_deg + 90.0) % 180.0) - 90.0


def _vector_horizontal_boundaries(
    chain: np.ndarray,
    theta_deg: np.ndarray,
    smin: float,
    smax: float,
    zero_tol_deg: float = 1e-6,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    chain = np.asarray(chain, dtype=float)
    theta_deg = np.asarray(theta_deg, dtype=float)
    if chain.ndim != 1 or theta_deg.ndim != 1 or chain.size != theta_deg.size or chain.size < 1:
        return out

    theta_norm = _normalized_vector_angle_deg(theta_deg)
    keep = np.isfinite(chain) & np.isfinite(theta_norm) & (chain >= float(smin)) & (chain <= float(smax))
    if int(np.count_nonzero(keep)) <= 0:
        return out

    c = np.asarray(chain[keep], dtype=float)
    t = np.asarray(theta_norm[keep], dtype=float)
    order = np.argsort(c)
    c = c[order]
    t = t[order]

    for i in range(c.size):
        if abs(float(t[i])) <= float(zero_tol_deg):
            out.append(_mk_boundary(float(c[i]), "vector_angle_zero_deg", score=0.0, fixed=False))

    for i in range(c.size - 1):
        c0 = float(c[i]); c1 = float(c[i + 1])
        t0 = float(t[i]); t1 = float(t[i + 1])
        if not (np.isfinite(c0) and np.isfinite(c1) and np.isfinite(t0) and np.isfinite(t1)):
            continue
        if c1 <= c0:
            continue
        if abs(t0) <= float(zero_tol_deg) or abs(t1) <= float(zero_tol_deg):
            continue
        if (t0 * t1) >= 0.0:
            continue
        den = (t1 - t0)
        if abs(den) <= 1e-12:
            continue
        frac = float(np.clip(-t0 / den, 0.0, 1.0))
        x_cross = c0 + frac * (c1 - c0)
        if smin <= x_cross <= smax:
            out.append(_mk_boundary(x_cross, "vector_angle_zero_deg", score=0.0, fixed=False))
    return _merge_close_boundaries(out, tol_m=1e-6)


def auto_group_profile_by_criteria(
    prof: Dict[str, Any],
    rdp_eps_m: float = 0.5,
    curvature_thr_abs: float = 0.02,
    smooth_radius_m: float = 0.0,
    include_curvature_threshold: bool = True,
    include_vector_angle_zero: bool = True,
) -> List[Dict[str, Any]]:
    chain = np.asarray(prof.get("chain", []), dtype=float)
    slip_mask = np.asarray(prof.get("slip_mask", [])) if prof.get("slip_mask", None) is not None else None
    if slip_mask is not None and slip_mask.shape == chain.shape:
        finite_mask = np.isfinite(chain) & (slip_mask == True)
        if int(np.count_nonzero(finite_mask)) >= 2:
            smin = float(np.nanmin(chain[finite_mask]))
            smax = float(np.nanmax(chain[finite_mask]))
        elif "slip_span" in prof and prof["slip_span"]:
            smin, smax = map(float, prof["slip_span"])
        else:
            elev_raw = np.asarray(prof.get("elev_orig", []), dtype=float)
            elev_fallback = np.asarray(prof.get("elev_s", []), dtype=float)
            if elev_raw.ndim != 1 or elev_raw.size != chain.size:
                elev_raw = elev_fallback
            finite = np.isfinite(chain) & np.isfinite(elev_raw)
            if int(np.count_nonzero(finite)) < 2:
                return []
            smin = float(np.nanmin(chain[finite]))
            smax = float(np.nanmax(chain[finite]))
    elif "slip_span" in prof and prof["slip_span"]:
        smin, smax = map(float, prof["slip_span"])
    else:
        elev_raw = np.asarray(prof.get("elev_orig", []), dtype=float)
        elev_fallback = np.asarray(prof.get("elev_s", []), dtype=float)
        if elev_raw.ndim != 1 or elev_raw.size != chain.size:
            elev_raw = elev_fallback
        finite = np.isfinite(chain) & np.isfinite(elev_raw)
        if int(np.count_nonzero(finite)) < 2:
            return []
        smin = float(np.nanmin(chain[finite]))
        smax = float(np.nanmax(chain[finite]))
    if smax < smin:
        smin, smax = smax, smin
    if smax <= smin:
        return []

    boundaries_meta: List[Dict[str, Any]] = [
        _mk_boundary(float(smin), "slip_span_start", score=0.0, fixed=True),
        _mk_boundary(float(smax), "slip_span_end", score=0.0, fixed=True),
    ]

    nodes = extract_curvature_rdp_nodes(
        prof,
        rdp_eps_m=float(rdp_eps_m),
        smooth_radius_m=float(smooth_radius_m),
        restrict_to_slip_span=False,
    )
    xs = np.asarray(nodes.get("chain", []), dtype=float)
    ks = np.asarray(nodes.get("curvature", []), dtype=float)
    if bool(include_curvature_threshold) and xs.size >= 3 and ks.size == xs.size:
        for i in range(1, xs.size - 1):
            x = float(xs[i])
            k = float(ks[i])
            if not (np.isfinite(x) and np.isfinite(k)):
                continue
            if not (smin < x < smax):
                continue
            if abs(k) > float(curvature_thr_abs):
                curv_boundary = _mk_boundary(
                    x,
                    f"curvature_gt_{float(curvature_thr_abs):.2f}",
                    score=abs(k),
                    fixed=False,
                )
                curv_boundary["curvature_value"] = float(k)
                boundaries_meta.append(curv_boundary)

    if bool(include_vector_angle_zero):
        theta = np.asarray(prof.get("theta", []), dtype=float)
        chain = np.asarray(prof.get("chain", []), dtype=float)
        boundaries_meta.extend(_vector_horizontal_boundaries(chain, theta, float(smin), float(smax)))

    boundaries_meta = _merge_close_boundaries(boundaries_meta, tol_m=1e-6)
    if bool(include_curvature_threshold):
        boundaries_meta = _prune_first_20m_descending_curvature_pair(boundaries_meta, window_m=20.0)
    if bool(include_vector_angle_zero):
        boundaries_meta = _prune_vector_zero_boundaries(
            boundaries_meta,
            first_curvature_gap_m=2.0,
            repeat_vector_gap_m=10.0,
        )
    boundaries_meta = [b for b in boundaries_meta if np.isfinite(float(b.get("x", np.nan)))]
    boundaries_meta.sort(key=lambda t: float(t["x"]))
    if len(boundaries_meta) < 2:
        return [{
            "id": "G1",
            "start": float(smin),
            "end": float(smax),
            "color": "#1f77b4",
            "start_reason": "slip_span_start",
            "end_reason": "slip_span_end",
        }]

    uniq_boundaries: List[Dict[str, Any]] = []
    for b in boundaries_meta:
        if not uniq_boundaries or abs(float(b["x"]) - float(uniq_boundaries[-1]["x"])) > 1e-6:
            uniq_boundaries.append(dict(b))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    groups: List[Dict[str, Any]] = []
    for bi, (left_b, right_b) in enumerate(zip(uniq_boundaries[:-1], uniq_boundaries[1:])):
        s = float(left_b["x"])
        e = float(right_b["x"])
        if e <= s:
            continue
        idx = len(groups) + 1
        groups.append({
            "id": f"G{idx}",
            "start": float(s),
            "end": float(e),
            "color": colors[(idx - 1) % len(colors)],
            "start_reason": "+".join(left_b.get("reasons", [])) or "boundary",
            "end_reason": "+".join(right_b.get("reasons", [])) or "boundary",
        })
    return _renumber_groups_visual_order(groups)


def clamp_groups_to_slip(prof: Dict[str, Any], groups: List[Dict[str, Any]], min_len: float = WORKFLOW_GROUP_MIN_LEN_M) -> List[Dict[str, Any]]:
    chain = np.asarray(prof.get("chain", []), dtype=float)
    elevs = np.asarray(prof.get("elev_s", []), dtype=float)
    slip_mask = np.asarray(prof.get("slip_mask", [])) if prof.get("slip_mask", None) is not None else None
    if slip_mask is not None and slip_mask.shape == chain.shape:
        keep = np.isfinite(chain) & (slip_mask == True)
        if int(np.count_nonzero(keep)) > 0:
            smin = float(np.nanmin(chain[keep]))
            smax = float(np.nanmax(chain[keep]))
        elif "slip_span" in prof and prof["slip_span"]:
            smin, smax = map(float, prof["slip_span"])
        else:
            keep = np.isfinite(chain) & np.isfinite(elevs)
            if int(np.count_nonzero(keep)) <= 0:
                return []
            smin = float(np.nanmin(chain[keep]))
            smax = float(np.nanmax(chain[keep]))
    elif "slip_span" in prof and prof["slip_span"]:
        smin, smax = map(float, prof["slip_span"])
    else:
        keep = np.isfinite(chain) & np.isfinite(elevs)
        if int(np.count_nonzero(keep)) <= 0:
            return []
        smin = float(np.nanmin(chain[keep]))
        smax = float(np.nanmax(chain[keep]))
    if smax < smin:
        smin, smax = smax, smin

    out: List[Dict[str, Any]] = []
    for g in (groups or []):
        try:
            s = float(g.get("start", g.get("start_chainage", 0.0)))
            e = float(g.get("end", g.get("end_chainage", 0.0)))
        except Exception:
            continue
        if e < s:
            s, e = e, s
        s = max(s, smin)
        e = min(e, smax)
        if (e - s) >= float(min_len):
            out.append({
                "id": str(g.get("id", f"G{len(out) + 1}")),
                "start": float(s),
                "end": float(e),
                "color": g.get("color", None),
                "start_reason": g.get("start_reason", ""),
                "end_reason": g.get("end_reason", ""),
            })
    return _renumber_groups_visual_order(out)


def _renumber_groups_visual_order(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    UI3 now displays chainage with origin at the right.
    Canonical group order is ascending chainage, so G1 always owns the
    rightmost span and the table is populated in that canonical order.
    """
    if not groups:
        return []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    ordered: List[Dict[str, Any]] = []
    sortable: List[Tuple[float, float, Dict[str, Any]]] = []
    for g in (groups or []):
        try:
            s = float(g.get("start", g.get("start_chainage", np.nan)))
            e = float(g.get("end", g.get("end_chainage", np.nan)))
        except Exception:
            continue
        if not (np.isfinite(s) and np.isfinite(e)):
            continue
        if e < s:
            s, e = e, s
        gg = dict(g or {})
        gg["start"] = float(s)
        gg["end"] = float(e)
        sortable.append((float(s), float(e), gg))
    sortable.sort(key=lambda t: (t[0], t[1]))
    for idx, (_, _, gg) in enumerate(sortable, start=1):
        gg["id"] = f"G{idx}"
        color = str(gg.get("color", "") or "").strip()
        if not color:
            color = colors[(idx - 1) % len(colors)]
        gg["color"] = color
        ordered.append(gg)
    return ordered


def _canonical_section_csv_row(
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


def _delete_legacy_ui3_outputs_for_run(run_dir: str) -> None:
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


def _ensure_sections_csv_current(csv_path: str, *, run_dir: str) -> bool:
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
        canonical_rows.append(_canonical_section_csv_row(
            int(row.get("idx") or i),
            p0,
            p1,
            line_id=str(row.get("line_id", row.get("name", "")) or "").strip(),
            line_role=str(row.get("line_role", row.get("role", "")) or "").strip(),
        ))
    if migrated:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SECTION_CSV_FIELDNAMES)
            writer.writeheader()
            for row in canonical_rows:
                writer.writerow(row)
        _delete_legacy_ui3_outputs_for_run(run_dir)
    return migrated


def _build_gdf_from_sections_csv(csv_path: str, dem_path: str) -> gpd.GeoDataFrame:
    """
    Đọc ui2/sections.csv (idx, x1, y1, x2, y2) và tạo GeoDataFrame lines.
    CRS lấy từ DEM (dem_path).
    """
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

    # Lấy CRS từ DEM
    crs = None
    try:
        with rasterio.open(dem_path) as ds:
            crs = ds.crs
    except Exception:
        pass

    idxs, xs1, ys1, xs2, ys2, line_ids, line_roles = zip(*rows)
    geoms = [LineString([(x1, y1), (x2, y2)]) for (_, x1, y1, x2, y2, _, _) in rows]

    lengths = []
    for g in geoms:
        try:
            L = float(g.length)
        except Exception:
            L = float("nan")
        lengths.append(L)

    names = [lid if str(lid).strip() else f"Line {i}" for i, lid in zip(idxs, line_ids)]

    gdf = gpd.GeoDataFrame(
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
    return gdf


class AnchorMarkerItem(QGraphicsEllipseItem):
    def __init__(
        self,
        x: float,
        y: float,
        r: float,
        tooltip: str,
        on_click: Optional[Callable[[], None]] = None,
    ):
        super().__init__(x - r, y - r, 2 * r, 2 * r)
        self._on_click = on_click
        self.setAcceptHoverEvents(True)
        self.setToolTip(tooltip)

    def mousePressEvent(self, event):
        try:
            if self._on_click is not None:
                self._on_click()
        except Exception:
            pass
        super().mousePressEvent(event)

class ZoomableGraphicsView(QGraphicsView):
    sceneMouseMoved = pyqtSignal(float, float)
    hoverExited = pyqtSignal()

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._zoom = 0

    def wheelEvent(self, e):
        zoom_in = 1.25
        zoom_out = 0.8
        if e.angleDelta().y() > 0:
            factor = zoom_in;  self._zoom += 1
        else:
            factor = zoom_out; self._zoom -= 1
        self.scale(factor, factor)

    def fit_to_scene(self):
        if not self.scene() or not self.scene().items():
            return
        r = self.scene().itemsBoundingRect()
        if r.isNull(): return
        self.resetTransform()
        self.fitInView(r, Qt.KeepAspectRatio)
        self._zoom = 0

    def set_100(self):
        self.resetTransform()
        self._zoom = 0

    def zoom_in(self):
        self.scale(1.25, 1.25)
        self._zoom += 1

    def zoom_out(self):
        self.scale(0.8, 0.8)
        self._zoom -= 1

    def mouseMoveEvent(self, e):
        try:
            sp = self.mapToScene(e.pos())
            self.sceneMouseMoved.emit(float(sp.x()), float(sp.y()))
        except Exception:
            pass
        super().mouseMoveEvent(e)

    def leaveEvent(self, e):
        try:
            self.hoverExited.emit()
        except Exception:
            pass
        super().leaveEvent(e)


class KeyboardOnlySpinBox(QSpinBox):
    """Disable wheel changes; allow keyboard input only."""
    def wheelEvent(self, event):
        event.ignore()


class KeyboardOnlyDoubleSpinBox(QDoubleSpinBox):
    """Disable wheel changes; allow keyboard input only."""
    def wheelEvent(self, event):
        event.ignore()


class NoWheelComboBox(QComboBox):
    """Ignore wheel to allow parent scroll area to consume mouse wheel."""
    def wheelEvent(self, event):
        event.ignore()


Section = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


class CurveAnalyzeTab(QWidget):
    """
    UI3 (refactor): khung làm việc phân tích đường cong.
    - Nhận context từ Analyze/Section (project/run/run_dir)
    - Đọc danh sách sections từ UI2/sections.csv
    - Cho phép chọn line; hiển thị status; vẽ placeholder đồ thị (dz & slope)
    """

    # (Optional) khi bạn muốn phát tín hiệu đã lưu JSON v.v.
    curve_saved = pyqtSignal(str)  # emit path
    _GROUND_EXPORT_STEP_M = 0.2

    def __init__(self, base_dir: str):
        super().__init__()
        self.base_dir = base_dir
        self._ctx: Dict[str, str] = {"project": "", "run_label": "", "run_dir": ""}
        self._splitter: Optional[QSplitter] = None
        self._left_scroll: Optional[QScrollArea] = None
        self._left_min_w = 380
        self._left_default_w = 490
        self._pending_init_splitter = True

        self._ax_top = None  # dict: {x_min,x_max,left_px,top_px,width_px,height_px}
        self._ax_bot = None

        # paths từ UI1/UI2
        self.dem_path = ""
        self.dem_path_raw = ""
        self.dem_path_smooth = ""
        self.ground_export_dem_path = ""
        self._default_profile_step_m = 0.20
        self.dx_path = ""
        self.dy_path = ""
        self.dz_path = ""
        self.lines_path = ""
        self.slip_path = ""
        self.profile_source_combo = None
        self.curvature_check = None
        self.vector_zero_check = None
        self.rdp_eps_spin = None
        self.smooth_radius_spin = None

        # UI widgets chính (để dùng lại)
        self.line_combo = None
        self.status = None
        self.scene = None
        self.view = None
        self.group_table = None
        # --- state for grouping/guide overlays (phải ở CurveAnalyzeTab) ---
        self._px_per_m: Optional[float] = None  # pixels per meter
        self._sec_len_m: Optional[float] = None  # chiều dài tuyến (m)
        self._group_bounds: Dict[str, List[float]] = {}  # {line_id: [x_m ...]}
        self._guide_lines_top: List[QGraphicsLineItem] = []
        self._guide_lines_bot: List[QGraphicsLineItem] = []
        self._group_bands_bot: List[QGraphicsRectItem] = []
        self._img_ground: Optional[QGraphicsPixmapItem] = None
        self._img_rate0: Optional[QGraphicsPixmapItem] = None
        self._curve_method_by_line: Dict[str, str] = {}
        self._active_prof: Optional[dict] = None
        self._active_groups: List[dict] = []
        self._active_base_curve: Optional[dict] = None
        self._active_curve: Optional[dict] = None
        self._curve_overlay_item: Optional[QGraphicsPathItem] = None
        self._cp_overlay_items: List[Any] = []
        self._anchor_overlay_items: List[Any] = []
        self._ui2_intersections_cache: Optional[Dict[str, Any]] = None
        self._anchors_xyz_cache: Optional[Dict[str, Any]] = None
        self._nurbs_params_by_line: Dict[str, Dict[str, Any]] = {}
        self._group_table_updating: bool = False
        self._nurbs_updating_ui: bool = False
        # True when background image already has a baked slip-curve (profile_*_nurbs.png).
        self._static_nurbs_bg_loaded: bool = False
        self._nurbs_live_timer = QTimer(self)
        self._nurbs_live_timer.setSingleShot(True)
        self._nurbs_live_timer.setInterval(30)
        self._nurbs_live_timer.timeout.connect(self._on_nurbs_live_tick)

        self._plot_x0_px = None  # ax_left_px trong PNG
        self._plot_w_px = None  # ax_width_px trong PNG
        self._x_min = None  # trục x (chainage) min trên hình
        self._x_max = None  # trục x (chainage) max trên hình
        self._profile_cursor_label = None

        self._build_ui()
    #
    # def _clear_group_guides(self) -> None:
    #     # Chỉ remove nếu item vẫn còn thuộc scene hiện tại
    #     cur_scene = getattr(self, "scene", None)
    #     for it in self._guide_lines_top:
    #         sc = it.scene()
    #         if sc is not None and (cur_scene is None or sc is cur_scene):
    #             sc.removeItem(it)
    #     for it in self._guide_lines_bot:
    #         sc = it.scene()
    #         if sc is not None and (cur_scene is None or sc is cur_scene):
    #             sc.removeItem(it)
    #     for it in self._group_bands_bot:
    #         sc = it.scene()
    #         if sc is not None and (cur_scene is None or sc is cur_scene):
    #             sc.removeItem(it)
    #     self._guide_lines_top.clear()
    #     self._guide_lines_bot.clear()
    #     self._group_bands_bot.clear()
    # #
    # def _chainage_to_xpx(self, x_m: float, panel: str = "top") -> float:
    #     ax = self._ax_top if panel == "top" else self._ax_bot
    #     if not ax or self._img_ground is None:
    #         return 0.0
    #     x_min = float(ax["x_min"]);
    #     x_max = float(ax["x_max"])
    #     left = float(ax["left_px"]);
    #     width = float(ax["width_px"])
    #     if x_max <= x_min or width <= 0.0:
    #         return 0.0
    #     px_per_m = width / (x_max - x_min)
    #     x_local = left + (float(x_m) - x_min) * px_per_m
    #     return float(self._img_ground.pos().x()) + x_local

    # def _line_id_current(self) -> str:
    #     """Trả về nhãn line hiện tại; fallback theo index nếu label trống."""
    #     if self.line_combo is None or self.line_combo.count() == 0:
    #         return "line_000"
    #     label = (self.line_combo.currentText() or "").strip()
    #     if label:
    #         return label
    #     return f"line_{self.line_combo.currentIndex() + 1:03d}"
    #
    # def _draw_group_guides_for_current_line(self) -> None:
    #     # guard
    #     if self.scene is None or self._img_ground is None:
    #         return
    #     if not self._ax_top or not self._ax_bot:
    #         return
    #
    #     line_id = self._line_id_current()
    #     bounds = self._group_bounds.get(line_id)
    #     if not bounds or len(bounds) < 2:
    #         self._clear_group_guides()
    #         return
    #
    #     self._clear_group_guides()
    #
    #     # Toạ độ gốc pixmap trong scene
    #     x0_img = float(self._img_ground.pos().x())
    #     y0_img = float(self._img_ground.pos().y())
    #
    #     # Pen dashed
    #     pen = QPen(QColor("#444444"))
    #     pen.setCosmetic(True)
    #     pen.setWidth(0)
    #     pen.setStyle(Qt.DashLine)
    #
    #     # Panel TOP
    #     t = self._ax_top  # {x_min,x_max,left_px,top_px,width_px,height_px}
    #     y_top = y0_img + float(t["top_px"])
    #     h_top = float(t["height_px"])
    #
    #     # Panel BOT
    #     b = self._ax_bot
    #     y_bot = y0_img + float(b["top_px"])
    #     h_bot = float(b["height_px"])
    #
    #     for x_m in bounds:
    #         # x theo từng panel (vì có thể x_min/x_max/bbox khác nhau)
    #         x_px_top = self._chainage_to_xpx(x_m, panel="top")
    #         x_px_bot = self._chainage_to_xpx(x_m, panel="bot")
    #
    #         ln_top = self.scene.addLine(x_px_top, y_top, x_px_top, y_top + h_top, pen)
    #         ln_bot = self.scene.addLine(x_px_bot, y_bot, x_px_bot, y_bot + h_bot, pen)
    #         self._guide_lines_top.append(ln_top)
    #         self._guide_lines_bot.append(ln_bot)
    #
    #     # Không tô band
    #     self._group_bands_bot.clear()

    @staticmethod
    def _normalize_curve_method(method: Optional[str]) -> str:
        m = str(method or "").strip().lower()
        if m == "nurbs":
            return "nurbs"
        return "bezier"

    @staticmethod
    def _curve_method_from_group_method(group_method: Optional[str]) -> str:
        gm = str(group_method or "").strip().lower()
        # Auto Group method only controls grouping strategy. Curve editing/rendering is NURBS for all methods.
        if gm in (
            "traditional",
            "new",
            "test",
            "raw_dem_curvature",
            "profile_dem_rdp_curvature_theta0",
            "profile_dem_rdp_curvature_only",
            "profile_dem_rdp_theta0_only",
            "profile_dem_rdp_span_only",
        ):
            return "nurbs"
        return "nurbs"

    def _set_curve_method_for_line(self, line_id: str, curve_method: Optional[str]) -> str:
        cm = self._normalize_curve_method(curve_method)
        self._curve_method_by_line[line_id] = cm
        return cm

    def _get_curve_method_for_line(self, line_id: str) -> str:
        cm = self._curve_method_by_line.get(line_id, "")
        if cm:
            return self._normalize_curve_method(cm)

        js_path = self._groups_json_path_for(line_id)
        if os.path.exists(js_path):
            try:
                with open(js_path, "r", encoding="utf-8") as f:
                    js = json.load(f) or {}
                cm = str(js.get("curve_method", "")).strip().lower()
                if not cm:
                    cm = self._curve_method_from_group_method(js.get("group_method"))
                cm = self._set_curve_method_for_line(line_id, cm)
                return cm
            except Exception:
                pass

        return self._set_curve_method_for_line(line_id, "bezier")

    def _save_groups_to_ui(
        self,
        groups: list,
        prof: dict,
        line_id: str,
        log_text: Optional[str] = None,
        curve_method: Optional[str] = None,
        group_method: Optional[str] = None,
    ) -> None:
        cm = self._set_curve_method_for_line(line_id, curve_method or self._curve_method_by_line.get(line_id))
        try:
            groups_for_json = self._groups_with_median_theta(groups, prof)
            curvature_points = self._curvature_points_for_json(prof, groups_for_json)
            js = {
                "line": self.line_combo.currentText(),
                "groups": groups_for_json,
                "curvature_points": curvature_points,
                "chainage_origin": self._ui3_chainage_origin(),
                "curve_method": cm,
                "profile_dem_source": self._current_profile_source_key(),
                "profile_dem_path": str(getattr(self, "dem_path", "") or "").replace("\\", "/"),
                "rdp_eps_m": self._current_rdp_eps_m(),
                "smooth_radius_m": self._current_smooth_radius_m(),
                "include_curvature_threshold": self._include_curvature_threshold(),
                "include_vector_angle_zero": self._include_vector_angle_zero(),
            }
            if group_method:
                js["group_method"] = str(group_method)
            with open(self._groups_json_path(), "w", encoding="utf-8") as f:
                json.dump(js, f, ensure_ascii=False, indent=2)
            self._log(f"[✓] Saved group definition: {self._groups_json_path()}")
        except Exception as e:
            self._warn(f"[UI3] Cannot save groups JSON: {e}")

        if "length_m" in prof and prof["length_m"] is not None:
            length_m = float(prof["length_m"])
        else:
            ch = prof.get("chain")
            length_m = float(ch[-1] - ch[0]) if ch is not None and len(ch) >= 2 else None

        if self.group_table is not None:
            self._group_table_updating = True
            try:
                self.group_table.setRowCount(0)
                for i, g in enumerate(groups, 1):
                    self.group_table.insertRow(self.group_table.rowCount())
                    self.group_table.setItem(i - 1, 0, QTableWidgetItem(str(g.get("id", f"G{i}"))))
                    self.group_table.setItem(i - 1, 1, QTableWidgetItem(f'{float(g.get("start", 0.0)):.3f}'))
                    self.group_table.setItem(i - 1, 2, QTableWidgetItem(f'{float(g.get("end", 0.0)):.3f}'))
                    self._set_group_boundary_reason(i - 1, 1, str(g.get("start_reason", "") or ""))
                    self._set_group_boundary_reason(i - 1, 2, str(g.get("end_reason", "") or ""))
                    self._set_color_cell(i - 1, str(g.get("color", "")).strip())
                # IMPORTANT: keep itemChanged suppressed for the synthetic UNGROUPED row too.
                self._append_ungrouped_row(groups, length_m)
            finally:
                self._group_table_updating = False

        bounds_set = set()
        for g in groups:
            s = float(g.get("start", 0.0))
            e = float(g.get("end", 0.0))
            if e < s:
                s, e = e, s
            if length_m:
                s = max(0.0, min(length_m, s))
                e = max(0.0, min(length_m, e))
            bounds_set.add(s)
            bounds_set.add(e)
        bounds_m = sorted(bounds_set)

        self._group_bounds[line_id] = bounds_m
        self._sec_len_m = length_m

        if self._px_per_m is None and getattr(self, "_img_ground", None) and self._sec_len_m:
            W = self._img_ground.pixmap().width()
            self._px_per_m = float(W) / float(self._sec_len_m)

        if log_text:
            self._ok(log_text)

    def _load_axes_meta(self, png_path: str) -> None:
        self._ax_top = None
        self._ax_bot = None
        try:
            meta_path = png_path.rsplit(".", 1)[0] + ".json"
            if not os.path.exists(meta_path):
                return
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
            top = meta.get("top") or {}
            bot = meta.get("bot") or {}
            if top:
                self._ax_top = top
            if bot:
                self._ax_bot = bot
        except Exception:
            self._ax_top = None
            self._ax_bot = None

    def _clear_curve_overlay(self) -> None:
        it = self._curve_overlay_item
        if it is not None:
            try:
                sc = it.scene()
                if sc is not None:
                    sc.removeItem(it)
            except Exception:
                pass
        self._curve_overlay_item = None
        self._clear_control_points_overlay()
        self._clear_anchor_overlay()

    def _clear_control_points_overlay(self) -> None:
        for it in (self._cp_overlay_items or []):
            try:
                sc = it.scene()
                if sc is not None:
                    sc.removeItem(it)
            except Exception:
                pass
        self._cp_overlay_items = []

    def _clear_anchor_overlay(self) -> None:
        for it in (self._anchor_overlay_items or []):
            try:
                sc = it.scene()
                if sc is not None:
                    sc.removeItem(it)
            except Exception:
                pass
        self._anchor_overlay_items = []

    def _chain_elev_to_scene_xy(self, chain_m: float, elev_m: float) -> Optional[Tuple[float, float]]:
        if self._img_ground is None or self._ax_top is None:
            return None
        try:
            ax = self._ax_top
            x0 = float(ax.get("x_min"))
            x1 = float(ax.get("x_max"))
            y_min = float(ax.get("y_min"))
            y_max = float(ax.get("y_max"))
            left_px = float(ax.get("left_px"))
            top_px = float(ax.get("top_px"))
            w_px = float(ax.get("width_px"))
            h_px = float(ax.get("height_px"))
            if not (abs(x1 - x0) > 1e-12 and y_max > y_min and w_px > 0 and h_px > 0):
                return None
            xr = (float(chain_m) - x0) / (x1 - x0)
            yr = (y_max - float(elev_m)) / (y_max - y_min)
            x_local = left_px + xr * w_px
            y_local = top_px + yr * h_px
            x_scene = float(self._img_ground.pos().x()) + x_local
            y_scene = float(self._img_ground.pos().y()) + y_local
            return x_scene, y_scene
        except Exception:
            return None

    def _scene_xy_to_chain_elev(self, scene_x: float, scene_y: float) -> Optional[Tuple[float, float]]:
        if self._img_ground is None or self._ax_top is None:
            return None
        try:
            ax = self._ax_top
            x0 = float(ax.get("x_min"))
            x1 = float(ax.get("x_max"))
            y_min = float(ax.get("y_min"))
            y_max = float(ax.get("y_max"))
            left_px = float(ax.get("left_px"))
            top_px = float(ax.get("top_px"))
            w_px = float(ax.get("width_px"))
            h_px = float(ax.get("height_px"))
            if not (abs(x1 - x0) > 1e-12 and y_max > y_min and w_px > 0 and h_px > 0):
                return None

            x_local = float(scene_x) - float(self._img_ground.pos().x())
            y_local = float(scene_y) - float(self._img_ground.pos().y())
            if not (left_px <= x_local <= (left_px + w_px) and top_px <= y_local <= (top_px + h_px)):
                return None

            xr = (x_local - left_px) / w_px
            yr = (y_local - top_px) / h_px
            chain_m = x0 + xr * (x1 - x0)
            elev_m = y_max - yr * (y_max - y_min)
            return float(chain_m), float(elev_m)
        except Exception:
            return None

    def _clear_profile_cursor_readout(self) -> None:
        if self._profile_cursor_label is not None:
            self._profile_cursor_label.setText("Cursor: —")

    def _on_profile_scene_mouse_moved(self, scene_x: float, scene_y: float) -> None:
        if self._profile_cursor_label is None:
            return
        vals = self._scene_xy_to_chain_elev(scene_x, scene_y)
        if vals is None:
            self._clear_profile_cursor_readout()
            return
        chain_m, elev_m = vals
        self._profile_cursor_label.setText(f"Cursor: chainage={chain_m:.3f} m, elev={elev_m:.3f} m")

    def _on_anchor_marker_clicked(self, anchor: dict) -> None:
        try:
            lbl = str(anchor.get("main_label_fixed", "")).strip() or "Anchor"
            self._log(
                f"[UI3] {lbl}: x={float(anchor.get('x')):.3f}, "
                f"y={float(anchor.get('y')):.3f}, z={float(anchor.get('z')):.3f}, "
                f"s_aux={float(anchor.get('s_on_cross')):.3f}"
            )
        except Exception:
            pass

    def _refresh_anchor_overlay(self) -> None:
        self._clear_anchor_overlay()
        if self.scene is None or self._img_ground is None or self._ax_top is None:
            return
        if self._current_ui2_line_role() != "cross":
            return
        cross_id = self._current_ui2_line_id()
        anchors = self._anchors_for_cross_line(cross_id, require_ready=True)
        if len(anchors) < 3:
            return

        colors = {
            1: "#e74c3c",  # L1
            2: "#2ecc71",  # L2
            3: "#3498db",  # L3
        }
        for a in anchors:
            try:
                s = float(a.get("s_on_cross"))
                z = float(a.get("z"))
            except Exception:
                continue
            pt = self._chain_elev_to_scene_xy(s, z)
            if pt is None:
                continue
            x, y = pt
            main_order = int(a.get("main_order", 0)) if str(a.get("main_order", "")).strip() else 0
            label = str(a.get("main_label_fixed", "")).strip() or f"L{main_order if main_order > 0 else '?'}"
            col = colors.get(main_order, "#f39c12")
            tip = (
                f"{label}\n"
                f"x={float(a.get('x')):.3f}\n"
                f"y={float(a.get('y')):.3f}\n"
                f"z={float(a.get('z')):.3f}\n"
                f"s_aux={float(a.get('s_on_cross')):.3f}"
            )
            marker = AnchorMarkerItem(
                x=x, y=y, r=7.0, tooltip=tip,
                on_click=lambda aa=dict(a): self._on_anchor_marker_clicked(aa)
            )
            marker.setBrush(QColor(col))
            pen = QPen(QColor("#ffffff"))
            pen.setWidth(2)
            pen.setCosmetic(True)
            marker.setPen(pen)
            marker.setZValue(145.0)
            self.scene.addItem(marker)
            self._anchor_overlay_items.append(marker)

            lbl = QGraphicsSimpleTextItem(label)
            lbl.setBrush(QColor("#111111"))
            fnt = lbl.font()
            psz = fnt.pointSizeF() if fnt.pointSizeF() > 0 else 9.0
            fnt.setPointSizeF(psz * 1.35)
            lbl.setFont(fnt)
            lbl.setToolTip(tip)
            lbl.setPos(x + 10.0, y - 22.0)
            lbl.setZValue(146.0)
            self.scene.addItem(lbl)
            self._anchor_overlay_items.append(lbl)

    def _draw_curve_overlay(self, chain_arr: np.ndarray, elev_arr: np.ndarray, color: str = "#bf00ff") -> None:
        # keep control-point markers; only refresh curve path here
        if self._curve_overlay_item is not None:
            try:
                sc = self._curve_overlay_item.scene()
                if sc is not None:
                    sc.removeItem(self._curve_overlay_item)
            except Exception:
                pass
            self._curve_overlay_item = None
        if self.scene is None:
            return
        ch = np.asarray(chain_arr, dtype=float)
        zz = np.asarray(elev_arr, dtype=float)
        m = np.isfinite(ch) & np.isfinite(zz)
        ch = ch[m]
        zz = zz[m]
        if ch.size < 2:
            return
        order = np.argsort(ch)
        ch = ch[order]
        zz = zz[order]

        path = QPainterPath()
        started = False
        for s, z in zip(ch, zz):
            pt = self._chain_elev_to_scene_xy(float(s), float(z))
            if pt is None:
                continue
            x, y = pt
            if not started:
                path.moveTo(x, y)
                started = True
            else:
                path.lineTo(x, y)
        if not started:
            return
        item = QGraphicsPathItem(path)
        pen = QPen(QColor(color))
        pen.setWidth(3)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setZValue(120.0)
        self.scene.addItem(item)
        self._curve_overlay_item = item
        self._refresh_anchor_overlay()

    def _draw_control_points_overlay(self, params: Optional[Dict[str, Any]] = None) -> None:
        self._clear_control_points_overlay()
        if self.scene is None:
            return
        p = params or self._collect_nurbs_params_from_ui()
        if not p:
            return
        cps = np.asarray(p.get("control_points", []), dtype=float)
        if cps.ndim != 2 or cps.shape[0] < 2:
            return

        for i, cp in enumerate(cps):
            pt = self._chain_elev_to_scene_xy(float(cp[0]), float(cp[1]))
            if pt is None:
                continue
            x, y = pt
            r = 10.0  # 2.5x from previous 4.0
            marker = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
            if i in (0, cps.shape[0] - 1):
                marker.setBrush(QColor("#ff4d4f"))
            else:
                marker.setBrush(QColor("#00a8ff"))
            marker.setPen(QPen(QColor("#ffffff")))
            marker.setZValue(130.0)
            self.scene.addItem(marker)
            self._cp_overlay_items.append(marker)

            lbl = QGraphicsSimpleTextItem(f"P{i}")
            lbl.setBrush(QColor("#111111"))
            fnt = lbl.font()
            psz = fnt.pointSizeF()
            if psz <= 0:
                psz = 10.0
            fnt.setPointSizeF(psz * 2.5)
            lbl.setFont(fnt)
            lbl.setPos(x + 20.0, y - 34.0)
            lbl.setZValue(131.0)
            self.scene.addItem(lbl)
            self._cp_overlay_items.append(lbl)
        self._refresh_anchor_overlay()

    @staticmethod
    def _profile_endpoints(prof: dict) -> Optional[Tuple[float, float, float, float]]:
        chain = np.asarray(prof.get("chain", []), dtype=float)
        elev = np.asarray(prof.get("elev_s", []), dtype=float)
        m = np.isfinite(chain) & np.isfinite(elev)
        chain = chain[m]
        elev = elev[m]
        if chain.size < 2:
            return None
        order = np.argsort(chain)
        chain = chain[order]
        elev = elev[order]
        return float(chain[0]), float(elev[0]), float(chain[-1]), float(elev[-1])

    @staticmethod
    def _grouped_vector_endpoints(prof: dict, groups: list) -> Optional[Tuple[float, float, float, float]]:
        chain = np.asarray(prof.get("chain", []), dtype=float)
        elev = np.asarray(prof.get("elev_s", []), dtype=float)
        finite = np.isfinite(chain) & np.isfinite(elev)
        if int(np.count_nonzero(finite)) < 2:
            return None
        chain = chain[finite]
        elev = elev[finite]
        if chain.size < 2:
            return None
        order = np.argsort(chain)
        chain = chain[order]
        elev = elev[order]

        grouped_mask = np.zeros(chain.shape, dtype=bool)
        for g in (groups or []):
            try:
                s = float(g.get("start", g.get("start_chainage", np.nan)))
                e = float(g.get("end", g.get("end_chainage", np.nan)))
            except Exception:
                continue
            if not (np.isfinite(s) and np.isfinite(e)):
                continue
            if e < s:
                s, e = e, s
            grouped_mask |= ((chain >= s) & (chain <= e))

        idx = np.flatnonzero(grouped_mask)
        if idx.size < 2:
            return None
        i0 = int(idx[0])
        i1 = int(idx[-1])
        return float(chain[i0]), float(elev[i0]), float(chain[i1]), float(elev[i1])

    def _nurbs_endpoint_targets(
        self,
        prof: dict,
        groups: Optional[list] = None,
        line_id: Optional[str] = None,
    ) -> Optional[Tuple[float, float, float, float]]:
        lid = line_id or self._line_id_current()
        curve_method = self._get_curve_method_for_line(lid)
        if curve_method == "nurbs":
            grouped = self._grouped_vector_endpoints(prof, groups or self._active_groups or [])
            if grouped is not None:
                return self._extend_endpoint_targets_with_cross_anchors(prof, grouped, line_id=lid)
        base = self._profile_endpoints(prof)
        return self._extend_endpoint_targets_with_cross_anchors(prof, base, line_id=lid)

    def _extend_endpoint_targets_with_cross_anchors(
        self,
        prof: dict,
        endpoints: Optional[Tuple[float, float, float, float]],
        line_id: Optional[str] = None,
    ) -> Optional[Tuple[float, float, float, float]]:
        if endpoints is None:
            return None
        lid = str(line_id or self._line_id_current())
        row_meta = self._line_row_meta()
        if str(row_meta.get("line_id", "")) != lid:
            # line_id passed here is save-path id (combo text-derived), not UI2 line_id. Use current role only.
            pass
        if self._current_ui2_line_role() != "cross":
            return endpoints
        anchors = self._anchors_for_cross_line(self._current_ui2_line_id(), require_ready=True)
        if not anchors:
            return endpoints
        s_vals = [float(a.get("s_on_cross")) for a in anchors if a.get("s_on_cross", None) is not None]
        s_vals = [s for s in s_vals if np.isfinite(s)]
        if not s_vals:
            return endpoints
        s0, z0, s1, z1 = map(float, endpoints)
        lo = min(s0, s1)
        hi = max(s0, s1)
        new_lo = min(lo, min(s_vals))
        new_hi = max(hi, max(s_vals))
        if not (new_hi > new_lo):
            return endpoints
        if abs(new_lo - lo) < 1e-9 and abs(new_hi - hi) < 1e-9:
            return endpoints
        chain = np.asarray(prof.get("chain", []), dtype=float)
        elev = np.asarray(prof.get("elev_s", []), dtype=float)
        m = np.isfinite(chain) & np.isfinite(elev)
        if int(np.count_nonzero(m)) < 2:
            return endpoints
        chain = chain[m]
        elev = elev[m]
        order = np.argsort(chain)
        chain = chain[order]
        elev = elev[order]
        z_lo = float(np.interp(new_lo, chain, elev))
        z_hi = float(np.interp(new_hi, chain, elev))
        return (new_lo, z_lo, new_hi, z_hi)

    def _constrain_curve_to_cross_anchors(self, curve: Optional[Dict[str, np.ndarray]]) -> Optional[Dict[str, np.ndarray]]:
        if not curve:
            return curve
        if self._current_ui2_line_role() != "cross":
            return curve
        anchors = self._anchors_for_cross_line(self._current_ui2_line_id(), require_ready=True)
        if len(anchors) < 3:
            return curve

        ch = np.asarray((curve or {}).get("chain", []), dtype=float)
        zz = np.asarray((curve or {}).get("elev", []), dtype=float)
        m = np.isfinite(ch) & np.isfinite(zz)
        ch = ch[m]
        zz = zz[m]
        if ch.size < 2:
            return curve
        order = np.argsort(ch)
        ch = ch[order]
        zz = zz[order]

        a_s = []
        a_z = []
        for a in anchors:
            try:
                s = float(a.get("s_on_cross"))
                z = float(a.get("z"))
            except Exception:
                continue
            if np.isfinite(s) and np.isfinite(z):
                a_s.append(s)
                a_z.append(z)
        if len(a_s) < 3:
            return curve
        a_s = np.asarray(a_s, dtype=float)
        a_z = np.asarray(a_z, dtype=float)
        o = np.argsort(a_s)
        a_s = a_s[o]
        a_z = a_z[o]

        # If anchors sit outside the current curve span (should be rare after endpoint extension),
        # expand sampled domain by inserting anchor chainages before applying correction.
        ch_aug = np.unique(np.concatenate([ch, a_s]))
        if ch_aug.size < 2:
            return curve
        base_zz = np.interp(ch_aug, ch, zz)

        base_at_anchor = np.interp(a_s, ch_aug, base_zz)
        residual = a_z - base_at_anchor
        node_x = np.concatenate([[float(ch_aug[0])], a_s, [float(ch_aug[-1])]])
        node_r = np.concatenate([[0.0], residual, [0.0]])
        # Monotonic x for interpolation (deduplicate if anchor hits endpoint exactly)
        keep = np.ones(node_x.shape, dtype=bool)
        for i in range(1, node_x.size):
            if not (node_x[i] > node_x[i - 1]):
                keep[i] = False
        node_x = node_x[keep]
        node_r = node_r[keep]
        if node_x.size < 2:
            return {"chain": ch_aug, "elev": base_zz}

        corr = np.interp(ch_aug, node_x, node_r)
        zz_adj = base_zz + corr
        for s, z in zip(a_s, a_z):
            hit = np.isclose(ch_aug, s, rtol=0.0, atol=1e-9)
            if np.any(hit):
                zz_adj[hit] = z

        return {"chain": ch_aug, "elev": zz_adj}

    def _clamp_curve_below_ground(
        self,
        curve: Optional[Dict[str, np.ndarray]],
        *,
        prof: Optional[dict] = None,
        clearance: float = 0.3,
        keep_endpoints: bool = True,
    ) -> Optional[Dict[str, np.ndarray]]:
        if not curve:
            return curve
        pf = prof if isinstance(prof, dict) else (self._active_prof or {})
        ch = np.asarray((curve or {}).get("chain", []), dtype=float)
        zz = np.asarray((curve or {}).get("elev", []), dtype=float)
        m = np.isfinite(ch) & np.isfinite(zz)
        ch = ch[m]
        zz = zz[m]
        if ch.size < 2:
            return curve
        order = np.argsort(ch)
        ch = ch[order]
        zz = zz[order]

        gch = np.asarray((pf or {}).get("chain", []), dtype=float)
        gz = np.asarray((pf or {}).get("elev_s", []), dtype=float)
        mg = np.isfinite(gch) & np.isfinite(gz)
        gch = gch[mg]
        gz = gz[mg]
        if gch.size < 2:
            return {"chain": ch, "elev": zz}
        go = np.argsort(gch)
        gch = gch[go]
        gz = gz[go]

        try:
            clear_m = float(clearance)
        except Exception:
            clear_m = 0.3
        clear_m = max(0.0, clear_m)
        g_at = np.interp(ch, gch, gz)
        zz2 = zz.copy()
        if keep_endpoints and zz2.size >= 3:
            zz2[1:-1] = np.minimum(zz2[1:-1], g_at[1:-1] - clear_m)
        else:
            zz2[:] = np.minimum(zz2, g_at - clear_m)
        return {"chain": ch, "elev": zz2}

    @staticmethod
    def _normalize_group_spans(groups: list) -> List[Tuple[float, float]]:
        spans: List[Tuple[float, float]] = []
        for g in (groups or []):
            try:
                s = float(g.get("start", g.get("start_chainage", np.nan)))
                e = float(g.get("end", g.get("end_chainage", np.nan)))
            except Exception:
                continue
            if not (np.isfinite(s) and np.isfinite(e)):
                continue
            if e < s:
                s, e = e, s
            spans.append((float(s), float(e)))
        spans.sort(key=lambda x: (x[0], x[1]))
        return spans

    def _build_default_nurbs_chainage(self, s0: float, s1: float, groups: list) -> List[float]:
        spans = self._normalize_group_spans(groups)
        n_ctrl = max(2, len(spans) + 1)
        if not spans:
            return np.linspace(float(s0), float(s1), n_ctrl).tolist()

        inner: List[float] = []
        for i in range(max(0, len(spans) - 1)):
            b = float(spans[i][1])  # shared boundary: end(i) == start(i+1)
            b = max(float(s0), min(float(s1), b))
            inner.append(b)

        cp_chain = [float(s0)] + inner + [float(s1)]
        if len(cp_chain) != n_ctrl:
            cp_chain = np.linspace(float(s0), float(s1), n_ctrl).tolist()
        return cp_chain

    def _build_default_nurbs_params(self, line_id: str, prof: dict, groups: list, base_curve: dict) -> Dict[str, Any]:
        ends = self._nurbs_endpoint_targets(prof, groups, line_id=line_id)
        if ends is None:
            return {"degree": 1, "control_points": [], "weights": []}
        s0, z0, s1, z1 = ends

        # Default count rule: number of groups + 1.
        # CP0/CP_last are NURBS endpoints; interior CPs sit on group boundaries.
        cp_chain = self._build_default_nurbs_chainage(float(s0), float(s1), groups)

        xb = np.asarray((base_curve or {}).get("chain", []), dtype=float)
        zb = np.asarray((base_curve or {}).get("elev", []), dtype=float)
        mb = np.isfinite(xb) & np.isfinite(zb)
        xb = xb[mb]
        zb = zb[mb]
        if xb.size >= 2:
            cp_elev = np.interp(np.asarray(cp_chain, dtype=float), xb, zb).tolist()
        else:
            chain = np.asarray(prof.get("chain", []), dtype=float)
            elev = np.asarray(prof.get("elev_s", []), dtype=float)
            m = np.isfinite(chain) & np.isfinite(elev)
            chain = chain[m]
            elev = elev[m]
            if chain.size >= 2:
                cp_elev = np.interp(np.asarray(cp_chain, dtype=float), chain, elev).tolist()
            else:
                cp_elev = np.linspace(z0, z1, len(cp_chain)).tolist()

        # Keep default interior CP always below ground profile.
        gch = np.asarray(prof.get("chain", []), dtype=float)
        gz = np.asarray(prof.get("elev_s", []), dtype=float)
        mg = np.isfinite(gch) & np.isfinite(gz)
        gch = gch[mg]
        gz = gz[mg]
        clearance = 0.35
        if gch.size >= 2 and len(cp_elev) >= 3:
            g_at_cp = np.interp(np.asarray(cp_chain, dtype=float), gch, gz)
            cp_elev_arr = np.asarray(cp_elev, dtype=float)
            cp_elev_arr[1:-1] = np.minimum(cp_elev_arr[1:-1], g_at_cp[1:-1] - clearance)
            cp_elev = cp_elev_arr.tolist()

        # Endpoint lock to first/last grouped vectors (fallback: profile endpoints).
        cp_elev[0] = z0
        cp_elev[-1] = z1
        cps = [[float(s), float(z)] for s, z in zip(cp_chain, cp_elev)]
        w = [1.0] * len(cps)
        deg = min(3, max(1, len(cps) - 1))
        params = {"degree": int(deg), "control_points": cps, "weights": w}
        self._nurbs_params_by_line[line_id] = params
        return params

    def _get_nurbs_params_for_line(self, line_id: str) -> Optional[Dict[str, Any]]:
        return self._nurbs_params_by_line.get(line_id)

    def _set_nurbs_params_for_line(self, line_id: str, params: Dict[str, Any]) -> None:
        self._nurbs_params_by_line[line_id] = params

    def _try_load_nurbs_params_file(self, line_id: str) -> Optional[Dict[str, Any]]:
        path = self._nurbs_json_path_for(line_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                js = json.load(f) or {}
            params = {
                "degree": int(js.get("degree", 3)),
                "control_points": js.get("control_points", []),
                "weights": js.get("weights", []),
            }
            cps = np.asarray(params.get("control_points", []), dtype=float)
            ws = np.asarray(params.get("weights", []), dtype=float)
            if cps.ndim != 2 or cps.shape[0] < 2:
                return None
            if ws.ndim != 1 or ws.size != cps.shape[0]:
                params["weights"] = np.ones(cps.shape[0], dtype=float).tolist()
            return params
        except Exception:
            return None

    def _sync_nurbs_panel_for_current_line(self, reset_defaults: bool = False) -> None:
        if self.line_combo is None or self.line_combo.count() == 0:
            return
        line_id = self._line_id_current()
        prof = self._active_prof
        if not prof:
            return
        groups = self._active_groups or []
        base = self._active_base_curve or {}

        params = None if reset_defaults else self._get_nurbs_params_for_line(line_id)
        if (not params) and (not reset_defaults):
            params = self._try_load_nurbs_params_file(line_id)
        if not params:
            params = self._build_default_nurbs_params(line_id, prof, groups, base)

        cps = params.get("control_points", []) or []
        ww = params.get("weights", []) or []
        deg = int(params.get("degree", 3))
        n_ctrl = max(2, len(cps))
        deg = max(1, min(deg, n_ctrl - 1))
        params["degree"] = deg
        if len(ww) != n_ctrl:
            ww = [1.0] * n_ctrl
            params["weights"] = ww
        self._set_nurbs_params_for_line(line_id, params)

        self._nurbs_updating_ui = True
        try:
            self.nurbs_cp_spin.setValue(n_ctrl)
            self.nurbs_deg_spin.setMaximum(max(1, n_ctrl - 1))
            self.nurbs_deg_spin.setValue(deg)
            self._populate_nurbs_table(params)
        finally:
            self._nurbs_updating_ui = False
        self._draw_control_points_overlay(params)

    def _populate_nurbs_table(self, params: Dict[str, Any]) -> None:
        cps = params.get("control_points", []) or []
        ww = params.get("weights", []) or []
        self.nurbs_table.setRowCount(0)
        for i, cp in enumerate(cps):
            r = self.nurbs_table.rowCount()
            self.nurbs_table.insertRow(r)
            item = QTableWidgetItem(f"P{i}")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.nurbs_table.setItem(r, 0, item)

            sbox = KeyboardOnlyDoubleSpinBox()
            sbox.setDecimals(3)
            sbox.setRange(-1e6, 1e6)
            sbox.setSingleStep(0.1)
            sbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
            sbox.setValue(float(cp[0]))
            sbox.valueChanged.connect(lambda _v, _r=r: self._on_nurbs_table_changed(_r))
            self.nurbs_table.setCellWidget(r, 1, sbox)

            zbox = KeyboardOnlyDoubleSpinBox()
            zbox.setDecimals(3)
            zbox.setRange(-1e6, 1e6)
            zbox.setSingleStep(0.1)
            zbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
            zbox.setValue(float(cp[1]))
            zbox.valueChanged.connect(lambda _v, _r=r: self._on_nurbs_table_changed(_r))
            self.nurbs_table.setCellWidget(r, 2, zbox)

            wbox = KeyboardOnlyDoubleSpinBox()
            wbox.setDecimals(3)
            wbox.setRange(0.001, 1e6)
            wbox.setSingleStep(0.1)
            wbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
            wbox.setValue(float(ww[i] if i < len(ww) else 1.0))
            wbox.valueChanged.connect(lambda _v, _r=r: self._on_nurbs_table_changed(_r))
            self.nurbs_table.setCellWidget(r, 3, wbox)

        self._enforce_nurbs_endpoint_lock()

    def _collect_nurbs_params_from_ui(self) -> Optional[Dict[str, Any]]:
        rc = self.nurbs_table.rowCount()
        if rc < 2:
            return None
        cps = []
        ws = []
        for r in range(rc):
            sbox = self.nurbs_table.cellWidget(r, 1)
            zbox = self.nurbs_table.cellWidget(r, 2)
            wbox = self.nurbs_table.cellWidget(r, 3)
            if not isinstance(sbox, QDoubleSpinBox) or not isinstance(zbox, QDoubleSpinBox) or not isinstance(wbox, QDoubleSpinBox):
                return None
            cps.append([float(sbox.value()), float(zbox.value())])
            ws.append(float(max(0.001, wbox.value())))
        cps_arr = np.asarray(cps, dtype=float)
        order = np.argsort(cps_arr[:, 0])
        cps_arr = cps_arr[order]
        ws_arr = np.asarray(ws, dtype=float)[order]
        deg = int(self.nurbs_deg_spin.value())
        deg = max(1, min(deg, len(cps) - 1))
        return {
            "degree": deg,
            "control_points": cps_arr.tolist(),
            "weights": ws_arr.tolist(),
        }

    def _enforce_nurbs_endpoint_lock(self) -> None:
        prof = self._active_prof
        if not prof:
            return
        ends = self._nurbs_endpoint_targets(prof, self._active_groups, line_id=self._line_id_current())
        if ends is None:
            return
        s0, z0, s1, z1 = ends
        rc = self.nurbs_table.rowCount()
        if rc < 2:
            return
        for row, s_val, z_val in ((0, s0, z0), (rc - 1, s1, z1)):
            sbox = self.nurbs_table.cellWidget(row, 1)
            zbox = self.nurbs_table.cellWidget(row, 2)
            if isinstance(sbox, QDoubleSpinBox):
                sbox.blockSignals(True)
                sbox.setValue(float(s_val))
                sbox.setEnabled(False)
                sbox.blockSignals(False)
            if isinstance(zbox, QDoubleSpinBox):
                zbox.blockSignals(True)
                zbox.setValue(float(z_val))
                zbox.setEnabled(False)
                zbox.blockSignals(False)

    def _resize_nurbs_control_points(self, new_count: int) -> None:
        line_id = self._line_id_current()
        params = self._get_nurbs_params_for_line(line_id)
        prof = self._active_prof
        if params is None or prof is None:
            return
        ends = self._nurbs_endpoint_targets(prof, self._active_groups, line_id=line_id)
        if ends is None:
            return
        s0, z0, s1, z1 = ends
        cps = np.asarray(params.get("control_points", []), dtype=float)
        ws = np.asarray(params.get("weights", []), dtype=float)
        if cps.ndim != 2 or cps.shape[0] < 2:
            params = self._build_default_nurbs_params(line_id, prof, self._active_groups, self._active_base_curve or {})
            cps = np.asarray(params.get("control_points", []), dtype=float)
            ws = np.asarray(params.get("weights", []), dtype=float)
        old_s = cps[:, 0]
        old_z = cps[:, 1]
        new_s = np.linspace(s0, s1, int(max(2, new_count)))
        new_z = np.interp(new_s, old_s, old_z)
        new_w = np.interp(new_s, old_s, ws if ws.size == old_s.size else np.ones_like(old_s))
        new_z[0] = z0
        new_z[-1] = z1
        new_params = {
            "degree": int(min(int(params.get("degree", 3)), len(new_s) - 1)),
            "control_points": np.vstack([new_s, new_z]).T.tolist(),
            "weights": np.where(np.isfinite(new_w) & (new_w > 0), new_w, 1.0).tolist(),
        }
        self._set_nurbs_params_for_line(line_id, new_params)
        self._sync_nurbs_panel_for_current_line(reset_defaults=False)
        self._schedule_nurbs_live_update()

    def _schedule_nurbs_live_update(self) -> None:
        if self._nurbs_updating_ui:
            return
        self._nurbs_live_timer.start()

    def _on_nurbs_live_tick(self) -> None:
        line_id = self._line_id_current()
        params = self._collect_nurbs_params_from_ui()
        if not params:
            return
        # If current background already contains a baked curve, refresh base image once
        # before drawing live overlay to avoid showing two slip-curves at the same time.
        if self._static_nurbs_bg_loaded:
            self._render_current_safe()
            self._static_nurbs_bg_loaded = False
        self._set_nurbs_params_for_line(line_id, params)
        self._draw_control_points_overlay(params)
        curve_method = self._get_curve_method_for_line(line_id)
        if curve_method != "nurbs":
            return
        curve = self._compute_nurbs_curve_from_params(params)
        if curve is None:
            return
        self._active_curve = curve
        self._draw_curve_overlay(np.asarray(curve["chain"], dtype=float), np.asarray(curve["elev"], dtype=float))

    def _compute_nurbs_curve_from_params(self, params: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        prof = self._active_prof
        if not prof:
            return None
        p = dict(params or {})
        cps = np.asarray(p.get("control_points", []), dtype=float)
        ww = np.asarray(p.get("weights", []), dtype=float)
        if cps.ndim != 2 or cps.shape[0] < 2:
            return None
        if ww.ndim != 1 or ww.size != cps.shape[0]:
            ww = np.ones(cps.shape[0], dtype=float)

        ends = self._nurbs_endpoint_targets(prof, self._active_groups, line_id=self._line_id_current())
        if ends is None:
            return None
        s0, z0, s1, z1 = ends
        cps = cps.copy()
        cps[0, 0], cps[0, 1] = s0, z0
        cps[-1, 0], cps[-1, 1] = s1, z1

        deg = int(max(1, min(int(p.get("degree", 3)), cps.shape[0] - 1)))
        ch = np.asarray(prof.get("chain", []), dtype=float)
        n_samples = int(max(120, np.count_nonzero(np.isfinite(ch)) * 2))
        out = evaluate_nurbs_curve(
            chain_ctrl=cps[:, 0],
            elev_ctrl=cps[:, 1],
            weights=ww,
            degree=deg,
            n_samples=n_samples,
        )
        sx = np.asarray(out.get("chain", []), dtype=float)
        sz = np.asarray(out.get("elev", []), dtype=float)
        m = np.isfinite(sx) & np.isfinite(sz)
        sx = sx[m]
        sz = sz[m]
        # Keep rendered NURBS strictly inside locked endpoint span.
        m_span = (sx >= min(s0, s1)) & (sx <= max(s0, s1))
        sx = sx[m_span]
        sz = sz[m_span]
        if sx.size < 2:
            return None
        out = {"chain": sx, "elev": sz}
        out = self._constrain_curve_to_cross_anchors(out)
        out = self._clamp_curve_below_ground(out, prof=prof, clearance=0.3, keep_endpoints=True)
        return out

    def _on_nurbs_cp_spin_changed(self, val: int) -> None:
        if self._nurbs_updating_ui:
            return
        self.nurbs_deg_spin.setMaximum(max(1, int(val) - 1))
        if self.nurbs_deg_spin.value() > self.nurbs_deg_spin.maximum():
            self.nurbs_deg_spin.setValue(self.nurbs_deg_spin.maximum())
        self._resize_nurbs_control_points(int(val))

    def _on_nurbs_deg_spin_changed(self, val: int) -> None:
        if self._nurbs_updating_ui:
            return
        max_deg = max(1, self.nurbs_cp_spin.value() - 1)
        if val > max_deg:
            self._nurbs_updating_ui = True
            try:
                self.nurbs_deg_spin.setValue(max_deg)
            finally:
                self._nurbs_updating_ui = False
        self._schedule_nurbs_live_update()

    def _on_nurbs_table_changed(self, _row: int) -> None:
        self._enforce_nurbs_endpoint_lock()
        self._schedule_nurbs_live_update()

    def _on_nurbs_reset_defaults(self) -> None:
        if not self._active_prof:
            self._warn("[UI3] Render/Draw curve first to initialize NURBS.")
            return
        self._sync_nurbs_panel_for_current_line(reset_defaults=True)
        self._schedule_nurbs_live_update()

    def _nurbs_png_path_for(self, line_id: str) -> str:
        return os.path.join(self._curve_dir(), f"profile_{line_id}_nurbs.png")

    def _nurbs_json_path_for(self, line_id: str) -> str:
        return os.path.join(self._preview_dir(), f"profile_{line_id}_nurbs.json")

    def _ground_json_path_for(self, line_id: str) -> str:
        src = self._current_profile_source_key()
        suffix = "smooth" if src == "smooth" else "raw"
        return os.path.join(self._curve_dir(), f"ground_{line_id}_{suffix}.json")

    def _on_nurbs_save(self) -> None:
        if not self._active_prof:
            self._warn("[UI3] No active profile to save NURBS.")
            return
        line_id = self._line_id_current()
        params = self._collect_nurbs_params_from_ui()
        if not params:
            self._warn("[UI3] Invalid NURBS parameters.")
            return
        curve = self._compute_nurbs_curve_from_params(params)
        if curve is None:
            self._warn("[UI3] Cannot evaluate NURBS curve.")
            return
        out_png = self._nurbs_png_path_for(line_id)
        msg, path = render_profile_png(
            self._active_prof, out_png,
            y_min=None, y_max=None,
            x_min=None, x_max=None,
            vec_scale=self.vscale.value(),
            vec_width=self.vwidth.value(),
            head_len=6.0, head_w=4.0,
            highlight_theta=None,
            group_ranges=self._active_groups if self._active_groups else None,
            ungrouped_color=self._get_ungrouped_color(),
            overlay_curves=[(curve["chain"], curve["elev"], "#bf00ff", "Slip curve")],
            curvature_rdp_eps_m=self._current_rdp_eps_m(),
            curvature_smooth_radius_m=self._current_smooth_radius_m(),
        )
        self._log(msg)
        if path and os.path.exists(path):
            payload = {
                "line_id": line_id,
                "curve_method": "nurbs",
                "degree": int(params.get("degree", 3)),
                "control_points": params.get("control_points", []),
                "weights": params.get("weights", []),
                "curve": {
                    "chain": np.asarray(curve["chain"], dtype=float).tolist(),
                    "elev": np.asarray(curve["elev"], dtype=float).tolist(),
                },
            }
            jpath = self._nurbs_json_path_for(line_id)
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            curve_dir = self._curve_dir()

            # 1) nurbs_Line_n__(n_m).json -> x,y,z of slip curve
            curve_chain = np.asarray(curve.get("chain", []), dtype=float)
            curve_elev = np.asarray(curve.get("elev", []), dtype=float)
            m_curve = np.isfinite(curve_chain) & np.isfinite(curve_elev)
            curve_chain = curve_chain[m_curve]
            curve_elev = curve_elev[m_curve]

            x_curve = np.full(curve_chain.shape, np.nan, dtype=float)
            y_curve = np.full(curve_chain.shape, np.nan, dtype=float)
            prof_chain = np.asarray(self._active_prof.get("chain", []), dtype=float)
            prof_x = np.asarray(self._active_prof.get("x", []), dtype=float)
            prof_y = np.asarray(self._active_prof.get("y", []), dtype=float)
            if prof_chain.size == prof_x.size and prof_chain.size == prof_y.size:
                m_xy = np.isfinite(prof_chain) & np.isfinite(prof_x) & np.isfinite(prof_y)
                if int(np.count_nonzero(m_xy)) >= 2 and curve_chain.size > 0:
                    ch = prof_chain[m_xy]
                    xx = prof_x[m_xy]
                    yy = prof_y[m_xy]
                    order = np.argsort(ch)
                    ch = ch[order]
                    xx = xx[order]
                    yy = yy[order]
                    ch_u, uniq_idx = np.unique(ch, return_index=True)
                    xx_u = xx[uniq_idx]
                    yy_u = yy[uniq_idx]
                    if ch_u.size >= 2:
                        x_curve = np.interp(curve_chain, ch_u, xx_u)
                        y_curve = np.interp(curve_chain, ch_u, yy_u)

            curve_rows = []
            for i, (s, x, y, z) in enumerate(zip(curve_chain, x_curve, y_curve, curve_elev)):
                curve_rows.append({
                    "index": int(i),
                    "chainage_m": float(s),
                    "x": (float(x) if np.isfinite(x) else None),
                    "y": (float(y) if np.isfinite(y) else None),
                    "z": float(z),
                })
            nurbs_curve_payload = {
                "line_id": line_id,
                "curve_method": "nurbs",
                "chainage_origin": self._ui3_chainage_origin(),
                "count": int(len(curve_rows)),
                "points": curve_rows,
            }
            curve_json = os.path.join(curve_dir, f"nurbs_{line_id}.json")
            with open(curve_json, "w", encoding="utf-8") as f:
                json.dump(nurbs_curve_payload, f, ensure_ascii=False, indent=2)

            # 2) group_Line_n__(n_m).json -> group ID, start, end
            table_groups = self._read_groups_from_table()
            group_rows = []
            for g in (table_groups or []):
                group_rows.append({
                    "group_id": str(g.get("id", "")),
                    "start": float(g.get("start")),
                    "end": float(g.get("end")),
                })
            group_payload = {
                "line_id": line_id,
                "count": int(len(group_rows)),
                "groups": group_rows,
                "chainage_origin": self._ui3_chainage_origin(),
                "rdp_eps_m": self._current_rdp_eps_m(),
                "smooth_radius_m": self._current_smooth_radius_m(),
                "include_curvature_threshold": self._include_curvature_threshold(),
                "include_vector_angle_zero": self._include_vector_angle_zero(),
            }
            group_json = os.path.join(curve_dir, f"group_{line_id}.json")
            with open(group_json, "w", encoding="utf-8") as f:
                json.dump(group_payload, f, ensure_ascii=False, indent=2)

            # 2b) ui3/groups/Line_n__(n_m).json -> canonical groups file for UI3
            groups_ui3_json = self._groups_json_path_for(line_id)
            groups_for_json = self._groups_with_median_theta(table_groups, self._active_prof)
            group_method = None
            if os.path.exists(groups_ui3_json):
                try:
                    with open(groups_ui3_json, "r", encoding="utf-8") as f:
                        old_js = json.load(f) or {}
                    if old_js.get("group_method", None):
                        group_method = str(old_js.get("group_method"))
                except Exception:
                    pass
            groups_ui3_payload = {
                "line": self.line_combo.currentText(),
                "groups": groups_for_json,
                "chainage_origin": self._ui3_chainage_origin(),
                "curve_method": self._get_curve_method_for_line(line_id),
                "rdp_eps_m": self._current_rdp_eps_m(),
                "smooth_radius_m": self._current_smooth_radius_m(),
                "include_curvature_threshold": self._include_curvature_threshold(),
                "include_vector_angle_zero": self._include_vector_angle_zero(),
            }
            if group_method:
                groups_ui3_payload["group_method"] = group_method
            with open(groups_ui3_json, "w", encoding="utf-8") as f:
                json.dump(groups_ui3_payload, f, ensure_ascii=False, indent=2)

            # 3) nurbs_info_Line_n__(n_m).json -> NURBS panel info
            cps = np.asarray(params.get("control_points", []), dtype=float)
            ww = np.asarray(params.get("weights", []), dtype=float)
            if ww.ndim != 1 or ww.size != cps.shape[0]:
                ww = np.ones(cps.shape[0], dtype=float)
            cp_rows = []
            for i in range(cps.shape[0]):
                cp_rows.append({
                    "cp_index": int(i),
                    "chainage_m": float(cps[i, 0]),
                    "elev_m": float(cps[i, 1]),
                    "weight": float(ww[i]),
                })
            nurbs_info_payload = {
                "line_id": line_id,
                "chainage_origin": self._ui3_chainage_origin(),
                "control_points_count": int(cps.shape[0]),
                "degree": int(params.get("degree", 3)),
                "control_points": cp_rows,
            }
            nurbs_info_json = os.path.join(curve_dir, f"nurbs_info_{line_id}.json")
            with open(nurbs_info_json, "w", encoding="utf-8") as f:
                json.dump(nurbs_info_payload, f, ensure_ascii=False, indent=2)

            self._ok(f"[UI3] Saved NURBS: {path}")
            self._log(f"[UI3] Saved NURBS params: {jpath}")
            self._log(f"[UI3] Saved NURBS curve: {curve_json}")
            self._log(f"[UI3] Saved group table: {group_json}")
            self._log(f"[UI3] Saved groups JSON: {groups_ui3_json}")
            self._log(f"[UI3] Saved NURBS info: {nurbs_info_json}")
            try:
                anchor_path, n_upd = self._update_anchors_xyz_for_saved_main_curve(curve)
                if anchor_path and n_upd > 0:
                    self._log(f"[UI3] Updated anchors_xyz: {anchor_path} (n={n_upd})")
            except Exception as e:
                self._warn(f"[UI3] Cannot update anchors_xyz: {e}")
            self._refresh_anchor_overlay()
            try:
                self.curve_saved.emit(curve_json)
            except Exception:
                pass

    def _on_auto_group(self) -> None:
        """Sinh group tự động như UI3, lưu JSON, cập nhật bảng, và vẽ guide (không re-render)."""
        try:
            try:
                self._nurbs_live_timer.stop()
            except Exception:
                pass
            # 1) Dữ liệu tuyến
            if not hasattr(self, "_gdf") or self._gdf is None or self._gdf.empty:
                self._warn("[UI3] No lines loaded from UI2.")
                return
            row = self.line_combo.currentIndex()
            if row < 0:
                self._warn("[UI3] No line selected.")
                return
            geom = self._gdf.geometry.iloc[row]

            # 2) Profile full line; curvature boundaries will be computed on the
            # full section and filtered back to the mask span.
            prof = self._compute_profile_for_geom(geom, slip_only=False)
            if not prof:
                self._err("[UI3] Empty profile.")
                return

            # 3) Auto-group from raw DEM curvature on slip span
            include_curvature = self._include_curvature_threshold()
            include_vector_zero = self._include_vector_angle_zero()
            parts = [f"{self._current_profile_source_key()} DEM", "RDP"]
            if include_curvature:
                parts.append("curvature")
            if include_vector_zero:
                parts.append("vector=0")
            self._log(
                "[UI3] Auto Group method: " + " + ".join(parts)
            )
            groups = auto_group_profile_by_criteria(prof, **self._grouping_params_current())
            if include_curvature and include_vector_zero:
                group_method = "profile_dem_rdp_curvature_theta0"
            elif include_curvature:
                group_method = "profile_dem_rdp_curvature_only"
            elif include_vector_zero:
                group_method = "profile_dem_rdp_theta0_only"
            else:
                group_method = "profile_dem_rdp_span_only"

            groups = clamp_groups_to_slip(prof, groups, min_len=WORKFLOW_GROUP_MIN_LEN_M)
            if not groups:
                self._warn("[UI3] Auto grouping produced no segments within slip zone.")
                return

            line_id = self._line_id_current()
            self._save_groups_to_ui(
                groups,
                prof,
                line_id,
                log_text=f"[UI3] Auto Group done for '{line_id}': {len(groups)} groups.",
                curve_method=self._curve_method_from_group_method(group_method),
                group_method=group_method,
            )
            # Re-render ngay để vẽ vector và tô màu theo group vừa tạo.
            self._render_current_safe()
            try:
                self._nurbs_live_timer.stop()
            except Exception:
                pass
        except Exception as e:
            self._err(f"[UI3] Auto Group error: {e}")

    def _on_draw_curve(self) -> None:
        """Tính và vẽ đường cong (overlay) vào PNG preview hiện tại."""
        try:
            line_id = self._line_id_current()
            _prev_curve_method = self._get_curve_method_for_line(line_id)
            curve_method = self._set_curve_method_for_line(line_id, "nurbs")
            if _prev_curve_method != "nurbs":
                self._log(f"[UI3] Curve method for '{line_id}': forced to NURBS (Bezier-like seed)")
            else:
                self._log(f"[UI3] Curve method for '{line_id}': NURBS (Bezier-like seed)")

            # 1) Lấy line và profile TRONG slip-zone
            if not hasattr(self, "_gdf") or self._gdf is None or self._gdf.empty:
                self._warn("[UI3] No lines.");
                return
            row = self.line_combo.currentIndex()
            if row < 0:
                self._warn("[UI3] Select a line first.");
                return

            geom = self._gdf.geometry.iloc[row]
            prof = self._compute_profile_for_geom(geom, slip_only=False)
            if not prof or len(prof.get("chain", [])) < 6:
                self._warn("[UI3] Empty/too-short slip profile.");
                return

            # 2) Lấy groups (ưu tiên bảng → file), rồi clamp vào slip-zone
            groups = self._load_groups_for_current_line()
            if not groups:
                try:
                    line_id = self._line_id_current()
                    gpath = self._groups_json_path_for(line_id)
                    # hoặc:
                    # gpath = self._groups_json_path()
                    if os.path.exists(gpath):
                        with open(gpath, "r", encoding="utf-8") as f:
                            groups = (json.load(f) or {}).get("groups", []) or []
                except Exception:
                    groups = []
            if not groups:
                groups = auto_group_profile_by_criteria(prof, **self._grouping_params_current())
                groups = clamp_groups_to_slip(prof, groups, min_len=WORKFLOW_GROUP_MIN_LEN_M)
                if not groups:
                    self._warn("[UI3] Auto grouping produced no segments within slip zone.");
                    return
                self._save_groups_to_ui(
                    groups,
                    prof,
                    line_id,
                    log_text=f"[UI3] Auto Group (implicit) for '{line_id}': {len(groups)} groups.",
                    curve_method=curve_method,
                )
            else:
                groups = clamp_groups_to_slip(prof, groups, min_len=WORKFLOW_GROUP_MIN_LEN_M)
                if not groups:
                    self._warn("[UI3] No groups within slip zone.");
                    return

            # 3) Base curve để lấy mục tiêu mặc định cho NURBS/Bezier
            base = estimate_slip_curve(
                prof, groups,
                ds=0.2, smooth_factor=0.06,
                depth_gain=14, min_depth=5
            )
            x_base = np.asarray(base.get("chain", []), dtype=float)
            z_base = np.asarray(base.get("elev", []), dtype=float)

            # Lọc NaN / inf
            mask = np.isfinite(x_base) & np.isfinite(z_base)
            x_base = x_base[mask]
            z_base = z_base[mask]

            if x_base.size < 2:
                self._warn("[UI3] Slip curve has too few valid points.")
                return

            # Sắp xếp theo chainage tăng dần cho chắc
            order = np.argsort(x_base)
            x_base = x_base[order]
            z_base = z_base[order]

            self._log(
                f"[UI3] Slip curve pts={x_base.size}, "
                f"chain=[{x_base.min():.2f}, {x_base.max():.2f}]"
            )

            # 4) Fit curve theo method
            curve = {"chain": x_base, "elev": z_base}  # fallback mặc định (base target)

            def _fit_bezier_curve() -> Optional[dict]:
                try:
                    bez = fit_bezier_smooth_curve(
                        chain=np.asarray(prof["chain"], dtype=float),
                        elevg=np.asarray(prof["elev_s"], dtype=float),
                        target_s=x_base,
                        target_z=z_base,
                        c0=0.20,
                        c1=0.40,
                        clearance=0.35,
                    )

                    xb = np.asarray(bez.get("chain", []), dtype=float)
                    zb = np.asarray(bez.get("elev", []), dtype=float)
                    m2 = np.isfinite(xb) & np.isfinite(zb)
                    xb = xb[m2]
                    zb = zb[m2]
                    if xb.size >= 2:
                        self._log(f"[UI3] Bezier-like seed curve OK: n={xb.size}")
                        return {"chain": xb, "elev": zb}
                except Exception as e:
                    self._warn(f"[UI3] Bezier-like seed fit failed, using base target. ({e})")
                return None

            bez_curve = _fit_bezier_curve()
            if bez_curve is not None:
                curve = bez_curve
            else:
                self._warn("[UI3] Bezier-like seed has too few points; using base target.")

            # Seed NURBS defaults from Bezier-like curve, but keep user's existing NURBS if already present.
            self._active_prof = prof
            self._active_groups = groups
            self._active_base_curve = {
                "chain": np.asarray(curve["chain"], dtype=float),
                "elev": np.asarray(curve["elev"], dtype=float),
            }
            self._sync_nurbs_panel_for_current_line(reset_defaults=False)

            def _eval_current_nurbs() -> Optional[dict]:
                params_now = self._collect_nurbs_params_from_ui()
                if not params_now:
                    return None
                return self._compute_nurbs_curve_from_params(params_now)

            nurbs_curve = _eval_current_nurbs()
            if nurbs_curve is None:
                # If current UI params are invalid (or stale), regenerate NURBS defaults from Bezier-like seed once.
                self._warn("[UI3] Current NURBS params invalid/unusable; reset to Bezier-like NURBS seed.")
                self._sync_nurbs_panel_for_current_line(reset_defaults=True)
                nurbs_curve = _eval_current_nurbs()

            if nurbs_curve is not None:
                curve = nurbs_curve
                self._log(f"[UI3] NURBS slip curve OK: n={len(nurbs_curve['chain'])}")
            else:
                # Last-resort fallback for display only; keep UI responsive.
                self._warn("[UI3] NURBS fit failed after reseed; showing Bezier-like seed curve.")
            curve = self._clamp_curve_below_ground(curve, prof=prof, clearance=0.3, keep_endpoints=True) or curve

            # 5) Re-render base PNG (no curve baked-in), then draw overlay in scene
            out_png = self._profile_png_path_for(line_id)

            msg, path = render_profile_png(
                prof, out_png,
                y_min=None, y_max=None,
                x_min=None, x_max=None,
                vec_scale=self.vscale.value(),
                vec_width=self.vwidth.value(),
                head_len=6.0, head_w=4.0,
                highlight_theta=None,
                group_ranges=groups,
                ungrouped_color=self._get_ungrouped_color(),
                curvature_rdp_eps_m=self._current_rdp_eps_m(),
                curvature_smooth_radius_m=self._current_smooth_radius_m(),
            )
            self._log(msg)
            if not path or not os.path.exists(path):
                return

            # 6) Nạp PNG mới (clear cache để tránh ảnh cũ)
            from PyQt5.QtGui import QPixmap, QPixmapCache
            QPixmapCache.clear()
            self.scene.clear()
            item = QGraphicsPixmapItem(QPixmap(path))
            self.scene.addItem(item)
            self._img_ground = item
            self._img_rate0 = item
            self._load_axes_meta(path)
            self._static_nurbs_bg_loaded = False

            self._active_curve = {
                "chain": np.asarray(curve["chain"], dtype=float),
                "elev": np.asarray(curve["elev"], dtype=float),
            }
            self._draw_curve_overlay(self._active_curve["chain"], self._active_curve["elev"])
            # scene.clear() above removes CP markers; redraw them immediately after Draw Curve.
            self._draw_control_points_overlay()

            if getattr(self, "_first_show", True):
                self.view.fit_to_scene()
                self._first_show = False

            self._ok("[UI3] Curve drawn on current section.")
        except Exception as e:
            self._err(f"[UI3] Draw Curve error: {e}")
            raise

    def _apply_button_style(self) -> None:
        """Áp style tab Curve Analyze (nền trắng + nút xanh, gồm cả QToolButton)."""
        style = """
        QWidget {
            background-color: #ffffff;
        }

        QGroupBox {
            font-weight: bold;
            font-size: 9pt;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 3px 6px;
        }

        QPushButton,
        QToolButton {
            background: #056832;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 6px 10px;
            font-weight: 600;
        }
        QPushButton:hover,
        QToolButton:hover {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #6fa34a, stop:1 #4f7a34
            );
        }
        QPushButton:pressed,
        QToolButton:pressed {
            background: #4f7a34;
        }
        QPushButton:disabled,
        QToolButton:disabled {
            background: #9dbb86;
            color: #eeeeee;
        }
        """
        # self.setStyleSheet(style)

    # -------------------- UI --------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ===== BODY: dùng QSplitter để panel trái/phải kéo được =====
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        self._splitter = splitter
        splitter.splitterMoved.connect(lambda *_: self._enforce_left_pane_bounds())
        root.addWidget(splitter)

        # ===== LEFT: controls (scrollable) =====
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setMinimumWidth(self._left_min_w)
        self._left_scroll = left_scroll
        splitter.addWidget(left_scroll)

        left_container = QWidget()
        left_container.setMinimumWidth(self._left_min_w)
        left = QVBoxLayout(left_container)
        left.setContentsMargins(6, 6, 6, 6)
        left.setSpacing(8)
        left_scroll.setWidget(left_container)

        # Project info – giống Section tab
        box_proj = QGroupBox("Project")
        lp = QVBoxLayout(box_proj)
        proj_input_h = 30
        fm = self.fontMetrics()
        proj_label_w = max(
            fm.horizontalAdvance("Name:"),
            fm.horizontalAdvance("Run label:")
        ) + 15

        def _fit_proj_label(text: str) -> QLabel:
            lb = QLabel(text)
            lb.setFixedWidth(proj_label_w)
            return lb

        row_proj = QHBoxLayout()
        row_proj.addWidget(_fit_proj_label("Name:"))
        self.edit_project = QLineEdit()
        self.edit_project.setPlaceholderText("—")
        self.edit_project.setReadOnly(True)
        self.edit_project.setFixedHeight(proj_input_h)
        self.edit_project.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_proj.addWidget(self.edit_project, 1)
        row_proj.addSpacing(6)
        row_proj.addWidget(_fit_proj_label("Run label:"))
        self.edit_runlabel = QLineEdit()
        self.edit_runlabel.setPlaceholderText("—")
        self.edit_runlabel.setReadOnly(True)
        self.edit_runlabel.setFixedHeight(proj_input_h)
        self.edit_runlabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_proj.addWidget(self.edit_runlabel, 1)
        lp.addLayout(row_proj)

        left.addWidget(box_proj)

        # Sections + Advanced display
        box_sel = QGroupBox("Sections Display")
        lsd = QVBoxLayout(box_sel)
        ls = QHBoxLayout()
        self.line_combo = NoWheelComboBox()
        self.line_combo.currentIndexChanged.connect(self._on_line_changed)
        btn_render = QPushButton("Render Section")
        btn_render.clicked.connect(self._render_current_safe)
        ls.addWidget(self.line_combo)
        ls.addWidget(btn_render)
        lsd.addLayout(ls)

        row_src = QHBoxLayout()
        row_src.setSpacing(6)
        lbl_src = QLabel("Profile DEM:")
        self.profile_source_combo = NoWheelComboBox()
        self.profile_source_combo.addItem("Raw DEM", "raw")
        self.profile_source_combo.addItem("Smoothed DEM", "smooth")
        self.profile_source_combo.setCurrentIndex(0)
        self.profile_source_combo.currentIndexChanged.connect(self._on_profile_source_changed)
        row_src.addWidget(lbl_src)
        row_src.addWidget(self.profile_source_combo, 1)
        lsd.addLayout(row_src)

        # Advanced controls
        la = QHBoxLayout()
        la.setSpacing(6)

        def _fit_adv_label(text: str) -> QLabel:
            lb = QLabel(text)
            min_w = lb.fontMetrics().horizontalAdvance(text) + 8
            lb.setFixedWidth(min_w)
            return lb

        lbl_step = _fit_adv_label("Step (m):")
        la.addWidget(lbl_step)
        self.step_box = KeyboardOnlyDoubleSpinBox()
        self.step_box.setDecimals(4)
        self.step_box.setValue(float(self._default_profile_step_m))
        self.step_box.setMaximum(1e6)
        self.step_box.setMinimumWidth(56)
        self.step_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        la.addWidget(self.step_box, 1)
        lbl_scale = _fit_adv_label("Scale:")
        la.addWidget(lbl_scale)
        self.vscale = KeyboardOnlyDoubleSpinBox()
        self.vscale.setDecimals(3)
        self.vscale.setValue(0.1)
        self.vscale.setMaximum(1e6)
        self.vscale.setMinimumWidth(56)
        self.vscale.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        la.addWidget(self.vscale, 1)
        lbl_width = _fit_adv_label("Width:")
        la.addWidget(lbl_width)
        self.vwidth = KeyboardOnlyDoubleSpinBox()
        self.vwidth.setDecimals(4)
        self.vwidth.setValue(0.0015)
        self.vwidth.setMaximum(1.0)
        self.vwidth.setMinimumWidth(56)
        self.vwidth.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        la.addWidget(self.vwidth, 1)
        lsd.addLayout(la)

        left.addWidget(box_sel)

        def _set_table_visible_rows(tbl: QTableWidget, rows: int, row_h: int) -> None:
            """Fix table height to show exactly `rows` data rows (+ header)."""
            tbl.verticalHeader().setDefaultSectionSize(int(row_h))
            header_h = int(tbl.horizontalHeader().sizeHint().height())
            frame_h = int(tbl.frameWidth()) * 2
            total_h = header_h + frame_h + int(rows) * int(row_h) + 2
            tbl.setMinimumHeight(total_h)
            tbl.setMaximumHeight(total_h)

        # Group table
        box_grp = QGroupBox("Group")
        lg = QVBoxLayout(box_grp)
        self.group_table = QTableWidget(0, 4)
        self.group_table.setHorizontalHeaderLabels(
            ["Group ID", "Start (m)", "End (m)", "Color"]
        )
        self.group_table.verticalHeader().setVisible(False)
        self.group_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.group_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.group_table.cellDoubleClicked.connect(self._on_group_cell_double_clicked)
        self.group_table.itemChanged.connect(self._on_group_table_item_changed)
        _set_table_visible_rows(self.group_table, rows=6, row_h=30)
        lg.addWidget(self.group_table)

        self.curvature_check = QCheckBox("Use curvature > 0.02 as boundary")
        self.curvature_check.setChecked(True)
        lg.addWidget(self.curvature_check)

        row_curv = QHBoxLayout()
        row_curv.addWidget(QLabel("RDP eps (m):"))
        self.rdp_eps_spin = KeyboardOnlyDoubleSpinBox()
        self.rdp_eps_spin.setDecimals(3)
        self.rdp_eps_spin.setRange(0.0, 1000.0)
        self.rdp_eps_spin.setSingleStep(0.1)
        self.rdp_eps_spin.setValue(float(WORKFLOW_GROUPING_PARAMS.get("rdp_eps_m", 0.5)))
        self.rdp_eps_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        row_curv.addWidget(self.rdp_eps_spin)
        row_curv.addWidget(QLabel("Smooth radius (m):"))
        self.smooth_radius_spin = KeyboardOnlyDoubleSpinBox()
        self.smooth_radius_spin.setDecimals(3)
        self.smooth_radius_spin.setRange(0.0, 1000.0)
        self.smooth_radius_spin.setSingleStep(0.1)
        self.smooth_radius_spin.setValue(float(WORKFLOW_GROUPING_PARAMS.get("smooth_radius_m", 0.0)))
        self.smooth_radius_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        row_curv.addWidget(self.smooth_radius_spin)
        lg.addLayout(row_curv)

        self.vector_zero_check = QCheckBox("Use vector = 0° as boundary")
        self.vector_zero_check.setChecked(True)
        lg.addWidget(self.vector_zero_check)

        rowg = QHBoxLayout()
        self.btn_add_g = QPushButton("Add")
        self.btn_add_g.clicked.connect(self._on_add_group)
        self.btn_del_g = QPushButton("Delete")
        self.btn_del_g.clicked.connect(self._on_delete_group)
        self.btn_draw_curve = QPushButton("Draw Curve")
        self.btn_draw_curve.clicked.connect(self._on_draw_curve)
        self.btn_auto_group = QPushButton("Auto Group")
        self.btn_auto_group.clicked.connect(self._on_auto_group)
        self.btn_load_group = QPushButton("Load Group")
        self.btn_load_group.clicked.connect(self._on_load_group_info)

        for btn in (self.btn_auto_group, self.btn_load_group, self.btn_add_g, self.btn_del_g, self.btn_draw_curve):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            rowg.addWidget(btn, 1)
        lg.addLayout(rowg)
        left.addWidget(box_grp, 1)

        # NURBS controls
        box_nurbs = QGroupBox("NURBS")
        ln = QVBoxLayout(box_nurbs)
        ln.setContentsMargins(8, 8, 8, 8)
        ln.setSpacing(6)

        row_cfg = QHBoxLayout()
        row_cfg.addWidget(QLabel("Control points:"))
        self.nurbs_cp_spin = KeyboardOnlySpinBox()
        self.nurbs_cp_spin.setRange(2, 20)
        self.nurbs_cp_spin.setValue(4)
        self.nurbs_cp_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        row_cfg.addWidget(self.nurbs_cp_spin)
        row_cfg.addSpacing(8)
        row_cfg.addWidget(QLabel("Degree:"))
        self.nurbs_deg_spin = KeyboardOnlySpinBox()
        self.nurbs_deg_spin.setRange(1, 10)
        self.nurbs_deg_spin.setValue(3)
        self.nurbs_deg_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        row_cfg.addWidget(self.nurbs_deg_spin)
        ln.addLayout(row_cfg)

        self.nurbs_table = QTableWidget(0, 4)
        self.nurbs_table.setHorizontalHeaderLabels(["CP", "Chainage (m)", "Elev (m)", "Weight"])
        self.nurbs_table.verticalHeader().setVisible(False)
        self.nurbs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.nurbs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        _set_table_visible_rows(self.nurbs_table, rows=6, row_h=34)
        ln.addWidget(self.nurbs_table)

        row_nurbs_btn = QHBoxLayout()
        self.btn_nurbs_load = QPushButton("Load NURBS")
        self.btn_nurbs_reset = QPushButton("Reset NURBS")
        self.btn_nurbs_save = QPushButton("Save")
        row_nurbs_btn.addWidget(self.btn_nurbs_load, 1)
        row_nurbs_btn.addWidget(self.btn_nurbs_reset, 1)
        row_nurbs_btn.addWidget(self.btn_nurbs_save, 1)
        ln.addLayout(row_nurbs_btn)

        self.nurbs_cp_spin.valueChanged.connect(self._on_nurbs_cp_spin_changed)
        self.nurbs_deg_spin.valueChanged.connect(self._on_nurbs_deg_spin_changed)
        self.btn_nurbs_load.clicked.connect(self._on_load_nurbs_info)
        self.btn_nurbs_reset.clicked.connect(self._on_nurbs_reset_defaults)
        self.btn_nurbs_save.clicked.connect(self._on_nurbs_save)

        left.addWidget(box_nurbs, 0)

        # Status
        box_st = QGroupBox("Status")
        ls = QVBoxLayout(box_st)
        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setMinimumHeight(170)
        self.status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ls.addWidget(self.status)
        left.addWidget(box_st, 1)

        # ===== RIGHT: preview (zoomable like AnalyzeTab) =====
        right_container = QWidget()
        right_wrap = QVBoxLayout(right_container)
        right_wrap.setContentsMargins(0, 0, 0, 0)

        zoombar = QToolBar()
        act_in = QAction("Zoom +", self)
        act_in.triggered.connect(lambda: self.view.zoom_in())
        act_out = QAction("Zoom –", self)
        act_out.triggered.connect(lambda: self.view.zoom_out())
        act_fit = QAction("Fit", self)
        act_fit.triggered.connect(lambda: self.view.fit_to_scene())
        act_100 = QAction("100%", self)
        act_100.triggered.connect(lambda: self.view.set_100())
        zoombar.addAction(act_in)
        zoombar.addAction(act_out)
        zoombar.addAction(act_fit)
        zoombar.addAction(act_100)
        zoombar.setIconSize(QSize(22, 22))
        zoombar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        zoombar.setContentsMargins(0, 0, 0, 0)
        zoombar.setStyleSheet("QToolBar { spacing: 6px; background: transparent; border: none; }")
        right_wrap.addWidget(zoombar)

        self._profile_cursor_label = QLabel("Cursor: —")
        self._profile_cursor_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        right_wrap.addWidget(self._profile_cursor_label)

        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.sceneMouseMoved.connect(self._on_profile_scene_mouse_moved)
        self.view.hoverExited.connect(self._clear_profile_cursor_readout)
        right_wrap.addWidget(self.view, 1)

        splitter.addWidget(right_container)

        # Tỉ lệ ban đầu giữa panel trái/phải
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([self._left_default_w, 900])

        self._apply_button_style()

    def _left_max_w(self) -> int:
        base_w = self.width()
        if self._splitter is not None and self._splitter.width() > 0:
            base_w = self._splitter.width()
        if base_w < (self._left_min_w * 2):
            return -1
        return max(self._left_min_w, int(base_w * 0.5))

    def _try_apply_initial_splitter_width(self) -> None:
        if not self._pending_init_splitter or self._splitter is None:
            return
        max_w = self._left_max_w()
        if max_w < 0:
            return
        init_left = max(self._left_min_w, min(self._left_default_w, max_w))
        total = sum(self._splitter.sizes())
        if total <= 0:
            total = max(self._splitter.width(), self.width(), init_left + 1)
        self._splitter.setSizes([init_left, max(1, total - init_left)])
        self._pending_init_splitter = False

    def _enforce_left_pane_bounds(self) -> None:
        if self._splitter is None:
            return
        self._try_apply_initial_splitter_width()
        sizes = self._splitter.sizes()
        if len(sizes) != 2:
            return
        left_w, right_w = sizes
        total = left_w + right_w
        max_w = self._left_max_w()
        if max_w < 0:
            return
        clamped_left = max(self._left_min_w, min(left_w, max_w))
        if clamped_left != left_w and total > 0:
            self._splitter.setSizes([clamped_left, max(1, total - clamped_left)])

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._enforce_left_pane_bounds()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._enforce_left_pane_bounds()

    @staticmethod
    def _pick_existing_path(*cands: str) -> str:
        for p in cands:
            if p and os.path.exists(p):
                return p
        return ""

    def _current_profile_source_key(self) -> str:
        try:
            if self.profile_source_combo is not None:
                key = str(self.profile_source_combo.currentData() or "").strip().lower()
                if key in ("raw", "smooth"):
                    return key
        except Exception:
            pass
        return "raw"

    def _set_profile_source_key(self, key: Optional[str], *, log_paths: bool = False) -> str:
        src = str(key or "").strip().lower()
        if src not in ("raw", "smooth"):
            return self._current_profile_source_key()
        try:
            if self.profile_source_combo is not None:
                idx = self.profile_source_combo.findData(src)
                if idx >= 0:
                    old = self.profile_source_combo.blockSignals(True)
                    self.profile_source_combo.setCurrentIndex(idx)
                    self.profile_source_combo.blockSignals(old)
        except Exception:
            pass
        self._refresh_profile_source_paths(log_paths=log_paths)
        return self._current_profile_source_key()

    def _refresh_profile_source_paths(self, log_paths: bool = False) -> None:
        src = self._current_profile_source_key()
        if src == "smooth":
            chosen = self._pick_existing_path(self.dem_path_smooth, self.dem_path_raw)
        else:
            chosen = self._pick_existing_path(self.dem_path_raw, self.dem_path_smooth)
        self.dem_path = chosen
        self.ground_export_dem_path = chosen
        self._sync_step_to_raw_dem(log_paths=log_paths)
        if log_paths:
            self._log(f"[UI3] Profile source: {src}")
            self._log(f"[UI3] Raw DEM: {self.dem_path_raw}")
            self._log(f"[UI3] Smoothed DEM: {self.dem_path_smooth}")
            self._log(f"[UI3] Active profile DEM: {self.dem_path}")

    def _raw_dem_pixel_step_m(self) -> Optional[float]:
        dem_path = str(getattr(self, "dem_path_raw", "") or "").strip()
        if not dem_path or not os.path.exists(dem_path):
            return None
        try:
            with rasterio.open(dem_path) as ds:
                xres, yres = ds.res
            vals = [float(v) for v in (xres, yres) if np.isfinite(v) and float(v) > 0.0]
            if not vals:
                return None
            return float(min(vals))
        except Exception:
            return None

    def _sync_step_to_raw_dem(self, *, log_paths: bool = False) -> None:
        if self.step_box is None:
            return
        step_m = self._raw_dem_pixel_step_m()
        if step_m is None:
            return
        try:
            old = self.step_box.blockSignals(True)
            self.step_box.setValue(float(step_m))
            self.step_box.blockSignals(old)
        except Exception:
            return
        if log_paths:
            self._log(f"[UI3] Sampling step synced to raw DEM pixel size: {step_m:.4f} m")

    def _include_vector_angle_zero(self) -> bool:
        try:
            if self.vector_zero_check is not None:
                return bool(self.vector_zero_check.isChecked())
        except Exception:
            pass
        return bool(WORKFLOW_GROUPING_PARAMS.get("include_vector_angle_zero", True))

    def _include_curvature_threshold(self) -> bool:
        try:
            if self.curvature_check is not None:
                return bool(self.curvature_check.isChecked())
        except Exception:
            pass
        return bool(WORKFLOW_GROUPING_PARAMS.get("include_curvature_threshold", True))

    def _current_rdp_eps_m(self) -> float:
        try:
            if self.rdp_eps_spin is not None:
                val = float(self.rdp_eps_spin.value())
                if np.isfinite(val) and val >= 0.0:
                    return float(val)
        except Exception:
            pass
        return float(WORKFLOW_GROUPING_PARAMS.get("rdp_eps_m", 0.5))

    def _current_smooth_radius_m(self) -> float:
        try:
            if self.smooth_radius_spin is not None:
                val = float(self.smooth_radius_spin.value())
                if np.isfinite(val) and val >= 0.0:
                    return float(val)
        except Exception:
            pass
        return float(WORKFLOW_GROUPING_PARAMS.get("smooth_radius_m", 0.0))

    def _grouping_params_current(self) -> Dict[str, Any]:
        params = dict(WORKFLOW_GROUPING_PARAMS)
        params["rdp_eps_m"] = self._current_rdp_eps_m()
        params["smooth_radius_m"] = self._current_smooth_radius_m()
        params["include_curvature_threshold"] = self._include_curvature_threshold()
        params["include_vector_angle_zero"] = self._include_vector_angle_zero()
        return params

    def _apply_group_json_settings(self, data: Dict[str, Any], *, log_paths: bool = False) -> None:
        try:
            if self.curvature_check is not None and "include_curvature_threshold" in data:
                self.curvature_check.setChecked(bool(data.get("include_curvature_threshold")))
        except Exception:
            pass
        try:
            if self.vector_zero_check is not None and "include_vector_angle_zero" in data:
                self.vector_zero_check.setChecked(bool(data.get("include_vector_angle_zero")))
        except Exception:
            pass
        try:
            if self.rdp_eps_spin is not None and "rdp_eps_m" in data:
                self.rdp_eps_spin.setValue(max(0.0, float(data.get("rdp_eps_m"))))
        except Exception:
            pass
        try:
            if self.smooth_radius_spin is not None and "smooth_radius_m" in data:
                self.smooth_radius_spin.setValue(max(0.0, float(data.get("smooth_radius_m"))))
        except Exception:
            pass
        try:
            if "profile_dem_source" in data:
                self._set_profile_source_key(str(data.get("profile_dem_source")), log_paths=log_paths)
        except Exception:
            pass

    @staticmethod
    def _group_signature(groups: List[dict]) -> List[Tuple[str, float, float, str, str]]:
        sig: List[Tuple[str, float, float, str, str]] = []
        for i, g in enumerate(groups or [], 1):
            try:
                gid = str(g.get("id", g.get("group_id", f"G{i}")) or f"G{i}").strip()
                s = float(g.get("start", g.get("start_chainage")))
                e = float(g.get("end", g.get("end_chainage")))
            except Exception:
                continue
            if e < s:
                s, e = e, s
            sig.append((
                gid,
                round(float(s), 3),
                round(float(e), 3),
                str(g.get("start_reason", "") or "").strip(),
                str(g.get("end_reason", "") or "").strip(),
            ))
        return sig

    @staticmethod
    def _ui3_chainage_origin() -> str:
        return SECTION_CHAINAGE_ORIGIN

    def _groups_to_current_chainage(
        self,
        groups: List[dict],
        *,
        source_origin: Optional[str] = None,
        length_m: Optional[float] = None,
    ) -> List[dict]:
        _ = source_origin
        _ = length_m
        out: List[Dict[str, Any]] = []
        for i, g in enumerate(groups or [], 1):
            try:
                s = float(g.get("start", g.get("start_chainage", np.nan)))
                e = float(g.get("end", g.get("end_chainage", np.nan)))
            except Exception:
                continue
            if not (np.isfinite(s) and np.isfinite(e)):
                continue
            gid = str(g.get("id", g.get("group_id", f"G{i}")) or f"G{i}").strip() or f"G{i}"
            start_reason = str(g.get("start_reason", "") or "").strip()
            end_reason = str(g.get("end_reason", "") or "").strip()
            if e < s:
                s, e = e, s
                start_reason, end_reason = end_reason, start_reason
            out.append({
                "id": gid,
                "start": float(s),
                "end": float(e),
                "start_reason": start_reason,
                "end_reason": end_reason,
                "color": str(g.get("color", "") or "").strip(),
            })
        return _renumber_groups_visual_order(out)

    @staticmethod
    def _interp_series_y(xs: np.ndarray, ys: np.ndarray, xq: float) -> Optional[float]:
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if xs.ndim != 1 or ys.ndim != 1 or xs.size < 2 or ys.size != xs.size:
            return None
        keep = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[keep]
        ys = ys[keep]
        if xs.size < 2:
            return None
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        if not np.isfinite(float(xq)):
            return None
        if xq < float(xs[0]) or xq > float(xs[-1]):
            return None
        if np.any(np.isclose(xs, float(xq), atol=1e-9)):
            idx = int(np.flatnonzero(np.isclose(xs, float(xq), atol=1e-9))[0])
            return float(ys[idx])
        idx = int(np.searchsorted(xs, float(xq)))
        if idx <= 0 or idx >= xs.size:
            return None
        x0 = float(xs[idx - 1]); x1 = float(xs[idx])
        y0 = float(ys[idx - 1]); y1 = float(ys[idx])
        if not (np.isfinite(x0) and np.isfinite(x1) and np.isfinite(y0) and np.isfinite(y1)):
            return None
        if abs(x1 - x0) <= 1e-12:
            return None
        t = (float(xq) - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0))

    def _repair_saved_curvature_points(
        self,
        points: List[dict],
        prof: Optional[dict],
    ) -> List[dict]:
        repaired = [dict(p) for p in (points or [])]
        if len(repaired) < 2 or not prof:
            return repaired
        try:
            params = self._grouping_params_current()
            nodes = extract_curvature_rdp_nodes(
                prof,
                rdp_eps_m=float(params.get("rdp_eps_m", 0.5)),
                smooth_radius_m=float(params.get("smooth_radius_m", 0.0)),
                restrict_to_slip_span=False,
            )
            full_x = np.asarray(nodes.get("chain", []), dtype=float)
            full_k = np.asarray(nodes.get("curvature", []), dtype=float)
        except Exception:
            return repaired
        if full_x.size < 2 or full_k.size != full_x.size:
            return repaired

        def _repair_one(item: dict, neighbor: Optional[dict], is_head: bool) -> None:
            try:
                idx = int(item.get("index", -1))
                kval = float(item.get("curvature", np.nan))
                xval = float(item.get("chain_m", np.nan))
            except Exception:
                return
            if not (np.isfinite(kval) and np.isfinite(xval)):
                return
            if abs(kval) > 1e-12:
                return
            if is_head:
                if idx != 0:
                    return
            else:
                try:
                    if neighbor is None:
                        return
                    nidx = int(neighbor.get("index", -1))
                    if idx != (nidx + 1):
                        return
                except Exception:
                    return
            kval_new = self._interp_series_y(full_x, full_k, float(xval))
            if kval_new is None or not np.isfinite(kval_new):
                return
            item["curvature"] = float(kval_new)
            item["curvature_abs"] = abs(float(kval_new))

        _repair_one(repaired[0], repaired[1] if len(repaired) > 1 else None, True)
        _repair_one(repaired[-1], repaired[-2] if len(repaired) > 1 else None, False)
        return repaired

    def _saved_curvature_series_for_line(
        self,
        line_id: str,
        current_groups: Optional[List[dict]],
        prof: Optional[dict] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        path = self._groups_json_path_for(line_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            return None

        saved_src = str(data.get("profile_dem_source", "") or "").strip().lower()
        current_src = self._current_profile_source_key()
        if saved_src in ("raw", "smooth") and current_src in ("raw", "smooth") and saved_src != current_src:
            return None

        saved_groups = self._groups_to_current_chainage(
            data.get("groups", []) or [],
            source_origin=data.get("chainage_origin"),
            length_m=(prof.get("length_m") if prof else None),
        )
        if current_groups is not None:
            if self._group_signature(saved_groups) != self._group_signature(current_groups):
                return None

        pts = self._repair_saved_curvature_points(data.get("curvature_points", []) or [], prof)
        xs: List[float] = []
        ks: List[float] = []
        for it in pts:
            try:
                x = float(it.get("chain_m"))
                k = float(it.get("curvature"))
            except Exception:
                continue
            if not (np.isfinite(x) and np.isfinite(k)):
                continue
            xs.append(float(x))
            ks.append(float(k))
        if len(xs) < 2:
            return None
        order = np.argsort(np.asarray(xs, dtype=float))
        return np.asarray(xs, dtype=float)[order], np.asarray(ks, dtype=float)[order]

    def _on_profile_source_changed(self, _idx: int) -> None:
        self._refresh_profile_source_paths(log_paths=True)
        self._active_prof = None
        if self.line_combo is None or self.line_combo.count() == 0:
            return
        self._render_current_safe()

    def _compute_profile_for_geom(self, geom, *, slip_only: bool) -> Optional[dict]:
        dem_path = str(getattr(self, "dem_path", "") or "").strip()
        dem_orig_path = str(getattr(self, "ground_export_dem_path", "") or "").strip()
        if not dem_path:
            return None
        prof = compute_profile(
            dem_path,
            self.dx_path,
            self.dy_path,
            self.dz_path,
            geom,
            step_m=self.step_box.value(),
            smooth_win=11,
            smooth_poly=2,
            slip_mask_path=self.slip_path,
            slip_only=bool(slip_only),
            dem_orig_path=(dem_orig_path or dem_path),
        )
        if not prof:
            return None
        prof["profile_dem_source"] = self._current_profile_source_key()
        prof["profile_dem_path"] = dem_path
        return prof

    # -------------------- Context --------------------
    def set_context(self, project: str, run_label: str, run_dir: str) -> None:
        """Du?c g?i t? MainWindow sau khi Analyze/Section xong."""
        self._ctx.update({"project": project, "run_label": run_label, "run_dir": run_dir})
        self._ui2_intersections_cache = None
        self._anchors_xyz_cache = None
        # uu tiˆn d?c t? ui_shared_data.json (do Analyze/Section ghi)
        shared_jsons = [
            os.path.join(run_dir, "ui_shared_data.json"),
            os.path.join(self.base_dir, "output", "ui_shared_data.json"),
            os.path.join(self.base_dir, "output", "UI1", "ui_shared_data.json"),
        ]
        js = {}
        for p in shared_jsons:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        js.update(json.load(f))
                except Exception:
                    pass

        # fallback to auto_paths when json missing
        ap = auto_paths()

        meta_inputs = {}
        meta_processed = {}
        try:
            meta_path = os.path.join(run_dir, "ingest_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}
                meta_inputs = meta.get("inputs") or {}
                meta_processed = meta.get("processed") or {}
        except Exception:
            pass

        def _pick_first(*cands: str) -> str:
            for p in cands:
                if p and os.path.exists(p):
                    return p
            return ""

        self.dem_path_smooth = _pick_first(
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
        self.dem_path_raw = _pick_first(
            meta_inputs.get("after_dem") or "",
            meta_inputs.get("before_dem") or "",
            os.path.join(run_dir, "input", "after_dem.tif"),
            os.path.join(run_dir, "input", "before_dem.tif"),
            js.get("dem_ground_path") or "",
            ap.get("dem_orig", ""),
            ap.get("dem", ""),
        )
        self._refresh_profile_source_paths(log_paths=True)
        self.dx_path = _pick_first(
            js.get("dx_path") or "",
            ap.get("dx", ""),
            os.path.join(run_dir, "ui1", "dx.tif"),
        )
        self.dy_path = _pick_first(
            js.get("dy_path") or "",
            ap.get("dy", ""),
            os.path.join(run_dir, "ui1", "dy.tif"),
        )
        self.dz_path = _pick_first(
            js.get("dz_path") or "",
            ap.get("dz", ""),
            os.path.join(run_dir, "ui1", "dz.tif"),
        )
        # QUAN TR?NG: lines t? UI2
        # QUAN TR?NG: kh“ng set lines_path legacy n?a
        self.lines_path = ""  # d? loader t? quy?t d?nh t? sections.csv

        # slip-zone mask (d? ch? v? trong v-ng tru?t)
        self.slip_path = js.get("slip_path") or ap.get("slip", "")
        # Prefer run-scoped slip mask from ingest_meta.json (pipeline step_detect)
        try:
            slip_mask = meta_processed.get("slip_mask")
            if slip_mask:
                self.slip_path = slip_mask
        except Exception:
            pass

        # Fallback to current run's ui1 output if still missing
        if not self.slip_path:
            self.slip_path = os.path.join(run_dir, "ui1", "landslide_mask.tif")
        # sau khi self.slip_path = os.path.join(..., "ui1", "step7_slipzone", "slip_zone.asc")
        if not self.slip_path or not os.path.exists(self.slip_path):
            self._warn("[UI3] Slip-zone mask not found. Vectors outside landslide may appear.")

        if not os.path.exists(self.slip_path):
            alt = self.slip_path.replace(".asc", ".tif")
            if os.path.exists(alt):
                self.slip_path = alt

        self._load_lines_into_combo()

    def reset_session(self) -> None:
        """
        Reset tab Curve Analyze cho New Session:
        - Xoá context project/run
        - Xoá line combo, bảng group
        - Clear hình nền / guide trên QGraphicsScene
        - Clear status
        """
        # 1) Reset context
        self._ctx = {"project": "", "run_label": "", "run_dir": ""}

        # 2) Reset thông tin hiển thị project/run nếu có label
        # 2) Reset thông tin hiển thị project/run nếu có
        if hasattr(self, "edit_project") and self.edit_project is not None:
            self.edit_project.clear()
        if hasattr(self, "edit_runlabel") and self.edit_runlabel is not None:
            self.edit_runlabel.clear()

        # 3) Reset các path dữ liệu
        self.dem_path = ""
        self.dem_path_raw = ""
        self.dem_path_smooth = ""
        self.ground_export_dem_path = ""
        self.dx_path = ""
        self.dy_path = ""
        self.dz_path = ""
        self.lines_path = ""
        self.slip_path = ""
        try:
            if self.profile_source_combo is not None:
                self.profile_source_combo.blockSignals(True)
                self.profile_source_combo.setCurrentIndex(0)
                self.profile_source_combo.blockSignals(False)
        except Exception:
            pass
        try:
            if self.step_box is not None:
                old = self.step_box.blockSignals(True)
                self.step_box.setValue(float(self._default_profile_step_m))
                self.step_box.blockSignals(old)
        except Exception:
            pass
        try:
            if self.curvature_check is not None:
                self.curvature_check.setChecked(True)
        except Exception:
            pass
        try:
            if self.rdp_eps_spin is not None:
                self.rdp_eps_spin.setValue(float(WORKFLOW_GROUPING_PARAMS.get("rdp_eps_m", 0.5)))
        except Exception:
            pass
        try:
            if self.smooth_radius_spin is not None:
                self.smooth_radius_spin.setValue(float(WORKFLOW_GROUPING_PARAMS.get("smooth_radius_m", 0.0)))
        except Exception:
            pass
        try:
            if self.vector_zero_check is not None:
                self.vector_zero_check.setChecked(True)
        except Exception:
            pass

        # 4) Clear combo line (list tuyến cắt)
        try:
            if self.line_combo is not None:
                self.line_combo.blockSignals(True)
                self.line_combo.clear()
                self.line_combo.blockSignals(False)
        except Exception:
            pass

        # 5) Clear bảng group
        try:
            if self.group_table is not None:
                self.group_table.setRowCount(0)
        except Exception:
            pass

        # 6) Clear scene & reset zoom viewer
        try:
            if self.scene is not None:
                self.scene.clear()
        except Exception:
            pass
        try:
            if self.view is not None:
                self.view.resetTransform()
        except Exception:
            pass

        # 7) Reset state nội bộ cho guide / overlay
        self._px_per_m = None
        self._sec_len_m = None
        self._group_bounds.clear()
        self._guide_lines_top.clear()
        self._guide_lines_bot.clear()
        self._group_bands_bot.clear()
        self._img_ground = None
        self._img_rate0 = None
        self._clear_curve_overlay()
        self._active_prof = None
        self._active_groups = []
        self._active_base_curve = None
        self._active_curve = None
        self._ui2_intersections_cache = None
        self._anchors_xyz_cache = None
        self._plot_x0_px = None
        self._plot_w_px = None
        self._x_min = None
        self._x_max = None
        self._current_idx = 0

        # 8) Clear status log
        try:
            if self.status is not None:
                self.status.clear()
        except Exception:
            pass

        # 9) Log nhẹ cho dễ debug
        try:
            self._log("[UI3] Curve tab session reset.")
        except Exception:
            pass

    def _load_lines_into_combo(self) -> None:
        """
        Đọc ui2/sections.csv của run hiện tại, build GeoDataFrame line,
        đổ vào combo và set self._gdf.
        (Không dùng selected_lines.gpkg cũ nữa.)
        """
        self.line_combo.blockSignals(True)
        self.line_combo.clear()
        try:
            run_dir = self._ctx.get("run_dir") or ""
            if not run_dir:
                self._log("[!] Run context is empty – cannot load sections.")
                return

            csv_path = os.path.join(run_dir, "ui2", "sections.csv")
            migrated = _ensure_sections_csv_current(csv_path, run_dir=run_dir)
            if migrated:
                self._log("[UI3] Migrated legacy sections.csv to direction_version=2 and cleared old UI3 derived outputs.")
            gdf = _build_gdf_from_sections_csv(csv_path, self.dem_path)
            if gdf is None or gdf.empty:
                self._log(f"[!] No sections in csv:\n{csv_path}")
                return

            # Gán lại cho UI3 dùng
            self._gdf = gdf

            # Tạo label cho combo: "Line i  (xxx.x m)"
            labels: List[str] = []
            for _, row in gdf.iterrows():
                base = row.get("name") or f"Line {int(row.get('idx', 0) or 0)}"
                L = float(row.get("length_m", float("nan")))
                if math.isfinite(L) and L > 0:
                    labels.append(f"{base}  ({L:.1f} m)")
                else:
                    labels.append(base)

            self.line_combo.addItems(labels)
            self._log(f"[i] Loaded {len(labels)} lines from ui2/sections.csv.")

            # (Tuỳ chọn) nếu bạn vẫn muốn có GPKG để các chỗ khác dùng:
            # try:
            #     os.makedirs(os.path.join(run_dir, "UI2", "step2_selected_lines"), exist_ok=True)
            #     gpkg_path = os.path.join(run_dir, "UI2", "step2_selected_lines", "selected_lines.gpkg")
            #     gdf.to_file(gpkg_path, driver="GPKG")
            #     self.lines_path = gpkg_path
            # except Exception:
            #     pass

        except Exception as e:
            self._log(f"[!] Cannot load lines from sections.csv: {e}")
        finally:
            self.line_combo.blockSignals(False)

        # cập nhật nhãn project
        # cập nhật ô Project / Run label giống các tab khác
        pj = self._ctx.get("project") or "—"
        rl = self._ctx.get("run_label") or "—"

        if hasattr(self, "edit_project") and self.edit_project is not None:
            self.edit_project.setText(pj)
        if hasattr(self, "edit_runlabel") and self.edit_runlabel is not None:
            self.edit_runlabel.setText(rl)

        if self.line_combo.count() > 0:
            self.line_combo.blockSignals(True)
            self.line_combo.setCurrentIndex(0)
            self.line_combo.blockSignals(False)
            self._on_line_changed(0)

    def _on_line_changed(self, _idx: int) -> None:
        self._log("[i] Line changed. Loading saved UI3 data if available.")
        self._clear_curve_overlay()
        self._active_prof = None
        self._active_groups = []
        self._active_base_curve = None
        self._active_curve = None
        try:
            row = self.line_combo.currentIndex()
            if row >= 0 and hasattr(self, "_gdf") and self._gdf is not None and not self._gdf.empty:
                geom = self._gdf.geometry.iloc[row]
                self._sec_len_m = float(getattr(geom, "length", np.nan))
        except Exception:
            self._sec_len_m = None

        self._load_saved_curve_state_for_current_line()
        self._refresh_anchor_overlay()

    def _load_group_table_from_path(self, path: str, line_id: str) -> bool:
        if not os.path.exists(path):
            return False
        loaded = 0
        self._group_table_updating = True
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            self._apply_group_json_settings(data)
            groups = self._groups_to_current_chainage(
                data.get("groups", []) or [],
                source_origin=data.get("chainage_origin"),
            )
            self.group_table.setRowCount(0)
            for g in (groups or []):
                try:
                    gid = str(g.get("group_id", g.get("id", ""))).strip()
                    s = float(g.get("start", g.get("start_chainage")))
                    e = float(g.get("end", g.get("end_chainage")))
                except Exception:
                    continue
                if e < s:
                    s, e = e, s
                r = self.group_table.rowCount()
                self.group_table.insertRow(r)
                self.group_table.setItem(r, 0, QTableWidgetItem(gid or f"G{r + 1}"))
                self.group_table.setItem(r, 1, QTableWidgetItem(f"{s:.3f}"))
                self.group_table.setItem(r, 2, QTableWidgetItem(f"{e:.3f}"))
                self._set_group_boundary_reason(r, 1, str(g.get("start_reason", "") or ""))
                self._set_group_boundary_reason(r, 2, str(g.get("end_reason", "") or ""))
                self._set_color_cell(r, str(g.get("color", "")).strip())
                loaded += 1
            if loaded:
                self._append_ungrouped_row(self._read_groups_from_table(), self._sec_len_m)
        except Exception as e:
            self._log(f"[!] Cannot read curve group file: {e}")
            return False
        finally:
            self._group_table_updating = False
        if loaded:
            self._set_curve_method_for_line(line_id, "nurbs")
            self._log(f"[UI3] Loaded group table from: {path}")
            return True
        return False

    def _try_load_group_table_from_curve(self, line_id: str) -> bool:
        path = self._curve_group_json_path_for(line_id)
        return self._load_group_table_from_path(path, line_id)

    def _on_load_group_info(self) -> None:
        if self.line_combo.count() == 0:
            self._warn("[UI3] No line selected.")
            return
        try:
            start_dir = self._curve_dir()
        except Exception:
            start_dir = self.base_dir
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load group_info",
            start_dir,
            "JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return
        line_id = self._line_id_current()
        if self._load_group_table_from_path(path, line_id):
            try:
                self._sync_nurbs_defaults_from_group_table()
            except Exception:
                pass
        else:
            self._warn("[UI3] Cannot load group_info file.")

    def _load_nurbs_table_from_path(self, path: str, line_id: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            cp_items = data.get("control_points", []) or []
            if not cp_items:
                return False
            cp_items = sorted(
                cp_items,
                key=lambda d: int(d.get("cp_index", 0)) if str(d.get("cp_index", "")).strip() else 0,
            )
            cps = []
            ws = []
            for cp in cp_items:
                try:
                    s = float(cp.get("chainage_m"))
                    z = float(cp.get("elev_m"))
                    w = float(cp.get("weight", 1.0))
                except Exception:
                    continue
                if not (np.isfinite(s) and np.isfinite(z)):
                    continue
                cps.append([float(s), float(z)])
                ws.append(float(w) if np.isfinite(w) and w > 0 else 1.0)
            if len(cps) < 2:
                return False
            params = {
                "degree": int(data.get("degree", 3)),
                "control_points": cps,
                "weights": ws if len(ws) == len(cps) else [1.0] * len(cps),
            }
            self._set_nurbs_params_for_line(line_id, params)
            n_ctrl = len(cps)
            deg = max(1, min(int(params.get("degree", 3)), n_ctrl - 1))
            self._nurbs_updating_ui = True
            try:
                self.nurbs_cp_spin.setValue(n_ctrl)
                self.nurbs_deg_spin.setMaximum(max(1, n_ctrl - 1))
                self.nurbs_deg_spin.setValue(deg)
                self._populate_nurbs_table(params)
            finally:
                self._nurbs_updating_ui = False
            self._draw_control_points_overlay(params)
            self._set_curve_method_for_line(line_id, "nurbs")
            # Ensure live NURBS overlay is recomputed from loaded control points.
            if not self._active_prof:
                self._active_prof = self._build_profile_for_current_line()
            groups_now = self._read_groups_from_table() or []
            if groups_now and self._active_prof:
                try:
                    groups_now = clamp_groups_to_slip(self._active_prof, groups_now, min_len=WORKFLOW_GROUP_MIN_LEN_M)
                except Exception:
                    pass
            self._active_groups = groups_now
            self._schedule_nurbs_live_update()
            self._log(f"[UI3] Loaded NURBS table from: {path}")
            return True
        except Exception as e:
            self._log(f"[!] Cannot read curve nurbs_info file: {e}")
            return False

    def _try_load_nurbs_table_from_curve(self, line_id: str) -> bool:
        path = self._curve_nurbs_info_json_path_for(line_id)
        return self._load_nurbs_table_from_path(path, line_id)

    def _on_load_nurbs_info(self) -> None:
        if self.line_combo.count() == 0:
            self._warn("[UI3] No line selected.")
            return
        try:
            start_dir = self._curve_dir()
        except Exception:
            start_dir = self.base_dir
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load nurbs_info",
            start_dir,
            "JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return
        line_id = self._line_id_current()
        if not self._load_nurbs_table_from_path(path, line_id):
            self._warn("[UI3] Cannot load nurbs_info file.")

    def _try_load_nurbs_preview_from_curve(self, line_id: str) -> bool:
        path = self._curve_nurbs_png_path_for(line_id)
        if not os.path.exists(path):
            return False
        pm = QPixmap(path)
        if pm.isNull():
            return False
        try:
            QPixmapCache.clear()
            self.scene.clear()
            item = QGraphicsPixmapItem(pm)
            self.scene.addItem(item)
            self._img_ground = item
            self._img_rate0 = item
            self._clear_curve_overlay()
            self._load_axes_meta(path)
            self._static_nurbs_bg_loaded = True
            if getattr(self, "_first_show", True):
                self.view.fit_to_scene()
                self._first_show = False
            self._refresh_anchor_overlay()
            self._log(f"[UI3] Loaded NURBS preview: {path}")
            return True
        except Exception:
            return False

    def _build_profile_for_current_line(self) -> Optional[dict]:
        try:
            if not hasattr(self, "_gdf") or self._gdf is None or self._gdf.empty:
                return None
            row = self.line_combo.currentIndex()
            if row < 0:
                return None
            geom = self._gdf.geometry.iloc[row]
            prof = self._compute_profile_for_geom(geom, slip_only=False)
            if not prof or len(prof.get("chain", [])) < 2:
                return None
            return prof
        except Exception:
            return None

    def _load_saved_curve_state_for_current_line(self) -> None:
        line_id = self._line_id_current()
        loaded_groups = self._try_load_group_table_from_curve(line_id)
        if not loaded_groups:
            self._populate_group_table_for_current_line()

        prof = self._build_profile_for_current_line()
        if prof is not None:
            self._active_prof = prof

        groups = self._read_groups_from_table() or []
        if groups and self._active_prof:
            try:
                groups = clamp_groups_to_slip(self._active_prof, groups, min_len=WORKFLOW_GROUP_MIN_LEN_M)
            except Exception:
                pass
        self._active_groups = groups

        loaded_preview = self._try_load_nurbs_preview_from_curve(line_id)
        self._try_load_nurbs_table_from_curve(line_id)
        if not loaded_preview:
            self._log("[i] No saved NURBS preview in ui3/curve. Click 'Render Section' to preview.")
        self._refresh_anchor_overlay()

    # def _groups_json_path(self) -> str:
    #     line_label = self.line_combo.currentText().strip() or f"line_{self.line_combo.currentIndex() + 1:03d}"
    #     safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in line_label)
    #     gdir = os.path.join(self.base_dir, "output", "UI3", "groups");
    #     os.makedirs(gdir, exist_ok=True)
    #     return os.path.join(gdir, f"{safe}.json")

    def _populate_group_table_for_current_line(self) -> None:
        path = self._groups_json_path()
        self._group_table_updating = True
        try:
            self.group_table.setRowCount(0)
            loaded = 0
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        js = json.load(f)
                    self._apply_group_json_settings(js)
                    cm = self._normalize_curve_method(js.get("curve_method"))
                    self._set_curve_method_for_line(self._line_id_current(), cm)
                    groups = self._groups_to_current_chainage(
                        js.get("groups", []) or [],
                        source_origin=js.get("chainage_origin"),
                    )
                    for g in groups:
                        r = self.group_table.rowCount();
                        self.group_table.insertRow(r)
                        self.group_table.setItem(r, 0, QTableWidgetItem(str(g.get("id", ""))))
                        self.group_table.setItem(r, 1, QTableWidgetItem(f'{float(g.get("start", 0.0)):.3f}'))
                        self.group_table.setItem(r, 2, QTableWidgetItem(f'{float(g.get("end", 0.0)):.3f}'))
                        self._set_group_boundary_reason(r, 1, str(g.get("start_reason", "") or ""))
                        self._set_group_boundary_reason(r, 2, str(g.get("end_reason", "") or ""))
                        self._set_color_cell(r, str(g.get("color", "")).strip())
                        loaded += 1
                except Exception as e:
                    self._log(f"[!] Cannot read groups: {e}")
            if loaded:
                self._append_ungrouped_row(self._read_groups_from_table(), self._sec_len_m)
            if loaded == 0:
                # 3 dòng trống mặc định
                for _ in range(3):
                    r = self.group_table.rowCount();
                    self.group_table.insertRow(r)
                    self.group_table.setItem(r, 0, QTableWidgetItem(""))
        finally:
            self._group_table_updating = False

    def _read_group_table(self) -> List[dict]:
        out = []
        rc = self.group_table.rowCount()
        for r in range(rc):
            gid = (self.group_table.item(r, 0).text().strip() if self.group_table.item(r, 0) else "")
            if gid.upper() == "UNGROUPED":
                continue
            s = (self.group_table.item(r, 1).text().strip() if self.group_table.item(r, 1) else "")
            e = (self.group_table.item(r, 2).text().strip() if self.group_table.item(r, 2) else "")
            color = self._get_color_cell_value(r)
            if not gid and not s and not e:
                continue
            try:
                s_val = float(s);
                e_val = float(e)
                if e_val < s_val: s_val, e_val = e_val, s_val
                out.append({"id": gid or f"G{len(out) + 1}", "start": s_val, "end": e_val, "color": color or None})
            except:
                self._log(f"[!] Row {r + 1}: start/end invalid.")
        return out

    def _get_groups_for_current_line(self):
        """Alias để tương thích với các chỗ gọi cũ; trả về list nhóm hiện hành."""
        return self._load_groups_for_current_line()

    # --- run-scoped path helpers (save under <run_dir>/ui3/...) ---
    def _ui3_run_dir(self) -> str:
        run_dir = (self._ctx.get("run_dir") or "").strip()
        if not run_dir:
            raise RuntimeError("[UI3] Run context is empty. Call set_context() first.")
        path = os.path.join(run_dir, "ui3")
        os.makedirs(path, exist_ok=True)
        return path

    def _preview_dir(self) -> str:
        path = os.path.join(self._ui3_run_dir(), "preview")
        os.makedirs(path, exist_ok=True)
        return path

    def _groups_dir(self) -> str:
        path = os.path.join(self._ui3_run_dir(), "groups")
        os.makedirs(path, exist_ok=True)
        return path

    def _vectors_dir(self) -> str:
        path = os.path.join(self._ui3_run_dir(), "vectors")
        os.makedirs(path, exist_ok=True)
        return path

    def _curve_dir(self) -> str:
        path = os.path.join(self._ui3_run_dir(), "curve")
        os.makedirs(path, exist_ok=True)
        return path

    def _curve_group_json_path_for(self, line_id: str) -> str:
        return os.path.join(self._curve_dir(), f"group_{line_id}.json")

    def _curve_nurbs_info_json_path_for(self, line_id: str) -> str:
        return os.path.join(self._curve_dir(), f"nurbs_info_{line_id}.json")

    def _curve_nurbs_png_path_for(self, line_id: str) -> str:
        return os.path.join(self._curve_dir(), f"profile_{line_id}_nurbs.png")

    def _line_id_current(self) -> str:
        # dùng id ổn định để tên file không đụng nhau (ưu tiên tên trong combo)
        if hasattr(self, "line_combo"):
            txt = self.line_combo.currentText().strip() or f"line_{self.line_combo.currentIndex() + 1:03d}"
            return txt.replace(" ", "_")
        # fallback
        row = getattr(self, "line_combo", None).currentIndex() if hasattr(self, "line_combo") else 0
        return f"line_{(row or 0) + 1:03d}"

    def _profile_png_path_for(self, line_id: str) -> str:
        return os.path.join(self._preview_dir(), f"profile_{line_id}.png")

    def _groups_json_path_for(self, line_id: str) -> str:
        return os.path.join(self._groups_dir(), f"{line_id}.json")

    def _vectors_json_path_for(self, line_id: str) -> str:
        return os.path.join(self._vectors_dir(), f"{line_id}.json")

    def _ui2_intersections_json_path(self) -> str:
        run_dir = (self._ctx.get("run_dir") or "").strip()
        if not run_dir:
            raise RuntimeError("[UI3] Run context is empty. Call set_context() first.")
        return os.path.join(run_dir, "ui2", "intersections_main_cross.json")

    def _anchors_xyz_json_path(self) -> str:
        return os.path.join(self._ui3_run_dir(), "anchors_xyz.json")

    @staticmethod
    def _normalize_line_role(line_role: str, line_id: str = "") -> str:
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

    def _line_row_meta(self, row: Optional[int] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"row": -1, "line_id": "", "line_role": ""}
        try:
            ridx = self.line_combo.currentIndex() if row is None else int(row)
        except Exception:
            ridx = -1
        out["row"] = ridx
        try:
            if ridx >= 0 and hasattr(self, "_gdf") and self._gdf is not None and ridx < len(self._gdf):
                g = self._gdf.iloc[ridx]
                line_id = str(g.get("line_id", g.get("name", "")) or "").strip()
                line_role = self._normalize_line_role(str(g.get("line_role", "")) or "", line_id)
                out["line_id"] = line_id
                out["line_role"] = line_role
                out["name"] = str(g.get("name", "") or "").strip()
        except Exception:
            pass
        if not out.get("line_id"):
            out["line_id"] = self._line_id_current()
        if not out.get("line_role"):
            out["line_role"] = self._normalize_line_role("", str(out.get("line_id", "")))
        return out

    def _current_ui2_line_id(self) -> str:
        return str(self._line_row_meta().get("line_id", "") or "")

    def _current_ui2_line_role(self) -> str:
        return str(self._line_row_meta().get("line_role", "") or "")

    def _load_ui2_intersections(self, force: bool = False) -> Dict[str, Any]:
        if (not force) and isinstance(self._ui2_intersections_cache, dict):
            return self._ui2_intersections_cache
        path = ""
        try:
            path = self._ui2_intersections_json_path()
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
            else:
                data = {}
        except Exception:
            data = {}
        items = data.get("items", []) if isinstance(data, dict) else []
        if not isinstance(items, list):
            items = []
        data = dict(data or {})
        data["items"] = items
        data["_path"] = path
        self._ui2_intersections_cache = data
        return data

    def _load_anchors_xyz(self, force: bool = False) -> Dict[str, Any]:
        if (not force) and isinstance(self._anchors_xyz_cache, dict):
            return self._anchors_xyz_cache
        path = ""
        try:
            path = self._anchors_xyz_json_path()
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
            else:
                data = {}
        except Exception:
            data = {}
        items = data.get("items", []) if isinstance(data, dict) else []
        if not isinstance(items, list):
            items = []
        data = dict(data or {})
        data["items"] = items
        data["_path"] = path
        self._anchors_xyz_cache = data
        return data

    def _save_anchors_xyz(self, data: Dict[str, Any]) -> str:
        path = self._anchors_xyz_json_path()
        payload = dict(data or {})
        payload.pop("_path", None)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        payload["_path"] = path
        self._anchors_xyz_cache = payload
        return path

    def _expected_main_line_ids(self) -> List[str]:
        data = self._load_ui2_intersections()
        vals = []
        for it in (data.get("items", []) or []):
            try:
                lid = str(it.get("main_line_id", "")).strip()
            except Exception:
                lid = ""
            if lid and lid not in vals:
                vals.append(lid)
        return vals

    def _anchors_ready_for_cross_constraints(self) -> bool:
        inter = self._load_ui2_intersections()
        inter_items = [it for it in (inter.get("items", []) or []) if str(it.get("status", "")).startswith(("ok", "multi_point"))]
        if not inter_items:
            return False
        expected = sorted({str(it.get("main_line_id", "")).strip() for it in inter_items if str(it.get("main_line_id", "")).strip()})
        if len(expected) < 3:
            return False
        anc = self._load_anchors_xyz()
        anc_items = [it for it in (anc.get("items", []) or []) if it.get("z", None) is not None]
        keys = {
            (
                str(it.get("main_line_id", "")).strip(),
                str(it.get("cross_line_id", "")).strip(),
            )
            for it in anc_items
        }
        for it in inter_items:
            key = (str(it.get("main_line_id", "")).strip(), str(it.get("cross_line_id", "")).strip())
            if key not in keys:
                return False
        saved_mains = sorted({k[0] for k in keys if k[0]})
        return all(m in saved_mains for m in expected[:3]) if expected else False

    def _anchors_for_cross_line(self, cross_line_id: str, require_ready: bool = True) -> List[dict]:
        cross_id = str(cross_line_id or "").strip()
        if not cross_id:
            return []
        if require_ready and not self._anchors_ready_for_cross_constraints():
            return []

        inter = self._load_ui2_intersections()
        anc = self._load_anchors_xyz()
        inter_items = [it for it in (inter.get("items", []) or []) if str(it.get("cross_line_id", "")).strip() == cross_id]
        anc_by_key: Dict[Tuple[str, str], dict] = {}
        for it in (anc.get("items", []) or []):
            m_id = str(it.get("main_line_id", "")).strip()
            c_id = str(it.get("cross_line_id", "")).strip()
            if m_id and c_id:
                anc_by_key[(m_id, c_id)] = it

        out = []
        for inter_it in inter_items:
            m_id = str(inter_it.get("main_line_id", "")).strip()
            rec = anc_by_key.get((m_id, cross_id))
            if not rec:
                continue
            try:
                s_cross = float(rec.get("s_on_cross", inter_it.get("s_on_cross")))
                z = float(rec.get("z"))
                x = float(rec.get("x", inter_it.get("x")))
                y = float(rec.get("y", inter_it.get("y")))
            except Exception:
                continue
            if not (np.isfinite(s_cross) and np.isfinite(z) and np.isfinite(x) and np.isfinite(y)):
                continue
            try:
                main_order = int(rec.get("main_order", inter_it.get("main_order", 999)))
            except Exception:
                main_order = 999
            label = str(rec.get("main_label_fixed", inter_it.get("main_label_fixed", ""))).strip() or f"L{main_order if main_order < 999 else len(out)+1}"
            out.append({
                "main_line_id": m_id,
                "cross_line_id": cross_id,
                "main_order": main_order,
                "main_label_fixed": label,
                "x": x,
                "y": y,
                "z": z,
                "s_on_cross": s_cross,
                "s_on_main": rec.get("s_on_main", inter_it.get("s_on_main")),
            })
        out.sort(key=lambda d: (int(d.get("main_order", 999)), str(d.get("main_line_id", ""))))
        return out

    def _update_anchors_xyz_for_saved_main_curve(self, curve: Dict[str, np.ndarray]) -> Tuple[Optional[str], int]:
        if self._current_ui2_line_role() != "main":
            return None, 0
        main_line_id = self._current_ui2_line_id()
        if not main_line_id:
            return None, 0

        inter = self._load_ui2_intersections(force=True)
        inter_items = []
        for it in (inter.get("items", []) or []):
            try:
                if str(it.get("main_line_id", "")).strip() != main_line_id:
                    continue
                status = str(it.get("status", "")).strip()
                if not status.startswith(("ok", "multi_point")):
                    continue
                s_main = float(it.get("s_on_main"))
                s_cross = float(it.get("s_on_cross"))
                x = float(it.get("x"))
                y = float(it.get("y"))
                if not (np.isfinite(s_main) and np.isfinite(s_cross) and np.isfinite(x) and np.isfinite(y)):
                    continue
                inter_items.append(it)
            except Exception:
                continue
        if not inter_items:
            return None, 0

        ch = np.asarray((curve or {}).get("chain", []), dtype=float)
        zz = np.asarray((curve or {}).get("elev", []), dtype=float)
        m = np.isfinite(ch) & np.isfinite(zz)
        ch = ch[m]
        zz = zz[m]
        if ch.size < 2:
            return None, 0
        o = np.argsort(ch)
        ch = ch[o]
        zz = zz[o]

        data = self._load_anchors_xyz(force=True)
        items = [dict(it) for it in (data.get("items", []) or []) if isinstance(it, dict)]
        index_by_key: Dict[Tuple[str, str], int] = {}
        for i, it in enumerate(items):
            key = (str(it.get("main_line_id", "")).strip(), str(it.get("cross_line_id", "")).strip())
            if key[0] and key[1]:
                index_by_key[key] = i

        try:
            import datetime as _dt
            saved_at = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            saved_at = ""

        updated = 0
        for it in inter_items:
            try:
                s_main = float(it.get("s_on_main"))
                z_val = float(np.interp(s_main, ch, zz))
            except Exception:
                continue
            rec = {
                "main_line_id": str(it.get("main_line_id", "")).strip(),
                "cross_line_id": str(it.get("cross_line_id", "")).strip(),
                "main_row_index": it.get("main_row_index"),
                "cross_row_index": it.get("cross_row_index"),
                "main_label_fixed": str(it.get("main_label_fixed", "")).strip(),
                "main_order": it.get("main_order"),
                "cross_order": it.get("cross_order"),
                "x": float(it.get("x")),
                "y": float(it.get("y")),
                "z": z_val,
                "s_on_main": float(it.get("s_on_main")),
                "s_on_cross": float(it.get("s_on_cross")),
                "status": str(it.get("status", "ok")),
                "saved_at": saved_at,
                "curve_method": "nurbs",
                "source_ui3_line_id": self._line_id_current(),
            }
            key = (rec["main_line_id"], rec["cross_line_id"])
            if key in index_by_key:
                items[index_by_key[key]] = rec
            else:
                index_by_key[key] = len(items)
                items.append(rec)
            updated += 1

        payload = {
            "version": 1,
            "items": items,
            "updated_at": saved_at,
            "source_intersections": inter.get("_path", ""),
        }
        out_path = self._save_anchors_xyz(payload)
        return out_path, updated

    @staticmethod
    def _group_for_chainage(groups: List[dict], chainage: float) -> Tuple[Optional[str], Optional[str]]:
        for g in (groups or []):
            try:
                s = float(g.get("start", g.get("start_chainage", 0.0)))
                e = float(g.get("end", g.get("end_chainage", 0.0)))
                if e < s:
                    s, e = e, s
                if s <= chainage <= e:
                    gid = str(g.get("id", "")) or None
                    color = g.get("color", None)
                    return gid, color
            except Exception:
                continue
        return None, None

    @staticmethod
    def _groups_with_median_theta(groups: List[dict], prof: Optional[dict]) -> List[dict]:
        """Return groups enriched with median vector angle (theta_deg) per chainage range."""
        out: List[dict] = []
        if not groups:
            return out

        chain = np.asarray((prof or {}).get("chain", []), dtype=float)
        theta = np.asarray((prof or {}).get("theta", []), dtype=float)
        n = int(min(chain.size, theta.size))
        if n > 0:
            chain = chain[:n]
            theta = theta[:n]

        for g in groups:
            gg = dict(g or {})
            med_theta = None
            try:
                if n > 0:
                    s = float(gg.get("start", gg.get("start_chainage", 0.0)))
                    e = float(gg.get("end", gg.get("end_chainage", 0.0)))
                    if e < s:
                        s, e = e, s
                    mask = (chain >= s) & (chain <= e)
                    if np.any(mask):
                        vals = theta[mask]
                        vals = vals[np.isfinite(vals)]
                        if vals.size > 0:
                            med_theta = float(np.median(vals))
            except Exception:
                med_theta = None
            gg["median_theta_deg"] = med_theta
            out.append(gg)
        return out

    def _curvature_points_for_json(self, prof: Optional[dict], groups: List[dict]) -> List[dict]:
        out: List[dict] = []
        if not prof:
            return out
        try:
            params = self._grouping_params_current()
            nodes = extract_curvature_rdp_nodes(
                prof,
                rdp_eps_m=float(params.get("rdp_eps_m", 0.5)),
                smooth_radius_m=float(params.get("smooth_radius_m", 0.0)),
                restrict_to_slip_span=False,
            )
            chain = np.asarray(nodes.get("chain", []), dtype=float)
            elev = np.asarray(nodes.get("elev", []), dtype=float)
            curv = np.asarray(nodes.get("curvature", []), dtype=float)
            slip_span = self._profile_slip_span_range(prof)
            if slip_span:
                smin, smax = slip_span
                keep = np.isfinite(chain) & np.isfinite(elev) & np.isfinite(curv) & (chain >= float(smin)) & (chain <= float(smax))
                chain = chain[keep]
                elev = elev[keep]
                curv = curv[keep]
            n = int(min(chain.size, elev.size, curv.size))
            curvature_thr_abs = float(params.get("curvature_thr_abs", 0.02))
            include_curvature = bool(params.get("include_curvature_threshold", True))
            for i in range(n):
                ch = float(chain[i])
                zz = float(elev[i])
                kk = float(curv[i])
                if not (np.isfinite(ch) and np.isfinite(zz) and np.isfinite(kk)):
                    continue
                gid, color = self._group_for_chainage(groups or [], ch)
                out.append({
                    "index": i,
                    "chain_m": ch,
                    "elev_m": zz,
                    "curvature": kk,
                    "curvature_abs": abs(kk),
                    "group_id": gid,
                    "group_color": color,
                    "passes_curvature_threshold": bool(include_curvature and (abs(kk) > curvature_thr_abs)),
                })
        except Exception:
            return []
        return out

    @staticmethod
    def _profile_slip_span_range(prof: Optional[dict]) -> Optional[Tuple[float, float]]:
        chain = np.asarray((prof or {}).get("chain", []), dtype=float)
        if chain.ndim != 1 or chain.size == 0:
            return None

        slip_mask = (prof or {}).get("slip_mask", None)
        if slip_mask is not None:
            try:
                mask = np.asarray(slip_mask)
                if mask.shape == chain.shape:
                    keep = np.isfinite(chain) & (mask == True)
                    if np.any(keep):
                        return float(np.nanmin(chain[keep])), float(np.nanmax(chain[keep]))
            except Exception:
                pass

        slip_span = (prof or {}).get("slip_span", None)
        if slip_span:
            try:
                smin, smax = map(float, slip_span)
                if smax < smin:
                    smin, smax = smax, smin
                return smin, smax
            except Exception:
                pass
        return None

    def _save_vectors_json_for_line(self, line_id: str, prof: dict, groups: Optional[List[dict]]) -> Optional[str]:
        def _to_float(arr, i: int) -> Optional[float]:
            if arr is None:
                return None
            try:
                v = float(arr[i])
            except Exception:
                return None
            return v if np.isfinite(v) else None

        chain = np.asarray(prof.get("chain", []), dtype=float)
        n = int(chain.size)
        if n == 0:
            return None

        x = np.asarray(prof.get("x", []), dtype=float) if prof.get("x", None) is not None else None
        y = np.asarray(prof.get("y", []), dtype=float) if prof.get("y", None) is not None else None
        elev = np.asarray(prof.get("elev", []), dtype=float) if prof.get("elev", None) is not None else None
        elev_s = np.asarray(prof.get("elev_s", []), dtype=float) if prof.get("elev_s", None) is not None else None
        theta = np.asarray(prof.get("theta", []), dtype=float) if prof.get("theta", None) is not None else None
        slip_mask = np.asarray(prof.get("slip_mask", [])) if prof.get("slip_mask", None) is not None else None
        slip_span = self._profile_slip_span_range(prof)
        if slip_span:
            span_min, span_max = slip_span
        else:
            span_min = span_max = None

        rows: List[dict] = []
        for i in range(n):
            ch = _to_float(chain, i)
            if ch is None:
                continue
            if (span_min is not None) and (span_max is not None) and not (float(span_min) <= float(ch) <= float(span_max)):
                continue
            in_slip = None
            if slip_mask is not None and i < slip_mask.size:
                try:
                    in_slip = bool(slip_mask[i])
                except Exception:
                    in_slip = None
            gid = None
            if in_slip is not False:
                gid, _ = self._group_for_chainage(groups or [], ch)
            rows.append({
                "index": i,
                "chain_m": ch,
                "x": _to_float(x, i),
                "y": _to_float(y, i),
                "elev_raw_m": _to_float(elev, i),
                "elev_s_m": _to_float(elev_s, i),
                "theta_deg": _to_float(theta, i),
                "in_slip_zone": in_slip,
                "group_id": gid,
            })

        payload = {
            "line_id": line_id,
            "count": len(rows),
            "chainage_origin": self._ui3_chainage_origin(),
            "profile_dem_source": self._current_profile_source_key(),
            "profile_dem_path": str(getattr(self, "ground_export_dem_path", "") or "").replace("\\", "/"),
            "slip_span": ([float(span_min), float(span_max)] if (span_min is not None and span_max is not None) else None),
            "groups": groups or [],
            "vectors": rows,
        }
        out_json = self._vectors_json_path_for(line_id)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out_json

    def _save_ground_json_for_line(self, line_id: str, geom, step_m: float) -> Optional[str]:
        _ = step_m
        profile_src = self._current_profile_source_key()
        dem_path = str(getattr(self, "ground_export_dem_path", "") or "").strip()
        if not dem_path or not os.path.exists(dem_path):
            if profile_src == "smooth":
                dem_path = self._pick_existing_path(
                    str(getattr(self, "dem_path_smooth", "") or "").strip(),
                    str(getattr(self, "dem_path_raw", "") or "").strip(),
                )
            else:
                dem_path = self._pick_existing_path(
                    str(getattr(self, "dem_path_raw", "") or "").strip(),
                    str(getattr(self, "dem_path_smooth", "") or "").strip(),
                )
        if not dem_path or not os.path.exists(dem_path):
            return None

        prof = compute_profile(
            dem_path,
            self.dx_path,
            self.dy_path,
            self.dz_path,
            geom,
            step_m=float(self._GROUND_EXPORT_STEP_M),
            smooth_win=11,
            smooth_poly=2,
            slip_mask_path=self.slip_path,
            slip_only=False,
        )
        if not prof:
            return None

        chain = np.asarray(prof.get("chain", []), dtype=float)
        x = np.asarray(prof.get("x", []), dtype=float) if prof.get("x", None) is not None else None
        y = np.asarray(prof.get("y", []), dtype=float) if prof.get("y", None) is not None else None
        elev = np.asarray(prof.get("elev_s", []), dtype=float) if prof.get("elev_s", None) is not None else None
        slip_mask = np.asarray(prof.get("slip_mask", [])) if prof.get("slip_mask", None) is not None else None
        if chain.size == 0 or elev is None:
            return None

        slip_start_idx = None
        slip_end_idx = None
        if slip_mask is not None and slip_mask.shape == chain.shape:
            try:
                slip_keep = np.isfinite(chain) & (slip_mask == True)
                slip_idx = np.flatnonzero(slip_keep)
                if slip_idx.size > 0:
                    slip_start_idx = int(slip_idx[0])
                    slip_end_idx = int(slip_idx[-1])
            except Exception:
                slip_start_idx = None
                slip_end_idx = None

        rows: List[dict] = []
        for i in range(int(chain.size)):
            ch = float(chain[i]) if np.isfinite(chain[i]) else None
            zz = float(elev[i]) if i < elev.size and np.isfinite(elev[i]) else None
            if ch is None or zz is None:
                continue
            xx = float(x[i]) if x is not None and i < x.size and np.isfinite(x[i]) else None
            yy = float(y[i]) if y is not None and i < y.size and np.isfinite(y[i]) else None
            in_mask = None
            if slip_mask is not None and i < slip_mask.size:
                try:
                    in_mask = bool(slip_mask[i])
                except Exception:
                    in_mask = None
            rows.append({
                "index": i,
                "chain_m": ch,
                "x": xx,
                "y": yy,
                "ground_m": zz,
                "in_mask": in_mask,
                "is_mask_start": bool(slip_start_idx is not None and i == slip_start_idx),
                "is_mask_end": bool(slip_end_idx is not None and i == slip_end_idx),
            })

        payload = {
            "line_id": line_id,
            "source_dem_path": dem_path.replace("\\", "/"),
            "count": len(rows),
            "chainage_origin": self._ui3_chainage_origin(),
            "profile_dem_source": profile_src,
            "step_m": float(self._GROUND_EXPORT_STEP_M),
            "mask_start_index": slip_start_idx,
            "mask_end_index": slip_end_idx,
            "mask_start_chain_m": (
                float(chain[slip_start_idx])
                if slip_start_idx is not None and 0 <= slip_start_idx < chain.size and np.isfinite(chain[slip_start_idx])
                else None
            ),
            "mask_end_chain_m": (
                float(chain[slip_end_idx])
                if slip_end_idx is not None and 0 <= slip_end_idx < chain.size and np.isfinite(chain[slip_end_idx])
                else None
            ),
            "ground": rows,
        }
        out_json = self._ground_json_path_for(line_id)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out_json

    # tiện gọi ở mọi nơi hiện tại (giữ chữ ký cũ không tham số)
    def _profile_png_path(self) -> str:
        return self._profile_png_path_for(self._line_id_current())

    def _groups_json_path(self) -> str:
        return self._groups_json_path_for(self._line_id_current())

    # def _groups_json_path(self, line_id: str) -> str:
    #     return os.path.join(self._groups_dir(), f"{line_id}.json")

    def _render_current(self) -> None:
        # 1) Kiểm tra dữ liệu tuyến
        if not hasattr(self, "_gdf") or self._gdf is None or self._gdf.empty:
            self._log("[!] No lines.")
            return
        row = self.line_combo.currentIndex()
        if row < 0:
            self._log("[!] Select a line first.")
            return

        # Log resolved inputs for debugging
        self._log(f"[UI3] DEM: {self.dem_path}")
        self._log(f"[UI3] DX:  {self.dx_path}")
        self._log(f"[UI3] DY:  {self.dy_path}")
        self._log(f"[UI3] DZ:  {self.dz_path}")
        self._log(f"[UI3] MASK:{self.slip_path}")

        # 2) Tính profile (ground full line, không giới hạn slip-zone)
        geom = self._gdf.geometry.iloc[row]
        prof = self._compute_profile_for_geom(geom, slip_only=False)
        if not prof:
            self._log("[!] Empty profile.")
            return

        # 3) Lấy nhóm hiện hành (ưu tiên JSON, fallback bảng), rồi clamp theo slip
        groups = self._get_groups_for_current_line()  # <- dùng helper trả list
        if groups:
            groups = clamp_groups_to_slip(prof, groups, min_len=WORKFLOW_GROUP_MIN_LEN_M)

        # 4) Gọi backend vẽ PNG (tô màu theo group nếu có)
        line_id = self._line_id_current()
        out_png = self._profile_png_path_for(line_id)

        msg, path = render_profile_png(
            prof, out_png,
            y_min=None, y_max=None,
            x_min=None, x_max=None,
            vec_scale=self.vscale.value(),
            vec_width=self.vwidth.value(),
            head_len=6.0, head_w=4.0,
            highlight_theta=None,
            group_ranges=groups if groups else None,
            ungrouped_color=self._get_ungrouped_color(),
            curvature_rdp_eps_m=self._current_rdp_eps_m(),
            curvature_smooth_radius_m=self._current_smooth_radius_m(),
        )
        self._log(msg)
        if not path or not os.path.exists(path):
            return

        # 5) Đưa PNG lên scene (luôn cập nhật reference mới)
        QPixmapCache.clear()

        pm = QPixmap(path)
        if pm.isNull():
            self._err("[UI3] Cannot load PNG with curve overlay.")
            return

        self.scene.clear()
        item = QGraphicsPixmapItem(pm)
        self.scene.addItem(item)
        self._img_ground = item
        self._img_rate0 = item
        self._clear_curve_overlay()
        self._load_axes_meta(path)
        self._static_nurbs_bg_loaded = False
        self._active_prof = prof
        self._active_groups = groups if groups else []
        self._active_base_curve = None
        self._active_curve = None
        self._refresh_anchor_overlay()

        # Fit lần đầu
        if getattr(self, "_first_show", True):
            self.view.fit_to_scene()
            self._first_show = False

        try:
            vec_json = self._save_vectors_json_for_line(line_id, prof, groups if groups else [])
            if vec_json:
                self._log(f"[UI3] Saved vectors JSON: {vec_json}")
        except Exception as e:
            self._warn(f"[UI3] Cannot save vectors JSON: {e}")
        try:
            ground_json = self._save_ground_json_for_line(line_id, geom, step_m=float(self.step_box.value()))
            if ground_json:
                self._log(f"[UI3] Saved ground JSON: {ground_json}")
        except Exception as e:
            self._warn(f"[UI3] Cannot save ground JSON: {e}")

        # KHÔNG vẽ overlay guide nữa vì đã vẽ sẵn trong backend

    def _render_current_safe(self) -> None:
        try:
            self._render_current()
        except Exception:
            msg = "[UI3] Render Section failed. See log for details."
            self._append_status(msg)
            log_path = ""
            try:
                run_dir = (self._ctx.get("run_dir") or "").strip()
                if run_dir:
                    log_dir = os.path.join(run_dir, "ui3")
                    os.makedirs(log_dir, exist_ok=True)
                    log_path = os.path.join(log_dir, "ui3_render_error.log")
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(traceback.format_exc())
                        f.write("\n")
            except Exception:
                pass

            try:
                details = msg
                if log_path:
                    details += f"\nLog: {log_path}"
                QMessageBox.critical(self, "UI3 Render Section Error", details)
            except Exception:
                pass

    def _sync_nurbs_defaults_from_group_table(self) -> None:
        """Rebuild default NURBS control points when Group table changes."""
        if self._group_table_updating:
            return
        if not self._active_prof:
            return
        try:
            line_id = self._line_id_current()
            groups = self._read_groups_from_table()
            if groups:
                groups = clamp_groups_to_slip(self._active_prof, groups, min_len=WORKFLOW_GROUP_MIN_LEN_M)
            self._active_groups = groups or []
            params = self._build_default_nurbs_params(
                line_id=line_id,
                prof=self._active_prof,
                groups=self._active_groups,
                base_curve=self._active_base_curve or {},
            )
            self._set_nurbs_params_for_line(line_id, params)
            self._sync_nurbs_panel_for_current_line(reset_defaults=False)
            self._schedule_nurbs_live_update()
        except Exception as e:
            self._warn(f"[UI3] Cannot sync NURBS from groups: {e}")

    def _set_group_chainage_cell(self, row: int, col: int, val: float) -> None:
        item = self.group_table.item(row, col)
        if item is None:
            item = QTableWidgetItem("")
            self.group_table.setItem(row, col, item)
        item.setText(f"{float(val):.3f}")

    def _set_group_boundary_reason(self, row: int, col: int, reason: str) -> None:
        if col not in (1, 2):
            return
        item = self.group_table.item(row, col)
        if item is None:
            item = QTableWidgetItem("")
            self.group_table.setItem(row, col, item)
        item.setData(Qt.UserRole + 1, str(reason or ""))

    def _get_group_boundary_reason(self, row: int, col: int) -> str:
        if col not in (1, 2):
            return ""
        item = self.group_table.item(row, col)
        if item is None:
            return ""
        try:
            return str(item.data(Qt.UserRole + 1) or "").strip()
        except Exception:
            return ""

    def _nearest_group_row(self, row: int, step: int) -> Optional[int]:
        r = int(row) + int(step)
        while 0 <= r < self.group_table.rowCount():
            gid_item = self.group_table.item(r, 0)
            gid = gid_item.text().strip().upper() if gid_item and gid_item.text() else ""
            if gid != "UNGROUPED":
                return r
            r += int(step)
        return None

    def _link_adjacent_group_boundaries(self, row: int, col: int) -> None:
        if col not in (1, 2):
            return
        gid_item = self.group_table.item(row, 0)
        gid = gid_item.text().strip().upper() if gid_item and gid_item.text() else ""
        if gid == "UNGROUPED":
            return
        cur_item = self.group_table.item(row, col)
        txt = cur_item.text().strip() if cur_item and cur_item.text() else ""
        if not txt:
            return
        try:
            val = float(txt)
        except Exception:
            return

        # Keep the edited value normalized.
        self._set_group_chainage_cell(row, col, val)

        # Enforce shared boundary: end(i) == start(i+1).
        if col == 2:  # end changed -> next start
            nxt = self._nearest_group_row(row, +1)
            if nxt is not None:
                self._set_group_chainage_cell(nxt, 1, val)
        elif col == 1:  # start changed -> previous end
            prv = self._nearest_group_row(row, -1)
            if prv is not None:
                self._set_group_chainage_cell(prv, 2, val)

    def _on_group_table_item_changed(self, _item) -> None:
        if self._group_table_updating:
            return
        changed_group_chainage = bool(_item is not None and _item.column() in (1, 2))
        if _item is not None and _item.column() in (1, 2):
            self._group_table_updating = True
            try:
                self._link_adjacent_group_boundaries(_item.row(), _item.column())
            finally:
                self._group_table_updating = False
        self._sync_nurbs_defaults_from_group_table()
        # Re-render immediately so dashed group boundaries update with table edits.
        if changed_group_chainage and self._active_prof:
            self._render_current_safe()

    def _on_add_group(self):
        r = self._find_ungrouped_row()
        if r is None:
            r = self.group_table.rowCount()
        self.group_table.insertRow(r)
        n_groups = len(self._read_groups_from_table()) + 1
        self.group_table.setItem(r, 0, QTableWidgetItem(f"G{n_groups}"))
        self._sync_nurbs_defaults_from_group_table()

    def _on_delete_group(self):
        rows = sorted({i.row() for i in self.group_table.selectedIndexes()}, reverse=True)
        if not rows:
            self._log("[!] Select row(s) to delete.");
            return
        for r in rows:
            gid = self.group_table.item(r, 0)
            if gid and gid.text().strip().upper() == "UNGROUPED":
                continue
            self.group_table.removeRow(r)
        self._sync_nurbs_defaults_from_group_table()


    def _read_groups_from_table(self):
        """Đọc nhóm từ bảng UI (cột: Group ID | Start | End | Color)."""
        rows = self.group_table.rowCount()
        out = []
        for r in range(rows):
            gid = self.group_table.item(r, 0)
            s = self.group_table.item(r, 1)
            e = self.group_table.item(r, 2)
            try:
                gid = gid.text().strip() if gid else f"G{r + 1}"
                if gid.upper() == "UNGROUPED":
                    continue
                s = float(s.text()) if s and s.text() not in ("", None) else None
                e = float(e.text()) if e and e.text() not in ("", None) else None
                if s is None or e is None:
                    continue
                if e < s: s, e = e, s
                out.append({
                    "id": gid,
                    "start": s,
                    "end": e,
                    "start_reason": self._get_group_boundary_reason(r, 1),
                    "end_reason": self._get_group_boundary_reason(r, 2),
                    "color": self._get_color_cell_value(r),
                })
            except Exception:
                continue
        return _renumber_groups_visual_order(out)

    def _find_ungrouped_row(self) -> Optional[int]:
        rows = self.group_table.rowCount()
        for r in range(rows):
            gid = self.group_table.item(r, 0)
            if gid and gid.text().strip().upper() == "UNGROUPED":
                return r
        return None

    def _compute_ungrouped_ranges(self, groups: list, smin: float, smax: float) -> List[tuple]:
        if smin is None or smax is None:
            return []
        if smax <= smin:
            return []
        norm = []
        for g in (groups or []):
            try:
                s = float(g.get("start", 0.0))
                e = float(g.get("end", 0.0))
            except Exception:
                continue
            if e < s:
                s, e = e, s
            s = max(s, smin)
            e = min(e, smax)
            if e > s:
                norm.append((s, e))
        norm.sort(key=lambda x: x[0])

        gaps = []
        cur = smin
        for s, e in norm:
            if s > cur:
                gaps.append((cur, s))
            if e > cur:
                cur = e
        if cur < smax:
            gaps.append((cur, smax))
        return gaps

    def _append_ungrouped_row(self, groups: list, length_m: Optional[float]) -> None:
        r = self._find_ungrouped_row()
        if r is not None:
            self.group_table.removeRow(r)

        if length_m is not None:
            smin, smax = 0.0, float(length_m)
        else:
            starts = [float(g.get("start", 0.0)) for g in (groups or []) if g.get("start", None) is not None]
            ends = [float(g.get("end", 0.0)) for g in (groups or []) if g.get("end", None) is not None]
            if not starts or not ends:
                return
            smin, smax = min(starts), max(ends)

        gaps = self._compute_ungrouped_ranges(groups, smin, smax)
        if not gaps:
            return

        starts = "; ".join([f"{s:.3f}" for s, _ in gaps])
        ends = "; ".join([f"{e:.3f}" for _, e in gaps])

        r = self.group_table.rowCount()
        self.group_table.insertRow(r)
        item_id = QTableWidgetItem("UNGROUPED")
        item_s = QTableWidgetItem(starts)
        item_e = QTableWidgetItem(ends)
        for it in (item_id, item_s, item_e):
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        self.group_table.setItem(r, 0, item_id)
        self.group_table.setItem(r, 1, item_s)
        self.group_table.setItem(r, 2, item_e)
        self._set_color_cell(r, "#bbbbbb")
        color_item = self.group_table.item(r, 3)
        if color_item:
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)

    def _normalize_color_hex(self, color_hex: str) -> str:
        if not color_hex:
            return ""
        c = color_hex.strip()
        if not c:
            return ""
        if not c.startswith("#"):
            c = f"#{c}"
        qc = QColor(c)
        return qc.name() if qc.isValid() else ""

    def _set_color_cell(self, row: int, color_hex: str) -> None:
        c = self._normalize_color_hex(color_hex)
        item = self.group_table.item(row, 3)
        if item is None:
            item = QTableWidgetItem("")
            self.group_table.setItem(row, 3, item)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        if c:
            item.setData(Qt.UserRole, c)
            item.setToolTip(c)
            item.setText("")
            item.setBackground(QColor(c))
        else:
            item.setData(Qt.UserRole, "")
            item.setToolTip("")
            item.setText("")
            item.setBackground(QColor(0, 0, 0, 0))

    def _get_color_cell_value(self, row: int) -> str:
        item = self.group_table.item(row, 3)
        if item is None:
            return ""
        val = item.data(Qt.UserRole)
        if isinstance(val, str) and val.strip():
            return val.strip()
        txt = item.text().strip() if item.text() else ""
        return self._normalize_color_hex(txt)

    def _on_group_cell_double_clicked(self, row: int, col: int) -> None:
        if col != 3:
            return
        current = self._get_color_cell_value(row)
        initial = QColor(current) if current else QColor(255, 255, 255)
        color = QColorDialog.getColor(initial, self, "Select group color")
        if color.isValid():
            self._set_color_cell(row, color.name())
            # Re-render to reflect new group colors on vectors
            self._render_current_safe()

    def _get_ungrouped_color(self) -> str:
        r = self._find_ungrouped_row()
        if r is None:
            return "#bbbbbb"
        c = self._get_color_cell_value(r)
        return c or "#bbbbbb"

    def _load_groups_for_current_line(self):
        """Ưu tiên đọc từ bảng (phản ánh chỉnh sửa/delete mới nhất); nếu trống thì đọc JSON."""
        line_id = self._line_id_current()
        table_groups = self._read_groups_from_table()
        if table_groups:
            return table_groups

        js_path = self._groups_json_path()
        if os.path.exists(js_path):
            try:
                import json
                with open(js_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._apply_group_json_settings(data)
                cm = self._normalize_curve_method(data.get("curve_method"))
                self._set_curve_method_for_line(line_id, cm)
                groups = self._groups_to_current_chainage(
                    data.get("groups", []) or [],
                    source_origin=data.get("chainage_origin"),
                )
                if groups:
                    return groups
            except Exception:
                pass
        return []

    def _on_confirm_groups(self):
        """Lưu JSON nhóm và re-render."""
        if self.line_combo.count() == 0:
            self._log("[!] No line.");
            return
        groups = self._read_group_table()
        prof_for_stats = None
        # nếu trống → auto
        if not groups:
            # cần profile để auto
            row = self.line_combo.currentIndex()
            geom = self._gdf.geometry.iloc[row]
            prof = self._compute_profile_for_geom(geom, slip_only=False)
            if prof:
                groups = auto_group_profile_by_criteria(prof, **self._grouping_params_current())

        # clamp trong slip-zone
        try:
            row = self.line_combo.currentIndex()
            geom = self._gdf.geometry.iloc[row]
            prof = self._compute_profile_for_geom(geom, slip_only=False)
            if prof:
                prof_for_stats = prof
                groups = clamp_groups_to_slip(prof, groups, min_len=WORKFLOW_GROUP_MIN_LEN_M)
        except Exception:
            pass

        # save
        line_id = self._line_id_current()
        curve_method = self._get_curve_method_for_line(line_id)
        groups_for_json = self._groups_with_median_theta(groups, prof_for_stats)
        curvature_points = self._curvature_points_for_json(prof_for_stats, groups_for_json)
        js = {
            "line": self.line_combo.currentText(),
            "groups": groups_for_json,
            "curvature_points": curvature_points,
            "chainage_origin": self._ui3_chainage_origin(),
            "curve_method": curve_method,
            "profile_dem_source": self._current_profile_source_key(),
            "profile_dem_path": str(getattr(self, "dem_path", "") or "").replace("\\", "/"),
            "rdp_eps_m": self._current_rdp_eps_m(),
            "smooth_radius_m": self._current_smooth_radius_m(),
            "include_curvature_threshold": self._include_curvature_threshold(),
            "include_vector_angle_zero": self._include_vector_angle_zero(),
        }
        try:
            with open(self._groups_json_path(), "w", encoding="utf-8") as f:
                json.dump(js, f, ensure_ascii=False, indent=2)
            self._log(f"[✓] Saved groups: {self._groups_json_path()}")
        except Exception as e:
            self._log(f"[!] Save groups failed: {e}")

        # re-render
        self._render_current_safe()
        # --- UPDATE: build bounds + đảm bảo scale + vẽ guides ---
        try:
            line_id = self._line_id_current()

            # 7.1) Thu groups (từ table bạn vừa confirm) → bounds [x0, x1, ...]
            groups = self._read_group_table()  # hoặc đọc lại từ JSON nếu bạn lưu ở trên
            bounds_set = set()
            for g in groups:
                try:
                    s = float(g.get("start", "nan"))
                    e = float(g.get("end", "nan"))
                    if not math.isnan(s) and not math.isnan(e):
                        if e < s: s, e = e, s
                        # clamp vào đoạn [0, _sec_len_m] nếu đã biết
                        if self._sec_len_m:
                            s = max(0.0, min(self._sec_len_m, s))
                            e = max(0.0, min(self._sec_len_m, e))
                        bounds_set.add(s);
                        bounds_set.add(e)
                except Exception:
                    pass
            bounds_m = sorted(bounds_set)
            if bounds_m:
                self._group_bounds[line_id] = bounds_m

            # 7.2) Đảm bảo có px_per_m
            if self._px_per_m is None and getattr(self, "_img_ground", None) and self._sec_len_m:
                W = self._img_ground.pixmap().width()
                if W and self._sec_len_m > 0:
                    self._px_per_m = float(W) / float(self._sec_len_m)

            # 7.3) Vẽ guides (gọi sau khi render)
            # self._draw_group_guides_for_current_line()
            self._ok("[UI3] Groups confirmed and guides updated.")
        except Exception as e:
            self._warn(f"[UI3] Confirm Groups: cannot update guides ({e})")

    # -------------------- Events --------------------
    # def _on_select_line(self, idx: int) -> None:
    #     if not self._sections:
    #         return
    #     self._current_idx = max(0, min(idx, len(self._sections) - 1))
    #     self._log(f"Selected line #{self._current_idx+1}.")
    #     self._draw_placeholder()
    #
    # def _on_auto_generate(self) -> None:
    #
    #     # TODO: gắn thuật toán auto-generate group/curve tại đây
    #     self._info("Auto generate curve – not implemented yet.")
    #
    # def _on_save_json(self) -> None:
    #
    #     # TODO: serialize kết quả đường cong đang hiển thị ra JSON
    #     self._info("Save curve JSON – not implemented yet.")

    # -------------------- Status helpers --------------------
    def _append_status(self, text: str) -> None:
        self.status.append(text)

    def _ok(self, msg: str) -> None:
        self._append_status(f"✅ {msg}")

    def _info(self, msg: str) -> None:
        self._append_status(f"[UI3] INFO: {msg}")

    def _warn(self, msg: str) -> None:
        self._append_status(f"⚠️ {msg}")

    def _err(self, msg: str) -> None:
        self._append_status(f"❌ {msg}")

    def _log(self, msg: str) -> None:
        self._append_status(msg)
