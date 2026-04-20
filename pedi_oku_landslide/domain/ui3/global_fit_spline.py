from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


SHORT_LINE_LENGTH_M = 0.1
FIT_PARAMETERIZATION = "chord_length"


def _normalize_groups(groups: Sequence[dict]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, group in enumerate(groups or [], 1):
        try:
            start = float(group.get("start", group.get("start_chainage", np.nan)))
            end = float(group.get("end", group.get("end_chainage", np.nan)))
        except Exception:
            continue
        if not (np.isfinite(start) and np.isfinite(end)):
            continue
        if end < start:
            start, end = end, start
        rows.append(
            {
                "index": int(idx - 1),
                "group_id": str(group.get("id", group.get("group_id", f"G{idx}")) or f"G{idx}"),
                "start": float(start),
                "end": float(end),
            }
        )
    rows.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return rows


def _normalize_theta_rows(theta_rows: Sequence[dict], group_rows: Sequence[dict]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(theta_rows or []):
        try:
            theta_deg = float(row.get("theta_deg", np.nan))
        except Exception:
            continue
        if not np.isfinite(theta_deg):
            continue
        try:
            boundary_chainage = float(row.get("boundary_chainage", np.nan))
        except Exception:
            boundary_chainage = np.nan
        rows.append(
            {
                "group_index": int(idx),
                "group_id": str(row.get("group_id", "") or ""),
                "theta_deg": float(theta_deg),
                "theta_source": str(row.get("theta_source", "") or "").strip(),
                "boundary_chainage": float(boundary_chainage) if np.isfinite(boundary_chainage) else None,
            }
        )
    if len(rows) != len(group_rows):
        raise ValueError(
            f"Theta rows count mismatch: expected {len(group_rows)} groups, found {len(rows)} rows."
        )
    for idx, (theta_row, group_row) in enumerate(zip(rows, group_rows)):
        gid = str(theta_row.get("group_id", "") or "").strip()
        if gid and gid != str(group_row.get("group_id", "")):
            raise ValueError(
                f"Theta row {idx + 1} belongs to '{gid}', expected '{group_row.get('group_id', '')}'."
            )
        theta_row["group_id"] = str(group_row.get("group_id", ""))
    return rows


def _profile_arrays(prof: dict) -> Tuple[np.ndarray, np.ndarray]:
    chain = np.asarray((prof or {}).get("chain", []), dtype=float)
    elev = np.asarray((prof or {}).get("elev_s", []), dtype=float)
    mask = np.isfinite(chain) & np.isfinite(elev)
    chain = chain[mask]
    elev = elev[mask]
    if chain.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(chain)
    chain = chain[order]
    elev = elev[order]
    chain_u, uniq_idx = np.unique(chain, return_index=True)
    elev_u = elev[uniq_idx]
    return chain_u, elev_u


def profile_point_at_chainage(prof: dict, chainage: float) -> Optional[Tuple[float, float]]:
    chain, elev = _profile_arrays(prof)
    if chain.size < 2:
        return None
    s = float(chainage)
    if s < float(chain[0]) - 1e-9 or s > float(chain[-1]) + 1e-9:
        return None
    z = float(np.interp(s, chain, elev))
    if not np.isfinite(z):
        return None
    return float(s), float(z)


def create_auxiliary_fit_point(
    anchor_point: Tuple[float, float],
    theta_deg: float,
    global_end_point: Tuple[float, float],
    *,
    short_length_m: float = SHORT_LINE_LENGTH_M,
) -> Tuple[float, float]:
    x0, z0 = map(float, anchor_point)
    x_toe, _z_toe = map(float, global_end_point)
    theta_rad = np.deg2rad(float(theta_deg))
    dx = float(short_length_m) * float(np.cos(theta_rad))
    dz = float(short_length_m) * float(np.sin(theta_rad))
    toe_sign = 1.0 if x_toe >= x0 else -1.0
    if dx == 0.0:
        dx = toe_sign * float(short_length_m)
    elif np.sign(dx) != np.sign(toe_sign):
        dx = -dx
        dz = -dz
    return float(x0 + dx), float(z0 + dz)


def _fit_parameter_values(fit_points: Sequence[Tuple[float, float]], parameterization: str) -> np.ndarray:
    pts = np.asarray(fit_points, dtype=float)
    n = int(pts.shape[0])
    if n <= 1:
        return np.zeros(n, dtype=float)
    if str(parameterization or "").strip().lower() != FIT_PARAMETERIZATION:
        raise ValueError(f"Unsupported fit parameterization: {parameterization!r}")
    delta = np.diff(pts, axis=0)
    dist = np.hypot(delta[:, 0], delta[:, 1])
    if not np.all(np.isfinite(dist)):
        raise ValueError("Invalid fit-point distances.")
    total = float(np.sum(dist))
    if total <= 1e-12:
        return np.linspace(0.0, 1.0, n, dtype=float)
    params = np.concatenate(([0.0], np.cumsum(dist)))
    return params / float(params[-1])


def _natural_cubic_second_derivatives(params: np.ndarray, values: np.ndarray) -> np.ndarray:
    t = np.asarray(params, dtype=float)
    y = np.asarray(values, dtype=float)
    n = int(min(t.size, y.size))
    if n <= 0:
        return np.array([], dtype=float)
    if n <= 2:
        return np.zeros(n, dtype=float)
    y2 = np.zeros(n, dtype=float)
    u = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        h_prev = float(t[i] - t[i - 1])
        h_next = float(t[i + 1] - t[i])
        span = float(t[i + 1] - t[i - 1])
        if h_prev <= 0.0 or h_next <= 0.0 or span <= 0.0:
            raise ValueError("Fit parameters must be strictly increasing.")
        sig = h_prev / span
        p = sig * y2[i - 1] + 2.0
        y2[i] = (sig - 1.0) / p
        slope_delta = (y[i + 1] - y[i]) / h_next - (y[i] - y[i - 1]) / h_prev
        u[i] = (6.0 * slope_delta / span - sig * u[i - 1]) / p
    for k in range(n - 2, -1, -1):
        y2[k] = y2[k] * y2[k + 1] + u[k]
    return y2


def build_global_fit_spline(
    fit_points: Sequence[Tuple[float, float]],
    *,
    parameterization: str = FIT_PARAMETERIZATION,
) -> Dict[str, Any]:
    pts = [(float(x), float(z)) for x, z in fit_points]
    if len(pts) < 2:
        raise ValueError("Need at least two fit points.")
    fit_params = _fit_parameter_values(pts, parameterization)
    arr = np.asarray(pts, dtype=float)
    second_x = _natural_cubic_second_derivatives(fit_params, arr[:, 0])
    second_z = _natural_cubic_second_derivatives(fit_params, arr[:, 1])
    return {
        "representation": "global_forward_fit_spline",
        "degree": 3,
        "parameterization": FIT_PARAMETERIZATION,
        "boundary_condition": "natural",
        "fit_points": [[float(x), float(z)] for x, z in pts],
        "fit_parameters": fit_params.astype(float).tolist(),
        "components": {
            "x": {"values": arr[:, 0].astype(float).tolist(), "second": second_x.astype(float).tolist()},
            "z": {"values": arr[:, 1].astype(float).tolist(), "second": second_z.astype(float).tolist()},
        },
        "evaluator": "parametric_natural_cubic",
    }


def _spline_piece_value(
    params: np.ndarray,
    values: np.ndarray,
    second: np.ndarray,
    u_val: float,
) -> Tuple[float, float]:
    t = np.asarray(params, dtype=float)
    y = np.asarray(values, dtype=float)
    y2 = np.asarray(second, dtype=float)
    if t.size == 1:
        return float(y[0]), 0.0
    u = float(np.clip(float(u_val), float(t[0]), float(t[-1])))
    hi = int(np.searchsorted(t, u, side="right"))
    hi = min(max(1, hi), t.size - 1)
    lo = hi - 1
    h = float(t[hi] - t[lo])
    if h <= 0.0:
        return float(y[lo]), 0.0
    a = (t[hi] - u) / h
    b = (u - t[lo]) / h
    val = (
        a * y[lo]
        + b * y[hi]
        + ((a**3 - a) * y2[lo] + (b**3 - b) * y2[hi]) * (h * h) / 6.0
    )
    deriv = (y[hi] - y[lo]) / h + ((1.0 - 3.0 * a * a) * y2[lo] + (3.0 * b * b - 1.0) * y2[hi]) * h / 6.0
    return float(val), float(deriv)


def evaluate_global_fit_spline(spline_def: Dict[str, Any], u_val: float) -> Tuple[float, float]:
    params = np.asarray(spline_def.get("fit_parameters", []), dtype=float)
    x_comp = spline_def.get("components", {}).get("x", {})
    z_comp = spline_def.get("components", {}).get("z", {})
    x_val, _dx = _spline_piece_value(
        params,
        np.asarray(x_comp.get("values", []), dtype=float),
        np.asarray(x_comp.get("second", []), dtype=float),
        u_val,
    )
    z_val, _dz = _spline_piece_value(
        params,
        np.asarray(z_comp.get("values", []), dtype=float),
        np.asarray(z_comp.get("second", []), dtype=float),
        u_val,
    )
    return float(x_val), float(z_val)


def evaluate_global_fit_spline_derivative(spline_def: Dict[str, Any], u_val: float) -> Tuple[float, float]:
    params = np.asarray(spline_def.get("fit_parameters", []), dtype=float)
    x_comp = spline_def.get("components", {}).get("x", {})
    z_comp = spline_def.get("components", {}).get("z", {})
    _x_val, dx = _spline_piece_value(
        params,
        np.asarray(x_comp.get("values", []), dtype=float),
        np.asarray(x_comp.get("second", []), dtype=float),
        u_val,
    )
    _z_val, dz = _spline_piece_value(
        params,
        np.asarray(z_comp.get("values", []), dtype=float),
        np.asarray(z_comp.get("second", []), dtype=float),
        u_val,
    )
    return float(dx), float(dz)


def sample_global_fit_spline(spline_def: Dict[str, Any], n_samples: int = 400) -> Dict[str, np.ndarray]:
    count = int(max(32, n_samples))
    u_vals = np.linspace(0.0, 1.0, count)
    pts = np.asarray([evaluate_global_fit_spline(spline_def, float(u)) for u in u_vals], dtype=float)
    return {
        "u": u_vals.astype(float),
        "chain": pts[:, 0].astype(float),
        "elev": pts[:, 1].astype(float),
    }


def locate_vertical_boundary_crossing(
    spline_def: Dict[str, Any],
    boundary_x: float,
    *,
    sample_count: int = 512,
) -> Tuple[float, float]:
    sampled = sample_global_fit_spline(spline_def, n_samples=int(max(64, sample_count)))
    u = np.asarray(sampled.get("u", []), dtype=float)
    x = np.asarray(sampled.get("chain", []), dtype=float)
    if u.size < 2 or x.size != u.size:
        raise ValueError("Spline sampling failed while locating boundary crossing.")
    f = x - float(boundary_x)
    brackets: List[Tuple[float, float]] = []
    tol = 1e-9
    for idx in range(u.size - 1):
        f0 = float(f[idx])
        f1 = float(f[idx + 1])
        u0 = float(u[idx])
        u1 = float(u[idx + 1])
        if abs(f0) <= tol:
            brackets.append((u0, u0))
            continue
        if abs(f1) <= tol:
            brackets.append((u1, u1))
            continue
        if (f0 < 0.0 < f1) or (f1 < 0.0 < f0):
            brackets.append((u0, u1))
    dedup: List[Tuple[float, float]] = []
    for bracket in brackets:
        if not dedup or max(abs(bracket[0] - dedup[-1][0]), abs(bracket[1] - dedup[-1][1])) > 1e-6:
            dedup.append(bracket)
    if len(dedup) != 1:
        raise ValueError(
            f"Boundary x={float(boundary_x):.3f} must intersect spline exactly once; found {len(dedup)}."
        )
    return dedup[0]


def refine_vertical_boundary_intersection(
    spline_def: Dict[str, Any],
    boundary_x: float,
    u_lo: float,
    u_hi: float,
    *,
    max_iter: int = 80,
    tol: float = 1e-9,
) -> Dict[str, float]:
    target = float(boundary_x)
    if abs(float(u_hi) - float(u_lo)) <= tol:
        x_val, z_val = evaluate_global_fit_spline(spline_def, float(u_lo))
        return {"u": float(u_lo), "chain": float(x_val), "elev": float(z_val)}
    lo = float(min(u_lo, u_hi))
    hi = float(max(u_lo, u_hi))
    x_lo, _z_lo = evaluate_global_fit_spline(spline_def, lo)
    x_hi, _z_hi = evaluate_global_fit_spline(spline_def, hi)
    f_lo = float(x_lo - target)
    f_hi = float(x_hi - target)
    if abs(f_lo) <= tol:
        x_val, z_val = evaluate_global_fit_spline(spline_def, lo)
        return {"u": float(lo), "chain": float(x_val), "elev": float(z_val)}
    if abs(f_hi) <= tol:
        x_val, z_val = evaluate_global_fit_spline(spline_def, hi)
        return {"u": float(hi), "chain": float(x_val), "elev": float(z_val)}
    if f_lo * f_hi > 0.0:
        raise ValueError("Refinement bracket does not straddle the target boundary.")
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        x_mid, z_mid = evaluate_global_fit_spline(spline_def, mid)
        f_mid = float(x_mid - target)
        if abs(f_mid) <= tol or abs(hi - lo) <= tol:
            return {"u": float(mid), "chain": float(x_mid), "elev": float(z_mid)}
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    x_mid, z_mid = evaluate_global_fit_spline(spline_def, 0.5 * (lo + hi))
    return {"u": float(0.5 * (lo + hi)), "chain": float(x_mid), "elev": float(z_mid)}


def _curve_sample_count(fit_point_count: int) -> int:
    return int(max(240, 120 * max(1, fit_point_count - 1)))


def build_global_forward_fit_spline(
    prof: dict,
    groups: Sequence[dict],
    theta_rows: Sequence[dict],
    *,
    short_length_m: float = SHORT_LINE_LENGTH_M,
    parameterization: str = FIT_PARAMETERIZATION,
) -> Dict[str, Any]:
    group_rows = _normalize_groups(groups)
    if not group_rows:
        raise ValueError("No valid groups for global fit spline generation.")
    theta_specs = _normalize_theta_rows(theta_rows, group_rows)
    global_start = profile_point_at_chainage(prof, float(group_rows[0]["start"]))
    global_end = profile_point_at_chainage(prof, float(group_rows[-1]["end"]))
    if global_start is None or global_end is None:
        raise ValueError("Cannot resolve slip start/end points on profile.")

    fit_points: List[Tuple[float, float]] = [global_start, global_end]
    steps: List[Dict[str, Any]] = []
    intersections: List[Dict[str, Any]] = []
    spline_def: Optional[Dict[str, Any]] = None

    for idx, (group_row, theta_spec) in enumerate(zip(group_rows, theta_specs)):
        anchor_point = global_start if idx == 0 else (
            float(intersections[-1]["chain"]),
            float(intersections[-1]["elev"]),
        )
        if idx > 0:
            fit_points.insert(-1, anchor_point)
        aux_point = create_auxiliary_fit_point(
            anchor_point,
            float(theta_spec["theta_deg"]),
            global_end,
            short_length_m=float(short_length_m),
        )
        fit_points.insert(-1, aux_point)
        spline_def = build_global_fit_spline(fit_points, parameterization=parameterization)
        sampled = sample_global_fit_spline(spline_def, n_samples=_curve_sample_count(len(fit_points)))
        step: Dict[str, Any] = {
            "group_index": int(idx + 1),
            "group_id": str(group_row.get("group_id", "")),
            "start_point": {"chain": float(anchor_point[0]), "elev": float(anchor_point[1])},
            "aux_fit_point": {"chain": float(aux_point[0]), "elev": float(aux_point[1])},
            "theta_deg": float(theta_spec["theta_deg"]),
            "theta_source": str(theta_spec.get("theta_source", "") or ""),
            "fit_point_count": int(len(fit_points)),
            "fit_points": [{"chain": float(x), "elev": float(z)} for x, z in fit_points],
            "full_curve": {
                "chain": np.asarray(sampled.get("chain", []), dtype=float).tolist(),
                "elev": np.asarray(sampled.get("elev", []), dtype=float).tolist(),
            },
            "boundary_x": None,
            "intersection_point": None,
            "intersection_u": None,
        }
        if idx < len(group_rows) - 1:
            boundary_x = float(group_row["end"])
            u_lo, u_hi = locate_vertical_boundary_crossing(spline_def, boundary_x)
            hit = refine_vertical_boundary_intersection(spline_def, boundary_x, u_lo, u_hi)
            step["boundary_x"] = float(boundary_x)
            step["intersection_u"] = float(hit["u"])
            step["intersection_point"] = {
                "chain": float(hit["chain"]),
                "elev": float(hit["elev"]),
            }
            intersections.append(
                {
                    "group_index": int(idx + 1),
                    "group_id": str(group_row.get("group_id", "")),
                    "boundary_x": float(boundary_x),
                    "u": float(hit["u"]),
                    "chain": float(hit["chain"]),
                    "elev": float(hit["elev"]),
                }
            )
        steps.append(step)

    if spline_def is None:
        raise ValueError("Global fit spline builder produced no spline.")

    sampled_final = sample_global_fit_spline(spline_def, n_samples=_curve_sample_count(len(fit_points)))
    short_lines = [
        {
            "group_index": int(step.get("group_index", 0)),
            "group_id": str(step.get("group_id", "")),
            "start_point": dict(step.get("start_point", {}) or {}),
            "aux_fit_point": dict(step.get("aux_fit_point", {}) or {}),
            "theta_deg": float(step.get("theta_deg", np.nan)),
        }
        for step in steps
    ]
    markers = [
        {
            "group_index": int(hit.get("group_index", 0)),
            "group_id": str(hit.get("group_id", "")),
            "chain": float(hit.get("chain", np.nan)),
            "elev": float(hit.get("elev", np.nan)),
            "boundary_x": float(hit.get("boundary_x", np.nan)),
        }
        for hit in intersections
    ]
    return {
        "curve_method": "global_fit_spline",
        "representation": "global_forward_fit_spline",
        "short_length_m": float(short_length_m),
        "fit_parameterization": FIT_PARAMETERIZATION,
        "global_start_point": {"chain": float(global_start[0]), "elev": float(global_start[1])},
        "global_end_point": {"chain": float(global_end[0]), "elev": float(global_end[1])},
        "fit_points": [{"chain": float(x), "elev": float(z)} for x, z in fit_points],
        "global_fit_points": [{"chain": float(x), "elev": float(z)} for x, z in fit_points],
        "theta_rows": [
            {
                "group_index": int(row["group_index"]),
                "group_id": str(row.get("group_id", "")),
                "theta_deg": float(row["theta_deg"]),
                "theta_source": str(row.get("theta_source", "") or ""),
                "boundary_chainage": row.get("boundary_chainage"),
            }
            for row in theta_specs
        ],
        "steps": steps,
        "short_lines": short_lines,
        "markers": markers,
        "boundary_intersections": intersections,
        "spline_definition": spline_def,
        "curve": {
            "chain": np.asarray(sampled_final.get("chain", []), dtype=float),
            "elev": np.asarray(sampled_final.get("elev", []), dtype=float),
        },
    }
