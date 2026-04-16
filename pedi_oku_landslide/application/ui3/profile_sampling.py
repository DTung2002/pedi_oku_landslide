import re
from typing import Optional

import numpy as np


def parse_nominal_length_m(line_id: str) -> Optional[float]:
    match = re.search(r"\(([-+]?\d+(?:\.\d+)?)_m\)", str(line_id or ""))
    if not match:
        return None
    try:
        length_m = float(match.group(1))
    except Exception:
        return None
    return length_m if np.isfinite(length_m) and length_m > 0.0 else None


def fixed_chainage_grid(length_m: float, step_m: float) -> np.ndarray:
    length_m = float(length_m)
    step_m = float(step_m)
    if not (np.isfinite(length_m) and length_m > 0.0 and np.isfinite(step_m) and step_m > 0.0):
        return np.array([], dtype=float)

    tol = max(1e-9, abs(step_m) * 1e-6)
    count = int(np.floor((length_m + tol) / step_m))
    chain = np.arange(count + 1, dtype=float) * step_m
    if chain.size == 0 or abs(float(chain[-1]) - length_m) > tol:
        chain = np.append(chain, length_m)
    chain = np.round(chain, 10)
    keep = np.ones(chain.shape, dtype=bool)
    if chain.size >= 2:
        keep[1:] = np.abs(np.diff(chain)) > tol
    return chain[keep]


def _interp_numeric_array(source_chain: np.ndarray, values: np.ndarray, target_chain: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size != source_chain.size:
        return values
    keep = np.isfinite(source_chain) & np.isfinite(values)
    if int(np.count_nonzero(keep)) < 2:
        return np.full(target_chain.shape, np.nan, dtype=float)
    xs = np.asarray(source_chain[keep], dtype=float)
    ys = np.asarray(values[keep], dtype=float)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    xs_u, uniq_idx = np.unique(xs, return_index=True)
    ys_u = ys[uniq_idx]
    if xs_u.size < 2:
        return np.full(target_chain.shape, np.nan, dtype=float)
    interp_mask = (target_chain >= float(xs_u[0])) & (target_chain <= float(xs_u[-1]))
    out = np.full(target_chain.shape, np.nan, dtype=float)
    out[interp_mask] = np.interp(target_chain[interp_mask], xs_u, ys_u)
    return out


def _resample_slip_mask(source_chain: np.ndarray, slip_mask, target_chain: np.ndarray):
    if slip_mask is None:
        return None
    mask = np.asarray(slip_mask)
    if mask.ndim != 1 or mask.size != source_chain.size:
        return None
    keep = np.isfinite(source_chain)
    if int(np.count_nonzero(keep)) < 1:
        return None
    xs = np.asarray(source_chain[keep], dtype=float)
    vals = np.asarray(mask[keep] == True, dtype=float)
    order = np.argsort(xs)
    xs = xs[order]
    vals = vals[order]
    xs_u, uniq_idx = np.unique(xs, return_index=True)
    vals_u = vals[uniq_idx]
    if xs_u.size <= 0:
        return None
    nearest_idx = np.searchsorted(xs_u, target_chain, side="left")
    nearest_idx = np.clip(nearest_idx, 0, max(0, xs_u.size - 1))
    prev_idx = np.clip(nearest_idx - 1, 0, max(0, xs_u.size - 1))
    choose_prev = np.abs(target_chain - xs_u[prev_idx]) <= np.abs(target_chain - xs_u[nearest_idx])
    nearest_idx = np.where(choose_prev, prev_idx, nearest_idx)
    return (vals_u[nearest_idx] >= 0.5)


def resample_profile_to_nominal_grid(
    prof: dict,
    *,
    line_id: str,
    target_step_m: float,
    nominal_length_m: Optional[float] = None,
) -> dict:
    if not prof:
        return prof
    source_chain = np.asarray((prof or {}).get("chain", []), dtype=float)
    if source_chain.ndim != 1 or source_chain.size < 2:
        return prof

    length_m = nominal_length_m
    if length_m is None:
        length_m = parse_nominal_length_m(line_id)
    if length_m is None:
        finite_chain = source_chain[np.isfinite(source_chain)]
        if finite_chain.size >= 2:
            length_m = float(np.nanmax(finite_chain))
    if length_m is None or not np.isfinite(length_m) or length_m <= 0.0:
        return prof

    target_chain = fixed_chainage_grid(float(length_m), float(target_step_m))
    if target_chain.size < 2:
        return prof
    finite_source = source_chain[np.isfinite(source_chain)]
    if finite_source.size >= 2:
        target_keep = (target_chain >= float(np.nanmin(finite_source))) & (target_chain <= float(np.nanmax(finite_source)))
        target_chain = target_chain[target_keep]
    if target_chain.size < 2:
        return prof

    out = dict(prof)
    out["chain"] = np.asarray(target_chain, dtype=float)

    for key in ("x", "y", "elev", "elev_s", "elev_orig", "dx", "dy", "dz", "d_para", "theta"):
        arr = np.asarray((prof or {}).get(key, []), dtype=float)
        if arr.ndim == 1 and arr.size == source_chain.size:
            out[key] = _interp_numeric_array(source_chain, arr, target_chain)

    resampled_mask = _resample_slip_mask(source_chain, (prof or {}).get("slip_mask", None), target_chain)
    out["slip_mask"] = resampled_mask
    if resampled_mask is not None and np.any(resampled_mask):
        keep = np.isfinite(target_chain) & (resampled_mask == True)
        out["slip_span"] = (float(np.nanmin(target_chain[keep])), float(np.nanmax(target_chain[keep])))
    else:
        elev_s = np.asarray(out.get("elev_s", []), dtype=float)
        keep = np.isfinite(target_chain) & np.isfinite(elev_s)
        if np.any(keep):
            out["slip_span"] = (float(np.nanmin(target_chain[keep])), float(np.nanmax(target_chain[keep])))

    return out
