"""UI4 variogram fitting, ordinary kriging solver, and prediction."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .types import (
    cdist,
    curve_fit,
    lu_factor,
    lu_solve,
)


def exp_variogram(h: np.ndarray, nugget: float, sill: float, rang: float) -> np.ndarray:
    rang = np.maximum(rang, 1e-6)
    return nugget + sill * (1.0 - np.exp(-h / rang))


def fit_exponential_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    pairs: int,
    bins_count: int,
    min_pairs_per_bin: int,
    percentile_max_h: float,
    random_seed: int,
) -> Dict[str, Any]:
    n = int(values.size)
    if n < 3:
        raise ValueError("Need at least 3 points for variogram fitting.")

    rng = np.random.default_rng(int(random_seed))
    pairs = int(max(1000, pairs))
    i = rng.integers(0, n, size=pairs)
    j = rng.integers(0, n, size=pairs)
    m = i != j
    i = i[m]
    j = j[m]
    if i.size < 10:
        raise ValueError("Not enough point pairs for variogram.")

    h = np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
    gamma = 0.5 * (values[i] - values[j]) ** 2

    valid_hg = np.isfinite(h) & np.isfinite(gamma)
    h = h[valid_hg]
    gamma = gamma[valid_hg]
    if h.size < 10:
        raise ValueError("Invalid pair distances/semivariance values.")

    hmax = float(np.percentile(h, float(percentile_max_h)))
    if not np.isfinite(hmax) or hmax <= 0:
        hmax = float(np.nanmax(h)) if np.isfinite(np.nanmax(h)) else 1.0
    bins_count = int(max(5, bins_count))
    bins = np.linspace(0.0, hmax, bins_count)

    bin_centers = []
    gamma_means = []
    counts = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mm = (h >= b0) & (h < b1)
        c = int(np.count_nonzero(mm))
        if c >= int(min_pairs_per_bin):
            bin_centers.append(0.5 * (b0 + b1))
            gamma_means.append(float(np.mean(gamma[mm])))
            counts.append(c)

    bin_centers_arr = np.asarray(bin_centers, dtype=float)
    gamma_means_arr = np.asarray(gamma_means, dtype=float)

    vvar = float(np.nanvar(values))
    if not np.isfinite(vvar) or vvar <= 0:
        vvar = max(float(np.nanmean(np.abs(values))) ** 2, 1e-6)

    # Fallback when fit data is too sparse.
    if bin_centers_arr.size < 3:
        span = float(np.nanmax(h) - np.nanmin(h)) if h.size else 1.0
        rang = max(1e-3, span / 3.0)
        params = np.array([0.05 * vvar, 0.95 * vvar, rang], dtype=float)
        return {
            "params": params,
            "nugget": float(params[0]),
            "sill": float(params[1]),
            "range": float(params[2]),
            "bin_centers": bin_centers_arr,
            "gamma_means": gamma_means_arr,
            "bin_counts": np.asarray(counts, dtype=int),
            "fit_method": "fallback_sparse",
        }

    p0 = [0.05 * vvar, 0.95 * vvar, max(1e-3, float(np.max(bin_centers_arr)) / 3.0)]
    try:
        params, _ = curve_fit(
            exp_variogram,
            bin_centers_arr,
            gamma_means_arr,
            p0=p0,
            bounds=([0.0, 0.0, 1e-3], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
        fit_method = "curve_fit"
    except Exception:
        params = np.asarray(p0, dtype=float)
        fit_method = "fallback_p0"

    params = np.asarray(params, dtype=float)
    return {
        "params": params,
        "nugget": float(params[0]),
        "sill": float(params[1]),
        "range": float(params[2]),
        "bin_centers": bin_centers_arr,
        "gamma_means": gamma_means_arr,
        "bin_counts": np.asarray(counts, dtype=int),
        "fit_method": fit_method,
    }


def build_ok_solver(coords: np.ndarray, values: np.ndarray, params: np.ndarray) -> Dict[str, Any]:
    n = int(values.size)
    if n < 2:
        raise ValueError("Need at least 2 points for ordinary kriging.")

    D = cdist(coords, coords)
    Gamma = exp_variogram(D, *params)
    np.fill_diagonal(Gamma, 0.0)

    K = np.empty((n + 1, n + 1), dtype=float)
    K[:n, :n] = Gamma
    K[:n, n] = 1.0
    K[n, :n] = 1.0
    K[n, n] = 0.0

    lu, piv = lu_factor(K)
    return {
        "coords": coords,
        "values": values,
        "params": np.asarray(params, dtype=float),
        "n": n,
        "lu": lu,
        "piv": piv,
    }


def ok_predict(solver: Dict[str, Any], points_xy: np.ndarray, chunk: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    coords = solver["coords"]
    values = solver["values"]
    params = solver["params"]
    n = int(solver["n"])
    lu = solver["lu"]
    piv = solver["piv"]

    points_xy = np.asarray(points_xy, dtype=float)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must be shape (m, 2)")
    m = int(points_xy.shape[0])
    preds = np.empty(m, dtype=float)
    vars_ = np.empty(m, dtype=float)
    chunk = int(max(1, chunk))

    for s in range(0, m, chunk):
        e = min(m, s + chunk)
        P = points_xy[s:e]
        d = cdist(coords, P)                   # (n, chunk)
        g0 = exp_variogram(d, *params)         # (n, chunk)
        rhs = np.vstack([g0, np.ones((1, e - s), dtype=float)])  # (n+1, chunk)
        sol = lu_solve((lu, piv), rhs)
        w = sol[:n, :]
        mu = sol[n, :]
        preds[s:e] = w.T @ values
        vars_[s:e] = np.sum(w * g0, axis=0) + mu
    return preds, vars_
