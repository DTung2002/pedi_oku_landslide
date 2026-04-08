"""UI4 surface building: curve point loading, DEM sampling, depth computation."""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .types import (
    _log,
    _require_ui4_runtime_deps,
    _safe_float,
    pd,
    rasterio,
)


def decimate_by_chainage(df, ds: float = 1.0):
    if df.empty:
        return df.copy()
    out = df.sort_values("chainage_m").copy()
    keep_idx = []
    last = -1e18
    for i, r in out.iterrows():
        ch = _safe_float(r.get("chainage_m"))
        if not np.isfinite(ch):
            continue
        if ch - last >= float(ds):
            keep_idx.append(i)
            last = ch
    if not keep_idx:
        return out.iloc[0:0].copy()
    return out.loc[keep_idx].copy()


def _curve_json_to_df(path: str):
    _require_ui4_runtime_deps()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = data.get("points", [])
    if not isinstance(pts, list):
        return pd.DataFrame(columns=["line_id", "index", "chainage_m", "x", "y", "z", "source_json"])

    rows: List[Dict[str, Any]] = []
    line_id = str(data.get("line_id") or os.path.splitext(os.path.basename(path))[0])
    for k, p in enumerate(pts):
        if not isinstance(p, dict):
            continue
        ch = p.get("chainage_m", p.get("chain"))
        z = p.get("z", p.get("elev_m", p.get("elev")))
        rows.append(
            {
                "line_id": line_id,
                "index": p.get("index", k),
                "chainage_m": _safe_float(ch),
                "x": _safe_float(p.get("x")),
                "y": _safe_float(p.get("y")),
                "z": _safe_float(z),
                "source_json": os.path.abspath(path),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["line_id", "index", "chainage_m", "x", "y", "z", "source_json"])

    df = pd.DataFrame(rows)
    for col in ("chainage_m", "x", "y", "z"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[np.isfinite(df["x"]) & np.isfinite(df["y"]) & np.isfinite(df["z"])].copy()
    if "chainage_m" not in df or df["chainage_m"].isna().all():
        # Fallback: build chainage from cumulative XY distance if missing.
        xy = df[["x", "y"]].to_numpy(dtype=float)
        if len(xy) > 0:
            d = np.zeros(len(xy), dtype=float)
            if len(xy) > 1:
                d[1:] = np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))
            df["chainage_m"] = d
    return df[["line_id", "index", "chainage_m", "x", "y", "z", "source_json"]].copy()


def load_ui4_curve_points(
    json_paths: List[str],
    chainage_step_m: float = 1.0,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[Any, Any]:
    _require_ui4_runtime_deps()
    dfs = []
    for p in json_paths:
        try:
            df = _curve_json_to_df(p)
        except Exception as e:
            _log(log_fn, f"[UI4] Skip invalid curve JSON: {p} ({e})")
            continue
        if df.empty:
            _log(log_fn, f"[UI4] Empty curve JSON: {p}")
            continue
        dfs.append(df)

    if not dfs:
        empty = pd.DataFrame(columns=["line_id", "index", "chainage_m", "x", "y", "z", "source_json"])
        return empty.copy(), empty.copy()

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df[
        np.isfinite(all_df["x"]) &
        np.isfinite(all_df["y"]) &
        np.isfinite(all_df["z"]) &
        np.isfinite(all_df["chainage_m"])
    ].copy()

    dec_list = []
    for lid, g in all_df.groupby("line_id", sort=True):
        dec = decimate_by_chainage(g, ds=float(chainage_step_m))
        if dec.empty and not g.empty:
            dec = g.sort_values("chainage_m").iloc[[0]].copy()
        dec_list.append(dec)
        _log(log_fn, f"[UI4] Curve {lid}: raw={len(g)} decimated={len(dec)}")

    dec_df = pd.concat(dec_list, ignore_index=True) if dec_list else all_df.iloc[0:0].copy()
    return all_df, dec_df


def sample_dem_and_compute_depth(
    dem_path: str,
    dec_df,
    duplicate_round_decimals: int = 3,
) -> Tuple[Any, Dict[str, Any]]:
    if dec_df.empty:
        return dec_df.copy(), {"n_input": 0, "n_valid_dem": 0, "n_after_dedupe": 0}

    work = dec_df.copy()
    with rasterio.open(dem_path) as src:
        samples = np.array([v[0] for v in src.sample(list(zip(work["x"], work["y"])))], dtype=float)
        nodata = src.nodata
        if nodata is not None:
            if np.isnan(nodata):
                samples[~np.isfinite(samples)] = np.nan
            else:
                samples[np.isclose(samples, float(nodata))] = np.nan
        work["z_dem"] = samples

    work = work[np.isfinite(work["z_dem"])].copy()
    # Keep raw depth (can be negative) for variogram/kriging stability and structure.
    # Final raster depth is clipped to non-negative after prediction.
    work["depth"] = (work["z_dem"] - work["z"])

    # Merge near-duplicate XY after depth computation to avoid singular kriging matrix.
    decs = int(max(0, int(duplicate_round_decimals)))
    work["_xk"] = np.round(work["x"].to_numpy(dtype=float), decs)
    work["_yk"] = np.round(work["y"].to_numpy(dtype=float), decs)
    dedup = (
        work.groupby(["_xk", "_yk"], as_index=False)
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            z=("z", "mean"),
            z_dem=("z_dem", "mean"),
            depth=("depth", "mean"),
            chainage_m=("chainage_m", "mean"),
            line_id=("line_id", "first"),
        )
    )
    dedup.drop(columns=["_xk", "_yk"], inplace=True, errors="ignore")

    stats = {
        "n_input": int(len(dec_df)),
        "n_valid_dem": int(len(work)),
        "n_after_dedupe": int(len(dedup)),
    }
    return dedup, stats
