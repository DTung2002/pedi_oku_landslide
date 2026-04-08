import csv
import os
from typing import Optional

import numpy as np

from pedi_oku_landslide.domain.ui3.grouping import extract_curvature_rdp_nodes, filter_rdp_nodes_to_slip_zone


def save_ground_csv_for_line(
    backend,
    *,
    line_id: str,
    geom,
    step_m: float,
    profile_source: str,
    ground_export_step_m: float,
    ground_export_dem_path: str,
    dx_path: str,
    dy_path: str,
    dz_path: str,
    slip_path: str,
    out_csv: str,
) -> Optional[str]:
    dem_path = str(ground_export_dem_path or "").strip()
    if dem_path and not os.path.exists(dem_path):
        dem_path = ""
    if not dem_path or not os.path.exists(dem_path):
        return None

    prof = backend.compute_profile_for_line(
        line_id,
        geom,
        profile_source,
        float(ground_export_step_m),
        False,
        dem_path=dem_path,
        dem_orig_path=dem_path,
        dx_path=dx_path,
        dy_path=dy_path,
        dz_path=dz_path,
        slip_mask_path=slip_path,
    )
    if not prof:
        return None

    chain = np.asarray(prof.get("chain", []), dtype=float)
    elev = np.asarray(prof.get("elev", []), dtype=float) if prof.get("elev", None) is not None else None
    if chain.size == 0 or elev is None:
        return None

    n = min(chain.size, elev.size)
    if n <= 0:
        return None

    chain = chain[:n]
    elev = elev[:n]
    keep = np.isfinite(chain) & np.isfinite(elev)
    if not np.any(keep):
        return None

    rows = [
        (
            f"{float(ch):.1f}",
            f"{float(zz):.10f}".rstrip("0").rstrip("."),
        )
        for ch, zz in zip(chain[keep], elev[keep])
        if np.isfinite(ch) and np.isfinite(zz)
    ]
    if not rows:
        return None

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return out_csv


def save_rdp_csv_for_line(
    *,
    line_id: str,
    prof: dict,
    rdp_eps_m: float,
    smooth_radius_m: float,
    out_csv: str,
) -> Optional[str]:
    if not prof:
        return None

    nodes = extract_curvature_rdp_nodes(
        prof,
        rdp_eps_m=float(rdp_eps_m),
        smooth_radius_m=float(smooth_radius_m),
        restrict_to_slip_span=False,
    )
    chain, elev, curv = filter_rdp_nodes_to_slip_zone(
        prof,
        nodes.get("chain", []),
        nodes.get("elev", []),
        nodes.get("curvature", []),
        rdp_eps_m=float(rdp_eps_m),
        smooth_radius_m=float(smooth_radius_m),
    )
    _ = line_id
    n = int(min(chain.size, elev.size, curv.size))
    if n <= 0:
        return None

    rows = []
    for i in range(n):
        ch = float(chain[i])
        zz = float(elev[i])
        kk = float(curv[i])
        if np.isfinite(ch) and np.isfinite(zz) and np.isfinite(kk):
            rows.append((ch, zz, kk))
    if not rows:
        return None

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chainage", "elevation", "curvature"])
        writer.writerows(rows)
    return out_csv
