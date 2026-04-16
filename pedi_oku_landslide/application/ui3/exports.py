import csv
import os
from typing import Optional

import numpy as np

from pedi_oku_landslide.application.ui3.profile_sampling import (
    parse_nominal_length_m,
    resample_profile_to_nominal_grid,
)
from pedi_oku_landslide.domain.ui3.curve_state import (
    median_displacement_theta_deg_for_group,
    start_theta_deg_for_cp1,
)
from pedi_oku_landslide.domain.ui3.grouping import extract_curvature_rdp_nodes, filter_rdp_nodes_to_slip_zone


def _format_csv_number(val: float) -> str:
    return f"{float(val):.10f}".rstrip("0").rstrip(".")


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

    nominal_length_m = parse_nominal_length_m(line_id)
    if nominal_length_m is None:
        try:
            nominal_length_m = round(float(getattr(geom, "length", np.nan)), 1)
        except Exception:
            nominal_length_m = None
    prof = resample_profile_to_nominal_grid(
        prof,
        line_id=line_id,
        target_step_m=float(ground_export_step_m),
        nominal_length_m=nominal_length_m,
    )

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


def save_theta_csv_for_line(
    *,
    line_id: str,
    prof: dict,
    groups: Optional[list],
    out_csv: str,
) -> Optional[str]:
    if not prof:
        return None
    _ = line_id

    group_rows = []
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
        group_rows.append({
            "id": str(g.get("id", g.get("group_id", "")) or "").strip(),
            "start": float(s),
            "end": float(e),
        })
    group_rows.sort(key=lambda it: (float(it["start"]), float(it["end"])))
    if len(group_rows) < 2:
        return None

    rows = []
    start_chainage = float(group_rows[0]["start"])
    start_theta = start_theta_deg_for_cp1(
        prof,
        start_chainage,
        percentile=20.0,
        neighbors_each_side=5,
    )
    first_boundary = float(group_rows[0]["end"])
    rows.append((
        "CP1",
        float(first_boundary),
        float(start_theta) if start_theta is not None and np.isfinite(start_theta) else None,
        "theta_percentile_20",
        "",
    ))

    for cp_idx in range(2, len(group_rows)):
        group = group_rows[cp_idx - 1]
        boundary_chainage = float(group_rows[cp_idx - 1]["end"])
        theta_deg = median_displacement_theta_deg_for_group(group, prof)
        rows.append((
            f"CP{cp_idx}",
            float(boundary_chainage),
            float(theta_deg) if theta_deg is not None and np.isfinite(theta_deg) else None,
            "median_theta_vector",
            str(group.get("id", "") or ""),
        ))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["control_point", "boundary_chainage", "theta_deg", "theta_source", "group_id"])
        writer.writerows(rows)
    return out_csv


def save_vectors_csv_for_line(
    *,
    line_id: str,
    prof: dict,
    out_csv: str,
) -> Optional[str]:
    _ = line_id
    if not prof:
        return None

    chain = np.asarray(prof.get("chain", []), dtype=float)
    elev = np.asarray(prof.get("elev_s", []), dtype=float)
    d_para = np.asarray(prof.get("d_para", []), dtype=float)
    dz = np.asarray(prof.get("dz", []), dtype=float)
    theta = np.asarray(prof.get("theta", []), dtype=float)

    n = min(chain.size, elev.size, d_para.size, dz.size, theta.size)
    if n <= 0:
        return None

    rows = []
    for i in range(n):
        s = chain[i]
        z = elev[i]
        dp = d_para[i]
        dzv = dz[i]
        th = theta[i]
        if np.isfinite(s) and np.isfinite(z) and np.isfinite(dp) and np.isfinite(dzv):
            rows.append((
                float(s),
                float(z),
                float(dp),
                float(dzv),
                float(th) if np.isfinite(th) else None,
            ))
    if not rows:
        return None

    rows.sort(key=lambda it: float(it[0]))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chainage", "elevation", "d_para", "dz", "theta_deg"])
        writer.writerows(rows)
    return out_csv
