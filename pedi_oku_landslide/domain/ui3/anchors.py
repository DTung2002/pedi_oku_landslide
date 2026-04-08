import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_json_items(path: str) -> Dict[str, Any]:
    try:
        if path and os.path.exists(path):
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
    return data


def save_json_items(path: str, data: Dict[str, Any]) -> str:
    payload = dict(data or {})
    payload.pop("_path", None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def anchors_ready_for_cross_constraints(intersections: Dict[str, Any], anchors: Dict[str, Any]) -> bool:
    inter_items = [it for it in (intersections.get("items", []) or []) if str(it.get("status", "")).startswith(("ok", "multi_point"))]
    if not inter_items:
        return False
    expected = sorted({str(it.get("main_line_id", "")).strip() for it in inter_items if str(it.get("main_line_id", "")).strip()})
    if len(expected) < 3:
        return False
    anc_items = [it for it in (anchors.get("items", []) or []) if it.get("z", None) is not None]
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


def anchors_for_cross_line(
    intersections: Dict[str, Any],
    anchors: Dict[str, Any],
    cross_line_id: str,
    *,
    require_ready: bool = True,
) -> List[dict]:
    cross_id = str(cross_line_id or "").strip()
    if not cross_id:
        return []
    if require_ready and not anchors_ready_for_cross_constraints(intersections, anchors):
        return []

    inter_items = [it for it in (intersections.get("items", []) or []) if str(it.get("cross_line_id", "")).strip() == cross_id]
    anc_by_key: Dict[Tuple[str, str], dict] = {}
    for it in (anchors.get("items", []) or []):
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


def extend_endpoint_targets_with_cross_anchors(
    prof: dict,
    endpoints: Optional[Tuple[float, float, float, float]],
    anchors: List[dict],
) -> Optional[Tuple[float, float, float, float]]:
    if endpoints is None or not anchors:
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


def constrain_curve_to_cross_anchors(curve: Optional[Dict[str, np.ndarray]], anchors: List[dict]) -> Optional[Dict[str, np.ndarray]]:
    if not curve or len(anchors) < 3:
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

    ch_aug = np.unique(np.concatenate([ch, a_s]))
    if ch_aug.size < 2:
        return curve
    base_zz = np.interp(ch_aug, ch, zz)

    base_at_anchor = np.interp(a_s, ch_aug, base_zz)
    residual = a_z - base_at_anchor
    node_x = np.concatenate([[float(ch_aug[0])], a_s, [float(ch_aug[-1])]])
    node_r = np.concatenate([[0.0], residual, [0.0]])
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
        zz_adj[hit] = z
    return {"chain": ch_aug, "elev": zz_adj}


def update_anchors_for_saved_main_curve(
    *,
    curve: Dict[str, np.ndarray],
    intersections: Dict[str, Any],
    existing_anchors: Dict[str, Any],
    main_line_id: str,
) -> Tuple[Dict[str, Any], int]:
    inter_items = []
    for it in (intersections.get("items", []) or []):
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
        return existing_anchors, 0

    ch = np.asarray((curve or {}).get("chain", []), dtype=float)
    zz = np.asarray((curve or {}).get("elev", []), dtype=float)
    m = np.isfinite(ch) & np.isfinite(zz)
    ch = ch[m]
    zz = zz[m]
    if ch.size < 2:
        return existing_anchors, 0
    o = np.argsort(ch)
    ch = ch[o]
    zz = zz[o]

    items = [dict(it) for it in (existing_anchors.get("items", []) or []) if isinstance(it, dict)]
    index_by_key: Dict[Tuple[str, str], int] = {}
    for i, it in enumerate(items):
        key = (str(it.get("main_line_id", "")).strip(), str(it.get("cross_line_id", "")).strip())
        if key[0] and key[1]:
            index_by_key[key] = i

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
    }
    return payload, updated
