import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.colors as mcolors
import numpy as np

from pedi_oku_landslide.domain.ui3.grouping import effective_profile_slip_span_range, extract_curvature_rdp_nodes

matplotlib.use("Agg")

CURVATURE_THRESHOLD_PLOT_ABS = 0.02


def _rdp_polyline(points, eps):
    if len(points) <= 2:
        return points
    x1, y1 = points[0]
    x2, y2 = points[-1]
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    idx, dmax = 0, -1.0
    for i, (x0, y0) in enumerate(points[1:-1], start=1):
        if L2 == 0:
            d = math.hypot(x0 - x1, y0 - y1)
        else:
            d = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / math.sqrt(L2)
        if d > dmax:
            idx, dmax = i, d
    if dmax > eps:
        left = _rdp_polyline(points[: idx + 1], eps)
        right = _rdp_polyline(points[idx:], eps)
        return left[:-1] + right
    return [points[0], points[-1]]


def _curvature_series(x, y):
    n = len(x)
    k = [0.0] * n
    for i in range(1, n - 1):
        x1, y1 = x[i - 1], y[i - 1]
        x2, y2 = x[i], y[i]
        x3, y3 = x[i + 1], y[i + 1]
        a = math.hypot(x2 - x3, y2 - y3)
        b = math.hypot(x3 - x1, y3 - y1)
        c = math.hypot(x1 - x2, y1 - y2)
        s = 0.5 * (a + b + c)
        area2 = max(s * (s - a) * (s - b) * (s - c), 0.0)
        if area2 <= 0:
            kk = 0.0
        else:
            R = (a * b * c) / (4.0 * math.sqrt(area2))
            kk = 0.0 if R == 0 else 1.0 / R
            cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
            if cross < 0:
                kk = -kk
        k[i] = kk
    return k


def _clip_polyline_to_span(xs: np.ndarray, ys: np.ndarray, smin: float, smax: float) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.ndim != 1 or ys.ndim != 1 or xs.size != ys.size or xs.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    if not (np.isfinite(smin) and np.isfinite(smax)):
        return np.array([], dtype=float), np.array([], dtype=float)
    if smax < smin:
        smin, smax = smax, smin
    out_x: List[float] = []
    out_y: List[float] = []
    for i in range(xs.size - 1):
        x0 = float(xs[i]); x1 = float(xs[i + 1])
        y0 = float(ys[i]); y1 = float(ys[i + 1])
        if not (np.isfinite(x0) and np.isfinite(x1) and np.isfinite(y0) and np.isfinite(y1)):
            continue
        if x1 == x0:
            continue
        seg_lo = max(min(x0, x1), float(smin))
        seg_hi = min(max(x0, x1), float(smax))
        if seg_hi < seg_lo:
            continue
        def _interp(xq: float) -> float:
            t = (float(xq) - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))
        xa = float(seg_lo); xb = float(seg_hi)
        ya = _interp(xa); yb = _interp(xb)
        if not out_x or abs(out_x[-1] - xa) > 1e-9:
            out_x.append(xa); out_y.append(ya)
        out_x.append(xb); out_y.append(yb)
    if len(out_x) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.asarray(out_x, dtype=float), np.asarray(out_y, dtype=float)


def _curvature_plot_series(prof: Dict[str, np.ndarray], *, rdp_eps_m: float = 0.5, smooth_radius_m: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    nodes = extract_curvature_rdp_nodes(
        prof,
        rdp_eps_m=float(rdp_eps_m),
        smooth_radius_m=float(smooth_radius_m),
        restrict_to_slip_span=False,
    )
    xs = np.asarray(nodes.get("chain", []), dtype=float)
    ys = np.asarray(nodes.get("curvature", []), dtype=float)
    if xs.size < 2 or ys.size != xs.size:
        return np.array([], dtype=float), np.array([], dtype=float)
    eff_span = effective_profile_slip_span_range(
        prof,
        rdp_eps_m=float(rdp_eps_m),
        smooth_radius_m=float(smooth_radius_m),
        xs=xs,
    )
    if eff_span is not None:
        smin, smax = eff_span
        return _clip_polyline_to_span(xs, ys, float(smin), float(smax))
    return xs, ys


def _group_boundary_style(reasons: List[str]) -> Dict[str, Any]:
    rs = [str(r).strip().lower() for r in (reasons or []) if str(r).strip()]
    if any(r in ("slip_span_start", "slip_span_end") for r in rs):
        return {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.3, "zorder": 10}
    if any(r.startswith("curvature_gt_") for r in rs):
        return {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.3, "zorder": 10}
    if any(r == "vector_angle_zero_deg" for r in rs):
        return {"color": "#d62728", "linestyle": "-", "linewidth": 1.3, "zorder": 10}
    return {"color": "#555555", "linestyle": (0, (4, 4)), "linewidth": 0.9, "zorder": 10}


def _infer_slip_curve_points(prof, group_ranges, eps_rdp=0.5, k_thr=0.0):
    _ = k_thr
    ch = prof.get("chain")
    if ch is None:
        ch = prof.get("chainage_m")
    gz = prof.get("ground_z_smooth")
    if gz is None:
        gz = prof.get("ground_z")
    if gz is None:
        gz = prof.get("z")
    if ch is None or gz is None or len(ch) < 4:
        return []
    ch = np.asarray(ch, dtype=float)
    gz = np.asarray(gz, dtype=float)
    pts = list(zip(map(float, ch), map(float, gz)))
    simp = _rdp_polyline(pts, eps_rdp)
    k = _curvature_series(list(map(float, ch)), list(map(float, gz)))
    bounds = []
    if group_ranges:
        for g in group_ranges:
            s = float(g.get("start", g.get("start_chainage", ch[0])))
            e = float(g.get("end", g.get("end_chainage", ch[-1])))
            if e < s:
                s, e = e, s
            bounds.append((s, e))
        bounds.sort()
    cps = []
    if bounds:
        cps.append((bounds[0][0], float(np.interp(bounds[0][0], ch, gz))))
        for s, e in bounds:
            i0 = max(0, int(np.searchsorted(ch, s)) - 1)
            i1 = min(len(ch) - 1, int(np.searchsorted(ch, e)))
            if i1 <= i0:
                continue
            window = list(range(i0, i1 + 1))
            idx = max(window, key=lambda i: (abs(k[i]), -k[i]))
            cps.append((float(ch[idx]), float(gz[idx])))
        cps.append((bounds[-1][1], float(np.interp(bounds[-1][1], ch, gz))))
    else:
        cps = simp
    xs, ys = zip(*sorted(set(cps)))
    if len(xs) < 3:
        return list(zip(xs, ys))
    try:
        from scipy.interpolate import splrep, splev
        tck = splrep(xs, ys, s=0)
        xs_new = np.linspace(xs[0], xs[-1], 200)
        ys_new = splev(xs_new, tck)
        return list(zip(xs_new.tolist(), ys_new.tolist()))
    except Exception:
        return list(zip(xs, ys))


def render_profile_png(
    prof: Dict[str, np.ndarray],
    out_png: str,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    vec_scale: float = 0.1,
    vec_width: float = 0.0015,
    head_len: float = 7.0,
    head_w: float = 5.0,
    highlight_theta: Optional[float] = None,
    group_ranges: Optional[List[dict]] = None,
    draw_curve: bool = False,
    save_curve_json: bool = False,
    overlay_curves: Optional[list] = None,
    figsize: Tuple[float, float] = (18, 10),
    dpi: int = 220,
    base_font: int = 20,
    label_font: int = 20,
    tick_font: int = 20,
    legend_font: int = 20,
    ground_lw: float = 2.2,
    ungrouped_color: str = "#bbbbbb",
    curvature_series: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    curvature_rdp_eps_m: float = 0.5,
    curvature_smooth_radius_m: float = 0.0,
) -> Tuple[str, Optional[str]]:
    if not prof:
        return "Empty profile", None
    font_scale = 0.8
    base_font = max(1, int(round(base_font * font_scale)))
    label_font = max(1, int(round(label_font * font_scale)))
    tick_font = max(1, int(round(tick_font * font_scale)))
    legend_font = max(1, int(round(legend_font * font_scale * 0.9)))
    axis_label_font = max(1, int(round(label_font * 0.9)))
    axis_tick_font = max(1, int(round(tick_font * 0.9)))
    x_user_min, x_user_max = x_min, x_max
    y_user_min, y_user_max = y_min, y_max
    curvature_plot_scale = -50.0
    curvature_plot_label = "Curvature plot (-50×k)"
    curvature_threshold_plot = abs(float(curvature_plot_scale)) * float(CURVATURE_THRESHOLD_PLOT_ABS)
    chain = prof["chain"]; elev_s = prof["elev_s"]
    d_para = prof["d_para"]; dz = prof["dz"]; theta = prof["theta"]
    profile_src = str(prof.get("profile_dem_source", "") or "").strip().lower()
    ground_label = "Ground"
    if profile_src == "raw":
        ground_label = "Ground (raw DEM)"
    elif profile_src == "smooth":
        ground_label = "Ground (smoothed DEM)"
    effective_span = effective_profile_slip_span_range(
        prof,
        rdp_eps_m=float(curvature_rdp_eps_m),
        smooth_radius_m=float(curvature_smooth_radius_m),
        chain=chain,
        finite_fallback=np.isfinite(chain) & np.isfinite(elev_s),
    )
    if (x_min is None) or (x_max is None):
        if effective_span is not None:
            smin, smax = effective_span
            if x_min is None:
                x_min = float(smin)
            if x_max is None:
                x_max = float(smax)
        else:
            finite_xy = np.isfinite(chain) & np.isfinite(elev_s)
            if finite_xy.any():
                if x_min is None:
                    x_min = float(np.nanmin(chain[finite_xy]))
                if x_max is None:
                    x_max = float(np.nanmax(chain[finite_xy]))
    if (y_min is None) or (y_max is None):
        finite_elev = np.isfinite(elev_s)
        if finite_elev.any():
            z_min = float(np.nanmin(elev_s[finite_elev]))
            z_max = float(np.nanmax(elev_s[finite_elev]))
            span = max(z_max - z_min, 0.5)
            pad = 0.02 * span
            if y_min is None:
                y_min = z_min - pad
            if y_max is None:
                y_max = z_max + pad

    def _set_chainage_xlim(ax_obj, left_val: float, right_val: float) -> None:
        ax_obj.set_xlim(float(min(left_val, right_val)), float(max(left_val, right_val)))

    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": base_font}):
        fig = plt.figure(figsize=figsize if figsize else (18, 12), dpi=dpi)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.0, 1.45], hspace=0.35)
        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
        ax.plot(chain, elev_s, "k-", lw=ground_lw, label=ground_label)
        finite_prof = np.isfinite(chain) & np.isfinite(elev_s) & np.isfinite(d_para) & np.isfinite(dz)
        slip_mask_arr = None
        try:
            sm = prof.get("slip_mask", None)
            if sm is not None:
                sm = np.asarray(sm)
                if sm.shape == chain.shape:
                    slip_mask_arr = sm == True
        except Exception:
            slip_mask_arr = None
        if effective_span is not None:
            plot_span_min, plot_span_max = map(float, effective_span)
        else:
            finite_span = np.isfinite(chain) & np.isfinite(elev_s)
            if finite_span.any():
                plot_span_min = float(np.nanmin(chain[finite_span]))
                plot_span_max = float(np.nanmax(chain[finite_span]))
            else:
                plot_span_min = None
                plot_span_max = None
        prof_curv = dict(prof)
        prof_curv["slip_span"] = (float(plot_span_min), float(plot_span_max)) if (plot_span_min is not None and plot_span_max is not None) else None

        def _resolve_curvature_series() -> Tuple[np.ndarray, np.ndarray]:
            if curvature_series is not None:
                try:
                    kx = np.asarray(curvature_series[0], dtype=float)
                    kv = np.asarray(curvature_series[1], dtype=float)
                    keep = np.isfinite(kx) & np.isfinite(kv)
                    kx = kx[keep]; kv = kv[keep]
                    if kx.size >= 2 and kv.size == kx.size:
                        order = np.argsort(kx)
                        kx = kx[order]; kv = kv[order]
                        if (plot_span_min is not None) and (plot_span_max is not None):
                            kx, kv = _clip_polyline_to_span(kx, kv, float(plot_span_min), float(plot_span_max))
                        if kx.size >= 2 and kv.size == kx.size:
                            return kx, kv
                except Exception:
                    pass
            return _curvature_plot_series(prof_curv, rdp_eps_m=float(curvature_rdp_eps_m), smooth_radius_m=float(curvature_smooth_radius_m))
        if group_ranges:
            if (plot_span_min is not None) and (plot_span_max is not None):
                smin, smax = float(plot_span_min), float(plot_span_max)
            else:
                smin = float(np.nanmin(prof["chain"][finite_prof]))
                smax = float(np.nanmax(prof["chain"][finite_prof]))
            cmap = plt.get_cmap("tab10")
            prepared = []
            for gi, gr in enumerate(group_ranges):
                gid = gr.get("id", f"G{gi + 1}")
                s = float(gr.get("start", gr.get("start_chainage", 0.0)))
                e = float(gr.get("end", gr.get("end_chainage", 0.0)))
                if e < s:
                    s, e = e, s
                s = max(s, smin); e = min(e, smax)
                if e <= s:
                    continue
                color = gr.get("color", None) or mcolors.to_hex(cmap(gi % 10))
                prepared.append((gi, gid, s, e, color))
            gidx = np.full(chain.shape, -1, dtype=int)
            for gi, _gid, s, e, _color in prepared:
                m = (chain >= s) & (chain <= e)
                if slip_mask_arr is not None:
                    m = m & slip_mask_arr
                gidx[m] = gi
            for gi, _gid, _s, _e, color in prepared:
                m = finite_prof & (gidx == gi)
                if np.any(m):
                    ax.quiver(chain[m], elev_s[m], d_para[m], dz[m], angles="xy", scale_units="xy", scale=vec_scale, width=vec_width, color=color, headlength=head_len, headwidth=head_w)
                    ax.plot([], [], color=color, lw=3, label="_nolegend_")
            order = np.argsort(chain)
            chain_s = chain[order]; d_para_s = d_para[order]; dz_s = dz[order]
            finite_s = np.isfinite(chain_s) & np.isfinite(d_para_s) & np.isfinite(dz_s)
            if (plot_span_min is not None) and (plot_span_max is not None):
                finite_s = finite_s & (chain_s >= float(plot_span_min)) & (chain_s <= float(plot_span_max))
            if finite_s.sum() >= 2:
                ch = chain_s[finite_s]
                gradient_deg = np.degrees(np.arctan2(dz_s[finite_s], d_para_s[finite_s]))
                gradient_deg = ((gradient_deg + 90.0) % 180.0) - 90.0
                ax2.plot(ch, gradient_deg, lw=2.2, color="#2ca02c", zorder=5, label="Gradient")
            ax2.axhline(0.0, color="0.5", lw=1.0, zorder=1)
            ax2.set_ylabel("Gradient (deg)", fontsize=axis_label_font)
            ax2r = ax2.twinx()
            ax2r.set_ylabel(curvature_plot_label, fontsize=axis_label_font)
            k_curve = None
            try:
                k_x, k_vals = _resolve_curvature_series()
                if k_x.size >= 3 and k_vals.size == k_x.size:
                    k_curve = np.asarray(k_vals, dtype=float)
                    k_curve_plot = float(curvature_plot_scale) * k_curve
                    ax2r.plot(k_x, k_curve_plot, lw=1.8, color="#222222", marker="o", markersize=4, zorder=6, label="Curvature")
            except Exception:
                pass
            ax2r.axhline(float(curvature_threshold_plot), color="#cc3333", lw=1.1, linestyle=(0, (4, 3)), zorder=4, label="_nolegend_")
            ax2r.axhline(-float(curvature_threshold_plot), color="#cc3333", lw=1.1, linestyle=(0, (4, 3)), zorder=4, label="_nolegend_")
            if k_curve is not None and np.any(np.isfinite(k_curve)):
                qk = np.nanpercentile(np.abs(float(curvature_plot_scale) * k_curve), 98)
                qk = max(float(qk), float(curvature_threshold_plot))
                if np.isfinite(qk) and qk > 0:
                    ax2r.set_ylim(-1.2 * qk, 1.2 * qk)
            else:
                ax2r.set_ylim(-1.2 * float(curvature_threshold_plot), 1.2 * float(curvature_threshold_plot))
            ax2r.grid(False)
            ax2.grid(ls="--", lw=0.8, alpha=0.35)
            try:
                if finite_s.sum() >= 2:
                    q = np.nanpercentile(np.abs(gradient_deg), 98)
                    if np.isfinite(q) and q > 0:
                        ax2.set_ylim(-1.2 * q, 1.2 * q)
            except Exception:
                pass
            handles = ax2.get_lines() + ax2r.get_lines()
            if handles:
                keep = [(h, h.get_label()) for h in handles if not h.get_label().startswith("_")]
                if keep:
                    h_keep, l_keep = zip(*keep)
                    ax2.legend(list(h_keep), list(l_keep), loc="upper left", bbox_to_anchor=(0.0, -0.18), fontsize=legend_font, frameon=False, ncol=2)
            m = finite_prof & (gidx == -1)
            if np.any(m):
                ug_color = ungrouped_color or "#bbbbbb"
                ax.quiver(chain[m], elev_s[m], d_para[m], dz[m], angles="xy", scale_units="xy", scale=vec_scale, width=vec_width, color=ug_color, alpha=0.9, headlength=head_len, headwidth=head_w)
                ax.plot([], [], color=ug_color, lw=3, label="_nolegend_")
        else:
            ax.quiver(chain, elev_s, d_para, dz, angles="xy", scale_units="xy", scale=vec_scale, width=vec_width, color="tab:red", headlength=head_len, headwidth=head_w)
            finite_th = np.isfinite(chain) & np.isfinite(d_para) & np.isfinite(dz)
            if (plot_span_min is not None) and (plot_span_max is not None):
                finite_th = finite_th & (chain >= float(plot_span_min)) & (chain <= float(plot_span_max))
            if finite_th.sum() >= 2:
                chain_f = chain[finite_th]
                gradient_deg = np.degrees(np.arctan2(dz[finite_th], d_para[finite_th]))
                gradient_deg = ((gradient_deg + 90.0) % 180.0) - 90.0
                ax2.plot(chain_f, gradient_deg, color="#2ca02c", lw=2.4, zorder=5, label="Gradient")
            ax2.axhline(0.0, color="0.5", lw=1.0, zorder=1)
            ax2.set_xlabel("Chainage (m)")
            ax2.set_ylabel("Gradient (deg)")
            ax2r = ax2.twinx()
            ax2r.set_ylabel(curvature_plot_label)
            k_curve = None
            try:
                k_x, k_vals = _resolve_curvature_series()
                if k_x.size >= 3 and k_vals.size == k_x.size:
                    k_curve = np.asarray(k_vals, dtype=float)
                    k_curve_plot = float(curvature_plot_scale) * k_curve
                    ax2r.plot(k_x, k_curve_plot, color="#222222", lw=2.0, marker="o", markersize=4, zorder=6, label="Curvature")
            except Exception:
                pass
            ax2r.axhline(float(curvature_threshold_plot), color="#cc3333", lw=1.1, linestyle=(0, (4, 3)), zorder=4, label="_nolegend_")
            ax2r.axhline(-float(curvature_threshold_plot), color="#cc3333", lw=1.1, linestyle=(0, (4, 3)), zorder=4, label="_nolegend_")
            if k_curve is not None and np.any(np.isfinite(k_curve)):
                qk = np.nanpercentile(np.abs(float(curvature_plot_scale) * k_curve), 98)
                qk = max(float(qk), float(curvature_threshold_plot))
                if np.isfinite(qk) and qk > 0:
                    ax2r.set_ylim(-1.2 * qk, 1.2 * qk)
            else:
                ax2r.set_ylim(-1.2 * float(curvature_threshold_plot), 1.2 * float(curvature_threshold_plot))
            ax2r.grid(False)
            ax2.grid(True, linestyle="--", alpha=0.4)
            try:
                if finite_th.sum() >= 2:
                    q = np.nanpercentile(np.abs(gradient_deg), 98)
                    if np.isfinite(q) and q > 0:
                        ax2.set_ylim(-1.2 * q, 1.2 * q)
            except Exception:
                pass
            handles = ax2.get_lines() + ax2r.get_lines()
            if handles:
                keep = [(h, h.get_label()) for h in handles if not h.get_label().startswith("_")]
                if keep:
                    h_keep, l_keep = zip(*keep)
                    ax2.legend(list(h_keep), list(l_keep), loc="upper left", bbox_to_anchor=(0.0, -0.18), fontsize=legend_font, frameon=False, ncol=2)
            if x_min is not None and x_max is not None and (abs(float(x_max) - float(x_min)) > 1e-12):
                _set_chainage_xlim(ax2, x_min, x_max)
            if group_ranges:
                for gi, gr in enumerate(group_ranges):
                    s = float(gr.get("start", gr.get("start_chainage", 0.0)))
                    e = float(gr.get("end", gr.get("end_chainage", 0.0)))
                    if e < s:
                        s, e = e, s
                    color = gr.get("color", None) or plt.get_cmap("tab10")(gi % 10)
                    ax2.axvspan(s, e, color=color, alpha=0.08, zorder=0)
        if group_ranges:
            try:
                xmin, xmax = ax.get_xlim()
            except Exception:
                xmin, xmax = None, None
            clip_lo = clip_hi = None
            if xmin is not None and xmax is not None:
                clip_lo = float(min(xmin, xmax))
                clip_hi = float(max(xmin, xmax))
            bounds_meta: Dict[float, Dict[str, Any]] = {}
            for g in group_ranges:
                try:
                    s = float(g.get("start", g.get("start_chainage", 0.0)))
                    e = float(g.get("end", g.get("end_chainage", 0.0)))
                except Exception:
                    continue
                if e < s:
                    s, e = e, s
                if (clip_lo is not None) and (clip_hi is not None):
                    s = max(clip_lo, min(clip_hi, s))
                    e = max(clip_lo, min(clip_hi, e))
                for x, reason in ((s, str(g.get("start_reason", "") or "").strip()), (e, str(g.get("end_reason", "") or "").strip())):
                    key = round(float(x), 9)
                    meta = bounds_meta.setdefault(key, {"x": float(x), "reasons": []})
                    if reason:
                        meta["reasons"].append(reason)
            bounds_items = sorted(bounds_meta.values(), key=lambda t: float(t["x"]))
            if bounds_items:
                for it in bounds_items:
                    x = float(it["x"])
                    vkw = _group_boundary_style(list(it.get("reasons", [])))
                    ax.axvline(x, **vkw)
                    ax2.axvline(x, **vkw)
                try:
                    import matplotlib.transforms as mtransforms
                    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
                    label_fs = max(8, int(round(tick_font * 0.9)))
                    label_items = []
                    for gi, gr in enumerate(group_ranges):
                        if bool(gr.get("suppress_label", False)):
                            continue
                        try:
                            s = float(gr.get("start", gr.get("start_chainage", np.nan)))
                            e = float(gr.get("end", gr.get("end_chainage", np.nan)))
                        except Exception:
                            continue
                        if not (np.isfinite(s) and np.isfinite(e)):
                            continue
                        if e < s:
                            s, e = e, s
                        if (clip_lo is not None) and (clip_hi is not None):
                            s = max(clip_lo, min(clip_hi, s))
                            e = max(clip_lo, min(clip_hi, e))
                        if not (np.isfinite(s) and np.isfinite(e)) or e <= s:
                            continue
                        label_items.append((0.5 * (s + e), int(gi)))
                    label_items.sort(key=lambda t: t[0])
                    for idx, (xm, _gi) in enumerate(label_items, start=1):
                        ax.text(float(xm), 0.995, str(idx), transform=trans, ha="center", va="top", fontsize=label_fs, color="#333333", zorder=60)
                except Exception:
                    pass
        if draw_curve:
            try:
                curve_pts = _infer_slip_curve_points(prof, group_ranges, eps_rdp=0.5)
            except Exception:
                curve_pts = []
            if curve_pts:
                cx, cz = zip(*curve_pts)
                cx = np.asarray(cx, dtype=float); cz = np.asarray(cz, dtype=float)
                m = np.isfinite(cx) & np.isfinite(cz)
                cx = cx[m]; cz = cz[m]
                try:
                    xmin, xmax = ax.get_xlim()
                    xmask = (cx >= xmin) & (cx <= xmax)
                    cx = cx[xmask]; cz = cz[xmask]
                except Exception:
                    pass
                try:
                    ymin, ymax = ax.get_ylim()
                    ymask = (cz >= ymin) & (cz <= ymax)
                    cx = cx[ymask]; cz = cz[ymask]
                except Exception:
                    pass
                if cx.size > 1:
                    ln, = ax.plot(cx, cz, color="#bf00ff", linewidth=3.0, zorder=50, label="Slip curve")
                    try:
                        import matplotlib.patheffects as pe
                        ln.set_path_effects([pe.Stroke(linewidth=5, foreground="white"), pe.Normal()])
                    except Exception:
                        pass
                if save_curve_json and out_png:
                    cjson = out_png.rsplit(".", 1)[0] + "_curve.json"
                    try:
                        with open(cjson, "w", encoding="utf-8") as f:
                            json.dump({"curve": [{"s": float(x), "z": float(y)} for x, y in curve_pts]}, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                if save_curve_json and out_png and "elev_s" in prof and prof["elev_s"] is not None:
                    gjson = out_png.rsplit(".", 1)[0] + "_ground_smoothed.json"
                    try:
                        fin = np.isfinite(prof["chain"]) & np.isfinite(prof["elev_s"])
                        ch_fin = prof["chain"][fin]
                        el_fin = prof["elev_s"][fin]
                        with open(gjson, "w", encoding="utf-8") as f:
                            json.dump({"ground_smoothed": [{"s": float(s), "z": float(z)} for s, z in zip(ch_fin, el_fin)]}, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
        if (highlight_theta is not None) and (float(highlight_theta) > 0):
            thr = float(highlight_theta)
            m = np.isfinite(theta) & (theta >= thr) & np.isfinite(dz)
            if np.any(m):
                ax.plot(chain[m], elev_s[m], "o", color="limegreen", ms=5, label=f"θ >= {thr:.1f} deg & dz>0")
        ax2.set_xlabel("Chainage (m)", fontsize=axis_label_font)
        ax.set_ylabel("Elevation (m)", fontsize=axis_label_font)
        ax2.set_ylabel("Gradient (deg)", fontsize=axis_label_font)
        try:
            ax2r.set_ylabel("Curvature (1/m)", fontsize=axis_label_font)
        except Exception:
            pass
        ax.tick_params(axis="both", labelsize=axis_tick_font)
        ax2.tick_params(axis="both", labelsize=axis_tick_font)
        try:
            ax2r.tick_params(axis="both", labelsize=axis_tick_font)
        except Exception:
            pass
        ax.grid(ls="--", lw=0.8, alpha=0.35)
        ax.margins(x=0.02)
        if overlay_curves:
            user_x_fixed = (x_user_min is not None) and (x_user_max is not None) and (x_user_max > x_user_min)
            user_y_fixed = (y_user_min is not None) and (y_user_max is not None) and (y_user_max > y_user_min)
            try:
                xs_list: List[np.ndarray] = []
                ys_list: List[np.ndarray] = []
                finite_xy = np.isfinite(prof["chain"]) & np.isfinite(prof["elev_s"])
                if np.any(finite_xy):
                    xs_list.append(np.asarray(prof["chain"][finite_xy], float))
                    ys_list.append(np.asarray(prof["elev_s"][finite_xy], float))
                for item in overlay_curves:
                    if len(item) == 2:
                        ch, zz = item
                    elif len(item) == 3:
                        ch, zz, _ = item
                    else:
                        ch, zz, _, _ = item
                    ch = np.asarray(ch, float); zz = np.asarray(zz, float)
                    m = np.isfinite(ch) & np.isfinite(zz)
                    if not np.any(m):
                        continue
                    xs_list.append(ch[m]); ys_list.append(zz[m])
                if xs_list and ys_list:
                    xs_all = np.concatenate(xs_list); ys_all = np.concatenate(ys_list)
                    x0, x1 = float(xs_all.min()), float(xs_all.max())
                    y0, y1 = float(ys_all.min()), float(ys_all.max())
                    xpad = 0.02 * ((x1 - x0) or 1.0)
                    ypad = 0.05 * ((y1 - y0) or 1.0)
                    if not user_x_fixed:
                        _set_chainage_xlim(ax, x0 - xpad, x1 + xpad)
                        _set_chainage_xlim(ax2, x0 - xpad, x1 + xpad)
                    if not user_y_fixed:
                        ax.set_ylim(y0 - ypad, y1 + ypad)
            except Exception:
                pass
        user_y_fixed = (y_user_min is not None) and (y_user_max is not None) and (y_user_max > y_user_min)
        user_x_fixed = (x_user_min is not None) and (x_user_max is not None) and (abs(float(x_user_max) - float(x_user_min)) > 1e-12)
        if user_y_fixed:
            ax.set_ylim(float(y_user_min), float(y_user_max))
        if user_x_fixed:
            _set_chainage_xlim(ax, float(x_user_min), float(x_user_max))
            _set_chainage_xlim(ax2, float(x_user_min), float(x_user_max))
        try:
            from matplotlib.ticker import MultipleLocator
            ax.xaxis.set_major_locator(MultipleLocator(20.0))
            ax2.xaxis.set_major_locator(MultipleLocator(20.0))
        except Exception:
            pass
        if overlay_curves:
            for item in overlay_curves:
                if len(item) == 2:
                    ch, zz = item; color, label = "#bf00ff", "Slip curve"
                elif len(item) == 3:
                    ch, zz, color = item; label = "Slip curve"
                else:
                    ch, zz, color, label = item
                ch = np.asarray(ch, float); zz = np.asarray(zz, float)
                m = np.isfinite(ch) & np.isfinite(zz)
                if not np.any(m):
                    continue
                ln, = ax.plot(ch[m], zz[m], color=color, lw=3.0, zorder=50, label=label)
                try:
                    import matplotlib.patheffects as pe
                    ln.set_path_effects([pe.Stroke(linewidth=5, foreground="white"), pe.Normal()])
                except Exception:
                    pass
        ax.legend(loc="best", fontsize=legend_font, frameon=False)
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox_top = ax.get_window_extent(renderer=renderer)
        x_top, y_top, w_top, h_top = map(float, bbox_top.bounds)
        x_min_top, x_max_top = ax.get_xlim()
        bbox_bot = ax2.get_window_extent(renderer=renderer)
        x_bot, y_bot, w_bot, h_bot = map(float, bbox_bot.bounds)
        x_min_bot, x_max_bot = ax2.get_xlim()
        pad_inches = 0.15
        tight = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
        pad_px = float(pad_inches) * float(fig.dpi)
        crop_x0 = float(tight.x0) - pad_px
        crop_y0 = float(tight.y0) - pad_px
        top_left_px = x_top - crop_x0
        bot_left_px = x_bot - crop_x0
        crop_h = float(tight.height) + (2.0 * pad_px)
        top_top_px = crop_h - ((y_top - crop_y0) + h_top)
        bot_top_px = crop_h - ((y_bot - crop_y0) + h_bot)
        meta = {
            "top": {"x_min": float(x_min_top), "x_max": float(x_max_top), "y_min": float(min(ax.get_ylim())), "y_max": float(max(ax.get_ylim())), "left_px": float(top_left_px), "top_px": float(top_top_px), "width_px": float(w_top), "height_px": float(h_top)},
            "bot": {"x_min": float(x_min_bot), "x_max": float(x_max_bot), "y_min": float(min(ax2.get_ylim())), "y_max": float(max(ax2.get_ylim())), "left_px": float(bot_left_px), "top_px": float(bot_top_px), "width_px": float(w_bot), "height_px": float(h_bot)},
        }
        meta_path = out_png.rsplit(".", 1)[0] + ".json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
        plt.close(fig)
        return f"Saved {out_png}", out_png
