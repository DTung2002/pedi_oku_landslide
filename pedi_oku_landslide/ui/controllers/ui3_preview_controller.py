import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtGui import QColor, QPainterPath, QPen, QPixmap, QPixmapCache
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsSimpleTextItem,
)

from pedi_oku_landslide.ui.scenes.ui3_preview_scene import AnchorMarkerItem


class UI3PreviewControllerMixin:
    def _render_profile_png_current_settings(
        self,
        prof: dict,
        out_png: str,
        *,
        groups: Optional[List[dict]] = None,
        overlay_curves: Optional[list] = None,
    ) -> Optional[str]:
        result = self._backend.render_preview(
            prof,
            {
                "out_png": out_png,
                "vec_scale": self.vscale.value(),
                "vec_width": self.vwidth.value(),
                "head_len": 6.0,
                "head_w": 4.0,
                "ungrouped_color": self._get_ungrouped_color(),
                "curvature_rdp_eps_m": self._current_rdp_eps_m(),
                "curvature_smooth_radius_m": 0.0,
            },
            groups=groups,
            overlay_curves=overlay_curves,
        )
        msg = str(result.get("message", "") or "")
        path = result.get("path")
        if msg:
            self._log(msg)
        if path and os.path.exists(path):
            return path
        return None

    def _load_preview_scene_from_path(self, path: str, *, static_nurbs_bg: bool) -> bool:
        pm = QPixmap(path)
        if pm.isNull():
            return False
        QPixmapCache.clear()
        self.scene.clear()
        item = QGraphicsPixmapItem(pm)
        self.scene.addItem(item)
        self._img_ground = item
        self._img_rate0 = item
        self._clear_curve_overlay()
        self._load_axes_meta(path)
        self._static_nurbs_bg_loaded = bool(static_nurbs_bg)
        if getattr(self, "_first_show", True):
            self.view.fit_to_scene()
            self._first_show = False
        self._refresh_anchor_overlay()
        return True

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

    def _on_boring_hole_marker_clicked(self, hole: dict) -> None:
        try:
            lbl = str(hole.get("label", hole.get("bh", "")) or "").strip() or "Boring hole"
            self._log(
                f"[UI3] {lbl}: x={float(hole.get('x')):.3f}, "
                f"y={float(hole.get('y')):.3f}, z={float(hole.get('z')):.3f}, "
                f"s={float(hole.get('s_on_line')):.3f}, d={float(hole.get('distance_to_line_m')):.3f}"
            )
        except Exception:
            pass

    def _refresh_anchor_overlay(self) -> None:
        self._clear_anchor_overlay()
        if self.scene is None or self._img_ground is None or self._ax_top is None:
            return
        boring_result = self._project_boring_holes_for_current_line(use_unsaved_table=True, log_skips=False)
        for hole in (boring_result.get("items", []) or []):
            try:
                s = float(hole.get("s_on_line"))
                z = float(hole.get("z"))
            except Exception:
                continue
            pt = self._chain_elev_to_scene_xy(s, z)
            if pt is None:
                continue
            x, y = pt
            label = str(hole.get("label", hole.get("bh", "")) or "").strip() or "BH"
            tip = (
                f"{label}\n"
                f"x={float(hole.get('x')):.3f}\n"
                f"y={float(hole.get('y')):.3f}\n"
                f"z={float(hole.get('z')):.3f}\n"
                f"s={float(hole.get('s_on_line')):.3f}\n"
                f"d={float(hole.get('distance_to_line_m')):.3f}"
            )
            marker = AnchorMarkerItem(
                x=x, y=y, r=9.0, tooltip=tip,
                on_click=lambda hh=dict(hole): self._on_boring_hole_marker_clicked(hh)
            )
            marker.setBrush(QColor("#ff3b30"))
            pen = QPen(QColor("#ffffff"))
            pen.setWidth(2)
            pen.setCosmetic(True)
            marker.setPen(pen)
            marker.setZValue(144.0)
            self.scene.addItem(marker)
            self._anchor_overlay_items.append(marker)

            lbl = QGraphicsSimpleTextItem(label)
            lbl.setBrush(QColor("#7a0c00"))
            fnt = lbl.font()
            psz = fnt.pointSizeF() if fnt.pointSizeF() > 0 else 8.0
            fnt.setPointSizeF(psz * 1.25)
            lbl.setFont(fnt)
            lbl.setToolTip(tip)
            lbl.setPos(x + 8.0, y - 18.0)
            lbl.setZValue(145.0)
            self.scene.addItem(lbl)
            self._anchor_overlay_items.append(lbl)
        if self._current_ui2_line_role() != "cross":
            return
        cross_id = self._current_ui2_line_id()
        anchors = self._anchors_for_cross_line(cross_id, require_ready=True)
        if len(anchors) < 3:
            return

        colors = {
            1: "#e74c3c",
            2: "#2ecc71",
            3: "#3498db",
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

    def _draw_global_fit_debug_overlay(self, result: Optional[Dict[str, Any]], *, draw_curve: bool = False) -> None:
        self._clear_control_points_overlay()
        if self.scene is None or not result:
            return
        curve = dict(result.get("curve", {}) or {})
        if draw_curve:
            self._draw_curve_overlay(
                np.asarray(curve.get("chain", []), dtype=float),
                np.asarray(curve.get("elev", []), dtype=float),
                color="#bf00ff",
            )

        for idx, seg in enumerate(result.get("steps", []) or [], 1):
            start_pt = dict(seg.get("start_point", {}) or {})
            aux_pt = dict(seg.get("aux_fit_point", {}) or {})
            g_pt = dict(seg.get("intersection_point", {}) or {})
            start_xy = self._chain_elev_to_scene_xy(float(start_pt.get("chain", np.nan)), float(start_pt.get("elev", np.nan)))
            aux_xy = self._chain_elev_to_scene_xy(float(aux_pt.get("chain", np.nan)), float(aux_pt.get("elev", np.nan)))
            if start_xy is not None and aux_xy is not None:
                line = QGraphicsLineItem(start_xy[0], start_xy[1], aux_xy[0], aux_xy[1])
                pen = QPen(QColor("#ff9f1a"))
                pen.setWidth(2)
                pen.setCosmetic(True)
                line.setPen(pen)
                line.setZValue(126.0)
                self.scene.addItem(line)
                self._cp_overlay_items.append(line)
            if aux_xy is not None:
                r = 7.0
                marker = QGraphicsEllipseItem(aux_xy[0] - r, aux_xy[1] - r, 2 * r, 2 * r)
                marker.setBrush(QColor("#ff9f1a"))
                marker.setPen(QPen(QColor("#ffffff")))
                marker.setZValue(127.0)
                self.scene.addItem(marker)
                self._cp_overlay_items.append(marker)

                lbl = QGraphicsSimpleTextItem(f"aux{idx}")
                lbl.setBrush(QColor("#7a4200"))
                lbl.setPos(aux_xy[0] + 8.0, aux_xy[1] - 18.0)
                lbl.setZValue(128.0)
                self.scene.addItem(lbl)
                self._cp_overlay_items.append(lbl)
            if g_pt:
                g_xy = self._chain_elev_to_scene_xy(float(g_pt.get("chain", np.nan)), float(g_pt.get("elev", np.nan)))
                if g_xy is not None:
                    r = 7.0
                    marker = QGraphicsEllipseItem(g_xy[0] - r, g_xy[1] - r, 2 * r, 2 * r)
                    marker.setBrush(QColor("#00c2a8"))
                    marker.setPen(QPen(QColor("#ffffff")))
                    marker.setZValue(127.0)
                    self.scene.addItem(marker)
                    self._cp_overlay_items.append(marker)

                    lbl = QGraphicsSimpleTextItem(f"G{idx}")
                    lbl.setBrush(QColor("#00695c"))
                    lbl.setPos(g_xy[0] + 8.0, g_xy[1] - 18.0)
                    lbl.setZValue(128.0)
                    self.scene.addItem(lbl)
                    self._cp_overlay_items.append(lbl)
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
            r = 10.0
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
