import os
from typing import Dict, Optional


class UI4PreviewControllerMixin:
    def _load_ui4_summary_for_current_run(self) -> Dict:
        run_dir = (self._ctx.get("run_dir") or "").strip()
        self._last_ui4_summary = self._backend_load_ui4_summary(run_dir)
        return self._last_ui4_summary

    def _summary_range_for_kind(self, kind: str) -> Optional[tuple]:
        summary = self._last_ui4_summary or self._load_ui4_summary_for_current_run()
        return self._backend_summary_range(summary, kind)

    def _populate_manual_range_from_summary(self, kind: str) -> None:
        rng = self._summary_range_for_kind(kind)
        if rng is None:
            self._append(f"[UI4] Cannot auto-fill {kind} manual range: summary min/max not available.")
            return
        zmin, zmax = rng
        if kind == "surface":
            self.surface_zmin.setValue(zmin)
            self.surface_zmax.setValue(zmax)
        else:
            self.depth_zmin.setValue(zmin)
            self.depth_zmax.setValue(zmax)
        self._append(f"[UI4] {kind} manual range auto-filled from raster stats: {zmin:g} .. {zmax:g}")

    def _validate_contour_params(self, params: Dict[str, float | None]) -> Optional[str]:
        if params.get("surface_z_min") is not None and params.get("surface_z_max") is not None:
            if float(params["surface_z_max"]) <= float(params["surface_z_min"]):
                return "Invalid surface manual range: zmax must be greater than zmin."
        if params.get("depth_z_min") is not None and params.get("depth_z_max") is not None:
            if float(params["depth_z_max"]) <= float(params["depth_z_min"]):
                return "Invalid depth manual range: zmax must be greater than zmin."
        return None

    def _contour_param_values(self) -> Dict[str, float | None]:
        surf_zmin = None if self.surface_auto_range.isChecked() else float(self.surface_zmin.value())
        surf_zmax = None if self.surface_auto_range.isChecked() else float(self.surface_zmax.value())
        depth_zmin = None if self.depth_auto_range.isChecked() else float(self.depth_zmin.value())
        depth_zmax = None if self.depth_auto_range.isChecked() else float(self.depth_zmax.value())
        return {
            "surface_interval_m": float(self.surface_step.value()),
            "depth_interval_m": float(self.depth_step.value()),
            "surface_z_min": surf_zmin,
            "surface_z_max": surf_zmax,
            "depth_z_min": depth_zmin,
            "depth_z_max": depth_zmax,
        }

    def _on_preview_file_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._preview_png_paths):
            self.preview_view.clear_image()
            return
        path = self._preview_png_paths[idx]
        ok = self.preview_view.load_image(path)
        if ok:
            self.lbl_preview_status.setText(
                f"Preview: {len(self._preview_png_paths)} PNG file(s) | Showing: {os.path.basename(path)} "
                f"(wheel/Zoom +/-/Fit)"
            )
        else:
            self.lbl_preview_status.setText(f"Preview: cannot load image ({path})")

    def _refresh_preview_pngs(self, prefer_surface: bool = False) -> None:
        run_dir = (self._ctx.get("run_dir") or "").strip()
        prev_selected = self.preview_file_combo.currentData()
        self.preview_file_combo.blockSignals(True)
        self.preview_file_combo.clear()
        self.preview_file_combo.blockSignals(False)
        self._preview_png_paths = []
        self.preview_view.clear_image()

        preview_info = self._backend_list_ui4_preview_pngs(run_dir)
        if not preview_info.get("ok", False):
            self.lbl_preview_status.setText(f"Preview: {preview_info.get('error', 'unavailable')}")
            return

        pngs = list(preview_info.get("pngs", []))
        preview_dir = str(preview_info.get("preview_dir", "") or "")
        self._preview_png_paths = pngs
        self.lbl_preview_status.setText(f"Preview: {len(pngs)} PNG file(s) in {preview_dir}")
        if not pngs:
            return

        self.preview_file_combo.blockSignals(True)
        for path in pngs:
            self.preview_file_combo.addItem(os.path.basename(path), path)
        self.preview_file_combo.blockSignals(False)

        idx = 0
        if prefer_surface:
            for i, path in enumerate(pngs):
                if os.path.basename(path).lower() == "contours_surface.png":
                    idx = i
                    break
        elif prev_selected:
            try:
                idx = pngs.index(prev_selected)
            except ValueError:
                idx = 0
        self.preview_file_combo.setCurrentIndex(idx)
        self._on_preview_file_changed(idx)
