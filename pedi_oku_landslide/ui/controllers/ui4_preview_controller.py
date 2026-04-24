import os
from typing import Dict


class UI4PreviewControllerMixin:
    @staticmethod
    def _preview_type_label_for_path(path: str) -> str:
        name = os.path.basename(str(path or "")).lower()
        if name == "contours_depth.png":
            return "Depth"
        if name == "contours_surface.png":
            return "Surface"
        return os.path.basename(str(path or "")) or "-"

    @staticmethod
    def _preview_type_kind_for_path(path: str) -> str:
        label = UI4PreviewControllerMixin._preview_type_label_for_path(path).lower()
        if label == "depth":
            return "depth"
        return "surface"

    def _load_ui4_summary_for_current_run(self) -> Dict:
        run_dir = (self._ctx.get("run_dir") or "").strip()
        self._last_ui4_summary = self._backend_load_ui4_summary(run_dir)
        return self._last_ui4_summary

    def _selected_preview_type(self) -> str:
        combo = getattr(self, "preview_file_combo", None)
        path = combo.currentData() if combo is not None else ""
        return self._preview_type_kind_for_path(str(path or ""))

    def _sync_step_visibility_for_preview_type(self) -> None:
        selected = self._selected_preview_type()
        self.surface_step.setVisible(selected == "surface")
        self.depth_step.setVisible(selected == "depth")

    def _contour_param_values(self) -> Dict[str, float | None]:
        return {
            "surface_interval_m": float(self.surface_step.value()),
            "depth_interval_m": float(self.depth_step.value()),
            "surface_z_min": None,
            "surface_z_max": None,
            "depth_z_min": None,
            "depth_z_max": None,
        }

    def _on_preview_file_changed(self, idx: int) -> None:
        self._sync_step_visibility_for_preview_type()
        if idx < 0 or idx >= len(self._preview_png_paths):
            self.preview_view.clear_image()
            return
        path = self._preview_png_paths[idx]
        ok = self.preview_view.load_image(path)
        if ok:
            label = self._preview_type_label_for_path(path)
            self.lbl_preview_status.setText(
                f"Preview: {len(self._preview_png_paths)} PNG file(s) | Showing: {label} "
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
        self._sync_step_visibility_for_preview_type()

        preview_info = self._backend_list_ui4_preview_pngs(run_dir)
        if not preview_info.get("ok", False):
            self.lbl_preview_status.setText(f"Preview: {preview_info.get('error', 'unavailable')}")
            return

        pngs = [
            p for p in list(preview_info.get("pngs", []))
            if os.path.basename(str(p or "")).lower() in ("contours_surface.png", "contours_depth.png")
        ]
        preview_dir = str(preview_info.get("preview_dir", "") or "")
        self._preview_png_paths = pngs
        self.lbl_preview_status.setText(f"Preview: {len(pngs)} PNG file(s) in {preview_dir}")
        if not pngs:
            return

        self.preview_file_combo.blockSignals(True)
        for path in pngs:
            self.preview_file_combo.addItem(self._preview_type_label_for_path(path), path)
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
