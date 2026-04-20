import os


class UI4RunControllerMixin:
    def set_context(self, project: str, run_label: str, run_dir: str) -> None:
        self._ctx = {
            "project": str(project or ""),
            "run_label": str(run_label or ""),
            "run_dir": str(run_dir or ""),
        }
        self.lbl_project_value.setText(self._ctx["project"] or "-")
        self.lbl_run_label_value.setText(self._ctx["run_label"] or "-")
        self.refresh_from_context()

    def on_upstream_curve_saved(self, curve_json_path: str = "") -> None:
        if curve_json_path:
            self._append(f"[UI3] curve_saved -> {curve_json_path}")
        self.refresh_from_context()

    def refresh_from_context(self) -> None:
        run_dir = (self._ctx.get("run_dir") or "").strip()
        if not run_dir:
            self.lbl_input_status_value.setText("Not Ready")
            self._append("[UI4] Waiting for run context from previous tabs.")
            self._refresh_preview_pngs()
            return

        info = self._backend_collect_ui4_run_inputs(run_dir)
        self._last_info = info
        self._load_ui4_summary_for_current_run()
        if not info.get("ok", False):
            self.lbl_input_status_value.setText("Not Ready")
            self._append(f"[UI4] Error: {info.get('error', 'unknown error')}")
            self._refresh_preview_pngs()
            return

        ready = bool(info.get("ready_for_ui4"))
        self.lbl_input_status_value.setText("Ready" if ready else "Not Ready")

        paths = info.get("paths", {})
        counts = info.get("counts", {})
        missing = info.get("missing_required", [])

        lines = [
            f"[UI4] Refresh run: {os.path.basename(run_dir)}",
            f"  Input dir: {paths.get('input_dir') or '-'}",
            f"  DEM: {paths.get('dem') or '-'}",
            f"  Mask (optional): {paths.get('mask_tif') or '-'}",
            f"  UI3 curve dir: {paths.get('ui3_curve_dir') or '-'}",
            f"  UI3 groups dir: {paths.get('ui3_groups_dir') or '-'}",
            f"  NURBS curves (CL/ML pattern): {counts.get('nurbs_curves', 0)}",
            f"  Groups: {counts.get('groups', 0)}",
            f"  NURBS info: {counts.get('nurbs_info', 0)}",
            f"  Anchors: {'yes' if paths.get('anchors_json') else 'no'}",
        ]
        if missing:
            lines.append("  Missing required: " + ", ".join(missing))
        else:
            lines.append("  Required inputs look ready for UI4 (DEM .tif + NURBS CL/ML curves).")

        self.status_box.clear()
        for ln in lines:
            self._append(ln)
        self._refresh_preview_pngs()

    def _on_generate_contours(self) -> None:
        run_dir = (self._ctx.get("run_dir") or "").strip()
        if not run_dir:
            self._append("[UI4] Cannot generate contours: missing run context.")
            return
        try:
            self._append("[UI4] Generating contour previews...")
            contour_kwargs = self._contour_param_values()
            err = self._validate_contour_params(contour_kwargs)
            if err:
                self._append(f"[UI4] {err}")
                return
            self._append(
                "[UI4] Contour settings: "
                f"surface(step={contour_kwargs['surface_interval_m']}, "
                f"zmin={contour_kwargs['surface_z_min']}, zmax={contour_kwargs['surface_z_max']}), "
                f"depth(step={contour_kwargs['depth_interval_m']}, "
                f"zmin={contour_kwargs['depth_z_min']}, zmax={contour_kwargs['depth_z_max']})"
            )
            res = self._backend_render_ui4_contours_for_run(run_dir, log_fn=self._append, **contour_kwargs)
            if not res.get("ok", False):
                self._append(f"[UI4] Contours failed: {res.get('error', 'unknown error')}")
                if "No UI4 kriging rasters found" in str(res.get("error", "")):
                    self._append("[UI4] Hint: click 'Calculate Kriging' first, then 'Preview'.")
                return
            items = res.get("items", {})
            for key in ("surface", "depth"):
                it = items.get(key, {})
                if it.get("ok"):
                    self._append(f"[UI4] {key} contours PNG: {it.get('png_path')}")
                elif it:
                    self._append(f"[UI4] {key} contours error: {it.get('error')}")
            if res.get("summary_json"):
                self._append(f"[UI4] Contour summary: {res.get('summary_json')}")
            self._append("[UI4] Contour previews generated.")
            self._refresh_preview_pngs(prefer_surface=True)
        except Exception as e:
            self._append(f"[UI4] Contours exception: {e}")

    def _on_run_ui4(self) -> None:
        run_dir = (self._ctx.get("run_dir") or "").strip()
        if not run_dir:
            self._append("[UI4] Cannot run UI4: missing run context.")
            return
        try:
            self._append("[UI4] Running kriging backend...")
            res = self._backend_run_ui4_kriging_for_run(run_dir, log_fn=self._append)
            if not res.get("ok", False):
                self._append(f"[UI4] Kriging failed: {res.get('error', 'unknown error')}")
                missing = res.get("missing_required", [])
                if missing:
                    self._append("[UI4] Missing required inputs: " + ", ".join(map(str, missing)))
                return
            outputs = res.get("outputs", {})
            self._append(f"[UI4] Surface raster: {outputs.get('slip_surface_tif')}")
            self._append(f"[UI4] Depth raster: {outputs.get('slip_depth_tif')}")
            self._append(f"[UI4] Variance raster: {outputs.get('slip_depth_variance_tif')}")
            if outputs.get("slip_surface_masked_tif"):
                self._append(f"[UI4] Surface raster (masked): {outputs.get('slip_surface_masked_tif')}")
            if outputs.get("slip_depth_masked_tif"):
                self._append(f"[UI4] Depth raster (masked): {outputs.get('slip_depth_masked_tif')}")
            if outputs.get("summary_json"):
                self._append(f"[UI4] Summary: {outputs.get('summary_json')}")
            self._load_ui4_summary_for_current_run()
            self.refresh_from_context()
            self._append("[UI4] Kriging completed.")
            self._refresh_preview_pngs()
        except Exception as e:
            self._append(f"[UI4] Kriging exception: {e}")

    def reset_session(self) -> None:
        self._ctx = {"project": "", "run_label": "", "run_dir": ""}
        self._last_info = {}
        self._last_ui4_summary = {}
        self.lbl_project_value.setText("-")
        self.lbl_run_label_value.setText("-")
        self.lbl_input_status_value.setText("Not Ready")
        self.status_box.clear()
        self.lbl_preview_status.setText("Preview: -")
        self.preview_file_combo.blockSignals(True)
        self.preview_file_combo.clear()
        self.preview_file_combo.blockSignals(False)
        self._preview_png_paths = []
        self.preview_view.clear_image()
        self.surface_auto_range.setChecked(True)
        self.depth_auto_range.setChecked(True)
        self.surface_step.setValue(1.0)
        self.depth_step.setValue(1.0)
        self.surface_zmin.setValue(0.0)
        self.surface_zmax.setValue(0.0)
        self.depth_zmin.setValue(0.0)
        self.depth_zmax.setValue(0.0)
