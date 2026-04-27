import os

from PyQt5.QtCore import QThread

from .ui1_workers import UI1SadWorker


class UI1ActionsMixin:
    def _on_confirm_input(self) -> None:
        project = (self.edit_project.text() or "").strip()
        run_label = (self.edit_runlabel.text() or "").strip()

        if not project:
            self._warn("Project name is required.")
            return

        files = {
            "after_dem":  self.fp_adem.path,
            "before_asc": self.fp_basc.path,
            "after_asc":  self.fp_aasc.path,
            "before_pz":  self.fp_bpz.path,
            "after_pz":   self.fp_apz.path,
        }
        if not all(files.values()):
            self._warn("Please select all 5 input files.")
            return

        try:
            ctx, info = self._backend.confirm_input(project, run_label if run_label else None, files)

            self._last_run_dir = info.get("run_dir")

            self.btn_open_run.setEnabled(True)
            self.btn_smooth.setEnabled(True)  # bật smooth sau khi confirm
            self.cmb_method.setEnabled(True)
            self.btn_calc_sad.setEnabled(True)
            self.btn_detect.setEnabled(False)
            self.cmb_detect_source.setEnabled(False)
            self.btn_vectors.setEnabled(False)

            self._ok(
                "Confirm Input completed.\n"
                f"Project: {ctx.project_id}\n"
                f"Run: {ctx.run_id}\n"
                f"Output: {self._last_run_dir}"
            )

            preview = info.get("preview", {})
            self.viewer.show_pair(
                preview.get("before_asc_hillshade_png"),
                preview.get("after_asc_hillshade_png"),
            )
            self._refresh_mask_source_ui(self._last_run_dir)

        except FileExistsError:
            self._err("Run folder already exists (rare). Please try again.")
        except Exception as e:
            self._err(f"Error: {e}")

    def _on_smooth(self) -> None:
        if not self._last_run_dir:
            self._warn("Please run 'Confirm Input' first.")
            return
        try:
            method = "Gaussian"
            param_m = float(self.spin_smooth_param.value())

            ctx = self._backend.context_from_run_dir(self._last_run_dir)
            out = self._backend.run_smooth(
                ctx,
                param_m=param_m,
                method=method,
                gaussian_sigma_percent=50.0,
            )

            smooth_note = f"filter={method}, radius={param_m}m"
            if method.lower() == "gaussian":
                smooth_note += ", sigma=50% radius, kernel=circle"

            self._ok(
                f"Smooth completed ({smooth_note}).\n"
                f"Outputs:\n"
                f" - {out['before_tif']}\n"
                f" - {out['after_tif']}\n"
                f" - {out['after_dem_tif']}"
            )
            self.viewer.show_pair(out["before_png"], out["after_png"])

        except Exception as e:
            self._err(f"Smooth error: {e}")

    def _on_calc_sad(self) -> None:
        if not self._last_run_dir:
            self._warn("Please run 'Confirm Input' first.")
            return
        try:
            ctx = self._backend.context_from_run_dir(self._last_run_dir)

            # --- disable nút khi đang chạy ---
            self.btn_calc_sad.setEnabled(False)
            self.btn_detect.setEnabled(False)
            self.btn_vectors.setEnabled(False)
            self.cmb_method.setEnabled(False)
            self._info("Calculating SAD + dZ in background...")

            # --- chạy nền bằng QThread ---
            self._sad_thread = QThread(self)
            self._sad_worker = UI1SadWorker(
                backend=self._backend,
                ctx=ctx,
                method=self._selected_sad_method_key(),
                patch_size_m=20.0,
                search_radius_m=2.0,
                use_smoothed=True
            )
            self._sad_worker.moveToThread(self._sad_thread)

            self._sad_thread.started.connect(self._sad_worker.run)
            self._sad_worker.finished.connect(self._on_sad_done)
            self._sad_worker.error.connect(self._on_sad_error)
            self._sad_worker.error.connect(self._sad_thread.quit)
            self._sad_worker.error.connect(self._sad_worker.deleteLater)
            self._sad_worker.finished.connect(self._sad_thread.quit)
            self._sad_worker.finished.connect(self._sad_worker.deleteLater)
            self._sad_thread.finished.connect(self._sad_thread.deleteLater)

            self._sad_thread.start()

        except Exception as e:
            self._err(f"SAD+dZ error: {e}")

    def _on_sad_done(self, out: dict, method: str) -> None:
        try:
            # bật lại nút
            self.cmb_method.setEnabled(True)
            self.btn_calc_sad.setEnabled(True)
            self.btn_detect.setEnabled(True)
            self.cmb_detect_source.setEnabled(True)
            self.btn_vectors.setEnabled(True)

            # thông báo
            self._ok(
                f"SAD + dZ completed (method: {method}).\n"
                f"- dX: {out.get('dx_tif', '')}\n"
                f"- dY: {out.get('dy_tif', '')}\n"
                f"- dZ: {out.get('dz_tif', '')}"
            )

            # hiển thị mặc định sau khi xong
            left_img, right_img = self._backend.post_sad_preview(str(self._last_run_dir or ""))
            self.viewer.show_pair(left_img, right_img)
            self._refresh_mask_source_ui(self._last_run_dir)

        except Exception as e:
            self._err(f"Post-process error: {e}")

    def _on_sad_error(self, msg: str) -> None:
        self.cmb_method.setEnabled(True)
        self.btn_calc_sad.setEnabled(True)
        self.btn_detect.setEnabled(True)
        self.btn_vectors.setEnabled(True)
        if self._last_run_dir:
            dx_ok = os.path.exists(os.path.join(self._last_run_dir, "ui1", "dx.tif"))
            dy_ok = os.path.exists(os.path.join(self._last_run_dir, "ui1", "dy.tif"))
            self.btn_detect.setEnabled(dx_ok and dy_ok)
            self.cmb_detect_source.setEnabled(dx_ok and dy_ok)
        else:
            self.btn_detect.setEnabled(False)
            self.cmb_detect_source.setEnabled(False)
        self._err(f"SAD+dZ error: {msg}")

    def _on_detect_requested(self) -> None:
        source = self.cmb_detect_source.currentData()
        if source == "dxf":
            self._on_import_dxf_mask()
        else:
            self._on_detect()

    def _on_detect(self) -> None:
        if not self._last_run_dir:
            self._warn("Please run 'Confirm Input' first.")
            return
        try:
            ctx = self._backend.context_from_run_dir(self._last_run_dir)

            thr_m = float(self.spin_detect_thr.value())
            out = self._backend.run_detect(ctx, method="threshold", threshold_m=thr_m)

            # Auto detect overrides manual DXF mask metadata.
            meta_json = os.path.join(ctx.out_ui1, "mask_from_dxf_meta.json")
            if os.path.exists(meta_json):
                try:
                    os.remove(meta_json)
                except Exception:
                    pass

            self._ok(f"Detected with threshold = {thr_m:.2f} m\n - {out['mask_tif']}")
            dz_png = os.path.join(ctx.out_ui1, "dz.png")
            overlay = out["mask_png"]  # bước detect đã lưu heatmap overlay
            self.viewer.show_pair(dz_png, overlay)
            self._refresh_mask_source_ui(ctx.run_dir)

        except Exception as e:
            self._err(f"Detect error: {e}")

    def _on_import_dxf_mask(self) -> None:
        if not self._last_run_dir:
            self._warn("Please run 'Confirm Input' first.")
            return
        dxf_path = (self.fp_mask_dxf.path or "").strip()
        if not dxf_path:
            self._warn("Please select a DXF boundary file first.")
            return
        try:
            ctx = self._backend.context_from_run_dir(self._last_run_dir)
            out = self._backend.run_mask_from_dxf(ctx, dxf_path)

            self._ok(
                f"DXF mask created.\n"
                f" - mask: {out.get('mask_tif', '')}\n"
                f" - polygons: {out.get('polygon_count', 0)}"
            )
            self._info(f"Mask pixels in-zone: {out.get('mask_pixels_positive', 0)}")

            dz_png = os.path.join(ctx.out_ui1, "dz.png")
            left_img = dz_png if os.path.exists(dz_png) else os.path.join(ctx.out_ui1, "after_asc_hillshade.png")
            right_img = out.get("mask_png", "")
            if left_img and right_img and os.path.exists(right_img):
                self.viewer.show_pair(left_img, right_img)
            self._refresh_mask_source_ui(ctx.run_dir)
        except Exception as e:
            self._err(f"DXF mask import error: {e}")

    def _on_render_vectors(self, quiet: bool = False, emit_signal: bool = True) -> None:
        if not self._last_run_dir:
            if not quiet:
                self._warn("Please run 'Confirm Input' first.")
            return
        try:
            step = int(self.spin_vec_step.value())
            scale = float(self.spin_vec_scale.value())
            color = str(self.combo_vec_color.currentText())
            size_mul = max(0.2, float(self.sld_vec_size.value()) / 100.0)
            width = 0.003 * size_mul
            opacity = max(0.0, min(1.0, float(self.sld_vec_opacity.value()) / 100.0))

            ctx = self._backend.context_from_run_dir(self._last_run_dir)
            out = self._backend.render_vectors(
                ctx,
                step=step,
                scale=scale,
                vector_color=color,
                vector_width=width,
                vector_opacity=opacity,
            )

            # Hiển thị: trái = overlay (nếu có) hoặc dz, phải = vectors overlay
            overlay = os.path.join(ctx.out_ui1, "landslide_overlay.png")
            left_img = overlay if os.path.exists(overlay) else os.path.join(ctx.out_ui1, "dz.png")
            self.viewer.show_pair(left_img, out["vectors_png"])

            if not quiet:
                self._ok(
                    f"Vectors rendered (step={step}, scale={scale}, "
                    f"size={self.sld_vec_size.value()}%, opacity={self.sld_vec_opacity.value()}%)."
                )

            # emit để MainWindow enable tab Section Selection
            if emit_signal:
                project = (self.edit_project.text() or "").strip()
                run_label = (self.edit_runlabel.text() or "").strip()
                self.vectors_rendered.emit(project, run_label, self._last_run_dir)

        except Exception as e:
            if quiet:
                return
            else:
                self._err(f"Render vectors error: {e}")

    def _on_vec_display_slider_changed(self) -> None:
        if not self._last_run_dir:
            return
        if not hasattr(self, "btn_vectors") or not self.btn_vectors.isEnabled():
            return
        self._vec_live_timer.start()

    def _on_vec_live_tick(self) -> None:
        self._on_render_vectors(quiet=True, emit_signal=False)

    def _selected_sad_method_key(self) -> str:
        data = self.cmb_method.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip()
        txt = (self.cmb_method.currentText() or "").strip().lower()
        return "ssd_opencv" if "opencv" in txt or "ssd" in txt else "traditional"

    def _set_sad_method_combo(self, method_key: str) -> None:
        idx = self.cmb_method.findData(method_key)
        if idx < 0:
            idx = self.cmb_method.findText("SAD")
        if idx >= 0:
            self.cmb_method.setCurrentIndex(idx)

    def _restore_sad_method_from_run(self, run_dir: str) -> None:
        self._set_sad_method_combo(self._backend.read_sad_method(run_dir, default="traditional"))
