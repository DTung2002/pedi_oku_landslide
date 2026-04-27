import os
import sys

from PyQt5.QtWidgets import QFileDialog


class UI1RunMixin:
    def _on_open_run(self) -> None:
        if not self._last_run_dir:
            self._warn("No run folder to open.")
            return
        try:
            if os.name == "nt":
                os.startfile(self._last_run_dir)  # type: ignore[attr-defined]
            else:
                import subprocess
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.Popen([opener, self._last_run_dir])
        except Exception as e:
            self._err(f"Cannot open folder: {e}")

    def _on_open_existing_run(self) -> None:
        """
        Let user pick an existing run folder under output/<Project>/<RunID>,
        then load context, enable actions, and show previews.
        """
        try:
            root = os.path.join(self.base_dir, "output")
            sel = QFileDialog.getExistingDirectory(self, "Open existing run", root)
            if not sel:
                return

            # Validate selection
            if not self._backend.is_valid_run_dir(sel):
                # Nếu user chọn nhầm thư mục project
                proj_candidate = os.path.join(sel, "input")
                if os.path.isdir(proj_candidate):
                    self._err(f"Selected folder is not a run folder:\n{sel}\n"
                              f"It must be .../output/<Project>/<RunID> and contain 'input' and 'ui*'.")
                    return
                self._err("Please select a run folder like:\n.../output/<Project>/<YYYYmmdd_HHMMSS[_label]>")
                return

            # Build context
            ctx = self._backend.load_context_from_run_dir(sel)
            self._last_run_dir = ctx.run_dir

            # Fill fields
            self.edit_project.setText(ctx.project_id)
            self.edit_runlabel.setText(self._backend.parse_label_from_run_id(ctx.run_id))

            # Khóa 2 ô khi đã load run
            self._set_project_run_locked(True)
            self._ok("Fields locked (loaded run).")

            # Enable buttons based on what exists
            self.btn_smooth.setEnabled(True)
            self.cmb_method.setEnabled(True)
            self.btn_calc_sad.setEnabled(True)

            dx_ok = os.path.exists(os.path.join(ctx.out_ui1, "dx.tif"))
            dy_ok = os.path.exists(os.path.join(ctx.out_ui1, "dy.tif"))
            self.btn_detect.setEnabled(dx_ok and dy_ok)
            self.cmb_detect_source.setEnabled(dx_ok and dy_ok)
            self.btn_vectors.setEnabled(dx_ok and dy_ok)

            # Bật Open run
            self.btn_open_run.setEnabled(True)

            # Show best available preview
            left_img, right_img = self._backend.existing_run_preview(ctx)
            if left_img and right_img:
                self.viewer.show_pair(left_img, right_img)
            self._refresh_mask_source_ui(ctx.run_dir)
            self._restore_sad_method_from_run(ctx.run_dir)

            self._ok(
                "Opened existing run.\n"
                f"Project: {ctx.project_id}\n"
                f"Run:     {ctx.run_id}\n"
                f"Folder:  {ctx.run_dir}"
            )

            # Nếu run đã có SAD (dx/dy) → enable UI2 ngay
            if dx_ok and dy_ok:
                project = ctx.project_id
                run_label = (self.edit_runlabel.text() or "").strip()
                self.vectors_rendered.emit(project, run_label, ctx.run_dir)

        except Exception as e:
            self._err(f"Open error: {e}")

    def reset_session(self) -> None:
        """
        Đưa tab Analyze (UI1) về trạng thái ban đầu cho New Session.
        Được MainWindow gọi khi user tạo session mới.
        """
        # 0) Best-effort dừng thread SAD nếu còn chạy
        if hasattr(self, "_sad_thread"):
            try:
                if self._sad_thread and self._sad_thread.isRunning():
                    self._sad_thread.requestInterruption()
                    self._sad_thread.quit()
                    self._sad_thread.wait(2000)
            except Exception:
                pass

        # 1) Quên run hiện tại
        self._last_run_dir = None

        # 2) Mở khoá & clear Project / Run label
        self._set_project_run_locked(False)
        self.edit_project.clear()
        self.edit_runlabel.clear()

        # 3) Clear input file pickers
        for fp in (self.fp_adem, self.fp_basc, self.fp_aasc, self.fp_bpz, self.fp_apz):
            try:
                if hasattr(fp, "clear"):
                    fp.clear()
                elif hasattr(fp, "set_path"):
                    fp.set_path("")
                elif hasattr(fp, "edit_path"):
                    fp.edit_path.clear()
            except Exception:
                # Không fatal, chỉ cố gắng hết sức
                pass
        try:
            self.fp_mask_dxf.clear()
        except Exception:
            pass

        # 4) Đưa các nút về trạng thái ban đầu
        self.btn_open_run.setEnabled(False)
        self.btn_smooth.setEnabled(False)
        self.cmb_method.setEnabled(False)
        self.btn_calc_sad.setEnabled(False)
        self.btn_detect.setEnabled(False)
        self.cmb_detect_source.setEnabled(False)
        self.btn_vectors.setEnabled(False)
        # Nút Confirm Input luôn bật
        self.btn_confirm.setEnabled(True)

        # 5) Reset các thông số xử lý
        try:
            self.spin_smooth_param.setValue(2.0)
        except Exception:
            pass
        try:
            self._set_sad_method_combo("traditional")
        except Exception:
            pass
        try:
            self.spin_detect_thr.setValue(0.8)
        except Exception:
            pass
        try:
            self.cmb_detect_source.setCurrentIndex(0)
        except Exception:
            pass
        try:
            self.spin_vec_step.setValue(25)
        except Exception:
            pass
        try:
            self.spin_vec_scale.setValue(1.0)
        except Exception:
            pass
        try:
            self.combo_vec_color.setCurrentText("Blue")
        except Exception:
            pass
        try:
            self.sld_vec_size.setValue(100)
        except Exception:
            pass
        try:
            self.sld_vec_opacity.setValue(100)
        except Exception:
            pass

        # 6) Clear viewer (ảnh bên phải)
        try:
            if hasattr(self.viewer, "scene"):
                self.viewer.scene.clear()
            if hasattr(self.viewer, "caption"):
                self.viewer.caption.setText("")
            if hasattr(self.viewer, "view") and hasattr(self.viewer.view, "reset_view_transform"):
                self.viewer.view.reset_view_transform()
        except Exception:
            pass

        # 7) Clear status log
        try:
            self.status_text.clear()
        except Exception:
            pass
        try:
            self.lbl_mask_source.setText("Mask source: not set")
        except Exception:
            pass

        # 8) Thông báo nhẹ để debug
        try:
            self._info("Session reset.")
        except Exception:
            pass
