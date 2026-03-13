from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import QMessageBox


class MainWorkflowCoordinator:
    """Owns cross-tab execution flow so the window class stays presentational."""

    def __init__(self, window) -> None:
        self.window = window

    def wire(self) -> None:
        analyze_tab = self.window.analyze_tab
        section_tab = self.window.section_tab
        curve_tab = self.window.curve_tab

        if hasattr(analyze_tab, "vectors_rendered"):
            analyze_tab.vectors_rendered.connect(self.on_vectors_ready)
        else:
            analyze_tab.vectors_ready.connect(self.on_vectors_ready)

        if hasattr(section_tab, "sections_confirmed"):
            section_tab.sections_confirmed.connect(self.on_sections_confirmed)
        if hasattr(curve_tab, "curve_saved"):
            curve_tab.curve_saved.connect(self.on_curve_saved)

    def on_vectors_ready(self, project: str, run_label: str, run_dir: str) -> None:
        try:
            vec_step = self._read_optional_spinbox("spin_vec_step", int)
            vec_scale = self._read_optional_spinbox("spin_vec_scale", float)

            self.window.section_tab.set_context(
                project,
                run_label,
                run_dir,
                vec_step=vec_step,
                vec_scale=vec_scale,
            )
            self.window.curve_tab.set_context(project, run_label, run_dir)
            self.window.ui4_tab.set_context(project, run_label, run_dir)
            self.window.tabs.setCurrentIndex(self.window._idx_section)
        except Exception as exc:
            QMessageBox.warning(self.window, "Error", f"Cannot open Section tab:\n{exc}")

    def on_sections_confirmed(self, project: str, run_label: str, run_dir: str) -> None:
        try:
            self.window.curve_tab.set_context(project, run_label, run_dir)
            self.window.ui4_tab.set_context(project, run_label, run_dir)
            self.window.tabs.setCurrentIndex(self.window._idx_curve)
        except Exception as exc:
            QMessageBox.warning(self.window, "Error", f"Cannot open Curve tab:\n{exc}")

    def on_curve_saved(self, curve_json_path: str) -> None:
        try:
            self.window.ui4_tab.on_upstream_curve_saved(curve_json_path)
        except Exception as exc:
            QMessageBox.warning(self.window, "Error", f"Cannot update UI4 tab:\n{exc}")

    def reset_session(self) -> None:
        for tab_name in ("analyze_tab", "section_tab", "curve_tab", "ui4_tab"):
            tab = getattr(self.window, tab_name, None)
            if tab is not None and hasattr(tab, "reset_session"):
                tab.reset_session()
        self.window.tabs.setCurrentIndex(self.window._idx_analyze)

    def _read_optional_spinbox(self, attr_name: str, cast):
        widget = getattr(self.window.analyze_tab, attr_name, None)
        if widget is None:
            return None
        try:
            return cast(widget.value())
        except Exception:
            return None
