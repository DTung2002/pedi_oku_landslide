# Repository Structure

Generated from the current repo layout after the staged folder refactor.

Notes:
- Excludes `.git`, `__pycache__`, `output`, `build`, and `dist`
- `ui/views/` and `pipeline/runners/ui3|ui4/` still contain compatibility shims during migration

```text
.
|-- .github/
|   `-- instructions/
|-- .gitignore
|-- Landslide.spec
|-- main.py
|-- recover.py
|-- REPO_STRUCTURE.md
|-- test_imports.py
`-- pedi_oku_landslide/
    |-- __init__.py
    |-- __main__.py
    |-- app/
    |   |-- __init__.py
    |   |-- bootstrap.py
    |   `-- workflow.py
    |-- application/
    |   |-- __init__.py
    |   |-- ui1/
    |   |   |-- __init__.py
    |   |   `-- service.py
    |   |-- ui2/
    |   |   |-- __init__.py
    |   |   `-- service.py
    |   |-- ui3/
    |   |   |-- __init__.py
    |   |   |-- exports.py
    |   |   |-- group_state.py
    |   |   |-- inputs.py
    |   |   |-- service.py
    |   |   `-- workflows.py
    |   `-- ui4/
    |       |-- __init__.py
    |       |-- inputs.py
    |       |-- service.py
    |       |-- state.py
    |       `-- workflows.py
    |-- assets/
    |   |-- OKUYAMA Boring.png
    |   `-- fonts/
    |-- config/
    |   |-- __init__.py
    |   |-- app_settings.json
    |   |-- config.yaml
    |   `-- settings.py
    |-- core/
    |   |-- __init__.py
    |   |-- analysis.py
    |   `-- gis_utils.py
    |-- domain/
    |   |-- __init__.py
    |   |-- ui1/
    |   |   `-- __init__.py
    |   |-- ui2/
    |   |   `-- __init__.py
    |   |-- ui3/
    |   |   |-- __init__.py
    |   |   |-- anchors.py
    |   |   |-- curve_fit.py
    |   |   |-- curve_state.py
    |   |   |-- grouping.py
    |   |   `-- profile.py
    |   `-- ui4/
    |       |-- __init__.py
    |       |-- boundary.py
    |       |-- contour.py
    |       |-- kriging.py
    |       |-- surface.py
    |       `-- types.py
    |-- infrastructure/
    |   |-- __init__.py
    |   |-- project/
    |   |   `-- __init__.py
    |   |-- raster/
    |   |   `-- __init__.py
    |   |-- rendering/
    |   |   |-- __init__.py
    |   |   `-- ui3_render.py
    |   `-- storage/
    |       |-- __init__.py
    |       |-- ui3_paths.py
    |       `-- ui3_storage.py
    |-- legacy/
    |   |-- README.md
    |   |-- core/
    |   |   `-- io_raster.py
    |   |-- pipeline/
    |   |   `-- runners/
    |   |       |-- ui2/
    |   |       |   `-- ui2_types.py
    |   |       `-- ui3/
    |   |           `-- ui3_types.py
    |   |-- project/
    |   |   `-- settings_store.py
    |   `-- ui/
    |       `-- views/
    |           |-- analyze_tab.py
    |           |-- curve_tab.py
    |           `-- section_tab.py
    |-- pipeline/
    |   |-- __init__.py
    |   |-- ingest.py
    |   |-- runners/
    |   |   |-- __init__.py
    |   |   |-- ui1_backend.py
    |   |   |-- ui2_backend.py
    |   |   |-- ui3_backend.py      # shim -> application/ui3/service.py
    |   |   |-- ui4_backend.py      # shim -> application/ui4/service.py
    |   |   |-- ui1/
    |   |   |   |-- __init__.py
    |   |   |   |-- ui1_context.py
    |   |   |   |-- ui1_ingest.py
    |   |   |   |-- ui1_processing.py
    |   |   |   |-- ui1_render.py
    |   |   |   |-- ui1_run_state.py
    |   |   |   `-- ui1_types.py
    |   |   |-- ui2/
    |   |   |   |-- __init__.py
    |   |   |   |-- ui2_auto_lines.py
    |   |   |   |-- ui2_intersections.py
    |   |   |   |-- ui2_paths.py
    |   |   |   |-- ui2_raster.py
    |   |   |   |-- ui2_sections_storage.py
    |   |   |   `-- ui2_visualization.py
    |   |   |-- ui3/                # compatibility shims during migration
    |   |   |   |-- __init__.py
    |   |   |   |-- ui3_anchors.py
    |   |   |   |-- ui3_curve_fit.py
    |   |   |   |-- ui3_curve_state.py
    |   |   |   |-- ui3_exports.py
    |   |   |   |-- ui3_group_state.py
    |   |   |   |-- ui3_grouping.py
    |   |   |   |-- ui3_inputs.py
    |   |   |   |-- ui3_paths.py
    |   |   |   |-- ui3_profile_math.py
    |   |   |   |-- ui3_render.py
    |   |   |   |-- ui3_storage.py
    |   |   |   `-- ui3_workflows.py
    |   |   `-- ui4/                # compatibility shims during migration
    |   |       |-- __init__.py
    |   |       |-- ui4_boundary.py
    |   |       |-- ui4_contour.py
    |   |       |-- ui4_inputs.py
    |   |       |-- ui4_kriging.py
    |   |       |-- ui4_render.py
    |   |       |-- ui4_state.py
    |   |       |-- ui4_surface.py
    |   |       `-- ui4_types.py
    |   `-- steps/
    |       |-- step_detect.py
    |       |-- step_mask_dxf.py
    |       |-- step_sad.py
    |       `-- step_smooth.py
    |-- project/
    |   `-- __init__.py
    |-- services/
    |   |-- __init__.py
    |   |-- app_settings.py
    |   `-- session_store.py
    `-- ui/
        |-- __init__.py
        |-- main_window.py
        |-- components/
        |   |-- __init__.py
        |   `-- image_pair_viewer.py
        |-- controllers/
        |   |-- __init__.py
        |   |-- ui3_curve_panel.py
        |   |-- ui3_group_panel.py
        |   |-- ui3_line_controller.py
        |   |-- ui3_preview_controller.py
        |   |-- ui4_preview_controller.py
        |   `-- ui4_run_controller.py
        |-- dialogs/
        |   |-- __init__.py
        |   |-- settings_dialog.py
        |   `-- ui2_dialogs.py
        |-- scenes/
        |   |-- __init__.py
        |   `-- ui3_preview_scene.py
        |-- views/
        |   |-- __init__.py
        |   |-- ui1_frontend.py
        |   |-- ui1_frontend_impl.py
        |   |-- ui1_viewer.py
        |   |-- ui1_workers.py
        |   |-- ui2_frontend.py
        |   |-- ui2_frontend_impl.py
        |   |-- ui2_layered_viewer.py
        |   |-- ui3_frontend.py
        |   |-- ui3_frontend_impl.py
        |   |-- ui4_frontend.py
        |   |-- settings_dialog.py      # shim
        |   |-- ui2_dialogs.py          # shim
        |   |-- ui2_widgets.py          # shim
        |   |-- ui3_curve_panel.py      # shim
        |   |-- ui3_group_panel.py      # shim
        |   |-- ui3_line_controller.py  # shim
        |   |-- ui3_preview_controller.py
        |   |-- ui3_preview_scene.py
        |   |-- ui3_widgets.py          # shim
        |   |-- ui4_preview_controller.py
        |   `-- ui4_run_controller.py
        `-- widgets/
            |-- __init__.py
            |-- file_picker.py
            |-- ui2_widgets.py
            `-- ui3_widgets.py
```
