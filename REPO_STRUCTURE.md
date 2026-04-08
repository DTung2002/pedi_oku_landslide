# Repository Structure

Generated from the current repo layout.

Notes:
- Excludes `.git`, `__pycache__`, `output`, `build`, and `dist`
- Includes the current `legacy/` area created during refactor cleanup

```text
.
|-- .github/
|   `-- instructions/
|       |-- 01-app-entry-and-packaging.instructions.md
|       |-- 02-config-and-settings.instructions.md
|       |-- 03-core-gis-utilities.instructions.md
|       |-- 04-project-run-management.instructions.md
|       |-- 05-pipeline-ingest.instructions.md
|       |-- 06-pipeline-steps-displacement-detect.instructions.md
|       |-- 07-pipeline-ui-runners.instructions.md
|       |-- 08-ui-main-window-and-tabs.instructions.md
|       `-- 09-ui-legacy-frontends.instructions.md
|-- Landslide.spec
|-- main.py
|-- recover.py
|-- REPO_STRUCTURE.md
`-- pedi_oku_landslide/
    |-- __init__.py
    |-- __main__.py
    |-- app/
    |   |-- __init__.py
    |   |-- bootstrap.py
    |   `-- workflow.py
    |-- assets/
    |   |-- OKUYAMA Boring.png
    |   `-- fonts/
    |       |-- Inter-Black.otf
    |       |-- Inter-BlackItalic.otf
    |       |-- Inter-Bold.otf
    |       |-- Inter-BoldItalic.otf
    |       |-- Inter-ExtraBold.otf
    |       |-- Inter-ExtraBoldItalic.otf
    |       |-- Inter-ExtraLight.otf
    |       |-- Inter-ExtraLightItalic.otf
    |       |-- Inter-Italic.otf
    |       |-- Inter-Light.otf
    |       |-- Inter-LightItalic.otf
    |       |-- Inter-Medium.otf
    |       |-- Inter-MediumItalic.otf
    |       |-- Inter-Regular.otf
    |       |-- Inter-SemiBold.otf
    |       |-- Inter-SemiBoldItalic.otf
    |       |-- Inter-Thin.otf
    |       |-- Inter-ThinItalic.otf
    |       `-- Inter-V.ttf
    |-- config/
    |   |-- __init__.py
    |   |-- app_settings.json
    |   |-- config.yaml
    |   `-- settings.py
    |-- core/
    |   |-- __init__.py
    |   |-- analysis.py
    |   |-- gis_utils.py
    |   `-- paths.py
    |-- legacy/
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
    |   |   |-- ui3_backend.py
    |   |   |-- ui4_backend.py
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
    |   |   `-- ui3/
    |   |       |-- __init__.py
    |   |       |-- ui3_anchors.py
    |   |       |-- ui3_curve_fit.py
    |   |       |-- ui3_curve_state.py
    |   |       |-- ui3_exports.py
    |   |       |-- ui3_group_state.py
    |   |       |-- ui3_grouping.py
    |   |       |-- ui3_inputs.py
    |   |       |-- ui3_paths.py
    |   |       |-- ui3_profile_math.py
    |   |       |-- ui3_render.py
    |   |       |-- ui3_storage.py
    |   |       `-- ui3_workflows.py
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
        |-- views/
        |   |-- __init__.py
        |   |-- settings_dialog.py
        |   |-- ui1_frontend.py
        |   |-- ui1_frontend_impl.py
        |   |-- ui1_viewer.py
        |   |-- ui1_workers.py
        |   |-- ui2_dialogs.py
        |   |-- ui2_frontend.py
        |   |-- ui2_frontend_impl.py
        |   |-- ui2_layered_viewer.py
        |   |-- ui2_widgets.py
        |   |-- ui3_curve_panel.py
        |   |-- ui3_frontend.py
        |   |-- ui3_frontend_impl.py
        |   |-- ui3_group_panel.py
        |   |-- ui3_line_controller.py
        |   |-- ui3_preview_controller.py
        |   |-- ui3_preview_scene.py
        |   |-- ui3_widgets.py
        |   |-- ui4_frontend.py
        |   `-- ui/
        |       `-- __init__.py
        `-- widgets/
            |-- __init__.py
            `-- file_picker.py
```
