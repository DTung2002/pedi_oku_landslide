# Repository Architecture

## Main execution pipeline

1. `main.py` or `pedi_oku_landslide/__main__.py`
2. `pedi_oku_landslide/app/bootstrap.py`
3. `pedi_oku_landslide/ui/main_window.py`
4. `pedi_oku_landslide/app/workflow.py`
5. Active tabs and backends:
   - `ui/views/analyze_tab.py` -> `pipeline/ingest.py` -> `pipeline/steps/step_smooth.py` -> `pipeline/steps/step_sad.py` -> `pipeline/steps/step_detect.py` / `step_mask_dxf.py`
   - `ui/views/section_tab.py` -> `pipeline/runners/ui2_backend.py`
   - `ui/views/curve_tab.py` -> `pipeline/runners/ui3_backend.py`
   - `ui/views/ui4_frontend.py` -> `pipeline/runners/ui4_backend.py`

## Classification

### Core

- `main.py`
- `pedi_oku_landslide/__main__.py`
- `pedi_oku_landslide/app/bootstrap.py`
- `pedi_oku_landslide/app/workflow.py`
- `pedi_oku_landslide/ui/main_window.py`
- `pedi_oku_landslide/ui/views/analyze_tab.py`
- `pedi_oku_landslide/ui/views/section_tab.py`
- `pedi_oku_landslide/ui/views/curve_tab.py`
- `pedi_oku_landslide/ui/views/ui4_frontend.py`
- `pedi_oku_landslide/pipeline/ingest.py`
- `pedi_oku_landslide/pipeline/steps/step_smooth.py`
- `pedi_oku_landslide/pipeline/steps/step_sad.py`
- `pedi_oku_landslide/pipeline/steps/step_detect.py`
- `pedi_oku_landslide/pipeline/steps/step_mask_dxf.py`
- `pedi_oku_landslide/pipeline/runners/ui2_backend.py`
- `pedi_oku_landslide/pipeline/runners/ui3_backend.py`
- `pedi_oku_landslide/pipeline/runners/ui4_backend.py`
- `pedi_oku_landslide/services/session_store.py`
- `pedi_oku_landslide/services/app_settings.py`
- `pedi_oku_landslide/ui/components/image_pair_viewer.py`
- `pedi_oku_landslide/core/paths.py`
- `pedi_oku_landslide/core/gis_utils.py`
- `pedi_oku_landslide/core/analysis.py`
- `pedi_oku_landslide/config/settings.py`

### Support

- `pedi_oku_landslide/project/path_manager.py` (compatibility shim)
- `pedi_oku_landslide/project/settings_store.py` (compatibility shim)
- `pedi_oku_landslide/ui/views/settings_dialog.py`
- `pedi_oku_landslide/ui/views/settings_tab.py`
- `pedi_oku_landslide/ui/widgets/file_picker.py`
- `pedi_oku_landslide/ui/views/ui/ui1_viewer.py` (compatibility shim)
- `pedi_oku_landslide/core/data_models.py`

### Experimental / Legacy

- `pedi_oku_landslide/experiments/legacy_ui/ui1_frontend.py`
- `pedi_oku_landslide/experiments/legacy_ui/ui2_frontend.py`
- `pedi_oku_landslide/experiments/legacy_ui/ui3_frontend_old.py`
- `pedi_oku_landslide/pipeline/runners/ui1_backend.py`
- `pedi_oku_landslide/ui/dual_viewer.py`

### Unused or likely unused in the live pipeline

- `pedi_oku_landslide/core/io_raster.py`
- `pedi_oku_landslide/core/data_models.py`

## Target structure

The repository now trends toward:

- `main.py`: readable entry point
- `pedi_oku_landslide/app`: bootstrap and workflow orchestration
- `pedi_oku_landslide/ui`: active UI shell, views, and shared components
- `pedi_oku_landslide/pipeline`: active processing steps and tab-specific backends
- `pedi_oku_landslide/services`: session and settings services
- `pedi_oku_landslide/core`: low-level reusable processing helpers
- `pedi_oku_landslide/config`: configuration loading and config files
- `pedi_oku_landslide/experiments`: isolated legacy UI prototypes
