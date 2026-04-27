import os
from typing import Any, Dict, List, Optional

from pedi_oku_landslide.pipeline.runners.ui2.ui2_auto_lines import generate_auto_lines_from_arrays
from pedi_oku_landslide.pipeline.runners.ui2.ui2_intersections import (
    build_main_cross_intersections,
    save_main_cross_intersections,
)
from pedi_oku_landslide.pipeline.runners.ui2.ui2_paths import validate_run_inputs
from pedi_oku_landslide.pipeline.runners.ui2.ui2_polylines_storage import (
    read_polylines_json,
    write_polylines_json,
)
from pedi_oku_landslide.pipeline.runners.ui2.ui2_raster import load_layers as load_run_layers
from pedi_oku_landslide.pipeline.runners.ui2.ui2_raster import validate_context as validate_run_context
from pedi_oku_landslide.pipeline.runners.ui2.ui2_sections_storage import (
    SECTION_CHAINAGE_ORIGIN,
    SECTION_CSV_FIELDNAMES,
    SECTION_DIRECTION_VERSION,
    ensure_sections_csv_current,
    read_sections_csv_rows,
    write_sections_csv_rows,
)


class UI2BackendService:
    def __init__(self) -> None:
        self._ctx: Dict[str, Any] = {}

    def set_context(self, project: str, run_label: str, run_dir: str, base_dir: str = "") -> Dict[str, Any]:
        validated = validate_run_context(run_dir)
        self._ctx = {
            "project": project or "",
            "run_label": run_label or "",
            "run_dir": run_dir or "",
            "base_dir": base_dir or "",
            **validated,
        }
        return dict(self._ctx)

    def validate_context(self, run_dir: str) -> Dict[str, Any]:
        return validate_run_context(run_dir)

    def load_layers(self, run_dir: str, vector_settings: Dict[str, Any]) -> Dict[str, Any]:
        return load_run_layers(run_dir, vector_settings)

    def load_sections(self, run_dir: str) -> Dict[str, Any]:
        csv_path = os.path.join(run_dir, "ui2", "sections.csv")
        if not os.path.isfile(csv_path):
            return {"rows": [], "migrated": False, "csv_path": csv_path}
        rows, migrated = ensure_sections_csv_current(csv_path, run_dir=run_dir)
        return {"rows": rows, "migrated": migrated, "csv_path": csv_path}

    def save_sections(self, run_dir: str, rows: List[Dict[str, Any]]) -> str:
        csv_path = os.path.join(run_dir, "ui2", "sections.csv")
        write_sections_csv_rows(csv_path, rows)
        return csv_path

    def load_polylines(self, run_dir: str) -> Dict[str, Any]:
        json_path = os.path.join(run_dir, "ui2", "polylines.json")
        if not os.path.isfile(json_path):
            return {"rows": [], "json_path": json_path}
        return {"rows": read_polylines_json(json_path), "json_path": json_path}

    def save_polylines(self, run_dir: str, rows: List[Dict[str, Any]]) -> str:
        json_path = os.path.join(run_dir, "ui2", "polylines.json")
        write_polylines_json(json_path, rows)
        return json_path

    def generate_auto_lines(self, dx, dy, mask, transform, params: Dict[str, Any]) -> Dict[str, Any]:
        return generate_auto_lines_from_arrays(
            dx=dx,
            dy=dy,
            mask=mask,
            transform=transform,
            main_num_even=int(params.get("main_num_even", 0)),
            main_offset_m=float(params.get("main_offset_m", 0.0)),
            cross_num_even=int(params.get("cross_num_even", 0)),
            cross_offset_m=float(params.get("cross_offset_m", 0.0)),
            base_length_m=params.get("base_length_m"),
            min_mag_thresh=float(params.get("min_mag_thresh", 1e-4)),
        )

    def build_main_cross_intersections(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return build_main_cross_intersections(rows)

    def save_main_cross_intersections(self, run_dir: str, rows: List[Dict[str, Any]]) -> Optional[str]:
        return save_main_cross_intersections(os.path.join(run_dir, "ui2"), rows)

    def read_sections_csv_rows(self, csv_path: str) -> List[Dict[str, Any]]:
        return read_sections_csv_rows(csv_path)

    def write_sections_csv_rows(self, csv_path: str, rows: List[Dict[str, Any]]) -> None:
        write_sections_csv_rows(csv_path, rows)

    def reset_context(self) -> None:
        self._ctx.clear()



__all__ = [
    "SECTION_CHAINAGE_ORIGIN",
    "SECTION_CSV_FIELDNAMES",
    "SECTION_DIRECTION_VERSION",
    "UI2BackendService",
    "validate_run_inputs",
]
