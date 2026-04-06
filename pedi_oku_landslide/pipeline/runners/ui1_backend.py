from typing import Optional

from pedi_oku_landslide.services.session_store import AnalysisContext

from .ui1.ui1_context import (
    build_context_from_run_dir,
    create_analysis_context,
    is_valid_run_dir,
    load_analysis_context,
    parse_label_from_run_id,
)
from .ui1.ui1_ingest import resolve_run_input_path, run_ingest
from .ui1.ui1_processing import render_vectors, run_detect, run_mask_from_dxf, run_sad, run_smooth
from .ui1.ui1_render import export_vectors_json
from .ui1.ui1_run_state import existing_run_preview, mask_source_info, post_sad_preview, read_sad_method


class UI1BackendService:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir

    def set_base_dir(self, base_dir: str) -> None:
        self.base_dir = base_dir

    def create_context(self, project: str, run_label: Optional[str] = None) -> AnalysisContext:
        return create_analysis_context(self.base_dir, project, run_label)

    def context_from_run_dir(self, run_dir: str) -> AnalysisContext:
        return build_context_from_run_dir(self.base_dir, run_dir)

    def load_context_from_run_dir(self, run_dir: str) -> AnalysisContext:
        return load_analysis_context(self.base_dir, run_dir)

    def is_valid_run_dir(self, run_dir: str) -> bool:
        return is_valid_run_dir(run_dir)

    def parse_label_from_run_id(self, run_id: str) -> str:
        return parse_label_from_run_id(run_id)

    def confirm_input(self, project: str, run_label: Optional[str], files: dict) -> tuple[AnalysisContext, dict]:
        ctx = self.create_context(project, run_label)
        info = run_ingest(ctx, files)
        return ctx, info

    def run_smooth(self, ctx: AnalysisContext, **kwargs) -> dict:
        return run_smooth(ctx, **kwargs)

    def run_sad(self, ctx: AnalysisContext, **kwargs) -> dict:
        return run_sad(ctx, **kwargs)

    def run_detect(self, ctx: AnalysisContext, **kwargs) -> dict:
        return run_detect(ctx, **kwargs)

    def run_mask_from_dxf(self, ctx: AnalysisContext, dxf_path: str) -> dict:
        return run_mask_from_dxf(ctx, dxf_path)

    def render_vectors(self, ctx: AnalysisContext, **kwargs) -> dict:
        return render_vectors(ctx, **kwargs)

    def export_vectors_json(self, ctx: AnalysisContext, **kwargs) -> str:
        return export_vectors_json(ctx, **kwargs)

    def resolve_run_input_path(self, run_dir: str, key: str) -> str:
        return resolve_run_input_path(run_dir, key)

    def mask_source_info(self, run_dir: str) -> dict:
        return mask_source_info(run_dir)

    def read_sad_method(self, run_dir: str, default: str = "traditional") -> str:
        return read_sad_method(run_dir, default=default)

    def existing_run_preview(self, ctx: AnalysisContext):
        return existing_run_preview(ctx)

    def post_sad_preview(self, run_dir: str):
        return post_sad_preview(run_dir)


__all__ = [
    "AnalysisContext",
    "UI1BackendService",
    "build_context_from_run_dir",
    "create_analysis_context",
    "export_vectors_json",
    "is_valid_run_dir",
    "load_analysis_context",
    "parse_label_from_run_id",
    "render_vectors",
    "resolve_run_input_path",
    "run_detect",
    "run_ingest",
    "run_mask_from_dxf",
    "run_sad",
    "run_smooth",
]
