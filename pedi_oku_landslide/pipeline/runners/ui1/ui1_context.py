import os

from pedi_oku_landslide.services.session_store import (
    AnalysisContext,
    create_context,
    is_valid_run_dir,
    load_context_from_run_dir,
)


def create_analysis_context(base_dir: str, project: str, run_label: str | None = None) -> AnalysisContext:
    return create_context(base_dir, project, run_label if run_label else None)


def load_analysis_context(base_dir: str, run_dir: str) -> AnalysisContext:
    return load_context_from_run_dir(base_dir, run_dir)


def build_context_from_run_dir(base_dir: str, run_dir: str) -> AnalysisContext:
    parts = os.path.normpath(run_dir).split(os.sep)
    if len(parts) < 2:
        raise RuntimeError("Unexpected run folder structure.")
    run_id = parts[-1]
    project = parts[-2]
    return AnalysisContext(
        project_id=project,
        run_id=run_id,
        base_dir=base_dir,
        project_dir=os.path.join(base_dir, "output", project),
        run_dir=run_dir,
        in_dir=os.path.join(run_dir, "input"),
        out_ui1=os.path.join(run_dir, "ui1"),
        out_ui2=os.path.join(run_dir, "ui2"),
        out_ui3=os.path.join(run_dir, "ui3"),
    )


def parse_label_from_run_id(run_id: str) -> str:
    parts = run_id.split("_", 2)
    return parts[2] if len(parts) == 3 else ""


__all__ = [
    "AnalysisContext",
    "build_context_from_run_dir",
    "create_analysis_context",
    "is_valid_run_dir",
    "load_analysis_context",
    "parse_label_from_run_id",
]
