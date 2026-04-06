import os
from typing import List, Optional, Tuple

from pedi_oku_landslide.pipeline.ingest import resolve_run_input_path


def file_exists(path: str) -> bool:
    return bool(path) and os.path.isfile(path)


def resolve_after_tif(run_dir: str) -> str:
    ui1_dir = os.path.join(run_dir, "ui1")
    after_smooth = os.path.join(ui1_dir, "after_asc_smooth.tif")
    after_base = resolve_run_input_path(run_dir, "after_asc")
    if file_exists(after_smooth):
        return after_smooth
    return after_base


def find_mask_tif(ui1_dir: str) -> Optional[str]:
    for name in ("detect_mask.tif", "mask.tif", "landslide_mask.tif", "mask_binary.tif"):
        path = os.path.join(ui1_dir, name)
        if file_exists(path):
            return path
    return None


def validate_run_inputs(run_dir: str) -> Tuple[List[str], str, Optional[str]]:
    ui1_dir = os.path.join(run_dir, "ui1")
    missing: List[str] = []
    for path in (
        os.path.join(ui1_dir, "dx.tif"),
        os.path.join(ui1_dir, "dy.tif"),
        os.path.join(ui1_dir, "dz.tif"),
    ):
        if not file_exists(path):
            missing.append(path)
    after_tif = resolve_after_tif(run_dir)
    if not file_exists(after_tif):
        missing.append(after_tif)
    mask_tif = find_mask_tif(ui1_dir)
    return missing, after_tif, mask_tif
