import json
import os
from typing import Optional, Tuple

from .ui1_types import AnalysisContext


def mask_source_info(run_dir: str) -> dict:
    rd = str(run_dir or "").strip()
    if not rd:
        return {"label": "Mask source: not set", "dxf_path": ""}

    ui1_dir = os.path.join(rd, "ui1")
    mask_tif = os.path.join(ui1_dir, "landslide_mask.tif")
    meta_json = os.path.join(ui1_dir, "mask_from_dxf_meta.json")

    if not os.path.exists(mask_tif):
        return {"label": "Mask source: not set", "dxf_path": ""}

    dxf_path = ""
    if os.path.exists(meta_json):
        try:
            with open(meta_json, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
            dxf_path = str(meta.get("dxf_path") or "").strip()
        except Exception:
            dxf_path = ""

    if dxf_path:
        return {
            "label": f"Mask source: DXF ({os.path.basename(dxf_path)})",
            "dxf_path": dxf_path,
        }
    return {"label": "Mask source: auto/existing mask raster", "dxf_path": ""}


def read_sad_method(run_dir: str, default: str = "traditional") -> str:
    meta_path = os.path.join(run_dir, "ui1", "sad_meta.json")
    if not os.path.isfile(meta_path):
        return default
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return str(meta.get("method", default))
    except Exception:
        return default


def existing_run_preview(ctx: AnalysisContext) -> Tuple[Optional[str], Optional[str]]:
    before_hs = os.path.join(ctx.out_ui1, "before_asc_hillshade.png")
    after_hs = os.path.join(ctx.out_ui1, "after_asc_hillshade.png")
    dz_png = os.path.join(ctx.out_ui1, "dz.png")
    overlay = os.path.join(ctx.out_ui1, "landslide_overlay.png")
    vectors = os.path.join(ctx.out_ui1, "vectors_overlay.png")

    def _exists(path: str) -> bool:
        return os.path.isfile(path)

    if _exists(dz_png) and _exists(overlay):
        return dz_png, overlay
    if _exists(before_hs) and _exists(after_hs):
        return before_hs, after_hs
    if _exists(dz_png) and _exists(vectors):
        return dz_png, vectors
    if _exists(dz_png):
        return dz_png, dz_png
    return None, None


def post_sad_preview(run_dir: str) -> Tuple[Optional[str], Optional[str]]:
    overlay = os.path.join(run_dir, "ui1", "landslide_overlay.png")
    vectors = os.path.join(run_dir, "ui1", "vectors_overlay.png")
    dz_png = os.path.join(run_dir, "ui1", "dz.png")
    left_img = overlay if os.path.exists(overlay) else dz_png
    right_img = vectors if os.path.exists(vectors) else dz_png
    return left_img, right_img


__all__ = [
    "existing_run_preview",
    "mask_source_info",
    "post_sad_preview",
    "read_sad_method",
]
