from typing import Any, Dict

from .ui3_frontend_impl import CurveAnalyzeTab
from pedi_oku_landslide.ui.controllers.ui3_line_controller import WORKFLOW_GROUPING_PARAMS
from pedi_oku_landslide.pipeline.runners.ui3.ui3_grouping import WORKFLOW_GROUP_MIN_LEN_M
from pedi_oku_landslide.pipeline.runners.ui3.ui3_storage import SECTION_CHAINAGE_ORIGIN

Section = Dict[str, Any]

__all__ = [
    "CurveAnalyzeTab",
    "SECTION_CHAINAGE_ORIGIN",
    "Section",
    "WORKFLOW_GROUP_MIN_LEN_M",
    "WORKFLOW_GROUPING_PARAMS",
]
