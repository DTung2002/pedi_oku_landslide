from typing import TypedDict

from pedi_oku_landslide.services.session_store import AnalysisContext


class UI1InputFiles(TypedDict):
    before_dem: str
    after_dem: str
    before_asc: str
    after_asc: str
    before_pz: str
    after_pz: str


class UI1VectorRenderSettings(TypedDict):
    step: int
    scale: float
    vector_color: str
    vector_width: float
    vector_opacity: float
