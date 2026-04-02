from typing import Any, Dict, List, Optional, Tuple, TypedDict


class UI3Paths(TypedDict, total=False):
    dem: str
    dem_orig: str
    dx: str
    dy: str
    dz: str
    lines: str
    slip: str


class UI3Group(TypedDict, total=False):
    id: str
    start: float
    end: float
    color: Optional[str]
    start_reason: str
    end_reason: str


class UI3Profile(TypedDict, total=False):
    chain: Any
    x: Any
    y: Any
    elev: Any
    elev_s: Any
    dx: Any
    dy: Any
    dz: Any
    d_para: Any
    theta: Any
    slip_mask: Any
    slip_span: Tuple[float, float]
    elev_orig: Any
    profile_dem_source: str
    profile_dem_path: str


class UI3Context(TypedDict, total=False):
    project: str
    run_label: str
    run_dir: str
    base_dir: str


class UI3RenderResult(TypedDict, total=False):
    message: str
    path: Optional[str]


JSONDict = Dict[str, Any]
JSONList = List[Dict[str, Any]]
