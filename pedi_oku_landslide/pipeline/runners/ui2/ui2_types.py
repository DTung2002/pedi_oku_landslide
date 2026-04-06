from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
from rasterio.transform import Affine


class UI2Context(TypedDict):
    project: str
    run_label: str
    run_dir: str
    ui1_dir: str
    ui2_dir: str
    after_tif: str
    mask_tif: Optional[str]
    missing: List[str]
    ready: bool


class UI2LayerPayload(TypedDict):
    dz: np.ndarray
    dx: Optional[np.ndarray]
    dy: Optional[np.ndarray]
    mask: np.ndarray
    after: np.ndarray
    transform: Affine
    inv_transform: Affine
    width: int
    height: int
    dem_path: str
    ui1_dir: str
    hillshade: np.ndarray
    heat_rgba: np.ndarray


class UI2SectionRow(TypedDict):
    idx: int
    x1: float
    y1: float
    x2: float
    y2: float
    line_id: str
    line_role: str
    direction_version: int
    chainage_origin: str


class UI2AutoLineParams(TypedDict, total=False):
    main_num_even: int
    main_offset_m: float
    cross_num_even: int
    cross_offset_m: float
    base_length_m: Optional[float]
    min_mag_thresh: float


class UI2AutoLineResult(TypedDict):
    main: List[Dict[str, Any]]
    cross: List[Dict[str, Any]]
    debug: Dict[str, Any]


class UI2IntersectionRecord(TypedDict, total=False):
    main_line_id: str
    cross_line_id: str
    x: float
    y: float
    row_index_main: int
    row_index_cross: int
