# pedi_oku_landslide/ui/views/ui2_frontend_impl.py
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rasterio.transform import Affine
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QGraphicsSimpleTextItem, QSplitter, QWidget

from pedi_oku_landslide.pipeline.runners.ui2_backend import UI2BackendService
from pedi_oku_landslide.ui.layout_constants import LEFT_DEFAULT_W, LEFT_MIN_W
from .ui2_auto_roles_mixin import UI2AutoRolesMixin
from .ui2_build_mixin import UI2BuildMixin
from .ui2_confirm_mixin import UI2ConfirmMixin
from .ui2_context_mixin import UI2ContextMixin
from .ui2_layout_mixin import UI2LayoutMixin
from .ui2_sections_mixin import UI2SectionsMixin
from .ui2_session_status_mixin import UI2SessionStatusMixin

# ---------- UI2: Section Selection tab ----------


class SectionSelectionTab(
    UI2BuildMixin,
    UI2ContextMixin,
    UI2SectionsMixin,
    UI2AutoRolesMixin,
    UI2ConfirmMixin,
    UI2SessionStatusMixin,
    UI2LayoutMixin,
    QWidget,
):
    sections_confirmed = pyqtSignal(str, str, str)  # project, run_label, run_dir

    def __init__(self, base_dir: str, parent=None) -> None:
        super().__init__(parent)
        self.base_dir = base_dir
        self._backend = UI2BackendService()
        self._splitter: Optional[QSplitter] = None
        self._left_min_w = LEFT_MIN_W
        self._left_default_w = LEFT_DEFAULT_W
        self._pending_init_splitter = True

        # run context
        self._ctx_ready: bool = False
        self._ctx_project: str = ""
        self._ctx_runlabel: str = ""
        self._ctx_run_dir: str = ""
        self._ui1_dir: str = ""
        self._ui2_dir: str = ""

        # viewer items list init
        self._grid_items = []
        self._vec_items = []
        self._hill_item = None
        self._heat_item = None

        self.project: Optional[str] = None
        self.run_label: Optional[str] = None
        self.run_dir: Optional[str] = None
        self._section_lines: list[object] = []
        self._section_line_labels: list[QGraphicsSimpleTextItem] = []
        self._poly_section_lines: list[object] = []
        self._poly_section_line_labels: list[QGraphicsSimpleTextItem] = []
        self._preview_line: Optional[object] = None
        self._preview_label: Optional[QGraphicsSimpleTextItem] = None
        self._preview_source: Optional[str] = None

        # caches
        self._tr: Optional[Affine] = None
        self._inv_tr: Optional[Affine] = None
        self._dz: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._dy: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None  # uint8 0/1 aligned to dz grid
        self._dem_path: Optional[str] = None

        # vector drawing params (UI2-local, độc lập UI1)
        self._vec_step: int = 25
        self._vec_scale: float = 1.0
        self._vec_size_pct: int = 100
        self._vec_opacity_pct: int = 100
        self._vec_color: str = "blue"
        self._vec_pen_base: int = 1
        self._vec_arrow_base: float = 12.0

        self._sections: List[Any] = []
        self._section_meta: List[Dict[str, Any]] = []
        self._poly_sections: List[List[Tuple[float, float]]] = []
        self._poly_section_meta: List[Dict[str, Any]] = []

        self._updating_table: bool = False

        self._build_ui()
        self._wire()
