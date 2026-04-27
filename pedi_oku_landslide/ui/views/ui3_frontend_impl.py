# pedi_oku_landslide/ui/views/ui3_frontend_impl.py
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5.QtWidgets import QGraphicsLineItem, QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsRectItem, QScrollArea, QSplitter, QWidget

from pedi_oku_landslide.pipeline.runners.ui3_backend import UI3BackendService
from pedi_oku_landslide.ui.controllers.ui3_preview_controller import UI3PreviewControllerMixin
from pedi_oku_landslide.ui.controllers.ui3_group_panel import UI3GroupPanelMixin
from pedi_oku_landslide.ui.controllers.ui3_curve_panel import UI3CurvePanelMixin
from pedi_oku_landslide.ui.controllers.ui3_line_controller import UI3LineControllerMixin
from pedi_oku_landslide.ui.layout_constants import LEFT_DEFAULT_W, LEFT_MIN_W
from .ui3_build_mixin import UI3BuildMixin
from .ui3_layout_mixin import UI3LayoutMixin
from .ui3_status_mixin import UI3StatusMixin


class CurveAnalyzeTab(
    UI3BuildMixin,
    UI3LayoutMixin,
    UI3StatusMixin,
    UI3PreviewControllerMixin,
    UI3GroupPanelMixin,
    UI3CurvePanelMixin,
    UI3LineControllerMixin,
    QWidget,
):
    """
    UI3 (refactor): khung làm việc phân tích đường cong.
    - Nhận context từ Analyze/Section (project/run/run_dir)
    - Đọc danh sách sections từ UI2/sections.csv
    - Cho phép chọn line; hiển thị status; vẽ placeholder đồ thị (dz & slope)
    """

    # (Optional) khi bạn muốn phát tín hiệu đã lưu JSON v.v.
    curve_saved = pyqtSignal(str)  # emit path
    _GROUND_EXPORT_STEP_M = 0.2

    def __init__(self, base_dir: str):
        super().__init__()
        self.base_dir = base_dir
        self._backend = UI3BackendService(base_dir=base_dir)
        self._ctx: Dict[str, str] = {"project": "", "run_label": "", "run_dir": ""}
        self._splitter: Optional[QSplitter] = None
        self._left_scroll: Optional[QScrollArea] = None
        self._left_min_w = LEFT_MIN_W
        self._left_default_w = LEFT_DEFAULT_W
        self._pending_init_splitter = True

        self._ax_top = None  # dict: {x_min,x_max,left_px,top_px,width_px,height_px}
        self._ax_bot = None

        # paths từ UI1/UI2
        self.dem_path = ""
        self.dem_path_raw = ""
        self.dem_path_smooth = ""
        self.ground_export_dem_path = ""
        self._default_profile_step_m = 0.20
        self.dx_path = ""
        self.dy_path = ""
        self.dz_path = ""
        self.lines_path = ""
        self.slip_path = ""
        self.profile_source_combo = None
        self.rdp_eps_spin = None

        # UI widgets chính (để dùng lại)
        self.line_combo = None
        self.status = None
        self.scene = None
        self.view = None
        self.group_table = None
        self.table_selector_combo = None
        # --- state for grouping/guide overlays (phải ở CurveAnalyzeTab) ---
        self._px_per_m: Optional[float] = None  # pixels per meter
        self._sec_len_m: Optional[float] = None  # chiều dài tuyến (m)
        self._group_bounds: Dict[str, List[float]] = {}  # {line_id: [x_m ...]}
        self._guide_lines_top: List[QGraphicsLineItem] = []
        self._guide_lines_bot: List[QGraphicsLineItem] = []
        self._group_bands_bot: List[QGraphicsRectItem] = []
        self._img_ground: Optional[QGraphicsPixmapItem] = None
        self._img_rate0: Optional[QGraphicsPixmapItem] = None
        self._curve_method_by_line: Dict[str, str] = {}
        self._active_prof: Optional[dict] = None
        self._active_groups: List[dict] = []
        self._active_base_curve: Optional[dict] = None
        self._active_curve: Optional[dict] = None
        self._active_global_fit_result: Optional[Dict[str, Any]] = None
        self._curve_overlay_item: Optional[QGraphicsPathItem] = None
        self._cp_overlay_items: List[Any] = []
        self._anchor_overlay_items: List[Any] = []
        self._ui2_intersections_cache: Optional[Dict[str, Any]] = None
        self._anchors_xyz_cache: Optional[Dict[str, Any]] = None
        self._boring_holes_data: Dict[str, Any] = {"version": 1, "distance_tolerance_m": 1.0, "items": []}
        self._nurbs_params_by_line: Dict[str, Dict[str, Any]] = {}
        self._nurbs_seed_method_by_line: Dict[str, str] = {}
        self._group_table_updating: bool = False
        self._boring_table_updating: bool = False
        self._nurbs_updating_ui: bool = False
        # True when background image already has a baked slip-curve (profile_*_nurbs.png).
        self._static_nurbs_bg_loaded: bool = False
        self._nurbs_live_timer = QTimer(self)
        self._nurbs_live_timer.setSingleShot(True)
        self._nurbs_live_timer.setInterval(30)
        self._nurbs_live_timer.timeout.connect(self._on_nurbs_live_tick)

        self._plot_x0_px = None  # ax_left_px trong PNG
        self._plot_w_px = None  # ax_width_px trong PNG
        self._x_min = None  # trục x (chainage) min trên hình
        self._x_max = None  # trục x (chainage) max trên hình
        self._profile_cursor_label = None

        self._build_ui()
