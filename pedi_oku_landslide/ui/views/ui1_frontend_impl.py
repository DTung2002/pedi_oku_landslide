# pedi_oku_landslide/ui/views/ui1_frontend_impl.py
from typing import Optional
from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5.QtWidgets import QWidget, QSplitter
from .ui1_build_mixin import UI1BuildMixin
from .ui1_actions_mixin import UI1ActionsMixin
from .ui1_layout_mixin import UI1LayoutMixin
from .ui1_run_mixin import UI1RunMixin
from .ui1_status_mixin import UI1StatusMixin
from pedi_oku_landslide.pipeline.runners.ui1_backend import UI1BackendService
from pedi_oku_landslide.ui.layout_constants import LEFT_DEFAULT_W, LEFT_MIN_W


class AnalyzeTab(UI1BuildMixin, UI1ActionsMixin, UI1RunMixin, UI1StatusMixin, UI1LayoutMixin, QWidget):
    # phát cho MainWindow khi đã render vectors xong → enable Section tab
    vectors_rendered = pyqtSignal(str, str, str)  # project, run_label, run_dir

    """
    Left: Project/Run + file pickers + Confirm Input + Processing + Status
    Right: UI1 preview (viewer with zoom toolbar inside)
    """

    def __init__(self, base_dir: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.base_dir = base_dir
        self._backend = UI1BackendService(base_dir)
        self._last_run_dir: Optional[str] = None
        self._splitter: Optional[QSplitter] = None
        self._left_min_w = LEFT_MIN_W
        self._left_default_w = LEFT_DEFAULT_W
        self._pending_init_splitter = True
        self._vec_live_timer = QTimer(self)
        self._vec_live_timer.setSingleShot(True)
        self._vec_live_timer.setInterval(80)
        self._vec_live_timer.timeout.connect(self._on_vec_live_tick)
        self._build_ui()
