# ui3_backend.py
from typing import Any, Dict

from pedi_oku_landslide.infrastructure.storage.ui3_paths import UI3RunPaths
from .anchor_boring_service import UI3AnchorBoringServiceMixin
from .group_curve_service import UI3GroupCurveServiceMixin
from .nurbs_state_service import UI3NurbsStateServiceMixin
from .profile_service import UI3ProfileServiceMixin
from .storage_export_service import UI3StorageExportServiceMixin


class UI3BackendService(
    UI3ProfileServiceMixin,
    UI3GroupCurveServiceMixin,
    UI3StorageExportServiceMixin,
    UI3AnchorBoringServiceMixin,
    UI3NurbsStateServiceMixin,
):
    def __init__(self, *, base_dir: str = "") -> None:
        self._ctx: Dict[str, str] = {"project": "", "run_label": "", "run_dir": "", "base_dir": base_dir}
        self._inputs: Dict[str, Any] = {}

    def set_context(self, project: str, run_label: str, run_dir: str, base_dir: str = "") -> Dict[str, Any]:
        if base_dir:
            self._ctx["base_dir"] = str(base_dir)
        self._ctx.update({"project": project, "run_label": run_label, "run_dir": run_dir})
        self._inputs = self.load_inputs()
        return dict(self._inputs)


__all__ = ["UI3BackendService"]
