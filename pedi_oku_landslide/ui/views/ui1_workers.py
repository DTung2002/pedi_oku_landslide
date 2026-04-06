from time import perf_counter

from PyQt5.QtCore import QObject, pyqtSignal

from pedi_oku_landslide.pipeline.runners.ui1_backend import UI1BackendService


class UI1SadWorker(QObject):
    finished = pyqtSignal(dict, str)
    error = pyqtSignal(str)

    def __init__(self, backend: UI1BackendService, ctx, method: str, patch_size_m: float, search_radius_m: float, use_smoothed: bool):
        super().__init__()
        self.backend = backend
        self.ctx = ctx
        self.method = str(method or "traditional")
        self.patch_size_m = patch_size_m
        self.search_radius_m = search_radius_m
        self.use_smoothed = use_smoothed

    def run(self):
        try:
            t0 = perf_counter()
            print("[SAD] run_sad start")
            out = self.backend.run_sad(
                self.ctx,
                patch_size_m=self.patch_size_m,
                search_radius_m=self.search_radius_m,
                use_smoothed=self.use_smoothed,
                method=self.method,
                vlim_dz=None,
            )
            dt = perf_counter() - t0
            print(f"[SAD] run_sad done in {dt:.2f}s")
            self.finished.emit(out, str(out.get("method_label") or out.get("method") or self.method))
        except Exception as e:
            self.error.emit(str(e))


__all__ = ["UI1SadWorker"]
