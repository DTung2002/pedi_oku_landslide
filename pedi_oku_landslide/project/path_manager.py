import os, re, json, time
from dataclasses import dataclass

def _slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s.strip()).strip("_")
    return s or "id"

def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def _ensure_dir(path: str, exist_ok: bool = True) -> str:
    os.makedirs(path, exist_ok=exist_ok)
    return path

@dataclass
class AnalysisContext:
    project_id: str
    run_id: str
    base_dir: str
    project_dir: str     # output/<project_id>
    run_dir: str         # output/<project_id>/<run_id>
    in_dir: str          # .../input
    out_ui1: str         # .../ui1
    out_ui2: str         # .../ui2
    out_ui3: str         # .../ui3

    def path_in(self, *parts: str) -> str:
        p = os.path.join(self.in_dir, *parts); os.makedirs(os.path.dirname(p), exist_ok=True); return p
    def path_ui1(self, *parts: str) -> str:
        p = os.path.join(self.out_ui1, *parts); os.makedirs(os.path.dirname(p), exist_ok=True); return p
    def path_ui2(self, *parts: str) -> str:
        p = os.path.join(self.out_ui2, *parts); os.makedirs(os.path.dirname(p), exist_ok=True); return p
    def path_ui3(self, *parts: str) -> str:
        p = os.path.join(self.out_ui3, *parts); os.makedirs(os.path.dirname(p), exist_ok=True); return p

def create_project(base_dir: str, project_id: str) -> str:
    """Create project folder (once)."""
    pid = _slugify(project_id)
    proj_dir = _ensure_dir(os.path.join(base_dir, "output", pid), exist_ok=True)
    # lưu metadata sơ bộ
    proj_meta = os.path.join(proj_dir, "project.json")
    if not os.path.exists(proj_meta):
        with open(proj_meta, "w", encoding="utf-8") as f:
            json.dump({"project_id": pid, "created_at": _timestamp()}, f, ensure_ascii=False, indent=2)
    return proj_dir

def create_context(base_dir: str, project_id: str, run_label: str | None = None) -> AnalysisContext:
    """Create a new run under a project: output/<project>/<timestamp>[_label]/..."""
    pid = _slugify(project_id)
    proj_dir = create_project(base_dir, pid)

    ts = _timestamp()
    run_suffix = f"_{_slugify(run_label)}" if run_label else ""
    run_id = f"{ts}{run_suffix}"
    run_dir = os.path.join(proj_dir, run_id)

    # strict tạo lần đầu để không ghi đè nếu vô tình trùng
    os.makedirs(run_dir, exist_ok=False)
    in_dir  = _ensure_dir(os.path.join(run_dir, "input"), exist_ok=True)
    out_ui1 = _ensure_dir(os.path.join(run_dir, "ui1"),   exist_ok=True)
    out_ui2 = _ensure_dir(os.path.join(run_dir, "ui2"),   exist_ok=True)
    out_ui3 = _ensure_dir(os.path.join(run_dir, "ui3"),   exist_ok=True)

    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "project_id": pid,
            "run_id": run_id,
            "created_at": ts,
            "run_dir": run_dir.replace("\\","/"),
        }, f, ensure_ascii=False, indent=2)

    return AnalysisContext(
        project_id=pid,
        run_id=run_id,
        base_dir=base_dir,
        project_dir=proj_dir,
        run_dir=run_dir,
        in_dir=in_dir,
        out_ui1=out_ui1,
        out_ui2=out_ui2,
        out_ui3=out_ui3,
    )
# ==== Helpers for opening an existing run ====

import os
from dataclasses import replace

def is_valid_run_dir(run_dir: str) -> bool:
    """A valid run dir must contain 'input' and at least one 'ui*' folder."""
    if not os.path.isdir(run_dir):
        return False
    has_input = os.path.isdir(os.path.join(run_dir, "input"))
    has_any_ui = any(
        os.path.isdir(os.path.join(run_dir, d))
        for d in ("ui1", "ui2", "ui3")
    )
    return has_input and has_any_ui

def load_context_from_run_dir(base_dir: str, run_dir: str) -> AnalysisContext:
    """
    Build AnalysisContext from a run directory:
    .../output/<project_id>/<run_id>
    """
    run_dir = os.path.normpath(run_dir)
    project_dir = os.path.dirname(run_dir)
    project_id = os.path.basename(project_dir)
    run_id = os.path.basename(run_dir)

    if not is_valid_run_dir(run_dir):
        raise RuntimeError(f"'{run_dir}' is not a valid run folder. It must contain 'input' and 'ui*'.")

    ctx = AnalysisContext(
        project_id=project_id,
        run_id=run_id,
        base_dir=base_dir,
        project_dir=project_dir,
        run_dir=run_dir,
        in_dir=os.path.join(run_dir, "input"),
        out_ui1=os.path.join(run_dir, "ui1"),
        out_ui2=os.path.join(run_dir, "ui2"),
        out_ui3=os.path.join(run_dir, "ui3"),
    )
    return ctx
# ---- Backward-compat PathManager (shim) ----
# Thêm đoạn này xuống cuối file project/path_manager.py

class PathManager:
    """
    Lớp tương thích cũ cho các module đang import PathManager.
    Bọc quanh AnalysisContext và cung cấp các method đường dẫn
    mà UI/pipeline đang gọi.
    """
    def __init__(self, run_dir: str | None = None, base_dir: str | None = None, project_id: str | None = None):
        self._ctx: AnalysisContext | None = None
        self._base_dir = base_dir if base_dir is not None else os.getcwd()
        if run_dir:
            # mở context có sẵn từ run_dir
            self._ctx = load_context_from_run_dir(self._base_dir, run_dir)
        elif project_id:
            # tạo context mới (dùng khi Confirm Input khởi tạo run)
            self._ctx = create_context(self._base_dir, project_id)
        # nếu cả run_dir và project_id đều None -> chờ set_run_dir()

    # ---------- context ----------
    def context(self) -> AnalysisContext | None:
        return self._ctx

    def set_run_dir(self, run_dir: str) -> None:
        self._ctx = load_context_from_run_dir(self._base_dir, run_dir)

    def run_dir(self) -> str:
        return self._ctx.run_dir if self._ctx else ""

    def ensure_dirs(self) -> None:
        if not self._ctx:
            raise RuntimeError("PathManager: context is not set.")
        # các thư mục đã được tạo khi create_context / load_context, nhưng đảm bảo lại
        for d in (self.input_dir(), self.ui1_dir(), self.ui2_dir(), self.ui3_dir()):
            os.makedirs(d, exist_ok=True)

    # ---------- dirs ----------
    def input_dir(self) -> str:
        return self._ctx.in_dir

    def ui1_dir(self) -> str:
        return self._ctx.out_ui1

    def ui2_dir(self) -> str:
        return self._ctx.out_ui2

    def ui3_dir(self) -> str:
        return self._ctx.out_ui3

    # ---------- input files (5) ----------
    def before_dem_tif(self) -> str:
        return os.path.join(self.input_dir(), "before_dem.tif")

    def asc_before(self) -> str:
        return os.path.join(self.input_dir(), "before.asc")

    def asc_after(self) -> str:
        return os.path.join(self.input_dir(), "after.asc")

    def asc_before_pz(self) -> str:
        return os.path.join(self.input_dir(), "before_pz.asc")

    def asc_after_pz(self) -> str:
        return os.path.join(self.input_dir(), "after_pz.asc")

    # ---------- UI1 outputs ----------
    def dx_path(self) -> str:
        return os.path.join(self.ui1_dir(), "dx.tif")

    def dy_path(self) -> str:
        return os.path.join(self.ui1_dir(), "dy.tif")

    def dz_path(self) -> str:
        return os.path.join(self.ui1_dir(), "dz.tif")

    def detect_mask_path(self) -> str:
        return os.path.join(self.ui1_dir(), "detect_mask.tif")

    def detect_overlay_path(self) -> str:
        return os.path.join(self.ui1_dir(), "detect_overlay.png")

    # ---------- UI2 ----------
    def sections_csv(self) -> str:
        return os.path.join(self.ui2_dir(), "sections.csv")

    def sections_json(self) -> str:
        return os.path.join(self.ui2_dir(), "sections.json")

    # ---------- UI3 ----------
    def ui3_groups_dir(self) -> str:
        return os.path.join(self.ui3_dir(), "groups")

    def ui3_curves_dir(self) -> str:
        return os.path.join(self.ui3_dir(), "curves")
