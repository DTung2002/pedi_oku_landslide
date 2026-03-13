from __future__ import annotations

import os
import sys
from pathlib import Path

from pedi_oku_landslide.core.paths import APP_ROOT, INTERNAL_ROOT
from pedi_oku_landslide.ui.main_window import run_ui


def configure_runtime() -> None:
    """Normalize runtime paths for both source and frozen builds."""
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)
        os.environ["PROJ_LIB"] = str(base / "proj-data")
        os.environ["GDAL_DATA"] = str(base / "gdal-data")
        os.chdir(Path(sys.executable).parent)
        return

    os.chdir(Path(__file__).resolve().parents[2])


def launch() -> None:
    configure_runtime()
    run_ui(str(APP_ROOT), str(INTERNAL_ROOT))
