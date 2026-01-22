import sys
import os
from pathlib import Path

if getattr(sys, "frozen", False):
    base = Path(sys._MEIPASS)
    os.environ["PROJ_LIB"]  = str(base / "proj-data")
    os.environ["GDAL_DATA"] = str(base / "gdal-data")
    os.chdir(Path(sys.executable).parent)
else:
    os.chdir(Path(__file__).parent)

from pedi_oku_landslide.core.paths import APP_ROOT, INTERNAL_ROOT
from pedi_oku_landslide.ui.main_window import run_ui

if __name__ == "__main__":
    # Use the robust paths
    run_ui(str(APP_ROOT), str(INTERNAL_ROOT))