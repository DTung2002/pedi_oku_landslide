import sys
import os
from pathlib import Path


def get_roots() -> tuple[str, str]:
    """
    Returns (APP_ROOT, INTERNAL_ROOT).

    1. APP_ROOT: Where the application runs and stores 'output'.
       - In Dev: Project Root
       - In Exe: The folder containing the .exe file.

    2. INTERNAL_ROOT: Where the assets/code live.
       - In Dev: Project Root
       - In Exe: The temp _MEIPASS folder (PyInstaller).
    """
    if getattr(sys, 'frozen', False):
        # --- EXE MODE (PyInstaller) ---
        app_root = os.path.dirname(sys.executable)
        internal_root = sys._MEIPASS
    else:
        # --- DEV MODE (Python) ---
        current_file = os.path.abspath(__file__)
        core_dir = os.path.dirname(current_file)
        pkg_dir = os.path.dirname(core_dir)
        project_root = os.path.dirname(pkg_dir)

        app_root = project_root
        internal_root = project_root

    return str(app_root), str(internal_root)

# Initialize globals
APP_ROOT, INTERNAL_ROOT = get_roots()

# Ensure output folder exists in the writable APP_ROOT
OUTPUT_ROOT = os.path.join(APP_ROOT, "output")
os.makedirs(OUTPUT_ROOT, exist_ok=True)