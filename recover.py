import os
import glob
import time
import shutil
import sys

with open("recover_log.txt", "w") as f:
    sys.stdout = f
    sys.stderr = f
    
    appdata = os.environ.get("APPDATA")
    if not appdata:
        appdata = os.path.expanduser("~\\AppData\\Roaming")
    
    history_dir = os.path.join(appdata, "Code", "User", "History")
    
    candidates = []
    print(f"Searching in {history_dir}")
    for root, dirs, files in os.walk(history_dir):
        for file_name in files:
            path = os.path.join(root, file_name)
            try:
                mtime = os.path.getmtime(path)
                # Only care about files from recent 2 days
                if time.time() - mtime < 86400 * 2:
                    with open(path, "r", encoding="utf-8", errors="ignore") as file_obj:
                        content = file_obj.read()
                        if "def render_profile_png(" in content and "CURVATURE_THRESHOLD_PLOT_ABS" in content:
                            lines = len(content.splitlines())
                            candidates.append((mtime, path, lines))
            except Exception as e:
                pass
    
    candidates.sort(reverse=True) # newest first
    found_1900 = False
    for i, (mtime, path, lines) in enumerate(candidates):
        print(f"Found candidate: {path} | {time.ctime(mtime)} | {lines} lines")
        if lines >= 1800 and not found_1900:
            print(f"--> MATCH! Restoring {path} to ui3_backend_recovered.py")
            with open(path, "r", encoding="utf-8", errors="ignore") as src, open("ui3_backend_recovered.py", "w", encoding="utf-8") as dst:
                dst.write(src.read())
            found_1900 = True
    
    if not found_1900:
        print("Could not find any file >= 1800 lines.")
