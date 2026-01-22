import os, json
from dataclasses import dataclass, asdict

@dataclass
class AppSettings:
    ui_scale_percent: int = 100

def _settings_path(base_dir: str) -> str:
    cfg_dir = os.path.join(base_dir, "pedi_oku_landslide", "config")
    os.makedirs(cfg_dir, exist_ok=True)
    return os.path.join(cfg_dir, "app_settings.json")

def load_settings(base_dir: str) -> AppSettings:
    path = _settings_path(base_dir)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return AppSettings(**data)
        except Exception:
            pass
    return AppSettings()

def save_settings(base_dir: str, settings: AppSettings) -> None:
    path = _settings_path(base_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(settings), f, ensure_ascii=False, indent=2)
