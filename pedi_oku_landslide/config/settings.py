import os, yaml
from dataclasses import dataclass

@dataclass
class AppConfig:
    ui_scale_percent: int = 100

@dataclass
class ProcessingConfig:
    smooth_size: int = 11
    sad_threshold: int = 800
    grouping_eps: float = 3.0
    grouping_min_cluster_size: int = 20

@dataclass
class CrsConfig:
    fallback_epsg: str | None = None

@dataclass
class Config:
    app: AppConfig
    processing: ProcessingConfig
    crs: CrsConfig

def load_config(base_dir: str) -> Config:
    path = os.path.join(base_dir, "pedi_oku_landslide", "config", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    app = y.get("app", {})
    proc = y.get("processing", {})
    crs  = y.get("crs", {})

    return Config(
        app=AppConfig(ui_scale_percent=app.get("ui_scale_percent", 100)),
        processing=ProcessingConfig(
            smooth_size=proc.get("smooth_size", 11),
            sad_threshold=proc.get("sad_threshold", 800),
            grouping_eps=proc.get("grouping", {}).get("eps", 3.0),
            grouping_min_cluster_size=proc.get("grouping", {}).get("min_cluster_size", 20),
        ),
        crs=CrsConfig(
            fallback_epsg=crs.get("fallback_epsg")
        )
    )
