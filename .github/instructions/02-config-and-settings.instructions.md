## What is the Problem
Quan ly cau hinh ung dung (UI scale, thong so xu ly, CRS fallback) va luu tru settings nguoi dung.

## Information Context
File/chuc nang:
- pedi_oku_landslide/config/config.yaml: cau hinh mac dinh (ui_scale_percent, smooth_size, sad_threshold, grouping, fallback_epsg).
- pedi_oku_landslide/config/settings.py: dataclass Config + ham load_config(base_dir).
- pedi_oku_landslide/config/app_settings.json: luu settings nguoi dung (duoc ghi qua settings_store).
- pedi_oku_landslide/project/settings_store.py: load_settings/save_settings, noi luu file app_settings.json.
Framework/thu vien: PyYAML, dataclasses.
Quy uoc/luu y:
- load_config doc file theo base_dir/pedi_oku_landslide/config/config.yaml.
- fallback_epsg duoc dung khi raster thieu CRS hoac khong co .prj (xem pipeline/ingest.py).
- app_settings.json luu UI scale, do do phong chu toan app.
Vi du nho:
```
cfg = load_config(base_dir)
scale = cfg.app.ui_scale_percent
```

## Steps to Complete
1) Xac dinh loai cau hinh: config.yaml (mac dinh, project) hay app_settings.json (nguoi dung).
2) Cap nhat dataclass trong settings.py neu them key moi, sau do cap nhat config.yaml.
3) Neu thong so can duoc UI sua, them vao SettingsDialog va settings_store.
4) Kiem tra ingest/sad/detect co dung key moi hay khong.
5) Test: chay app, mo Settings, doi scale, kiem tra app_settings.json duoc cap nhat.
