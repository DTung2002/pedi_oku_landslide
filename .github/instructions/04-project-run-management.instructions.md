## What is the Problem
Quan ly context du an/run va cau truc thu muc output/<project>/<run>, bao gom helper tao duong dan va luu metadata.

## Information Context
File chinh:
- pedi_oku_landslide/project/path_manager.py: AnalysisContext + create_context + load_context_from_run_dir + PathManager shim.
- pedi_oku_landslide/project/settings_store.py: luu tai app_settings.json.
Cau truc thu muc:
- output/<project_id>/<run_id>/
  - input/ (5 inputs)
  - ui1/, ui2/, ui3/
  - run_meta.json
- ingest_meta.json duoc tao boi pipeline/ingest.py tai run_dir.
Quy uoc/luu y:
- run_id = YYYYmmdd_HHMMSS[_label], slugify label.
- create_context tao thu muc moi (exist_ok=False) de tranh ghi de.
- PathManager la lop tuong thich cu; uu tien dung AnalysisContext cho code moi.
Vi du nho:
```
ctx = create_context(base_dir, project_id, run_label)
print(ctx.run_dir, ctx.in_dir)
```

## Steps to Complete
1) Neu them output moi, them method path_ui* trong AnalysisContext hoac noi ro vi tri file.
2) Neu thay doi quy uoc folder, cap nhat create_context/load_context_from_run_dir va UI analyze_tab.
3) Giu backward-compat: neu code cu dung PathManager, can cap nhat shim neu them duong dan moi.
4) Test: tao run moi, kiem tra run_meta.json, ingest_meta.json, va cac thu muc ui1/ui2/ui3.
