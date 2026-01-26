## What is the Problem
Giao dien PyQt5 cho luong phan tich: Input Analyze (UI1), Section Selection (UI2), Curve Analyze (UI3), va Settings.

## Information Context
File chinh:
- pedi_oku_landslide/ui/main_window.py: cua so chinh, tab flow, QSS, load fonts/branding.
- pedi_oku_landslide/ui/views/analyze_tab.py: ingest + smooth + SAD + detect + vectors, viewer UI1.
- pedi_oku_landslide/ui/views/section_tab.py: chon/tao section lines, thao tac tren hillshade.
- pedi_oku_landslide/ui/views/curve_tab.py: xu ly curvature/profile tu lines.
- pedi_oku_landslide/ui/views/settings_dialog.py: UI scale.
- pedi_oku_landslide/ui/widgets/file_picker.py: file input widget.
Design system:
- QSS trong main_window.py; mau xanh (#056832) va Inter fonts tu assets/fonts.
Quy uoc/luu y:
- Flow: Analyze -> Section -> Curve. MainWindow bat signal vectors_rendered va sections_confirmed.
- app_settings.json luu UI scale; apply_scale cap nhat QFont va QSS.
- Assets logo: assets/OKUYAMA Boring.png.
Vi du nho:
```
self.analyze_tab.vectors_rendered.connect(self._on_vectors_ready)
```

## Steps to Complete
1) Xac dinh tab anh huong: Analyze/Section/Curve/Settings.
2) Map input/output: Analyze tao run_dir va ui1 outputs; Section doc ui1 + tao sections.csv; Curve doc ui2 + tao ui3.
3) Neu thay doi giao dien, giu signal/slot va enable/disable tab logic.
4) Neu doi branding, cap nhat assets va get_app_icon/load_inter_fonts.
5) Test luong day du: Confirm Input -> SAD -> Detect -> Render Vectors -> Sections -> Curve.
