# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['pedi_oku_landslide\\__main__.py'],
    pathex=[],
    binaries=[],
    datas=[('pedi_oku_landslide\\assets', 'assets'), ('pedi_oku_landslide\\config', 'config')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PEDI_Landslide',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
