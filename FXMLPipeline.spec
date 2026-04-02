# FXMLPipeline.spec

from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
)

hiddenimports = []
hiddenimports += collect_submodules("sklearn")
hiddenimports += collect_submodules("lightgbm")

datas = []
datas += collect_data_files("sklearn")
datas += collect_data_files("lightgbm")
datas += [("data", "data")]

binaries = []
binaries += collect_dynamic_libs("sklearn")
binaries += collect_dynamic_libs("lightgbm")

a = Analysis(
    ["app/gui_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    [],
    exclude_binaries=True,
    name="FXMLPipeline",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FXMLPipeline",
)
