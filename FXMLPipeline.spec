# FXMLPipeline.spec

from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
    get_package_paths,
)

hiddenimports = []
hiddenimports += collect_submodules("sklearn")
hiddenimports += collect_submodules("lightgbm")
hiddenimports += collect_submodules("numpy")
hiddenimports += collect_submodules("scipy")

datas = []
datas += collect_data_files("sklearn")
datas += collect_data_files("lightgbm")
datas += collect_data_files("numpy")
datas += collect_data_files("scipy")
datas += [("data", "data")]

binaries = []
binaries += collect_dynamic_libs("sklearn")
binaries += collect_dynamic_libs("lightgbm")
binaries += collect_dynamic_libs("numpy")
binaries += collect_dynamic_libs("scipy")

# --- EXPLICITLY ADD sklearn/.libs if it exists ---
sklearn_base, sklearn_pkg = get_package_paths("sklearn")
sklearn_pkg_path = Path(sklearn_pkg)
sklearn_libs = sklearn_pkg_path / ".libs"

if sklearn_libs.exists():
    for dll in sklearn_libs.glob("*"):
        binaries.append((str(dll), "sklearn/.libs"))

# --- EXPLICITLY ADD scipy/.libs if it exists ---
scipy_base, scipy_pkg = get_package_paths("scipy")
scipy_pkg_path = Path(scipy_pkg)
scipy_libs = scipy_pkg_path / ".libs"

if scipy_libs.exists():
    for dll in scipy_libs.glob("*"):
        binaries.append((str(dll), "scipy/.libs"))

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
