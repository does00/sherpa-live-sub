# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：在 Windows 上运行 pyinstaller sherpa-live-sub.spec"""

from PyInstaller.utils.hooks import collect_dynamic_libs

# sherpa_onnx 自带的 onnxruntime 等 DLL（sherpa_onnx.libs 目录）
sherpa_bins = collect_dynamic_libs("sherpa_onnx")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=sherpa_bins,
    datas=[],
    hiddenimports=["soundcard", "sounddevice", "sherpa_onnx"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="实时字幕",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI 程序，不弹黑窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
