# PyInstaller build spec for the Mob Cam desktop app (Linux onedir build).
# Entry point is receiver_gui.py. Bundles mediapipe/cv2/pyvirtualcam/sounddevice
# data files, which normal PyInstaller hidden-import detection misses.

from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas = [('logo.png', '.')]
binaries = []
hidden_imports = ['PIL._tkinter_finder']

for pkg in ('mediapipe', 'cv2', 'pyvirtualcam', 'sounddevice', 'zeroconf', 'qrcode'):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hidden_imports += pkg_hidden

a = Analysis(
    ['receiver_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='mobcam',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='mobcam',
)