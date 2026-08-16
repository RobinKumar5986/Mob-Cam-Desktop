# PyInstaller build spec for the Mob Cam desktop app (Windows onedir build).
# Same Analysis as the Linux spec, plus an icon embedded straight into the exe -
# there is no separate BUNDLE() step like macOS, PyInstaller's EXE() does it directly.

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
    icon='packaging/windows/mobcam.ico',
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

# NOTE: icon= on EXE() is what sets the .exe's file icon and the taskbar icon
# on Windows - equivalent role to BUNDLE(icon=...) on macOS. iconphoto() inside
# receiver_gui.py still runs (harmless) but has no visible effect here, same as
# on macOS. tk.Tk(className="mobcam") is also a no-op on Windows - WM_CLASS is
# an X11/Linux concept, nothing to match here.
