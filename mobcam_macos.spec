# PyInstaller build spec for the Mob Cam desktop app (macOS .app bundle).
# Same Analysis as the Linux spec, but adds a BUNDLE() step to produce a
# proper .app with Info.plist and a dock icon instead of a plain onedir folder.
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
app = BUNDLE(
    coll,
    name='Mob Cam.app',
    icon='packaging/macos/mobcam.icns',
    bundle_identifier='com.kgjr.mobcam',
    info_plist={
        'CFBundleName': 'Mob Cam',
        'CFBundleDisplayName': 'Mob Cam',
        'CFBundleShortVersionString': '0.1.0',
        'NSHighResolutionCapable': True,
        # PortAudio enumerates input devices even though this app only ever
        # writes to output/loopback devices; some macOS versions still gate
        # that enumeration behind the mic permission prompt.
        'NSMicrophoneUsageDescription': 'Mob Cam lists audio output devices to send your phone microphone to.',
    },
)