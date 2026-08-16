"""Converts logo.png into the .ico PyInstaller and Inno Setup need on Windows.

Unlike macOS's make_icns.sh, this needs no OS-specific tool - Pillow can write
.ico on any platform, so this can be run from Linux ahead of time.
"""
import os
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "logo.png")
DST = os.path.join(ROOT, "packaging", "windows", "mobcam.ico")

SIZES = [16, 32, 48, 64, 128, 256]


def main():
    img = Image.open(SRC).convert("RGBA")
    img.save(DST, sizes=[(s, s) for s in SIZES])
    print(f"==> icon built: {DST}")


if __name__ == "__main__":
    main()
