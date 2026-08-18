# adb_tools.py
"""Locating the adb binary regardless of how the app was launched.

A GUI app started from Finder, a .desktop entry or a Start Menu shortcut
inherits almost no PATH, so a bare "adb" subprocess call fails with
FileNotFoundError even though the same call works from a terminal. This
resolves adb from the usual SDK and package-manager locations, caches it, and
puts its folder on PATH so nested calls agree.
"""

import os
import platform
import shutil

_cached = None


def _binary_name():
    return "adb.exe" if platform.system() == "Windows" else "adb"


def _sdk_roots():
    """Android SDK roots worth checking, most authoritative first."""
    home = os.path.expanduser("~")
    roots = [os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT")]
    system = platform.system()
    if system == "Darwin":
        roots.append(os.path.join(home, "Library", "Android", "sdk"))
    elif system == "Windows":
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            roots.append(os.path.join(local, "Android", "Sdk"))
    else:
        roots += [os.path.join(home, "Android", "Sdk"), "/usr/lib/android-sdk"]
    return [root for root in roots if root]


def _extra_dirs():
    """Package-manager install locations that are not SDK layouts."""
    system = platform.system()
    if system == "Darwin":
        return ["/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"]
    if system == "Windows":
        return []
    return ["/usr/bin", "/usr/local/bin", "/snap/bin"]


def adb_path():
    """Absolute path to a usable adb, or None if none was found."""
    global _cached
    if _cached is not None:
        return _cached or None

    explicit = os.environ.get("ADB_PATH", "")
    found = explicit if explicit and os.access(explicit, os.X_OK) else shutil.which("adb")

    if not found:
        directories = [os.path.join(root, "platform-tools") for root in _sdk_roots()]
        directories += _extra_dirs()
        for directory in directories:
            candidate = os.path.join(directory, _binary_name())
            if os.access(candidate, os.X_OK):
                found = candidate
                break

    _cached = found or ""
    if found:
        folder = os.path.dirname(found)
        if folder not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
    return found or None


def adb_command(*args):
    """Full argv for an adb call, or None if adb is not installed."""
    binary = adb_path()
    return [binary, *args] if binary else None


def adb_hint():
    """User-facing message for when adb cannot be found."""
    system = platform.system()
    if system == "Darwin":
        install = ("Install it with:  brew install --cask android-platform-tools\n"
                   "or install Android Studio, which puts it in\n"
                   "  ~/Library/Android/sdk/platform-tools")
    elif system == "Windows":
        install = ("Install Android Platform Tools and keep them in\n"
                   "  %LOCALAPPDATA%\\Android\\Sdk\\platform-tools")
    else:
        install = "Install it with:  sudo apt install android-tools-adb"
    return ("adb was not found.\n\n" + install +
            "\n\nIf adb is somewhere unusual, set ADB_PATH to its full path "
            "before launching Mob Cam.")