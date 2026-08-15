"""Persisted desktop preferences.

Everything the user picks in the config window is written to a small JSON file
so the app comes back the way they left it. Values are type-checked against the
defaults on load, so a hand-edited or older file can never crash startup - a bad
entry just falls back to its default.

Location: $MOBCAM_CONFIG, else ~/.mobcam/settings.json
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict

DEFAULTS: Dict[str, Any] = {
    "shape": "square",
    "output_size": [720, 720],
    "fps": 30,
    "mirror": False,
    "rotation": 0,
    "pin_preview": False,
    "show_preview": True,
    "virtual_camera": True,
    "segmenter_model": "selfie_landscape",
    "mask_sharpness": 75,
    "device_serial": "",
    "window_geometry": "",
    "audio_enabled": True,
    "audio_device": "",
    "audio_monitor": False,
    "audio_gain": 0,
    "audio_mute": False,
}


def config_path() -> str:
    """Absolute path of the settings file."""
    override = os.environ.get("MOBCAM_CONFIG")
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), ".mobcam", "settings.json")


class DesktopSettings:
    """Dict-like store that saves atomically and never raises on bad input."""

    def __init__(self, values: Dict[str, Any] | None = None, path: str | None = None):
        self.path = path or config_path()
        self._values = dict(DEFAULTS)
        if values:
            self.update(values)

    @classmethod
    def load(cls, path: str | None = None) -> "DesktopSettings":
        """Read the settings file, falling back to defaults for anything odd."""
        settings = cls(path=path)
        try:
            with open(settings.path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return settings
        if isinstance(raw, dict):
            settings.update(raw)
        return settings

    def save(self) -> bool:
        """Write the file atomically. Returns False if the disk refused."""
        try:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=directory or ".",
                prefix=".settings-", suffix=".tmp", delete=False,
            )
            with handle:
                json.dump(self._values, handle, indent=2, sort_keys=True)
            os.replace(handle.name, self.path)
            return True
        except OSError as exc:
            print(f"[desktop_settings] could not save settings: {exc}")
            return False

    # ------------------------------------------------------------ accessors

    def get(self, key: str, fallback: Any = None) -> Any:
        return self._values.get(key, DEFAULTS.get(key, fallback))

    def set(self, key: str, value: Any) -> None:
        """Store one value, ignoring it if the type does not match the default."""
        coerced = _coerce(key, value)
        if coerced is not None or DEFAULTS.get(key) is None:
            self._values[key] = coerced

    def update(self, values: Dict[str, Any]) -> None:
        for key, value in values.items():
            if key in DEFAULTS:
                self.set(key, value)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._values)

    # ---------------------------------------------------------- convenience

    def output_size(self) -> tuple:
        """Persisted resolution as a (width, height) tuple."""
        size = self.get("output_size")
        try:
            width, height = int(size[0]), int(size[1])
        except (TypeError, ValueError, IndexError):
            width, height = DEFAULTS["output_size"]
        return max(2, width), max(2, height)

    def choose(self, key: str, allowed) -> Any:
        """Persisted value if it is still one of allowed, else the default."""
        value = self.get(key)
        if value in allowed:
            return value
        return DEFAULTS.get(key)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)


def _coerce(key: str, value: Any) -> Any:
    """Force a loaded value into the shape of its default, or None if hopeless."""
    default = DEFAULTS.get(key)
    if default is None:
        return value
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if isinstance(default, str):
        return str(value)
    if isinstance(default, list):
        if not isinstance(value, (list, tuple)) or len(value) != len(default):
            return None
        try:
            return [int(item) for item in value]
        except (TypeError, ValueError):
            return None
    return value