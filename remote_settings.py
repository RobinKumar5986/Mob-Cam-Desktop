"""Settings received from the phone in the HELLO handshake."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

DEFAULT_CHROMA_BGR = (64, 177, 0)  # #00B140 in BGR

MIN_ISO = 100
MAX_ISO = 3200


@dataclass
class RemoteSettings:
    """Mirror of the Android SettingSharePreference values."""

    use_pc: bool = False
    contrast: int = 100
    brightness: int = 100
    saturation: int = 100
    iso: int = 400
    wb_position: int = 50
    wb_mode: str = "MANUAL"
    target_fps: int = 24
    front_camera: bool = False

    head_lock: bool = False
    remove_background: bool = False
    chroma_color: Tuple[int, int, int] = DEFAULT_CHROMA_BGR
    background_option: Optional[str] = None
    blur_enabled: bool = False
    blur_intensity: int = 0

    frame_width: int = 0
    frame_height: int = 0
    protocol_version: int = 1
    device_model: str = ""

    background_image: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_hello(cls, data: dict) -> "RemoteSettings":
        """Build settings from a decoded HELLO payload."""
        get = data.get
        return cls(
            use_pc=bool(get("usePc", False)),
            contrast=_as_int(get("contrast"), 100),
            brightness=_as_int(get("brightness"), 100),
            saturation=_as_int(get("saturation"), 100),
            iso=_as_int(get("iso"), 400),
            wb_position=_as_int(get("wbPosition"), 50),
            wb_mode=str(get("wbMode", "MANUAL")),
            target_fps=_as_int(get("targetFps"), 24),
            front_camera=bool(get("frontCamera", False)),
            head_lock=bool(get("headLock", False)),
            remove_background=bool(get("removeBackground", False)),
            chroma_color=argb_to_bgr(_as_int(get("chromaColor"), 0xFF00B140)),
            background_option=_as_optional_str(get("backgroundOption")),
            blur_enabled=bool(get("blurEnabled", False)),
            blur_intensity=_as_int(get("blurIntensity"), 0),
            frame_width=_as_int(get("frameWidth"), 0),
            frame_height=_as_int(get("frameHeight"), 0),
            protocol_version=_as_int(get("protocolVersion"), 1),
            device_model=str(get("deviceModel", "")),
        )

    def update_from(self, data: dict) -> None:
        """Apply a live SETTINGS message in place, leaving absent keys alone."""
        fresh = RemoteSettings.from_hello({**self.as_hello(), **data})
        for name in self.__dataclass_fields__:
            if name == "background_image":
                continue
            setattr(self, name, getattr(fresh, name))

    def as_hello(self) -> dict:
        """Round-trip back to HELLO key names, so partial updates can merge."""
        return {
            "usePc": self.use_pc,
            "contrast": self.contrast,
            "brightness": self.brightness,
            "saturation": self.saturation,
            "iso": self.iso,
            "wbPosition": self.wb_position,
            "wbMode": self.wb_mode,
            "targetFps": self.target_fps,
            "frontCamera": self.front_camera,
            "headLock": self.head_lock,
            "removeBackground": self.remove_background,
            "chromaColor": bgr_to_argb(self.chroma_color),
            "backgroundOption": self.background_option or "",
            "blurEnabled": self.blur_enabled,
            "blurIntensity": self.blur_intensity,
            "frameWidth": self.frame_width,
            "frameHeight": self.frame_height,
            "protocolVersion": self.protocol_version,
            "deviceModel": self.device_model,
        }

    @property
    def needs_segmentation(self) -> bool:
        """True when any enabled effect depends on the person mask."""
        return self.use_pc and (self.remove_background or self.blur_enabled)

    @property
    def needs_face_tracking(self) -> bool:
        return self.use_pc and self.head_lock

    @property
    def uses_background_image(self) -> bool:
        return bool(self.background_option) and self.background_image is not None

    def summary(self) -> str:
        """One-line description for the UI."""
        effects = []
        if self.head_lock:
            effects.append("head-lock")
        if self.remove_background:
            effects.append("bg-replace" if self.uses_background_image else "chroma")
        if self.blur_enabled:
            effects.append(f"blur {self.blur_intensity}")
        mode = "PC processing" if self.use_pc else "phone processing"
        detail = ", ".join(effects) or "no effects"
        return f"{mode} - {detail} @ {self.target_fps} fps"


def _as_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_optional_str(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def argb_to_bgr(argb: int) -> Tuple[int, int, int]:
    """Convert an Android packed ARGB int to an OpenCV BGR tuple."""
    argb = int(argb) & 0xFFFFFFFF
    return (argb & 0xFF, (argb >> 8) & 0xFF, (argb >> 16) & 0xFF)


def bgr_to_argb(bgr: Tuple[int, int, int]) -> int:
    """Convert an OpenCV BGR tuple back to a packed ARGB int."""
    blue, green, red = (int(c) & 0xFF for c in bgr)
    return (0xFF << 24) | (red << 16) | (green << 8) | blue


def clamp_iso_gain(iso: int) -> float:
    """Reproduce the Android ISO-to-gain curve."""
    span = float(MAX_ISO - MIN_ISO)
    normalised = (max(MIN_ISO, min(MAX_ISO, int(iso))) - MIN_ISO) / span
    return 1.0 + normalised * 0.4