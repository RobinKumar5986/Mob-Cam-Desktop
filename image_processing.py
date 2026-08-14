"""Image processing layer.

Sits between the data receiver and the virtual camera:

    device frames -> [ image_processing ] -> virtual camera / preview

Stage order mirrors the Android ImageProcessor, so a frame looks the same
whether the phone processed it or the desktop did:

    rotate -> mirror -> color_filter -> head_lock -> background -> blur
           -> crop_shape -> fit_output -> shape_mask

The AI stages (head_lock, background, blur) only do work when the phone is in
"USE PC" mode; otherwise the phone has already applied them and they pass
through. Register future stages with pipeline.insert_before(...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

import ai_processing
from remote_settings import RemoteSettings, clamp_iso_gain

SHAPE_SQUARE = "square"
SHAPE_CIRCLE = "circle"
SHAPE_SOURCE = "source"

GREEN_BGR = (0, 255, 0)
BLACK_BGR = (0, 0, 0)

_LUMA = (0.213, 0.715, 0.072)


@dataclass
class ProcessingConfig:
    """Live-mutable settings read once per frame."""

    shape: str = SHAPE_SQUARE
    output_size: Tuple[int, int] = (720, 720)
    mirror: bool = False
    rotation: int = 0
    background: Tuple[int, int, int] = GREEN_BGR

    remote: RemoteSettings = field(default_factory=RemoteSettings)
    ai: Optional[ai_processing.AIEngine] = None

    frame_state: dict = field(default_factory=dict, repr=False)
    extra: dict = field(default_factory=dict)

    def normalized_rotation(self) -> int:
        return int(self.rotation) % 360

    @property
    def pc_mode(self) -> bool:
        return bool(self.remote and self.remote.use_pc)


Stage = Callable[[np.ndarray, ProcessingConfig], np.ndarray]


# ------------------------------------------------------------ geometry stages


def stage_rotate(frame: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Rotate the frame in 90 degree steps."""
    rotation = config.normalized_rotation()
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def stage_mirror(frame: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Flip horizontally for a selfie-style image."""
    return cv2.flip(frame, 1) if config.mirror else frame


def stage_crop_shape(frame: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Centre-crop to a square unless the source aspect is being kept."""
    if config.shape == SHAPE_SOURCE:
        return frame
    height, width = frame.shape[:2]
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return frame[top:top + side, left:left + side]


def stage_fit_output(frame: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Scale to fit the fixed output canvas, letterboxing the remainder."""
    out_w, out_h = even_size(config.output_size)
    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        return np.full((out_h, out_w, 3), config.background, dtype=np.uint8)

    scale = min(out_w / width, out_h / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    if (new_w, new_h) == (out_w, out_h):
        return resized

    canvas = np.full((out_h, out_w, 3), config.background, dtype=np.uint8)
    x = (out_w - new_w) // 2
    y = (out_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


_CIRCLE_CACHE: dict = {}


def circle_alpha(width: int, height: int) -> np.ndarray:
    """Cached anti-aliased circular alpha for the current output size."""
    key = (width, height)
    alpha = _CIRCLE_CACHE.get(key)
    if alpha is None:
        mask = np.zeros((height, width), dtype=np.uint8)
        radius = min(width, height) // 2
        cv2.circle(mask, (width // 2, height // 2), radius, 255, -1,
                   lineType=cv2.LINE_AA)
        alpha = mask.astype(np.float32) / 255.0
        _CIRCLE_CACHE.clear()
        _CIRCLE_CACHE[key] = alpha
    return alpha


def stage_shape_mask(frame: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Knock the corners off for the circle shape."""
    if config.shape != SHAPE_CIRCLE:
        return frame

    height, width = frame.shape[:2]
    plate = ai_processing.solid_plate(width, height, config.background)
    return ai_processing.composite(frame, plate, circle_alpha(width, height))


# --------------------------------------------------------------- colour stage


def stage_color_filter(frame: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Apply contrast, brightness, saturation, ISO gain and white balance."""
    settings = config.remote
    if settings is None or not settings.use_pc:
        return frame

    matrix = color_matrix_bgr(
        settings.contrast, settings.brightness, settings.saturation,
        settings.iso, settings.wb_position,
    )
    if matrix is None:
        return frame
    return cv2.transform(frame, matrix)


def color_matrix_bgr(contrast: int, brightness: int, saturation: int,
                     iso: int, wb_position: int) -> Optional[np.ndarray]:
    """Build the 3x4 BGR colour matrix equivalent to the Android ColorMatrix."""
    contrast_f = contrast / 100.0
    brightness_f = brightness - 100.0
    saturation_f = saturation / 100.0
    scale = contrast_f * clamp_iso_gain(iso)

    wb_shift = (wb_position - 50) / 50.0
    offsets_rgb = np.array([
        brightness_f + wb_shift * 35.0,
        brightness_f - abs(wb_shift) * 8.0,
        brightness_f - wb_shift * 35.0,
    ], dtype=np.float32)

    if (abs(scale - 1.0) < 1e-3 and abs(saturation_f - 1.0) < 1e-3
            and np.all(np.abs(offsets_rgb) < 1e-3)):
        return None

    base = np.eye(3, dtype=np.float32) * scale

    inv_sat = 1.0 - saturation_f
    lr, lg, lb = (c * inv_sat for c in _LUMA)
    sat = np.array([
        [lr + saturation_f, lg, lb],
        [lr, lg + saturation_f, lb],
        [lr, lg, lb + saturation_f],
    ], dtype=np.float32)

    linear_rgb = sat @ base
    offset_rgb = sat @ offsets_rgb

    swap = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    linear_bgr = swap @ linear_rgb @ swap
    offset_bgr = swap @ offset_rgb

    return np.hstack([linear_bgr, offset_bgr.reshape(3, 1)]).astype(np.float32)


# ------------------------------------------------------------------ AI stages


def stage_head_lock(frame: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Crop and zoom so the tracked face stays centred."""
    settings = config.remote
    if not config.pc_mode or not settings.head_lock or config.ai is None:
        return frame
    box = config.ai.face_box(frame)
    if box is None:
        return frame
    return ai_processing.zoom_to_face(frame, box)


def stage_background(frame: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Replace the background with the phone's image or its chroma colour."""
    settings = config.remote
    if not config.pc_mode or not settings.remove_background or config.ai is None:
        return frame

    mask = _person_mask(frame, config)
    if mask is None:
        return frame

    background = None
    if settings.background_option:
        background = config.ai.background_for(
            settings.background_image, frame.shape[1], frame.shape[0]
        )
    return ai_processing.replace_background(
        frame, mask, background, settings.chroma_color
    )


def stage_blur(frame: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Blur the background while keeping the person sharp."""
    settings = config.remote
    if not config.pc_mode or not settings.blur_enabled or settings.blur_intensity <= 0:
        return frame
    mask = _person_mask(frame, config) if config.ai is not None else None
    return ai_processing.blur_background(frame, mask, settings.blur_intensity)


def _person_mask(frame: np.ndarray, config: ProcessingConfig) -> Optional[np.ndarray]:
    """Segment once per frame and share the mask between stages."""
    cached = config.frame_state.get("person_mask")
    if cached is not None and cached.shape[:2] == frame.shape[:2]:
        return cached
    mask = config.ai.person_mask(frame) if config.ai is not None else None
    if mask is not None:
        config.frame_state["person_mask"] = mask
    return mask


# ------------------------------------------------------------------- pipeline


DEFAULT_STAGES: List[Tuple[str, Stage]] = [
    ("rotate", stage_rotate),
    ("mirror", stage_mirror),
    ("color_filter", stage_color_filter),
    ("head_lock", stage_head_lock),
    ("background", stage_background),
    ("blur", stage_blur),
    ("crop_shape", stage_crop_shape),
    ("fit_output", stage_fit_output),
    ("shape_mask", stage_shape_mask),
]


def even_size(size: Tuple[int, int]) -> Tuple[int, int]:
    """Round a size down to even numbers, which virtual camera drivers require."""
    width, height = int(size[0]), int(size[1])
    return max(2, width - width % 2), max(2, height - height % 2)


class ProcessingPipeline:
    """Ordered, mutable chain of stages sharing one live config object."""

    def __init__(self, config: ProcessingConfig | None = None, stages=None):
        self.config = config or ProcessingConfig()
        self._stages: List[Tuple[str, Stage]] = list(
            DEFAULT_STAGES if stages is None else stages
        )

    @property
    def stage_names(self) -> List[str]:
        return [name for name, _ in self._stages]

    def _index_of(self, name: str) -> int:
        for index, (stage_name, _) in enumerate(self._stages):
            if stage_name == name:
                return index
        raise KeyError(f"No such stage: {name!r} (have {self.stage_names})")

    def append(self, name: str, fn: Stage) -> None:
        self._stages.append((name, fn))

    def insert_before(self, existing: str, name: str, fn: Stage) -> None:
        self._stages.insert(self._index_of(existing), (name, fn))

    def insert_after(self, existing: str, name: str, fn: Stage) -> None:
        self._stages.insert(self._index_of(existing) + 1, (name, fn))

    def replace(self, name: str, fn: Stage) -> None:
        self._stages[self._index_of(name)] = (name, fn)

    def remove(self, name: str) -> None:
        del self._stages[self._index_of(name)]

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run every stage in order; a failing stage is skipped, not fatal."""
        if frame is None:
            return None
        config = self.config
        config.frame_state.clear()
        for name, fn in self._stages:
            try:
                result = fn(frame, config)
            except Exception as exc:  # noqa: BLE001
                print(f"[image_processing] stage {name!r} failed: {exc}")
                continue
            if result is not None:
                frame = result
        return np.ascontiguousarray(frame)

    __call__ = process


def build_pipeline(config: ProcessingConfig | None = None) -> ProcessingPipeline:
    return ProcessingPipeline(config)