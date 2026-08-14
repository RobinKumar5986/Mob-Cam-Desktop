"""Image processing layer.

Sits between the frame receiver and the virtual camera:

    device frames -> [ image_processing ] -> virtual camera / preview

Everything here operates on OpenCV BGR numpy arrays and returns numpy arrays,
so no PIL round-trips happen in the hot path.

The pipeline is an ordered list of named stages. Adding future processing
(background removal, face tracking, beautify, LUTs, overlays...) means writing
a callable ``fn(frame, config) -> frame`` and registering it:

    pipeline.insert_before("fit_output", "segment", my_background_removal)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import cv2
import numpy as np

# Shapes
SHAPE_SQUARE = "square"
SHAPE_CIRCLE = "circle"
SHAPE_SOURCE = "source"  # keep the phone's native aspect ratio

# Chroma key green, in BGR. Matches BG_KEY_COLOR in the GUI so the preview and
# the virtual camera letterbox to the same colour.
GREEN_BGR = (0, 255, 0)
BLACK_BGR = (0, 0, 0)


@dataclass
class ProcessingConfig:
    """Live-mutable settings. The pipeline reads these per frame, so the UI can
    change them at any time without rebuilding anything."""

    shape: str = SHAPE_SQUARE
    output_size: Tuple[int, int] = (720, 720)  # (width, height) sent to the vcam
    mirror: bool = False       # horizontal flip, selfie-style
    rotation: int = 0          # 0 / 90 / 180 / 270, clockwise
    background: Tuple[int, int, int] = GREEN_BGR
    # Reserved for future stages so they have somewhere to keep their options.
    extra: dict = field(default_factory=dict)

    def normalized_rotation(self) -> int:
        return int(self.rotation) % 360


Stage = Callable[[np.ndarray, ProcessingConfig], np.ndarray]


# --------------------------------------------------------------------- stages


def stage_rotate(frame: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    rotation = config.normalized_rotation()
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def stage_mirror(frame: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    return cv2.flip(frame, 1) if config.mirror else frame


def stage_crop_shape(frame: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Centre-crop to a square for the square/circle shapes."""
    if config.shape == SHAPE_SOURCE:
        return frame

    height, width = frame.shape[:2]
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return frame[top:top + side, left:left + side]


def stage_fit_output(frame: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Scale to fit the fixed output canvas without distorting, letterboxing the
    remainder with the background colour.

    The output size must stay constant frame to frame, because a virtual camera
    device advertises one fixed resolution.
    """
    out_w, out_h = config.output_size
    out_w = max(2, out_w - out_w % 2)
    out_h = max(2, out_h - out_h % 2)

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


def stage_shape_mask(frame: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Knock the corners off for the circle shape.

    Note: a virtual camera carries no alpha channel, so the circle is sent as a
    circle on a solid background. The receiving app has to chroma-key the green
    itself if it wants a real cut-out.
    """
    if config.shape != SHAPE_CIRCLE:
        return frame

    height, width = frame.shape[:2]
    radius = min(width, height) // 2
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (width // 2, height // 2), radius, 255, -1, lineType=cv2.LINE_AA)

    background = np.full_like(frame, config.background)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    return (frame * alpha + background * (1.0 - alpha)).astype(np.uint8)


DEFAULT_STAGES: List[Tuple[str, Stage]] = [
    ("rotate", stage_rotate),
    ("mirror", stage_mirror),
    ("crop_shape", stage_crop_shape),
    ("fit_output", stage_fit_output),
    ("shape_mask", stage_shape_mask),
]


# ------------------------------------------------------------------- pipeline


class ProcessingPipeline:
    """Ordered, mutable chain of stages sharing one live config object."""

    def __init__(self, config: ProcessingConfig | None = None, stages=None):
        self.config = config or ProcessingConfig()
        self._stages: List[Tuple[str, Stage]] = list(
            DEFAULT_STAGES if stages is None else stages
        )

    # -- introspection

    @property
    def stage_names(self) -> List[str]:
        return [name for name, _ in self._stages]

    def _index_of(self, name: str) -> int:
        for i, (stage_name, _) in enumerate(self._stages):
            if stage_name == name:
                return i
        raise KeyError(f"No such stage: {name!r} (have {self.stage_names})")

    # -- mutation, for future processing modules

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

    # -- hot path

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Run every stage in order. A stage that raises is skipped rather than
        killing the stream, so one bad experimental filter cannot take the
        camera down mid-call."""
        if frame is None:
            return None
        config = self.config
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