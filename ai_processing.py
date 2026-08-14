"""AI layer: person segmentation and face tracking, run on the desktop.

This is the half of the pipeline the phone hands over when "USE PC" is on. It
does the same jobs as the Android ImageProcessor but takes advantage of the
desktop having real CPU: full-resolution compositing, temporal mask smoothing,
guided-filter edge refinement and a proper separable background blur.
"""

from __future__ import annotations

import os
import threading
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    mp = None
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_IMPORT_ERROR = str(exc)

try:
    _GUIDED_FILTER = cv2.ximgproc.guidedFilter
except AttributeError:
    _GUIDED_FILTER = None


MODEL_DIR = os.environ.get(
    "MOBCAM_MODEL_DIR", os.path.join(os.path.expanduser("~"), ".mobcam", "models")
)

MODEL_SPECS = {
    "selfie_multiclass": (
        "selfie_multiclass_256x256.tflite",
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
    ),
    "selfie_landscape": (
        "selfie_segmenter_landscape.tflite",
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
    ),
    "face_detector": (
        "blaze_face_short_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    ),
}

SEGMENT_INPUT_WIDTH = 256
REFINE_MAX_WIDTH = 640
SHARPEN_MAX_WIDTH = 0.60
SHARPEN_MIN_WIDTH = 0.04
FACE_INPUT_WIDTH = 320
BLUR_DOWNSCALE_THRESHOLD = 15
FACE_ZOOM_MARGIN = 2.5
MIN_CROP_RATIO = 0.35
MAX_MISSED_FRAMES = 8


def model_path(key: str, search_dirs=()) -> Optional[str]:
    """Locate a model file locally without downloading it."""
    filename = MODEL_SPECS[key][0]
    candidates = list(search_dirs) + [
        os.path.join(os.getcwd(), "models"),
        os.getcwd(),
        MODEL_DIR,
    ]
    for directory in candidates:
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


def missing_models(keys, search_dirs=()) -> list:
    """Model keys that are not on disk yet, i.e. the ones a run would download."""
    return [key for key in keys if model_path(key, search_dirs) is None]


def ensure_model(key: str, search_dirs=(), on_progress=None) -> str:
    """Return a local path to a model, downloading it into MODEL_DIR if absent.

    on_progress(key, filename, downloaded_bytes, total_bytes) is called during a
    download only; a model already on disk returns without calling it at all.
    """
    found = model_path(key, search_dirs)
    if found:
        return found

    filename, url = MODEL_SPECS[key]
    os.makedirs(MODEL_DIR, exist_ok=True)
    destination = os.path.join(MODEL_DIR, filename)
    partial = destination + ".part"

    hook = None
    if on_progress is not None:
        def hook(blocks, block_size, total):
            done = blocks * block_size
            if total > 0:
                done = min(done, total)
            on_progress(key, filename, done, total)

    try:
        urllib.request.urlretrieve(url, partial, hook)
        os.replace(partial, destination)
    except BaseException:
        if os.path.exists(partial):
            try:
                os.remove(partial)
            except OSError:
                pass
        raise
    return destination


def prefetch_models(keys=("selfie_landscape", "face_detector"), on_status=None,
                    on_progress=None) -> dict:
    """Resolve every model up front so the first frame is not stalled."""
    results = {}
    for key in keys:
        try:
            results[key] = ensure_model(key, on_progress=on_progress)
        except Exception as exc:
            results[key] = None
            if on_status:
                on_status(f"{MODEL_SPECS[key][0]} unavailable: {exc}")
    if on_status:
        ready = sum(1 for value in results.values() if value)
        on_status(f"{ready}/{len(keys)} models ready")
    return results


def sharpen_mask(mask: np.ndarray, sharpness: float = 0.75, feather: int = 1,
                 cleanup: bool = True) -> np.ndarray:
    """Turn a soft confidence mask into a decisive one with a thin soft edge.

    A segmentation model outputs a probability per pixel, and using that
    directly as alpha makes every uncertain pixel semi-transparent - the whole
    subject washes out into the background. This remaps the probability so only
    a narrow band around 0.5 stays partial: the interior goes fully opaque, the
    background fully clear, and just a couple of pixels at the boundary keep the
    soft ramp that hair and shoulders need.

    sharpness  0 keeps the model's original ramp, 1 is nearly a hard cut
    feather    radius in mask pixels of the remaining anti-aliased edge
    cleanup    remove speckle and fill pinholes before feathering
    """
    sharpness = float(np.clip(sharpness, 0.0, 1.0))
    width = SHARPEN_MAX_WIDTH - (SHARPEN_MAX_WIDTH - SHARPEN_MIN_WIDTH) * sharpness
    low = 0.5 - width / 2.0
    hardened = np.clip((mask - low) / width, 0.0, 1.0)

    if cleanup:
        kernel = np.ones((3, 3), np.uint8)
        core = (hardened > 0.5).astype(np.uint8)
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN, kernel)
        core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, kernel)
        # Only keep alpha next to a surviving region, which deletes the isolated
        # low-confidence islands that read as haze in the background.
        band = cv2.dilate(core, kernel, iterations=max(1, int(feather) + 1))
        hardened = hardened * band.astype(np.float32)

    if feather > 0:
        size = int(feather) * 2 + 1
        hardened = cv2.GaussianBlur(hardened, (size, size), 0)
    return hardened


_RECRISP_LUT: dict = {}


def recrisp(mask: np.ndarray, strength: float) -> np.ndarray:
    """Restore edge contrast lost when a mask is scaled up to frame size.

    Applied through a cached 256-entry lookup table: the curve is fixed per
    strength, so quantising the mask to 8 bits and looking it up is several
    times faster than the equivalent float arithmetic at full frame size, and
    the mask is quantised to 8 bits for compositing anyway.
    """
    if strength <= 0:
        return mask

    key = round(float(strength), 3)
    lut = _RECRISP_LUT.get(key)
    if lut is None:
        levels = np.arange(256, dtype=np.float32) / 255.0
        lut = np.clip((levels - 0.5) * (1.0 + 3.0 * key) + 0.5, 0.0, 1.0)
        _RECRISP_LUT.clear()
        _RECRISP_LUT[key] = lut

    quantised = cv2.convertScaleAbs(mask, alpha=255.0)
    return cv2.LUT(quantised, lut)


# --------------------------------------------------------------- segmentation


class SelfieSegmenter:
    """Wraps the MediaPipe ImageSegmenter and returns a smoothed person mask."""

    def __init__(self, model_key: str = "selfie_landscape", smoothing: float = 0.35,
                 refine_edges: bool = True, sharpness: float = 0.75,
                 feather: int = 1):
        self.model_key = model_key
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self.refine_edges = refine_edges
        self.sharpness = float(np.clip(sharpness, 0.0, 1.0))
        self.feather = max(0, int(feather))
        self.model_file = None
        self._segmenter = None
        self._previous_mask = None
        self._lock = threading.Lock()

    def open(self) -> None:
        """Load the model. Raises RuntimeError when unavailable."""
        if self._segmenter is not None:
            return
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                f"mediapipe is not installed ({MEDIAPIPE_IMPORT_ERROR}). "
                "Run: pip install mediapipe"
            )
        self.model_file = ensure_model(self.model_key)
        options = mp_vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.model_file),
            running_mode=mp_vision.RunningMode.IMAGE,
            output_confidence_masks=True,
        )
        self._segmenter = mp_vision.ImageSegmenter.create_from_options(options)
        self._previous_mask = None

    def close(self) -> None:
        with self._lock:
            if self._segmenter is not None:
                try:
                    self._segmenter.close()
                except Exception:
                    pass
            self._segmenter = None
            self._previous_mask = None

    @property
    def is_open(self) -> bool:
        return self._segmenter is not None

    def reset(self) -> None:
        """Drop temporal state, e.g. after a camera flip."""
        self._previous_mask = None

    def person_mask(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return a float32 HxW mask in 0..1 where 1 is person, or None."""
        with self._lock:
            if self._segmenter is None:
                return None

            height, width = frame_bgr.shape[:2]
            small = _resize_to_width(frame_bgr, SEGMENT_INPUT_WIDTH)

            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            try:
                result = self._segmenter.segment(mp_image)
            except Exception:
                return None

            masks = result.confidence_masks
            if not masks:
                return None

            person = self._person_channel(masks)
            if person is None:
                return None

            person = self._smooth(person)

            if self.refine_edges:
                guide = _resize_to_width(frame_bgr, REFINE_MAX_WIDTH)
                coarse = cv2.resize(
                    person, (guide.shape[1], guide.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                person = self._refine(coarse, guide)

            # Sharpened before the upscale: cheaper, and the morphology works on
            # a scale where speckle is still speckle rather than blobs.
            person = sharpen_mask(person, self.sharpness, self.feather)

            mask = cv2.resize(person, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = recrisp(mask, self.sharpness)
            np.nan_to_num(mask, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
            return np.clip(mask, 0.0, 1.0)

    @staticmethod
    def _person_channel(masks) -> Optional[np.ndarray]:
        """Pick the person channel, whichever segmenter model is loaded.

        numpy_view() aliases memory owned by the MediaPipe result, so every
        channel is copied before the result goes out of scope.
        """
        arrays = [np.array(m.numpy_view(), dtype=np.float32, copy=True) for m in masks]
        for array in arrays:
            np.nan_to_num(array, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        if len(arrays) == 1:
            return arrays[0]
        if len(arrays) == 2:
            return arrays[1]
        return np.clip(1.0 - arrays[0], 0.0, 1.0)

    def _smooth(self, mask: np.ndarray) -> np.ndarray:
        """Blend with the previous frame's mask to stop edge flicker."""
        if self.smoothing <= 0:
            return mask
        previous = self._previous_mask
        if previous is not None and previous.shape == mask.shape:
            mask = previous * self.smoothing + mask * (1.0 - self.smoothing)
        self._previous_mask = mask
        return mask

    @staticmethod
    def _refine(mask: np.ndarray, frame_bgr: np.ndarray) -> np.ndarray:
        """Snap mask edges to image edges so hair and shoulders stay clean."""
        try:
            if _GUIDED_FILTER is not None:
                guide = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                guide = guide.astype(np.float32) / 255.0
                refined = _GUIDED_FILTER(guide, mask, radius=8, eps=1e-2)
            else:
                refined = cv2.bilateralFilter(mask, d=5, sigmaColor=0.15, sigmaSpace=5)
        except cv2.error:
            return mask
        if not np.isfinite(refined).all():
            return mask
        return refined


# ------------------------------------------------------------- face tracking


@dataclass
class FaceBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def area(self) -> int:
        return self.width * self.height


class FaceTracker:
    """Detects the most prominent face and smooths the box across frames."""

    def __init__(self, smoothing: float = 0.75):
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self.model_file = None
        self._detector = None
        self._cascade = None
        self._locked: Optional[FaceBox] = None
        self._missed = 0
        self._lock = threading.Lock()

    def open(self) -> None:
        """Load MediaPipe's face detector, falling back to a Haar cascade."""
        if self._detector is not None or self._cascade is not None:
            return
        if MEDIAPIPE_AVAILABLE:
            try:
                self.model_file = ensure_model("face_detector")
                options = mp_vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=self.model_file),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    min_detection_confidence=0.5,
                )
                self._detector = mp_vision.FaceDetector.create_from_options(options)
                return
            except Exception:
                self._detector = None
        cascade_file = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        cascade = cv2.CascadeClassifier(cascade_file)
        if cascade.empty():
            raise RuntimeError("no face detector available")
        self._cascade = cascade

    def close(self) -> None:
        with self._lock:
            if self._detector is not None:
                try:
                    self._detector.close()
                except Exception:
                    pass
            self._detector = None
            self._cascade = None
            self._locked = None
            self._missed = 0

    @property
    def is_open(self) -> bool:
        return self._detector is not None or self._cascade is not None

    @property
    def backend(self) -> str:
        if self._detector is not None:
            return "mediapipe"
        return "haar" if self._cascade is not None else "none"

    def reset(self) -> None:
        self._locked = None
        self._missed = 0

    def track(self, frame_bgr: np.ndarray) -> Optional[FaceBox]:
        """Return the smoothed locked face box, or None while nothing is found."""
        with self._lock:
            if not self.is_open:
                return None

            boxes = self._detect(frame_bgr)
            if not boxes:
                self._missed += 1
                if self._missed >= MAX_MISSED_FRAMES:
                    self._locked = None
                return self._locked

            self._missed = 0
            chosen = self._pick(boxes)
            self._locked = self._blend(self._locked, chosen)
            return self._locked

    def _detect(self, frame_bgr: np.ndarray):
        height, width = frame_bgr.shape[:2]
        scale = min(1.0, FACE_INPUT_WIDTH / max(1, width))
        small = (
            cv2.resize(frame_bgr, (int(width * scale), int(height * scale)),
                       interpolation=cv2.INTER_AREA)
            if scale < 1.0 else frame_bgr
        )
        inverse = 1.0 / scale if scale > 0 else 1.0

        if self._detector is not None:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            try:
                result = self._detector.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                )
            except Exception:
                return []
            boxes = []
            for detection in result.detections:
                box = detection.bounding_box
                boxes.append(FaceBox(
                    int(box.origin_x * inverse), int(box.origin_y * inverse),
                    int(box.width * inverse), int(box.height * inverse),
                ))
            return boxes

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        found = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        return [
            FaceBox(int(x * inverse), int(y * inverse),
                    int(w * inverse), int(h * inverse))
            for (x, y, w, h) in found
        ]

    def _pick(self, boxes) -> FaceBox:
        """Prefer the face nearest the previous lock, else the largest."""
        if self._locked is None:
            return max(boxes, key=lambda b: b.area)
        px, py = self._locked.center
        return min(boxes, key=lambda b: (b.center[0] - px) ** 2 + (b.center[1] - py) ** 2)

    def _blend(self, previous: Optional[FaceBox], fresh: FaceBox) -> FaceBox:
        if previous is None or self.smoothing <= 0:
            return fresh
        a = self.smoothing
        return FaceBox(
            int(previous.x * a + fresh.x * (1 - a)),
            int(previous.y * a + fresh.y * (1 - a)),
            int(previous.width * a + fresh.width * (1 - a)),
            int(previous.height * a + fresh.height * (1 - a)),
        )


# ---------------------------------------------------------------- compositing


def _resize_to_width(image: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale so the width is at most max_width, keeping the aspect ratio."""
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / float(width)
    return cv2.resize(
        image, (max_width, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def zoom_to_face(frame: np.ndarray, box: FaceBox) -> np.ndarray:
    """Crop around the face and rescale back to the original frame size."""
    frame_h, frame_w = frame.shape[:2]
    if box is None or box.width <= 0 or box.height <= 0:
        return frame

    crop_w = box.width * FACE_ZOOM_MARGIN
    crop_w = max(crop_w, frame_w * MIN_CROP_RATIO)
    crop_h = crop_w * (frame_h / float(frame_w))

    if crop_w > frame_w or crop_h > frame_h:
        return frame

    center_x, center_y = box.center
    left = int(round(min(max(center_x - crop_w / 2.0, 0), frame_w - crop_w)))
    top = int(round(min(max(center_y - crop_h / 2.0, 0), frame_h - crop_h)))
    width = min(int(round(crop_w)), frame_w - left)
    height = min(int(round(crop_h)), frame_h - top)
    if width < 2 or height < 2:
        return frame

    cropped = frame[top:top + height, left:left + width]
    return cv2.resize(cropped, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)


def composite(foreground: np.ndarray, background: np.ndarray,
              mask: np.ndarray) -> np.ndarray:
    """Alpha-blend foreground over background using a float mask.

    Done entirely in 8-bit through OpenCV, which is about six times faster than
    the equivalent float32 numpy expression at 720p. The alpha is quantised to
    1/255 steps, which is invisible and is the precision the output has anyway.
    """
    alpha = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]

    alpha8 = cv2.cvtColor(
        np.clip(alpha * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )
    inverse8 = cv2.bitwise_not(alpha8)
    return cv2.add(
        cv2.multiply(foreground, alpha8, scale=1 / 255.0, dtype=cv2.CV_8U),
        cv2.multiply(background, inverse8, scale=1 / 255.0, dtype=cv2.CV_8U),
    )


_SOLID_CACHE: dict = {}


def solid_plate(width: int, height: int, bgr: Tuple[int, int, int]) -> np.ndarray:
    """Cached flat-colour frame, so a full-frame fill is not allocated per frame."""
    key = (width, height, tuple(int(c) for c in bgr))
    plate = _SOLID_CACHE.get(key)
    if plate is None:
        plate = np.full((height, width, 3), key[2], dtype=np.uint8)
        _SOLID_CACHE.clear()
        _SOLID_CACHE[key] = plate
    return plate


def replace_background(frame: np.ndarray, mask: np.ndarray,
                       background: Optional[np.ndarray],
                       chroma_bgr: Tuple[int, int, int]) -> np.ndarray:
    """Swap the background for an image, or for a flat chroma colour."""
    if background is None:
        plate = solid_plate(frame.shape[1], frame.shape[0], chroma_bgr)
    else:
        plate = fit_background(background, frame.shape[1], frame.shape[0])
    return composite(frame, plate, mask)


def blur_background(frame: np.ndarray, mask: Optional[np.ndarray],
                    intensity: int) -> np.ndarray:
    """Blur everything except the person. Blurs the whole frame if mask is None."""
    strength = int(np.clip(intensity, 0, 100))
    if strength <= 0:
        return frame

    radius = max(3, int(round(strength / 100.0 * min(frame.shape[:2]) * 0.08)))
    if radius % 2 == 0:
        radius += 1

    if radius > BLUR_DOWNSCALE_THRESHOLD:
        # Blurring a shrunken copy then scaling back looks the same and is far
        # cheaper than a large kernel at full resolution.
        height, width = frame.shape[:2]
        factor = min(4, max(2, radius // BLUR_DOWNSCALE_THRESHOLD))
        small = cv2.resize(frame, (max(2, width // factor), max(2, height // factor)),
                           interpolation=cv2.INTER_AREA)
        small_radius = max(3, (radius // factor) | 1)
        small = cv2.GaussianBlur(small, (small_radius, small_radius), 0)
        blurred = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        blurred = cv2.GaussianBlur(frame, (radius, radius), 0)

    if mask is None:
        return blurred
    return composite(frame, blurred, mask)


def fit_background(background: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale-and-crop a background image to fill the target size."""
    if background.shape[1] == width and background.shape[0] == height:
        return background

    src_h, src_w = background.shape[:2]
    target_ratio = width / float(height)
    src_ratio = src_w / float(src_h)

    if src_ratio > target_ratio:
        crop_h = src_h
        crop_w = int(round(crop_h * target_ratio))
    else:
        crop_w = src_w
        crop_h = int(round(crop_w / target_ratio))

    x = (src_w - crop_w) // 2
    y = (src_h - crop_h) // 2
    cropped = background[y:y + crop_h, x:x + crop_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------- engine


class AIEngine:
    """Owns the models and turns them on and off to match the phone's settings."""

    def __init__(self, segmenter_model: str = "selfie_landscape", on_status=None,
                 mask_sharpness: float = 0.75):
        self.segmenter = SelfieSegmenter(model_key=segmenter_model,
                                         sharpness=mask_sharpness)
        self.face_tracker = FaceTracker()
        self.on_status = on_status
        self.last_error: Optional[str] = None
        self._background_cache: Optional[np.ndarray] = None
        self._background_key = None

    def prefetch(self, on_progress=None) -> dict:
        """Resolve the models this engine can use, before any frame arrives."""
        return prefetch_models(
            (self.segmenter.model_key, "face_detector"), self._status, on_progress
        )

    def configure(self, settings) -> None:
        """Load or release models so only what the settings need stays resident."""
        self._ensure(self.segmenter, settings.needs_segmentation, "segmenter")
        self._ensure(self.face_tracker, settings.needs_face_tracking, "face tracker")

    def _ensure(self, component, wanted: bool, label: str) -> None:
        if wanted and not component.is_open:
            try:
                component.open()
                self._status(f"{label} ready")
            except Exception as exc:
                self.last_error = f"{label}: {exc}"
                self._status(f"{label} unavailable - {exc}")
        elif not wanted and component.is_open:
            component.close()

    def person_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        return self.segmenter.person_mask(frame)

    def face_box(self, frame: np.ndarray) -> Optional[FaceBox]:
        return self.face_tracker.track(frame)

    def background_for(self, image: Optional[np.ndarray], width: int, height: int):
        """Cached scale-and-crop of the phone's background image."""
        if image is None:
            return None
        key = (id(image), width, height)
        if key != self._background_key:
            self._background_cache = fit_background(image, width, height)
            self._background_key = key
        return self._background_cache

    def set_mask_sharpness(self, sharpness: float) -> None:
        """Live edge-hardness control, 0 soft to 1 nearly a hard cut."""
        self.segmenter.sharpness = float(np.clip(sharpness, 0.0, 1.0))

    def reset(self) -> None:
        self.segmenter.reset()
        self.face_tracker.reset()

    def close(self) -> None:
        self.segmenter.close()
        self.face_tracker.close()
        self._background_cache = None
        self._background_key = None

    def status(self) -> str:
        """Short description of which models are live."""
        parts = []
        if self.segmenter.is_open:
            parts.append(f"segmenter({self.segmenter.model_key})")
        if self.face_tracker.is_open:
            parts.append(f"face({self.face_tracker.backend})")
        return ", ".join(parts) or "idle"

    def _status(self, message: str) -> None:
        if self.on_status:
            try:
                self.on_status(message)
            except Exception:
                pass