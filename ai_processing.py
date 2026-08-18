"""AI layer: person segmentation and face tracking, run on the desktop.

This is the half of the pipeline the phone hands over when "USE PC" is on. It
does the same jobs as the Android ImageProcessor but takes advantage of the
desktop having real CPU: full-resolution compositing, temporal mask smoothing,
guided-filter edge refinement and a proper separable background blur.

Face tracking has three interchangeable backends because MediaPipe's face
detector cannot be used everywhere: on macOS its graph asks for a Metal service
that the wheel often does not register, and the failure is a C++ CHECK that
calls abort(), taking the whole process down before any Python handler runs. It
is therefore probed out of process once, and OpenCV's YuNet is used instead
where it is unsafe, with a Haar cascade as the last resort.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import urllib.parse
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

# The primary URL stays a plain string so existing callers and tests that do
# `filename, url = MODEL_SPECS[key]` keep working.
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
    # opencv_zoo stores this in Git LFS. raw.githubusercontent.com serves the
    # 131-byte pointer file instead of the model, so the /raw/ redirect (which
    # lands on media.githubusercontent.com) is the only usable path.
    "face_yunet": (
        "face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ),
    # OpenCV 5 wheels ship an empty cv2/data package, so the cascade the Haar
    # fallback needs has to be fetched like any other model.
    "haar_frontalface": (
        "haarcascade_frontalface_default.xml",
        "https://raw.githubusercontent.com/opencv/opencv/master/data/"
        "haarcascades/haarcascade_frontalface_default.xml",
    ),
}

MODEL_MIRRORS = {
    "face_yunet": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ),
    "haar_frontalface": (
        "https://raw.githubusercontent.com/opencv/opencv/4.x/data/"
        "haarcascades/haarcascade_frontalface_default.xml",
    ),
}

# A model shorter than this is a pointer file, an error page or a truncated
# transfer, never a usable model.
DEFAULT_MIN_MODEL_BYTES = 50_000
MODEL_MIN_BYTES = {
    "selfie_multiclass": 1_000_000,
    "selfie_landscape": 100_000,
    "face_detector": 100_000,
    "face_yunet": 200_000,
    "haar_frontalface": 300_000,
}

DOWNLOAD_TIMEOUT = float(os.environ.get("MOBCAM_DOWNLOAD_TIMEOUT", "20"))
DOWNLOAD_BLOCK = 64 * 1024
DOWNLOAD_USER_AGENT = "MobCam/1.0 (+urllib)"
LFS_POINTER_PREFIX = b"version https://git-lfs"

SEGMENT_INPUT_WIDTH = 256
REFINE_MAX_WIDTH = 640
SHARPEN_MAX_WIDTH = 0.60
SHARPEN_MIN_WIDTH = 0.04
FACE_INPUT_WIDTH = 320
BLUR_DOWNSCALE_THRESHOLD = 15
FACE_ZOOM_MARGIN = 2.5
MIN_CROP_RATIO = 0.35
MAX_MISSED_FRAMES = 8

FACE_BACKEND_OVERRIDE = os.environ.get("MOBCAM_FACE_BACKEND", "auto").strip().lower()
FACE_PROBE_FILE = os.path.join(MODEL_DIR, "face_backend.json")
FACE_PROBE_TIMEOUT = 120
YUNET_SCORE_THRESHOLD = 0.6
YUNET_NMS_THRESHOLD = 0.3
HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml"


# ------------------------------------------------------------------ models


def model_urls(key: str) -> tuple:
    """Every source to try for a model, primary first."""
    return (MODEL_SPECS[key][1],) + tuple(MODEL_MIRRORS.get(key, ()))


def model_min_bytes(key: str) -> int:
    """Smallest size a genuine copy of this model can have."""
    return MODEL_MIN_BYTES.get(key, DEFAULT_MIN_MODEL_BYTES)


def _is_usable_model(path: str, min_bytes: int) -> bool:
    """Whether a file on disk is a real model rather than a failed download.

    Git LFS repositories answer a plain raw request with a short text pointer,
    and a proxy or captive portal answers with an HTML page. Both arrive as
    HTTP 200 and both get cached, so size and magic bytes are checked instead
    of trusting that the file exists.
    """
    try:
        if os.path.getsize(path) < min_bytes:
            return False
        with open(path, "rb") as handle:
            head = handle.read(64)
    except OSError:
        return False
    return not head.startswith(LFS_POINTER_PREFIX)


def _discard(path: str) -> None:
    """Delete a file if it is there, ignoring failure."""
    try:
        os.remove(path)
    except OSError:
        pass


def _download_to(url: str, partial: str, key: str, filename: str,
                 min_bytes: int, on_progress) -> None:
    """Fetch one URL to a temporary path, reporting progress as blocks arrive.

    Read in a loop rather than through urlretrieve so the socket timeout applies
    to the transfer as well as the connect. Without it a stalled connection
    hangs the download thread forever with the progress dialog frozen at zero.
    """
    request = urllib.request.Request(url, headers={"User-Agent": DOWNLOAD_USER_AGENT})
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "html" in content_type:
            raise ValueError(f"got an HTML page, not a model ({content_type})")
        try:
            total = int(response.headers.get("Content-Length") or 0)
        except ValueError:
            total = 0
        if 0 < total < min_bytes:
            raise ValueError(f"server offered only {total} bytes")

        done = 0
        with open(partial, "wb") as handle:
            while True:
                block = response.read(DOWNLOAD_BLOCK)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                if on_progress is not None:
                    on_progress(key, filename, done, total)

    if not _is_usable_model(partial, min_bytes):
        size = os.path.getsize(partial) if os.path.exists(partial) else 0
        raise ValueError(f"downloaded file is not a model ({size} bytes)")


def _fetch_with_curl(url: str, partial: str, key: str, filename: str,
                     min_bytes: int, on_progress) -> None:
    """Second attempt through curl, which trusts the OS certificate store.

    A python.org macOS build whose Install Certificates step was never run fails
    every urllib request with CERTIFICATE_VERIFY_FAILED while curl succeeds, so
    it is worth one more try before declaring a model unavailable.
    """
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is not installed")

    result = subprocess.run(
        [curl, "-fsSL", "--connect-timeout", str(int(DOWNLOAD_TIMEOUT)),
         "--max-time", str(int(DOWNLOAD_TIMEOUT * 6)),
         "-A", DOWNLOAD_USER_AGENT, "-o", partial, url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or f"curl exited {result.returncode}")

    if not _is_usable_model(partial, min_bytes):
        size = os.path.getsize(partial) if os.path.exists(partial) else 0
        raise ValueError(f"curl fetched something that is not a model ({size} bytes)")

    if on_progress is not None:
        size = os.path.getsize(partial)
        on_progress(key, filename, size, size)


def model_path(key: str, search_dirs=()) -> Optional[str]:
    """Locate a usable model file locally without downloading it.

    A previously cached failed download is treated as absent, so a bad file
    from an earlier run is replaced instead of being trusted forever.
    """
    filename = MODEL_SPECS[key][0]
    min_bytes = model_min_bytes(key)
    candidates = list(search_dirs) + [
        os.path.join(os.getcwd(), "models"),
        os.getcwd(),
        MODEL_DIR,
    ]
    for directory in candidates:
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate) and _is_usable_model(candidate, min_bytes):
            return candidate
    return None


def missing_models(keys, search_dirs=()) -> list:
    """Model keys that are not on disk yet, i.e. the ones a run would download."""
    return [key for key in keys if model_path(key, search_dirs) is None]


def ensure_model(key: str, search_dirs=(), on_progress=None) -> str:
    """Return a local path to a model, downloading it into MODEL_DIR if absent.

    Every source is tried with urllib and then with curl, and the result is
    validated before it is committed to the cache. on_progress(key, filename,
    done, total) is called during a download only; total is 0 when the server
    sends no length.
    """
    found = model_path(key, search_dirs)
    if found:
        return found

    filename = MODEL_SPECS[key][0]
    min_bytes = model_min_bytes(key)
    os.makedirs(MODEL_DIR, exist_ok=True)
    destination = os.path.join(MODEL_DIR, filename)
    partial = destination + ".part"

    errors = []
    for url in model_urls(key):
        host = urllib.parse.urlsplit(url).netloc or url
        for label, fetch in (("urllib", _download_to), ("curl", _fetch_with_curl)):
            try:
                fetch(url, partial, key, filename, min_bytes, on_progress)
                os.replace(partial, destination)
                return destination
            except Exception as exc:
                errors.append(f"{host} via {label}: {exc}")
            finally:
                if os.path.exists(partial):
                    _discard(partial)

    raise RuntimeError(f"could not download {filename} - " + "; ".join(errors))


def prefetch_models(keys=("selfie_landscape", "face_detector"), on_status=None,
                    on_progress=None) -> dict:
    """Resolve every model up front so the first frame is not stalled."""
    keys = [key for key in keys if key]
    results = {}
    failures = []
    for key in keys:
        try:
            results[key] = ensure_model(key, on_progress=on_progress)
        except Exception as exc:
            results[key] = None
            failures.append(f"{MODEL_SPECS[key][0]}: {exc}")
            print(f"[ai_processing] {MODEL_SPECS[key][0]} unavailable: {exc}")
            if on_status:
                on_status(f"{MODEL_SPECS[key][0]} unavailable: {exc}")
    if on_status and not failures:
        on_status(f"{len(results)}/{len(keys)} models ready")
    elif on_status:
        ready = sum(1 for value in results.values() if value)
        on_status(f"{ready}/{len(keys)} models ready - " + "; ".join(failures))
    return results


# ------------------------------------------------------------------- masking


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


# ------------------------------------------------------- face backend choice


_FACE_PROBE_CODE = (
    "import sys, numpy as np, mediapipe as mp\n"
    "from mediapipe.tasks import python as mp_python\n"
    "from mediapipe.tasks.python import vision as mp_vision\n"
    "opts = mp_vision.FaceDetectorOptions(\n"
    "    base_options=mp_python.BaseOptions(model_asset_path=sys.argv[1]),\n"
    "    running_mode=mp_vision.RunningMode.IMAGE)\n"
    "det = mp_vision.FaceDetector.create_from_options(opts)\n"
    "det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,\n"
    "                   data=np.zeros((128, 128, 3), dtype=np.uint8)))\n"
    "det.close()\n"
)


def _mediapipe_version() -> str:
    """Version string of the installed mediapipe, or 'none'."""
    return getattr(mp, "__version__", "unknown") if mp is not None else "none"


def _face_probe_key() -> str:
    """Cache key: the verdict is only valid for this mediapipe on this OS."""
    return f"{_mediapipe_version()}|{platform.system()}|{platform.mac_ver()[0]}"


def _load_face_probe(key: str) -> Optional[bool]:
    """Previously cached probe verdict for this key, or None."""
    try:
        with open(FACE_PROBE_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if data.get("key") != key or not isinstance(data.get("safe"), bool):
        return None
    return data["safe"]


def _save_face_probe(key: str, safe: bool) -> None:
    """Remember the probe verdict so it costs one subprocess ever."""
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(FACE_PROBE_FILE, "w", encoding="utf-8") as handle:
            json.dump({"key": key, "safe": bool(safe)}, handle)
    except OSError:
        pass


def _run_face_probe(model_file: str) -> bool:
    """Create the MediaPipe face detector in a child process and report survival."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _FACE_PROBE_CODE, model_file],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=FACE_PROBE_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def mediapipe_face_safe(search_dirs=()) -> bool:
    """Whether MediaPipe's face detector can open here without killing the process.

    Its graph asks for a Metal service that macOS builds often do not register,
    and the failure is a C++ CHECK that calls abort(), so it cannot be caught in
    process. The detector is therefore created once in a child process and the
    verdict cached; other platforms are trusted directly.
    """
    if not MEDIAPIPE_AVAILABLE:
        return False
    if platform.system() != "Darwin":
        return True
    # A frozen build has no interpreter to re-invoke, so assume the worst.
    if getattr(sys, "frozen", False):
        return False

    key = _face_probe_key()
    cached = _load_face_probe(key)
    if cached is not None:
        return cached
    try:
        model_file = ensure_model("face_detector", search_dirs)
    except Exception:
        return False
    safe = _run_face_probe(model_file)
    _save_face_probe(key, safe)
    return safe


def face_model_key(search_dirs=()) -> Optional[str]:
    """Model file the face tracker will actually load on this machine."""
    if FACE_BACKEND_OVERRIDE == "haar":
        return "haar_frontalface"
    if FACE_BACKEND_OVERRIDE == "mediapipe":
        return "face_detector"
    if FACE_BACKEND_OVERRIDE == "yunet":
        return "face_yunet"
    return "face_detector" if mediapipe_face_safe(search_dirs) else "face_yunet"


def _open_yunet(search_dirs=()):
    """Create OpenCV's YuNet face detector, fetching the ONNX model if absent."""
    factory = getattr(cv2, "FaceDetectorYN", None)
    if factory is None:
        raise RuntimeError("this OpenCV build has no FaceDetectorYN")
    model_file = ensure_model("face_yunet", search_dirs)
    return factory.create(
        model_file, "", (FACE_INPUT_WIDTH, FACE_INPUT_WIDTH),
        YUNET_SCORE_THRESHOLD, YUNET_NMS_THRESHOLD, 5000,
    )


def haar_cascade_path(search_dirs=()) -> str:
    """Path to the frontal-face cascade, downloading it if OpenCV ships none.

    OpenCV 5 wheels contain an empty cv2/data package - the XML cascades were
    dropped - so the last-resort detector has to fetch its own data file rather
    than trust cv2.data.haarcascades to point at anything.
    """
    bundled_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
    if bundled_dir:
        bundled = os.path.join(bundled_dir, HAAR_CASCADE_FILE)
        if os.path.isfile(bundled) and os.path.getsize(bundled) > 0:
            return bundled
    return ensure_model("haar_frontalface", search_dirs)


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
        self.last_error: Optional[str] = None
        self._detector = None
        self._yunet = None
        self._cascade = None
        self._locked: Optional[FaceBox] = None
        self._missed = 0
        self._lock = threading.Lock()

    def open(self) -> None:
        """Load the best face detector this machine can actually run.

        Every rejected backend is reported, not just the last one, because the
        interesting failure is usually the first: which of MediaPipe, YuNet or
        the cascade was unavailable and why.
        """
        if self.is_open:
            return

        errors = []
        wanted = face_model_key()

        if wanted == "face_detector":
            try:
                self.model_file = ensure_model("face_detector")
                options = mp_vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=self.model_file),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    min_detection_confidence=0.5,
                )
                self._detector = mp_vision.FaceDetector.create_from_options(options)
                self.last_error = None
                return
            except Exception as exc:
                self._detector = None
                errors.append(f"mediapipe: {exc}")

        if wanted != "haar_frontalface":
            try:
                self._yunet = _open_yunet()
                self.model_file = model_path("face_yunet")
                self.last_error = None
                return
            except Exception as exc:
                self._yunet = None
                errors.append(f"yunet: {exc}")

        try:
            # OpenCV 5 dropped Haar cascades from some builds entirely, so the
            # class itself has to be checked, not just the data file.
            factory = getattr(cv2, "CascadeClassifier", None)
            if factory is None:
                raise RuntimeError("this OpenCV build has no CascadeClassifier")
            cascade_file = haar_cascade_path()
            cascade = factory(cascade_file)
            if cascade.empty():
                raise RuntimeError(f"{cascade_file} did not load")
            self._cascade = cascade
            self.model_file = cascade_file
            self.last_error = "; ".join(errors) or None
            if errors:
                print("[ai_processing] face detector fell back to haar - "
                      + "; ".join(errors))
            return
        except Exception as exc:
            errors.append(f"haar: {exc}")

        detail = "; ".join(errors) or "no usable backend"
        self.last_error = detail
        print(f"[ai_processing] no face detector available - {detail}")
        raise RuntimeError(f"no face detector available ({detail})")

    def close(self) -> None:
        with self._lock:
            if self._detector is not None:
                try:
                    self._detector.close()
                except Exception:
                    pass
            self._detector = None
            self._yunet = None
            self._cascade = None
            self._locked = None
            self._missed = 0

    @property
    def is_open(self) -> bool:
        return (self._detector is not None or self._yunet is not None
                or self._cascade is not None)

    @property
    def backend(self) -> str:
        if self._detector is not None:
            return "mediapipe"
        if self._yunet is not None:
            return "yunet"
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

        if self._yunet is not None:
            # YuNet needs the exact input size declared before every detect.
            try:
                self._yunet.setInputSize((small.shape[1], small.shape[0]))
                _, faces = self._yunet.detect(np.ascontiguousarray(small))
            except cv2.error:
                return []
            if faces is None:
                return []
            return [
                FaceBox(int(face[0] * inverse), int(face[1] * inverse),
                        int(face[2] * inverse), int(face[3] * inverse))
                for face in faces
                if face[2] > 1 and face[3] > 1
            ]

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
        keys = tuple(k for k in (self.segmenter.model_key, face_model_key()) if k)
        return prefetch_models(keys, self._status, on_progress)

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
                print(f"[ai_processing] {label} unavailable - {exc}")
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