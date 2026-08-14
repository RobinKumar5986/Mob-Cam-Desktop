"""Virtual camera sink.

Last stage of the pipeline: takes processed BGR frames and pushes them into a
real OS-level camera device so that Zoom / Meet / Teams / OBS / browsers see a
camera in their dropdown.

A window is only pixels on screen; applications enumerate camera *devices* from
the OS (DirectShow / Media Foundation on Windows, AVFoundation on macOS, V4L2 on
Linux). pyvirtualcam writes into a loopback driver already registered with the
OS, which is what makes us appear as a source.

Backends per OS:
    Windows  obs (OBS Virtual Camera) or unitycapture
    macOS    obs (OBS Virtual Camera, OBS >= 26.1)
    Linux    v4l2loopback (kernel module, creates /dev/videoN)

A camera device advertises exactly one resolution for as long as it is open, so
the device format is fixed at open time and every frame is conformed to it. A
resolution change therefore means closing and reopening the device, which is
done only through reopen_at() - never implicitly on a frame - and which retries,
then falls back to the previous format if the driver refuses, so the feed keeps
running instead of dying.
"""

from __future__ import annotations

import platform
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat

    PYVIRTUALCAM_AVAILABLE = True
    PYVIRTUALCAM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    pyvirtualcam = None
    PixelFormat = None
    PYVIRTUALCAM_AVAILABLE = False
    PYVIRTUALCAM_IMPORT_ERROR = str(exc)


OS_WINDOWS = "Windows"
OS_MACOS = "Darwin"
OS_LINUX = "Linux"

BACKEND_PREFERENCE = {
    OS_WINDOWS: ("obs", "unitycapture"),
    OS_MACOS: ("obs",),
    OS_LINUX: ("v4l2loopback",),
}

SETUP_INSTRUCTIONS = {
    OS_WINDOWS: (
        "No virtual camera driver found.\n\n"
        "Install OBS Studio (https://obsproject.com) and start it once so the\n"
        "OBS Virtual Camera driver gets registered. You do not need to keep OBS\n"
        "running afterwards.\n\n"
        "Alternative: install the UnityCapture driver."
    ),
    OS_MACOS: (
        "No virtual camera driver found.\n\n"
        "Install OBS Studio 26.1 or newer (https://obsproject.com), open it once\n"
        "and click 'Start Virtual Camera' so macOS registers the device.\n\n"
        "On first use macOS will ask you to allow the camera extension in\n"
        "System Settings > Privacy & Security."
    ),
    OS_LINUX: (
        "No virtual camera device found.\n\n"
        "Install and load the v4l2loopback kernel module:\n\n"
        "    sudo apt install v4l2loopback-dkms v4l-utils\n"
        "    sudo modprobe v4l2loopback devices=1 video_nr=10 \\\n"
        "         card_label='Mob Cam' exclusive_caps=1\n\n"
        "exclusive_caps=1 matters: without it Chrome and Firefox will not list\n"
        "the device. To make it permanent, add 'v4l2loopback' to\n"
        "/etc/modules-load.d/ and the options to /etc/modprobe.d/."
    ),
}


REOPEN_ATTEMPTS = 4
REOPEN_SETTLE_SECONDS = 0.2


class VirtualCameraError(RuntimeError):
    """Raised when no virtual camera device could be opened."""


def current_os() -> str:
    """Return 'Windows', 'Darwin' (macOS) or 'Linux'."""
    return platform.system()


def preferred_backends(os_name: Optional[str] = None) -> Tuple[str, ...]:
    """Backends to try, in order, for the given OS."""
    return BACKEND_PREFERENCE.get(os_name or current_os(), ())


def setup_instructions(os_name: Optional[str] = None) -> str:
    """Human-readable driver setup steps for the given OS."""
    os_name = os_name or current_os()
    return SETUP_INSTRUCTIONS.get(
        os_name, f"Virtual camera output is not supported on {os_name}."
    )


def even_size(width: int, height: int) -> Tuple[int, int]:
    """Round a size down to even numbers, which some drivers require."""
    width, height = int(width), int(height)
    return max(2, width - width % 2), max(2, height - height % 2)


def probe(width: int = 640, height: int = 480, fps: int = 30) -> Tuple[bool, str]:
    """Check whether a virtual camera can be opened, then release it."""
    if not PYVIRTUALCAM_AVAILABLE:
        return False, (
            "pyvirtualcam is not installed.\n\n    pip install pyvirtualcam\n\n"
            f"Import error: {PYVIRTUALCAM_IMPORT_ERROR}"
        )

    cam = VirtualCamera(fps=fps)
    try:
        cam.open(width, height)
        return True, f"{cam.device_name} (backend: {cam.backend})"
    except VirtualCameraError as exc:
        return False, str(exc)
    finally:
        cam.close()


def conform(frame: np.ndarray, width: int, height: int,
            pad_color=(0, 0, 0)) -> np.ndarray:
    """Fit a frame to an exact size without distorting it, padding the rest."""
    frame_h, frame_w = frame.shape[:2]
    if (frame_w, frame_h) == (width, height):
        return frame

    scale = min(width / frame_w, height / frame_h)
    new_w = max(1, min(width, int(round(frame_w * scale))))
    new_h = max(1, min(height, int(round(frame_h * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    if (new_w, new_h) == (width, height):
        return resized

    canvas = np.full((height, width, 3), pad_color, dtype=np.uint8)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


class VirtualCamera:
    """Thread-safe pyvirtualcam wrapper with a stable device format."""

    def __init__(self, fps: int = 30, device: Optional[str] = None,
                 pace: bool = False, on_status=None):
        self.fps = fps
        self.device = device
        self.pace = pace
        self.on_status = on_status
        self.pad_color: Tuple[int, int, int] = (0, 0, 0)
        self.last_error: Optional[str] = None

        self._cam = None
        self._size: Optional[Tuple[int, int]] = None
        self._pending_size: Optional[Tuple[int, int]] = None
        self._lock = threading.Lock()
        self._attempt_errors: list[str] = []
        self.frames_sent = 0

    @property
    def is_open(self) -> bool:
        return self._cam is not None

    @property
    def device_name(self) -> str:
        return getattr(self._cam, "device", "") or ""

    @property
    def backend(self) -> str:
        return getattr(self._cam, "backend", "") or ""

    @property
    def size(self) -> Optional[Tuple[int, int]]:
        return self._size

    def request_size(self, width: int, height: int) -> None:
        """Set the resolution the device will use the next time it is opened.

        Only affects an open that has not happened yet. Use reopen_at() to change
        the format of a device that is already live.
        """
        with self._lock:
            self._pending_size = even_size(width, height)

    def reopen_at(self, width: int, height: int,
                  attempts: int = REOPEN_ATTEMPTS) -> bool:
        """Close and reopen the device at a new resolution.

        Retries, because a driver that still has the old handle in flight reports
        the device busy for a moment. If every attempt fails the previous format
        is restored, so a rejected change costs the user nothing.

        Call this from the thread that sends frames, never from the UI thread.
        """
        target = even_size(width, height)
        with self._lock:
            if self._cam is not None and self._size == target:
                return True

            previous = self._size
            self._close_locked()
            time.sleep(REOPEN_SETTLE_SECONDS)

            failure = None
            for attempt in range(max(1, attempts)):
                try:
                    self._open_locked(*target)
                    self.last_error = None
                    return True
                except VirtualCameraError as exc:
                    failure = exc
                    time.sleep(REOPEN_SETTLE_SECONDS * (attempt + 1))

            self.last_error = str(failure)
            if previous is not None:
                try:
                    self._open_locked(*previous)
                    self._notify(
                        f"kept {previous[0]}x{previous[1]}, "
                        f"{target[0]}x{target[1]} was refused"
                    )
                except VirtualCameraError:
                    pass
            return False

    def open(self, width: int, height: int) -> None:
        """Open the device at the given resolution."""
        with self._lock:
            self._open_locked(*even_size(width, height))

    def _open_locked(self, width: int, height: int) -> None:
        if not PYVIRTUALCAM_AVAILABLE:
            raise VirtualCameraError(
                "pyvirtualcam is not installed (pip install pyvirtualcam)."
            )

        self._close_locked()

        os_name = current_os()
        backends = preferred_backends(os_name)
        if not backends:
            raise VirtualCameraError(
                f"Virtual camera output is not supported on {os_name}."
            )

        self._attempt_errors = []
        for backend in backends:
            kwargs = dict(
                width=width, height=height, fps=self.fps,
                fmt=PixelFormat.BGR, backend=backend, print_fps=False,
            )
            if self.device:
                kwargs["device"] = self.device
            try:
                self._cam = pyvirtualcam.Camera(**kwargs)
                self._size = (width, height)
                self._pending_size = None
                self.frames_sent = 0
                self._notify(
                    f"{self.device_name} {width}x{height}@{self.fps} ({backend})"
                )
                return
            except Exception as exc:
                self._attempt_errors.append(f"{backend}: {exc}")

        detail = "\n".join(f"  - {e}" for e in self._attempt_errors)
        raise VirtualCameraError(
            f"{setup_instructions(os_name)}\n\nBackends tried:\n{detail}"
        )

    def close(self) -> None:
        """Release the device."""
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
        self._cam = None
        self._size = None

    def send(self, frame: np.ndarray) -> None:
        """Push one BGR frame, conforming it to the device resolution."""
        height, width = frame.shape[:2]
        with self._lock:
            if self._cam is None:
                self._open_locked(*(self._pending_size or even_size(width, height)))

            target_w, target_h = self._size
            if (width, height) != (target_w, target_h):
                # Letterboxed rather than reopened: changing the device format
                # under a connected app is what breaks its stream.
                frame = conform(frame, target_w, target_h, self.pad_color)

            self._cam.send(np.ascontiguousarray(frame))
            self.frames_sent += 1
            if self.pace:
                self._cam.sleep_until_next_frame()

    def _notify(self, message: str) -> None:
        if self.on_status:
            try:
                self.on_status(message)
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()