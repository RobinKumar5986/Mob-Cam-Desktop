"""Virtual camera sink.

Last stage of the pipeline: takes processed BGR frames and pushes them into a
real OS-level camera device so that Zoom / Meet / Teams / OBS / browsers see
"Mob Cam" in their camera dropdown.

A Tk window is only pixels on screen; applications enumerate camera *devices*
from the OS (DirectShow / Media Foundation on Windows, AVFoundation on macOS,
V4L2 on Linux). pyvirtualcam writes into a loopback driver that is already
registered with the OS, which is what makes us appear as a source.

Backends per OS:
    Windows  obs (OBS Virtual Camera) or unitycapture
    macOS    obs (OBS Virtual Camera, OBS >= 26.1)
    Linux    v4l2loopback (kernel module, creates /dev/videoN)
"""

from __future__ import annotations

import platform
import threading

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat

    PYVIRTUALCAM_AVAILABLE = True
    PYVIRTUALCAM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on environment
    pyvirtualcam = None
    PixelFormat = None
    PYVIRTUALCAM_AVAILABLE = False
    PYVIRTUALCAM_IMPORT_ERROR = str(exc)


OS_WINDOWS = "Windows"
OS_MACOS = "Darwin"
OS_LINUX = "Linux"

# Tried in order; the first one that opens wins.
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


class VirtualCameraError(RuntimeError):
    """Raised when no virtual camera device could be opened."""


def current_os() -> str:
    """'Windows', 'Darwin' (macOS) or 'Linux'."""
    return platform.system()


def preferred_backends(os_name: str | None = None) -> tuple[str, ...]:
    return BACKEND_PREFERENCE.get(os_name or current_os(), ())


def setup_instructions(os_name: str | None = None) -> str:
    os_name = os_name or current_os()
    return SETUP_INSTRUCTIONS.get(
        os_name, f"Virtual camera output is not supported on {os_name}."
    )


def probe(width: int = 640, height: int = 480, fps: int = 30) -> tuple[bool, str]:
    """Check whether a virtual camera can be opened, without keeping it open.

    Returns (ok, message). Safe to call from a background thread at startup so
    the UI can grey out the toggle and show the setup hint instead of failing
    only once the user tries to stream.
    """
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


class VirtualCamera:
    """Thread-safe wrapper around pyvirtualcam.

    Opens lazily on the first frame, because the frame size is only known once
    the pipeline has produced something, and transparently reopens the device
    if the output resolution changes mid-session.
    """

    def __init__(
        self,
        fps: int = 30,
        device: str | None = None,
        pace: bool = False,
        on_status=None,
    ):
        """
        fps     advertised frame rate of the virtual device
        device  explicit device to use, e.g. '/dev/video10' on Linux
        pace    block each send until the next frame slot is due. Leave False
                when the phone drives the timing, else the receiver thread
                gets throttled and frames back up.
        """
        self.fps = fps
        self.device = device
        self.pace = pace
        self.on_status = on_status

        self._cam = None
        self._size: tuple[int, int] | None = None
        self._lock = threading.Lock()
        self._attempt_errors: list[str] = []
        self.frames_sent = 0

    # ---------------------------------------------------------------- state

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
    def size(self) -> tuple[int, int] | None:
        return self._size

    # ----------------------------------------------------------- lifecycle

    def open(self, width: int, height: int) -> None:
        with self._lock:
            self._open_locked(width, height)

    def _open_locked(self, width: int, height: int) -> None:
        if not PYVIRTUALCAM_AVAILABLE:
            raise VirtualCameraError(
                "pyvirtualcam is not installed (pip install pyvirtualcam)."
            )

        self._close_locked()

        # Odd dimensions break some drivers' YUV conversion.
        width -= width % 2
        height -= height % 2

        os_name = current_os()
        backends = preferred_backends(os_name)
        if not backends:
            raise VirtualCameraError(
                f"Virtual camera output is not supported on {os_name}."
            )

        self._attempt_errors = []
        for backend in backends:
            kwargs = dict(
                width=width,
                height=height,
                fps=self.fps,
                fmt=PixelFormat.BGR,  # matches OpenCV, no extra conversion
                backend=backend,
                print_fps=False,
            )
            if self.device:
                kwargs["device"] = self.device
            try:
                self._cam = pyvirtualcam.Camera(**kwargs)
                self._size = (width, height)
                self.frames_sent = 0
                self._notify(
                    f"Virtual camera live: {self.device_name} "
                    f"{width}x{height}@{self.fps} ({backend})"
                )
                return
            except Exception as exc:
                self._attempt_errors.append(f"{backend}: {exc}")

        detail = "\n".join(f"  - {e}" for e in self._attempt_errors)
        raise VirtualCameraError(
            f"{setup_instructions(os_name)}\n\nBackends tried:\n{detail}"
        )

    def close(self) -> None:
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

    # ---------------------------------------------------------------- send

    def send(self, frame) -> None:
        """Push one BGR numpy frame. Opens/reopens the device as needed."""
        height, width = frame.shape[:2]
        with self._lock:
            if self._cam is None or self._size != (width, height):
                self._open_locked(width, height)
                # Dimensions may have been rounded down to even numbers.
                if self._size != (width, height):
                    target_w, target_h = self._size
                    frame = frame[:target_h, :target_w]

            self._cam.send(frame)
            self.frames_sent += 1
            if self.pace:
                self._cam.sleep_until_next_frame()

    # --------------------------------------------------------------- misc

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