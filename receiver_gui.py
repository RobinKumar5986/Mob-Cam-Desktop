"""Mob Cam receiver GUI.

Thin UI layer over the pipeline:

    data_receiver  ->  image_processing (+ ai_processing)  ->  virtual_camera
                                                          \\->  preview window
    audio_receiver ->  audio_output (loopback device + optional monitor)

The window is only a local monitor. What other applications see is the virtual
camera device and, when a loopback audio device is installed, the virtual
microphone, so Zoom / Meet / Teams / OBS / browsers list both sources.
"""

import os
import platform
import re
import shutil
import subprocess
import threading
import queue

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

from adb_wifi_dialog import WifiPairDialog
import ai_processing
import audio_output
from data_receiver import DEFAULT_PORT
from desktop_settings import DesktopSettings
from image_processing import (
    GREEN_BGR, SHAPE_CIRCLE, SHAPE_SOURCE, SHAPE_SQUARE, ProcessingConfig,
)
from protocol import AUDIO_PORT
from stream_pipeline import StreamPipeline
import virtual_mic
from audio_processing import MAX_GAIN_DB, MIN_GAIN_DB
from virtual_camera import current_os, probe, setup_instructions

PORT = DEFAULT_PORT
AUDIO_STREAM_PORT = AUDIO_PORT
BG_KEY_COLOR = "#00FF00"

RESOLUTIONS = [
    ("720 x 720 (square)", (720, 720)),
    ("480 x 480 (square)", (480, 480)),
    ("1080 x 1080 (square)", (1080, 1080)),
    ("1280 x 720 (16:9)", (1280, 720)),
    ("1920 x 1080 (16:9)", (1920, 1080)),
    ("640 x 480 (4:3)", (640, 480)),
]
FPS_OPTIONS = [30, 15, 24, 60]
# Fast model first; it is also the default selection.
SEGMENTER_MODELS = [
    ("Landscape (fastest)", "selfie_landscape"),
    ("Multiclass 256 (best quality)", "selfie_multiclass"),
]
DEFAULT_SEGMENTER = SEGMENTER_MODELS[0][1]
SHAPES = (SHAPE_SQUARE, SHAPE_CIRCLE, SHAPE_SOURCE)

# The camera device and the microphone device are deliberately separate: two
# kernel/sound-server objects, two names, two independent toggles. Nothing in
# one path can take the other down.
VCAM_CARD_LABEL = "Mob Cam"
VCAM_VIDEO_NR = 10
V4L2_MODULE = "v4l2loopback"


def v4l2_module_loaded() -> bool:
    """True when the v4l2loopback kernel module is currently loaded."""
    try:
        with open("/proc/modules", "r", encoding="utf-8") as handle:
            return any(line.startswith(V4L2_MODULE) for line in handle)
    except OSError:
        return False


def v4l2_parameter(name: str) -> str:
    """Read one live module parameter, or '' if it cannot be read."""
    try:
        with open(f"/sys/module/{V4L2_MODULE}/parameters/{name}",
                  "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def v4l2_exclusive_caps() -> bool:
    """Whether the loaded module advertises capture-only devices.

    This is the single reason a loopback camera shows up in OBS but not in
    Chrome, Firefox or Google Meet. Without exclusive_caps the device reports
    both OUTPUT and CAPTURE, and browsers skip anything that is not purely a
    capture device. OBS is not that fussy, which is why it keeps working and
    makes the device look fine.

    The parameter is one value per device, so only the first matters here.
    Kernels render booleans as either 1/0 or Y/N depending on version.
    """
    raw = v4l2_parameter("exclusive_caps")
    if not raw:
        return False
    return raw.split(",")[0].strip() in ("1", "y", "Y")


def v4l2_devices() -> list:
    """Paths of the loopback video nodes that currently exist."""
    raw = v4l2_parameter("video_nr")
    numbers = []
    for part in raw.split(","):
        part = part.strip()
        if part.lstrip("-").isdigit() and int(part) >= 0:
            numbers.append(int(part))
    if not numbers:
        numbers = [VCAM_VIDEO_NR]
    return [f"/dev/video{n}" for n in numbers if os.path.exists(f"/dev/video{n}")]


def camera_diagnosis():
    """Classify why the virtual camera is or is not usable by browsers.

    Returns (state, message) where state is one of:
        ok            usable everywhere
        no_module     the kernel module is not loaded
        no_exclusive  loaded, but browsers will refuse to list it
        no_device     loaded correctly but no node appeared
    """
    if platform.system() != "Linux":
        return "ok", ""

    if not v4l2_module_loaded():
        return "no_module", f"{V4L2_MODULE} is not loaded - click 'Load driver'."

    if not v4l2_exclusive_caps():
        return "no_exclusive", (
            f"{V4L2_MODULE} is loaded WITHOUT exclusive_caps=1, so OBS can see "
            "the camera but Chrome, Firefox and Google Meet will not list it. "
            "Click 'Reload driver' to fix it.")

    if not v4l2_devices():
        return "no_device", (
            f"{V4L2_MODULE} is loaded but no /dev/video node appeared. "
            "Try 'Reload driver'.")

    return "ok", f"{V4L2_MODULE} ready on {', '.join(v4l2_devices())}"


PERMANENT_HINT = (
    "This lasts until you reboot. To make it permanent:\n"
    f"  echo {V4L2_MODULE} | sudo tee /etc/modules-load.d/{V4L2_MODULE}.conf\n"
    f"  echo 'options {V4L2_MODULE} devices=1 video_nr={VCAM_VIDEO_NR} "
    f"card_label=\"{VCAM_CARD_LABEL}\" exclusive_caps=1' | \\\n"
    f"    sudo tee /etc/modprobe.d/{V4L2_MODULE}.conf")

MANUAL_HINT = (
    f"  sudo apt install {V4L2_MODULE}-dkms v4l-utils\n"
    f"  sudo modprobe -r {V4L2_MODULE}\n"
    f"  sudo modprobe {V4L2_MODULE} devices=1 video_nr={VCAM_VIDEO_NR} \\\n"
    f"       card_label='{VCAM_CARD_LABEL}' exclusive_caps=1")


def _run_privileged(arguments, timeout=60):
    """Run a root command, trying a graphical prompt before a terminal one."""
    attempts = []
    if shutil.which("pkexec"):
        attempts.append(["pkexec"] + arguments)
    if shutil.which("sudo"):
        attempts.append(["sudo", "-n"] + arguments)
    attempts.append(arguments)

    errors = []
    for command in attempts:
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"  {command[0]}: {exc}")
            continue
        if result.returncode == 0:
            return True, ""
        detail = (result.stderr or result.stdout or "").strip()
        errors.append(f"  {command[0]}: {detail or f'exit {result.returncode}'}")
    return False, "\n".join(errors)


def load_v4l2loopback(force_reload: bool = False):
    """Load, or reload, the loopback module with browser-friendly options.

    Reloading matters as much as loading. A plain 'modprobe v4l2loopback' with
    no options produces a device OBS accepts and browsers ignore, and modprobe
    will not change the options of a module that is already resident - it has
    to come out first.
    """
    if platform.system() != "Linux":
        return False, "The kernel module is a Linux-only step."

    modprobe = shutil.which("modprobe") or "/sbin/modprobe"
    arguments = [
        modprobe, V4L2_MODULE, "devices=1", f"video_nr={VCAM_VIDEO_NR}",
        f"card_label={VCAM_CARD_LABEL}", "exclusive_caps=1",
    ]

    if v4l2_module_loaded():
        state, _ = camera_diagnosis()
        if state == "ok" and not force_reload:
            return True, (
                f"{V4L2_MODULE} is already loaded correctly on "
                f"{', '.join(v4l2_devices())}.")

        ok, errors = _run_privileged([modprobe, "-r", V4L2_MODULE])
        if not ok:
            return False, (
                "The module is loaded with the wrong options and could not be "
                "removed, usually because something is still using it.\n\n"
                "Close OBS and any browser tab holding the camera, then try "
                "again. Or run it yourself:\n\n" + MANUAL_HINT +
                f"\n\nAttempts made:\n{errors}")

    ok, errors = _run_privileged(arguments)
    if ok and v4l2_module_loaded() and v4l2_exclusive_caps():
        devices = ", ".join(v4l2_devices()) or f"/dev/video{VCAM_VIDEO_NR}"
        return True, (
            f"{V4L2_MODULE} loaded as '{VCAM_CARD_LABEL}' on {devices}, with "
            "exclusive_caps=1 so browsers will list it.\n\n"
            "In Google Meet: start Mob Cam first, then reload the Meet tab and "
            f"pick '{VCAM_CARD_LABEL}'. Chrome only enumerates cameras when the "
            "page loads, so a tab opened earlier will not show it.\n\n"
            + PERMANENT_HINT)

    if ok and v4l2_module_loaded() and not v4l2_exclusive_caps():
        return False, (
            "The module loaded but exclusive_caps did not take effect, so "
            "browsers still will not list the camera.\n\n"
            "Your v4l2loopback build may be older than the option. Check the "
            "version with 'modinfo v4l2loopback' and update the dkms package:\n"
            f"  sudo apt install --reinstall {V4L2_MODULE}-dkms")

    return False, (
        "Could not load the module automatically. Run this in a terminal:\n\n"
        + MANUAL_HINT + f"\n\nAttempts made:\n{errors}")


def resolution_label(size):
    """Combobox label for a (width, height), falling back to the first entry."""
    for label, value in RESOLUTIONS:
        if tuple(value) == tuple(size):
            return label
    return RESOLUTIONS[0][0]


def model_label(key):
    """Combobox label for a segmenter model key."""
    for label, value in SEGMENTER_MODELS:
        if value == key:
            return label
    return SEGMENTER_MODELS[0][0]


def run_adb(*args):
    """Run an adb command, tolerating adb not being installed."""
    try:
        return subprocess.run(["adb", *args], capture_output=True)
    except (FileNotFoundError, OSError) as exc:
        print(f"[receiver_gui] adb unavailable: {exc}")
        return None


def list_adb_devices():
    """Return (serial, model) tuples for connected, authorized devices."""
    try:
        output = subprocess.check_output(
            ["adb", "devices", "-l"], stderr=subprocess.STDOUT, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    devices = []
    for line in output.splitlines()[1:]:
        line = line.strip()
        if not line or "device" not in line.split():
            continue
        parts = line.split()
        serial = parts[0]
        model = next((p.split(":")[1] for p in parts if p.startswith("model:")), "")
        devices.append((serial, model or serial))
    return devices


class ConfigWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Mob Cam - Config")
        # Small minimum on purpose: the panels scroll, so the window only needs
        # to be wide enough for the widest row, not tall enough for everything.
        self.root.minsize(640, 320)
        self.root.protocol("WM_DELETE_WINDOW", self.on_app_close)

        self.settings = DesktopSettings.load()
        self.root.geometry(self._start_geometry())

        self.stream = None
        self.viewer = None
        self._devices = []
        self._closing_stream = False
        self._vcam_available = None
        self._vcam_message = ""
        # The user's intent, kept apart from whether the driver happens to be
        # loaded. A probe failure must never be written back as a preference,
        # or the app permanently disables its own camera and cannot recover
        # once the module is loaded again.
        self._vcam_preference = bool(self.settings.get("virtual_camera"))
        self._suppress_vcam_trace = 0
        self._vcam_state = ""
        self._audio_available = None
        self._audio_message = ""
        self._download_window = None
        self._save_job = None
        self._stats_job = None
        self._resize_job = None
        self._level_job = None

        self.config = ProcessingConfig(
            shape=self.settings.choose("shape", SHAPES),
            output_size=self.settings.output_size(),
            mirror=bool(self.settings.get("mirror")),
            rotation=int(self.settings.get("rotation")),
            background=GREEN_BGR,
        )

        self._build_ui()
        self.refresh_devices()
        self.check_virtual_camera()
        self.check_audio()
        self.prefetch_models()

    # ------------------------------------------------------------------ UI

    def _start_geometry(self) -> str:
        """Saved window geometry, clamped to the screen.

        A size saved on a larger display used to open taller than the screen,
        putting the Connect button under the taskbar with no way to reach it.
        Now anything too tall is trimmed and the panels scroll instead.
        """
        max_width = self.root.winfo_screenwidth()
        # Leave room for a panel or taskbar rather than filling edge to edge.
        max_height = max(320, self.root.winfo_screenheight() - 120)
        default = f"{min(700, max_width)}x{min(780, max_height)}"

        saved = self.settings.get("window_geometry") or ""
        match = re.match(r"^(\d+)x(\d+)(.*)$", saved)
        if not match:
            return default

        width = max(640, min(int(match.group(1)), max_width))
        height = max(320, min(int(match.group(2)), max_height))
        return f"{width}x{height}{match.group(3)}"

    def _build_ui(self):
        # The panels outgrew a fixed window once audio arrived, so everything
        # lives in a scroller. The window can now be resized down to anything
        # and the controls stay reachable.
        self.scroller = ScrollableFrame(self.root)
        self.scroller.pack(fill="both", expand=True)

        frm = ttk.Frame(self.scroller.interior, padding=15)
        frm.pack(fill="both", expand=True)
        for col in range(4):
            frm.columnconfigure(col, weight=1)

        row = 0

        ttk.Label(frm, text="Phone:").grid(row=row, column=0, sticky="w", padx=5, pady=6)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(
            frm, textvariable=self.device_var, state="readonly", width=30
        )
        self.device_combo.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=6)
        ttk.Button(frm, text="Rescan", command=self.refresh_devices).grid(
            row=row, column=3, sticky="e", padx=5, pady=6
        )
        row += 1

        ttk.Label(frm, text="Crop shape:").grid(row=row, column=0, sticky="w", padx=5, pady=6)
        self.shape_var = tk.StringVar(value=self.config.shape)
        self.shape_var.trace_add("write", self._on_shape_changed)
        ttk.Radiobutton(frm, text="Square", variable=self.shape_var, value=SHAPE_SQUARE).grid(
            row=row, column=1, sticky="w", padx=5, pady=6)
        ttk.Radiobutton(frm, text="Circle", variable=self.shape_var, value=SHAPE_CIRCLE).grid(
            row=row, column=2, sticky="w", padx=5, pady=6)
        ttk.Radiobutton(frm, text="Full frame", variable=self.shape_var, value=SHAPE_SOURCE).grid(
            row=row, column=3, sticky="w", padx=5, pady=6)
        row += 1

        ttk.Label(frm, text="Video size / fps:").grid(
            row=row, column=0, sticky="w", padx=5, pady=6)
        self.res_var = tk.StringVar(value=resolution_label(self.config.output_size))
        ttk.Combobox(
            frm, textvariable=self.res_var, state="readonly",
            values=[label for label, _ in RESOLUTIONS], width=22,
        ).grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=6)
        self.res_var.trace_add("write", self._on_resolution_changed)

        self.fps_var = tk.StringVar(value=str(self.settings.choose("fps", FPS_OPTIONS)))
        self.fps_var.trace_add("write", lambda *_: self._schedule_save())
        ttk.Combobox(
            frm, textvariable=self.fps_var, state="readonly",
            values=[str(f) for f in FPS_OPTIONS], width=6,
        ).grid(row=row, column=3, sticky="e", padx=5, pady=6)
        row += 1

        self.mirror_var = tk.BooleanVar(value=self.config.mirror)
        self.mirror_var.trace_add("write", self._on_mirror_changed)
        ttk.Checkbutton(frm, text="Mirror video", variable=self.mirror_var).grid(
            row=row, column=0, sticky="w", padx=5, pady=6)

        self.preview_var = tk.BooleanVar(value=bool(self.settings.get("show_preview")))
        self.preview_var.trace_add("write", lambda *_: self._schedule_save())
        ttk.Checkbutton(
            frm, text="Show preview window", variable=self.preview_var
        ).grid(row=row, column=1, columnspan=2, sticky="w", padx=5, pady=6)

        self.pin_var = tk.BooleanVar(value=bool(self.settings.get("pin_preview")))
        self.pin_var.trace_add("write", self._on_pin_changed)
        ttk.Checkbutton(frm, text="Keep preview on top", variable=self.pin_var).grid(
            row=row, column=3, sticky="w", padx=5, pady=6)
        row += 1

        vcam_frame = ttk.LabelFrame(
            frm, text="Video out - what apps see as a camera", padding=10)
        vcam_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5, pady=(10, 6))
        vcam_frame.columnconfigure(1, weight=1)

        self.vcam_var = tk.BooleanVar(value=self._vcam_preference)
        self.vcam_var.trace_add("write", self._on_vcam_changed)
        ttk.Checkbutton(
            vcam_frame, text="Publish as a camera", variable=self.vcam_var
        ).grid(row=0, column=0, sticky="w")

        # Mirrors the microphone panel: a button that fixes the device rather
        # than instructions telling you to go and fix it yourself.
        self.vcam_load_btn = ttk.Button(
            vcam_frame, text="Load driver", command=self.load_camera_driver)
        self.vcam_load_btn.grid(row=0, column=2, sticky="e", padx=(0, 6))

        self.vcam_help_btn = ttk.Button(
            vcam_frame, text="Fix / help", command=self.show_vcam_help)
        self.vcam_help_btn.grid(row=0, column=3, sticky="e")

        self.vcam_recheck_btn = ttk.Button(
            vcam_frame, text="Re-check", command=self.check_virtual_camera)
        self.vcam_recheck_btn.grid(row=0, column=1, sticky="e", padx=(0, 6))

        self.vcam_status_var = tk.StringVar(value="Checking for a driver...")
        ttk.Label(
            vcam_frame, textvariable=self.vcam_status_var, foreground="gray",
            wraplength=560, justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
        row += 1

        mic_frame = ttk.LabelFrame(
            frm, text="Audio out - what apps see as a microphone", padding=10)
        mic_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5, pady=6)
        mic_frame.columnconfigure(1, weight=1)

        self.audio_var = tk.BooleanVar(value=bool(self.settings.get("audio_enabled")))
        self.audio_var.trace_add("write", self._on_audio_changed)
        ttk.Checkbutton(
            mic_frame, text="Publish as a microphone", variable=self.audio_var
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        self.mic_create_btn = ttk.Button(
            mic_frame, text="Create virtual mic", command=self.create_virtual_mic)
        self.mic_create_btn.grid(row=0, column=2, sticky="e", padx=(0, 6))
        ttk.Button(mic_frame, text="Fix / help", command=self.show_audio_help).grid(
            row=0, column=3, sticky="e")

        ttk.Label(mic_frame, text="Send audio to:").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        self.audio_device_var = tk.StringVar(value=self.settings.get("audio_device") or "")
        self.audio_device_combo = ttk.Combobox(
            mic_frame, textvariable=self.audio_device_var, state="readonly", width=30)
        self.audio_device_combo.grid(
            row=1, column=1, columnspan=2, sticky="ew", padx=8, pady=(8, 0))
        self.audio_device_var.trace_add("write", self._on_audio_device_changed)
        ttk.Button(mic_frame, text="Rescan", command=self.refresh_audio_devices).grid(
            row=1, column=3, sticky="e", pady=(8, 0))

        ttk.Label(mic_frame, text="Input level:").grid(
            row=2, column=0, sticky="w", pady=(8, 0))
        self.level_bar = ttk.Progressbar(
            mic_frame, mode="determinate", maximum=100, length=200)
        self.level_bar.grid(row=2, column=1, sticky="ew", padx=8, pady=(8, 0))

        self.audio_mute_var = tk.BooleanVar(value=bool(self.settings.get("audio_mute")))
        self.audio_mute_var.trace_add("write", self._on_audio_mute_changed)
        ttk.Checkbutton(mic_frame, text="Mute", variable=self.audio_mute_var).grid(
            row=2, column=2, sticky="w", pady=(8, 0))

        ttk.Label(mic_frame, text="Mic gain:").grid(
            row=3, column=0, sticky="w", pady=(8, 0))
        self.audio_gain_var = tk.IntVar(value=int(self.settings.get("audio_gain")))
        ttk.Scale(
            mic_frame, from_=MIN_GAIN_DB, to=MAX_GAIN_DB, orient="horizontal",
            variable=self.audio_gain_var, command=self._on_audio_gain_changed,
        ).grid(row=3, column=1, sticky="ew", padx=8, pady=(8, 0))
        self.audio_gain_label_var = tk.StringVar(
            value=f"{self.audio_gain_var.get():+d} dB")
        ttk.Label(
            mic_frame, textvariable=self.audio_gain_label_var, foreground="gray",
        ).grid(row=3, column=2, sticky="w", pady=(8, 0))

        self.audio_monitor_var = tk.BooleanVar(
            value=bool(self.settings.get("audio_monitor")))
        self.audio_monitor_var.trace_add("write", self._on_audio_monitor_changed)
        ttk.Checkbutton(
            mic_frame, text="Also play on my speakers (for testing)",
            variable=self.audio_monitor_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.audio_status_var = tk.StringVar(value="Checking for a device...")
        ttk.Label(
            mic_frame, textvariable=self.audio_status_var, foreground="gray",
            wraplength=560, justify="left",
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(8, 0))
        row += 1

        phone_frame = ttk.LabelFrame(frm, text="Phone settings and status", padding=10)
        phone_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5, pady=6)
        phone_frame.columnconfigure(1, weight=1)

        ttk.Label(phone_frame, text="Phone reports:").grid(row=0, column=0, sticky="w")
        self.phone_settings_var = tk.StringVar(value="waiting for handshake")
        ttk.Label(
            phone_frame, textvariable=self.phone_settings_var, foreground="gray",
            wraplength=430, justify="left",
        ).grid(row=0, column=1, columnspan=2, sticky="w", padx=8)

        ttk.Label(phone_frame, text="Cut-out edge:").grid(
            row=3, column=0, sticky="w", pady=(6, 0))
        self.sharpness_var = tk.IntVar(
            value=int(self.settings.get("mask_sharpness")))
        ttk.Scale(
            phone_frame, from_=0, to=100, orient="horizontal",
            variable=self.sharpness_var, command=self._on_sharpness_changed,
        ).grid(row=3, column=1, sticky="ew", padx=8, pady=(6, 0))
        self.sharpness_label_var = tk.StringVar(
            value=f"{self.sharpness_var.get()}%")
        ttk.Label(
            phone_frame, textvariable=self.sharpness_label_var, foreground="gray",
        ).grid(row=3, column=2, sticky="w", pady=(6, 0))

        ttk.Label(phone_frame, text="Video rate:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.stats_var = tk.StringVar(value="idle")
        ttk.Label(
            phone_frame, textvariable=self.stats_var, foreground="gray",
        ).grid(row=2, column=1, columnspan=2, sticky="w", padx=8, pady=(6, 0))

        ttk.Label(phone_frame, text="AI models:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.ai_status_var = tk.StringVar(value="idle")
        ttk.Label(
            phone_frame, textvariable=self.ai_status_var, foreground="gray",
            wraplength=430, justify="left",
        ).grid(row=1, column=1, sticky="w", padx=8, pady=(6, 0))

        self.model_var = tk.StringVar(
            value=model_label(self.settings.choose("segmenter_model",
                                                  [k for _, k in SEGMENTER_MODELS])))
        ttk.Combobox(
            phone_frame, textvariable=self.model_var, state="readonly",
            values=[label for label, _ in SEGMENTER_MODELS], width=24,
        ).grid(row=1, column=2, sticky="e", pady=(6, 0))
        self.model_var.trace_add("write", lambda *_: self._on_model_changed())
        row += 1

        self.connect_btn = ttk.Button(frm, text="Connect", command=self.on_connect_clicked)
        self.connect_btn.grid(row=row, column=0, sticky="w", padx=5, pady=(10, 6))

        self.status_dot_var = tk.StringVar(value="")
        ttk.Label(
            frm, textvariable=self.status_dot_var, foreground="#2ecc71",
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=row, column=1, sticky="w", padx=(5, 0), pady=(10, 6))

        self.status_var = tk.StringVar(value="Not connected")
        ttk.Label(frm, textvariable=self.status_var, foreground="gray").grid(
            row=row, column=2, columnspan=2, sticky="w", padx=5, pady=(10, 6))
        row += 1

        ttk.Button(frm, text="Start ADB / find phone", command=self.connect_adb).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5, pady=6)
        ttk.Button(frm, text="Pair phone over WiFi", command=self.pair_wifi).grid(
            row=row, column=2, columnspan=2, sticky="w", padx=5, pady=6)

    # -------------------------------------------------------- config changes

    def _on_shape_changed(self, *_):
        self.config.shape = self.shape_var.get()
        self._schedule_save()

    def _on_mirror_changed(self, *_):
        self.config.mirror = self.mirror_var.get()
        self._schedule_save()

    def _on_resolution_changed(self, *_):
        """Debounced so scrolling through the list does not thrash the driver."""
        size = dict(RESOLUTIONS).get(self.res_var.get())
        if not size:
            return
        self.config.output_size = size
        self._schedule_save()

        if self.stream is None:
            return
        if self._resize_job is not None:
            try:
                self.root.after_cancel(self._resize_job)
            except tk.TclError:
                pass
        self.status_var.set(f"Switching output to {size[0]}x{size[1]}...")
        self._resize_job = self.root.after(600, lambda: self._apply_resolution(size))

    def _apply_resolution(self, size):
        self._resize_job = None
        if self.stream is not None:
            self.stream.set_output_size(*size)

    def _on_pin_changed(self, *_):
        if self.viewer is not None:
            self.viewer.set_pinned(self.pin_var.get())
        self._schedule_save()

    def _on_vcam_changed(self, *_):
        """Only a real click counts as a preference worth remembering."""
        if self._suppress_vcam_trace:
            return
        self._vcam_preference = self.vcam_var.get()
        if self.stream is not None:
            self.stream.set_virtual_camera_enabled(self.vcam_var.get())
        self._schedule_save()

    def _set_vcam_checkbox(self, value: bool) -> None:
        """Move the checkbox from code without recording it as a preference."""
        self._suppress_vcam_trace += 1
        try:
            self.vcam_var.set(bool(value))
        finally:
            self._suppress_vcam_trace -= 1
        if self.stream is not None:
            self.stream.set_virtual_camera_enabled(self.vcam_var.get())

    def _on_model_changed(self):
        self._schedule_save()
        self.prefetch_models()

    def _schedule_save(self, delay_ms: int = 400):
        """Coalesce rapid changes into a single write."""
        if self._save_job is not None:
            try:
                self.root.after_cancel(self._save_job)
            except tk.TclError:
                pass
        self._save_job = self.root.after(delay_ms, self.save_settings)

    def save_settings(self):
        """Persist every user-visible choice."""
        self._save_job = None
        try:
            fps = int(self.fps_var.get())
        except ValueError:
            fps = FPS_OPTIONS[0]
        self.settings.update({
            "shape": self.shape_var.get(),
            "output_size": list(self.config.output_size),
            "fps": fps,
            "mirror": self.mirror_var.get(),
            "rotation": self.config.rotation,
            "pin_preview": self.pin_var.get(),
            "show_preview": self.preview_var.get(),
            # The remembered choice, not the current forced-off state.
            "virtual_camera": self._vcam_preference,
            "segmenter_model": self.selected_segmenter(),
            "mask_sharpness": int(float(self.sharpness_var.get())),
            "audio_enabled": self.audio_var.get(),
            "audio_device": self.audio_device_var.get(),
            "audio_monitor": self.audio_monitor_var.get(),
            "audio_gain": int(float(self.audio_gain_var.get())),
            "audio_mute": self.audio_mute_var.get(),
            "device_serial": self.get_selected_serial() or self.settings.get("device_serial"),
        })
        self.settings.save()

    def _on_sharpness_changed(self, _value=None):
        """Hardens or softens the background-removal edge while streaming."""
        value = int(float(self.sharpness_var.get()))
        self.sharpness_label_var.set(f"{value}%")
        if self.stream is not None:
            self.stream.set_mask_sharpness(value / 100.0)
        self._schedule_save()

    def selected_segmenter(self):
        """Model key for the chosen segmentation model."""
        return dict(SEGMENTER_MODELS).get(self.model_var.get(), DEFAULT_SEGMENTER)

    # ------------------------------------------------------ virtual camera

    def check_virtual_camera(self):
        """Probe for a driver in the background so the UI stays responsive."""
        self.vcam_status_var.set("Checking for a driver...")
        threading.Thread(target=self._probe_worker, daemon=True).start()

    def _probe_worker(self):
        ok, message = probe()
        self.root.after(0, lambda: self._apply_probe_result(ok, message))

    def _apply_probe_result(self, ok, message):
        """Reflect the driver state without ever overwriting the preference.

        pyvirtualcam's probe only asks 'can I open a device'. That is true even
        when the module was loaded without exclusive_caps, which is exactly the
        state where OBS works and browsers do not - so the diagnosis is checked
        as well, and a working-for-OBS-only device is reported as a problem
        rather than a tick.
        """
        state, detail = camera_diagnosis()
        self._vcam_message = message
        linux = platform.system() == "Linux"

        self._vcam_state = state
        self.vcam_load_btn.state(["!disabled"] if linux else ["disabled"])
        self.vcam_load_btn.config(
            text="Reload driver" if state == "no_exclusive" else "Load driver")

        if state == "no_exclusive":
            # The device works, so keep publishing for OBS, but say plainly why
            # Meet cannot see it.
            self._vcam_available = ok
            self.vcam_status_var.set(f"⚠ {detail}")
            self._set_vcam_checkbox(self._vcam_preference if ok else False)
            return

        self._vcam_available = ok
        if ok:
            self.vcam_status_var.set(
                f"✔ {detail or message}"
                + (f" - pick '{VCAM_CARD_LABEL}' in your app" if linux else ""))
            # The driver is back, so honour what the user actually asked for.
            self._set_vcam_checkbox(self._vcam_preference)
            return

        if linux and state in ("no_module", "no_device"):
            self.vcam_status_var.set(f"{detail} It does not survive a reboot.")
        else:
            self.vcam_status_var.set("No camera device found - click 'Fix / help'")

        # Off because it cannot work right now, not because they chose it.
        self._set_vcam_checkbox(False)

    def load_camera_driver(self):
        """Load or reload v4l2loopback, re-probe, and restore the camera."""
        reload_needed = getattr(self, "_vcam_state", "") == "no_exclusive"
        if reload_needed and not messagebox.askokcancel(
                "Reload the camera driver",
                "The driver has to be unloaded and reloaded to add "
                "exclusive_caps=1, which is what makes browsers list it.\n\n"
                "Any app currently using the Mob Cam camera will lose it and "
                "must reselect it. Continue?"):
            return

        self.vcam_status_var.set(
            "Reloading the kernel module..." if reload_needed
            else "Loading the kernel module...")
        self.vcam_load_btn.state(["disabled"])

        def worker():
            ok, message = load_v4l2loopback(force_reload=reload_needed)
            self.root.after(0, lambda: finish(ok, message))

        def finish(ok, message):
            self.vcam_load_btn.state(["!disabled"])
            HelpWindow(
                self.root,
                "Virtual camera" if ok else "Virtual camera - manual step needed",
                message)
            self.check_virtual_camera()

        threading.Thread(target=worker, daemon=True).start()

    def show_vcam_help(self):
        state, detail = camera_diagnosis()
        text = setup_instructions()

        if state == "no_exclusive":
            text = (
                "Works in OBS but not in Google Meet / Chrome / Firefox\n"
                "-----------------------------------------------------\n"
                "The driver is loaded without exclusive_caps=1. Without it the\n"
                "device reports itself as both an output and a capture device,\n"
                "and browsers only list pure capture devices. OBS does not care,\n"
                "which is why it keeps working and hides the problem.\n\n"
                "Click 'Reload driver', or run:\n\n" + MANUAL_HINT + "\n\n"
                "Then start Mob Cam BEFORE reloading the Meet tab: Chrome only\n"
                "enumerates cameras when a page loads.\n\n"
                "----------------------------------------\n\n" + text)
        elif detail:
            text = f"{text}\n\n---\nCurrent state:\n{detail}"

        if self._vcam_message and not self._vcam_available:
            text = f"{text}\n\n---\nDetail:\n{self._vcam_message}"
        HelpWindow(self.root, "Virtual camera setup", text)

    def _on_vcam_status(self, message):
        self.root.after(0, lambda: self.vcam_status_var.set(f"✔ {message}"))

    def _on_vcam_error(self, message):
        recoverable = "Could not switch" in message

        def show():
            if recoverable:
                # The device is still live at its previous size, so the toggle
                # stays on and only the message changes.
                self.vcam_status_var.set("Resolution change refused - see details")
                HelpWindow(self.root, "Virtual camera", message)
                return
            self.vcam_status_var.set("Virtual camera stopped - click 'Fix / help'")
            self._vcam_available = False
            self._vcam_message = message
            # Same rule as the probe: a runtime failure is not a preference.
            self._set_vcam_checkbox(False)
            HelpWindow(self.root, "Virtual camera error", message)

        self.root.after(0, show)

    # ------------------------------------------------------------ microphone

    def check_audio(self):
        """Find a microphone device and select it, so nothing starts unset."""
        self.refresh_audio_devices()
        available, message = audio_output.is_available()
        self._audio_available = available
        self._audio_message = message

        # Always land on a valid device. A saved choice wins, but only while it
        # still exists; otherwise fall back to whatever is actually usable.
        current = self.audio_device_var.get()
        if not current or current not in self.audio_device_combo["values"]:
            self.audio_device_var.set(audio_output.guess_loopback_device())

        self.mic_create_btn.state(
            ["!disabled"] if virtual_mic.can_create() else ["disabled"])

        if available:
            target = self.audio_device_var.get()
            self.audio_status_var.set(
                f"✔ {message}. Apps should pick "
                f"'{virtual_mic.FRIENDLY_NAME}' as their microphone."
                if virtual_mic.is_installed()
                else f"✔ {message} - sending to '{target}'")
        elif virtual_mic.can_create():
            self.audio_status_var.set(
                "No microphone device yet - click 'Create virtual mic'.")
        else:
            self.audio_status_var.set(f"{message}")

    def create_virtual_mic(self):
        """Build the loopback microphone, then select and use it immediately."""
        ok, message = virtual_mic.create()
        if not ok:
            self.audio_status_var.set("Could not create the virtual mic")
            HelpWindow(self.root, "Virtual microphone", message)
            return

        self.refresh_audio_devices()
        self.audio_device_var.set(audio_output.guess_loopback_device())
        self.check_audio()
        # Repoint a live stream at the new device rather than making them reconnect.
        if self.stream is not None:
            self.stream.set_audio_device(self.audio_device_var.get())
        HelpWindow(self.root, "Virtual microphone ready", message)

    def refresh_audio_devices(self):
        """Re-list playback devices, keeping the current choice if it survived."""
        names = audio_output.output_devices()
        self.audio_device_combo["values"] = names
        current = self.audio_device_var.get()
        if current and current not in names:
            self.audio_device_var.set(audio_output.guess_loopback_device())

    def show_audio_help(self):
        text = audio_output.setup_instructions()
        if self._audio_message and not self._audio_available:
            text = f"{text}\n\n---\nDetail:\n{self._audio_message}"
        HelpWindow(self.root, "Virtual microphone setup", text)

    def _on_audio_changed(self, *_):
        if self.stream is not None:
            self.stream.set_audio_enabled(self.audio_var.get())
        self._schedule_save()

    def _on_audio_device_changed(self, *_):
        if self.stream is not None:
            self.stream.set_audio_device(self.audio_device_var.get())
        self._schedule_save()

    def _on_audio_monitor_changed(self, *_):
        if self.stream is not None:
            self.stream.set_audio_monitor(self.audio_monitor_var.get())
        self._schedule_save()

    def _on_audio_gain_changed(self, _value=None):
        value = int(float(self.audio_gain_var.get()))
        self.audio_gain_label_var.set(f"{value:+d} dB")
        if self.stream is not None:
            self.stream.set_audio_gain_db(value)
        self._schedule_save()

    def _on_audio_mute_changed(self, *_):
        if self.stream is not None:
            self.stream.set_audio_mute(self.audio_mute_var.get())
        self._schedule_save()

    def _on_audio_status(self, message):
        self.root.after(0, lambda: self.audio_status_var.set(message))

    def _on_audio_error(self, message):
        def show():
            self.audio_status_var.set("Microphone output failed - see details")
            HelpWindow(self.root, "Virtual microphone", message)

        self.root.after(0, show)

    def _poll_level(self):
        """Drive the level meter. Faster than the stats tick so it looks live."""
        self._level_job = None
        stream = self.stream
        if stream is None:
            self.level_bar["value"] = 0
            return
        try:
            peak = stream.audio_stats()["peak"]
        except Exception:  # noqa: BLE001
            peak = 0.0
        self.level_bar["value"] = max(0, min(100, int(peak * 100)))
        self._level_job = self.root.after(100, self._poll_level)

    def _on_settings_received(self, settings):
        self.root.after(0, lambda: self.phone_settings_var.set(settings.summary()))

    def _on_ai_status(self, message):
        self.root.after(0, lambda: self.ai_status_var.set(message))

    def prefetch_models(self):
        """Resolve the AI models so the first frame is not stalled."""
        self.ai_status_var.set("checking AI models...")
        threading.Thread(target=self._prefetch_worker, daemon=True).start()

    def _prefetch_worker(self):
        keys = (self.selected_segmenter(), "face_detector")
        missing = ai_processing.missing_models(keys)
        if missing:
            # Only ever shown when something actually has to be fetched.
            self.root.after(0, lambda: self._open_download_window(missing))
        ai_processing.prefetch_models(
            keys, on_status=self._on_ai_status, on_progress=self._on_model_progress)
        self.root.after(0, self._close_download_window)

    def _open_download_window(self, missing):
        if self._download_window is not None:
            return
        self._download_window = ModelDownloadWindow(self.root, len(missing))

    def _close_download_window(self):
        window, self._download_window = self._download_window, None
        if window is not None:
            window.close()

    def _on_model_progress(self, key, filename, done, total):
        """Called from the download thread for each block received."""
        self.root.after(0, lambda: self._apply_model_progress(filename, done, total))

    def _apply_model_progress(self, filename, done, total):
        window = self._download_window
        if window is not None:
            window.update_progress(filename, done, total)
        megabytes = done / (1024 * 1024)
        if total > 0:
            self.ai_status_var.set(
                f"downloading {filename} - {done * 100 // total}%")
        else:
            self.ai_status_var.set(f"downloading {filename} - {megabytes:.1f} MB")

    # --------------------------------------------------------------- adb

    def refresh_devices(self):
        """Re-list adb devices and select the first one."""
        devices = list_adb_devices()
        if not devices:
            self.device_combo["values"] = []
            self.device_var.set("")
            self.status_var.set("No devices found")
            self._devices = []
            return

        self._devices = devices
        self.device_combo["values"] = [f"{s}  ({m})" for s, m in devices]
        remembered = self.settings.get("device_serial")
        index = next(
            (i for i, (serial, _) in enumerate(devices) if serial == remembered), 0)
        self.device_combo.current(index)
        self.status_var.set(f"{len(devices)} device(s) found")

    def get_selected_serial(self):
        idx = self.device_combo.current()
        if idx < 0 or not self._devices:
            return None
        return self._devices[idx][0]

    def connect_adb(self):
        self.status_var.set("Starting ADB and scanning for devices...")
        threading.Thread(target=self._connect_adb_worker, daemon=True).start()

    def _connect_adb_worker(self):
        run_adb("start-server")
        self.root.after(0, self.refresh_devices)

    def pair_wifi(self):
        WifiPairDialog(self.root, on_paired=self.refresh_devices)

    # ------------------------------------------------------------- stream

    def on_connect_clicked(self):
        """Toggle the stream on or off."""
        if self.stream is not None:
            self.stop_stream()
            return

        serial = self.get_selected_serial()
        if not serial:
            messagebox.showwarning("No device", "Select a device first.")
            return

        try:
            subprocess.check_call(
                ["adb", "-s", serial, "forward", f"tcp:{PORT}", f"tcp:{PORT}"])
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            messagebox.showerror("adb forward failed", str(e))
            return

        # Audio has its own socket. A failure here is not fatal: video still
        # works, the mic simply never connects.
        if self.audio_var.get():
            try:
                subprocess.check_call([
                    "adb", "-s", serial, "forward",
                    f"tcp:{AUDIO_STREAM_PORT}", f"tcp:{AUDIO_STREAM_PORT}",
                ])
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                self.audio_status_var.set(f"Audio port forward failed: {e}")

        try:
            fps = int(self.fps_var.get())
        except ValueError:
            fps = 30

        self.stream = StreamPipeline(
            port=PORT,
            config=self.config,
            fps=fps,
            virtual_camera_enabled=self.vcam_var.get(),
            segmenter_model=self.selected_segmenter(),
            mask_sharpness=int(float(self.sharpness_var.get())) / 100.0,
            audio_enabled=self.audio_var.get(),
            audio_port=AUDIO_STREAM_PORT,
            audio_device=self.audio_device_var.get(),
            audio_monitor=self.audio_monitor_var.get(),
            audio_gain_db=int(float(self.audio_gain_var.get())),
            audio_mute=self.audio_mute_var.get(),
            on_preview_frame=self._on_preview_frame,
            on_connected=lambda: self.root.after(0, self.on_stream_connected),
            on_disconnected=lambda: self.root.after(0, self.on_stream_disconnected),
            on_vcam_status=self._on_vcam_status,
            on_vcam_error=self._on_vcam_error,
            on_settings_received=self._on_settings_received,
            on_ai_status=self._on_ai_status,
            on_audio_status=self._on_audio_status,
            on_audio_error=self._on_audio_error,
        )

        if self.preview_var.get():
            self.viewer = ViewerWindow(
                self.root, pinned=self.pin_var.get(), on_closed=self.on_viewer_closed)

        self.save_settings()
        self.stream.start()
        self._poll_stats()
        self._poll_level()
        self.status_dot_var.set("")
        self.status_var.set("Waiting for frames from device...")
        self.connect_btn.config(text="Disconnect")

    def stop_stream(self):
        """Stop the stream and release the camera device."""
        if self._closing_stream:
            return
        self._closing_stream = True
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream = None
            if self.viewer is not None:
                viewer, self.viewer = self.viewer, None
                viewer.close()
            self.connect_btn.config(text="Connect")
            self.status_dot_var.set("")
            self.status_var.set("Disconnected")
            self.phone_settings_var.set("waiting for handshake")
            self.ai_status_var.set("idle")
            self.stats_var.set("idle")
            self.level_bar["value"] = 0
            for attribute in ("_stats_job", "_level_job"):
                job = getattr(self, attribute)
                if job is not None:
                    try:
                        self.root.after_cancel(job)
                    except tk.TclError:
                        pass
                    setattr(self, attribute, None)
            self.check_audio()
        finally:
            self._closing_stream = False

    def _poll_stats(self):
        """Show frames in, frames processed and frames dropped once a second."""
        self._stats_job = None
        stream = self.stream
        if stream is None:
            return
        data = stream.stats()
        if data["in_fps"] < 0.05:
            self.stats_var.set("waiting for frames")
        else:
            self.stats_var.set(
                f"in {data['in_fps']:.0f} / out {data['out_fps']:.0f} fps, "
                f"dropped {data['dropped_fps']:.0f}/s, {data['avg_ms']:.0f} ms/frame"
            )
        self._poll_audio_stats(stream)
        self._stats_job = self.root.after(1000, self._poll_stats)

    def _poll_audio_stats(self, stream):
        """Keep the mic line current while streaming, without hiding errors."""
        if not self.audio_var.get():
            return
        # An error message is more useful than a chunk count, so leave it alone.
        if "failed" in self.audio_status_var.get().lower():
            return
        audio = stream.audio_stats()
        if not audio["connected"]:
            return
        target = (virtual_mic.FRIENDLY_NAME if virtual_mic.is_installed()
                  else self.audio_device_var.get() or "output")
        detail = audio["format"] or "connecting"
        if audio["processing"] not in ("flat", ""):
            detail += f", {audio['processing']}"
        if audio["underruns"]:
            detail += f", {audio['underruns']} underruns"
        self.audio_status_var.set(f"✔ live on '{target}' - {detail}")

    def _on_preview_frame(self, frame):
        viewer = self.viewer
        if viewer is not None:
            viewer.submit(frame)

    def on_stream_connected(self):
        self.status_dot_var.set("✔")
        self.status_var.set("Connected - receiving frames")

    def on_stream_disconnected(self):
        self.status_dot_var.set("")
        self.status_var.set("Waiting for frames from device...")

    def on_viewer_closed(self):
        self.viewer = None
        self.preview_var.set(False)

    def on_app_close(self):
        self.settings.set("window_geometry", self.root.winfo_geometry())
        self.save_settings()
        self.stop_stream()
        run_adb("disconnect")
        self.root.destroy()


class ScrollableFrame(ttk.Frame):
    """A vertically scrolling container to put the config panels in.

    Tk has no scrollable frame, so the standard construction applies: a Canvas
    scrolls, and the real frame is a window item inside it. Two bindings keep
    them in step - the interior's size drives the scroll region, and the
    canvas's width is pushed back onto the interior so the content stretches
    instead of leaving a gap at the right.
    """

    def __init__(self, parent):
        super().__init__(parent)

        self._canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self.interior = ttk.Frame(self._canvas)
        self._window = self._canvas.create_window(
            (0, 0), window=self.interior, anchor="nw")

        self.interior.bind("<Configure>", self._on_interior_resize)
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        # The wheel is bound globally only while the pointer is over the canvas,
        # so it does not hijack scrolling in the preview or help windows.
        self._canvas.bind("<Enter>", self._bind_wheel)
        self._canvas.bind("<Leave>", self._unbind_wheel)

    def _on_interior_resize(self, _event=None):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_resize(self, event):
        self._canvas.itemconfigure(self._window, width=event.width)

    def _bind_wheel(self, _event=None):
        self._canvas.bind_all("<MouseWheel>", self._on_wheel)
        self._canvas.bind_all("<Button-4>", self._on_wheel)
        self._canvas.bind_all("<Button-5>", self._on_wheel)

    def _unbind_wheel(self, _event=None):
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._canvas.unbind_all(sequence)

    def _on_wheel(self, event):
        """Normalise the three wheel conventions X11, Windows and macOS use."""
        if event.num == 4:
            step = -1
        elif event.num == 5:
            step = 1
        elif event.delta:
            step = -1 if event.delta > 0 else 1
        else:
            return
        self._canvas.yview_scroll(step, "units")

    def scroll_to_top(self):
        self._canvas.yview_moveto(0.0)


class ViewerWindow:
    """Local monitor for processed frames; the camera runs without it."""

    def __init__(self, parent, pinned=False, on_closed=None):
        self.on_closed = on_closed
        self._closed = False

        self.win = tk.Toplevel(parent)
        self.win.title("Mob Cam Preview")
        self.win.geometry("400x400")
        self.win.minsize(150, 150)
        self.win.configure(bg=BG_KEY_COLOR)
        self.win.attributes("-topmost", bool(pinned))

        self.label = tk.Label(self.win, bg=BG_KEY_COLOR, bd=0, highlightthickness=0)
        self.label.pack(fill="both", expand=True)

        self.frame_queue = queue.Queue(maxsize=1)
        self.last_frame = None

        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.win.after(30, self._tick)

    def set_pinned(self, pinned: bool):
        if not self._closed:
            self.win.attributes("-topmost", bool(pinned))

    def submit(self, frame):
        """Hand a frame over from a worker thread; newest frame wins."""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except (queue.Empty, queue.Full):
                pass

    def _tick(self):
        if self._closed:
            return
        try:
            self.last_frame = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        if self.last_frame is not None:
            self._render(self.last_frame)
        self.win.after(30, self._tick)

    def _render(self, frame):
        label_w = self.label.winfo_width()
        label_h = self.label.winfo_height()
        if label_w < 2 or label_h < 2:
            return

        height, width = frame.shape[:2]
        scale = min(label_w / width, label_h / height)
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

        canvas = np.full((label_h, label_w, 3), GREEN_BGR, dtype=np.uint8)
        x = (label_w - new_w) // 2
        y = (label_h - new_h) // 2
        canvas[y:y + new_h, x:x + new_w] = resized

        photo = ImageTk.PhotoImage(
            image=Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
        self.label.configure(image=photo)
        self.label.image = photo

    def close(self):
        """Destroy the window and notify the owner once."""
        if self._closed:
            return
        self._closed = True
        try:
            self.win.destroy()
        except tk.TclError:
            pass
        if self.on_closed:
            self.on_closed()


class ModelDownloadWindow(tk.Toplevel):
    """Progress dialog shown only while AI models are actually downloading."""

    def __init__(self, parent, file_count):
        super().__init__(parent)
        self.title("Downloading AI models")
        self.geometry("420x150")
        self.resizable(False, False)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self.close)

        self._closed = False
        self._seen = []
        self._file_count = max(1, file_count)

        frame = ttk.Frame(self, padding=18)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame, text="One-time download. They are cached on disk afterwards.",
            wraplength=380, foreground="gray",
        ).pack(anchor="w")

        self.file_var = tk.StringVar(value="Starting...")
        ttk.Label(frame, textvariable=self.file_var).pack(anchor="w", pady=(10, 4))

        self.bar = ttk.Progressbar(frame, mode="determinate", maximum=100, length=380)
        self.bar.pack(fill="x")

        self.detail_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.detail_var, foreground="gray").pack(
            anchor="w", pady=(4, 0))

    def update_progress(self, filename, done, total):
        """Advance the bar; falls back to indeterminate if the size is unknown."""
        if self._closed:
            return
        if filename not in self._seen:
            self._seen.append(filename)
        index = self._seen.index(filename) + 1

        self.file_var.set(f"{filename}  ({index} of {self._file_count})")
        if total > 0:
            if str(self.bar.cget("mode")) != "determinate":
                self.bar.stop()
                self.bar.configure(mode="determinate", maximum=100)
            self.bar["value"] = done * 100 / total
            self.detail_var.set(
                f"{done / 1024:.0f} KB of {total / 1024:.0f} KB")
        else:
            if str(self.bar.cget("mode")) != "indeterminate":
                self.bar.configure(mode="indeterminate")
                self.bar.start(12)
            self.detail_var.set(f"{done / 1024:.0f} KB")

    def close(self):
        """Destroy the dialog once, whether finished or dismissed."""
        if self._closed:
            return
        self._closed = True
        try:
            self.bar.stop()
            self.destroy()
        except tk.TclError:
            pass


class HelpWindow(tk.Toplevel):
    """Scrollable text window so setup commands can be copied."""

    def __init__(self, parent, title, text):
        super().__init__(parent)
        self.title(title)
        self.geometry("560x360")

        box = scrolledtext.ScrolledText(self, wrap="word", padx=10, pady=10)
        box.pack(fill="both", expand=True)
        box.insert("1.0", text)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=8)


def main():
    root = tk.Tk()
    ConfigWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()