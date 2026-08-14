"""Mob Cam receiver GUI.

Thin UI layer over the pipeline:

    data_receiver  ->  image_processing (+ ai_processing)  ->  virtual_camera
                                                          \\->  preview window

The window is only a local monitor. What other applications see is the virtual
camera device, so Zoom / Meet / Teams / OBS / browsers list a camera source.
"""

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
from data_receiver import DEFAULT_PORT
from desktop_settings import DesktopSettings
from image_processing import (
    GREEN_BGR, SHAPE_CIRCLE, SHAPE_SOURCE, SHAPE_SQUARE, ProcessingConfig,
)
from stream_pipeline import StreamPipeline
from virtual_camera import current_os, probe, setup_instructions

PORT = DEFAULT_PORT
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
        self.root.minsize(640, 560)
        self.root.protocol("WM_DELETE_WINDOW", self.on_app_close)

        self.settings = DesktopSettings.load()
        self.root.geometry(self.settings.get("window_geometry") or "640x560")

        self.stream = None
        self.viewer = None
        self._devices = []
        self._closing_stream = False
        self._vcam_available = None
        self._vcam_message = ""
        self._download_window = None
        self._save_job = None
        self._stats_job = None

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
        self.prefetch_models()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=15)
        frm.pack(fill="both", expand=True)
        for col in range(4):
            frm.columnconfigure(col, weight=1)

        row = 0

        ttk.Label(frm, text="Device:").grid(row=row, column=0, sticky="w", padx=5, pady=6)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(
            frm, textvariable=self.device_var, state="readonly", width=30
        )
        self.device_combo.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=6)
        ttk.Button(frm, text="Refresh", command=self.refresh_devices).grid(
            row=row, column=3, sticky="e", padx=5, pady=6
        )
        row += 1

        ttk.Label(frm, text="Frame shape:").grid(row=row, column=0, sticky="w", padx=5, pady=6)
        self.shape_var = tk.StringVar(value=self.config.shape)
        self.shape_var.trace_add("write", self._on_shape_changed)
        ttk.Radiobutton(frm, text="Square", variable=self.shape_var, value=SHAPE_SQUARE).grid(
            row=row, column=1, sticky="w", padx=5, pady=6)
        ttk.Radiobutton(frm, text="Circle", variable=self.shape_var, value=SHAPE_CIRCLE).grid(
            row=row, column=2, sticky="w", padx=5, pady=6)
        ttk.Radiobutton(frm, text="Full frame", variable=self.shape_var, value=SHAPE_SOURCE).grid(
            row=row, column=3, sticky="w", padx=5, pady=6)
        row += 1

        ttk.Label(frm, text="Output:").grid(row=row, column=0, sticky="w", padx=5, pady=6)
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
        ttk.Checkbutton(frm, text="Mirror", variable=self.mirror_var).grid(
            row=row, column=0, sticky="w", padx=5, pady=6)

        self.pin_var = tk.BooleanVar(value=bool(self.settings.get("pin_preview")))
        self.pin_var.trace_add("write", self._on_pin_changed)
        ttk.Checkbutton(frm, text="Pin preview on top", variable=self.pin_var).grid(
            row=row, column=1, columnspan=2, sticky="w", padx=5, pady=6)

        self.preview_var = tk.BooleanVar(value=bool(self.settings.get("show_preview")))
        self.preview_var.trace_add("write", lambda *_: self._schedule_save())
        ttk.Checkbutton(frm, text="Show preview", variable=self.preview_var).grid(
            row=row, column=3, sticky="w", padx=5, pady=6)
        row += 1

        vcam_frame = ttk.LabelFrame(
            frm, text=f"Virtual camera  ({current_os()})", padding=10)
        vcam_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5, pady=(10, 6))
        vcam_frame.columnconfigure(1, weight=1)

        self.vcam_var = tk.BooleanVar(value=bool(self.settings.get("virtual_camera")))
        self.vcam_var.trace_add("write", self._on_vcam_changed)
        ttk.Checkbutton(
            vcam_frame, text="Output as camera device", variable=self.vcam_var
        ).grid(row=0, column=0, sticky="w")

        self.vcam_status_var = tk.StringVar(value="Checking for a driver...")
        ttk.Label(
            vcam_frame, textvariable=self.vcam_status_var, foreground="gray",
            wraplength=330, justify="left",
        ).grid(row=0, column=1, sticky="w", padx=10)

        self.vcam_recheck_btn = ttk.Button(
            vcam_frame, text="Re-check", command=self.check_virtual_camera)
        self.vcam_recheck_btn.grid(row=0, column=2, sticky="e", padx=(0, 6))

        self.vcam_help_btn = ttk.Button(
            vcam_frame, text="Setup help", command=self.show_vcam_help)
        self.vcam_help_btn.grid(row=0, column=3, sticky="e")
        row += 1

        phone_frame = ttk.LabelFrame(frm, text="Phone", padding=10)
        phone_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5, pady=6)
        phone_frame.columnconfigure(1, weight=1)

        ttk.Label(phone_frame, text="Settings:").grid(row=0, column=0, sticky="w")
        self.phone_settings_var = tk.StringVar(value="waiting for handshake")
        ttk.Label(
            phone_frame, textvariable=self.phone_settings_var, foreground="gray",
            wraplength=430, justify="left",
        ).grid(row=0, column=1, columnspan=2, sticky="w", padx=8)

        ttk.Label(phone_frame, text="Edge sharpness:").grid(
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

        ttk.Label(phone_frame, text="Throughput:").grid(row=2, column=0, sticky="w", pady=(6, 0))
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

        ttk.Button(frm, text="Connect to ADB", command=self.connect_adb).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5, pady=6)
        ttk.Button(frm, text="Pair over WiFi", command=self.pair_wifi).grid(
            row=row, column=2, columnspan=2, sticky="w", padx=5, pady=6)

    # -------------------------------------------------------- config changes

    def _on_shape_changed(self, *_):
        self.config.shape = self.shape_var.get()
        self._schedule_save()

    def _on_mirror_changed(self, *_):
        self.config.mirror = self.mirror_var.get()
        self._schedule_save()

    def _on_resolution_changed(self, *_):
        size = dict(RESOLUTIONS).get(self.res_var.get())
        if not size:
            return
        self.config.output_size = size
        self._schedule_save()
        if self.stream is not None:
            self.stream.set_output_size(*size)
            self.status_var.set(
                f"Output {size[0]}x{size[1]} - reselect the camera in your app")

    def _on_pin_changed(self, *_):
        if self.viewer is not None:
            self.viewer.set_pinned(self.pin_var.get())
        self._schedule_save()

    def _on_vcam_changed(self, *_):
        if self.stream is not None:
            self.stream.set_virtual_camera_enabled(self.vcam_var.get())
        self._schedule_save()

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
            "virtual_camera": self.vcam_var.get(),
            "segmenter_model": self.selected_segmenter(),
            "mask_sharpness": int(float(self.sharpness_var.get())),
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
        self._vcam_available = ok
        self._vcam_message = message
        if ok:
            self.vcam_status_var.set(f"✔ {message}")
            if self.settings.get("virtual_camera"):
                self.vcam_var.set(True)
        else:
            self.vcam_status_var.set("No driver found - click Setup help")
            self.vcam_var.set(False)

    def show_vcam_help(self):
        text = setup_instructions()
        if self._vcam_message and not self._vcam_available:
            text = f"{text}\n\n---\nDetail:\n{self._vcam_message}"
        HelpWindow(self.root, "Virtual camera setup", text)

    def _on_vcam_status(self, message):
        self.root.after(0, lambda: self.vcam_status_var.set(f"✔ {message}"))

    def _on_vcam_error(self, message):
        def show():
            self.vcam_status_var.set("Virtual camera stopped - click Setup help")
            self._vcam_available = False
            self._vcam_message = message
            self.vcam_var.set(False)
            HelpWindow(self.root, "Virtual camera error", message)

        self.root.after(0, show)

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
            on_preview_frame=self._on_preview_frame,
            on_connected=lambda: self.root.after(0, self.on_stream_connected),
            on_disconnected=lambda: self.root.after(0, self.on_stream_disconnected),
            on_vcam_status=self._on_vcam_status,
            on_vcam_error=self._on_vcam_error,
            on_settings_received=self._on_settings_received,
            on_ai_status=self._on_ai_status,
        )

        if self.preview_var.get():
            self.viewer = ViewerWindow(
                self.root, pinned=self.pin_var.get(), on_closed=self.on_viewer_closed)

        self.save_settings()
        self.stream.start()
        self._poll_stats()
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
            if self._stats_job is not None:
                try:
                    self.root.after_cancel(self._stats_job)
                except tk.TclError:
                    pass
                self._stats_job = None
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
        self._stats_job = self.root.after(1000, self._poll_stats)

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