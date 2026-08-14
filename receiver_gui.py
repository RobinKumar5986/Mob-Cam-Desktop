"""Mob Cam receiver GUI.

Thin UI layer over the pipeline:

    frame_receiver  ->  image_processing  ->  virtual_camera
                                          \\->  preview window (this file)

The window is only a local monitor now. What other applications actually see is
the virtual camera device, so Zoom / Meet / Teams / OBS / browsers list "Mob Cam"
(or "OBS Virtual Camera", depending on the driver) in their camera dropdown.
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
from frame_receiver import DEFAULT_PORT
from image_processing import (
    GREEN_BGR,
    SHAPE_CIRCLE,
    SHAPE_SOURCE,
    SHAPE_SQUARE,
    ProcessingConfig,
)
from stream_pipeline import StreamPipeline
from virtual_camera import current_os, probe, setup_instructions

PORT = DEFAULT_PORT
BG_KEY_COLOR = "#00FF00"  # same green as image_processing.GREEN_BGR

RESOLUTIONS = [
    ("720 x 720 (square)", (720, 720)),
    ("480 x 480 (square)", (480, 480)),
    ("1080 x 1080 (square)", (1080, 1080)),
    ("1280 x 720 (16:9)", (1280, 720)),
    ("1920 x 1080 (16:9)", (1920, 1080)),
    ("640 x 480 (4:3)", (640, 480)),
]
FPS_OPTIONS = [30, 15, 24, 60]


def list_adb_devices():
    """Return list of (serial, model) tuples for connected/authorized devices."""
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
        self.root.minsize(600, 470)
        self.root.geometry("600x470")
        self.root.protocol("WM_DELETE_WINDOW", self.on_app_close)

        self.stream = None
        self.viewer = None
        self._devices = []
        self._closing_stream = False
        self._vcam_available = None

        self.config = ProcessingConfig(
            shape=SHAPE_SQUARE,
            output_size=RESOLUTIONS[0][1],
            background=GREEN_BGR,
        )

        self._build_ui()
        self.refresh_devices()
        self.check_virtual_camera()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=15)
        frm.pack(fill="both", expand=True)
        for col in range(4):
            frm.columnconfigure(col, weight=1)

        row = 0

        # -- device
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

        # -- shape
        ttk.Label(frm, text="Frame shape:").grid(row=row, column=0, sticky="w", padx=5, pady=6)
        self.shape_var = tk.StringVar(value=SHAPE_SQUARE)
        self.shape_var.trace_add("write", self._on_shape_changed)
        ttk.Radiobutton(frm, text="Square", variable=self.shape_var, value=SHAPE_SQUARE).grid(
            row=row, column=1, sticky="w", padx=5, pady=6
        )
        ttk.Radiobutton(frm, text="Circle", variable=self.shape_var, value=SHAPE_CIRCLE).grid(
            row=row, column=2, sticky="w", padx=5, pady=6
        )
        ttk.Radiobutton(frm, text="Full frame", variable=self.shape_var, value=SHAPE_SOURCE).grid(
            row=row, column=3, sticky="w", padx=5, pady=6
        )
        row += 1

        # -- output resolution / fps
        ttk.Label(frm, text="Output:").grid(row=row, column=0, sticky="w", padx=5, pady=6)
        self.res_var = tk.StringVar(value=RESOLUTIONS[0][0])
        res_combo = ttk.Combobox(
            frm, textvariable=self.res_var, state="readonly",
            values=[label for label, _ in RESOLUTIONS], width=22,
        )
        res_combo.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=6)
        self.res_var.trace_add("write", self._on_resolution_changed)

        self.fps_var = tk.StringVar(value=str(FPS_OPTIONS[0]))
        fps_combo = ttk.Combobox(
            frm, textvariable=self.fps_var, state="readonly",
            values=[str(f) for f in FPS_OPTIONS], width=6,
        )
        fps_combo.grid(row=row, column=3, sticky="e", padx=5, pady=6)
        row += 1

        # -- toggles
        self.mirror_var = tk.BooleanVar(value=False)
        self.mirror_var.trace_add("write", self._on_mirror_changed)
        ttk.Checkbutton(frm, text="Mirror", variable=self.mirror_var).grid(
            row=row, column=0, sticky="w", padx=5, pady=6
        )

        self.pin_var = tk.BooleanVar(value=False)
        self.pin_var.trace_add("write", self._on_pin_changed)
        ttk.Checkbutton(frm, text="Pin preview on top", variable=self.pin_var).grid(
            row=row, column=1, columnspan=2, sticky="w", padx=5, pady=6
        )

        self.preview_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Show preview", variable=self.preview_var).grid(
            row=row, column=3, sticky="w", padx=5, pady=6
        )
        row += 1

        # -- virtual camera
        vcam_frame = ttk.LabelFrame(
            frm, text=f"Virtual camera  ({current_os()})", padding=10
        )
        vcam_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5, pady=(10, 6))
        vcam_frame.columnconfigure(1, weight=1)

        self.vcam_var = tk.BooleanVar(value=True)
        self.vcam_var.trace_add("write", self._on_vcam_changed)
        self.vcam_check = ttk.Checkbutton(
            vcam_frame, text="Output as camera device", variable=self.vcam_var
        )
        self.vcam_check.grid(row=0, column=0, sticky="w")

        self.vcam_status_var = tk.StringVar(value="Checking for a driver...")
        ttk.Label(
            vcam_frame, textvariable=self.vcam_status_var, foreground="gray",
            wraplength=380, justify="left",
        ).grid(row=0, column=1, sticky="w", padx=10)

        self.vcam_help_btn = ttk.Button(
            vcam_frame, text="Setup help", command=self.show_vcam_help
        )
        self.vcam_help_btn.grid(row=0, column=2, sticky="e")
        row += 1

        # -- connect
        self.connect_btn = ttk.Button(frm, text="Connect", command=self.on_connect_clicked)
        self.connect_btn.grid(row=row, column=0, sticky="w", padx=5, pady=(10, 6))

        self.status_dot_var = tk.StringVar(value="")
        ttk.Label(
            frm, textvariable=self.status_dot_var, foreground="#2ecc71",
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=row, column=1, sticky="w", padx=(5, 0), pady=(10, 6))

        self.status_var = tk.StringVar(value="Not connected")
        ttk.Label(frm, textvariable=self.status_var, foreground="gray").grid(
            row=row, column=2, columnspan=2, sticky="w", padx=5, pady=(10, 6)
        )
        row += 1

        ttk.Button(frm, text="Connect to ADB", command=self.connect_adb).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5, pady=6
        )
        ttk.Button(frm, text="Pair over WiFi", command=self.pair_wifi).grid(
            row=row, column=2, columnspan=2, sticky="w", padx=5, pady=6
        )

    # -------------------------------------------------------- config changes

    def _on_shape_changed(self, *_):
        self.config.shape = self.shape_var.get()

    def _on_mirror_changed(self, *_):
        self.config.mirror = self.mirror_var.get()

    def _on_resolution_changed(self, *_):
        size = dict(RESOLUTIONS).get(self.res_var.get())
        if size:
            self.config.output_size = size

    def _on_pin_changed(self, *_):
        if self.viewer is not None:
            self.viewer.set_pinned(self.pin_var.get())

    def _on_vcam_changed(self, *_):
        if self.stream is not None:
            self.stream.set_virtual_camera_enabled(self.vcam_var.get())

    # ------------------------------------------------------ virtual camera

    def check_virtual_camera(self):
        threading.Thread(target=self._probe_worker, daemon=True).start()

    def _probe_worker(self):
        ok, message = probe()
        self.root.after(0, lambda: self._apply_probe_result(ok, message))

    def _apply_probe_result(self, ok, message):
        self._vcam_available = ok
        self._vcam_message = message
        if ok:
            self.vcam_status_var.set(f"✔ {message}")
            self.vcam_help_btn.state(["disabled"])
        else:
            self.vcam_status_var.set("No driver found - click Setup help")
            self.vcam_var.set(False)
            self.vcam_help_btn.state(["!disabled"])

    def show_vcam_help(self):
        detail = getattr(self, "_vcam_message", "") or ""
        text = setup_instructions()
        if detail and not self._vcam_available:
            text = f"{text}\n\n---\nDetail:\n{detail}"
        HelpWindow(self.root, "Virtual camera setup", text)

    def _on_vcam_status(self, message):
        self.root.after(0, lambda: self.vcam_status_var.set(f"✔ {message}"))

    def _on_vcam_error(self, message):
        def show():
            self.vcam_status_var.set("Virtual camera stopped - click Setup help")
            self._vcam_available = False
            self._vcam_message = message
            self.vcam_help_btn.state(["!disabled"])
            self.vcam_var.set(False)
            HelpWindow(self.root, "Virtual camera error", message)

        self.root.after(0, show)

    # --------------------------------------------------------------- adb

    def refresh_devices(self):
        devices = list_adb_devices()
        if not devices:
            self.device_combo["values"] = []
            self.device_var.set("")
            self.status_var.set("No devices found")
            self._devices = []
            return

        self._devices = devices
        self.device_combo["values"] = [f"{s}  ({m})" for s, m in devices]
        self.device_combo.current(0)
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
        subprocess.run(["adb", "start-server"], capture_output=True)
        self.root.after(0, self.refresh_devices)

    def pair_wifi(self):
        WifiPairDialog(self.root, on_paired=self.refresh_devices)

    # ------------------------------------------------------------- stream

    def on_connect_clicked(self):
        if self.stream is not None:
            self.stop_stream()
            return

        serial = self.get_selected_serial()
        if not serial:
            messagebox.showwarning("No device", "Select a device first.")
            return

        try:
            subprocess.check_call(
                ["adb", "-s", serial, "forward", f"tcp:{PORT}", f"tcp:{PORT}"]
            )
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
            on_preview_frame=self._on_preview_frame,
            on_connected=lambda: self.root.after(0, self.on_stream_connected),
            on_disconnected=lambda: self.root.after(0, self.on_stream_disconnected),
            on_vcam_status=self._on_vcam_status,
            on_vcam_error=self._on_vcam_error,
        )

        if self.preview_var.get():
            self.viewer = ViewerWindow(
                self.root, pinned=self.pin_var.get(), on_closed=self.on_viewer_closed
            )

        self.stream.start()
        self.status_dot_var.set("")
        self.status_var.set("Waiting for frames from device...")
        self.connect_btn.config(text="Disconnect")

    def stop_stream(self):
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
        finally:
            self._closing_stream = False

    def _on_preview_frame(self, frame):
        """Called on the receiver thread; hand off to the viewer's queue."""
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
        # Closing the preview must not kill the camera feed.
        self.viewer = None
        self.preview_var.set(False)

    def on_app_close(self):
        self.stop_stream()
        # drops any wireless-debugging (TCP/IP) connections; USB devices are unaffected
        subprocess.run(["adb", "disconnect"], capture_output=True)
        self.root.destroy()


class ViewerWindow:
    """Local monitor for the processed frames. Purely cosmetic: the virtual
    camera keeps running whether or not this window is open."""

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
        """Thread-safe: newest frame wins, stale ones are dropped."""
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

        image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image=image)
        self.label.configure(image=photo)
        self.label.image = photo

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.win.destroy()
        except tk.TclError:
            pass
        if self.on_closed:
            self.on_closed()


class HelpWindow(tk.Toplevel):
    """Scrollable, selectable text window, so setup commands can be copied."""

    def __init__(self, parent, title, text):
        super().__init__(parent)
        self.title(title)
        self.geometry("560x360")

        box = scrolledtext.ScrolledText(self, wrap="word", padx=10, pady=10)
        box.pack(fill="both", expand=True)
        box.insert("1.0", text)
        box.configure(state="normal")  # keep editable so selection/copy works

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=8)


def main():
    root = tk.Tk()
    ConfigWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()