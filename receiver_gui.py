#File Info: name = receiver_gui.py


import socket
import struct
import subprocess
import threading
import queue
import time

import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import ttk, messagebox

from adb_wifi_dialog import WifiPairDialog

PORT = 4343
BG_KEY_COLOR = "#00FF00"
RECONNECT_DELAY = 1.0


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


def recv_exact(sock, size):
    buf = b""
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


class ConfigWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Mob Cam - Config")
        self.root.minsize(520, 320)
        self.root.geometry("520x320")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_app_close)

        self.viewer = None
        self._devices = []

        self._build_ui()
        self.refresh_devices()

    def on_app_close(self):
        if self.viewer is not None:
            self.viewer.close()
        # drops any wireless-debugging (TCP/IP) connections; USB devices are unaffected
        subprocess.run(["adb", "disconnect"], capture_output=True)
        self.root.destroy()
    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=15)
        frm.pack(fill="both", expand=True)
        for col in range(4):
            frm.columnconfigure(col, weight=1)

        ttk.Label(frm, text="Device:").grid(row=0, column=0, sticky="w", padx=5, pady=8)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(
            frm, textvariable=self.device_var, state="readonly", width=32
        )
        self.device_combo.grid(row=0, column=1, columnspan=2, sticky="ew", padx=5, pady=8)

        ttk.Button(frm, text="Refresh", command=self.refresh_devices).grid(
            row=0, column=3, sticky="e", padx=5, pady=8
        )

        ttk.Label(frm, text="Frame shape:").grid(row=1, column=0, sticky="w", padx=5, pady=8)
        self.shape_var = tk.StringVar(value="square")
        self.shape_var.trace_add("write", self._on_shape_changed)
        ttk.Radiobutton(
            frm, text="Square", variable=self.shape_var, value="square"
        ).grid(row=1, column=1, sticky="w", padx=5, pady=8)
        ttk.Radiobutton(
            frm, text="Circle", variable=self.shape_var, value="circle"
        ).grid(row=1, column=2, sticky="w", padx=5, pady=8)

        self.pin_var = tk.BooleanVar(value=False)
        self.pin_var.trace_add("write", self._on_pin_changed)
        ttk.Checkbutton(
            frm, text="Pin on top", variable=self.pin_var
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=8)

        self.connect_btn = ttk.Button(
            frm, text="Connect", command=self.on_connect_clicked
        )
        self.connect_btn.grid(row=3, column=0, sticky="w", padx=5, pady=(15, 8))

        self.status_dot_var = tk.StringVar(value="")
        ttk.Label(
            frm, textvariable=self.status_dot_var, foreground="#2ecc71",
            font=("TkDefaultFont", 11, "bold")
        ).grid(row=3, column=1, sticky="w", padx=(5, 0), pady=(15, 8))

        self.status_var = tk.StringVar(value="Not connected")
        ttk.Label(frm, textvariable=self.status_var, foreground="gray").grid(
            row=3, column=2, columnspan=2, sticky="w", padx=5, pady=(15, 8)
        )

        ttk.Button(frm, text="Connect to ADB", command=self.connect_adb).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=5, pady=8
        )
        ttk.Button(frm, text="Pair over WiFi", command=self.pair_wifi).grid(
            row=4, column=2, columnspan=2, sticky="w", padx=5, pady=8
        )

    def _on_shape_changed(self, *_):
        if self.viewer is not None:
            self.viewer.set_shape(self.shape_var.get())

    def _on_pin_changed(self, *_):
        if self.viewer is not None:
            self.viewer.set_pinned(self.pin_var.get())

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

    def on_connect_clicked(self):
        if self.viewer is not None:
            self.viewer.close()
            return

        serial = self.get_selected_serial()
        if not serial:
            messagebox.showwarning("No device", "Select a device first.")
            return

        try:
            subprocess.check_call(
                ["adb", "-s", serial, "forward", f"tcp:{PORT}", f"tcp:{PORT}"]
            )
        except subprocess.CalledProcessError as e:
            messagebox.showerror("adb forward failed", str(e))
            return

        self.status_dot_var.set("")
        self.status_var.set("Waiting for frames from device...")
        self.connect_btn.config(text="Disconnect")

        self.viewer = ViewerWindow(
            self.root,
            shape=self.shape_var.get(),
            pinned=self.pin_var.get(),
            on_closed=self.on_viewer_closed,
            on_connected=self.on_stream_connected,
            on_disconnected=self.on_stream_disconnected,
        )

    def on_stream_connected(self):
        self.status_dot_var.set("\u2714")
        self.status_var.set("Connected - receiving frames")

    def on_stream_disconnected(self):
        self.status_dot_var.set("")
        self.status_var.set("Waiting for frames from device...")

    def on_viewer_closed(self):
        self.viewer = None
        self.connect_btn.config(text="Connect")
        self.status_dot_var.set("")
        self.status_var.set("Disconnected")


class ViewerWindow:
    """Normal, WM-managed, resizable window that displays incoming frames on
    a green-screen background, with live shape switching and pin/unpin."""

    def __init__(self, parent, shape="square", pinned=False, on_closed=None,
                 on_connected=None, on_disconnected=None):
        self.shape = shape
        self.pinned = pinned
        self.on_closed = on_closed
        self.on_connected = on_connected
        self.on_disconnected = on_disconnected

        self.win = tk.Toplevel(parent)
        self.win.title("Mob Cam Feed")
        self.win.geometry("400x400")
        self.win.minsize(150, 150)
        self.win.configure(bg=BG_KEY_COLOR)
        self.win.attributes("-topmost", self.pinned)

        self.label = tk.Label(self.win, bg=BG_KEY_COLOR, bd=0, highlightthickness=0)
        self.label.pack(fill="both", expand=True)

        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.last_frame = None

        self.thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.thread.start()

        self.win.after(30, self._update_frame)
        self.win.protocol("WM_DELETE_WINDOW", self.close)

    def set_pinned(self, pinned: bool):
        self.pinned = pinned
        self.win.attributes("-topmost", pinned)

    def set_shape(self, shape_name: str):
        self.shape = shape_name
        if self.last_frame is not None:
            self._render(self.last_frame)

    def _stream_worker(self):
        # keeps retrying the connection so it doesn't matter whether the
        # device starts sending before or after this window opens, and
        # reconnects automatically if the stream drops
        while not self.stop_event.is_set():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.settimeout(2)
                sock.connect(("localhost", PORT))
                sock.settimeout(None)
            except OSError:
                sock.close()
                time.sleep(RECONNECT_DELAY)
                continue

            if self.on_connected:
                self.win.after(0, self.on_connected)

            try:
                while not self.stop_event.is_set():
                    header = recv_exact(sock, 4)
                    if header is None:
                        break

                    frame_len = struct.unpack(">I", header)[0]
                    jpeg_bytes = recv_exact(sock, frame_len)
                    if jpeg_bytes is None:
                        break

                    frame_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass
            finally:
                sock.close()
                if self.on_disconnected:
                    self.win.after(0, self.on_disconnected)

            if not self.stop_event.is_set():
                time.sleep(RECONNECT_DELAY)

    def _update_frame(self):
        try:
            frame = self.frame_queue.get_nowait()
            self.last_frame = frame
            self._render(frame)
        except queue.Empty:
            pass

        if not self.stop_event.is_set():
            self.win.after(30, self._update_frame)
        else:
            self.close()

    def _render(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        square_img = crop_to_square(Image.fromarray(frame_rgb))

        label_w = self.label.winfo_width()
        label_h = self.label.winfo_height()
        if label_w < 2 or label_h < 2:
            return

        # fit the square frame into whatever size/aspect the window currently is,
        # letterboxed with the green background so it's never stretched/distorted
        side = min(label_w, label_h)
        resized = square_img.resize((side, side), Image.LANCZOS)

        if self.shape == "circle":
            mask = Image.new("L", (side, side), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, side, side), fill=255)
            canvas = Image.new("RGB", (side, side), BG_KEY_COLOR)
            resized = Image.composite(resized, canvas, mask)

        canvas_full = Image.new("RGB", (label_w, label_h), BG_KEY_COLOR)
        offset = ((label_w - side) // 2, (label_h - side) // 2)
        canvas_full.paste(resized, offset)

        photo = ImageTk.PhotoImage(image=canvas_full)
        self.label.configure(image=photo)
        self.label.image = photo

    def close(self):
        self.stop_event.set()
        try:
            self.win.destroy()
        except tk.TclError:
            pass
        if self.on_closed:
            self.on_closed()


def main():
    root = tk.Tk()
    ConfigWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()