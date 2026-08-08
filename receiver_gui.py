import socket
import struct
import subprocess
import threading
import queue

import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

PORT = 5000


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


class ReceiverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mob Cam Receiver")
        self.root.geometry("900x700")

        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.stream_thread = None
        self.connected = False

        self._build_ui()
        self.refresh_devices()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Device:").pack(side="left")

        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(
            top, textvariable=self.device_var, state="readonly", width=50
        )
        self.device_combo.pack(side="left", padx=5)

        ttk.Button(top, text="Refresh", command=self.refresh_devices).pack(
            side="left", padx=5
        )

        self.connect_btn = ttk.Button(
            top, text="Connect", command=self.on_connect_clicked
        )
        self.connect_btn.pack(side="left", padx=5)

        self.status_var = tk.StringVar(value="Not connected")
        ttk.Label(top, textvariable=self.status_var, foreground="gray").pack(
            side="left", padx=10
        )

        self.video_label = ttk.Label(self.root, background="black")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def refresh_devices(self):
        devices = list_adb_devices()
        if not devices:
            self.device_combo["values"] = []
            self.device_var.set("")
            self.status_var.set("No devices found")
            return

        display_values = [f"{serial}  ({model})" for serial, model in devices]
        self.device_combo["values"] = display_values
        self.device_combo.current(0)
        self._devices = devices
        self.status_var.set(f"{len(devices)} device(s) found")

    def get_selected_serial(self):
        idx = self.device_combo.current()
        if idx < 0 or not hasattr(self, "_devices"):
            return None
        return self._devices[idx][0]

    def on_connect_clicked(self):
        if self.connected:
            self.disconnect()
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

        self.stop_event.clear()
        self.status_var.set(f"Connecting to {serial} ...")
        self.connect_btn.config(text="Disconnect")
        self.connected = True

        self.stream_thread = threading.Thread(
            target=self.stream_worker, daemon=True
        )
        self.stream_thread.start()

        self.root.after(30, self.update_frame)

    def disconnect(self):
        self.stop_event.set()
        self.connected = False
        self.connect_btn.config(text="Connect")
        self.status_var.set("Disconnected")

    def stream_worker(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(5)
            sock.connect(("localhost", PORT))
            sock.settimeout(None)
            self.status_var.set("Connected — streaming")
        except OSError as e:
            self.status_var.set(f"Connection failed: {e}")
            self.connected = False
            self.connect_btn.config(text="Connect")
            return

        try:
            while not self.stop_event.is_set():
                header = recv_exact(sock, 4)
                if header is None:
                    self.status_var.set("Device closed connection")
                    break

                frame_len = struct.unpack(">I", header)[0]
                jpeg_bytes = recv_exact(sock, frame_len)
                if jpeg_bytes is None:
                    self.status_var.set("Device closed connection")
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
            self.connected = False
            self.connect_btn.config(text="Connect")

    def update_frame(self):
        try:
            frame = self.frame_queue.get_nowait()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            label_w = self.video_label.winfo_width() or frame_rgb.shape[1]
            label_h = self.video_label.winfo_height() or frame_rgb.shape[0]
            img = Image.fromarray(frame_rgb)
            img.thumbnail((label_w, label_h))

            photo = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=photo)
            self.video_label.image = photo
        except queue.Empty:
            pass

        if self.connected:
            self.root.after(30, self.update_frame)

    def on_close(self):
        self.stop_event.set()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ReceiverApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()