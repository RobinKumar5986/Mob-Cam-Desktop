"""Wireless-debugging pairing dialog.

Two tabs onto the same flow:
  QR Code       - generates the WIFI:T:ADB payload the phone's "Pair with QR
                  code" scanner expects, then waits for the mDNS broadcast.
  Pairing Code  - manual IP:port + 6-digit code fallback.

Both end in `adb pair`, then auto-discovery of the connect address and
`adb connect`, so the device shows up in the main window's device list.
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox

import qrcode
from PIL import ImageTk

from adb_wifi import (
    random_service_name, random_password, run_adb,
    wait_for_pairing_service, wait_for_connect_service,
)

CONNECTED_COLOR = "#2ecc71"
DEFAULT_COLOR = "gray"


class WifiPairDialog(tk.Toplevel):
    def __init__(self, parent, on_paired=None):
        """on_paired, if given, is called as on_paired(address) with the
        'host:port' string that was just connected, once adb connect succeeds.
        """
        super().__init__(parent)
        self.title("Pair over WiFi")
        self.geometry("360x460")
        self.resizable(False, False)
        self.on_paired = on_paired
        self._pairing_started = False
        self._service_name = None
        self._destroyed = False

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.qr_tab = ttk.Frame(notebook)
        self.pin_tab = ttk.Frame(notebook)
        notebook.add(self.qr_tab, text="QR Code")
        notebook.add(self.pin_tab, text="Pairing Code")

        self._build_qr_tab()
        self._build_pin_tab()
        notebook.bind("<<NotebookTabChanged>>", lambda e: self._on_tab_changed(notebook))

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        # background mDNS threads are daemons and will time out on their own;
        # this flag stops them touching a dead widget
        self._destroyed = True
        self.destroy()

    def _later(self, fn):
        """Schedule on the Tk thread, unless the dialog is already gone."""
        if not self._destroyed:
            try:
                self.after(0, fn)
            except tk.TclError:
                pass

    # ------------------------------------------------------------- QR tab

    def _build_qr_tab(self):
        self.qr_label = ttk.Label(self.qr_tab)
        self.qr_label.pack(pady=10)

        self.qr_status_var = tk.StringVar(
            value="Open Wireless debugging > Pair with QR code on your phone"
        )
        self.qr_status_label = ttk.Label(
            self.qr_tab, textvariable=self.qr_status_var, wraplength=320,
            foreground=DEFAULT_COLOR
        )
        self.qr_status_label.pack(pady=5)

    def _start_qr_pairing(self):
        if self._pairing_started:
            return
        self._pairing_started = True

        self._service_name = random_service_name()
        password = random_password()
        payload = f"WIFI:T:ADB;S:{self._service_name};P:{password};;"

        img = qrcode.make(payload).resize((260, 260))
        self._qr_photo = ImageTk.PhotoImage(img)
        self.qr_label.configure(image=self._qr_photo)

        threading.Thread(
            target=wait_for_pairing_service,
            args=(self._service_name, self._on_qr_pairing_found),
            kwargs={"timeout": 90},
            daemon=True,
        ).start()

    def _on_qr_pairing_found(self, host, port):
        if host is None:
            self._later(lambda: self._set_status(
                self.qr_status_label, self.qr_status_var, "No scan detected, try again."
            ))
            self._pairing_started = False
            return
        # for QR pairing adb uses the service name itself as the pairing code
        ok, output = run_adb("pair", f"{host}:{port}", self._service_name)
        self._later(lambda: self._finish_pairing(ok, output))

    # ------------------------------------------------------------ PIN tab

    def _build_pin_tab(self):
        frm = self.pin_tab
        ttk.Label(frm, text="IP:Port from 'Pair device with pairing code':").pack(
            anchor="w", pady=(10, 0)
        )
        self.pin_addr_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.pin_addr_var).pack(fill="x", pady=5)

        ttk.Label(frm, text="6-digit pairing code:").pack(anchor="w")
        self.pin_code_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.pin_code_var).pack(fill="x", pady=5)

        ttk.Button(frm, text="Pair", command=self._pair_with_pin).pack(pady=10)

        self.pin_status_var = tk.StringVar()
        self.pin_status_label = ttk.Label(
            frm, textvariable=self.pin_status_var, foreground=DEFAULT_COLOR,
            wraplength=320
        )
        self.pin_status_label.pack()

    def _pair_with_pin(self):
        addr = self.pin_addr_var.get().strip()
        code = self.pin_code_var.get().strip()
        if not addr or not code:
            messagebox.showwarning("Missing info", "Enter both the address and the code.")
            return
        self._set_status(self.pin_status_label, self.pin_status_var, "Pairing...")
        threading.Thread(target=self._pin_worker, args=(addr, code), daemon=True).start()

    def _pin_worker(self, addr, code):
        ok, output = run_adb("pair", addr, code)
        self._later(lambda: self._finish_pairing(ok, output))

    # ------------------------------------------------------------- shared

    def _set_status(self, label, var, text, color=DEFAULT_COLOR, big=False):
        var.set(text)
        label.configure(
            foreground=color,
            font=("TkDefaultFont", 14, "bold") if big else ("TkDefaultFont", 9, "normal"),
        )

    def _set_both(self, text, color=DEFAULT_COLOR, big=False):
        self._set_status(self.qr_status_label, self.qr_status_var, text, color, big)
        self._set_status(self.pin_status_label, self.pin_status_var, text, color, big)

    def _finish_pairing(self, ok, output):
        if not ok:
            messagebox.showerror("Pairing failed", output)
            self._set_both("Pairing failed, try again.")
            self._pairing_started = False
            return

        self._set_both("Paired! Looking for the connect address...")
        threading.Thread(target=self._connect_worker, daemon=True).start()

    def _connect_worker(self):
        wait_for_connect_service(
            lambda h, p: self._later(lambda: self._finish_connect(h, p))
        )

    def _finish_connect(self, host, port):
        if host is None:
            self._set_both("Paired, but couldn't auto-detect the connect address.")
            return

        address = f"{host}:{port}"
        ok, output = run_adb("connect", address)
        if ok:
            self._set_both(f"✔ Connected to {address}", CONNECTED_COLOR, big=True)
            if self.on_paired:
                self.on_paired(address)
        else:
            messagebox.showerror("Connect failed", output)

    def _on_tab_changed(self, notebook):
        if notebook.tab(notebook.select(), "text") == "QR Code":
            self._start_qr_pairing()