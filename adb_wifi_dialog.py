#File Info: name = wifi_dialog.py

import threading
import tkinter as tk
from tkinter import ttk, messagebox

import qrcode
from PIL import ImageTk

from adb_wifi import (
    random_service_name, random_password, run_adb,
    wait_for_pairing_service, wait_for_connect_service,
)


class WifiPairDialog(tk.Toplevel):
    def __init__(self, parent, on_paired=None):
        super().__init__(parent)
        self.title("Pair over WiFi")
        self.geometry("360x420")
        self.resizable(False, False)
        self.on_paired = on_paired
        self._pairing_started = False
        self._service_name = None

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.qr_tab = ttk.Frame(notebook)
        self.pin_tab = ttk.Frame(notebook)
        notebook.add(self.qr_tab, text="QR Code")
        notebook.add(self.pin_tab, text="Pairing Code")

        self._build_qr_tab()
        self._build_pin_tab()
        notebook.bind("<<NotebookTabChanged>>", lambda e: self._on_tab_changed(notebook))

    def _build_qr_tab(self):
        self.qr_label = ttk.Label(self.qr_tab)
        self.qr_label.pack(pady=10)
        self.qr_status = tk.StringVar(
            value="Open Wireless debugging > Pair with QR code on your phone"
        )
        ttk.Label(self.qr_tab, textvariable=self.qr_status, wraplength=320,
                  foreground="gray").pack(pady=5)

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
            self.after(0, lambda: self.qr_status.set("No scan detected, try again."))
            return
        # for QR pairing adb uses the service name itself as the pairing code
        ok, output = run_adb("pair", f"{host}:{port}", self._service_name)
        self.after(0, lambda: self._finish_pairing(ok, output))

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
        self.pin_status = tk.StringVar()
        ttk.Label(frm, textvariable=self.pin_status, foreground="gray",
                  wraplength=320).pack()

    def _pair_with_pin(self):
        addr = self.pin_addr_var.get().strip()
        code = self.pin_code_var.get().strip()
        if not addr or not code:
            messagebox.showwarning("Missing info", "Enter both the address and the code.")
            return
        self.pin_status.set("Pairing...")
        threading.Thread(target=self._pin_worker, args=(addr, code), daemon=True).start()

    def _pin_worker(self, addr, code):
        ok, output = run_adb("pair", addr, code)
        self.after(0, lambda: self._finish_pairing(ok, output))

    def _finish_pairing(self, ok, output):
        if not ok:
            messagebox.showerror("Pairing failed", output)
            self.qr_status.set("Pairing failed, try again.")
            self.pin_status.set("Pairing failed, try again.")
            self._pairing_started = False
            return

        self.qr_status.set("Paired! Looking for the connect address...")
        self.pin_status.set("Paired! Looking for the connect address...")
        threading.Thread(target=self._connect_worker, daemon=True).start()

    def _connect_worker(self):
        wait_for_connect_service(lambda h, p: self.after(0, lambda: self._finish_connect(h, p)))

    def _finish_connect(self, host, port):
        if host is None:
            msg = "Paired, but couldn't auto-detect the connect address."
            self.qr_status.set(msg)
            self.pin_status.set(msg)
            return

        ok, output = run_adb("connect", f"{host}:{port}")
        if ok:
            self.qr_status.set(f"Connected to {host}:{port}")
            self.pin_status.set(f"Connected to {host}:{port}")
            if self.on_paired:
                self.on_paired()
        else:
            messagebox.showerror("Connect failed", output)

    def _on_tab_changed(self, notebook):
        if notebook.tab(notebook.select(), "text") == "QR Code":
            self._start_qr_pairing()