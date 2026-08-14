"""Frame receiver.

First stage of the pipeline: reads length-prefixed JPEG frames off the adb-
forwarded TCP socket and hands decoded BGR numpy frames to a callback.

No Tk, no processing, no camera. Keeping this headless means the same receiver
can drive the GUI preview, the virtual camera, a headless service or a test.

Wire format, per frame:
    4 bytes  big-endian unsigned frame length
    N bytes  JPEG payload
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import cv2
import numpy as np

DEFAULT_PORT = 4343
RECONNECT_DELAY = 1.0
CONNECT_TIMEOUT = 2.0
MAX_FRAME_BYTES = 32 * 1024 * 1024  # sanity guard against a desynced stream


def recv_exact(sock: socket.socket, size: int) -> bytes | None:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class FrameReceiver:
    """Background thread that keeps a frame stream alive.

    Retries the connection forever, so it does not matter whether the phone
    starts sending before or after this is started, and it recovers on its own
    if the stream drops.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        on_frame=None,
        on_connected=None,
        on_disconnected=None,
        reconnect_delay: float = RECONNECT_DELAY,
    ):
        self.host = host
        self.port = port
        self.on_frame = on_frame
        self.on_connected = on_connected
        self.on_disconnected = on_disconnected
        self.reconnect_delay = reconnect_delay

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = False
        self.frames_received = 0

    # ----------------------------------------------------------- lifecycle

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="frame-receiver", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._thread = None

    # -------------------------------------------------------------- worker

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.settimeout(CONNECT_TIMEOUT)
                sock.connect((self.host, self.port))
                sock.settimeout(None)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                sock.close()
                self._sleep(self.reconnect_delay)
                continue

            self._set_connected(True)
            try:
                self._read_loop(sock)
            finally:
                sock.close()
                self._set_connected(False)

            self._sleep(self.reconnect_delay)

    def _read_loop(self, sock: socket.socket) -> None:
        while not self._stop_event.is_set():
            header = recv_exact(sock, 4)
            if header is None:
                return

            frame_len = struct.unpack(">I", header)[0]
            if frame_len == 0 or frame_len > MAX_FRAME_BYTES:
                # Stream is out of sync; drop the connection and resynchronise.
                return

            jpeg_bytes = recv_exact(sock, frame_len)
            if jpeg_bytes is None:
                return

            frame = cv2.imdecode(
                np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                continue

            self.frames_received += 1
            if self.on_frame is not None:
                try:
                    self.on_frame(frame)
                except Exception as exc:  # noqa: BLE001
                    print(f"[frame_receiver] on_frame failed: {exc}")

    # --------------------------------------------------------------- utils

    def _set_connected(self, connected: bool) -> None:
        if connected == self._connected:
            return
        self._connected = connected
        callback = self.on_connected if connected else self.on_disconnected
        if callback is not None:
            try:
                callback()
            except Exception as exc:  # noqa: BLE001
                print(f"[frame_receiver] status callback failed: {exc}")

    def _sleep(self, seconds: float) -> None:
        self._stop_event.wait(seconds)