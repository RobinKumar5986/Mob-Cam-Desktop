"""Data receiver.

First stage of the pipeline. Reads the typed message stream off the adb-forwarded
TCP socket and dispatches it: settings, background image and JPEG frames.

Handshake, phone-driven:
    phone -> HELLO       settings JSON, including the USE PC flag
    phone -> BACKGROUND  the initial background image (only if one is selected)
    PC    -> ACK         processing is configured, start sending
    phone -> FRAME ...   frames, only after the ACK

Frames that arrive before the ACK are still accepted, so an older APK that never
handshakes keeps working.

Latency control: the socket is drained as fast as the phone can send, but only
the newest frame is kept. Decoding and processing happen on a separate thread,
so a frame that arrives while the previous one is still being processed simply
replaces it. Without this the kernel receive buffer becomes an unbounded queue
of stale frames and the preview drifts further behind real time the longer it
runs.
"""

from __future__ import annotations

import socket
import threading
from typing import Optional

import cv2
import numpy as np

import protocol
from protocol import (
    MessageReader, PROTOCOL_VERSION, ProtocolError, TYPE_ACK, TYPE_BACKGROUND,
    TYPE_BYE, TYPE_FRAME, TYPE_HELLO, TYPE_SETTINGS,
)
from remote_settings import RemoteSettings

DEFAULT_PORT = protocol.PORT
RECONNECT_DELAY = 1.0
CONNECT_TIMEOUT = 2.0
SLOT_POLL_TIMEOUT = 0.2


class LatestSlot:
    """Single-slot handoff where a new item replaces one not yet consumed."""

    def __init__(self):
        self._item = None
        self._condition = threading.Condition()
        self._closed = False
        self.dropped = 0

    def put(self, item) -> None:
        """Store an item, counting the replaced one as dropped."""
        with self._condition:
            if self._closed:
                return
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self._condition.notify()

    def get(self, timeout: float = SLOT_POLL_TIMEOUT):
        """Take the pending item, waiting briefly if there is none."""
        with self._condition:
            if self._item is None and not self._closed:
                self._condition.wait(timeout)
            item, self._item = self._item, None
            return item

    def clear(self) -> None:
        with self._condition:
            self._item = None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._item = None
            self._condition.notify_all()

    def reopen(self) -> None:
        with self._condition:
            self._closed = False
            self._item = None


class DataReceiver:
    """Background thread that keeps a data stream alive and dispatches messages."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        on_frame=None,
        on_hello=None,
        on_background=None,
        on_settings=None,
        on_connected=None,
        on_disconnected=None,
        ack_payload=None,
        reconnect_delay: float = RECONNECT_DELAY,
    ):
        self.host = host
        self.port = port
        self.on_frame = on_frame
        self.on_hello = on_hello
        self.on_background = on_background
        self.on_settings = on_settings
        self.on_connected = on_connected
        self.on_disconnected = on_disconnected
        self.ack_payload = ack_payload
        self.reconnect_delay = reconnect_delay

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._worker: Optional[threading.Thread] = None
        self._connected = False
        self._socket: Optional[socket.socket] = None
        self._awaiting_background = False
        self._frame_slot = LatestSlot()

        self.frames_received = 0
        self.frames_processed = 0
        self.handshake_done = False
        self.settings: Optional[RemoteSettings] = None

    @property
    def frames_dropped(self) -> int:
        """Frames discarded because a newer one arrived mid-processing."""
        return self._frame_slot.dropped

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Begin connecting, retrying until stopped."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._frame_slot.reopen()
        self._worker = threading.Thread(
            target=self._process_loop, name="frame-processor", daemon=True
        )
        self._worker.start()
        self._thread = threading.Thread(
            target=self._run, name="data-receiver", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop reading and close the connection."""
        self._stop_event.set()
        sock = self._socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._frame_slot.close()
        for thread in (self._thread, self._worker):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=timeout)
        self._thread = None
        self._worker = None

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
                self._stop_event.wait(self.reconnect_delay)
                continue

            self._socket = sock
            self._reset_session()
            self._set_connected(True)
            try:
                self._read_loop(sock)
            except ProtocolError as exc:
                print(f"[data_receiver] {exc}")
            except OSError:
                pass
            finally:
                self._socket = None
                sock.close()
                self._set_connected(False)

            self._stop_event.wait(self.reconnect_delay)

    def _reset_session(self) -> None:
        self.handshake_done = False
        self._awaiting_background = False
        self.settings = None
        self._frame_slot.clear()

    def _read_loop(self, sock: socket.socket) -> None:
        reader = MessageReader(sock)
        while not self._stop_event.is_set():
            message = reader.read()
            if message is None:
                return
            msg_type, payload = message

            if msg_type == TYPE_FRAME:
                self._handle_frame(payload)
            elif msg_type == TYPE_HELLO:
                self._handle_hello(sock, payload)
            elif msg_type == TYPE_BACKGROUND:
                self._handle_background(sock, payload)
            elif msg_type == TYPE_SETTINGS:
                self._handle_settings(payload)
            elif msg_type == TYPE_BYE:
                return

    def _handle_frame(self, payload: bytes) -> None:
        """Queue the newest frame, replacing any frame not yet processed."""
        self.frames_received += 1
        self._frame_slot.put(payload)

    def _process_loop(self) -> None:
        """Decode and dispatch the newest available frame, one at a time."""
        while not self._stop_event.is_set():
            payload = self._frame_slot.get()
            if payload is None:
                continue
            frame = cv2.imdecode(
                np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                continue
            self.frames_processed += 1
            self._safe_call(self.on_frame, frame)

    def _handle_hello(self, sock: socket.socket, payload: bytes) -> None:
        data = protocol.decode_json(payload)
        settings = RemoteSettings.from_hello(data)
        self.settings = settings
        self._safe_call(self.on_hello, settings)

        self._awaiting_background = bool(data.get("hasBackground")) and settings.use_pc
        if not self._awaiting_background:
            self._send_ack(sock)

    def _handle_background(self, sock: socket.socket, payload: bytes) -> None:
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            if self.settings is not None:
                self.settings.background_image = image
            self._safe_call(self.on_background, image)
        if self._awaiting_background:
            self._awaiting_background = False
            self._send_ack(sock)

    def _handle_settings(self, payload: bytes) -> None:
        data = protocol.decode_json(payload)
        if self.settings is not None:
            self.settings.update_from(data)
        self._safe_call(self.on_settings, data)

    def _send_ack(self, sock: socket.socket) -> None:
        """Tell the phone the desktop is configured and ready for frames."""
        payload = {"ok": True, "protocolVersion": PROTOCOL_VERSION}
        if self.ack_payload is not None:
            try:
                extra = self.ack_payload()
                if isinstance(extra, dict):
                    payload.update(extra)
            except Exception as exc:  # noqa: BLE001
                print(f"[data_receiver] ack_payload failed: {exc}")
        try:
            protocol.send_json(sock, TYPE_ACK, payload)
            self.handshake_done = True
        except OSError as exc:
            print(f"[data_receiver] failed to send ACK: {exc}")

    def _set_connected(self, connected: bool) -> None:
        if connected == self._connected:
            return
        self._connected = connected
        self._safe_call(self.on_connected if connected else self.on_disconnected)

    @staticmethod
    def _safe_call(callback, *args) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as exc:  # noqa: BLE001
            print(f"[data_receiver] callback failed: {exc}")