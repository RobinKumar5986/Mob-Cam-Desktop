"""Audio receiver.

Mirror of data_receiver for the microphone stream. Connects to the phone's
second adb-forwarded port, reads the format announcement, acknowledges it and
then hands raw PCM chunks straight to a sink.

Handshake, phone-driven:
    phone -> HELLO   {sampleRate, channels, encoding, chunkBytes}
    PC    -> ACK     output device is open, start sending
    phone -> AUDIO   20 ms PCM chunks

Nothing is buffered here. A chunk goes to the sink the moment it is read; the
sink owns the only buffer in the chain and keeps it deliberately short, because
audio that arrives late is worse than audio that never arrives.
"""

from __future__ import annotations

import socket
import threading
from typing import Optional

import protocol
from protocol import (
    AUDIO_PORT, MessageReader, PROTOCOL_VERSION, ProtocolError, TYPE_ACK,
    TYPE_AUDIO, TYPE_BYE, TYPE_HELLO,
)

RECONNECT_DELAY = 1.0
CONNECT_TIMEOUT = 2.0


class AudioFormat:
    """Capture format announced by the phone."""

    def __init__(self, sample_rate=16000, channels=1, chunk_bytes=640, name=""):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.chunk_bytes = int(chunk_bytes)
        self.name = str(name)

    @classmethod
    def from_hello(cls, data: dict) -> "AudioFormat":
        """Build a format from a decoded HELLO payload, clamping absurd values."""
        rate = _as_int(data.get("sampleRate"), 16000)
        channels = _as_int(data.get("channels"), 1)
        rate = rate if 8000 <= rate <= 192000 else 16000
        channels = channels if channels in (1, 2) else 1
        default_chunk = (rate // 50) * channels * 2
        return cls(
            sample_rate=rate,
            channels=channels,
            chunk_bytes=_as_int(data.get("chunkBytes"), default_chunk),
            name=str(data.get("formatName", "")),
        )

    @property
    def bytes_per_frame(self) -> int:
        return self.channels * 2

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, AudioFormat)
            and other.sample_rate == self.sample_rate
            and other.channels == self.channels
        )

    def __str__(self) -> str:
        layout = "stereo" if self.channels == 2 else "mono"
        return f"{self.sample_rate / 1000:g} kHz {layout} PCM16"


class AudioReceiver:
    """Background thread that keeps the audio stream alive and feeds a sink."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = AUDIO_PORT,
        on_chunk=None,
        on_format=None,
        on_connected=None,
        on_disconnected=None,
        reconnect_delay: float = RECONNECT_DELAY,
    ):
        self.host = host
        self.port = port
        self.on_chunk = on_chunk
        self.on_format = on_format
        self.on_connected = on_connected
        self.on_disconnected = on_disconnected
        self.reconnect_delay = reconnect_delay

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._socket: Optional[socket.socket] = None
        self._connected = False

        self.audio_format: Optional[AudioFormat] = None
        self.chunks_received = 0
        self.bytes_received = 0

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
        self._thread = threading.Thread(
            target=self._run, name="audio-receiver", daemon=True
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
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._thread = None

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
            self.audio_format = None
            self._set_connected(True)
            try:
                self._read_loop(sock)
            except ProtocolError as exc:
                print(f"[audio_receiver] {exc}")
            except OSError:
                pass
            finally:
                self._socket = None
                sock.close()
                self._set_connected(False)

            self._stop_event.wait(self.reconnect_delay)

    def _read_loop(self, sock: socket.socket) -> None:
        reader = MessageReader(sock)
        while not self._stop_event.is_set():
            message = reader.read()
            if message is None:
                return
            msg_type, payload = message

            if msg_type == TYPE_AUDIO:
                self._handle_chunk(payload)
            elif msg_type == TYPE_HELLO:
                self._handle_hello(sock, payload)
            elif msg_type == TYPE_BYE:
                return

    def _handle_hello(self, sock: socket.socket, payload: bytes) -> None:
        fmt = AudioFormat.from_hello(protocol.decode_json(payload))
        self.audio_format = fmt
        self._safe_call(self.on_format, fmt)
        try:
            protocol.send_json(sock, TYPE_ACK, {
                "ok": True, "protocolVersion": PROTOCOL_VERSION,
                "sampleRate": fmt.sample_rate, "channels": fmt.channels,
            })
        except OSError as exc:
            print(f"[audio_receiver] failed to send ACK: {exc}")

    def _handle_chunk(self, payload: bytes) -> None:
        if not payload:
            return
        self.chunks_received += 1
        self.bytes_received += len(payload)
        self._safe_call(self.on_chunk, payload)

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
            print(f"[audio_receiver] callback failed: {exc}")


def _as_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback
