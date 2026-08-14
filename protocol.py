"""Wire protocol shared by the phone and the desktop.

Message layout:
    [1 byte type][4 bytes big-endian payload length][payload]

Legacy streams (older APKs that sent bare length-prefixed JPEGs) are still
readable: a legacy frame starts with the high byte of a 4-byte length, which is
0x00 for any frame under 16 MB, and 0x00 is not a valid message type.
"""

from __future__ import annotations

import json
import socket
import struct

PORT = 4343
PROTOCOL_VERSION = 1

TYPE_LEGACY_FRAME = 0x00
TYPE_HELLO = 0x01
TYPE_BACKGROUND = 0x02
TYPE_ACK = 0x03
TYPE_FRAME = 0x04
TYPE_SETTINGS = 0x05
TYPE_BYE = 0x06

TYPE_NAMES = {
    TYPE_LEGACY_FRAME: "LEGACY_FRAME",
    TYPE_HELLO: "HELLO",
    TYPE_BACKGROUND: "BACKGROUND",
    TYPE_ACK: "ACK",
    TYPE_FRAME: "FRAME",
    TYPE_SETTINGS: "SETTINGS",
    TYPE_BYE: "BYE",
}

MAX_PAYLOAD_BYTES = 32 * 1024 * 1024

MODE_PC = "pc"
MODE_PHONE = "phone"


class ProtocolError(Exception):
    """Raised when the stream is malformed and the connection must be dropped."""


def encode_message(msg_type: int, payload: bytes = b"") -> bytes:
    """Serialise one message."""
    return struct.pack(">BI", msg_type, len(payload)) + payload


def encode_json(msg_type: int, obj) -> bytes:
    """Serialise one message whose payload is UTF-8 JSON."""
    return encode_message(msg_type, json.dumps(obj).encode("utf-8"))


def send_message(sock: socket.socket, msg_type: int, payload: bytes = b"") -> None:
    """Write one message to a socket."""
    sock.sendall(encode_message(msg_type, payload))


def send_json(sock: socket.socket, msg_type: int, obj) -> None:
    """Write one JSON message to a socket."""
    sock.sendall(encode_json(msg_type, obj))


def decode_json(payload: bytes) -> dict:
    """Parse a JSON payload, returning {} rather than raising on junk."""
    try:
        value = json.loads(payload.decode("utf-8"))
        return value if isinstance(value, dict) else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}


class MessageReader:
    """Buffered message reader over a blocking socket."""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._buffer = bytearray()

    def _fill(self, count: int) -> bool:
        """Block until the buffer holds count bytes; False if the peer closed."""
        while len(self._buffer) < count:
            chunk = self._sock.recv(65536)
            if not chunk:
                return False
            self._buffer.extend(chunk)
        return True

    def _take(self, count: int) -> bytes:
        data = bytes(self._buffer[:count])
        del self._buffer[:count]
        return data

    def read(self):
        """Return the next (type, payload), or None if the peer closed."""
        if not self._fill(1):
            return None

        msg_type = self._buffer[0]

        if msg_type == TYPE_LEGACY_FRAME:
            if not self._fill(4):
                return None
            length = struct.unpack(">I", self._take(4))[0]
            if length == 0 or length > MAX_PAYLOAD_BYTES:
                raise ProtocolError(f"legacy frame length out of range: {length}")
            if not self._fill(length):
                return None
            return TYPE_FRAME, self._take(length)

        if msg_type not in TYPE_NAMES:
            raise ProtocolError(f"unknown message type 0x{msg_type:02X}")

        if not self._fill(5):
            return None
        _, length = struct.unpack(">BI", self._take(5))
        if length > MAX_PAYLOAD_BYTES:
            raise ProtocolError(f"payload length out of range: {length}")
        if length and not self._fill(length):
            return None
        return msg_type, self._take(length)