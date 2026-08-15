"""Virtual microphone output.

pyvirtualcam publishes video only, so the audio half is done the same way any
other app does it: the PCM is written to a loopback audio device whose input
side the operating system already exposes as a recording device. Zoom, Meet,
Teams, OBS and browsers then list it in their microphone picker.

    VB-CABLE      Windows
    BlackHole     macOS
    snd-aloop / module-null-sink   Linux

An optional monitor stream plays the same audio on the normal speakers, so the
sound coming off the phone can be checked without joining a call.

Rate negotiation: the phone picks the capture format, but a real sound card
will refuse most of them - ALSA hardware devices typically accept only 44.1 or
48 kHz, so a 16 kHz stream fails outright with PaErrorCode -9997. Rather than
surface that, the device is opened at a rate it does accept and the chunks are
resampled on the way in. Mono is expanded to the device's channel count the
same way.

Buffering is deliberately tiny. Each stream holds at most MAX_BUFFER_BLOCKS
blocks; when the phone runs ahead the oldest bytes are dropped rather than
queued, which trades an occasional click for staying in sync with the video.
An underrun is filled with silence instead of stalling the device.
"""

from __future__ import annotations

import contextlib
import os
import threading
from typing import List, Optional, Tuple

import numpy as np

import virtual_mic

try:
    import sounddevice as sd
except Exception as _import_error:  # noqa: BLE001
    sd = None
    _SD_ERROR = str(_import_error)
else:
    _SD_ERROR = ""

# 20 ms blocks in, so four blocks is 80 ms of slack: enough to ride out a
# scheduling hiccup, short enough that nobody hears the lag.
MAX_BUFFER_BLOCKS = 4
BLOCK_MS = 20

# Rates worth trying when the phone's own rate is refused. 48 kHz first: it is
# what nearly every modern device runs natively.
FALLBACK_RATES = (48000, 44100, 32000, 16000, 8000)

# Substrings that identify a loopback device across the three platforms.
LOOPBACK_HINTS = (
    "cable input", "vb-audio", "vb audio", "cable", "blackhole", "soundflower",
    "loopback", "virtual audio", "null", "mob cam", "mobcam", "aloop",
)


class AudioOutputError(Exception):
    """Raised when the output device cannot be opened."""


class _Ring:
    """Byte ring that drops the oldest data instead of growing without bound."""

    def __init__(self, capacity: int):
        self._capacity = max(1, capacity)
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self.dropped_bytes = 0

    def write(self, data: bytes) -> None:
        with self._lock:
            self._buffer.extend(data)
            excess = len(self._buffer) - self._capacity
            if excess > 0:
                del self._buffer[:excess]
                self.dropped_bytes += excess

    def read(self, count: int) -> Tuple[bytes, int]:
        """Take count bytes, zero-padding and reporting the shortfall."""
        with self._lock:
            available = min(count, len(self._buffer))
            data = bytes(self._buffer[:available])
            del self._buffer[:available]
        missing = count - available
        if missing:
            data += b"\x00" * missing
        return data, missing

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()


class _Converter:
    """Rate and channel conversion for one output stream.

    Linear interpolation, carrying the phase and the last input frame across
    chunk boundaries so the seam between two 20 ms chunks does not click.
    Quality is adequate for speech and costs almost nothing; a polyphase filter
    would be better and would also add latency, which is the one thing this
    path cannot spend.
    """

    def __init__(self, src_rate: int, src_channels: int,
                 dst_rate: int, dst_channels: int):
        self.src_rate = int(src_rate)
        self.src_channels = max(1, int(src_channels))
        self.dst_rate = int(dst_rate)
        self.dst_channels = max(1, int(dst_channels))
        self.ratio = self.src_rate / float(self.dst_rate)
        self.passthrough = (
            self.src_rate == self.dst_rate
            and self.src_channels == self.dst_channels
        )
        self._tail: Optional[np.ndarray] = None
        self._phase = 0.0

    def __str__(self) -> str:
        if self.passthrough:
            return "no conversion"
        return (f"{self.src_rate / 1000:g}k/{self.src_channels}ch -> "
                f"{self.dst_rate / 1000:g}k/{self.dst_channels}ch")

    def process(self, pcm: bytes) -> bytes:
        """Convert one chunk of interleaved int16 PCM."""
        if self.passthrough:
            return pcm

        frames = np.frombuffer(pcm, dtype="<i2")
        usable = (frames.size // self.src_channels) * self.src_channels
        if usable == 0:
            return b""
        frames = frames[:usable].reshape(-1, self.src_channels).astype(np.float32)

        frames = self._resample(frames)
        if frames.size == 0:
            return b""
        frames = self._map_channels(frames)
        return np.clip(frames, -32768, 32767).astype("<i2").tobytes()

    def _resample(self, frames: np.ndarray) -> np.ndarray:
        if self.src_rate == self.dst_rate:
            return frames

        # Prepending the previous chunk's last frame makes the first output
        # sample interpolate across the seam instead of restarting from it.
        tail = self._tail if self._tail is not None else frames[:1]
        buffer = np.concatenate((tail, frames), axis=0)
        self._tail = frames[-1:].copy()

        span = len(buffer) - 1
        if span <= 0:
            return np.empty((0, self.src_channels), dtype=np.float32)

        count = int(np.ceil((span - self._phase) / self.ratio))
        if count <= 0:
            self._phase -= span
            return np.empty((0, self.src_channels), dtype=np.float32)

        positions = self._phase + np.arange(count, dtype=np.float64) * self.ratio
        positions = positions[positions <= span]
        if positions.size == 0:
            self._phase -= span
            return np.empty((0, self.src_channels), dtype=np.float32)

        # Carry the leftover fraction into the next chunk.
        self._phase = positions[-1] + self.ratio - span

        left = np.floor(positions).astype(np.int64)
        right = np.minimum(left + 1, span)
        weight = (positions - left).astype(np.float32)[:, None]
        return buffer[left] * (1.0 - weight) + buffer[right] * weight

    def _map_channels(self, frames: np.ndarray) -> np.ndarray:
        src, dst = self.src_channels, self.dst_channels
        if src == dst:
            return frames
        if src == 1:
            return np.repeat(frames, dst, axis=1)
        if dst == 1:
            return frames.mean(axis=1, keepdims=True)
        if dst < src:
            return frames[:, :dst]
        pad = np.repeat(frames[:, -1:], dst - src, axis=1)
        return np.concatenate((frames, pad), axis=1)


class _Sink:
    """One open output stream plus the ring and converter feeding it."""

    def __init__(self, stream, ring: _Ring, converter: _Converter, label: str):
        self.stream = stream
        self.ring = ring
        self.converter = converter
        self.label = label

    def write(self, pcm: bytes) -> None:
        self.ring.write(self.converter.process(pcm))

    def close(self) -> None:
        try:
            self.stream.stop()
            self.stream.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[audio_output] failed to close stream: {exc}")


class AudioOutput:
    """Feeds PCM to a loopback device, and optionally to the speakers."""

    def __init__(self, device: Optional[str] = None, monitor: bool = False,
                 on_status=None, on_error=None):
        self.device = device or ""
        self.monitor = monitor
        self.on_status = on_status
        self.on_error = on_error

        self._lock = threading.Lock()
        self._sink: Optional[_Sink] = None
        self._monitor_sink: Optional[_Sink] = None
        self._format = None
        self.underruns = 0
        self.last_error = ""

    # ------------------------------------------------------------ lifecycle

    @property
    def is_open(self) -> bool:
        return self._sink is not None or self._monitor_sink is not None

    def open(self, audio_format) -> None:
        """Open the devices for a format, reopening if the format changed."""
        if sd is None:
            raise AudioOutputError(
                "sounddevice is not installed.\n\npip install sounddevice\n\n"
                f"Import error: {_SD_ERROR}")

        with self._lock:
            if self.is_open and self._format == audio_format:
                return
            self._close_locked()
            self._format = audio_format

            if self.device:
                self._sink = self._build_sink(self.device, audio_format)
                self._emit_status(self._describe(self._sink))

            if self.monitor:
                try:
                    self._monitor_sink = self._build_sink(None, audio_format)
                except AudioOutputError as exc:
                    # Monitoring is a convenience; never let it take the mic down.
                    self._monitor_sink = None
                    self._emit_error(f"Monitor unavailable: {exc}")

            if self._sink is None and self._monitor_sink is None:
                raise AudioOutputError(
                    "No microphone output selected. Pick a loopback device, or "
                    "enable Monitor to listen locally.")

    def _build_sink(self, device, audio_format) -> _Sink:
        """Open the first format the device accepts, converting if needed."""
        errors = []
        for rate, channels in _candidate_formats(device, audio_format):
            block_frames = max(1, rate * BLOCK_MS // 1000)
            capacity = block_frames * channels * 2 * MAX_BUFFER_BLOCKS
            ring = _Ring(capacity)
            converter = _Converter(
                audio_format.sample_rate, audio_format.channels, rate, channels)
            try:
                stream = self._open_stream(device, rate, channels, block_frames, ring)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"  {rate} Hz / {channels} ch: {exc}")
                continue
            return _Sink(stream, ring, converter, f"{rate} Hz / {channels} ch")

        detail = "\n".join(errors) or "  no candidate formats"
        self.last_error = detail
        raise AudioOutputError(
            f"Could not open '{describe_device(device)}'.\n\n"
            f"Every format was refused:\n{detail}\n\n"
            "If this is your sound card rather than a loopback device, pick a "
            "loopback device instead - click Setup help."
        )

    def _open_stream(self, device, rate, channels, block_frames, ring):
        """Open one raw output stream that pulls from the given ring."""
        frame_bytes = channels * 2

        def callback(outdata, frames, _time, status):
            if status:
                self.underruns += 1
            wanted = frames * frame_bytes
            data, missing = ring.read(wanted)
            if missing:
                self.underruns += 1
            outdata[:wanted] = data

        with _pulse_sink(_sink_for(device)):
            stream = sd.RawOutputStream(
                samplerate=rate,
                channels=channels,
                dtype="int16",
                blocksize=block_frames,
                latency="low",
                device=resolve_device(device) if device else None,
                callback=callback,
            )
            stream.start()
        return stream

    def _describe(self, sink: Optional[_Sink]) -> str:
        if sink is None:
            return "no output"
        text = f"{describe_device(self.device)} - {sink.label}"
        if not sink.converter.passthrough:
            text += f" (resampling {sink.converter})"
        return text

    def feed(self, pcm: bytes) -> None:
        """Hand one chunk to every open stream. Never blocks."""
        sink = self._sink
        if sink is not None:
            sink.write(pcm)
        monitor = self._monitor_sink
        if monitor is not None:
            monitor.write(pcm)

    def set_monitor(self, enabled: bool) -> None:
        """Turn speaker monitoring on or off without dropping the mic stream."""
        if enabled == self.monitor:
            return
        self.monitor = enabled
        audio_format = self._format
        if audio_format is None:
            return
        with self._lock:
            if not enabled:
                if self._monitor_sink is not None:
                    self._monitor_sink.close()
                self._monitor_sink = None
                return
            try:
                self._monitor_sink = self._build_sink(None, audio_format)
            except AudioOutputError as exc:
                self._monitor_sink = None
                self._emit_error(f"Monitor unavailable: {exc}")

    def set_device(self, device: Optional[str]) -> None:
        """Switch the loopback device, reopening if a stream is already live."""
        device = device or ""
        if device == self.device:
            return
        self.device = device
        audio_format = self._format
        if audio_format is None:
            return
        with self._lock:
            if self._sink is not None:
                self._sink.close()
            self._sink = None
            if not device:
                return
            try:
                self._sink = self._build_sink(device, audio_format)
                self._emit_status(self._describe(self._sink))
            except AudioOutputError as exc:
                self._sink = None
                self._emit_error(str(exc))

    def close(self) -> None:
        """Stop and release every stream."""
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        for sink in (self._sink, self._monitor_sink):
            if sink is not None:
                sink.close()
        self._sink = None
        self._monitor_sink = None
        self._format = None

    def _emit_status(self, message: str) -> None:
        self._safe(self.on_status, message)

    def _emit_error(self, message: str) -> None:
        self.last_error = message
        self._safe(self.on_error, message)

    @staticmethod
    def _safe(callback, *args) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------- device list

def output_devices() -> List[str]:
    """Names of every device that can play audio."""
    if sd is None:
        return []
    try:
        devices = sd.query_devices()
    except Exception as exc:  # noqa: BLE001
        print(f"[audio_output] could not list devices: {exc}")
        return []

    names = []
    for device in devices:
        if device.get("max_output_channels", 0) > 0:
            name = device.get("name", "")
            if name and name not in names:
                names.append(name)
    return names


def resolve_device(name: str):
    """Map a device name back to an index, falling back to the name itself."""
    if sd is None or not name:
        return None
    try:
        for index, device in enumerate(sd.query_devices()):
            if device.get("name") == name and device.get("max_output_channels", 0) > 0:
                return index
    except Exception:  # noqa: BLE001
        pass
    return name


def device_defaults(device) -> Tuple[int, int]:
    """Native sample rate and channel count of a device."""
    if sd is None:
        return 48000, 2
    try:
        info = sd.query_devices(
            resolve_device(device) if device else None, "output")
        return int(info["default_samplerate"]), int(info["max_output_channels"])
    except Exception:  # noqa: BLE001
        return 48000, 2


def _candidate_formats(device, audio_format) -> List[Tuple[int, int]]:
    """Formats to try, best first: the phone's own, then what the device likes."""
    native_rate, max_channels = device_defaults(device)
    max_channels = max(1, max_channels)
    wanted = max(1, min(audio_format.channels, max_channels))
    stereo = min(2, max_channels)

    candidates: List[Tuple[int, int]] = []

    def add(rate, channels):
        entry = (int(rate), max(1, min(int(channels), max_channels)))
        if entry[0] > 0 and entry not in candidates:
            candidates.append(entry)

    add(audio_format.sample_rate, wanted)
    add(native_rate, wanted)
    add(native_rate, stereo)
    for rate in FALLBACK_RATES:
        add(rate, wanted)
        add(rate, stereo)
    return candidates


def describe_device(name) -> str:
    return name or "system default output"


def _is_pulse_device(device) -> bool:
    """True when a device name routes through PulseAudio / PipeWire."""
    if not device:
        return False
    lowered = str(device).lower()
    return lowered.startswith("pulse") or lowered.startswith("default")


def _sink_for(device) -> str:
    """Which sound-server sink a stream on this device should land on.

    PortAudio exposes PulseAudio as a single 'pulse' device that follows the
    system default sink, which is the speakers - the reason audio came out loud
    instead of into the microphone. PULSE_SINK overrides that per stream, so the
    mic path lands on the Mob Cam sink while a monitor stream opened on the same
    'pulse' device still goes to the speakers.
    """
    if not _is_pulse_device(device):
        return ""
    return virtual_mic.sink_target()


@contextlib.contextmanager
def _pulse_sink(sink: str):
    """Point the next stream at a specific sink, then restore the environment."""
    if not sink:
        yield
        return
    previous = os.environ.get("PULSE_SINK")
    os.environ["PULSE_SINK"] = sink
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PULSE_SINK", None)
        else:
            os.environ["PULSE_SINK"] = previous


def guess_loopback_device() -> str:
    """The device to preselect: the Mob Cam mic if present, else any loopback."""
    names = output_devices()
    if not names:
        return ""

    # With the Mob Cam sink loaded, the right target is the sound server itself;
    # _sink_for then steers the stream onto our sink rather than the speakers.
    if virtual_mic.is_installed():
        for prefix in ("pulse", "default"):
            for name in names:
                if name.lower().startswith(prefix):
                    return name

    for name in names:
        lowered = name.lower()
        if any(hint in lowered for hint in LOOPBACK_HINTS):
            return name
    return ""


def is_available() -> Tuple[bool, str]:
    """Whether a virtual microphone exists, with a message for the UI."""
    if sd is None:
        return False, f"sounddevice not installed ({_SD_ERROR})"
    if virtual_mic.is_installed():
        return True, f"'{virtual_mic.FRIENDLY_NAME}' ready"
    device = guess_loopback_device()
    if device:
        return True, f"loopback device found: {device}"
    return virtual_mic.probe()


def setup_instructions() -> str:
    """Per-OS instructions, plus the rate note that applies everywhere."""
    return (
        virtual_mic.setup_instructions()
        + "\n\nSample rates: a real sound card usually accepts only 44.1 or\n"
          "48 kHz. Mob Cam resamples automatically, so the phone can stay on\n"
          "16 kHz whatever the device wants."
    )