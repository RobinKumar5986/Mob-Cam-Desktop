"""Audio stage pipeline.

The audio equivalent of image_processing: a fixed list of stages over int16 PCM
sitting between the receiver and the virtual microphone.

    audio_receiver -> AudioPipeline -> AudioOutput -> virtual mic -> your app

Stage order: mute, dc_block, gain, limiter, meter.

Same rules as the video pipeline. A stage that fails is skipped rather than
taken as fatal, and nothing here buffers: one chunk in, one chunk out, so
adding a stage can cost CPU but never latency. Chunks are 20 ms, which is the
whole budget - anything needing a longer window (noise gates with lookahead,
FFT denoise) has to carry its own state across calls the way DcBlock does,
not stall waiting for more samples.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

MIN_GAIN_DB = -12
MAX_GAIN_DB = 24

# Above this the limiter starts pulling peaks down, leaving a little headroom
# below full scale so the conversion in audio_output never wraps.
LIMIT_CEILING = 0.97 * 32767.0


class AudioConfig:
    """Live-tunable settings for the audio stages."""

    def __init__(self, gain_db: float = 0.0, mute: bool = False,
                 dc_block: bool = True, limiter: bool = True):
        self.gain_db = float(gain_db)
        self.mute = bool(mute)
        self.dc_block = bool(dc_block)
        self.limiter = bool(limiter)

    @property
    def gain(self) -> float:
        """Linear gain for the configured decibel value."""
        return float(10.0 ** (self.gain_db / 20.0))

    def summary(self) -> str:
        if self.mute:
            return "muted"
        parts = []
        if abs(self.gain_db) >= 0.5:
            parts.append(f"{self.gain_db:+.0f} dB")
        if self.dc_block:
            parts.append("dc-block")
        return ", ".join(parts) or "flat"


class _DcBlock:
    """One-pole high-pass that removes the DC offset some phone mics carry.

    Left in permanently because a constant offset eats headroom before the
    limiter and is inaudible, so it is never obvious that it is the cause.
    State is per channel and survives across chunks; resetting it mid-stream
    would put a step back into the signal.
    """

    # ~20 Hz corner at 16 kHz, and lower at higher rates, which is fine: the
    # point is to remove DC, not to shape the bottom end.
    COEFFICIENT = 0.995

    def __init__(self):
        self._last_in: Optional[np.ndarray] = None
        self._last_out: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._last_in = None
        self._last_out = None

    def process(self, frames: np.ndarray) -> np.ndarray:
        channels = frames.shape[1]
        if self._last_in is None or self._last_in.shape[0] != channels:
            self._last_in = np.zeros(channels, dtype=np.float32)
            self._last_out = np.zeros(channels, dtype=np.float32)

        out = np.empty_like(frames)
        last_in = self._last_in
        last_out = self._last_out
        coefficient = self.COEFFICIENT

        # A vectorised form exists but needs a full cumulative product per
        # chunk; at 320 frames the loop is cheaper and far easier to follow.
        for index in range(frames.shape[0]):
            current = frames[index]
            last_out = coefficient * (last_out + current - last_in)
            last_in = current
            out[index] = last_out

        self._last_in = last_in
        self._last_out = last_out
        return out


class AudioPipeline:
    """Runs the stages over one chunk at a time and tracks the output level."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._dc = _DcBlock()
        self.peak = 0.0
        self.rms = 0.0
        self.clipped_chunks = 0

    def reset(self) -> None:
        """Drop any carried state, for a fresh connection."""
        self._dc.reset()
        self.peak = 0.0
        self.rms = 0.0

    def process(self, pcm: bytes, channels: int = 1) -> bytes:
        """Run every stage over one chunk of interleaved int16 PCM."""
        channels = max(1, int(channels))
        if not pcm:
            return pcm

        if self.config.mute:
            self.peak = 0.0
            self.rms = 0.0
            return b"\x00" * len(pcm)

        try:
            samples = np.frombuffer(pcm, dtype="<i2")
            usable = (samples.size // channels) * channels
            if usable == 0:
                return pcm
            frames = samples[:usable].reshape(-1, channels).astype(np.float32)
        except (ValueError, TypeError) as exc:
            print(f"[audio_processing] could not decode chunk: {exc}")
            return pcm

        frames = self._stage(self._dc_stage, frames, "dc_block")
        frames = self._stage(self._gain_stage, frames, "gain")
        frames = self._stage(self._limiter_stage, frames, "limiter")
        self._stage(self._meter_stage, frames, "meter")

        return np.clip(frames, -32768, 32767).astype("<i2").tobytes()

    def _stage(self, function, frames: np.ndarray, name: str) -> np.ndarray:
        """Run one stage, skipping it if it raises rather than dropping audio."""
        try:
            result = function(frames)
        except Exception as exc:  # noqa: BLE001
            print(f"[audio_processing] stage '{name}' failed, skipped: {exc}")
            return frames
        return frames if result is None else result

    # ------------------------------------------------------------- stages

    def _dc_stage(self, frames):
        if not self.config.dc_block:
            return frames
        return self._dc.process(frames)

    def _gain_stage(self, frames):
        gain = self.config.gain
        if abs(gain - 1.0) < 1e-3:
            return frames
        return frames * gain

    def _limiter_stage(self, frames):
        """Scales a chunk down whole rather than clipping individual samples.

        Per-sample clipping is what makes boosted speech sound gritty; pulling
        the whole 20 ms window down by one factor is inaudible by comparison.
        """
        if not self.config.limiter:
            return frames
        peak = float(np.max(np.abs(frames))) if frames.size else 0.0
        if peak <= LIMIT_CEILING:
            return frames
        self.clipped_chunks += 1
        return frames * (LIMIT_CEILING / peak)

    def _meter_stage(self, frames):
        """Records the output level for the UI meter. Never changes the audio."""
        if not frames.size:
            return frames
        self.peak = float(np.max(np.abs(frames))) / 32767.0
        self.rms = float(np.sqrt(np.mean(np.square(frames)))) / 32767.0
        return frames

    # -------------------------------------------------------------- config

    def set_gain_db(self, gain_db: float) -> None:
        self.config.gain_db = float(
            max(MIN_GAIN_DB, min(MAX_GAIN_DB, gain_db)))

    def set_mute(self, mute: bool) -> None:
        self.config.mute = bool(mute)
        if mute:
            self._dc.reset()

    def level(self) -> dict:
        """Current output level, 0.0 to 1.0, for the meter."""
        return {"peak": self.peak, "rms": self.rms}