"""Pipeline controller.

Owns the whole chain and hides the threading from the UI:

    DataReceiver  -> ProcessingPipeline (+ AIEngine) -> VirtualCamera
                                                    \\-> preview callback
    AudioReceiver -> AudioPipeline -> AudioOutput -> virtual microphone
                                                 \\-> optional speaker monitor

The two halves mirror each other: receive, run a stage pipeline, publish as an
OS device other applications can select.

The virtual camera runs on its own sender thread behind a depth-1 queue, so a
slow driver drops frames instead of stalling the socket reader. The receiver
does the same one stage earlier, so nothing anywhere in the chain queues more
than one frame and the output stays in real time.

Audio is a second socket on its own port and never shares a thread with the
video path: a chunk queued behind a JPEG would arrive one frame time late, and
lip sync is the whole point.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from ai_processing import AIEngine
from audio_output import AudioOutput, AudioOutputError
from audio_processing import AudioConfig, AudioPipeline
from audio_receiver import AudioReceiver
from data_receiver import DEFAULT_PORT, DataReceiver
from image_processing import (
    ProcessingConfig, ProcessingPipeline, build_pipeline, even_size,
)
from protocol import AUDIO_PORT
from remote_settings import RemoteSettings
from virtual_camera import VirtualCamera, VirtualCameraError


class StreamPipeline:
    def __init__(
        self,
        port: int = DEFAULT_PORT,
        config: Optional[ProcessingConfig] = None,
        fps: int = 30,
        virtual_camera_enabled: bool = True,
        vcam_device: Optional[str] = None,
        segmenter_model: str = "selfie_landscape",
        mask_sharpness: float = 0.75,
        audio_enabled: bool = True,
        audio_port: int = AUDIO_PORT,
        audio_device: Optional[str] = None,
        audio_monitor: bool = False,
        audio_gain_db: float = 0.0,
        audio_mute: bool = False,
        on_preview_frame=None,
        on_connected=None,
        on_disconnected=None,
        on_vcam_status=None,
        on_vcam_error=None,
        on_settings_received=None,
        on_ai_status=None,
        on_audio_status=None,
        on_audio_error=None,
    ):
        self.config = config or ProcessingConfig()
        self.pipeline: ProcessingPipeline = build_pipeline(self.config)

        self.on_preview_frame = on_preview_frame
        self.on_vcam_status = on_vcam_status
        self.on_vcam_error = on_vcam_error
        self.on_settings_received = on_settings_received
        self.on_audio_status = on_audio_status
        self.on_audio_error = on_audio_error

        self.ai = AIEngine(segmenter_model=segmenter_model, on_status=on_ai_status,
                           mask_sharpness=mask_sharpness)
        self.config.ai = self.ai

        self.receiver = DataReceiver(
            port=port,
            on_frame=self._handle_frame,
            on_hello=self._handle_hello,
            on_background=self._handle_background,
            on_settings=self._handle_settings,
            on_connected=on_connected,
            on_disconnected=on_disconnected,
            ack_payload=self._build_ack,
        )

        self.camera = VirtualCamera(
            fps=fps, device=vcam_device, on_status=self._emit_vcam_status
        )
        self._vcam_enabled = virtual_camera_enabled
        self._vcam_queue: queue.Queue = queue.Queue(maxsize=1)
        self._vcam_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._vcam_failed = False
        self._pending_device_size: Optional[tuple] = None
        self.camera.pad_color = self.config.background

        self._audio_enabled = audio_enabled
        self._audio_failed = False
        self._audio_channels = 1
        self.audio_pipeline = AudioPipeline(
            AudioConfig(gain_db=audio_gain_db, mute=audio_mute))
        self.audio_output = AudioOutput(
            device=audio_device, monitor=audio_monitor,
            on_status=self._emit_audio_status, on_error=self._emit_audio_error,
        )
        self.audio_receiver = AudioReceiver(
            port=audio_port,
            on_chunk=self._handle_audio_chunk,
            on_format=self._handle_audio_format,
            on_connected=lambda: self._emit_audio_status("Phone mic connected"),
            on_disconnected=self._handle_audio_disconnected,
        )

        self.frames_dropped = 0
        self._process_seconds = 0.0
        self._processed = 0
        self._stats_mark = time.perf_counter()
        self._stats_baseline = (0, 0, 0)

    # ----------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Start the receivers and, if enabled, the virtual camera sender."""
        self._stop_event.clear()
        self._vcam_failed = False
        self._audio_failed = False
        self.camera.request_size(*even_size(self.config.output_size))
        self._process_seconds = 0.0
        self._processed = 0
        self._stats_mark = time.perf_counter()
        self._stats_baseline = (0, 0, 0)
        if self._vcam_enabled:
            self._start_vcam_thread()
        self.receiver.start()
        if self._audio_enabled:
            self.audio_receiver.start()
            self._emit_audio_status("Waiting for the phone mic...")

    def stop(self) -> None:
        """Tear everything down, releasing the devices and the AI models."""
        self._stop_event.set()
        self.receiver.stop()
        self.audio_receiver.stop()
        self.audio_output.close()
        self._drain_vcam_queue()
        thread = self._vcam_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._vcam_thread = None
        self.camera.close()
        self.ai.close()

    # -------------------------------------------------------------- config

    @property
    def virtual_camera_enabled(self) -> bool:
        return self._vcam_enabled

    @property
    def audio_enabled(self) -> bool:
        return self._audio_enabled

    @property
    def settings(self) -> RemoteSettings:
        return self.config.remote

    def set_virtual_camera_enabled(self, enabled: bool) -> None:
        """Turn the camera device output on or off without touching the stream."""
        if enabled == self._vcam_enabled:
            return
        self._vcam_enabled = enabled
        if enabled:
            self._vcam_failed = False
            self.camera.request_size(*even_size(self.config.output_size))
            self._start_vcam_thread()
        else:
            self._drain_vcam_queue()
            self.camera.close()
            self._emit_vcam_status("off")

    def set_audio_enabled(self, enabled: bool) -> None:
        """Turn the microphone path on or off without touching the video."""
        if enabled == self._audio_enabled:
            return
        self._audio_enabled = enabled
        if enabled:
            self._audio_failed = False
            self.audio_receiver.start()
            self._emit_audio_status("Waiting for the phone mic...")
        else:
            self.audio_receiver.stop()
            self.audio_output.close()
            self._emit_audio_status("off")

    def set_audio_device(self, device: Optional[str]) -> None:
        """Switch the loopback device the microphone audio is written to."""
        self._audio_failed = False
        self.audio_output.set_device(device)

    def set_audio_monitor(self, enabled: bool) -> None:
        """Play the phone audio on the local speakers as well."""
        self.audio_output.set_monitor(enabled)

    def set_shape(self, shape: str) -> None:
        self.config.shape = shape

    def set_mirror(self, mirror: bool) -> None:
        self.config.mirror = mirror

    def set_rotation(self, degrees: int) -> None:
        self.config.rotation = degrees

    def set_mask_sharpness(self, sharpness: float) -> None:
        """Live control over how hard the background-removal edge is."""
        self.ai.set_mask_sharpness(sharpness)

    def set_output_size(self, width: int, height: int) -> None:
        """Change the output resolution.

        The composition changes immediately. Reopening the camera device is queued
        for the sender thread, because the driver call can block for a moment and
        must not run on the UI thread.
        """
        size = even_size((width, height))
        self.config.output_size = size
        self.camera.pad_color = self.config.background

        if self.camera.is_open:
            self._pending_device_size = size
        else:
            self.camera.request_size(*size)

    # ------------------------------------------------------------ handshake

    def _handle_hello(self, settings: RemoteSettings) -> None:
        """Adopt the phone's settings and load whatever models they need."""
        previous = self.config.remote
        if previous is not None and settings.background_image is None:
            settings.background_image = previous.background_image
        self.config.remote = settings
        self.ai.reset()
        self.ai.configure(settings)
        self._emit_settings(settings)

    def _handle_background(self, image) -> None:
        """Store the background image the phone sent during the handshake."""
        self.config.remote.background_image = image
        self.ai.configure(self.config.remote)

    def _handle_settings(self, data: dict) -> None:
        """Apply a live settings change from the phone."""
        self.ai.configure(self.config.remote)
        self._emit_settings(self.config.remote)

    def _build_ack(self) -> dict:
        """Payload the phone receives before it starts sending frames."""
        width, height = even_size(self.config.output_size)
        settings = self.config.remote
        return {
            "mode": "pc" if settings.use_pc else "phone",
            "outputWidth": width,
            "outputHeight": height,
            "virtualCamera": self._vcam_enabled,
            "audio": self._audio_enabled,
            "ai": self.ai.status(),
            "aiError": self.ai.last_error or "",
        }

    # ----------------------------------------------------------- audio path

    def _handle_audio_format(self, audio_format) -> None:
        """Open the output device to match the format the phone announced.

        Runs on the audio reader thread, before the ACK is sent, so the device is
        already live by the time the first chunk arrives.
        """
        self._audio_channels = audio_format.channels
        self.audio_pipeline.reset()
        try:
            self.audio_output.open(audio_format)
            self._audio_failed = False
        except AudioOutputError as exc:
            self._audio_failed = True
            self._emit_audio_error(str(exc))

    def _handle_audio_chunk(self, pcm: bytes) -> None:
        """Run one chunk through the stages and hand it to the output.

        Done inline on the reader thread on purpose. The stages are a few
        microseconds on a 20 ms chunk, and a thread hop here would mean a queue,
        which is the one thing this path must not have.
        """
        if self._audio_failed:
            return
        self.audio_output.feed(
            self.audio_pipeline.process(pcm, self._audio_channels))

    def _handle_audio_disconnected(self) -> None:
        self.audio_output.close()
        self.audio_pipeline.reset()
        if self._audio_enabled:
            self._emit_audio_status("Phone mic disconnected")

    def set_audio_gain_db(self, gain_db: float) -> None:
        """Live gain trim on the microphone signal."""
        self.audio_pipeline.set_gain_db(gain_db)

    def set_audio_mute(self, mute: bool) -> None:
        """Mute at the pipeline, so the device stays open and apps see silence."""
        self.audio_pipeline.set_mute(mute)

    def audio_stats(self) -> dict:
        """Current audio state for the UI."""
        fmt = self.audio_receiver.audio_format
        level = self.audio_pipeline.level()
        return {
            "connected": self.audio_receiver.is_connected,
            "format": str(fmt) if fmt else "",
            "chunks": self.audio_receiver.chunks_received,
            "underruns": self.audio_output.underruns,
            "peak": level["peak"],
            "rms": level["rms"],
            "processing": self.audio_pipeline.config.summary(),
        }

    # ---------------------------------------------------------- frame path

    def _handle_frame(self, frame) -> None:
        """Process one decoded frame on the processing thread."""
        started = time.perf_counter()
        processed = self.pipeline.process(frame)
        self._process_seconds += time.perf_counter() - started
        self._processed += 1
        if processed is None:
            return

        if self.on_preview_frame is not None:
            try:
                self.on_preview_frame(processed)
            except Exception as exc:  # noqa: BLE001
                print(f"[stream_pipeline] preview callback failed: {exc}")

        if self._vcam_enabled and not self._vcam_failed:
            try:
                self._vcam_queue.put_nowait(processed)
            except queue.Full:
                self.frames_dropped += 1
                try:
                    self._vcam_queue.get_nowait()
                    self._vcam_queue.put_nowait(processed)
                except (queue.Empty, queue.Full):
                    pass

    def _start_vcam_thread(self) -> None:
        if self._vcam_thread is not None and self._vcam_thread.is_alive():
            return
        self._vcam_thread = threading.Thread(
            target=self._vcam_worker, name="virtual-camera", daemon=True
        )
        self._vcam_thread.start()

    def _vcam_worker(self) -> None:
        while not self._stop_event.is_set():
            pending = self._pending_device_size
            if pending is not None:
                self._pending_device_size = None
                self._apply_device_size(pending)

            try:
                frame = self._vcam_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if not self._vcam_enabled:
                continue
            try:
                self.camera.send(frame)
            except VirtualCameraError as exc:
                self._vcam_failed = True
                self._emit_vcam_error(str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                self._vcam_failed = True
                self._emit_vcam_error(f"Virtual camera send failed: {exc}")
                return

    def _apply_device_size(self, size) -> None:
        """Reopen the camera device at a new resolution, on the sender thread."""
        self._drain_vcam_queue()
        if self.camera.reopen_at(*size):
            self._emit_vcam_status(
                f"{self.camera.device_name} {size[0]}x{size[1]} - "
                "reselect the camera in your app"
            )
        else:
            current = self.camera.size
            detail = f"{current[0]}x{current[1]}" if current else "closed"
            self._emit_vcam_error(
                f"Could not switch the camera device to {size[0]}x{size[1]}.\n\n"
                f"Still running at {detail}. Close any app currently using the "
                f"camera and try again.\n\n{self.camera.last_error or ''}"
            )

    def _drain_vcam_queue(self) -> None:
        while True:
            try:
                self._vcam_queue.get_nowait()
            except queue.Empty:
                return

    def stats(self) -> dict:
        """Throughput since the previous call, for the UI to display."""
        now = time.perf_counter()
        elapsed = max(1e-6, now - self._stats_mark)

        received = self.receiver.frames_received
        dropped = self.receiver.frames_dropped
        processed = self._processed
        base_received, base_dropped, base_processed = self._stats_baseline

        frames = processed - base_processed
        result = {
            "in_fps": (received - base_received) / elapsed,
            "out_fps": frames / elapsed,
            "dropped_fps": (dropped - base_dropped) / elapsed,
            "avg_ms": (self._process_seconds / frames * 1000.0) if frames else 0.0,
            "dropped_total": dropped,
        }

        self._stats_mark = now
        self._stats_baseline = (received, dropped, processed)
        self._process_seconds = 0.0
        return result

    # -------------------------------------------------------------- events

    def _emit_vcam_status(self, message: str) -> None:
        self._safe(self.on_vcam_status, message)

    def _emit_vcam_error(self, message: str) -> None:
        self._safe(self.on_vcam_error, message)

    def _emit_audio_status(self, message: str) -> None:
        self._safe(self.on_audio_status, message)

    def _emit_audio_error(self, message: str) -> None:
        self._safe(self.on_audio_error, message)

    def _emit_settings(self, settings: RemoteSettings) -> None:
        self._safe(self.on_settings_received, settings)

    @staticmethod
    def _safe(callback, *args) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            pass