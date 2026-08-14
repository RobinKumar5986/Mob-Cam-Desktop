"""Pipeline controller.

Owns the whole chain and hides the threading from the UI:

    DataReceiver -> ProcessingPipeline (+ AIEngine) -> VirtualCamera
                                                   \\-> preview callback

The virtual camera runs on its own sender thread behind a depth-1 queue, so a
slow driver drops frames instead of stalling the socket reader. The receiver
does the same one stage earlier, so nothing anywhere in the chain queues more
than one frame and the output stays in real time.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from ai_processing import AIEngine
from data_receiver import DEFAULT_PORT, DataReceiver
from image_processing import (
    ProcessingConfig, ProcessingPipeline, build_pipeline, even_size,
)
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
        on_preview_frame=None,
        on_connected=None,
        on_disconnected=None,
        on_vcam_status=None,
        on_vcam_error=None,
        on_settings_received=None,
        on_ai_status=None,
    ):
        self.config = config or ProcessingConfig()
        self.pipeline: ProcessingPipeline = build_pipeline(self.config)

        self.on_preview_frame = on_preview_frame
        self.on_vcam_status = on_vcam_status
        self.on_vcam_error = on_vcam_error
        self.on_settings_received = on_settings_received

        self.ai = AIEngine(segmenter_model=segmenter_model, on_status=on_ai_status)
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

        self.frames_dropped = 0
        self._process_seconds = 0.0
        self._processed = 0
        self._stats_mark = time.perf_counter()
        self._stats_baseline = (0, 0, 0)

    # ----------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Start the receiver and, if enabled, the virtual camera sender."""
        self._stop_event.clear()
        self._vcam_failed = False
        self.camera.request_size(*even_size(self.config.output_size))
        self._process_seconds = 0.0
        self._processed = 0
        self._stats_mark = time.perf_counter()
        self._stats_baseline = (0, 0, 0)
        if self._vcam_enabled:
            self._start_vcam_thread()
        self.receiver.start()

    def stop(self) -> None:
        """Tear everything down, releasing the device and the AI models."""
        self._stop_event.set()
        self.receiver.stop()
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

    def set_shape(self, shape: str) -> None:
        self.config.shape = shape

    def set_mirror(self, mirror: bool) -> None:
        self.config.mirror = mirror

    def set_rotation(self, degrees: int) -> None:
        self.config.rotation = degrees

    def set_output_size(self, width: int, height: int) -> None:
        """Change the output resolution; the device reopens on the next frame."""
        size = even_size((width, height))
        self.config.output_size = size
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
            "ai": self.ai.status(),
            "aiError": self.ai.last_error or "",
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