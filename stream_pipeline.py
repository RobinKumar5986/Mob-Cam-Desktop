"""Pipeline controller.

Owns the whole chain and hides the threading from the UI:

    FrameReceiver -> ProcessingPipeline -> VirtualCamera
                                       \\-> preview callback

The virtual camera runs on its own sender thread behind a depth-1 queue. A slow
or stalled driver then drops frames instead of blocking the socket reader, which
is what would otherwise make the incoming stream lag and back up.
"""

from __future__ import annotations

import queue
import threading

from frame_receiver import DEFAULT_PORT, FrameReceiver
from image_processing import ProcessingConfig, ProcessingPipeline, build_pipeline
from virtual_camera import VirtualCamera, VirtualCameraError


class StreamPipeline:
    def __init__(
        self,
        port: int = DEFAULT_PORT,
        config: ProcessingConfig | None = None,
        fps: int = 30,
        virtual_camera_enabled: bool = True,
        vcam_device: str | None = None,
        on_preview_frame=None,
        on_connected=None,
        on_disconnected=None,
        on_vcam_status=None,
        on_vcam_error=None,
    ):
        self.config = config or ProcessingConfig()
        self.pipeline: ProcessingPipeline = build_pipeline(self.config)

        self.on_preview_frame = on_preview_frame
        self.on_vcam_status = on_vcam_status
        self.on_vcam_error = on_vcam_error

        self.receiver = FrameReceiver(
            port=port,
            on_frame=self._handle_frame,
            on_connected=on_connected,
            on_disconnected=on_disconnected,
        )

        self.camera = VirtualCamera(
            fps=fps, device=vcam_device, on_status=self._emit_vcam_status
        )
        self._vcam_enabled = virtual_camera_enabled
        self._vcam_queue: queue.Queue = queue.Queue(maxsize=1)
        self._vcam_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._vcam_failed = False

        self.frames_dropped = 0

    # ----------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._stop_event.clear()
        self._vcam_failed = False
        if self._vcam_enabled:
            self._start_vcam_thread()
        self.receiver.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.receiver.stop()
        self._drain_vcam_queue()
        thread = self._vcam_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._vcam_thread = None
        self.camera.close()

    # -------------------------------------------------------------- config

    @property
    def virtual_camera_enabled(self) -> bool:
        return self._vcam_enabled

    def set_virtual_camera_enabled(self, enabled: bool) -> None:
        if enabled == self._vcam_enabled:
            return
        self._vcam_enabled = enabled
        if enabled:
            self._vcam_failed = False
            self._start_vcam_thread()
        else:
            self._drain_vcam_queue()
            self.camera.close()
            self._emit_vcam_status("Virtual camera off")

    def set_shape(self, shape: str) -> None:
        self.config.shape = shape

    def set_mirror(self, mirror: bool) -> None:
        self.config.mirror = mirror

    def set_rotation(self, degrees: int) -> None:
        self.config.rotation = degrees

    def set_output_size(self, width: int, height: int) -> None:
        # VirtualCamera notices the change on the next frame and reopens itself.
        self.config.output_size = (width, height)

    # ---------------------------------------------------------- frame path

    def _handle_frame(self, frame) -> None:
        """Runs on the receiver thread."""
        processed = self.pipeline.process(frame)
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
                # Newest frame wins: throw away the stale one and retry once.
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
                # No driver / driver refused: report once and stop trying so we
                # do not spam the UI on every frame.
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

    # -------------------------------------------------------------- events

    def _emit_vcam_status(self, message: str) -> None:
        if self.on_vcam_status is not None:
            try:
                self.on_vcam_status(message)
            except Exception:
                pass

    def _emit_vcam_error(self, message: str) -> None:
        if self.on_vcam_error is not None:
            try:
                self.on_vcam_error(message)
            except Exception:
                pass