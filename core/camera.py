"""
SmartAttend camera module.

The camera runs in a dedicated capture thread and always keeps only the latest
frame in memory. This avoids growing buffers and keeps the Raspberry Pi feed
responsive even when recognition is slower than capture.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("smartattend.camera")

DEFAULT_RESOLUTION = (
    int(os.environ.get("CAM_WIDTH", "640")),
    int(os.environ.get("CAM_HEIGHT", "480")),
)
DEFAULT_FPS = int(os.environ.get("CAM_FPS", "20"))
WARMUP_FRAMES = int(os.environ.get("CAM_WARMUP_FRAMES", "5"))
DEFAULT_STREAM_QUALITY = int(os.environ.get("STREAM_JPEG_QUALITY", "70"))
CAM_BRIGHTNESS = float(os.environ.get("CAM_BRIGHTNESS", "0.0"))
CAM_CONTRAST = float(os.environ.get("CAM_CONTRAST", "1.0"))
CAM_SATURATION = float(os.environ.get("CAM_SATURATION", "1.0"))
CAM_SHARPNESS = float(os.environ.get("CAM_SHARPNESS", "1.0"))
CAM_AWB_ENABLE = os.environ.get("CAM_AWB_ENABLE", "true").strip().lower() == "true"


class PiCamera2Backend:
    """Camera backend for Raspberry Pi CSI cameras."""

    def __init__(self, resolution=DEFAULT_RESOLUTION, fps=DEFAULT_FPS):
        self.resolution = resolution
        self.fps = fps
        self._cam = None

    def start(self):
        from picamera2 import Picamera2

        self._cam = Picamera2()
        try:
            config = self._cam.create_preview_configuration(
                main={"size": self.resolution, "format": "BGR888"},
                buffer_count=2,
                queue=False,
            )
        except TypeError:
            config = self._cam.create_preview_configuration(
                main={"size": self.resolution, "format": "BGR888"},
            )
        self._cam.configure(config)

        frame_us = int(1_000_000 / max(self.fps, 1))
        try:
            self._cam.set_controls({"FrameDurationLimits": (frame_us, frame_us)})
        except Exception:
            logger.debug("PiCamera2 did not accept FrameDurationLimits control", exc_info=True)

        color_controls = {
            "AwbEnable": CAM_AWB_ENABLE,
            "Brightness": CAM_BRIGHTNESS,
            "Contrast": CAM_CONTRAST,
            "Saturation": CAM_SATURATION,
            "Sharpness": CAM_SHARPNESS,
        }
        try:
            self._cam.set_controls(color_controls)
        except Exception:
            logger.debug("PiCamera2 did not accept color controls", exc_info=True)

        self._cam.start()
        for _ in range(WARMUP_FRAMES):
            self._cam.capture_array()
        logger.info(
            "PiCamera2 started at %s @ %sfps (brightness=%s contrast=%s saturation=%s sharpness=%s awb=%s)",
            self.resolution,
            self.fps,
            CAM_BRIGHTNESS,
            CAM_CONTRAST,
            CAM_SATURATION,
            CAM_SHARPNESS,
            CAM_AWB_ENABLE,
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cam is None:
            return False, None
        try:
            return True, self._cam.capture_array()
        except Exception as exc:
            logger.error("PiCamera2 read error: %s", exc)
            return False, None

    def stop(self):
        if self._cam is not None:
            self._cam.stop()
            self._cam = None


class OpenCVBackend:
    """Camera backend for USB webcams or V4L2 devices."""

    def __init__(self, device_id: int = 0, resolution=DEFAULT_RESOLUTION, fps=DEFAULT_FPS):
        self.device_id = device_id
        self.resolution = resolution
        self.fps = fps
        self._cap = None

    def start(self):
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self.device_id}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for _ in range(WARMUP_FRAMES):
            self._cap.read()
        logger.info("OpenCV camera %s started at %s @ %sfps", self.device_id, self.resolution, self.fps)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def stop(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class CameraManager:
    """Unified camera API with a latest-frame buffer."""

    def __init__(
        self,
        backend: str = "auto",
        device_id: int = 0,
        resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
        fps: int = DEFAULT_FPS,
    ):
        self._backend = self._build_backend(backend, device_id, resolution, fps)
        self._frame: Optional[np.ndarray] = None
        self._frame_ts = 0.0
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread = None
        self._frame_count = 0
        self._capture_started_at = 0.0

    def start(self):
        self._backend.start()
        self._running = True
        self._capture_started_at = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCapture")
        self._thread.start()

        deadline = time.time() + 5.0
        while self._frame is None and time.time() < deadline:
            time.sleep(0.05)
        if self._frame is None:
            raise RuntimeError("Camera did not produce a frame within 5 seconds")

        logger.info("Camera ready")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._backend.stop()
        logger.info("Camera stopped")

    def get_frame(self, copy: bool = True) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._frame is None:
                return None
            return self._frame.copy() if copy else self._frame

    def get_frame_packet(self, copy: bool = True) -> Tuple[Optional[np.ndarray], float]:
        with self._frame_lock:
            if self._frame is None:
                return None, 0.0
            frame = self._frame.copy() if copy else self._frame
            return frame, self._frame_ts

    def get_jpeg(self, quality: int = DEFAULT_STREAM_QUALITY) -> Optional[bytes]:
        frame = self.get_frame(copy=False)
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def capture_still(self, path: str, n_frames: int = 5) -> bool:
        frames = []
        last_ts = 0.0

        for _ in range(max(n_frames, 1) * 3):
            frame, ts = self.get_frame_packet(copy=True)
            if frame is None or ts <= last_ts:
                time.sleep(0.03)
                continue
            frames.append(frame.astype(np.float32))
            last_ts = ts
            if len(frames) >= n_frames:
                break
            time.sleep(0.03)

        if not frames:
            return False

        avg = np.mean(frames, axis=0).astype(np.uint8)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(path, avg))

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def capture_fps(self) -> float:
        elapsed = max(time.time() - self._capture_started_at, 1e-6)
        return self._frame_count / elapsed

    def _capture_loop(self):
        while self._running:
            ok, frame = self._backend.read()
            if ok and frame is not None:
                with self._frame_lock:
                    self._frame = frame
                    self._frame_ts = time.time()
                self._frame_count += 1
            else:
                time.sleep(0.01)

    @staticmethod
    def _build_backend(backend: str, device_id, resolution, fps):
        if backend == "picamera2":
            return PiCamera2Backend(resolution, fps)
        if backend == "opencv":
            return OpenCVBackend(device_id, resolution, fps)

        try:
            import picamera2  # noqa: F401

            logger.info("Auto-detected PiCamera2 backend")
            return PiCamera2Backend(resolution, fps)
        except ImportError:
            logger.info("PiCamera2 not found, using OpenCV backend")
            return OpenCVBackend(device_id, resolution, fps)


def mjpeg_stream_generator(
    camera: CameraManager,
    overlay_callback: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    fps_limit: int = 12,
    quality: int = DEFAULT_STREAM_QUALITY,
):
    """Yield MJPEG frames without re-running recognition inside the stream."""

    interval = 1.0 / max(fps_limit, 1)

    while True:
        started = time.time()
        frame, frame_ts = camera.get_frame_packet(copy=False)
        if frame is None:
            time.sleep(interval)
            continue

        display_frame = frame.copy()
        if overlay_callback is not None:
            display_frame = overlay_callback(display_frame, frame_ts)

        cv2.putText(
            display_frame,
            f"Stream {fps_limit}fps",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 100),
            1,
            cv2.LINE_AA,
        )

        ok, buf = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )

        elapsed = time.time() - started
        if elapsed < interval:
            time.sleep(interval - elapsed)
