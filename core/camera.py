"""
=============================================================================
  SmartAttend Camera Module
=============================================================================
  Provides a unified camera interface that works with:
    • Raspberry Pi Camera Module (via picamera2)
    • Standard USB webcam (via OpenCV VideoCapture)
    • Fallback to image files (for offline testing)

  HOW RASPBERRY PI CAMERA WORKS
  ─────────────────────────────────────────────────────────────────────────
  The Pi Camera Module connects to the CSI (Camera Serial Interface) port
  on the Raspberry Pi board.  The camera sensor (IMX219 / IMX477) streams
  raw Bayer data to the GPU over a dedicated high-bandwidth lane.

  picamera2 is the modern Python library (replacing the old picamera) that:
    1. Configures the sensor (resolution, framerate, exposure)
    2. Uses the ISP (Image Signal Processor) built into the BCM2711/2712
       chip to convert raw → JPEG/YUV/BGR
    3. Returns frames as numpy arrays compatible with OpenCV

  For USB webcams, OpenCV's VideoCapture opens /dev/video0 and reads
  V4L2 (Video for Linux 2) frames directly.
=============================================================================
"""

import cv2
import time
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("smartattend.camera")

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_RESOLUTION = (640, 480)    # width × height
DEFAULT_FPS        = 20
WARMUP_FRAMES      = 5             # discard first N frames (auto-exposure settle)


# ─────────────────────────────────────────────────────────────────────────────
#  Camera backends
# ─────────────────────────────────────────────────────────────────────────────

class PiCamera2Backend:
    """Uses picamera2 — for Raspberry Pi Camera Module (CSI)."""

    def __init__(self, resolution=DEFAULT_RESOLUTION, fps=DEFAULT_FPS):
        self.resolution = resolution
        self.fps        = fps
        self._cam       = None

    def start(self):
        from picamera2 import Picamera2
        self._cam = Picamera2()
        config = self._cam.create_preview_configuration(
            main={"size": self.resolution, "format": "BGR888"},
        )
        self._cam.configure(config)
        self._cam.start()
        # Warm up
        for _ in range(WARMUP_FRAMES):
            self._cam.capture_array()
        logger.info("PiCamera2 started at %s", self.resolution)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cam is None:
            return False, None
        try:
            frame = self._cam.capture_array()
            return True, frame
        except Exception as exc:
            logger.error("PiCamera2 read error: %s", exc)
            return False, None

    def stop(self):
        if self._cam:
            self._cam.stop()
            self._cam = None


class OpenCVBackend:
    """Uses OpenCV VideoCapture — for USB webcams or V4L2 devices."""

    def __init__(self, device_id: int = 0,
                 resolution=DEFAULT_RESOLUTION, fps=DEFAULT_FPS):
        self.device_id  = device_id
        self.resolution = resolution
        self.fps        = fps
        self._cap       = None

    def start(self):
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self.device_id}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Flush warm-up frames
        for _ in range(WARMUP_FRAMES):
            self._cap.read()
        logger.info("OpenCV camera %d started at %s", self.device_id, self.resolution)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def stop(self):
        if self._cap:
            self._cap.release()
            self._cap = None


# ─────────────────────────────────────────────────────────────────────────────
#  Unified Camera Manager
# ─────────────────────────────────────────────────────────────────────────────

class CameraManager:
    """
    Unified camera API with threaded frame buffering.

    The capture loop runs in a background thread and continuously reads
    the latest frame into `self._frame`.  Consumers call `get_frame()`
    to get the most recent frame without blocking on I/O.

    Usage
    -----
        cam = CameraManager(backend="auto")
        cam.start()
        frame = cam.get_frame()   # BGR numpy array
        cam.stop()
    """

    def __init__(self, backend: str = "auto",
                 device_id: int = 0,
                 resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
                 fps: int = DEFAULT_FPS):
        """
        backend : "picamera2" | "opencv" | "auto"
                  "auto" tries picamera2 first, falls back to opencv
        """
        self._backend     = self._build_backend(backend, device_id, resolution, fps)
        self._frame       = None
        self._frame_lock  = threading.Lock()
        self._running     = False
        self._thread      = None
        self._frame_count = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        self._backend.start()
        self._running = True
        self._thread  = threading.Thread(
            target=self._capture_loop, daemon=True, name="CameraCapture"
        )
        self._thread.start()
        # Wait until first frame is ready
        deadline = time.time() + 5.0
        while self._frame is None and time.time() < deadline:
            time.sleep(0.05)
        if self._frame is None:
            raise RuntimeError("Camera did not produce a frame within 5 s")
        logger.info("Camera ready ✓")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._backend.stop()
        logger.info("Camera stopped")

    # ── Frame access ──────────────────────────────────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the most recent captured frame (BGR)."""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def get_jpeg(self, quality: int = 80) -> Optional[bytes]:
        """Return the most recent frame encoded as JPEG bytes."""
        frame = self.get_frame()
        if frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    def capture_still(self, path: str, n_frames: int = 5) -> bool:
        """
        Capture a high-quality still by averaging N consecutive frames
        (reduces noise — useful for registration photos).
        """
        frames = []
        for _ in range(n_frames):
            ok, frame = self._backend.read()
            if ok and frame is not None:
                frames.append(frame.astype(np.float32))
            time.sleep(0.05)

        if not frames:
            return False

        avg = np.mean(frames, axis=0).astype(np.uint8)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(path, avg)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ── Internal ──────────────────────────────────────────────────────────

    def _capture_loop(self):
        while self._running:
            ok, frame = self._backend.read()
            if ok and frame is not None:
                with self._frame_lock:
                    self._frame = frame
                self._frame_count += 1
            else:
                time.sleep(0.01)

    @staticmethod
    def _build_backend(backend: str, device_id, resolution, fps):
        if backend == "picamera2":
            return PiCamera2Backend(resolution, fps)
        if backend == "opencv":
            return OpenCVBackend(device_id, resolution, fps)
        # auto-detect
        try:
            import picamera2  # noqa: F401
            logger.info("Auto-detected PiCamera2 backend")
            return PiCamera2Backend(resolution, fps)
        except ImportError:
            logger.info("PiCamera2 not found — using OpenCV backend")
            return OpenCVBackend(device_id, resolution, fps)


# ─────────────────────────────────────────────────────────────────────────────
#  MJPEG stream generator (for Flask live feed)
# ─────────────────────────────────────────────────────────────────────────────

def mjpeg_stream_generator(camera: CameraManager,
                            face_engine=None,
                            fps_limit: int = 15):
    """
    Generator that yields MJPEG boundary frames for Flask's Response.
    Optionally overlays face recognition bounding boxes in real time.

    Flask route example:
        @app.route("/video_feed")
        def video_feed():
            return Response(
                mjpeg_stream_generator(cam, engine),
                mimetype="multipart/x-mixed-replace; boundary=frame"
            )
    """
    interval = 1.0 / fps_limit
    while True:
        t0    = time.time()
        frame = camera.get_frame()
        if frame is None:
            time.sleep(interval)
            continue

        # Optionally run recognition overlay
        if face_engine and face_engine.is_model_loaded():
            result = face_engine.process_frame(frame, draw_boxes=True)
            display_frame = result.frame_with_boxes if result.frame_with_boxes is not None else frame
        else:
            display_frame = frame

        # Add FPS counter overlay
        fps_text = f"FPS: {1.0/max(interval,0.001):.0f}"
        cv2.putText(display_frame, fps_text, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 1)

        _, buf = cv2.imencode(".jpg", display_frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_bytes = buf.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        # Throttle to fps_limit
        elapsed = time.time() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)
