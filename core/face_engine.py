"""
=============================================================================
  SmartAttend Face Recognition Engine
=============================================================================
  Author  : SmartAttend Project (B.Tech ECE/CS Final Year)

  HOW IT WORKS
  ─────────────────────────────────────────────────────────────────────────
  This engine uses a two-stage pipeline:

  STAGE 1 — Detection (HOG or CNN):
    face_recognition.face_locations() runs a Histogram of Oriented Gradients
    (HOG) detector (fast, CPU-friendly) or a CNN detector (accurate) to find
    bounding boxes of all faces in a frame.

  STAGE 2 — Encoding:
    face_recognition.face_encodings() passes each detected face through a
    deep neural network (ResNet-34 variant) that maps the face into a
    128-dimensional embedding vector.  Faces of the same person cluster
    together; different people are far apart in this space.

  STAGE 3 — Matching:
    We compare a new 128-D vector against every stored encoding using
    Euclidean distance.  If distance < tolerance (default 0.50), it's a
    match.  We pick the closest known encoding as the identity.

  STAGE 4 — Confidence:
    Confidence = 1.0 - (distance / tolerance), clipped to [0, 1].
    We only accept a match when confidence ≥ MIN_CONFIDENCE.

  TRAINING:
    For each registered student we store all of their face encodings from
    their dataset images in a pickle file (models/encodings.pkl).
    The trainer averages multiple encodings per student to build a robust
    mean embedding, reducing noise from lighting / angle variation.
=============================================================================
"""

import os
import cv2
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

logger = logging.getLogger("smartattend.face_engine")

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

TOLERANCE      = float(os.environ.get("FACE_TOLERANCE", "0.60"))  # match if distance <= tolerance
MIN_CONFIDENCE = float(os.environ.get("FACE_MIN_CONF", "0.0"))    # extra gate; default off
SCALE_FACTOR   = float(os.environ.get("FACE_SCALE", "0.5"))       # downscale frame before detection
MODEL_PATH     = Path("models/encodings.pkl")
DATASET_DIR    = Path("dataset")

# HOG is faster on Raspberry Pi; switch to "cnn" for more accuracy
DETECTION_MODEL = os.environ.get("FACE_DETECTOR", "hog")
ENCODING_MODEL  = os.environ.get("FACE_ENCODER", "small")          # "small" or "large"
ENCODING_JITTERS = int(os.environ.get("FACE_JITTERS", "1"))        # >1 increases robustness (slower)


# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RecognitionResult:
    """Result for a single face detected in a frame."""
    name: str               # "John Doe" or "Unknown"
    student_id: str         # DB student_id or ""
    confidence: float       # 0.0 – 1.0
    bounding_box: Tuple[int, int, int, int]  # (top, right, bottom, left) pixels
    is_known: bool


@dataclass
class FrameResult:
    """Aggregated result for one camera frame."""
    faces: List[RecognitionResult] = field(default_factory=list)
    frame_with_boxes: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Face Engine
# ─────────────────────────────────────────────────────────────────────────────

class FaceEngine:
    """
    Manages face encodings and recognises faces in OpenCV frames.

    Usage
    -----
        engine = FaceEngine()
        engine.load_model()               # call once at startup
        result = engine.process_frame(frame)
        for face in result.faces:
            print(face.name, face.confidence)
    """

    def __init__(self):
        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []
        self.known_ids: List[str] = []
        self._model_loaded = False
        self._fr = None          # lazy import of face_recognition

    # ── Model management ──────────────────────────────────────────────────

    def _ensure_fr(self):
        """Lazy-import face_recognition (slow to import, only do once)."""
        if self._fr is None:
            try:
                import face_recognition as fr
                self._fr = fr
            except ImportError:
                raise RuntimeError(
                    "face_recognition library not installed.\n"
                    "Run: pip install face-recognition"
                )

    def load_model(self) -> bool:
        """
        Load pre-trained encodings from disk.
        Returns True on success, False if model file doesn't exist yet.
        """
        self._ensure_fr()
        if not MODEL_PATH.exists():
            logger.warning("No model found at %s — run train_model.py first", MODEL_PATH)
            return False
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self.known_encodings = data["encodings"]
            self.known_names     = data["names"]
            self.known_ids       = data.get("student_ids", [""] * len(self.known_names))
            self._model_loaded   = True
            logger.info("Model loaded: %d known face encodings for %d unique students",
                        len(self.known_encodings),
                        len(set(self.known_names)))
            return True
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            return False

    def save_model(self, encodings, names, student_ids):
        """Persist encodings to disk after training."""
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "encodings":   encodings,
                "names":       names,
                "student_ids": student_ids,
            }, f)
        self.known_encodings = encodings
        self.known_names     = names
        self.known_ids       = student_ids
        self._model_loaded   = True
        logger.info("Model saved with %d encodings", len(encodings))

    def is_model_loaded(self) -> bool:
        return self._model_loaded and len(self.known_encodings) > 0

    # ── Frame processing ──────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray,
                      draw_boxes: bool = True) -> FrameResult:
        """
        Detect and recognise all faces in an OpenCV BGR frame.

        Parameters
        ----------
        frame      : BGR numpy array from cv2
        draw_boxes : Whether to annotate the frame with bounding boxes

        Returns
        -------
        FrameResult with list of RecognitionResult objects
        """
        import time
        t0 = time.time()
        self._ensure_fr()
        fr = self._fr

        result = FrameResult()

        # ── 1. Downscale for speed ─────────────────────────────────────
        small = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # ── 2. Detect face bounding boxes ──────────────────────────────
        locations = fr.face_locations(rgb, model=DETECTION_MODEL)
        if not locations:
            result.frame_with_boxes = frame.copy() if draw_boxes else None
            result.processing_time_ms = (time.time() - t0) * 1000
            return result

        # ── 3. Compute 128-D embeddings ────────────────────────────────
        encodings = fr.face_encodings(
            rgb,
            locations,
            num_jitters=ENCODING_JITTERS,
            model=ENCODING_MODEL,
        )

        # ── 4. Scale bounding boxes back to original resolution ────────
        scale_inv = 1.0 / SCALE_FACTOR
        scaled_locations = [
            (int(t * scale_inv), int(r * scale_inv),
             int(b * scale_inv), int(l * scale_inv))
            for (t, r, b, l) in locations
        ]

        annotated = frame.copy() if draw_boxes else None

        for enc, (top, right, bottom, left) in zip(encodings, scaled_locations):
            rec = self._match(enc, top, right, bottom, left)
            result.faces.append(rec)

            if draw_boxes:
                self._draw_box(annotated, rec)

        result.frame_with_boxes  = annotated
        result.processing_time_ms = (time.time() - t0) * 1000
        return result

    # ── Matching ──────────────────────────────────────────────────────────

    def _match(self, encoding: np.ndarray,
               top: int, right: int, bottom: int, left: int) -> RecognitionResult:
        """Match a single encoding against all known encodings."""
        fr = self._fr

        if not self.is_model_loaded():
            return RecognitionResult("Unknown", "", 0.0,
                                     (top, right, bottom, left), False)

        distances = fr.face_distance(self.known_encodings, encoding)
        best_idx  = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        # Confidence is for display/attendance gating; distance drives the match.
        confidence = max(0.0, 1.0 - best_dist)

        if best_dist <= TOLERANCE and confidence >= MIN_CONFIDENCE:
            return RecognitionResult(
                name         = self.known_names[best_idx],
                student_id   = self.known_ids[best_idx],
                confidence   = round(confidence, 3),
                bounding_box = (top, right, bottom, left),
                is_known     = True,
            )

        return RecognitionResult("Unknown", "", 0.0,
                                 (top, right, bottom, left), False)

    # ── Drawing ───────────────────────────────────────────────────────────

    @staticmethod
    def _draw_box(frame: np.ndarray, rec: RecognitionResult):
        """Draw annotated bounding box on frame (modifies in-place)."""
        top, right, bottom, left = rec.bounding_box
        color = (0, 200, 0) if rec.is_known else (0, 60, 220)

        # Box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Label background
        label_h = 26
        cv2.rectangle(frame,
                       (left, bottom - label_h), (right, bottom),
                       color, cv2.FILLED)

        # Label text
        label = f"{rec.name} ({rec.confidence:.0%})" if rec.is_known else "Unknown"
        font_scale = 0.5
        cv2.putText(frame, label,
                    (left + 4, bottom - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Utility ───────────────────────────────────────────────────────────

    def encode_image_file(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract a single 128-D encoding from an image file.
        Returns None if no face found.
        """
        self._ensure_fr()
        fr = self._fr
        img = cv2.imread(image_path)
        if img is None:
            return None
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locs = fr.face_locations(rgb, model=DETECTION_MODEL)
        if not locs:
            return None
        encs = fr.face_encodings(
            rgb,
            locs,
            num_jitters=ENCODING_JITTERS,
            model=ENCODING_MODEL,
        )
        return encs[0] if encs else None

    def get_model_stats(self) -> Dict:
        return {
            "loaded":         self._model_loaded,
            "total_encodings": len(self.known_encodings),
            "unique_students": len(set(self.known_names)),
            "model_path":     str(MODEL_PATH),
        }
