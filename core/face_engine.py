"""
SmartAttend Face Recognition Engine.

This module supports two embedding backends:
1. face_recognition / dlib (default, CPU-friendly on Raspberry Pi)
2. ArcFace via insightface (optional, enabled when installed/configured)

Models are stored as multiple embeddings per student. During matching we:
- normalize embeddings
- compare the query against every stored sample
- aggregate the best sample distance with the profile centroid distance
- reject weak or ambiguous matches as Unknown
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("smartattend.face_engine")

MATCH_THRESHOLD = float(os.environ.get("FACE_THRESHOLD", "0.47"))
MIN_CONFIDENCE = float(os.environ.get("FACE_MIN_CONF", "0.35"))
SCALE_FACTOR = float(os.environ.get("FACE_SCALE", "0.5"))
MODEL_PATH = Path("models/encodings.pkl")
DATASET_DIR = Path("dataset")

DETECTION_MODEL = os.environ.get("FACE_DETECTOR", "hog")
ENCODING_MODEL = os.environ.get("FACE_ENCODER", "small")
ENCODING_JITTERS = int(os.environ.get("FACE_JITTERS", "1"))

FACE_BACKEND = os.environ.get("FACE_BACKEND", "auto").strip().lower()
FACE_MATCH_METRIC = os.environ.get("FACE_MATCH_METRIC", "l2").strip().lower()
PROFILE_CENTROID_WEIGHT = float(os.environ.get("FACE_PROFILE_WEIGHT", "0.35"))
MIN_MATCH_MARGIN = float(os.environ.get("FACE_MIN_MARGIN", "0.04"))
MIN_FACE_SIZE = int(os.environ.get("FACE_MIN_SIZE", "48"))
ARC_DET_SIZE = int(os.environ.get("ARC_DET_SIZE", "320"))


@dataclass
class RecognitionResult:
    name: str
    student_id: str
    confidence: float
    distance: float
    similarity: float
    matched_index: int
    bounding_box: Tuple[int, int, int, int]
    is_known: bool
    status: str = "processing"
    status_text: str = "Processing"
    challenge_text: str = ""
    live_verified: bool = False
    liveness_score: float = 0.0
    spoof_score: float = 0.0
    spoof_detected: bool = False
    spoof_reasons: List[str] = field(default_factory=list)
    blink_count: int = 0
    left_ear: float = 0.0
    right_ear: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    track_key: str = ""
    should_log_spoof: bool = False


@dataclass
class FrameResult:
    faces: List[RecognitionResult] = field(default_factory=list)
    frame_with_boxes: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0
    backend: str = "unknown"


class BackendUnavailableError(RuntimeError):
    """Raised when the configured recognition backend cannot be used."""


class FaceEngine:
    """Face recognition engine with strict thresholding and profile matching."""

    def __init__(self):
        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []
        self.known_ids: List[str] = []

        self.known_matrix: Optional[np.ndarray] = None
        self.profile_centroids: Optional[np.ndarray] = None
        self.profile_index_groups: List[np.ndarray] = []
        self.profile_names: List[str] = []
        self.profile_ids: List[str] = []

        self._model_loaded = False
        self._model_backend = "face_recognition"
        self._embedding_dim = 0

        self._fr = None
        self._arcface_app = None
        self._runtime_backend = None

        self.match_threshold = MATCH_THRESHOLD
        self.min_confidence = MIN_CONFIDENCE
        self.profile_centroid_weight = min(max(PROFILE_CENTROID_WEIGHT, 0.0), 0.95)
        self.min_match_margin = max(MIN_MATCH_MARGIN, 0.0)
        self.match_metric = FACE_MATCH_METRIC if FACE_MATCH_METRIC in {"l2", "cosine"} else "l2"

    # Backend loading ----------------------------------------------------

    def _ensure_face_recognition(self):
        if self._fr is None:
            try:
                import face_recognition as fr
            except ImportError as exc:
                message = str(exc)
                message_lower = message.lower()

                if "application control policy has blocked this file" in message_lower:
                    raise BackendUnavailableError(
                        "face_recognition is installed, but Windows Application Control "
                        "blocked dlib (_dlib_pybind11). Recognition cannot run on this "
                        "machine until that policy is relaxed or the app is moved to "
                        "an allowed Linux/WSL environment."
                    ) from exc

                if "no module named" in message_lower and "face_recognition" in message_lower:
                    raise BackendUnavailableError(
                        "face_recognition is not installed. Run: "
                        "venv\\Scripts\\python.exe -m pip install face-recognition"
                    ) from exc

                if "no module named" in message_lower and "dlib" in message_lower:
                    raise BackendUnavailableError(
                        "face_recognition is installed, but its dlib dependency is missing "
                        "or blocked on this machine."
                    ) from exc

                raise BackendUnavailableError(
                    f"face_recognition backend is unavailable: {message}"
                ) from exc
            self._fr = fr
        self._runtime_backend = "face_recognition"
        return self._fr

    def _ensure_arcface(self):
        if self._arcface_app is None:
            try:
                from insightface.app import FaceAnalysis
            except ImportError as exc:
                raise BackendUnavailableError(
                    "ArcFace backend requested but insightface is not installed."
                ) from exc

            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(ARC_DET_SIZE, ARC_DET_SIZE))
            self._arcface_app = app

        self._runtime_backend = "arcface"
        return self._arcface_app

    def _preferred_backend(self) -> str:
        if self._model_loaded and self._model_backend:
            return self._model_backend
        return FACE_BACKEND if FACE_BACKEND in {"face_recognition", "arcface"} else "auto"

    def _ensure_backend(self):
        preferred = self._preferred_backend()
        if preferred == "arcface":
            return self._ensure_arcface()
        if preferred == "face_recognition":
            return self._ensure_face_recognition()

        try:
            return self._ensure_arcface()
        except Exception:
            logger.info("ArcFace backend unavailable, falling back to face_recognition")
            return self._ensure_face_recognition()

    def ensure_runtime_ready(self):
        self._ensure_backend()
        return self._runtime_backend or self._preferred_backend()

    # Model management ---------------------------------------------------

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms <= 1e-12, 1.0, norms)
        return arr / norms

    def _distance_from_similarity(self, similarity: np.ndarray) -> np.ndarray:
        similarity = np.clip(similarity.astype(np.float32), -1.0, 1.0)
        if self.match_metric == "cosine":
            return 1.0 - similarity
        return np.sqrt(np.clip(2.0 - (2.0 * similarity), 0.0, 4.0))

    def _prepare_profiles(self):
        if not self.known_encodings:
            self.known_matrix = None
            self.profile_centroids = None
            self.profile_index_groups = []
            self.profile_names = []
            self.profile_ids = []
            self._embedding_dim = 0
            self._model_loaded = False
            return

        matrix = np.vstack([np.asarray(enc, dtype=np.float32) for enc in self.known_encodings])
        self.known_matrix = self._normalize_vectors(matrix)
        self._embedding_dim = int(self.known_matrix.shape[1])

        grouped: Dict[Tuple[str, str], List[int]] = {}
        for idx, (student_id, name) in enumerate(zip(self.known_ids, self.known_names)):
            grouped.setdefault((student_id, name), []).append(idx)

        centroid_rows: List[np.ndarray] = []
        self.profile_index_groups = []
        self.profile_names = []
        self.profile_ids = []

        for (student_id, name), indices in grouped.items():
            group_idx = np.asarray(indices, dtype=np.int32)
            vectors = self.known_matrix[group_idx]
            centroid = self._normalize_vectors(np.mean(vectors, axis=0))[0]

            self.profile_index_groups.append(group_idx)
            self.profile_names.append(name)
            self.profile_ids.append(student_id)
            centroid_rows.append(centroid)

        self.profile_centroids = np.vstack(centroid_rows).astype(np.float32)
        self._model_loaded = True

    def load_model(self) -> bool:
        if not MODEL_PATH.exists():
            logger.warning("No model found at %s; train the model first", MODEL_PATH)
            return False

        try:
            with open(MODEL_PATH, "rb") as fh:
                data = pickle.load(fh)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            return False

        encodings = data.get("encodings", [])
        names = data.get("names", [])
        student_ids = data.get("student_ids", [""] * len(names))

        clean_encodings: List[np.ndarray] = []
        clean_names: List[str] = []
        clean_ids: List[str] = []
        embedding_dim = None

        for enc, name, student_id in zip(encodings, names, student_ids):
            arr = np.asarray(enc, dtype=np.float32)
            if arr.ndim != 1 or arr.size == 0:
                continue
            if embedding_dim is None:
                embedding_dim = int(arr.size)
            if int(arr.size) != embedding_dim:
                logger.warning("Skipping inconsistent embedding for %s (%s)", name, student_id)
                continue
            clean_encodings.append(arr)
            clean_names.append(str(name))
            clean_ids.append(str(student_id))

        self.known_encodings = clean_encodings
        self.known_names = clean_names
        self.known_ids = clean_ids
        self._model_backend = str(data.get("backend") or "face_recognition")

        self._prepare_profiles()
        if not self._model_loaded:
            logger.warning("Model file contained no usable encodings")
            return False

        logger.info(
            "Model loaded: %d encodings across %d students using %s backend",
            len(self.known_encodings),
            len(self.profile_names),
            self._model_backend,
        )
        return True

    def save_model(self, encodings, names, student_ids):
        backend = self._runtime_backend or self._preferred_backend()
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as fh:
            pickle.dump(
                {
                    "version": 2,
                    "backend": backend,
                    "encodings": encodings,
                    "names": names,
                    "student_ids": student_ids,
                },
                fh,
            )

        self.known_encodings = [np.asarray(enc, dtype=np.float32) for enc in encodings]
        self.known_names = [str(item) for item in names]
        self.known_ids = [str(item) for item in student_ids]
        self._model_backend = backend
        self._prepare_profiles()

        logger.info(
            "Model saved with %d encodings across %d students (%s backend)",
            len(self.known_encodings),
            len(self.profile_names),
            self._model_backend,
        )

    def is_model_loaded(self) -> bool:
        return self._model_loaded and self.known_matrix is not None and len(self.profile_names) > 0

    # Frame processing ---------------------------------------------------

    def process_frame(self, frame: np.ndarray, draw_boxes: bool = True) -> FrameResult:
        import time

        started = time.time()
        self._ensure_backend()
        backend = self._runtime_backend or "unknown"
        result = FrameResult(backend=backend)

        if backend == "arcface":
            result = self._process_frame_arcface(frame, draw_boxes=draw_boxes)
        else:
            result = self._process_frame_face_recognition(frame, draw_boxes=draw_boxes)

        result.processing_time_ms = (time.time() - started) * 1000.0
        result.backend = backend
        return result

    def _process_frame_face_recognition(self, frame: np.ndarray, draw_boxes: bool) -> FrameResult:
        fr = self._ensure_face_recognition()
        result = FrameResult(backend="face_recognition")

        if SCALE_FACTOR != 1.0:
            small = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        else:
            small = frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = fr.face_locations(rgb, model=DETECTION_MODEL)
        if not locations:
            result.frame_with_boxes = frame.copy() if draw_boxes else None
            return result

        encodings = fr.face_encodings(
            rgb,
            locations,
            num_jitters=ENCODING_JITTERS,
            model=ENCODING_MODEL,
        )

        scale_inv = 1.0 / SCALE_FACTOR if SCALE_FACTOR else 1.0
        scaled_locations = [
            (
                int(top * scale_inv),
                int(right * scale_inv),
                int(bottom * scale_inv),
                int(left * scale_inv),
            )
            for (top, right, bottom, left) in locations
        ]

        annotated = frame.copy() if draw_boxes else None
        for enc, box in zip(encodings, scaled_locations):
            rec = self._match(enc, box)
            result.faces.append(rec)
            if draw_boxes and annotated is not None:
                self._draw_box(annotated, rec)

        result.frame_with_boxes = annotated
        return result

    def _process_frame_arcface(self, frame: np.ndarray, draw_boxes: bool) -> FrameResult:
        app = self._ensure_arcface()
        result = FrameResult(backend="arcface")

        faces = app.get(frame)
        if not faces:
            result.frame_with_boxes = frame.copy() if draw_boxes else None
            return result

        annotated = frame.copy() if draw_boxes else None
        for face in faces:
            embedding = getattr(face, "embedding", None)
            bbox = getattr(face, "bbox", None)
            if embedding is None and isinstance(face, dict):
                embedding = face.get("embedding")
                bbox = face.get("bbox")
            if embedding is None or bbox is None:
                continue

            left, top, right, bottom = [int(v) for v in bbox]
            rec = self._match(
                np.asarray(embedding, dtype=np.float32),
                (top, right, bottom, left),
            )
            result.faces.append(rec)
            if draw_boxes and annotated is not None:
                self._draw_box(annotated, rec)

        result.frame_with_boxes = annotated
        return result

    # Matching -----------------------------------------------------------

    def _unknown_result(
        self,
        box: Tuple[int, int, int, int],
        distance: float = 1.0,
        similarity: float = 0.0,
        matched_index: int = -1,
    ) -> RecognitionResult:
        return RecognitionResult(
            name="Unknown",
            student_id="",
            confidence=0.0,
            distance=round(float(distance), 4),
            similarity=round(float(similarity), 4),
            matched_index=int(matched_index),
            bounding_box=box,
            is_known=False,
        )

    def _match(self, encoding: np.ndarray, box: Tuple[int, int, int, int]) -> RecognitionResult:
        if not self.is_model_loaded() or self.known_matrix is None or self.profile_centroids is None:
            return self._unknown_result(box)

        top, right, bottom, left = box
        width = max(right - left, 0)
        height = max(bottom - top, 0)
        if min(width, height) < MIN_FACE_SIZE:
            logger.info(
                "match skipped: face too small (%dx%d), result=Unknown",
                width,
                height,
            )
            return self._unknown_result(box, distance=1.0)

        query = self._normalize_vectors(np.asarray(encoding, dtype=np.float32))[0]
        if self._embedding_dim and int(query.size) != self._embedding_dim:
            logger.warning(
                "Embedding dimension mismatch: query=%d model=%d",
                int(query.size),
                self._embedding_dim,
            )
            return self._unknown_result(box)

        sample_similarity = self.known_matrix @ query
        sample_distance = self._distance_from_similarity(sample_similarity)

        centroid_similarity = self.profile_centroids @ query
        centroid_distance = self._distance_from_similarity(centroid_similarity)

        best_profile_idx = -1
        best_sample_idx = -1
        best_distance = float("inf")
        best_sample_distance = float("inf")
        best_centroid_distance = float("inf")
        best_similarity = 0.0
        runner_up_distance = float("inf")

        sample_weight = 1.0 - self.profile_centroid_weight

        for profile_idx, group_indices in enumerate(self.profile_index_groups):
            profile_sample_distance = sample_distance[group_indices]
            rel_idx = int(np.argmin(profile_sample_distance))
            sample_idx = int(group_indices[rel_idx])

            min_sample_distance = float(profile_sample_distance[rel_idx])
            centroid_dist = float(centroid_distance[profile_idx])
            composite_distance = (
                (sample_weight * min_sample_distance)
                + (self.profile_centroid_weight * centroid_dist)
            )

            if composite_distance < best_distance:
                runner_up_distance = best_distance
                best_distance = composite_distance
                best_profile_idx = profile_idx
                best_sample_idx = sample_idx
                best_sample_distance = min_sample_distance
                best_centroid_distance = centroid_dist
                best_similarity = float(sample_similarity[sample_idx])
            elif composite_distance < runner_up_distance:
                runner_up_distance = composite_distance

        margin = (
            runner_up_distance - best_distance
            if np.isfinite(runner_up_distance)
            else float("inf")
        )
        confidence = max(0.0, 1.0 - (best_distance / max(self.match_threshold, 1e-6)))

        threshold_pass = best_distance < self.match_threshold
        confidence_pass = confidence >= self.min_confidence
        margin_pass = margin >= self.min_match_margin
        is_known = threshold_pass and confidence_pass and margin_pass

        predicted_name = self.profile_names[best_profile_idx] if best_profile_idx >= 0 else "Unknown"
        logger.info(
            "match idx=%s sample=%s candidate=%s distance=%.4f sample=%.4f centroid=%.4f "
            "similarity=%.4f margin=%.4f result=%s",
            best_profile_idx,
            best_sample_idx,
            predicted_name,
            best_distance,
            best_sample_distance,
            best_centroid_distance,
            best_similarity,
            margin,
            predicted_name if is_known else "Unknown",
        )

        if not is_known or best_profile_idx < 0:
            return self._unknown_result(
                box,
                distance=best_distance,
                similarity=best_similarity,
                matched_index=best_profile_idx,
            )

        return RecognitionResult(
            name=self.profile_names[best_profile_idx],
            student_id=self.profile_ids[best_profile_idx],
            confidence=round(confidence, 4),
            distance=round(best_distance, 4),
            similarity=round(best_similarity, 4),
            matched_index=int(best_profile_idx),
            bounding_box=box,
            is_known=True,
        )

    # Drawing ------------------------------------------------------------

    def draw_results(self, frame: np.ndarray, faces: Sequence[RecognitionResult]):
        for rec in faces:
            self._draw_box(frame, rec)
        return frame

    @staticmethod
    def _draw_box(frame: np.ndarray, rec: RecognitionResult):
        top, right, bottom, left = rec.bounding_box
        if rec.status == "verified" and rec.live_verified:
            color = (40, 190, 90)
        elif rec.status == "spoof_detected":
            color = (55, 55, 220)
        elif rec.is_known:
            color = (0, 210, 255)
        else:
            color = (0, 70, 220)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        info_rows = [
            f"{rec.name if rec.is_known else 'Unknown'} {rec.confidence:.0%}",
            f"{rec.status_text} | L {rec.liveness_score:.0%}",
        ]
        if rec.challenge_text and rec.status not in {"verified", "spoof_detected"}:
            info_rows.append(rec.challenge_text[:28])
        label_height = 20 * len(info_rows) + 8
        cv2.rectangle(frame, (left, max(bottom - label_height, 0)), (right, bottom), color, cv2.FILLED)

        for idx, line in enumerate(info_rows):
            cv2.putText(
                frame,
                line,
                (left + 4, max(bottom - label_height + 18 + (idx * 18), 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Utility ------------------------------------------------------------

    def encode_image_file(self, image_path: str) -> Optional[np.ndarray]:
        backend = self._preferred_backend()

        if backend == "arcface":
            app = self._ensure_arcface()
            img = cv2.imread(image_path)
            if img is None:
                return None
            faces = app.get(img)
            if len(faces) != 1:
                return None
            embedding = getattr(faces[0], "embedding", None)
            if embedding is None and isinstance(faces[0], dict):
                embedding = faces[0].get("embedding")
            return np.asarray(embedding, dtype=np.float32) if embedding is not None else None

        fr = self._ensure_face_recognition()
        img = cv2.imread(image_path)
        if img is None:
            return None

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locations = fr.face_locations(rgb, model=DETECTION_MODEL)
        if len(locations) != 1:
            return None

        encodings = fr.face_encodings(
            rgb,
            locations,
            num_jitters=ENCODING_JITTERS,
            model=ENCODING_MODEL,
        )
        if len(encodings) != 1:
            return None
        return np.asarray(encodings[0], dtype=np.float32)

    def get_model_stats(self) -> Dict[str, Any]:
        return {
            "loaded": self._model_loaded,
            "total_encodings": len(self.known_encodings),
            "unique_students": len(self.profile_names),
            "model_path": str(MODEL_PATH),
            "backend": self._model_backend,
            "match_metric": self.match_metric,
            "threshold": self.match_threshold,
        }
