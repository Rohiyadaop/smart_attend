"""
SmartAttend model trainer.

Training filters low-quality images, rejects ambiguous training photos, keeps
multiple embeddings per student, and caps the number of stored samples so
recognition stays fast on Raspberry Pi hardware.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger("smartattend.trainer")

DATASET_DIR = Path("dataset")
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_SHARPNESS = float(os.environ.get("TRAIN_MIN_SHARPNESS", "55.0"))
MIN_BRIGHTNESS = float(os.environ.get("TRAIN_MIN_BRIGHTNESS", "35.0"))
MAX_BRIGHTNESS = float(os.environ.get("TRAIN_MAX_BRIGHTNESS", "225.0"))
MAX_SAMPLES_PER_STUDENT = int(os.environ.get("TRAIN_MAX_SAMPLES_PER_STUDENT", "25"))
DEDUP_DISTANCE = float(os.environ.get("TRAIN_DUP_DISTANCE", "0.08"))


class ModelTrainer:
    """Trains and saves the face recognition model from dataset images."""

    def __init__(self, face_engine):
        self.engine = face_engine

    def train(self, progress_callback=None) -> dict:
        logger.info("Training started")
        started_at = datetime.now()

        if not DATASET_DIR.exists():
            return {"success": False, "message": "dataset/ directory not found"}

        student_dirs = sorted(
            d for d in DATASET_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        if not student_dirs:
            return {
                "success": False,
                "message": "No student folders found in dataset/. Register at least one student first.",
            }

        image_jobs = []
        for student_dir in student_dirs:
            parsed = self._parse_folder_name(student_dir.name)
            if not parsed:
                logger.warning("Skipping folder '%s' due to unexpected format", student_dir.name)
                continue

            student_id, name = parsed
            for image_path in sorted(student_dir.iterdir()):
                if image_path.suffix.lower() in SUPPORTED:
                    image_jobs.append((image_path, name, student_id))

        if not image_jobs:
            return {"success": False, "message": "No valid images found in dataset/"}

        encodings = []
        names = []
        student_ids = []

        per_student_vectors: Dict[str, list[np.ndarray]] = defaultdict(list)
        failed = 0
        skipped_quality = 0
        skipped_duplicate = 0
        skipped_cap = 0

        total = len(image_jobs)
        for index, (image_path, name, student_id) in enumerate(image_jobs, start=1):
            if progress_callback:
                progress_callback(index, total, f"Encoding {image_path.name} ({name})")

            if len(per_student_vectors[student_id]) >= MAX_SAMPLES_PER_STUDENT:
                skipped_cap += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                failed += 1
                logger.warning("Could not read %s", image_path)
                continue

            quality = self._assess_quality(image)
            if not quality["ok"]:
                skipped_quality += 1
                logger.info(
                    "Skipping %s due to image quality: sharpness=%.1f brightness=%.1f",
                    image_path.name,
                    quality["sharpness"],
                    quality["brightness"],
                )
                continue

            enc = self.engine.encode_image_file(str(image_path))
            if enc is None:
                failed += 1
                logger.warning("Skipping %s: expected exactly one usable face", image_path)
                continue

            enc = np.asarray(enc, dtype=np.float32)
            norm_enc = self.engine._normalize_vectors(enc)[0]
            existing = per_student_vectors[student_id]
            if existing:
                min_distance = min(float(np.linalg.norm(item - norm_enc)) for item in existing)
                if min_distance < DEDUP_DISTANCE:
                    skipped_duplicate += 1
                    logger.info(
                        "Skipping near-duplicate sample %s for %s (distance=%.4f)",
                        image_path.name,
                        student_id,
                        min_distance,
                    )
                    continue

            per_student_vectors[student_id].append(norm_enc)
            encodings.append(enc)
            names.append(name)
            student_ids.append(student_id)

        if not encodings:
            return {
                "success": False,
                "message": "Could not encode any usable faces. Check dataset quality and lighting.",
            }

        self.engine.save_model(encodings, names, student_ids)

        elapsed = (datetime.now() - started_at).total_seconds()
        stats = {
            "total_images": total,
            "encoded": len(encodings),
            "failed": failed,
            "skipped_quality": skipped_quality,
            "skipped_duplicate": skipped_duplicate,
            "skipped_cap": skipped_cap,
            "unique_students": len(set(student_ids)),
            "elapsed_seconds": round(elapsed, 1),
            "max_samples_per_student": MAX_SAMPLES_PER_STUDENT,
        }

        logger.info(
            "Training complete in %.1fs: encoded=%d failed=%d quality_skips=%d duplicate_skips=%d cap_skips=%d",
            elapsed,
            len(encodings),
            failed,
            skipped_quality,
            skipped_duplicate,
            skipped_cap,
        )

        return {
            "success": True,
            "message": (
                f"Model trained successfully with {len(encodings)} embeddings for "
                f"{len(set(student_ids))} students in {elapsed:.1f}s."
            ),
            "stats": stats,
        }

    @staticmethod
    def _assess_quality(image: np.ndarray) -> Dict[str, float | bool]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())

        ok = (
            sharpness >= MIN_SHARPNESS
            and MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS
        )
        return {
            "ok": ok,
            "sharpness": sharpness,
            "brightness": brightness,
        }

    @staticmethod
    def _parse_folder_name(folder_name: str) -> Optional[tuple]:
        parts = folder_name.split("_")
        if len(parts) < 2:
            return None
        student_id = parts[0]
        name = " ".join(parts[1:])
        return student_id, name
