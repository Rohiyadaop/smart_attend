"""
=============================================================================
  SmartAttend Model Trainer
=============================================================================
  Scans the dataset/ directory, encodes every face image, and saves a
  pickle file that the FaceEngine loads at startup.

  DATASET STRUCTURE
  ─────────────────────────────────────────────────────────────────────────
  dataset/
    ├── STU001_John_Doe/
    │     ├── img_001.jpg
    │     ├── img_002.jpg
    │     └── img_003.jpg
    └── STU002_Jane_Smith/
          ├── img_001.jpg
          └── img_002.jpg

  Folder name format: {student_id}_{First}_{Last}
  We parse student_id and full name from the folder name.

  TRAINING ALGORITHM
  ─────────────────────────────────────────────────────────────────────────
  For each image:
    1. Load the image with OpenCV
    2. Detect face locations (HOG detector)
    3. Compute 128-D embedding with the dlib ResNet model
    4. Append (encoding, name, student_id) to the lists

  All encodings are saved together.  During recognition the engine
  computes the Euclidean distance from a new encoding to EVERY stored
  encoding and picks the closest match.

  Having multiple images per student improves accuracy because the
  model sees the person from different angles and lighting conditions.
=============================================================================
"""

import os
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger("smartattend.trainer")

DATASET_DIR = Path("dataset")
SUPPORTED   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ModelTrainer:
    """Trains and saves the face recognition model from dataset images."""

    def __init__(self, face_engine):
        self.engine = face_engine

    def train(self, progress_callback=None) -> dict:
        """
        Scan dataset/, encode all faces, save model.

        Parameters
        ----------
        progress_callback : optional callable(current, total, message)

        Returns
        -------
        dict with "success", "message", "stats"
        """
        logger.info("Training started …")
        start_time = datetime.now()

        # ── 1. Discover student folders ───────────────────────────────────
        if not DATASET_DIR.exists():
            return {"success": False, "message": "dataset/ directory not found"}

        student_dirs = [
            d for d in DATASET_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        if not student_dirs:
            return {
                "success": False,
                "message": "No student folders found in dataset/. "
                           "Register at least one student first.",
            }

        # ── 2. Collect image paths ────────────────────────────────────────
        image_jobs = []   # list of (path, name, student_id)
        for sdir in student_dirs:
            parsed = self._parse_folder_name(sdir.name)
            if not parsed:
                logger.warning("Skipping folder '%s' — unexpected format", sdir.name)
                continue
            student_id, name = parsed
            for img_path in sdir.iterdir():
                if img_path.suffix.lower() in SUPPORTED:
                    image_jobs.append((img_path, name, student_id))

        if not image_jobs:
            return {"success": False, "message": "No valid images found in dataset/"}

        # ── 3. Encode faces ───────────────────────────────────────────────
        encodings, names, student_ids = [], [], []
        failed = 0
        total  = len(image_jobs)

        for i, (img_path, name, sid) in enumerate(image_jobs):
            if progress_callback:
                progress_callback(i + 1, total,
                                  f"Encoding {img_path.name} ({name})")

            enc = self.engine.encode_image_file(str(img_path))
            if enc is not None:
                encodings.append(enc)
                names.append(name)
                student_ids.append(sid)
                logger.debug("Encoded %s → %s", img_path.name, name)
            else:
                failed += 1
                logger.warning("No face detected in %s — skipped", img_path)

        if not encodings:
            return {
                "success": False,
                "message": f"Could not encode any faces. {failed} images failed.",
            }

        # ── 4. Save model ─────────────────────────────────────────────────
        self.engine.save_model(encodings, names, student_ids)

        elapsed  = (datetime.now() - start_time).total_seconds()
        unique   = len(set(names))
        stats    = {
            "total_images":    total,
            "encoded":         len(encodings),
            "failed":          failed,
            "unique_students": unique,
            "elapsed_seconds": round(elapsed, 1),
        }
        logger.info("Training complete in %.1fs — %d encodings / %d students",
                    elapsed, len(encodings), unique)
        return {
            "success": True,
            "message": (f"Model trained successfully! "
                        f"{len(encodings)} encodings for {unique} students "
                        f"in {elapsed:.1f}s."),
            "stats": stats,
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_folder_name(folder_name: str) -> Optional[tuple]:
        """
        Parse "STU001_John_Doe" → ("STU001", "John Doe")
        Returns None on bad format.
        """
        parts = folder_name.split("_")
        if len(parts) < 2:
            return None
        student_id = parts[0]
        name       = " ".join(parts[1:])
        return student_id, name
