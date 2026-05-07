from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from config import settings
from utils.image_utils import crop_box


class SnapshotService:
    def save_snapshot(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        bucket: str,
        student_id: str = "",
        session_id: Optional[int] = None,
        prefix: str = "face",
    ) -> str:
        target_root = (
            settings.attendance_snapshot_dir
            if bucket == "attendance"
            else settings.spoof_snapshot_dir
        )
        day_folder = target_root / datetime.now().strftime("%Y-%m-%d")
        day_folder.mkdir(parents=True, exist_ok=True)

        crop = crop_box(frame, box, margin=settings.min_face_box_margin_px)
        if crop.size == 0:
            crop = frame.copy()

        name_bits = [
            prefix,
            student_id or "unknown",
            f"s{session_id}" if session_id else "nosession",
            datetime.now().strftime("%H%M%S_%f"),
        ]
        filename = "_".join(name_bits) + ".jpg"
        path = day_folder / filename
        cv2.imwrite(str(path), crop)
        return str(path.relative_to(settings.evidence_dir)).replace("\\", "/")
