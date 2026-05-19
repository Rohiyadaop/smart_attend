from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv(override=False)


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
EVIDENCE_DIR = LOG_DIR / "evidence"
ATTENDANCE_EVIDENCE_DIR = EVIDENCE_DIR / "attendance"
SPOOF_EVIDENCE_DIR = EVIDENCE_DIR / "spoof"

for path in (LOG_DIR, EVIDENCE_DIR, ATTENDANCE_EVIDENCE_DIR, SPOOF_EVIDENCE_DIR):
    path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    face_threshold: float = float(os.environ.get("FACE_THRESHOLD", "0.47"))
    face_min_conf: float = float(os.environ.get("FACE_MIN_CONF", "0.35"))
    attend_min_conf: float = float(
        os.environ.get("ATTEND_MIN_CONF", os.environ.get("FACE_MIN_CONF", "0.45"))
    )
    liveness_min_score: float = float(os.environ.get("LIVENESS_MIN_SCORE", "0.68"))
    spoof_max_score: float = float(os.environ.get("SPOOF_MAX_SCORE", "0.64"))
    blink_ear_threshold: float = float(os.environ.get("BLINK_EAR_THRESHOLD", "0.20"))
    blink_ear_reopen_threshold: float = float(
        os.environ.get("BLINK_EAR_REOPEN_THRESHOLD", "0.235")
    )
    challenge_timeout_sec: float = float(os.environ.get("CHALLENGE_TIMEOUT_SEC", "10"))
    security_track_ttl_sec: float = float(os.environ.get("SECURITY_TRACK_TTL_SEC", "6"))
    spoof_log_cooldown_sec: float = float(os.environ.get("SPOOF_LOG_COOLDOWN_SEC", "8"))
    challenge_head_turn_yaw: float = float(os.environ.get("CHALLENGE_HEAD_TURN_YAW", "16"))
    challenge_smile_ratio: float = float(os.environ.get("CHALLENGE_SMILE_RATIO", "1.78"))
    challenge_mouth_open_ratio: float = float(
        os.environ.get("CHALLENGE_MOUTH_OPEN_RATIO", "0.20")
    )
    challenge_blink_count: int = int(os.environ.get("CHALLENGE_BLINK_COUNT", "0"))
    min_face_box_margin_px: int = int(os.environ.get("FACE_BOX_MARGIN_PX", "18"))
    max_security_faces: int = int(os.environ.get("MAX_SECURITY_FACES", "3"))
    attendance_snapshot_dir: Path = ATTENDANCE_EVIDENCE_DIR
    spoof_snapshot_dir: Path = SPOOF_EVIDENCE_DIR
    evidence_dir: Path = EVIDENCE_DIR


settings = Settings()
