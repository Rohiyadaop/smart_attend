from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from config import settings
from liveness.challenge_engine import RandomChallengeEngine
from liveness.landmark_analyzer import FaceLandmarkAnalyzer
from spoof_detection.heuristics import SpoofHeuristics
from utils.image_utils import crop_box


@dataclass
class SecurityTrackState:
    key: str
    last_seen_at: float = 0.0
    frames_seen: int = 0
    eyes_closed: bool = False
    total_blinks: int = 0
    natural_blink_done: bool = False
    challenge_code: str = ""
    challenge_text: str = "Blink naturally to begin"
    challenge_started_at: float = 0.0
    challenge_passed: bool = False
    challenge_progress: float = 0.0
    challenge_blinks: int = 0
    last_spoof_logged_at: float = 0.0
    brightness_history: Deque[float] = field(default_factory=lambda: deque(maxlen=12))
    yaw_history: Deque[float] = field(default_factory=lambda: deque(maxlen=12))
    pitch_history: Deque[float] = field(default_factory=lambda: deque(maxlen=12))


class SecurityPipeline:
    def __init__(self):
        self._landmarks = FaceLandmarkAnalyzer()
        self._challenges = RandomChallengeEngine()
        self._spoof = SpoofHeuristics()
        self._tracks: Dict[str, SecurityTrackState] = {}
        self._session_marker: Optional[int] = None

    def reset(self):
        self._tracks.clear()
        self._session_marker = None

    def set_session(self, session_id: Optional[int]):
        if session_id != self._session_marker:
            self.reset()
            self._session_marker = session_id

    def evaluate(self, frame, faces):
        now = time.time()
        self._expire_tracks(now)

        for face in faces[: settings.max_security_faces]:
            track_key = face.student_id or self._box_track_key(face.bounding_box)
            face.track_key = track_key
            state = self._tracks.setdefault(track_key, SecurityTrackState(key=track_key))
            state.last_seen_at = now
            state.frames_seen += 1

            metrics = self._landmarks.analyze_face(frame, face.bounding_box)
            if metrics is None:
                face.status = "processing"
                face.status_text = "Align face for liveness"
                face.challenge_text = state.challenge_text
                face.live_verified = False
                face.liveness_score = 0.0
                face.spoof_score = 0.0
                face.spoof_detected = False
                face.spoof_reasons = []
                continue

            blink_detected = self._update_blink_state(state, metrics.avg_ear)
            if blink_detected:
                state.total_blinks += 1
                if not state.natural_blink_done:
                    state.natural_blink_done = True
                if state.challenge_code == "blink_twice" and not state.challenge_passed:
                    state.challenge_blinks += 1

            state.yaw_history.append(metrics.yaw)
            state.pitch_history.append(metrics.pitch)

            state_metrics = {
                "yaw": round(metrics.yaw, 2),
                "pitch": round(metrics.pitch, 2),
                "roll": round(metrics.roll, 2),
                "ear": round(metrics.avg_ear, 4),
                "mouth_open_ratio": round(metrics.mouth_open_ratio, 4),
                "smile_score": round(metrics.smile_score, 4),
            }

            if state.natural_blink_done and not state.challenge_code:
                challenge = self._challenges.pick()
                state.challenge_code = challenge.code
                state.challenge_text = challenge.text
                state.challenge_started_at = now
                state.challenge_blinks = 0
                state.challenge_progress = 0.0

            if (
                state.challenge_code
                and not state.challenge_passed
                and (now - state.challenge_started_at) > settings.challenge_timeout_sec
            ):
                challenge = self._challenges.pick()
                state.challenge_code = challenge.code
                state.challenge_text = challenge.text
                state.challenge_started_at = now
                state.challenge_blinks = 0
                state.challenge_progress = 0.0

            if state.challenge_code and not state.challenge_passed:
                state.challenge_passed, state.challenge_progress = self._challenges.evaluate(
                    state.challenge_code,
                    state_metrics,
                    state.challenge_blinks,
                )

            crop = crop_box(frame, face.bounding_box, margin=settings.min_face_box_margin_px)
            spoof = self._spoof.analyze(crop, state_metrics, state)
            state.brightness_history.append(spoof.metrics.get("brightness", 0.0))

            face.left_ear = round(metrics.left_ear, 4)
            face.right_ear = round(metrics.right_ear, 4)
            face.yaw = round(metrics.yaw, 2)
            face.pitch = round(metrics.pitch, 2)
            face.roll = round(metrics.roll, 2)
            face.blink_count = state.total_blinks
            face.challenge_text = state.challenge_text if state.natural_blink_done else "Blink naturally"
            face.liveness_score = round(self._liveness_score(state, spoof.spoof_score), 4)
            face.spoof_score = round(spoof.spoof_score, 4)
            face.spoof_detected = spoof.detected
            face.spoof_reasons = spoof.reasons

            if spoof.detected:
                face.status = "spoof_detected"
                face.status_text = "Spoof detected"
                face.live_verified = False
                if (now - state.last_spoof_logged_at) >= settings.spoof_log_cooldown_sec:
                    face.should_log_spoof = True
                    state.last_spoof_logged_at = now
                continue

            if not state.natural_blink_done:
                face.status = "awaiting_blink"
                face.status_text = "Waiting for natural blink"
                face.live_verified = False
                continue

            if not state.challenge_passed:
                face.status = "challenge_pending"
                face.status_text = f"Challenge: {state.challenge_text}"
                face.live_verified = False
                continue

            if face.liveness_score < settings.liveness_min_score:
                face.status = "liveness_pending"
                face.status_text = "Liveness below threshold"
                face.live_verified = False
                continue

            if face.is_known:
                face.status = "verified"
                face.status_text = "Live verified"
                face.live_verified = True
            else:
                face.status = "unknown_live"
                face.status_text = "Live face - unknown"
                face.live_verified = False

        for face in faces[settings.max_security_faces :]:
            face.status = "processing"
            face.status_text = "Waiting in queue"
            face.challenge_text = "Move closer to camera"

        return faces

    @staticmethod
    def _box_track_key(box) -> str:
        top, right, bottom, left = box
        return f"bbox:{left // 24}:{top // 24}:{(right - left) // 24}:{(bottom - top) // 24}"

    @staticmethod
    def _update_blink_state(state: SecurityTrackState, avg_ear: float) -> bool:
        blink = False
        if avg_ear <= settings.blink_ear_threshold:
            state.eyes_closed = True
        elif state.eyes_closed and avg_ear >= settings.blink_ear_reopen_threshold:
            state.eyes_closed = False
            blink = True
        else:
            state.eyes_closed = False
        return blink

    @staticmethod
    def _liveness_score(state: SecurityTrackState, spoof_score: float) -> float:
        blink_score = 0.30 if state.natural_blink_done else min(state.total_blinks * 0.12, 0.24)
        challenge_score = 0.40 if state.challenge_passed else (state.challenge_progress * 0.30)
        pose_motion = 0.0
        if len(state.yaw_history) > 1:
            yaw_span = max(state.yaw_history) - min(state.yaw_history)
            pitch_span = max(state.pitch_history) - min(state.pitch_history)
            pose_motion = min((yaw_span + pitch_span) / 24.0, 0.20)
        presence_score = min(state.frames_seen / 15.0, 1.0) * 0.10
        total = blink_score + challenge_score + pose_motion + presence_score - (spoof_score * 0.30)
        return max(0.0, min(total, 1.0))

    def _expire_tracks(self, now: float):
        expired = [
            key
            for key, state in self._tracks.items()
            if (now - state.last_seen_at) > settings.security_track_ttl_sec
        ]
        for key in expired:
            self._tracks.pop(key, None)
