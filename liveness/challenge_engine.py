from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

from config import settings


@dataclass(frozen=True)
class Challenge:
    code: str
    text: str


class RandomChallengeEngine:
    def __init__(self):
        self._random = random.SystemRandom()
        self._challenges = (
            Challenge("blink_twice", "Blink twice"),
            Challenge("turn_left", "Turn head left"),
            Challenge("turn_right", "Turn head right"),
            Challenge("smile", "Smile naturally"),
            Challenge("open_mouth", "Open your mouth"),
        )

    def pick(self) -> Challenge:
        return self._random.choice(self._challenges)

    def evaluate(
        self,
        challenge_code: str,
        metrics: Dict[str, float],
        challenge_blinks: int,
    ) -> Tuple[bool, float]:
        if challenge_code == "blink_twice":
            score = min(challenge_blinks / max(settings.challenge_blink_count, 1), 1.0)
            return challenge_blinks >= settings.challenge_blink_count, score
        if challenge_code == "turn_left":
            yaw = abs(min(metrics.get("yaw", 0.0), 0.0))
            score = min(yaw / max(settings.challenge_head_turn_yaw, 1.0), 1.0)
            return metrics.get("yaw", 0.0) <= -settings.challenge_head_turn_yaw, score
        if challenge_code == "turn_right":
            yaw = max(metrics.get("yaw", 0.0), 0.0)
            score = min(yaw / max(settings.challenge_head_turn_yaw, 1.0), 1.0)
            return metrics.get("yaw", 0.0) >= settings.challenge_head_turn_yaw, score
        if challenge_code == "smile":
            score = min(metrics.get("smile_score", 0.0) / max(settings.challenge_smile_ratio, 0.1), 1.0)
            return metrics.get("smile_score", 0.0) >= settings.challenge_smile_ratio, score
        if challenge_code == "open_mouth":
            score = min(
                metrics.get("mouth_open_ratio", 0.0) / max(settings.challenge_mouth_open_ratio, 0.05),
                1.0,
            )
            return metrics.get("mouth_open_ratio", 0.0) >= settings.challenge_mouth_open_ratio, score
        return False, 0.0
