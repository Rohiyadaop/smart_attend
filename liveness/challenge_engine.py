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
        # Only use a head-turn challenge for liveness.
        self._challenges = (Challenge("head_turn", "Turn your head left or right"),)

    def pick(self) -> Challenge:
        # Always pick the head-turn challenge (only one present).
        return self._random.choice(self._challenges)

    def evaluate(
        self,
        challenge_code: str,
        metrics: Dict[str, float],
    ) -> Tuple[bool, float]:
        if challenge_code == "head_turn":
            head_turn_delta = max(float(metrics.get("head_turn_delta", 0.0)), 0.0)
            threshold = max(settings.challenge_head_turn_yaw, 1.0)
            score = min(head_turn_delta / threshold, 1.0)
            return head_turn_delta >= threshold, score
        return False, 0.0
