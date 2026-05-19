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
        # Only use a blink challenge for liveness (no head turns / smile / mouth)
        self._challenges = (Challenge("blink", "Blink your eyes"),)

    def pick(self) -> Challenge:
        # Always pick the blink challenge (only one present)
        return self._random.choice(self._challenges)

    def evaluate(
        self,
        challenge_code: str,
        metrics: Dict[str, float],
        challenge_blinks: int,
    ) -> Tuple[bool, float]:
        # Only evaluate blink challenge
        if challenge_code == "blink":
            score = min(challenge_blinks / max(settings.challenge_blink_count, 1), 1.0)
            return challenge_blinks >= settings.challenge_blink_count, score
        return False, 0.0
