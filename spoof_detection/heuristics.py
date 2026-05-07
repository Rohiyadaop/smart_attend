from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from config import settings
from utils.image_utils import border_edge_ratio, brightness_stats, highlight_ratio, variance_of_laplacian


@dataclass
class SpoofAssessment:
    spoof_score: float
    detected: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class SpoofHeuristics:
    def analyze(self, crop, state_metrics: Dict[str, float], track_state) -> SpoofAssessment:
        mean_brightness, dynamic_range = brightness_stats(crop)
        lap_var = variance_of_laplacian(crop)
        highlights = highlight_ratio(crop)
        border_ratio = border_edge_ratio(crop)

        prev_brightness = track_state.brightness_history[-1] if track_state.brightness_history else mean_brightness
        brightness_jump = abs(mean_brightness - prev_brightness)
        brightness_window = list(track_state.brightness_history) + [mean_brightness]
        flicker_std = float(np.std(brightness_window)) if brightness_window else 0.0
        yaw_motion = (
            max(track_state.yaw_history) - min(track_state.yaw_history)
            if len(track_state.yaw_history) > 1
            else 0.0
        )
        pitch_motion = (
            max(track_state.pitch_history) - min(track_state.pitch_history)
            if len(track_state.pitch_history) > 1
            else 0.0
        )
        motion_score = yaw_motion + pitch_motion

        reasons: List[str] = []
        score = 0.0

        if highlights > 0.038:
            reasons.append("screen_reflection")
            score += 0.26
        if brightness_jump > 26.0 and flicker_std > 10.0:
            reasons.append("brightness_flicker")
            score += 0.22
        if border_ratio > 0.34 and motion_score < 3.0:
            reasons.append("flat_surface_edges")
            score += 0.20
        if lap_var < 38.0 and dynamic_range < 52.0:
            reasons.append("printed_texture")
            score += 0.20
        if track_state.frames_seen >= 18 and motion_score < 1.2 and not track_state.natural_blink_done:
            reasons.append("static_pose_pattern")
            score += 0.17

        detected = score >= settings.spoof_max_score or len(reasons) >= 3
        return SpoofAssessment(
            spoof_score=min(score, 1.0),
            detected=detected,
            reasons=reasons,
            metrics={
                "brightness": round(mean_brightness, 2),
                "dynamic_range": round(dynamic_range, 2),
                "laplacian_var": round(lap_var, 2),
                "highlight_ratio": round(highlights, 4),
                "border_edge_ratio": round(border_ratio, 4),
                "brightness_jump": round(brightness_jump, 2),
                "flicker_std": round(flicker_std, 2),
                "motion_score": round(motion_score, 2),
                **state_metrics,
            },
        )
