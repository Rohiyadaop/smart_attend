from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]


def clamp_box(box: Box, width: int, height: int, margin: int = 0) -> Box:
    top, right, bottom, left = box
    return (
        max(0, int(top - margin)),
        min(width, int(right + margin)),
        min(height, int(bottom + margin)),
        max(0, int(left - margin)),
    )


def crop_box(frame: np.ndarray, box: Box, margin: int = 0) -> np.ndarray:
    height, width = frame.shape[:2]
    top, right, bottom, left = clamp_box(box, width, height, margin=margin)
    if bottom <= top or right <= left:
        return np.empty((0, 0, 3), dtype=frame.dtype)
    return frame[top:bottom, left:right].copy()


def euclidean(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return float(np.linalg.norm(np.asarray(p1, dtype=np.float32) - np.asarray(p2, dtype=np.float32)))


def centroid(points: Iterable[Tuple[int, int]]) -> Tuple[int, int]:
    pts = np.asarray(list(points), dtype=np.float32)
    if pts.size == 0:
        return (0, 0)
    x, y = np.mean(pts, axis=0)
    return (int(x), int(y))


def variance_of_laplacian(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_stats(image: np.ndarray) -> Tuple[float, float]:
    if image.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(np.mean(gray)), float(np.percentile(gray, 95) - np.percentile(gray, 5))


def highlight_ratio(image: np.ndarray, threshold: int = 245) -> float:
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(np.mean(gray >= threshold))


def border_edge_ratio(image: np.ndarray, border_pct: float = 0.12) -> float:
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 80, 180)
    if not np.any(edges):
        return 0.0

    h, w = edges.shape[:2]
    border_y = max(int(h * border_pct), 1)
    border_x = max(int(w * border_pct), 1)
    mask = np.zeros_like(edges, dtype=np.uint8)
    mask[:border_y, :] = 1
    mask[h - border_y :, :] = 1
    mask[:, :border_x] = 1
    mask[:, w - border_x :] = 1
    border_edges = np.count_nonzero(edges * mask)
    total_edges = np.count_nonzero(edges)
    return float(border_edges / max(total_edges, 1))


def safe_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False
