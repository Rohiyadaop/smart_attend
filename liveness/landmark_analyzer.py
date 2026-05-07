from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import settings
from utils.image_utils import centroid, clamp_box, crop_box, euclidean

logger = logging.getLogger("smartattend.landmarks")

Point = Tuple[int, int]
Box = Tuple[int, int, int, int]


@dataclass
class FaceLandmarkMetrics:
    provider: str
    bounding_box: Box
    left_eye_points: List[Point] = field(default_factory=list)
    right_eye_points: List[Point] = field(default_factory=list)
    mouth_points: List[Point] = field(default_factory=list)
    nose_tip: Point = (0, 0)
    chin: Point = (0, 0)
    mouth_left: Point = (0, 0)
    mouth_right: Point = (0, 0)
    left_eye_outer: Point = (0, 0)
    right_eye_outer: Point = (0, 0)
    left_ear: float = 0.0
    right_ear: float = 0.0
    mouth_open_ratio: float = 0.0
    smile_score: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    landmarks: Dict[str, List[Point]] = field(default_factory=dict)

    @property
    def avg_ear(self) -> float:
        return (self.left_ear + self.right_ear) / 2.0


class FaceLandmarkAnalyzer:
    def __init__(self):
        self._mp_face_mesh = None
        self._face_recognition = None

    def _ensure_mediapipe(self):
        if self._mp_face_mesh is not None:
            return self._mp_face_mesh
        try:
            import mediapipe as mp
        except ImportError:
            return None

        self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return self._mp_face_mesh

    def _ensure_face_recognition(self):
        if self._face_recognition is None:
            import face_recognition as fr

            self._face_recognition = fr
        return self._face_recognition

    def analyze_face(self, frame: np.ndarray, box: Box) -> Optional[FaceLandmarkMetrics]:
        metrics = self._analyze_with_mediapipe(frame, box)
        if metrics is not None:
            return metrics
        return self._analyze_with_face_recognition(frame, box)

    def _analyze_with_mediapipe(self, frame: np.ndarray, box: Box) -> Optional[FaceLandmarkMetrics]:
        mesh = self._ensure_mediapipe()
        if mesh is None:
            return None

        crop = crop_box(frame, box, margin=settings.min_face_box_margin_px)
        if crop.size == 0:
            return None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        top, right, bottom, left = clamp_box(
            box, frame.shape[1], frame.shape[0], margin=settings.min_face_box_margin_px
        )
        crop_h, crop_w = crop.shape[:2]
        face_landmarks = result.multi_face_landmarks[0]

        def point(index: int) -> Point:
            landmark = face_landmarks.landmark[index]
            x = int(left + landmark.x * crop_w)
            y = int(top + landmark.y * crop_h)
            return (x, y)

        left_eye = [point(idx) for idx in (33, 160, 158, 133, 153, 144)]
        right_eye = [point(idx) for idx in (362, 385, 387, 263, 373, 380)]
        mouth_points = [point(idx) for idx in (61, 13, 291, 14, 78, 308)]

        metrics = FaceLandmarkMetrics(
            provider="mediapipe",
            bounding_box=box,
            left_eye_points=left_eye,
            right_eye_points=right_eye,
            mouth_points=mouth_points,
            nose_tip=point(1),
            chin=point(152),
            mouth_left=point(61),
            mouth_right=point(291),
            left_eye_outer=point(33),
            right_eye_outer=point(263),
        )
        metrics.left_ear = self._eye_aspect_ratio(left_eye)
        metrics.right_ear = self._eye_aspect_ratio(right_eye)
        metrics.mouth_open_ratio = self._mouth_open_ratio(metrics)
        metrics.smile_score = self._smile_ratio(metrics)
        metrics.yaw, metrics.pitch, metrics.roll = self._estimate_head_pose(frame, metrics)
        metrics.landmarks = {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "mouth": mouth_points,
        }
        return metrics

    def _analyze_with_face_recognition(self, frame: np.ndarray, box: Box) -> Optional[FaceLandmarkMetrics]:
        fr = self._ensure_face_recognition()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        top, right, bottom, left = box
        landmarks_list = fr.face_landmarks(rgb, [(top, right, bottom, left)])
        if not landmarks_list:
            return None

        landmarks = landmarks_list[0]
        left_eye = [(int(x), int(y)) for x, y in landmarks.get("left_eye", [])]
        right_eye = [(int(x), int(y)) for x, y in landmarks.get("right_eye", [])]
        top_lip = [(int(x), int(y)) for x, y in landmarks.get("top_lip", [])]
        bottom_lip = [(int(x), int(y)) for x, y in landmarks.get("bottom_lip", [])]
        mouth = top_lip + bottom_lip
        nose_tip_points = [(int(x), int(y)) for x, y in landmarks.get("nose_tip", [])]
        chin_points = [(int(x), int(y)) for x, y in landmarks.get("chin", [])]

        if not left_eye or not right_eye or not top_lip or not bottom_lip or not nose_tip_points or not chin_points:
            return None

        mouth_left = top_lip[0]
        mouth_right = top_lip[6] if len(top_lip) > 6 else top_lip[-1]
        metrics = FaceLandmarkMetrics(
            provider="face_recognition",
            bounding_box=box,
            left_eye_points=left_eye,
            right_eye_points=right_eye,
            mouth_points=mouth,
            nose_tip=centroid(nose_tip_points),
            chin=chin_points[len(chin_points) // 2],
            mouth_left=mouth_left,
            mouth_right=mouth_right,
            left_eye_outer=left_eye[0],
            right_eye_outer=right_eye[3] if len(right_eye) > 3 else right_eye[-1],
        )
        metrics.left_ear = self._eye_aspect_ratio(left_eye)
        metrics.right_ear = self._eye_aspect_ratio(right_eye)
        metrics.mouth_open_ratio = self._mouth_open_ratio(metrics)
        metrics.smile_score = self._smile_ratio(metrics)
        metrics.yaw, metrics.pitch, metrics.roll = self._estimate_head_pose(frame, metrics)
        metrics.landmarks = {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "mouth": mouth,
            "nose_tip": nose_tip_points,
            "chin": chin_points,
        }
        return metrics

    @staticmethod
    def _eye_aspect_ratio(points: Sequence[Point]) -> float:
        if len(points) < 6:
            return 0.0
        p1, p2, p3, p4, p5, p6 = points[:6]
        vertical = euclidean(p2, p6) + euclidean(p3, p5)
        horizontal = max(euclidean(p1, p4), 1e-6)
        return float(vertical / (2.0 * horizontal))

    @staticmethod
    def _mouth_open_ratio(metrics: FaceLandmarkMetrics) -> float:
        mouth_width = max(euclidean(metrics.mouth_left, metrics.mouth_right), 1e-6)
        mouth_center_top = centroid(metrics.mouth_points[: len(metrics.mouth_points) // 2])
        mouth_center_bottom = centroid(metrics.mouth_points[len(metrics.mouth_points) // 2 :])
        mouth_open = euclidean(mouth_center_top, mouth_center_bottom)
        return float(mouth_open / mouth_width)

    @staticmethod
    def _smile_ratio(metrics: FaceLandmarkMetrics) -> float:
        eye_span = max(euclidean(metrics.left_eye_outer, metrics.right_eye_outer), 1e-6)
        mouth_width = euclidean(metrics.mouth_left, metrics.mouth_right)
        return float(mouth_width / eye_span)

    def _estimate_head_pose(
        self,
        frame: np.ndarray,
        metrics: FaceLandmarkMetrics,
    ) -> Tuple[float, float, float]:
        image_points = np.array(
            [
                metrics.nose_tip,
                metrics.chin,
                metrics.left_eye_outer,
                metrics.right_eye_outer,
                metrics.mouth_left,
                metrics.mouth_right,
            ],
            dtype=np.float64,
        )
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1),
            ],
            dtype=np.float64,
        )
        focal_length = float(frame.shape[1])
        center = (frame.shape[1] / 2.0, frame.shape[0] / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            ok, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                return 0.0, 0.0, 0.0
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
            pitch, yaw, roll = [float(value) for value in euler_angles.flatten()]
            return yaw, pitch, roll
        except Exception as exc:  # pragma: no cover - OpenCV numerical errors
            logger.debug("Head pose estimation failed: %s", exc)
            return 0.0, 0.0, 0.0
