from __future__ import annotations


def serialize_face(face):
    return {
        "name": face.name,
        "student_id": face.student_id,
        "confidence": face.confidence,
        "distance": face.distance,
        "similarity": face.similarity,
        "matched_index": face.matched_index,
        "bounding_box": list(face.bounding_box),
        "is_known": face.is_known,
        "status": face.status,
        "status_text": face.status_text,
        "challenge_text": face.challenge_text,
        "live_verified": face.live_verified,
        "liveness_score": face.liveness_score,
        "spoof_score": face.spoof_score,
        "spoof_detected": face.spoof_detected,
        "spoof_reasons": list(face.spoof_reasons),
        "blink_count": face.blink_count,
        "left_ear": face.left_ear,
        "right_ear": face.right_ear,
        "yaw": face.yaw,
        "pitch": face.pitch,
        "roll": face.roll,
    }
