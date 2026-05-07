"""
=============================================================================
  SmartAttend Attendance Manager
=============================================================================
  Central business logic for class-session attendance.
=============================================================================
"""

from __future__ import annotations

import csv
import io
import logging
import os
from datetime import date, datetime
from typing import Dict, List, Optional

from database.db_manager import DBManager

logger = logging.getLogger("smartattend.attendance")

CONFIDENCE_THRESHOLD = float(
    os.environ.get("ATTEND_MIN_CONF", os.environ.get("FACE_MIN_CONF", "0.45"))
)


class AttendanceManager:
    """Manages all attendance operations."""

    def __init__(self, db: DBManager):
        self.db = db
        self._session_marked: Dict[int, set[str]] = {}
        self._cache_day = date.today()

    # Mark attendance ------------------------------------------------------

    def mark_attendance(self, student_id: str, name: str,
                        confidence: float,
                        liveness_score: float = 0.0,
                        active_session: Optional[Dict] = None) -> Dict:
        """
        Attempt to mark attendance for a recognised student.

        Returns dict with:
            status  : marked | duplicate | low_confidence | unknown
                      | no_active_session | not_in_session
            message : human-readable text
            record  : the attendance row when available
        """
        self._reset_cache_if_needed()

        if not student_id:
            logger.info("Attendance rejected: unknown face")
            return {"status": "unknown", "message": "Unknown face", "record": None}

        if confidence < CONFIDENCE_THRESHOLD:
            logger.info(
                "Attendance rejected: confidence %.3f below threshold %.3f",
                confidence,
                CONFIDENCE_THRESHOLD,
            )
            return {
                "status": "low_confidence",
                "message": f"Confidence {confidence:.0%} below threshold",
                "record": None,
            }

        if not active_session or not active_session.get("id"):
            logger.info("Attendance rejected for %s (%s): no active session", name, student_id)
            return {
                "status": "no_active_session",
                "message": "Start the running class session before marking attendance",
                "record": None,
            }

        student = self.db.get_student(student_id)
        if not student:
            logger.info("Attendance rejected for %s (%s): student not found", name, student_id)
            return {"status": "unknown", "message": "Student not found", "record": None}

        if (
            student.get("department") != active_session.get("department")
            or int(student.get("year") or 0) != int(active_session.get("year") or 0)
        ):
            logger.info(
                "Attendance rejected for %s (%s): student batch=%s year=%s session batch=%s year=%s",
                name,
                student_id,
                student.get("department"),
                student.get("year"),
                active_session.get("department"),
                active_session.get("year"),
            )
            return {
                "status": "not_in_session",
                "message": f"{name} does not belong to {active_session['batch_label']}",
                "record": None,
            }

        session_id = int(active_session["id"])
        marked_ids = self._get_marked_set(session_id)
        if student_id in marked_ids:
            record = self.db.get_session_record(session_id, student_id)
            logger.info(
                "Attendance duplicate: %s (%s) already marked in session %s",
                name,
                student_id,
                session_id,
            )
            return {
                "status": "duplicate",
                "message": f"{name} already marked in this class",
                "record": record,
            }

        now = datetime.now()
        record = self.db.insert_attendance(
            student_id=student_id,
            session_id=session_id,
            name=name,
            date_str=now.strftime("%Y-%m-%d"),
            time_str=now.strftime("%H:%M:%S"),
            confidence=round(confidence, 4),
            liveness_score=round(liveness_score, 4),
        )

        if record:
            marked_ids.add(student_id)
            logger.info(
                "Attendance marked: %s (%s) in session %s @ %s - confidence %.0f%%",
                name,
                student_id,
                session_id,
                now.strftime("%H:%M:%S"),
                confidence * 100,
            )
            return {
                "status": "marked",
                "message": f"{name} marked present in {active_session['display_label']}",
                "record": record,
            }

        return {
            "status": "duplicate",
            "message": f"{name} already marked in this class",
            "record": self.db.get_session_record(session_id, student_id),
        }

    # Queries --------------------------------------------------------------

    def get_today_attendance(self, session_id: Optional[int] = None) -> List[Dict]:
        return self.db.get_attendance_by_date(
            date.today().strftime("%Y-%m-%d"),
            session_id=session_id,
        )

    def get_attendance_by_date(self, date_str: str,
                               session_id: Optional[int] = None) -> List[Dict]:
        return self.db.get_attendance_by_date(date_str, session_id=session_id)

    def get_attendance_range(self, start: str, end: str) -> List[Dict]:
        return self.db.get_attendance_range(start, end)

    def get_student_history(self, student_id: str) -> List[Dict]:
        return self.db.get_student_attendance(student_id)

    def get_summary(self, date_str: Optional[str] = None,
                    session_id: Optional[int] = None) -> Dict:
        if session_id:
            return self._get_session_summary(session_id)

        if date_str is None:
            date_str = date.today().strftime("%Y-%m-%d")

        total_students = self.db.count_students()
        present_today = self.db.count_present_students_by_date(date_str)
        pct = (present_today / total_students * 100) if total_students else 0

        return {
            "scope": "day",
            "date": date_str,
            "total_students": total_students,
            "present": present_today,
            "absent": max(total_students - present_today, 0),
            "attendance_pct": round(pct, 1),
        }

    def get_weekly_trend(self) -> List[Dict]:
        return self.db.get_weekly_trend()

    def search_student(self, query: str) -> List[Dict]:
        return self.db.search_students(query)

    # Export ---------------------------------------------------------------

    def export_csv(self, date_str: Optional[str] = None,
                   session_id: Optional[int] = None) -> str:
        records = (
            self.get_today_attendance(session_id=session_id)
            if date_str is None
            else self.get_attendance_by_date(date_str, session_id=session_id)
        )
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=[
                "student_id",
                "name",
                "department",
                "year",
                "date",
                "time",
                "room",
                "subject",
                "session_label",
                "confidence",
                "liveness_score",
                "snapshot_path",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)
        return buf.getvalue()

    def export_range_csv(self, start: str, end: str) -> str:
        records = self.get_attendance_range(start, end)
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=[
                "student_id",
                "name",
                "department",
                "year",
                "date",
                "time",
                "room",
                "subject",
                "session_label",
                "confidence",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)
        return buf.getvalue()

    def export_presence_interval_csv(self, date_str: str,
                                     session_id: Optional[int] = None,
                                     interval_minutes: int = 60) -> str:
        interval_records = self.db.get_presence_interval_report(
            date_str,
            session_id=session_id,
            interval_minutes=interval_minutes,
        )
        records = [
            {
                "date": item.get("date"),
                "session_label": item.get("session_label"),
                "interval_start": item.get("interval_start"),
                "interval_end": item.get("interval_end"),
                "present_students": item.get("student_names", ""),
            }
            for item in interval_records
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=[
                "date",
                "session_label",
                "interval_start",
                "interval_end",
                "present_students",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)
        return buf.getvalue()

    # Internal -------------------------------------------------------------

    def clear_session_cache(self, session_id: Optional[int] = None):
        if session_id is None:
            self._session_marked.clear()
            return
        self._session_marked.pop(int(session_id), None)

    def _get_session_summary(self, session_id: int) -> Dict:
        session = self.db.get_session(session_id)
        if not session:
            return {
                "scope": "session",
                "date": date.today().strftime("%Y-%m-%d"),
                "total_students": 0,
                "present": 0,
                "absent": 0,
                "attendance_pct": 0.0,
                "session": None,
            }

        total_students = self.db.count_students_for_batch(
            session["department"],
            session["year"],
        )
        present = self.db.count_attendance_for_session(session_id)
        pct = (present / total_students * 100) if total_students else 0
        return {
            "scope": "session",
            "date": session["session_date"],
            "total_students": total_students,
            "present": present,
            "absent": max(total_students - present, 0),
            "attendance_pct": round(pct, 1),
            "session": session,
        }

    def _reset_cache_if_needed(self):
        today = date.today()
        if today != self._cache_day:
            self._session_marked.clear()
            self._cache_day = today

    def _get_marked_set(self, session_id: int) -> set[str]:
        if session_id not in self._session_marked:
            records = self.db.get_session_attendance(session_id)
            self._session_marked[session_id] = {r["student_id"] for r in records}
        return self._session_marked[session_id]
