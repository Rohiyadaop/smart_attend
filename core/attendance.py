"""
=============================================================================
  SmartAttend Attendance Manager
=============================================================================
  Central business logic for marking, querying, and exporting attendance.

  DUPLICATE PREVENTION
  ─────────────────────────────────────────────────────────────────────────
  Each recognition event passes through mark_attendance().
  The DB has a UNIQUE constraint on (student_id, date), so any duplicate
  INSERT silently fails (INSERT OR IGNORE).  Additionally, we maintain an
  in-memory set `_today_marked` for O(1) lookups before hitting the DB
  — this prevents hammering the database for every camera frame.
=============================================================================
"""

import csv
import os
import io
import logging
from datetime import datetime, date
from typing import List, Optional, Dict
from pathlib import Path

from database.db_manager import DBManager

logger = logging.getLogger("smartattend.attendance")

CONFIDENCE_THRESHOLD = float(os.environ.get("ATTEND_MIN_CONF", "0.40"))


class AttendanceManager:
    """Manages all attendance operations."""

    def __init__(self, db: DBManager):
        self.db            = db
        self._today_marked = set()          # student_ids already marked today
        self._last_refresh = date.min       # last time we refreshed the set
        self._refresh_today_set()

    # ── Mark Attendance ───────────────────────────────────────────────────

    def mark_attendance(self, student_id: str, name: str,
                        confidence: float) -> Dict:
        """
        Attempt to mark attendance for a recognised student.

        Returns dict with:
            status   : "marked" | "duplicate" | "low_confidence" | "unknown"
            message  : human-readable string
            record   : the attendance row (if marked or duplicate)
        """
        # Refresh the in-memory set if date has changed (midnight rollover)
        today = date.today()
        if today != self._last_refresh:
            self._refresh_today_set()
            self._last_refresh = today

        # Unknown faces
        if not student_id:
            return {"status": "unknown", "message": "Unknown face", "record": None}

        # Low confidence
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "status":  "low_confidence",
                "message": f"Confidence {confidence:.0%} below threshold",
                "record":  None,
            }

        # Duplicate check (fast in-memory)
        if student_id in self._today_marked:
            record = self.db.get_today_record(student_id)
            return {
                "status":  "duplicate",
                "message": f"{name} already marked today",
                "record":  record,
            }

        # Mark in DB
        now = datetime.now()
        record = self.db.insert_attendance(
            student_id = student_id,
            name       = name,
            date_str   = now.strftime("%Y-%m-%d"),
            time_str   = now.strftime("%H:%M:%S"),
            confidence = round(confidence, 4),
        )

        if record:
            self._today_marked.add(student_id)
            logger.info("Attendance marked: %s (%s) @ %s — confidence %.0f%%",
                        name, student_id, now.strftime("%H:%M:%S"),
                        confidence * 100)
            return {"status": "marked", "message": f"{name} marked present", "record": record}

        # INSERT OR IGNORE fired — already in DB (edge case)
        return {
            "status":  "duplicate",
            "message": f"{name} already marked today",
            "record":  self.db.get_today_record(student_id),
        }

    # ── Query ─────────────────────────────────────────────────────────────

    def get_today_attendance(self) -> List[Dict]:
        return self.db.get_attendance_by_date(date.today().strftime("%Y-%m-%d"))

    def get_attendance_by_date(self, date_str: str) -> List[Dict]:
        return self.db.get_attendance_by_date(date_str)

    def get_attendance_range(self, start: str, end: str) -> List[Dict]:
        return self.db.get_attendance_range(start, end)

    def get_student_history(self, student_id: str) -> List[Dict]:
        return self.db.get_student_attendance(student_id)

    def get_summary(self, date_str: Optional[str] = None) -> Dict:
        """Return KPI summary for a given date (defaults to today)."""
        if date_str is None:
            date_str = date.today().strftime("%Y-%m-%d")

        total_students = self.db.count_students()
        present_today  = len(self.db.get_attendance_by_date(date_str))
        pct = (present_today / total_students * 100) if total_students else 0

        return {
            "date":             date_str,
            "total_students":   total_students,
            "present":          present_today,
            "absent":           total_students - present_today,
            "attendance_pct":   round(pct, 1),
        }

    def get_weekly_trend(self) -> List[Dict]:
        """Last 7 days attendance count — for trend chart."""
        return self.db.get_weekly_trend()

    def search_student(self, query: str) -> List[Dict]:
        return self.db.search_students(query)

    # ── Export ────────────────────────────────────────────────────────────

    def export_csv(self, date_str: Optional[str] = None) -> str:
        """Return CSV string of attendance for a given date."""
        records = (
            self.get_today_attendance()
            if date_str is None
            else self.get_attendance_by_date(date_str)
        )
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=["student_id", "name", "date", "time", "confidence"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)
        return buf.getvalue()

    def export_range_csv(self, start: str, end: str) -> str:
        """Export a date range as CSV."""
        records = self.get_attendance_range(start, end)
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=["student_id", "name", "date", "time", "confidence"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)
        return buf.getvalue()

    # ── Internal ──────────────────────────────────────────────────────────

    def _refresh_today_set(self):
        today_str = date.today().strftime("%Y-%m-%d")
        records   = self.db.get_attendance_by_date(today_str)
        self._today_marked = {r["student_id"] for r in records}
        self._last_refresh = date.today()
        logger.debug("Today-set refreshed: %d students already marked",
                     len(self._today_marked))
