"""
=============================================================================
  SmartAttend Database Manager
=============================================================================
  Thread-safe SQLite wrapper using per-thread connections.

  SCHEMA
  ─────────────────────────────────────────────────────────────────────────
  students
    • student_id   TEXT PK  e.g. "STU001"
    • name         TEXT
    • department   TEXT
    • year         INTEGER  (1–4)
    • email        TEXT
    • phone        TEXT
    • photo_path   TEXT     (relative path to representative photo)
    • registered_at TEXT

  attendance
    • id           INTEGER PK AUTOINCREMENT
    • student_id   TEXT FK → students.student_id
    • name         TEXT     (denormalised for quick reads)
    • date         TEXT     "YYYY-MM-DD"
    • time         TEXT     "HH:MM:SS"
    • confidence   REAL     (0.0–1.0)
    UNIQUE (student_id, date)   ← prevents duplicate entries per day

  SECURITY NOTES
  ─────────────────────────────────────────────────────────────────────────
  • All queries use parameterised statements — no string interpolation →
    protects against SQL injection.
  • File permissions: chmod 600 smartattend.db (set in setup script).
  • No plain-text passwords stored (admin auth handled at app level).
=============================================================================
"""

import sqlite3
import threading
import logging
from pathlib import Path
from typing import List, Optional, Dict
from datetime import date, timedelta

logger = logging.getLogger("smartattend.db")

DB_PATH = Path("smartattend.db")


class DBManager:
    """Thread-safe SQLite database manager."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._local  = threading.local()
        self._init_db()

    # ── Connection ────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")  # better concurrency
            self._local.conn = conn
        return self._local.conn

    def _cursor(self) -> sqlite3.Cursor:
        return self._conn().cursor()

    def _commit(self):
        self._conn().commit()

    # ── Schema ────────────────────────────────────────────────────────────

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript("""
            PRAGMA foreign_keys = ON;
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS students (
                student_id    TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                department    TEXT DEFAULT '',
                year          INTEGER DEFAULT 1,
                email         TEXT DEFAULT '',
                phone         TEXT DEFAULT '',
                photo_path    TEXT DEFAULT '',
                registered_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  TEXT NOT NULL REFERENCES students(student_id),
                name        TEXT NOT NULL,
                date        TEXT NOT NULL,
                time        TEXT NOT NULL,
                confidence  REAL DEFAULT 0,
                UNIQUE (student_id, date)
            );

            CREATE INDEX IF NOT EXISTS idx_att_date       ON attendance(date);
            CREATE INDEX IF NOT EXISTS idx_att_student    ON attendance(student_id);
            CREATE INDEX IF NOT EXISTS idx_att_date_name  ON attendance(date, name);
        """)
        conn.commit()
        conn.close()
        logger.info("Database initialised at %s", self.db_path)

    # ── Students ──────────────────────────────────────────────────────────

    def insert_student(self, student_id: str, name: str, department: str = "",
                       year: int = 1, email: str = "", phone: str = "",
                       photo_path: str = "") -> Optional[Dict]:
        from datetime import datetime
        try:
            cur = self._cursor()
            cur.execute("""
                INSERT INTO students
                    (student_id, name, department, year, email, phone,
                     photo_path, registered_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (student_id, name, department, year, email, phone,
                  photo_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            self._commit()
            return self.get_student(student_id)
        except sqlite3.IntegrityError:
            logger.error("Student %s already exists", student_id)
            return None

    def update_student(self, student_id: str, **kwargs) -> bool:
        allowed = {"name", "department", "year", "email", "phone", "photo_path"}
        fields  = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values     = list(fields.values()) + [student_id]
        self._cursor().execute(
            f"UPDATE students SET {set_clause} WHERE student_id = ?", values
        )
        self._commit()
        return True

    def delete_student(self, student_id: str) -> bool:
        cur = self._cursor()
        cur.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        cur.execute("DELETE FROM students    WHERE student_id = ?", (student_id,))
        self._commit()
        return cur.rowcount > 0

    def get_student(self, student_id: str) -> Optional[Dict]:
        cur = self._cursor()
        cur.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_all_students(self) -> List[Dict]:
        cur = self._cursor()
        cur.execute("SELECT * FROM students ORDER BY name")
        return [dict(r) for r in cur.fetchall()]

    def count_students(self) -> int:
        cur = self._cursor()
        cur.execute("SELECT COUNT(*) FROM students")
        return cur.fetchone()[0]

    def search_students(self, query: str) -> List[Dict]:
        q = f"%{query}%"
        cur = self._cursor()
        cur.execute("""
            SELECT * FROM students
            WHERE name LIKE ? OR student_id LIKE ? OR department LIKE ?
            ORDER BY name
        """, (q, q, q))
        return [dict(r) for r in cur.fetchall()]

    def student_exists(self, student_id: str) -> bool:
        cur = self._cursor()
        cur.execute("SELECT 1 FROM students WHERE student_id = ?", (student_id,))
        return cur.fetchone() is not None

    # ── Attendance ────────────────────────────────────────────────────────

    def insert_attendance(self, student_id: str, name: str,
                          date_str: str, time_str: str,
                          confidence: float = 0.0) -> Optional[Dict]:
        try:
            cur = self._cursor()
            cur.execute("""
                INSERT OR IGNORE INTO attendance
                    (student_id, name, date, time, confidence)
                VALUES (?,?,?,?,?)
            """, (student_id, name, date_str, time_str, confidence))
            self._commit()
            if cur.rowcount == 0:
                return None     # duplicate — ignored
            return {
                "id":         cur.lastrowid,
                "student_id": student_id,
                "name":       name,
                "date":       date_str,
                "time":       time_str,
                "confidence": confidence,
            }
        except Exception as exc:
            logger.error("Attendance insert error: %s", exc)
            return None

    def get_today_record(self, student_id: str) -> Optional[Dict]:
        today = date.today().strftime("%Y-%m-%d")
        cur   = self._cursor()
        cur.execute("""
            SELECT * FROM attendance
            WHERE student_id = ? AND date = ?
        """, (student_id, today))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_attendance_by_date(self, date_str: str) -> List[Dict]:
        cur = self._cursor()
        cur.execute("""
            SELECT a.*, s.department, s.year
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE a.date = ?
            ORDER BY a.time DESC
        """, (date_str,))
        return [dict(r) for r in cur.fetchall()]

    def get_attendance_range(self, start: str, end: str) -> List[Dict]:
        cur = self._cursor()
        cur.execute("""
            SELECT a.*, s.department, s.year
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE a.date BETWEEN ? AND ?
            ORDER BY a.date DESC, a.time DESC
        """, (start, end))
        return [dict(r) for r in cur.fetchall()]

    def get_student_attendance(self, student_id: str) -> List[Dict]:
        cur = self._cursor()
        cur.execute("""
            SELECT * FROM attendance
            WHERE student_id = ?
            ORDER BY date DESC
        """, (student_id,))
        return [dict(r) for r in cur.fetchall()]

    def get_weekly_trend(self) -> List[Dict]:
        """Last 7 days — date + count of present students."""
        today = date.today()
        rows  = []
        cur   = self._cursor()
        for i in range(6, -1, -1):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            cur.execute(
                "SELECT COUNT(*) FROM attendance WHERE date = ?", (d,)
            )
            rows.append({"date": d, "count": cur.fetchone()[0]})
        return rows

    def get_dept_breakdown(self, date_str: str) -> List[Dict]:
        """Count present students per department for a given date."""
        cur = self._cursor()
        cur.execute("""
            SELECT s.department, COUNT(*) as present
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE a.date = ?
            GROUP BY s.department
            ORDER BY present DESC
        """, (date_str,))
        return [dict(r) for r in cur.fetchall()]

    def delete_attendance(self, attendance_id: int) -> bool:
        cur = self._cursor()
        cur.execute("DELETE FROM attendance WHERE id = ?", (attendance_id,))
        self._commit()
        return cur.rowcount > 0
