"""
=============================================================================
  SmartAttend Database Manager
=============================================================================
  Thread-safe SQLite wrapper using per-thread connections.
=============================================================================
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from core.college import (
    BTECH_BRANCHES,
    BRANCH_CODES,
    PROGRAM_NAME,
    format_batch_label,
    format_session_label,
    get_branch_name,
    normalize_branch,
)

logger = logging.getLogger("smartattend.db")

DB_PATH = Path("smartattend.db")


class DBManager:
    """Thread-safe SQLite database manager."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    # Connection -----------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA busy_timeout = 5000")
            self._local.conn = conn
        return self._local.conn

    def _cursor(self) -> sqlite3.Cursor:
        return self._conn().cursor()

    def _commit(self):
        self._conn().commit()

    # Schema ---------------------------------------------------------------

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id    TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                department    TEXT DEFAULT '',
                year          INTEGER DEFAULT 1,
                email         TEXT DEFAULT '',
                phone         TEXT DEFAULT '',
                photo_path    TEXT DEFAULT '',
                registered_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS class_sessions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                program      TEXT NOT NULL DEFAULT 'B.Tech',
                department   TEXT NOT NULL,
                year         INTEGER NOT NULL,
                room         TEXT NOT NULL,
                subject      TEXT DEFAULT '',
                section      TEXT DEFAULT '',
                session_date TEXT NOT NULL,
                started_at   TEXT NOT NULL,
                ended_at     TEXT DEFAULT '',
                status       TEXT NOT NULL DEFAULT 'active'
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_date ON class_sessions(session_date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_status ON class_sessions(status)"
        )

        self._ensure_attendance_schema(conn)
        self._ensure_presence_schema(conn)
        self._normalize_existing_departments(conn)

        conn.commit()
        conn.close()
        logger.info("Database initialised at %s", self.db_path)

    def _ensure_attendance_schema(self, conn: sqlite3.Connection):
        if not self._table_exists(conn, "attendance"):
            self._create_attendance_table(conn, "attendance")
            self._create_attendance_indexes(conn)
            return

        columns = set(self._table_columns(conn, "attendance"))
        required = {"id", "student_id", "name", "date", "time", "confidence", "session_id"}
        if required.issubset(columns):
            self._create_attendance_indexes(conn)
            return

        logger.info("Migrating legacy attendance table to class-session schema")
        self._create_attendance_table(conn, "attendance_new")
        conn.execute("""
            INSERT INTO attendance_new (id, student_id, name, date, time, confidence)
            SELECT id, student_id, name, date, time, confidence
            FROM attendance
        """)
        conn.execute("DROP TABLE attendance")
        conn.execute("ALTER TABLE attendance_new RENAME TO attendance")
        self._create_attendance_indexes(conn)

    def _create_attendance_table(self, conn: sqlite3.Connection, table_name: str):
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  TEXT NOT NULL REFERENCES students(student_id),
                session_id  INTEGER REFERENCES class_sessions(id) ON DELETE SET NULL,
                name        TEXT NOT NULL,
                date        TEXT NOT NULL,
                time        TEXT NOT NULL,
                confidence  REAL DEFAULT 0,
                UNIQUE (student_id, session_id)
            )
        """)

    def _create_attendance_indexes(self, conn: sqlite3.Connection):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_att_date ON attendance(date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_att_student ON attendance(student_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_att_session ON attendance(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_att_date_session ON attendance(date, session_id)"
        )

    def _ensure_presence_schema(self, conn: sqlite3.Connection):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS student_presence (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id   TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
                session_id   INTEGER NOT NULL REFERENCES class_sessions(id) ON DELETE CASCADE,
                bucket_date  TEXT NOT NULL,
                bucket_time  TEXT NOT NULL,
                bucket_start TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                seen_count   INTEGER NOT NULL DEFAULT 1,
                UNIQUE (student_id, bucket_start)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_presence_student_date
            ON student_presence(student_id, bucket_date)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_presence_session
            ON student_presence(session_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_presence_bucket
            ON student_presence(bucket_start)
        """)

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return [row[1] for row in rows]

    def _normalize_existing_departments(self, conn: sqlite3.Connection):
        rows = conn.execute(
            "SELECT student_id, department FROM students"
        ).fetchall()
        for row in rows:
            current = row["department"] or ""
            normalised = normalize_branch(current)
            if current != normalised and normalised in BRANCH_CODES:
                conn.execute(
                    "UPDATE students SET department = ? WHERE student_id = ?",
                    (normalised, row["student_id"]),
                )

    # Students -------------------------------------------------------------

    def insert_student(self, student_id: str, name: str, department: str = "",
                       year: int = 1, email: str = "", phone: str = "",
                       photo_path: str = "") -> Optional[Dict]:
        department = normalize_branch(department)
        try:
            cur = self._cursor()
            cur.execute("""
                INSERT INTO students
                    (student_id, name, department, year, email, phone,
                     photo_path, registered_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                student_id,
                name,
                department,
                int(year),
                email,
                phone,
                photo_path,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ))
            self._commit()
            return self.get_student(student_id)
        except sqlite3.IntegrityError:
            logger.error("Student %s already exists", student_id)
            return None

    def update_student(self, student_id: str, **kwargs) -> bool:
        allowed = {"name", "department", "year", "email", "phone", "photo_path"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
        if "department" in fields:
            fields["department"] = normalize_branch(fields["department"])
        if "year" in fields:
            fields["year"] = int(fields["year"])
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [student_id]
        self._cursor().execute(
            f"UPDATE students SET {set_clause} WHERE student_id = ?",
            values,
        )
        self._commit()
        return True

    def delete_student(self, student_id: str) -> bool:
        cur = self._cursor()
        cur.execute("DELETE FROM student_presence WHERE student_id = ?", (student_id,))
        cur.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        cur.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
        self._commit()
        return cur.rowcount > 0

    def get_student(self, student_id: str) -> Optional[Dict]:
        cur = self._cursor()
        cur.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
        row = cur.fetchone()
        return self._decorate_student(dict(row)) if row else None

    def get_all_students(self) -> List[Dict]:
        cur = self._cursor()
        cur.execute("SELECT * FROM students ORDER BY year, department, name")
        return [self._decorate_student(dict(r)) for r in cur.fetchall()]

    def count_students(self) -> int:
        cur = self._cursor()
        cur.execute("SELECT COUNT(*) FROM students")
        return cur.fetchone()[0]

    def count_students_for_batch(self, department: str, year: int) -> int:
        cur = self._cursor()
        cur.execute("""
            SELECT COUNT(*)
            FROM students
            WHERE department = ? AND year = ?
        """, (normalize_branch(department), int(year)))
        return cur.fetchone()[0]

    def count_unassigned_students(self) -> int:
        cur = self._cursor()
        cur.execute("""
            SELECT COUNT(*)
            FROM students
            WHERE COALESCE(TRIM(department), '') = ''
        """)
        return cur.fetchone()[0]

    def search_students(self, query: str) -> List[Dict]:
        q = f"%{query}%"
        cur = self._cursor()
        cur.execute("""
            SELECT * FROM students
            WHERE name LIKE ? OR student_id LIKE ? OR department LIKE ?
            ORDER BY year, department, name
        """, (q, q, q))
        return [self._decorate_student(dict(r)) for r in cur.fetchall()]

    def student_exists(self, student_id: str) -> bool:
        cur = self._cursor()
        cur.execute("SELECT 1 FROM students WHERE student_id = ?", (student_id,))
        return cur.fetchone() is not None

    def get_btech_batch_matrix(self) -> List[Dict]:
        cur = self._cursor()
        cur.execute("""
            SELECT year, department, COUNT(*) AS total
            FROM students
            WHERE COALESCE(TRIM(department), '') != ''
            GROUP BY year, department
        """)
        lookup = {
            (int(row["year"]), normalize_branch(row["department"])): row["total"]
            for row in cur.fetchall()
        }

        rows = []
        for year in [1, 2, 3, 4]:
            branches = []
            total = 0
            for branch in BTECH_BRANCHES:
                count = lookup.get((year, branch["code"]), 0)
                total += count
                branches.append({
                    "code": branch["code"],
                    "name": branch["name"],
                    "count": count,
                })
            rows.append({"year": year, "branches": branches, "total": total})
        return rows

    # Class sessions -------------------------------------------------------

    def start_class_session(self, program: str, department: str, year: int,
                            room: str, subject: str = "",
                            section: str = "") -> Dict:
        program = (program or PROGRAM_NAME).strip() or PROGRAM_NAME
        department = normalize_branch(department)
        room = (room or "").strip().upper()
        subject = (subject or "").strip()
        section = (section or "").strip().upper()

        active = self.get_active_session()
        if active:
            same_session = (
                active["program"] == program
                and active["department"] == department
                and int(active["year"]) == int(year)
                and active["room"] == room
                and (active.get("subject") or "") == subject
                and (active.get("section") or "") == section
            )
            if same_session:
                return active
            self.stop_class_session(active["id"])

        now = datetime.now()
        cur = self._cursor()
        cur.execute("""
            INSERT INTO class_sessions
                (program, department, year, room, subject, section,
                 session_date, started_at, ended_at, status)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            program,
            department,
            int(year),
            room,
            subject,
            section,
            now.strftime("%Y-%m-%d"),
            now.strftime("%Y-%m-%d %H:%M:%S"),
            "",
            "active",
        ))
        self._commit()
        return self.get_session(cur.lastrowid)

    def get_session(self, session_id: int) -> Optional[Dict]:
        cur = self._cursor()
        cur.execute("""
            SELECT cs.*, COUNT(a.id) AS attendance_count
            FROM class_sessions cs
            LEFT JOIN attendance a ON a.session_id = cs.id
            WHERE cs.id = ?
            GROUP BY cs.id
        """, (session_id,))
        row = cur.fetchone()
        return self._decorate_session(dict(row)) if row else None

    def get_active_session(self) -> Optional[Dict]:
        cur = self._cursor()
        cur.execute("""
            SELECT cs.*, COUNT(a.id) AS attendance_count
            FROM class_sessions cs
            LEFT JOIN attendance a ON a.session_id = cs.id
            WHERE cs.status = 'active'
            GROUP BY cs.id
            ORDER BY cs.started_at DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        return self._decorate_session(dict(row)) if row else None

    def stop_class_session(self, session_id: Optional[int] = None) -> Optional[Dict]:
        session = self.get_session(session_id) if session_id else self.get_active_session()
        if not session:
            return None
        self._cursor().execute("""
            UPDATE class_sessions
            SET status = 'completed', ended_at = ?
            WHERE id = ?
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session["id"]))
        self._commit()
        return self.get_session(session["id"])

    def get_sessions_by_date(self, date_str: str) -> List[Dict]:
        cur = self._cursor()
        cur.execute("""
            SELECT cs.*, COUNT(a.id) AS attendance_count
            FROM class_sessions cs
            LEFT JOIN attendance a ON a.session_id = cs.id
            WHERE cs.session_date = ?
            GROUP BY cs.id
            ORDER BY CASE WHEN cs.status = 'active' THEN 0 ELSE 1 END,
                     cs.started_at DESC
        """, (date_str,))
        return [self._decorate_session(dict(r)) for r in cur.fetchall()]

    def log_student_presence(self, student_id: str, session_id: int,
                             seen_at: Optional[datetime] = None) -> bool:
        if not student_id or not session_id:
            return False

        seen_at = seen_at or datetime.now()
        bucket_start_dt = seen_at.replace(
            minute=(seen_at.minute // 5) * 5,
            second=0,
            microsecond=0,
        )
        bucket_start = bucket_start_dt.strftime("%Y-%m-%d %H:%M:%S")
        bucket_date = bucket_start_dt.strftime("%Y-%m-%d")
        bucket_time = bucket_start_dt.strftime("%H:%M")
        last_seen_at = seen_at.strftime("%Y-%m-%d %H:%M:%S")

        try:
            cur = self._cursor()
            cur.execute("""
                INSERT OR IGNORE INTO student_presence
                    (student_id, session_id, bucket_date, bucket_time,
                     bucket_start, last_seen_at, seen_count)
                VALUES (?,?,?,?,?,?,1)
            """, (
                student_id,
                int(session_id),
                bucket_date,
                bucket_time,
                bucket_start,
                last_seen_at,
            ))
            if cur.rowcount == 0:
                return False
            self._commit()
            return True
        except Exception as exc:
            logger.error("Presence log error for %s: %s", student_id, exc)
            return False

    # Attendance -----------------------------------------------------------

    def insert_attendance(self, student_id: str, name: str,
                          date_str: str, time_str: str,
                          confidence: float = 0.0,
                          session_id: Optional[int] = None) -> Optional[Dict]:
        try:
            cur = self._cursor()
            cur.execute("""
                INSERT OR IGNORE INTO attendance
                    (student_id, session_id, name, date, time, confidence)
                VALUES (?,?,?,?,?,?)
            """, (student_id, session_id, name, date_str, time_str, confidence))
            self._commit()
            if cur.rowcount == 0:
                return None
            return self.get_attendance_record(cur.lastrowid)
        except Exception as exc:
            logger.error("Attendance insert error: %s", exc)
            return None

    def get_attendance_record(self, attendance_id: int) -> Optional[Dict]:
        cur = self._cursor()
        cur.execute("""
            SELECT a.*, s.department, s.year,
                   cs.program, cs.department AS session_department,
                   cs.year AS session_year, cs.room, cs.subject, cs.section,
                   cs.started_at, cs.ended_at, cs.status AS session_status
            FROM attendance a
            LEFT JOIN students s ON a.student_id = s.student_id
            LEFT JOIN class_sessions cs ON a.session_id = cs.id
            WHERE a.id = ?
        """, (attendance_id,))
        row = cur.fetchone()
        return self._decorate_attendance(dict(row)) if row else None

    def get_today_record(self, student_id: str) -> Optional[Dict]:
        today = date.today().strftime("%Y-%m-%d")
        cur = self._cursor()
        cur.execute("""
            SELECT a.id
            FROM attendance a
            WHERE a.student_id = ? AND a.date = ?
            ORDER BY a.time DESC
            LIMIT 1
        """, (student_id, today))
        row = cur.fetchone()
        return self.get_attendance_record(row["id"]) if row else None

    def get_session_record(self, session_id: int, student_id: str) -> Optional[Dict]:
        cur = self._cursor()
        cur.execute("""
            SELECT a.id
            FROM attendance a
            WHERE a.session_id = ? AND a.student_id = ?
            LIMIT 1
        """, (session_id, student_id))
        row = cur.fetchone()
        return self.get_attendance_record(row["id"]) if row else None

    def get_attendance_by_date(self, date_str: str,
                               session_id: Optional[int] = None) -> List[Dict]:
        cur = self._cursor()
        if session_id:
            cur.execute("""
                SELECT a.*, s.department, s.year,
                       cs.program, cs.department AS session_department,
                       cs.year AS session_year, cs.room, cs.subject, cs.section,
                       cs.started_at, cs.ended_at, cs.status AS session_status
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                LEFT JOIN class_sessions cs ON a.session_id = cs.id
                WHERE a.date = ? AND a.session_id = ?
                ORDER BY a.time DESC
            """, (date_str, session_id))
        else:
            cur.execute("""
                SELECT a.*, s.department, s.year,
                       cs.program, cs.department AS session_department,
                       cs.year AS session_year, cs.room, cs.subject, cs.section,
                       cs.started_at, cs.ended_at, cs.status AS session_status
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                LEFT JOIN class_sessions cs ON a.session_id = cs.id
                WHERE a.date = ?
                ORDER BY a.time DESC
            """, (date_str,))
        return [self._decorate_attendance(dict(r)) for r in cur.fetchall()]

    def get_attendance_range(self, start: str, end: str) -> List[Dict]:
        cur = self._cursor()
        cur.execute("""
            SELECT a.*, s.department, s.year,
                   cs.program, cs.department AS session_department,
                   cs.year AS session_year, cs.room, cs.subject, cs.section,
                   cs.started_at, cs.ended_at, cs.status AS session_status
            FROM attendance a
            LEFT JOIN students s ON a.student_id = s.student_id
            LEFT JOIN class_sessions cs ON a.session_id = cs.id
            WHERE a.date BETWEEN ? AND ?
            ORDER BY a.date DESC, a.time DESC
        """, (start, end))
        return [self._decorate_attendance(dict(r)) for r in cur.fetchall()]

    def get_student_attendance(self, student_id: str) -> List[Dict]:
        cur = self._cursor()
        cur.execute("""
            SELECT a.*, s.department, s.year,
                   cs.program, cs.department AS session_department,
                   cs.year AS session_year, cs.room, cs.subject, cs.section,
                   cs.started_at, cs.ended_at, cs.status AS session_status
            FROM attendance a
            LEFT JOIN students s ON a.student_id = s.student_id
            LEFT JOIN class_sessions cs ON a.session_id = cs.id
            WHERE a.student_id = ?
            ORDER BY a.date DESC, a.time DESC
        """, (student_id,))
        return [self._decorate_attendance(dict(r)) for r in cur.fetchall()]

    def get_student_presence_by_date(self, student_id: str, date_str: str) -> List[Dict]:
        cur = self._cursor()
        cur.execute("""
            SELECT sp.*, cs.program, cs.department AS session_department,
                   cs.year AS session_year, cs.room, cs.subject, cs.section,
                   cs.started_at, cs.ended_at, cs.status AS session_status
            FROM student_presence sp
            LEFT JOIN class_sessions cs ON sp.session_id = cs.id
            WHERE sp.student_id = ? AND sp.bucket_date = ?
            ORDER BY sp.bucket_start ASC
        """, (student_id, date_str))
        return [self._decorate_presence(dict(r)) for r in cur.fetchall()]

    def get_student_presence_timeline(self, student_id: str, date_str: str) -> Dict:
        day_start = datetime.strptime(date_str, "%Y-%m-%d")
        entries = self.get_student_presence_by_date(student_id, date_str)
        entry_lookup = {entry["bucket_time"]: entry for entry in entries}

        labels: List[str] = []
        values: List[int] = []
        slots: List[Optional[Dict]] = []

        for i in range(288):
            slot_time = (day_start + timedelta(minutes=i * 5)).strftime("%H:%M")
            labels.append(slot_time)
            entry = entry_lookup.get(slot_time)
            values.append(1 if entry else 0)
            slots.append(entry if entry else None)

        total_windows = sum(values)
        total_detections = sum(int(entry.get("seen_count") or 0) for entry in entries)
        first_seen = entries[0]["last_seen_at"][11:19] if entries else None
        last_seen = entries[-1]["last_seen_at"][11:19] if entries else None

        return {
            "success": True,
            "student_id": student_id,
            "date": date_str,
            "interval_minutes": 5,
            "labels": labels,
            "values": values,
            "slots": slots,
            "entries": entries,
            "slots_present": total_windows,
            "total_detections": total_detections,
            "coverage_pct": round((total_windows / len(labels)) * 100, 1) if labels else 0.0,
            "first_seen": first_seen,
            "last_seen": last_seen,
        }

    def get_year_student_presence_overview(self, year: int, date_str: str) -> Dict:
        datetime.strptime(date_str, "%Y-%m-%d")
        year = int(year)

        cur = self._cursor()
        cur.execute("""
            SELECT s.student_id, s.name, s.department, s.year, s.registered_at,
                   COUNT(sp.id) AS slots_present,
                   COALESCE(SUM(sp.seen_count), 0) AS total_detections,
                   MIN(sp.last_seen_at) AS first_seen_at,
                   MAX(sp.last_seen_at) AS last_seen_at
            FROM students s
            LEFT JOIN student_presence sp
              ON sp.student_id = s.student_id
             AND sp.bucket_date = ?
            WHERE s.year = ?
            GROUP BY s.student_id, s.name, s.department, s.year, s.registered_at
            ORDER BY slots_present DESC, s.name ASC
        """, (date_str, year))

        students = []
        labels: List[str] = []
        values: List[int] = []

        for row in cur.fetchall():
            student = self._decorate_student(dict(row))
            slots_present = int(student.get("slots_present") or 0)
            total_detections = int(student.get("total_detections") or 0)
            first_seen_at = student.get("first_seen_at")
            last_seen_at = student.get("last_seen_at")
            student["slots_present"] = slots_present
            student["total_detections"] = total_detections
            student["coverage_pct"] = round((slots_present / 288) * 100, 1)
            student["first_seen"] = first_seen_at[11:19] if first_seen_at else None
            student["last_seen"] = last_seen_at[11:19] if last_seen_at else None
            students.append(student)
            labels.append(student["name"])
            values.append(slots_present)

        detected_students = sum(1 for student in students if student["slots_present"] > 0)
        total_slots = sum(student["slots_present"] for student in students)

        return {
            "success": True,
            "year": year,
            "date": date_str,
            "labels": labels,
            "values": values,
            "students": students,
            "summary": {
                "total_students": len(students),
                "detected_students": detected_students,
                "total_slots": total_slots,
                "avg_slots": round((total_slots / len(students)), 1) if students else 0.0,
            },
        }

    def get_presence_interval_report(self, date_str: str,
                                     session_id: Optional[int] = None,
                                     interval_minutes: int = 60) -> List[Dict]:
        day_start = datetime.strptime(date_str, "%Y-%m-%d")
        interval_minutes = max(int(interval_minutes or 60), 5)

        cur = self._cursor()
        params: List[object] = [date_str]
        session_filter = ""
        if session_id:
            session_filter = " AND sp.session_id = ?"
            params.append(int(session_id))

        cur.execute(f"""
            SELECT sp.student_id, sp.session_id, sp.bucket_start, sp.last_seen_at,
                   s.name, s.department, s.year,
                   cs.program, cs.department AS session_department,
                   cs.year AS session_year, cs.room, cs.subject, cs.section,
                   cs.started_at, cs.ended_at, cs.status AS session_status
            FROM student_presence sp
            JOIN students s ON sp.student_id = s.student_id
            LEFT JOIN class_sessions cs ON sp.session_id = cs.id
            WHERE sp.bucket_date = ?{session_filter}
            ORDER BY sp.session_id ASC, sp.bucket_start ASC, s.name ASC
        """, params)

        grouped: Dict[tuple, Dict] = {}
        for row in cur.fetchall():
            record = dict(row)
            bucket_dt = datetime.strptime(record["bucket_start"], "%Y-%m-%d %H:%M:%S")
            minutes_from_start = int((bucket_dt - day_start).total_seconds() // 60)
            interval_index = max(minutes_from_start // interval_minutes, 0)
            interval_start_dt = day_start + timedelta(minutes=interval_index * interval_minutes)
            interval_end_dt = interval_start_dt + timedelta(minutes=interval_minutes)
            key = (
                int(record["session_id"] or 0),
                interval_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            )

            if key not in grouped:
                session_department = normalize_branch(record.get("session_department", ""))
                session_year = int(record.get("session_year") or record.get("year") or 1)
                section = (record.get("section") or "").strip().upper()
                program = record.get("program") or PROGRAM_NAME
                grouped[key] = {
                    "date": date_str,
                    "session_id": int(record["session_id"] or 0),
                    "session_label": format_session_label(
                        program,
                        session_department,
                        session_year,
                        record.get("room", ""),
                        record.get("subject", ""),
                        section,
                    ) if record.get("session_id") else "Unassigned session",
                    "batch_label": format_batch_label(
                        session_department,
                        session_year,
                        section,
                    ) if session_department else "",
                    "room": (record.get("room") or "").strip().upper(),
                    "subject": (record.get("subject") or "").strip(),
                    "section": section,
                    "department": session_department,
                    "year": session_year,
                    "interval_start": interval_start_dt.strftime("%H:%M"),
                    "interval_end": interval_end_dt.strftime("%H:%M"),
                    "interval_minutes": interval_minutes,
                    "student_count": 0,
                    "student_ids": [],
                    "student_names": [],
                    "first_seen_at": record.get("last_seen_at", "")[11:19],
                    "last_seen_at": record.get("last_seen_at", "")[11:19],
                    "_seen_ids": set(),
                }

            bucket = grouped[key]
            student_id = str(record.get("student_id") or "").strip()
            student_name = str(record.get("name") or "").strip()
            if student_id and student_id not in bucket["_seen_ids"]:
                bucket["_seen_ids"].add(student_id)
                bucket["student_ids"].append(student_id)
                bucket["student_names"].append(student_name)
                bucket["student_count"] += 1

            seen_time = record.get("last_seen_at", "")[11:19]
            if seen_time:
                if not bucket["first_seen_at"] or seen_time < bucket["first_seen_at"]:
                    bucket["first_seen_at"] = seen_time
                if not bucket["last_seen_at"] or seen_time > bucket["last_seen_at"]:
                    bucket["last_seen_at"] = seen_time

        rows: List[Dict] = []
        for item in grouped.values():
            item["student_ids"] = ", ".join(item["student_ids"])
            item["student_names"] = ", ".join(item["student_names"])
            item.pop("_seen_ids", None)
            rows.append(item)

        rows.sort(key=lambda row: (
            row["session_label"],
            row["interval_start"],
            row["room"],
        ))
        return rows

    def get_session_attendance(self, session_id: int) -> List[Dict]:
        cur = self._cursor()
        cur.execute("""
            SELECT a.*, s.department, s.year,
                   cs.program, cs.department AS session_department,
                   cs.year AS session_year, cs.room, cs.subject, cs.section,
                   cs.started_at, cs.ended_at, cs.status AS session_status
            FROM attendance a
            LEFT JOIN students s ON a.student_id = s.student_id
            LEFT JOIN class_sessions cs ON a.session_id = cs.id
            WHERE a.session_id = ?
            ORDER BY a.time DESC
        """, (session_id,))
        return [self._decorate_attendance(dict(r)) for r in cur.fetchall()]

    def count_attendance_for_session(self, session_id: int) -> int:
        cur = self._cursor()
        cur.execute(
            "SELECT COUNT(*) FROM attendance WHERE session_id = ?",
            (session_id,),
        )
        return cur.fetchone()[0]

    def count_attendance_records_by_date(self, date_str: str) -> int:
        cur = self._cursor()
        cur.execute("SELECT COUNT(*) FROM attendance WHERE date = ?", (date_str,))
        return cur.fetchone()[0]

    def count_present_students_by_date(self, date_str: str) -> int:
        cur = self._cursor()
        cur.execute("""
            SELECT COUNT(DISTINCT student_id)
            FROM attendance
            WHERE date = ?
        """, (date_str,))
        return cur.fetchone()[0]

    def get_weekly_trend(self) -> List[Dict]:
        today = date.today()
        rows = []
        cur = self._cursor()
        for i in range(6, -1, -1):
            day_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            cur.execute("""
                SELECT COUNT(DISTINCT student_id)
                FROM attendance
                WHERE date = ?
            """, (day_str,))
            rows.append({"date": day_str, "count": cur.fetchone()[0]})
        return rows

    def get_dept_breakdown(self, date_str: str) -> List[Dict]:
        cur = self._cursor()
        cur.execute("""
            SELECT s.department, COUNT(DISTINCT a.student_id) AS present
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE a.date = ?
            GROUP BY s.department
            ORDER BY present DESC
        """, (date_str,))
        rows = []
        for row in cur.fetchall():
            department = normalize_branch(row["department"]) or "Unassigned"
            rows.append({
                "department": department,
                "department_name": get_branch_name(department) if department != "Unassigned" else department,
                "present": row["present"],
            })
        return rows

    def delete_attendance(self, attendance_id: int) -> bool:
        cur = self._cursor()
        cur.execute("DELETE FROM attendance WHERE id = ?", (attendance_id,))
        self._commit()
        return cur.rowcount > 0

    # Decorators -----------------------------------------------------------

    def _decorate_student(self, student: Dict) -> Dict:
        student["department"] = normalize_branch(student.get("department", ""))
        student["branch_name"] = get_branch_name(student["department"]) if student["department"] else ""
        return student

    def _decorate_session(self, session: Dict) -> Dict:
        session["department"] = normalize_branch(session.get("department", ""))
        session["room"] = (session.get("room") or "").strip().upper()
        session["section"] = (session.get("section") or "").strip().upper()
        session["program"] = session.get("program") or PROGRAM_NAME
        session["branch_name"] = get_branch_name(session["department"])
        session["batch_label"] = format_batch_label(
            session["department"],
            int(session.get("year") or 1),
            session["section"],
        )
        session["display_label"] = format_session_label(
            session["program"],
            session["department"],
            int(session.get("year") or 1),
            session["room"],
            session.get("subject", ""),
            session["section"],
        )
        session["is_active"] = session.get("status") == "active"
        return session

    def _decorate_attendance(self, record: Dict) -> Dict:
        record["department"] = normalize_branch(record.get("department", ""))
        record["branch_name"] = get_branch_name(record["department"]) if record["department"] else ""
        session_department = normalize_branch(record.get("session_department", ""))
        session_year = int(record.get("session_year") or record.get("year") or 1)
        if record.get("session_id"):
            record["session_department"] = session_department
            record["session_year"] = session_year
            record["batch_label"] = format_batch_label(
                session_department,
                session_year,
                record.get("section", ""),
            )
            record["session_label"] = format_session_label(
                record.get("program") or PROGRAM_NAME,
                session_department,
                session_year,
                record.get("room", ""),
                record.get("subject", ""),
                record.get("section", ""),
            )
        else:
            if record.get("department") and record.get("year"):
                record["batch_label"] = format_batch_label(
                    record["department"],
                    int(record["year"]),
                )
            else:
                record["batch_label"] = "Legacy"
            record["session_label"] = "Legacy daily attendance"
            record["session_department"] = ""
            record["session_year"] = None
            record["room"] = ""
            record["subject"] = ""
            record["section"] = ""
            record["program"] = PROGRAM_NAME
        return record

    def _decorate_presence(self, record: Dict) -> Dict:
        session_department = normalize_branch(record.get("session_department", ""))
        session_year = int(record.get("session_year") or 1)
        record["session_department"] = session_department
        record["session_year"] = session_year
        record["session_label"] = format_session_label(
            record.get("program") or PROGRAM_NAME,
            session_department,
            session_year,
            record.get("room", ""),
            record.get("subject", ""),
            record.get("section", ""),
        ) if session_department else "Unassigned session"
        return record
