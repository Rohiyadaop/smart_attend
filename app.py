"""
=============================================================================
  SmartAttend — Flask Application
=============================================================================
  Serves the web dashboard and REST API.
  Manages the live recognition loop in a background thread.

  HOW THE DASHBOARD CONNECTS TO THE BACKEND
  ─────────────────────────────────────────────────────────────────────────
  1. Browser loads dashboard.html from Flask's template system.
  2. JavaScript polls /api/live_status every 2 seconds via fetch().
  3. /video_feed returns an MJPEG stream — displayed in an <img> tag.
  4. Chart.js calls /api/stats and /api/weekly_trend for graph data.
  5. Admin actions POST to /api/register, /api/delete_student, /api/train.

  RECOGNITION LOOP
  ─────────────────────────────────────────────────────────────────────────
  The recognition loop thread runs independently of Flask:
    while running:
        frame  = camera.get_frame()
        result = engine.process_frame(frame)
        for face in result.faces:
            if face.is_known:
                attendance.mark_attendance(face.student_id, ...)
                gpio.on_attendance_marked(...)
            else:
                gpio.on_unknown_face()
        sleep(0.1)   ← 10 fps recognition rate
=============================================================================
"""

import os
import io
import time
import uuid
import shutil
import logging
import threading
from pathlib import Path
from datetime import datetime, date
from functools import wraps

from flask import (Flask, render_template, request, jsonify,
                   Response, redirect, url_for, session, send_file,
                   flash, abort)

# SmartAttend modules
from core.face_engine      import FaceEngine
from core.camera           import CameraManager, mjpeg_stream_generator
from core.attendance       import AttendanceManager
from core.trainer          import ModelTrainer
from core.gpio_indicator   import GPIOIndicator
from database.db_manager   import DBManager

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    handlers= [
        logging.StreamHandler(),
        logging.FileHandler("logs/smartattend.log"),
    ],
)
logger = logging.getLogger("smartattend.app")

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR   = Path("dataset")
PHOTOS_PER_STUDENT = 200       # still shots captured during registration
ADMIN_USERNAME = os.environ.get("ADMIN_USER",  "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASS",  "smartattend2024")
CAMERA_BACKEND = os.environ.get("CAMERA",      "auto")   # auto|picamera2|opencv
CAMERA_DEVICE  = int(os.environ.get("CAM_DEV", "0"))
GPIO_ENABLED   = os.environ.get("GPIO", "false").lower() == "true"

# ─────────────────────────────────────────────────────────────────────────────
#  Singletons
# ─────────────────────────────────────────────────────────────────────────────

app       = Flask(__name__, template_folder="web/templates",
                  static_folder="web/static")
app.secret_key = os.environ.get("SECRET_KEY", "smartattend-secret-" + str(uuid.uuid4()))

db        = DBManager()
engine    = FaceEngine()
attendmgr = AttendanceManager(db)
gpio      = GPIOIndicator(enabled=GPIO_ENABLED)
trainer   = ModelTrainer(engine)
camera    = None        # initialised lazily on first /video_feed request

# ── Shared recognition state ──────────────────────────────────────────────────
recog_state = {
    "running":          False,
    "last_face_name":   None,
    "last_status":      None,
    "last_timestamp":   None,
    "frames_processed": 0,
    "today_count":      0,
    "training_progress": None,     # set during model training
}
recog_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
#  Auth
# ─────────────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if (request.form.get("username") == ADMIN_USERNAME
                and request.form.get("password") == ADMIN_PASSWORD):
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ─────────────────────────────────────────────────────────────────────────────
#  Page routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
@login_required
def dashboard():
    summary = attendmgr.get_summary()
    trend   = attendmgr.get_weekly_trend()
    today   = attendmgr.get_today_attendance()[:10]
    dept    = db.get_dept_breakdown(date.today().strftime("%Y-%m-%d"))
    stats   = engine.get_model_stats()
    return render_template("dashboard.html",
                           summary=summary, trend=trend,
                           recent=today, dept=dept,
                           model_stats=stats,
                           recog=recog_state)


@app.route("/attendance")
@login_required
def attendance_page():
    date_str  = request.args.get("date", date.today().strftime("%Y-%m-%d"))
    records   = attendmgr.get_attendance_by_date(date_str)
    summary   = attendmgr.get_summary(date_str)
    return render_template("attendance.html",
                           records=records, summary=summary,
                           selected_date=date_str)


@app.route("/students")
@login_required
def students_page():
    query    = request.args.get("q", "")
    students = (db.search_students(query) if query else db.get_all_students())
    return render_template("students.html",
                           students=students, query=query)


@app.route("/register")
@login_required
def register_page():
    return render_template("register.html")


@app.route("/admin")
@login_required
def admin_page():
    stats  = engine.get_model_stats()
    return render_template("admin.html",
                           model_stats=stats, recog=recog_state)


@app.route("/live")
@login_required
def live_page():
    return render_template("live.html")


# ─────────────────────────────────────────────────────────────────────────────
#  Video feed
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/video_feed")
@login_required
def video_feed():
    _ensure_camera()
    gen = mjpeg_stream_generator(camera, engine, fps_limit=30)
    return Response(gen, mimetype="multipart/x-mixed-replace; boundary=frame")


def _ensure_camera():
    global camera
    if camera is None:
        try:
            camera = CameraManager(backend=CAMERA_BACKEND,
                                   device_id=CAMERA_DEVICE)
            camera.start()
            logger.info("Camera started on demand")
        except Exception as exc:
            logger.error("Camera start failed: %s", exc)
            camera = None
            abort(503, description=f"Camera unavailable: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  Recognition loop
# ─────────────────────────────────────────────────────────────────────────────

def recognition_loop():
    """Background thread — continuously recognises faces and marks attendance."""
    logger.info("Recognition loop started")
    _ensure_camera()

    while recog_state["running"]:
        try:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            result = engine.process_frame(frame, draw_boxes=False)

            with recog_lock:
                recog_state["frames_processed"] += 1

            for face in result.faces:
                outcome = attendmgr.mark_attendance(
                    student_id = face.student_id,
                    name       = face.name,
                    confidence = face.confidence,
                )
                with recog_lock:
                    recog_state["last_face_name"] = face.name
                    recog_state["last_status"]    = outcome["status"]
                    recog_state["last_timestamp"] = datetime.now().strftime("%H:%M:%S")
                    if outcome["status"] == "marked":
                        recog_state["today_count"] += 1

                # GPIO feedback
                if outcome["status"] == "marked":
                    gpio.on_attendance_marked(face.name)
                elif outcome["status"] == "duplicate":
                    gpio.on_duplicate(face.name)
                else:
                    gpio.on_unknown_face()

            time.sleep(0.10)    # ~10 fps recognition

        except Exception as exc:
            logger.exception("Recognition loop error: %s", exc)
            time.sleep(1.0)

    logger.info("Recognition loop stopped")


@app.route("/api/recognition/start", methods=["POST"])
@login_required
def api_start_recognition():
    if recog_state["running"]:
        return jsonify({"status": "already_running"})

    if not engine.is_model_loaded():
        ok = engine.load_model()
        if not ok:
            return jsonify({"status": "error",
                            "message": "No model loaded. Train first."}), 400

    recog_state["running"] = True
    t = threading.Thread(target=recognition_loop, daemon=True,
                         name="RecognitionLoop")
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/recognition/stop", methods=["POST"])
@login_required
def api_stop_recognition():
    recog_state["running"] = False
    return jsonify({"status": "stopped"})


# ─────────────────────────────────────────────────────────────────────────────
#  REST API — Stats / Attendance
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/live_status")
@login_required
def api_live_status():
    with recog_lock:
        return jsonify({
            "running":        recog_state["running"],
            "last_name":      recog_state["last_face_name"],
            "last_status":    recog_state["last_status"],
            "last_time":      recog_state["last_timestamp"],
            "today_count":    recog_state["today_count"],
            "frames":         recog_state["frames_processed"],
        })


@app.route("/api/stats")
@login_required
def api_stats():
    date_str = request.args.get("date", date.today().strftime("%Y-%m-%d"))
    return jsonify({
        "summary": attendmgr.get_summary(date_str),
        "dept":    db.get_dept_breakdown(date_str),
    })


@app.route("/api/weekly_trend")
@login_required
def api_weekly_trend():
    return jsonify(attendmgr.get_weekly_trend())


@app.route("/api/today_attendance")
@login_required
def api_today_attendance():
    records = attendmgr.get_today_attendance()
    return jsonify({"records": records, "count": len(records)})


# ─────────────────────────────────────────────────────────────────────────────
#  REST API — Students
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
@login_required
def api_register():
    """Register a new student and capture face photos from live camera."""
    data       = request.json or {}
    student_id = data.get("student_id", "").strip().upper()
    name       = data.get("name", "").strip()
    department = data.get("department", "").strip()
    year       = int(data.get("year", 1))
    email      = data.get("email", "").strip()
    phone      = data.get("phone", "").strip()

    if not student_id or not name:
        return jsonify({"success": False, "message": "student_id and name required"}), 400

    if db.student_exists(student_id):
        return jsonify({"success": False, "message": f"Student {student_id} already exists"}), 409

    # Create dataset folder
    folder_name = f"{student_id}_{name.replace(' ', '_')}"
    folder      = DATASET_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    # Capture photos
    _ensure_camera()
    photo_paths = []
    photo_path  = ""
    for i in range(PHOTOS_PER_STUDENT):
        time.sleep(0.4)     # brief delay between shots
        path = str(folder / f"img_{i+1:03d}.jpg")
        if camera and camera.capture_still(path, n_frames=3):
            photo_paths.append(path)
            if not photo_path:
                photo_path = path
        else:
            logger.warning("Could not capture photo %d for %s", i+1, name)

    if not photo_paths:
        shutil.rmtree(folder, ignore_errors=True)
        return jsonify({"success": False,
                        "message": "Camera capture failed — no photos taken"}), 500

    # Insert into DB
    student = db.insert_student(student_id, name, department, year,
                                 email, phone, photo_path)
    if not student:
        return jsonify({"success": False,
                        "message": "Database error — student may already exist"}), 500

    logger.info("Registered student: %s (%s) — %d photos", name, student_id, len(photo_paths))
    return jsonify({
        "success":      True,
        "message":      f"{name} registered with {len(photo_paths)} photos. Retrain model.",
        "student":      student,
        "photos_taken": len(photo_paths),
    })


@app.route("/api/delete_student", methods=["POST"])
@login_required
def api_delete_student():
    student_id = (request.json or {}).get("student_id", "").strip().upper()
    if not student_id:
        return jsonify({"success": False, "message": "student_id required"}), 400

    student = db.get_student(student_id)
    if not student:
        return jsonify({"success": False, "message": "Student not found"}), 404

    # Remove dataset folder
    name        = student["name"]
    folder_name = f"{student_id}_{name.replace(' ', '_')}"
    folder      = DATASET_DIR / folder_name
    shutil.rmtree(folder, ignore_errors=True)

    db.delete_student(student_id)
    logger.info("Deleted student %s (%s)", name, student_id)
    return jsonify({"success": True, "message": f"{name} deleted"})


@app.route("/api/students")
@login_required
def api_students():
    return jsonify(db.get_all_students())


# ─────────────────────────────────────────────────────────────────────────────
#  REST API — Model training
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
@login_required
def api_train():
    """Kick off training in a background thread and return immediately."""

    if recog_state.get("training_progress") == "running":
        return jsonify({"success": False, "message": "Training already in progress"}), 409

    def _train_bg():
        recog_state["training_progress"] = "running"
        recog_state["running"] = False          # pause recognition during training

        def progress_cb(cur, total, msg):
            recog_state["training_progress"] = f"{cur}/{total}: {msg}"

        result = trainer.train(progress_callback=progress_cb)
        recog_state["training_progress"] = (
            "done" if result["success"] else f"error: {result['message']}"
        )
        logger.info("Training result: %s", result["message"])

    t = threading.Thread(target=_train_bg, daemon=True, name="ModelTrainer")
    t.start()
    return jsonify({"success": True, "message": "Training started in background"})


@app.route("/api/train_status")
@login_required
def api_train_status():
    return jsonify({
        "progress": recog_state.get("training_progress"),
        "model":    engine.get_model_stats(),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  CSV Download
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/download/attendance")
@login_required
def download_attendance():
    date_str = request.args.get("date", date.today().strftime("%Y-%m-%d"))
    csv_data = attendmgr.export_csv(date_str)
    buf      = io.BytesIO(csv_data.encode("utf-8"))
    return send_file(
        buf,
        mimetype    = "text/csv",
        as_attachment = True,
        download_name = f"attendance_{date_str}.csv",
    )


@app.route("/download/attendance_range")
@login_required
def download_attendance_range():
    start = request.args.get("start", date.today().strftime("%Y-%m-%d"))
    end   = request.args.get("end",   date.today().strftime("%Y-%m-%d"))
    csv_data = attendmgr.export_range_csv(start, end)
    buf      = io.BytesIO(csv_data.encode("utf-8"))
    return send_file(
        buf,
        mimetype      = "text/csv",
        as_attachment  = True,
        download_name  = f"attendance_{start}_to_{end}.csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Startup
# ─────────────────────────────────────────────────────────────────────────────

def startup():
    Path("logs").mkdir(exist_ok=True)
    DATASET_DIR.mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    engine.load_model()
    recog_state["today_count"] = len(attendmgr.get_today_attendance())
    logger.info("=" * 55)
    logger.info("  SmartAttend — Face Recognition Attendance System")
    logger.info("  Open  http://<raspberry-pi-ip>:5000  in browser")
    logger.info("=" * 55)


if __name__ == "__main__":
    startup()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
