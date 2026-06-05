# """
# =============================================================================
#   SmartAttend — Flask Application
# =============================================================================
#   Serves the web dashboard and REST API.
#   Manages the live recognition loop in a background thread.

#   HOW THE DASHBOARD CONNECTS TO THE BACKEND
#   ─────────────────────────────────────────────────────────────────────────
#   1. Browser loads dashboard.html from Flask's template system.
#   2. JavaScript polls /api/live_status every 2 seconds via fetch().
#   3. /video_feed returns an MJPEG stream — displayed in an <img> tag.
#   4. Chart.js calls /api/stats and /api/weekly_trend for graph data.
#   5. Admin actions POST to /api/register, /api/delete_student, /api/train.

#   RECOGNITION LOOP
#   ─────────────────────────────────────────────────────────────────────────
#   The recognition loop thread runs independently of Flask:
#     while running:
#         frame  = camera.get_frame()
#         result = engine.process_frame(frame)
#         for face in result.faces:
#             if face.is_known:
#                 attendance.mark_attendance(face.student_id, ...)
#                 gpio.on_attendance_marked(...)
#             else:
#                 gpio.on_unknown_face()
#         sleep(0.1)   ← 10 fps recognition rate
# =============================================================================
# """

# import os
# import io
# import time
# import uuid
# import shutil
# import logging
# import threading
# from pathlib import Path
# from datetime import datetime, date
# from functools import wraps

# from flask import (Flask, render_template, request, jsonify,
#                    Response, redirect, url_for, session, send_file,
#                    flash, abort)

# # SmartAttend modules
# from core.face_engine      import FaceEngine
# from core.camera           import CameraManager, mjpeg_stream_generator
# from core.attendance       import AttendanceManager
# from core.college          import (
#     BTECH_BRANCHES,
#     BRANCH_CODES,
#     PROGRAM_NAME,
#     YEAR_OPTIONS,
#     normalize_branch,
# )
# from core.trainer          import ModelTrainer
# from core.gpio_indicator   import GPIOIndicator
# from database.db_manager   import DBManager

# # ─────────────────────────────────────────────────────────────────────────────
# #  Logging
# # ─────────────────────────────────────────────────────────────────────────────

# logging.basicConfig(
#     level   = logging.INFO,
#     format  = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
#     handlers= [
#         logging.StreamHandler(),
#         logging.FileHandler("logs/smartattend.log"),
#     ],
# )
# logger = logging.getLogger("smartattend.app")

# # ─────────────────────────────────────────────────────────────────────────────
# #  Configuration
# # ─────────────────────────────────────────────────────────────────────────────

# DATASET_DIR   = Path("dataset")
# PHOTOS_PER_STUDENT = int(os.environ.get("PHOTOS_PER_STUDENT", "60"))
# ADMIN_USERNAME = os.environ.get("ADMIN_USER",  "admin")
# ADMIN_PASSWORD = os.environ.get("ADMIN_PASS",  "smartattend2024")
# CAMERA_BACKEND = os.environ.get("CAMERA",      "auto")   # auto|picamera2|opencv
# CAMERA_DEVICE  = int(os.environ.get("CAM_DEV", "0"))
# GPIO_ENABLED   = os.environ.get("GPIO", "false").lower() == "true"
# RECOGNITION_FPS = max(float(os.environ.get("RECOG_FPS", "5")), 1.0)
# STREAM_FPS      = max(int(os.environ.get("STREAM_FPS", "12")), 1)
# STREAM_QUALITY  = max(int(os.environ.get("STREAM_JPEG_QUALITY", "70")), 40)
# OVERLAY_TTL_SEC = max(float(os.environ.get("OVERLAY_TTL_SEC", "0.45")), 0.1)

# # ─────────────────────────────────────────────────────────────────────────────
# #  Singletons
# # ─────────────────────────────────────────────────────────────────────────────

# app       = Flask(__name__, template_folder="web/templates",
#                   static_folder="web/static")
# app.secret_key = os.environ.get("SECRET_KEY", "smartattend-secret-" + str(uuid.uuid4()))

# db        = DBManager()
# engine    = FaceEngine()
# attendmgr = AttendanceManager(db)
# gpio      = GPIOIndicator(enabled=GPIO_ENABLED)
# trainer   = ModelTrainer(engine)
# camera    = None        # initialised lazily on first /video_feed request

# # ── Shared recognition state ──────────────────────────────────────────────────
# recog_state = {
#     "running":          False,
#     "last_face_name":   None,
#     "last_status":      None,
#     "last_message":     None,
#     "last_timestamp":   None,
#     "last_marked_name": None,
#     "last_marked_time": None,
#     "frames_processed": 0,
#     "today_count":      0,
#     "last_distance":    None,
#     "last_similarity":  None,
#     "last_confidence":  None,
#     "last_match_index": None,
#     "last_processing_ms": 0.0,
#     "last_backend":     None,
#     "camera_fps":       0.0,
#     "recognition_fps":  0.0,
#     "last_faces":       [],
#     "last_overlay_at":  0.0,
#     "training_progress": None,     # set during model training
# }
# recog_lock = threading.Lock()

# # ─────────────────────────────────────────────────────────────────────────────
# #  Auth
# # ─────────────────────────────────────────────────────────────────────────────

# def login_required(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         if not session.get("logged_in"):
#             return redirect(url_for("login"))
#         return f(*args, **kwargs)
#     return decorated


# @app.route("/login", methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         if (request.form.get("username") == ADMIN_USERNAME
#                 and request.form.get("password") == ADMIN_PASSWORD):
#             session["logged_in"] = True
#             return redirect(url_for("dashboard"))
#         flash("Invalid credentials", "error")
#     return render_template("login.html")


# @app.route("/logout")
# def logout():
#     session.clear()
#     return redirect(url_for("login"))


# @app.context_processor
# def inject_college_context():
#     return {
#         "program_name": PROGRAM_NAME,
#         "branch_options": BTECH_BRANCHES,
#         "year_options": YEAR_OPTIONS,
#         "photos_per_student": PHOTOS_PER_STUDENT,
#     }


# def _today_str() -> str:
#     return date.today().strftime("%Y-%m-%d")


# def _active_session_bundle():
#     active_session = db.get_active_session()
#     active_summary = (
#         attendmgr.get_summary(session_id=active_session["id"])
#         if active_session else None
#     )
#     return active_session, active_summary


# def _serialize_face(face):
#     return {
#         "name": face.name,
#         "student_id": face.student_id,
#         "confidence": face.confidence,
#         "distance": face.distance,
#         "similarity": face.similarity,
#         "matched_index": face.matched_index,
#         "bounding_box": list(face.bounding_box),
#         "is_known": face.is_known,
#     }


# def _attendance_payload(date_str: str, session_id: int | None = None):
#     records = attendmgr.get_attendance_by_date(date_str, session_id=session_id)
#     summary = (
#         attendmgr.get_summary(session_id=session_id)
#         if session_id
#         else attendmgr.get_summary(date_str)
#     )
#     sessions = db.get_sessions_by_date(date_str)
#     selected_session = next((item for item in sessions if item["id"] == session_id), None)
#     return {
#         "success": True,
#         "date": date_str,
#         "session_id": session_id,
#         "selected_session": selected_session,
#         "summary": summary,
#         "count": len(records),
#         "records": records,
#     }


# def _stream_overlay_callback(frame, frame_ts):
#     with recog_lock:
#         faces = recog_state["last_faces"]
#         overlay_at = recog_state["last_overlay_at"]

#     if not faces or (frame_ts and (time.time() - overlay_at) > OVERLAY_TTL_SEC):
#         return frame

#     engine.draw_results(frame, faces)
#     return frame


# # ─────────────────────────────────────────────────────────────────────────────
# #  Page routes
# # ─────────────────────────────────────────────────────────────────────────────

# @app.route("/")
# @login_required
# def dashboard():
#     today_str = _today_str()
#     summary = attendmgr.get_summary(today_str)
#     active_session, active_summary = _active_session_bundle()
#     trend = attendmgr.get_weekly_trend()
#     today = attendmgr.get_today_attendance()[:10]
#     dept = db.get_dept_breakdown(today_str)
#     stats = engine.get_model_stats()
#     return render_template("dashboard.html",
#                            summary=summary, trend=trend,
#                            recent=today, dept=dept,
#                            active_session=active_session,
#                            active_summary=active_summary,
#                            batch_matrix=db.get_btech_batch_matrix(),
#                            today_sessions=db.get_sessions_by_date(today_str),
#                            unassigned_students=db.count_unassigned_students(),
#                            model_stats=stats,
#                            recog=recog_state)


# @app.route("/attendance")
# @login_required
# def attendance_page():
#     date_str = request.args.get("date", _today_str())
#     session_id = request.args.get("session", type=int)
#     sessions = db.get_sessions_by_date(date_str)
#     valid_session_ids = {item["id"] for item in sessions}
#     if session_id not in valid_session_ids:
#         session_id = None
#     selected_session = next(
#         (item for item in sessions if item["id"] == session_id),
#         None,
#     )
#     records = attendmgr.get_attendance_by_date(date_str, session_id=session_id)
#     summary = (
#         attendmgr.get_summary(session_id=session_id)
#         if session_id else attendmgr.get_summary(date_str)
#     )
#     wants_json = (
#         request.args.get("format") == "json"
#         or request.accept_mimetypes.best == "application/json"
#     )
#     if wants_json:
#         return jsonify(_attendance_payload(date_str, session_id=session_id))
#     return render_template("attendance.html",
#                            records=records, summary=summary,
#                            selected_date=date_str,
#                            sessions=sessions,
#                            selected_session_id=session_id,
#                            selected_session=selected_session)


# @app.route("/students")
# @login_required
# def students_page():
#     query    = request.args.get("q", "")
#     students = (db.search_students(query) if query else db.get_all_students())
#     return render_template("students.html",
#                            students=students, query=query)


# @app.route("/register")
# @login_required
# def register_page():
#     return render_template("register.html")


# @app.route("/admin")
# @login_required
# def admin_page():
#     stats  = engine.get_model_stats()
#     return render_template("admin.html",
#                            model_stats=stats, recog=recog_state)


# @app.route("/live")
# @login_required
# def live_page():
#     active_session, active_summary = _active_session_bundle()
#     return render_template("live.html",
#                            active_session=active_session,
#                            active_summary=active_summary)


# # ─────────────────────────────────────────────────────────────────────────────
# #  Video feed
# # ─────────────────────────────────────────────────────────────────────────────

# @app.route("/video_feed")
# @login_required
# def video_feed():
#     _ensure_camera()
#     gen = mjpeg_stream_generator(
#         camera,
#         overlay_callback=_stream_overlay_callback,
#         fps_limit=STREAM_FPS,
#         quality=STREAM_QUALITY,
#     )
#     return Response(gen, mimetype="multipart/x-mixed-replace; boundary=frame")


# def _ensure_camera():
#     global camera
#     if camera is None:
#         try:
#             camera = CameraManager(backend=CAMERA_BACKEND,
#                                    device_id=CAMERA_DEVICE)
#             camera.start()
#             logger.info("Camera started on demand")
#         except Exception as exc:
#             logger.error("Camera start failed: %s", exc)
#             camera = None
#             abort(503, description=f"Camera unavailable: {exc}")


# # ─────────────────────────────────────────────────────────────────────────────
# #  Recognition loop
# # ─────────────────────────────────────────────────────────────────────────────

# def recognition_loop():
#     """Background thread — continuously recognises faces and marks attendance."""
#     logger.info("Recognition loop started")
#     _ensure_camera()
#     target_interval = 1.0 / RECOGNITION_FPS
#     last_loop_started = time.time()

#     while recog_state["running"]:
#         loop_started = time.time()
#         try:
#             frame = camera.get_frame(copy=False)
#             if frame is None:
#                 time.sleep(0.1)
#                 continue

#             result = engine.process_frame(frame, draw_boxes=False)

#             with recog_lock:
#                 recog_state["frames_processed"] += 1
#                 elapsed = max(loop_started - last_loop_started, 1e-6)
#                 recog_state["recognition_fps"] = round(1.0 / elapsed, 2)
#                 recog_state["camera_fps"] = round(camera.capture_fps if camera else 0.0, 2)
#                 recog_state["last_processing_ms"] = round(result.processing_time_ms, 2)
#                 recog_state["last_backend"] = result.backend
#                 recog_state["last_faces"] = list(result.faces)
#                 recog_state["last_overlay_at"] = time.time()
#             last_loop_started = loop_started

#             active_session = db.get_active_session()
#             for face in result.faces:
#                 outcome = attendmgr.mark_attendance(
#                     student_id = face.student_id,
#                     name       = face.name,
#                     confidence = face.confidence,
#                     active_session = active_session,
#                 )
#                 with recog_lock:
#                     recog_state["last_face_name"] = face.name
#                     recog_state["last_status"]    = outcome["status"]
#                     recog_state["last_message"]   = outcome.get("message")
#                     recog_state["last_timestamp"] = datetime.now().strftime("%H:%M:%S")
#                     recog_state["last_distance"] = face.distance
#                     recog_state["last_similarity"] = face.similarity
#                     recog_state["last_confidence"] = face.confidence
#                     recog_state["last_match_index"] = face.matched_index
#                     if outcome["status"] == "marked":
#                         recog_state["last_marked_name"] = face.name
#                         recog_state["last_marked_time"] = recog_state["last_timestamp"]
#                         recog_state["today_count"] = db.count_attendance_records_by_date(_today_str())

#                 # GPIO feedback
#                 if outcome["status"] == "marked":
#                     gpio.on_attendance_marked(face.name)
#                 elif outcome["status"] == "duplicate":
#                     gpio.on_duplicate(face.name)
#                 else:
#                     gpio.on_unknown_face()

#             if not result.faces:
#                 with recog_lock:
#                     recog_state["last_faces"] = []

#             elapsed = time.time() - loop_started
#             if elapsed < target_interval:
#                 time.sleep(target_interval - elapsed)

#         except Exception as exc:
#             logger.exception("Recognition loop error: %s", exc)
#             time.sleep(1.0)

#     logger.info("Recognition loop stopped")


# @app.route("/api/recognition/start", methods=["POST"])
# @login_required
# def api_start_recognition():
#     if recog_state["running"]:
#         return jsonify({
#             "success": True,
#             "status": "already_running",
#             "message": "Recognition is already running.",
#         })

#     active_session = db.get_active_session()
#     if not active_session:
#         return jsonify({
#             "success": False,
#             "status": "error",
#             "message": "Start the running class session first.",
#         }), 400

#     if not engine.is_model_loaded():
#         ok = engine.load_model()
#         if not ok:
#             return jsonify({"success": False, "status": "error",
#                             "message": "No model loaded. Train first."}), 400

#     recog_state["running"] = True
#     t = threading.Thread(target=recognition_loop, daemon=True,
#                          name="RecognitionLoop")
#     t.start()
#     return jsonify({"success": True, "status": "started", "message": "Recognition started."})


# @app.route("/api/recognition/stop", methods=["POST"])
# @login_required
# def api_stop_recognition():
#     recog_state["running"] = False
#     with recog_lock:
#         recog_state["last_faces"] = []
#     return jsonify({"success": True, "status": "stopped", "message": "Recognition stopped."})


# @app.route("/api/session/start", methods=["POST"])
# @login_required
# def api_start_session():
#     data = request.json or {}
#     department = normalize_branch(data.get("department", ""))
#     room = data.get("room", "").strip().upper()
#     subject = data.get("subject", "").strip()
#     section = data.get("section", "").strip().upper()

#     try:
#         year = int(data.get("year", 1))
#     except (TypeError, ValueError):
#         year = 0

#     if department not in BRANCH_CODES:
#         return jsonify({"success": False, "message": "Select a valid B.Tech branch."}), 400
#     if year not in YEAR_OPTIONS:
#         return jsonify({"success": False, "message": "Select a valid year."}), 400
#     if not room:
#         return jsonify({"success": False, "message": "Room is required."}), 400

#     session_row = db.start_class_session(
#         PROGRAM_NAME,
#         department,
#         year,
#         room,
#         subject,
#         section,
#     )
#     attendmgr.clear_session_cache(session_row["id"])
#     return jsonify({
#         "success": True,
#         "message": f"Class session started for {session_row['display_label']}.",
#         "session": session_row,
#     })


# @app.route("/api/session/stop", methods=["POST"])
# @login_required
# def api_stop_session():
#     session_row = db.stop_class_session()
#     recog_state["running"] = False
#     with recog_lock:
#         recog_state["last_faces"] = []
#     if not session_row:
#         return jsonify({"success": False, "message": "No active class session."}), 404
#     attendmgr.clear_session_cache(session_row["id"])
#     return jsonify({
#         "success": True,
#         "message": f"Class session closed for {session_row['display_label']}.",
#         "session": session_row,
#     })


# # ─────────────────────────────────────────────────────────────────────────────
# #  REST API — Stats / Attendance
# # ─────────────────────────────────────────────────────────────────────────────

# @app.route("/api/live_status")
# @login_required
# def api_live_status():
#     active_session, active_summary = _active_session_bundle()
#     today_count = db.count_attendance_records_by_date(_today_str())
#     with recog_lock:
#         return jsonify({
#             "success":        True,
#             "running":        recog_state["running"],
#             "last_name":      recog_state["last_face_name"],
#             "last_status":    recog_state["last_status"],
#             "last_message":   recog_state["last_message"],
#             "last_time":      recog_state["last_timestamp"],
#             "last_marked_name": recog_state["last_marked_name"],
#             "last_marked_time": recog_state["last_marked_time"],
#             "last_distance":  recog_state["last_distance"],
#             "last_similarity": recog_state["last_similarity"],
#             "last_confidence": recog_state["last_confidence"],
#             "last_match_index": recog_state["last_match_index"],
#             "last_processing_ms": recog_state["last_processing_ms"],
#             "backend":        recog_state["last_backend"],
#             "camera_fps":     recog_state["camera_fps"],
#             "recognition_fps": recog_state["recognition_fps"],
#             "faces":          [_serialize_face(face) for face in recog_state["last_faces"]],
#             "today_count":    today_count,
#             "session_count":  (active_summary or {}).get("present", 0),
#             "active_session": active_session,
#             "active_summary": active_summary,
#             "frames":         recog_state["frames_processed"],
#         })


# @app.route("/api/stats")
# @login_required
# def api_stats():
#     date_str = request.args.get("date", _today_str())
#     active_session, active_summary = _active_session_bundle()
#     return jsonify({
#         "success": True,
#         "summary": attendmgr.get_summary(date_str),
#         "dept":    db.get_dept_breakdown(date_str),
#         "active_session": active_session,
#         "active_summary": active_summary,
#     })


# @app.route("/api/weekly_trend")
# @login_required
# def api_weekly_trend():
#     return jsonify(attendmgr.get_weekly_trend())


# @app.route("/api/today_attendance")
# @login_required
# def api_today_attendance():
#     session_id = request.args.get("session", type=int)
#     return jsonify(_attendance_payload(_today_str(), session_id=session_id))


# @app.route("/api/attendance")
# @login_required
# def api_attendance():
#     date_str = request.args.get("date", _today_str())
#     session_id = request.args.get("session", type=int)
#     return jsonify(_attendance_payload(date_str, session_id=session_id))


# # ─────────────────────────────────────────────────────────────────────────────
# #  REST API — Students
# # ─────────────────────────────────────────────────────────────────────────────

# @app.route("/api/register", methods=["POST"])
# @login_required
# def api_register():
#     """Register a new student and capture face photos from live camera."""
#     data       = request.json or {}
#     student_id = data.get("student_id", "").strip().upper()
#     name       = data.get("name", "").strip()
#     department = normalize_branch(data.get("department", "").strip())
#     try:
#         year = int(data.get("year", 1))
#     except (TypeError, ValueError):
#         year = 0
#     email      = data.get("email", "").strip()
#     phone      = data.get("phone", "").strip()

#     if not student_id or not name:
#         return jsonify({"success": False, "message": "student_id and name required"}), 400
#     if department not in BRANCH_CODES:
#         return jsonify({"success": False, "message": "Select a valid B.Tech branch"}), 400
#     if year not in YEAR_OPTIONS:
#         return jsonify({"success": False, "message": "Year must be between 1 and 4"}), 400

#     if db.student_exists(student_id):
#         return jsonify({"success": False, "message": f"Student {student_id} already exists"}), 409

#     # Create dataset folder
#     folder_name = f"{student_id}_{name.replace(' ', '_')}"
#     folder      = DATASET_DIR / folder_name
#     folder.mkdir(parents=True, exist_ok=True)

#     # Capture photos
#     _ensure_camera()
#     photo_paths = []
#     photo_path  = ""
#     for i in range(PHOTOS_PER_STUDENT):
#         time.sleep(0.4)     # brief delay between shots
#         path = str(folder / f"img_{i+1:03d}.jpg")
#         if camera and camera.capture_still(path, n_frames=3):
#             photo_paths.append(path)
#             if not photo_path:
#                 photo_path = path
#         else:
#             logger.warning("Could not capture photo %d for %s", i+1, name)

#     if not photo_paths:
#         shutil.rmtree(folder, ignore_errors=True)
#         return jsonify({"success": False,
#                         "message": "Camera capture failed — no photos taken"}), 500

#     # Insert into DB
#     student = db.insert_student(student_id, name, department, year,
#                                  email, phone, photo_path)
#     if not student:
#         return jsonify({"success": False,
#                         "message": "Database error — student may already exist"}), 500

#     logger.info("Registered student: %s (%s) — %d photos", name, student_id, len(photo_paths))
#     return jsonify({
#         "success":      True,
#         "message":      f"{name} registered with {len(photo_paths)} photos. Retrain model.",
#         "student":      student,
#         "photos_taken": len(photo_paths),
#     })


# @app.route("/api/delete_student", methods=["POST"])
# @login_required
# def api_delete_student():
#     student_id = (request.json or {}).get("student_id", "").strip().upper()
#     if not student_id:
#         return jsonify({"success": False, "message": "student_id required"}), 400

#     student = db.get_student(student_id)
#     if not student:
#         return jsonify({"success": False, "message": "Student not found"}), 404

#     # Remove dataset folder
#     name        = student["name"]
#     folder_name = f"{student_id}_{name.replace(' ', '_')}"
#     folder      = DATASET_DIR / folder_name
#     shutil.rmtree(folder, ignore_errors=True)

#     db.delete_student(student_id)
#     logger.info("Deleted student %s (%s)", name, student_id)
#     return jsonify({"success": True, "message": f"{name} deleted"})


# @app.route("/api/students")
# @login_required
# def api_students():
#     return jsonify(db.get_all_students())


# # ─────────────────────────────────────────────────────────────────────────────
# #  REST API — Model training
# # ─────────────────────────────────────────────────────────────────────────────

# @app.route("/api/train", methods=["POST"])
# @login_required
# def api_train():
#     """Kick off training in a background thread and return immediately."""

#     if recog_state.get("training_progress") == "running":
#         return jsonify({"success": False, "message": "Training already in progress"}), 409

#     def _train_bg():
#         recog_state["training_progress"] = "running"
#         recog_state["running"] = False          # pause recognition during training

#         def progress_cb(cur, total, msg):
#             recog_state["training_progress"] = f"{cur}/{total}: {msg}"

#         result = trainer.train(progress_callback=progress_cb)
#         recog_state["training_progress"] = (
#             "done" if result["success"] else f"error: {result['message']}"
#         )
#         logger.info("Training result: %s", result["message"])

#     t = threading.Thread(target=_train_bg, daemon=True, name="ModelTrainer")
#     t.start()
#     return jsonify({"success": True, "message": "Training started in background"})


# @app.route("/api/train_status")
# @login_required
# def api_train_status():
#     return jsonify({
#         "progress": recog_state.get("training_progress"),
#         "model":    engine.get_model_stats(),
#     })


# # ─────────────────────────────────────────────────────────────────────────────
# #  CSV Download
# # ─────────────────────────────────────────────────────────────────────────────

# @app.route("/download/attendance")
# @login_required
# def download_attendance():
#     date_str = request.args.get("date", _today_str())
#     session_id = request.args.get("session", type=int)
#     csv_data = attendmgr.export_csv(date_str, session_id=session_id)
#     buf      = io.BytesIO(csv_data.encode("utf-8"))
#     name_suffix = f"_{session_id}" if session_id else ""
#     return send_file(
#         buf,
#         mimetype    = "text/csv",
#         as_attachment = True,
#         download_name = f"attendance_{date_str}{name_suffix}.csv",
#     )


# @app.route("/download/attendance_range")
# @login_required
# def download_attendance_range():
#     start = request.args.get("start", _today_str())
#     end   = request.args.get("end", _today_str())
#     csv_data = attendmgr.export_range_csv(start, end)
#     buf      = io.BytesIO(csv_data.encode("utf-8"))
#     return send_file(
#         buf,
#         mimetype      = "text/csv",
#         as_attachment  = True,
#         download_name  = f"attendance_{start}_to_{end}.csv",
#     )


# # ─────────────────────────────────────────────────────────────────────────────
# #  Startup
# # ─────────────────────────────────────────────────────────────────────────────

# def startup():
#     Path("logs").mkdir(exist_ok=True)
#     DATASET_DIR.mkdir(exist_ok=True)
#     Path("models").mkdir(exist_ok=True)
#     active_session = db.get_active_session()
#     if active_session and active_session["session_date"] != _today_str():
#         db.stop_class_session(active_session["id"])
#     engine.load_model()
#     recog_state["today_count"] = db.count_attendance_records_by_date(_today_str())
#     logger.info("=" * 55)
#     logger.info("  SmartAttend — Face Recognition Attendance System")
#     logger.info("  Open  http://<raspberry-pi-ip>:5000  in browser")
#     logger.info("=" * 55)


# if __name__ == "__main__":
#     startup()
#     app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)




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
import webbrowser
from pathlib import Path
from datetime import datetime, date
from functools import wraps

from flask import (Flask, render_template, request, jsonify,
                   Response, redirect, url_for, session, send_file,
                   flash, abort)

from config import settings
# SmartAttend modules
from core.face_engine      import BackendUnavailableError, FaceEngine
from core.camera           import CameraManager, mjpeg_stream_generator
from core.attendance       import AttendanceManager
from core.college          import (
    BTECH_BRANCHES,
    BRANCH_CODES,
    PROGRAM_NAME,
    YEAR_OPTIONS,
    normalize_branch,
)
from core.trainer          import ModelTrainer
from core.gpio_indicator   import GPIOIndicator
from database.db_manager   import DBManager
from routes.serializers    import serialize_face
from services.audio_feedback import AttendanceAudioFeedback
from services.security_pipeline import SecurityPipeline
from services.snapshot_service import SnapshotService
from utils.image_utils     import safe_relative_to


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
PHOTOS_PER_STUDENT = int(os.environ.get("PHOTOS_PER_STUDENT", "60"))
ADMIN_USERNAME = os.environ.get("ADMIN_USER",  "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASS",  "smartattend2024")
CAMERA_BACKEND = os.environ.get("CAMERA",      "auto")   # auto|picamera2|opencv
CAMERA_DEVICE  = int(os.environ.get("CAM_DEV", "0"))
GPIO_ENABLED   = os.environ.get("GPIO", "false").lower() == "true"
AUDIO_FEEDBACK_ENABLED = os.environ.get("ATTENDANCE_AUDIO", "true").lower() == "true"
AUTO_OPEN_BROWSER = os.environ.get(
    "AUTO_OPEN_BROWSER",
    "true" if os.name == "nt" else "false",
).lower() == "true"
AUTO_OPEN_BROWSER_URL = os.environ.get("AUTO_OPEN_BROWSER_URL", "http://127.0.0.1:5000")
AUTO_OPEN_BROWSER_DELAY_SEC = max(
    float(os.environ.get("AUTO_OPEN_BROWSER_DELAY_SEC", "1.5")),
    0.0,
)
RECOGNITION_FPS = max(float(os.environ.get("RECOG_FPS", "30")), 1.0)
STREAM_FPS      = max(int(os.environ.get("STREAM_FPS", "30")), 1)
STREAM_QUALITY  = max(int(os.environ.get("STREAM_JPEG_QUALITY", "70")), 40)
OVERLAY_TTL_SEC = max(float(os.environ.get("OVERLAY_TTL_SEC", "0.45")), 0.1)
ATTEND_CONFIRM_FRAMES = max(int(os.environ.get("ATTEND_CONFIRM_FRAMES", "5")), 1)

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
audio_feedback = AttendanceAudioFeedback(
    enabled=AUDIO_FEEDBACK_ENABLED,
    beep_enabled=not GPIO_ENABLED,
)
trainer   = ModelTrainer(engine)
camera    = None        # initialised lazily on first /video_feed request
security_pipeline = SecurityPipeline()
snapshot_service = SnapshotService()

# ── Shared recognition state ──────────────────────────────────────────────────
recog_state = {
    "running":          False,
    "last_face_name":   None,
    "last_status":      None,
    "last_message":     None,
    "last_timestamp":   None,
    "last_marked_name": None,
    "last_marked_time": None,
    "frames_processed": 0,
    "today_count":      0,
    "last_distance":    None,
    "last_similarity":  None,
    "last_confidence":  None,
    "last_match_index": None,
    "last_processing_ms": 0.0,
    "last_backend":     None,
    "camera_fps":       0.0,
    "recognition_fps":  0.0,
    "last_faces":       [],
    "last_overlay_at":  0.0,
    "last_liveness_score": 0.0,
    "last_spoof_score": 0.0,
    "last_challenge":   None,
    "last_spoof_reason": None,
    "spoof_attempts_today": 0,
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


@app.context_processor
def inject_college_context():
    return {
        "program_name": PROGRAM_NAME,
        "branch_options": BTECH_BRANCHES,
        "year_options": YEAR_OPTIONS,
        "photos_per_student": PHOTOS_PER_STUDENT,
    }


def _today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


def _active_session_bundle():
    active_session = db.get_active_session()
    active_summary = (
        attendmgr.get_summary(session_id=active_session["id"])
        if active_session else None
    )
    return active_session, active_summary


def _serialize_face(face):
    return serialize_face(face)


def _attendance_payload(date_str: str, session_id: int | None = None):
    records = attendmgr.get_attendance_by_date(date_str, session_id=session_id)
    summary = (
        attendmgr.get_summary(session_id=session_id)
        if session_id
        else attendmgr.get_summary(date_str)
    )
    sessions = db.get_sessions_by_date(date_str)
    selected_session = next((item for item in sessions if item["id"] == session_id), None)
    return {
        "success": True,
        "date": date_str,
        "session_id": session_id,
        "selected_session": selected_session,
        "summary": summary,
        "count": len(records),
        "records": records,
    }


def _stream_overlay_callback(frame, frame_ts):
    with recog_lock:
        faces = recog_state["last_faces"]
        overlay_at = recog_state["last_overlay_at"]

    if not faces or (frame_ts and (time.time() - overlay_at) > OVERLAY_TTL_SEC):
        return frame

    engine.draw_results(frame, faces)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  Page routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
@login_required
def dashboard():
    today_str = _today_str()
    summary = attendmgr.get_summary(today_str)
    active_session, active_summary = _active_session_bundle()
    trend = attendmgr.get_weekly_trend()
    today = attendmgr.get_today_attendance()[:10]
    dept = db.get_dept_breakdown(today_str)
    stats = engine.get_model_stats()
    return render_template("dashboard.html",
                           summary=summary, trend=trend,
                           recent=today, dept=dept,
                           active_session=active_session,
                           active_summary=active_summary,
                           batch_matrix=db.get_btech_batch_matrix(),
                           today_sessions=db.get_sessions_by_date(today_str),
                           unassigned_students=db.count_unassigned_students(),
                           model_stats=stats,
                           recog=recog_state)


@app.route("/attendance")
@login_required
def attendance_page():
    date_str = request.args.get("date", _today_str())
    session_id = request.args.get("session", type=int)
    sessions = db.get_sessions_by_date(date_str)
    valid_session_ids = {item["id"] for item in sessions}
    if session_id not in valid_session_ids:
        session_id = None
    selected_session = next(
        (item for item in sessions if item["id"] == session_id),
        None,
    )
    records = attendmgr.get_attendance_by_date(date_str, session_id=session_id)
    summary = (
        attendmgr.get_summary(session_id=session_id)
        if session_id else attendmgr.get_summary(date_str)
    )
    wants_json = (
        request.args.get("format") == "json"
        or request.accept_mimetypes.best == "application/json"
    )
    if wants_json:
        return jsonify(_attendance_payload(date_str, session_id=session_id))
    return render_template("attendance.html",
                           records=records, summary=summary,
                           selected_date=date_str,
                           sessions=sessions,
                           selected_session_id=session_id,
                           selected_session=selected_session)


@app.route("/students")
@login_required
def students_page():
    query    = request.args.get("q", "")
    students = (db.search_students(query) if query else db.get_all_students())
    return render_template("students.html",
                           students=students, query=query)


@app.route("/students/graphs")
@login_required
def student_graphs_page():
    selected_date = request.args.get("date", _today_str())
    selected_year = request.args.get("year", type=int) or YEAR_OPTIONS[0]
    if selected_year not in YEAR_OPTIONS:
        selected_year = YEAR_OPTIONS[0]

    try:
        initial_overview = db.get_year_student_presence_overview(selected_year, selected_date)
    except ValueError:
        selected_date = _today_str()
        initial_overview = db.get_year_student_presence_overview(selected_year, selected_date)

    return render_template(
        "student_graphs.html",
        selected_date=selected_date,
        selected_year=selected_year,
        initial_overview=initial_overview,
    )


@app.route("/students/<student_id>")
@login_required
def student_detail_page(student_id: str):
    student_id = student_id.strip().upper()
    student = db.get_student(student_id)
    if not student:
        abort(404, description="Student not found")

    selected_date = request.args.get("date", _today_str())
    try:
        initial_timeline = db.get_student_presence_timeline(student_id, selected_date)
    except ValueError:
        selected_date = _today_str()
        initial_timeline = db.get_student_presence_timeline(student_id, selected_date)

    return render_template(
        "student_detail.html",
        student=student,
        selected_date=selected_date,
        initial_timeline=initial_timeline,
        attendance_history=attendmgr.get_student_history(student_id)[:20],
    )


@app.route("/register")
@login_required
def register_page():
    return render_template("register.html")


@app.route("/admin")
@login_required
def admin_page():
    stats  = engine.get_model_stats()
    return render_template("admin.html",
                           model_stats=stats,
                           recog=recog_state,
                           today_spoof_count=db.count_spoof_logs_by_date(_today_str()))


@app.route("/live")
@login_required
def live_page():
    active_session, active_summary = _active_session_bundle()
    return render_template("live.html",
                           active_session=active_session,
                           active_summary=active_summary)


# ─────────────────────────────────────────────────────────────────────────────
#  Video feed
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/video_feed")
@login_required
def video_feed():
    _ensure_camera()
    gen = mjpeg_stream_generator(
        camera,
        overlay_callback=_stream_overlay_callback,
        fps_limit=STREAM_FPS,
        quality=STREAM_QUALITY,
    )
    return Response(gen, mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/evidence/<path:relpath>")
@login_required
def evidence_file(relpath: str):
    target = settings.evidence_dir / relpath
    if not safe_relative_to(target, settings.evidence_dir) or not target.exists():
        abort(404)
    return send_file(target)


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
    target_interval = 1.0 / RECOGNITION_FPS
    last_loop_started = time.time()
    confirmation_counts: dict[str, int] = {}
    confirmation_session_id = None

    while recog_state["running"]:
        loop_started = time.time()
        try:
            frame = camera.get_frame(copy=False)
            if frame is None:
                time.sleep(0.1)
                continue

            result = engine.process_frame(frame, draw_boxes=False)

            active_session = db.get_active_session()
            active_session_id = (
                int(active_session["id"])
                if active_session and active_session.get("id")
                else None
            )
            if active_session_id != confirmation_session_id:
                confirmation_counts.clear()
                confirmation_session_id = active_session_id
                security_pipeline.set_session(active_session_id)

            security_pipeline.evaluate(frame, result.faces)

            with recog_lock:
                recog_state["frames_processed"] += 1
                elapsed = max(loop_started - last_loop_started, 1e-6)
                recog_state["recognition_fps"] = round(1.0 / elapsed, 2)
               # recog_state["camera_fps"] = round(camera.capture_fps if camera else 0.0, 2)
                recog_state["camera_fps"] = 0.0
                recog_state["last_processing_ms"] = round(result.processing_time_ms, 2)
                recog_state["last_backend"] = result.backend
                recog_state["last_faces"] = list(result.faces)
                recog_state["last_overlay_at"] = time.time()
            last_loop_started = loop_started

            current_frame_ids = set()
            for face in result.faces:
                if (
                    not face.student_id
                    or face.student_id in current_frame_ids
                    or not face.live_verified
                    or face.spoof_detected
                ):
                    continue
                current_frame_ids.add(face.student_id)
                confirmation_counts[face.student_id] = confirmation_counts.get(face.student_id, 0) + 1
            confirmation_counts = {
                student_id: confirmation_counts[student_id]
                for student_id in current_frame_ids
                if student_id in confirmation_counts
            }

            for face in result.faces:
                if face.should_log_spoof:
                    snapshot_path = snapshot_service.save_snapshot(
                        frame=frame,
                        box=face.bounding_box,
                        bucket="spoof",
                        student_id=face.student_id,
                        session_id=active_session_id,
                        prefix="spoof",
                    )
                    db.insert_spoof_log(
                        student_id=face.student_id,
                        name=face.name,
                        session_id=active_session_id,
                        date_str=_today_str(),
                        time_str=datetime.now().strftime("%H:%M:%S"),
                        reason=", ".join(face.spoof_reasons) or "spoof_suspected",
                        confidence=face.confidence,
                        liveness_score=face.liveness_score,
                        spoof_score=face.spoof_score,
                        challenge=face.challenge_text,
                        snapshot_path=snapshot_path,
                        metadata={
                            "status": face.status,
                            "yaw": face.yaw,
                            "pitch": face.pitch,
                            "roll": face.roll,
                            "blink_count": face.blink_count,
                        },
                    )

                confirmed_frames = confirmation_counts.get(face.student_id, 0)
                if face.spoof_detected:
                    outcome = {
                        "status": "spoof_detected",
                        "message": "Spoof attempt rejected",
                        "record": None,
                    }
                elif not face.is_known or not face.student_id:
                    outcome = {
                        "status": "unknown",
                        "message": "Unknown face",
                        "record": None,
                    }
                elif not face.live_verified:
                    outcome = {
                        "status": face.status,
                        "message": face.status_text,
                        "record": None,
                    }
                elif confirmed_frames < ATTEND_CONFIRM_FRAMES:
                    outcome = {
                        "status": "confirming",
                        "message": (
                            f"{face.name} detected {confirmed_frames}/"
                            f"{ATTEND_CONFIRM_FRAMES} frames"
                        ),
                        "record": None,
                    }
                else:
                    outcome = attendmgr.mark_attendance(
                        student_id = face.student_id,
                        name       = face.name,
                        confidence = face.confidence,
                        liveness_score = face.liveness_score,
                        active_session = active_session,
                    )
                    if outcome["status"] == "marked" and outcome.get("record"):
                        snapshot_path = snapshot_service.save_snapshot(
                            frame=frame,
                            box=face.bounding_box,
                            bucket="attendance",
                            student_id=face.student_id,
                            session_id=active_session_id,
                            prefix="attendance",
                        )
                        record = db.update_attendance_evidence(
                            attendance_id=outcome["record"]["id"],
                            liveness_score=face.liveness_score,
                            snapshot_path=snapshot_path,
                        )
                        if record:
                            outcome["record"] = record
                if (
                    outcome["status"] in {"marked", "duplicate"}
                    and active_session
                    and face.student_id
                ):
                    db.log_student_presence(
                        student_id=face.student_id,
                        session_id=int(active_session["id"]),
                        seen_at=datetime.now(),
                    )
                with recog_lock:
                    recog_state["last_face_name"] = face.name
                    recog_state["last_status"]    = outcome["status"]
                    recog_state["last_message"]   = outcome.get("message")
                    recog_state["last_timestamp"] = datetime.now().strftime("%H:%M:%S")
                    recog_state["last_distance"] = face.distance
                    recog_state["last_similarity"] = face.similarity
                    recog_state["last_confidence"] = face.confidence
                    recog_state["last_match_index"] = face.matched_index
                    recog_state["last_liveness_score"] = face.liveness_score
                    recog_state["last_spoof_score"] = face.spoof_score
                    recog_state["last_challenge"] = face.challenge_text
                    recog_state["last_spoof_reason"] = ", ".join(face.spoof_reasons) if face.spoof_reasons else None
                    recog_state["spoof_attempts_today"] = db.count_spoof_logs_by_date(_today_str())
                    if outcome["status"] == "marked":
                        recog_state["last_marked_name"] = face.name
                        recog_state["last_marked_time"] = recog_state["last_timestamp"]
                        recog_state["today_count"] = db.count_attendance_records_by_date(_today_str())

                if outcome["status"] == "marked":
                    audio_feedback.clear_prompt_cache(face.track_key)
                    audio_feedback.on_attendance_marked(face.name)
                    gpio.on_attendance_marked(face.name)
                elif outcome["status"] == "duplicate":
                    gpio.on_duplicate(face.name)
                elif outcome["status"] in {"challenge_pending", "liveness_pending", "processing"}:
                    audio_feedback.on_guidance(
                        face.track_key,
                        face.challenge_text or face.status_text,
                    )
                elif outcome["status"] in {"confirming", "awaiting_blink"}:
                    pass
                else:
                    gpio.on_unknown_face()

            if not result.faces:
                confirmation_counts.clear()
                with recog_lock:
                    recog_state["last_faces"] = []

            elapsed = time.time() - loop_started
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

        except BackendUnavailableError as exc:
            with recog_lock:
                recog_state["running"] = False
                recog_state["last_status"] = "error"
                recog_state["last_message"] = str(exc)
                recog_state["last_timestamp"] = datetime.now().strftime("%H:%M:%S")
                recog_state["last_faces"] = []
                recog_state["recognition_fps"] = 0.0
            logger.error("Recognition stopped: %s", exc)
            break

        except Exception as exc:
            logger.exception("Recognition loop error: %s", exc)
            time.sleep(1.0)

    logger.info("Recognition loop stopped")


@app.route("/api/recognition/start", methods=["POST"])
@login_required
def api_start_recognition():
    if recog_state["running"]:
        return jsonify({
            "success": True,
            "status": "already_running",
            "message": "Recognition is already running.",
        })

    active_session = db.get_active_session()
    if not active_session:
        return jsonify({
            "success": False,
            "status": "error",
            "message": "Start the running class session first.",
        }), 400

    if not engine.is_model_loaded():
        ok = engine.load_model()
        if not ok:
            return jsonify({"success": False, "status": "error",
                            "message": "No model loaded. Train first."}), 400

    try:
        engine.ensure_runtime_ready()
    except BackendUnavailableError as exc:
        with recog_lock:
            recog_state["last_status"] = "error"
            recog_state["last_message"] = str(exc)
            recog_state["last_timestamp"] = datetime.now().strftime("%H:%M:%S")
            recog_state["last_faces"] = []
            recog_state["recognition_fps"] = 0.0
        return jsonify({
            "success": False,
            "status": "error",
            "message": str(exc),
        }), 400

    security_pipeline.set_session(int(active_session["id"]))
    recog_state["running"] = True
    t = threading.Thread(target=recognition_loop, daemon=True,
                         name="RecognitionLoop")
    t.start()
    return jsonify({"success": True, "status": "started", "message": "Recognition started."})


@app.route("/api/recognition/stop", methods=["POST"])
@login_required
def api_stop_recognition():
    recog_state["running"] = False
    security_pipeline.reset()
    with recog_lock:
        recog_state["last_faces"] = []
    return jsonify({"success": True, "status": "stopped", "message": "Recognition stopped."})


@app.route("/api/session/start", methods=["POST"])
@login_required
def api_start_session():
    data = request.json or {}
    department = normalize_branch(data.get("department", ""))
    room = data.get("room", "").strip().upper()
    subject = data.get("subject", "").strip()
    section = data.get("section", "").strip().upper()

    try:
        year = int(data.get("year", 1))
    except (TypeError, ValueError):
        year = 0

    if department not in BRANCH_CODES:
        return jsonify({"success": False, "message": "Select a valid B.Tech branch."}), 400
    if year not in YEAR_OPTIONS:
        return jsonify({"success": False, "message": "Select a valid year."}), 400
    if not room:
        return jsonify({"success": False, "message": "Room is required."}), 400

    session_row = db.start_class_session(
        PROGRAM_NAME,
        department,
        year,
        room,
        subject,
        section,
    )
    attendmgr.clear_session_cache(session_row["id"])
    security_pipeline.set_session(int(session_row["id"]))
    return jsonify({
        "success": True,
        "message": f"Class session started for {session_row['display_label']}.",
        "session": session_row,
    })


@app.route("/api/session/stop", methods=["POST"])
@login_required
def api_stop_session():
    session_row = db.stop_class_session()
    recog_state["running"] = False
    security_pipeline.reset()
    with recog_lock:
        recog_state["last_faces"] = []
    if not session_row:
        return jsonify({"success": False, "message": "No active class session."}), 404
    attendmgr.clear_session_cache(session_row["id"])
    return jsonify({
        "success": True,
        "message": f"Class session closed for {session_row['display_label']}.",
        "session": session_row,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  REST API — Stats / Attendance
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/live_status")
@login_required
def api_live_status():
    active_session, active_summary = _active_session_bundle()
    today_count = db.count_attendance_records_by_date(_today_str())
    with recog_lock:
        return jsonify({
            "success":        True,
            "running":        recog_state["running"],
            "last_name":      recog_state["last_face_name"],
            "last_status":    recog_state["last_status"],
            "last_message":   recog_state["last_message"],
            "last_time":      recog_state["last_timestamp"],
            "last_marked_name": recog_state["last_marked_name"],
            "last_marked_time": recog_state["last_marked_time"],
            "last_distance":  recog_state["last_distance"],
            "last_similarity": recog_state["last_similarity"],
            "last_confidence": recog_state["last_confidence"],
            "last_match_index": recog_state["last_match_index"],
            "last_processing_ms": recog_state["last_processing_ms"],
            "last_liveness_score": recog_state["last_liveness_score"],
            "last_spoof_score": recog_state["last_spoof_score"],
            "last_challenge": recog_state["last_challenge"],
            "last_spoof_reason": recog_state["last_spoof_reason"],
            "backend":        recog_state["last_backend"],
            "camera_fps":     recog_state["camera_fps"],
            "recognition_fps": recog_state["recognition_fps"],
            "faces":          [_serialize_face(face) for face in recog_state["last_faces"]],
            "today_count":    today_count,
            "spoof_attempts_today": db.count_spoof_logs_by_date(_today_str()),
            "session_count":  (active_summary or {}).get("present", 0),
            "active_session": active_session,
            "active_summary": active_summary,
            "frames":         recog_state["frames_processed"],
        })


@app.route("/api/stats")
@login_required
def api_stats():
    date_str = request.args.get("date", _today_str())
    active_session, active_summary = _active_session_bundle()
    return jsonify({
        "success": True,
        "summary": attendmgr.get_summary(date_str),
        "dept":    db.get_dept_breakdown(date_str),
        "active_session": active_session,
        "active_summary": active_summary,
    })


@app.route("/api/weekly_trend")
@login_required
def api_weekly_trend():
    return jsonify(attendmgr.get_weekly_trend())


@app.route("/api/today_attendance")
@login_required
def api_today_attendance():
    session_id = request.args.get("session", type=int)
    return jsonify(_attendance_payload(_today_str(), session_id=session_id))


@app.route("/api/attendance")
@login_required
def api_attendance():
    date_str = request.args.get("date", _today_str())
    session_id = request.args.get("session", type=int)
    return jsonify(_attendance_payload(date_str, session_id=session_id))


@app.route("/api/spoof_logs")
@login_required
def api_spoof_logs():
    date_str = request.args.get("date") or _today_str()
    session_id = request.args.get("session", type=int)
    limit = request.args.get("limit", default=25, type=int)
    return jsonify({
        "success": True,
        "date": date_str,
        "count": db.count_spoof_logs_by_date(date_str),
        "logs": db.get_spoof_logs(limit=limit, date_str=date_str, session_id=session_id),
    })


@app.route("/api/student_presence_timeline/<student_id>")
@login_required
def api_student_presence_timeline(student_id: str):
    student_id = student_id.strip().upper()
    if not db.student_exists(student_id):
        return jsonify({"success": False, "message": "Student not found"}), 404

    date_str = request.args.get("date", _today_str())
    try:
        return jsonify(db.get_student_presence_timeline(student_id, date_str))
    except ValueError:
        return jsonify({
            "success": False,
            "message": "Use date format YYYY-MM-DD.",
        }), 400


@app.route("/api/student_presence_overview")
@login_required
def api_student_presence_overview():
    date_str = request.args.get("date", _today_str())
    year = request.args.get("year", type=int) or YEAR_OPTIONS[0]
    if year not in YEAR_OPTIONS:
        return jsonify({
            "success": False,
            "message": "Select a valid year.",
        }), 400

    try:
        return jsonify(db.get_year_student_presence_overview(year, date_str))
    except ValueError:
        return jsonify({
            "success": False,
            "message": "Use date format YYYY-MM-DD.",
        }), 400


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
    department = normalize_branch(data.get("department", "").strip())
    try:
        year = int(data.get("year", 1))
    except (TypeError, ValueError):
        year = 0
    email      = data.get("email", "").strip()
    phone      = data.get("phone", "").strip()

    if not student_id or not name:
        return jsonify({"success": False, "message": "student_id and name required"}), 400
    if department not in BRANCH_CODES:
        return jsonify({"success": False, "message": "Select a valid B.Tech branch"}), 400
    if year not in YEAR_OPTIONS:
        return jsonify({"success": False, "message": "Year must be between 1 and 4"}), 400

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
    date_str = request.args.get("date", _today_str())
    session_id = request.args.get("session", type=int)
    csv_data = attendmgr.export_csv(date_str, session_id=session_id)
    buf      = io.BytesIO(csv_data.encode("utf-8"))
    name_suffix = f"_{session_id}" if session_id else ""
    return send_file(
        buf,
        mimetype    = "text/csv",
        as_attachment = True,
        download_name = f"attendance_{date_str}{name_suffix}.csv",
    )


@app.route("/download/presence_report")
@login_required
def download_presence_report():
    date_str = request.args.get("date", _today_str())
    session_id = request.args.get("session", type=int)
    interval_minutes = max(request.args.get("interval", default=60, type=int) or 60, 5)
    csv_data = attendmgr.export_presence_interval_csv(
        date_str,
        session_id=session_id,
        interval_minutes=interval_minutes,
    )
    buf = io.BytesIO(csv_data.encode("utf-8"))
    name_suffix = f"_{session_id}" if session_id else ""
    return send_file(
        buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"presence_{date_str}{name_suffix}_{interval_minutes}min.csv",
    )


@app.route("/download/attendance_range")
@login_required
def download_attendance_range():
    start = request.args.get("start", _today_str())
    end   = request.args.get("end", _today_str())
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
    active_session = db.get_active_session()
    if active_session and active_session["session_date"] != _today_str():
        db.stop_class_session(active_session["id"])
    engine.load_model()
    recog_state["today_count"] = db.count_attendance_records_by_date(_today_str())
    if AUTO_OPEN_BROWSER:
        threading.Timer(AUTO_OPEN_BROWSER_DELAY_SEC, _open_browser_on_startup).start()
    logger.info("=" * 55)
    logger.info("  SmartAttend — Face Recognition Attendance System")
    logger.info("  Open  http://<raspberry-pi-ip>:5000  in browser")
    logger.info("=" * 55)


def _open_browser_on_startup():
    try:
        opened = webbrowser.open(AUTO_OPEN_BROWSER_URL, new=2)
        if opened:
            logger.info("Opened SmartAttend in browser: %s", AUTO_OPEN_BROWSER_URL)
        else:
            logger.warning("Browser auto-open returned false for %s", AUTO_OPEN_BROWSER_URL)
    except Exception as exc:
        logger.warning("Browser auto-open failed: %s", exc)


if __name__ == "__main__":
    startup()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)































































































































