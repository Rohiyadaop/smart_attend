# SmartAttend — Smart Face Recognition Attendance System
### B.Tech Final Year Project · Computer Science & Electronics Engineering

> Automatically detects and recognises student faces using a Raspberry Pi Camera Module, marks attendance in SQLite, and serves a real-time web dashboard for monitoring and management.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     SmartAttend Architecture                     │
├──────────────────┬───────────────────────────────────────────────┤
│  HARDWARE LAYER  │  Raspberry Pi 4B  +  Camera Module v2        │
│                  │  Optional: LED (GPIO 17), Buzzer (GPIO 22)    │
├──────────────────┼───────────────────────────────────────────────┤
│  CAPTURE LAYER   │  picamera2 / OpenCV VideoCapture             │
│  (camera.py)     │  Threaded frame buffer (latest frame always  │
│                  │  available without blocking)                  │
├──────────────────┼───────────────────────────────────────────────┤
│  DPI LAYER       │  face_recognition (dlib ResNet-34)           │
│  (face_engine)   │  HOG detection → 128-D embedding → matching  │
├──────────────────┼───────────────────────────────────────────────┤
│  LOGIC LAYER     │  AttendanceManager — mark, dedup, export     │
│  (attendance.py) │  ModelTrainer — dataset scan + pickle save   │
│  (trainer.py)    │  GPIOIndicator — LED / buzzer feedback       │
├──────────────────┼───────────────────────────────────────────────┤
│  DATA LAYER      │  SQLite database — students + attendance      │
│  (db_manager.py) │  Thread-safe WAL mode · Parameterised queries │
├──────────────────┼───────────────────────────────────────────────┤
│  WEB LAYER       │  Flask REST API + Jinja2 templates           │
│  (app.py)        │  MJPEG live stream  /  Chart.js charts       │
│                  │  Session auth  /  CSV download               │
└──────────────────┴───────────────────────────────────────────────┘
                              │
              Browser ← HTTP/WebSocket → Flask
```

---

## How Face Recognition Works

### 1. HOG Detection (Stage 1)
The system uses **Histogram of Oriented Gradients (HOG)** to detect face bounding boxes:
- The camera frame is downscaled to 50% for speed
- HOG extracts gradient orientations in 8×8-pixel cells
- A sliding window + SVM classifier identifies face regions
- Result: bounding box `(top, right, bottom, left)` for each face

### 2. 128-D Embedding (Stage 2)
Each detected face region is passed through a **dlib ResNet-34** deep neural network:
- Network outputs a **128-dimensional vector** — a unique numeric "fingerprint"
- Same person from different angles/lighting → similar vectors (close in Euclidean space)
- Different people → far apart in the 128-D space
- This is why it generalises to new angles without retraining the neural network

### 3. Matching (Stage 3)
The new 128-D vector is compared to every stored training encoding:
```
distance = euclidean_distance(new_encoding, known_encoding)
match    = distance < 0.50  (tolerance)
confidence = 1.0 - (distance / 0.50)
```
The closest known encoding whose distance is below tolerance is declared the match.

### 4. Training
Training scans `dataset/{ID}_{Name}/` folders:
- Loads each image with OpenCV
- Detects face → computes 128-D embedding
- Saves all (encoding, name, student_id) tuples to `models/encodings.pkl`
- **More photos per student = better accuracy** (aim for 5–10 varied photos)

---

## Project Structure

```
smartattend/
├── app.py                       ← Flask application (all routes + API)
├── requirements.txt
├── README.md
│
├── core/
│   ├── face_engine.py           ← DPI + face recognition engine
│   ├── camera.py                ← PiCamera2 / OpenCV camera manager
│   ├── attendance.py            ← Attendance business logic
│   ├── trainer.py               ← Model trainer from dataset images
│   └── gpio_indicator.py        ← LED / buzzer GPIO control
│
├── database/
│   └── db_manager.py            ← SQLite wrapper (students + attendance)
│
├── web/
│   └── templates/
│       ├── login.html           ← Admin login page
│       ├── dashboard.html       ← Main overview + live feed
│       ├── attendance.html      ← Attendance records + date filter
│       ├── students.html        ← Student management (search/delete)
│       ├── register.html        ← New student registration
│       ├── admin.html           ← Model training + system controls
│       └── live.html            ← Fullscreen live camera view
│
├── dataset/                     ← Student face images
│   └── STU001_John_Doe/
│       ├── img_001.jpg
│       └── img_002.jpg
│
├── models/
│   └── encodings.pkl            ← Trained face encodings
│
├── logs/
│   └── smartattend.log
│
└── scripts/
    └── train_model.py           ← CLI training utility
```

---

## Raspberry Pi Setup Instructions

### Step 1 — Flash OS & Enable Camera
```bash
# Use Raspberry Pi Imager to flash Raspberry Pi OS (64-bit recommended)
# Enable camera in raspi-config:
sudo raspi-config
# → Interface Options → Camera → Enable → Reboot
```

### Step 2 — Update System
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git cmake build-essential \
    libboost-all-dev libatlas-base-dev libhdf5-dev \
    python3-picamera2
```

### Step 3 — Install dlib (required for face_recognition)
```bash
# dlib must be compiled from source on ARM — takes 15–30 min on Pi 4
pip3 install dlib
# If the above fails, use pre-compiled wheel:
pip3 install https://github.com/Melvyn-Braz/face-recognition-rpi/releases/download/v1.0/dlib-19.22.99-cp311-cp311-linux_aarch64.whl
```

### Step 4 — Clone & Install Project
```bash
git clone https://github.com/yourusername/smartattend.git
cd smartattend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 5 — Configure Environment
```bash
# Create .env file (optional — all have defaults)
export ADMIN_USER=admin
export ADMIN_PASS=YourSecurePassword
export CAMERA=picamera2          # or "opencv" for USB webcam
export GPIO=true                 # enable LED/buzzer
export SECRET_KEY=your-secret-key
```

### Step 6 — Register Students & Train Model
```bash
# Option A: Use the web dashboard at http://pi-ip:5000/register
# Option B: Manually add images to dataset/ then:
python scripts/train_model.py
```

### Step 7 — Run the Application
```bash
# Development:
python app.py

# Production (Gunicorn):
gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 app:app
```

Open in browser: **http://\<raspberry-pi-ip\>:5000**
Default login: `admin` / `smartattend2024`

---

## Run on Boot (systemd)

```bash
sudo nano /etc/systemd/system/smartattend.service
```

Paste:
```ini
[Unit]
Description=SmartAttend Face Recognition System
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/smartattend
Environment=PATH=/home/pi/smartattend/venv/bin
ExecStart=/home/pi/smartattend/venv/bin/gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 app:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable smartattend
sudo systemctl start smartattend
sudo systemctl status smartattend
```

---

## Hardware Wiring

```
Raspberry Pi 4          LED / Buzzer
─────────────           ────────────
GPIO 17 (Pin 11)  ──►  220Ω ──► GREEN LED (+)
GPIO 27 (Pin 13)  ──►  220Ω ──► RED LED (+)
GPIO 22 (Pin 15)  ──►  Buzzer (+)
GND (Pin 6, 9…)   ──►  All component (−) terminals

Camera Module     ──►  CSI connector (silver cable, blue side up)
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard (login required) |
| `/video_feed` | GET | MJPEG live camera stream |
| `/api/live_status` | GET | Current recognition state |
| `/api/stats` | GET | KPI stats for a date |
| `/api/weekly_trend` | GET | 7-day attendance chart data |
| `/api/register` | POST | Register new student + capture photos |
| `/api/delete_student` | POST | Remove student and their records |
| `/api/train` | POST | Start model training in background |
| `/api/train_status` | GET | Training progress |
| `/api/recognition/start` | POST | Start recognition loop |
| `/api/recognition/stop` | POST | Stop recognition loop |
| `/download/attendance` | GET | Download attendance CSV by date |

---

## College Deployment Guide

### Network Setup
1. Connect Raspberry Pi to college LAN via Ethernet (preferred for stability)
2. Assign a static IP via router or: `sudo nano /etc/dhcpcd.conf`
3. Optionally configure a hostname: `sudo raspi-config → Hostname → smartattend`
4. Share URL with teachers: `http://smartattend.local:5000`

### Camera Placement
- Mount at eye level (1.6–1.7m height), facing the classroom door
- Ensure **even, diffuse lighting** — avoid strong backlight from windows
- Test with `http://pi-ip:5000/live` before final installation
- Field of view: 1–1.5m detection distance works best for Pi Camera v2

### Multi-class Support
- Register students with their full roll number as Student ID
- Create department-specific folders if needed
- Export daily CSVs per class and upload to Google Sheets

### Privacy & Security
- Change default admin password immediately
- Set file permissions: `chmod 600 smartattend.db`
- Run on a private VLAN or password-protected Wi-Fi
- Face encodings are stored locally — no cloud uploads
- Comply with your institution's data protection policy
- Consider posting a camera notice at the classroom entrance

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `No module named face_recognition` | `pip install face-recognition` (needs cmake + build-essential) |
| Camera black screen | Check CSI cable; run `libcamera-hello` to test |
| Slow recognition (>5s) | Reduce frame resolution to 320×240; ensure no other heavy processes |
| `ModuleNotFoundError: picamera2` | On non-Pi: set `CAMERA=opencv` env variable |
| GPIO error | Run with `sudo` or add user to gpio group: `sudo usermod -aG gpio pi` |
| Low accuracy | Add more training photos per student (5–10 varied angles) |

---

## Performance on Raspberry Pi 4

| Metric | Value |
|---|---|
| Recognition latency | 200–400ms per frame (HOG model) |
| Camera FPS | 15–20 fps (640×480) |
| Concurrent students | Up to ~50 (higher = slower training) |
| Model training time | ~1 min for 50 students × 5 photos |
| RAM usage | ~350–500 MB |
| Storage (encodings) | ~500 KB per 100 students |

---

*Developed as a B.Tech Computer Science / ECE Final Year Project*
*Veer Madho Singh Bhandari Uttarakhand Technical University, Dehradun*
"# smart_attend" 
