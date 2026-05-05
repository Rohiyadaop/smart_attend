# SmartAttend Pilot Deployment Guide for Raspberry Pi 4

Prepared for pilot deployment in a college classroom or lab.

## 1. Pilot deployment goal

This pilot setup is for:

- one Raspberry Pi 4
- one camera
- one classroom, lab, or entry point
- local web dashboard for teacher/admin use

Use the pilot first before full college-wide deployment.

## 2. Recommended hardware

- Raspberry Pi 4 Model B, 4 GB or 8 GB RAM
- 32 GB or larger microSD card, Class 10
- Official Raspberry Pi power adapter
- Raspberry Pi Camera Module or a supported USB webcam
- Ethernet connection preferred over Wi-Fi
- Monitor, keyboard, and mouse for first-time setup
- Optional: case, heatsink, cooling fan, UPS backup

## 3. Before you start

Keep these ready:

- admin username and password
- classroom name or room number
- list of students with roll numbers
- 5 to 10 face images per student if you want to pre-train
- permission from HOD or college authority for pilot use

## 4. Install Raspberry Pi OS

1. Flash Raspberry Pi OS 64-bit using Raspberry Pi Imager.
2. Boot the Raspberry Pi 4.
3. Complete the first-time setup.
4. Connect the Pi to the college LAN using Ethernet if possible.
5. Open terminal and update the system.

```bash
sudo apt update
sudo apt upgrade -y
```

## 5. Enable camera

If using Raspberry Pi Camera Module:

```bash
sudo raspi-config
```

Then:

- Interface Options
- Camera
- Enable
- Reboot

After reboot, test camera:

```bash
libcamera-hello
```

If using USB webcam, this step is not required.

## 6. Install system packages

```bash
sudo apt install -y python3-pip python3-venv git cmake build-essential \
libboost-all-dev libatlas-base-dev libhdf5-dev python3-picamera2
```

If you are using USB webcam only, `python3-picamera2` is optional.

## 7. Clone project and create virtual environment

```bash
git clone <your-repository-url>
cd smartattend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 8. Install dlib / face-recognition dependencies

On Raspberry Pi, `dlib` may take time to install.

```bash
pip install dlib
pip install face-recognition
```

If `dlib` build fails, use a compatible wheel for your Pi OS and Python version.

## 9. Configure SmartAttend environment

Set environment variables before running:

```bash
export ADMIN_USER=admin
export ADMIN_PASS=ChangeThisNow
export CAMERA=picamera2
export GPIO=false
export SECRET_KEY=change-this-secret-key
```

If using USB webcam:

```bash
export CAMERA=opencv
```

For permanent setup, place these in the systemd service file later.

## 10. Prepare project folders

When the app starts, it uses:

- `dataset/`
- `models/`
- `logs/`
- `smartattend.db`

Make sure your project folder has write permission for the Pi user.

## 11. Start the app for first test

```bash
python app.py
```

Open in browser:

```text
http://<raspberry-pi-ip>:5000
```

Login with the admin username and password you configured.

## 12. Register students

There are two good pilot options:

### Option A: register from live camera

- open the Register Student page
- enter student details
- capture photos from the camera

### Option B: add student folders manually

Use folder pattern:

```text
dataset/STUDENTID_Name/
```

Example:

```text
dataset/221340105006_Rudranil_Adhikari/
```

Put multiple clear images inside each folder.

## 13. Train the face model

After registering students:

```bash
python scripts/train_model.py
```

Or use the training option from the web admin page.

Confirm that `models/encodings.pkl` is created.

## 14. Test pilot in real classroom conditions

Check all of these before live use:

- camera sees faces clearly
- lighting is even
- room entry angle is good
- student IDs and names are correct
- recognition marks attendance correctly
- year/session filtering works correctly
- individual graph and year graph load correctly

## 15. Recommended camera placement

- mount at face level, about 1.5 m to 1.7 m height
- avoid strong backlight from windows
- do not point directly into sunlight
- keep the camera stable
- prefer fixed placement near classroom entry or front wall

## 16. Run SmartAttend with Gunicorn

For pilot production use:

```bash
source venv/bin/activate
gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 app:app
```

This is better than running `python app.py` all the time.

## 17. Run on boot using systemd

Create service file:

```bash
sudo nano /etc/systemd/system/smartattend.service
```

Paste this and update paths if needed:

```ini
[Unit]
Description=SmartAttend Face Recognition System
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/smartattend
Environment=PATH=/home/pi/smartattend/venv/bin
Environment=ADMIN_USER=admin
Environment=ADMIN_PASS=ChangeThisNow
Environment=CAMERA=picamera2
Environment=GPIO=false
Environment=SECRET_KEY=change-this-secret-key
ExecStart=/home/pi/smartattend/venv/bin/gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 app:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable smartattend
sudo systemctl start smartattend
sudo systemctl status smartattend
```

## 18. Set a fixed IP

For teacher access, avoid changing IP addresses.

You can:

- reserve the Pi IP in the college router, or
- configure a static IP on the Pi

Also set hostname if useful:

```bash
sudo raspi-config
```

Then teachers may access using:

```text
http://smartattend.local:5000
```

if local name resolution works on your network.

## 19. Daily operating flow for pilot

1. Power on Raspberry Pi.
2. Confirm network is connected.
3. Open SmartAttend dashboard.
4. Start class session.
5. Start recognition.
6. Monitor live camera and attendance marking.
7. Stop session after class.
8. Export attendance CSV if needed.

## 20. Backup plan

Back up these items regularly:

- `smartattend.db`
- `models/encodings.pkl`
- `dataset/`
- `logs/`

Example backup command:

```bash
mkdir -p ~/smartattend-backups
cp smartattend.db ~/smartattend-backups/smartattend-$(date +%F).db
```

## 21. Security and privacy checklist

- change default admin password
- do not expose the app to the public internet
- keep it on college LAN only
- restrict admin access to authorized staff
- inform students about pilot usage
- follow college data/privacy policy
- keep a manual attendance fallback during pilot

## 22. Troubleshooting

### Camera not working

```bash
libcamera-hello
```

If that fails, recheck camera cable and camera enable setting.

### USB webcam not detected

Set:

```bash
export CAMERA=opencv
```

### App not opening in browser

Check service:

```bash
sudo systemctl status smartattend
```

Check whether port 5000 is reachable on LAN.

### Face recognition accuracy is low

- add more images
- improve lighting
- avoid blurry capture
- keep camera steady
- retrain the model

### Student graph has no data

The graph depends on live presence logging while recognition is running.
Make sure recognition was active for that class/day.

## 23. Pilot success criteria

Your pilot is successful if:

- the device runs for full class hours without crashing
- attendance is marked correctly for most students
- graphs and history are visible to staff
- teachers can export records easily
- false matches stay very low

## 24. Recommended next step after pilot

After pilot success, upgrade for department or college use:

- move from SQLite to PostgreSQL/MySQL
- add teacher-specific login
- add HTTPS and reverse proxy
- add automatic backups
- add central dashboard for multiple rooms

## 25. Final note

Deploy one classroom first, observe for one or two weeks, then scale.
Pilot deployment is the safest and most practical path for SmartAttend.
