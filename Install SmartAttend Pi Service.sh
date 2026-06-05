#!/bin/bash
set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="/etc/systemd/system/smartattend.service"
CURRENT_USER="${SUDO_USER:-$USER}"

sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=SmartAttend Face Recognition Attendance System
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$BASE_DIR
Environment=AUTO_OPEN_BROWSER=false
ExecStart=$BASE_DIR/venv/bin/python $BASE_DIR/app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable smartattend.service
echo "SmartAttend system service installed."
echo "Start it with: sudo systemctl start smartattend.service"
