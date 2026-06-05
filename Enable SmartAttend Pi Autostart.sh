#!/bin/bash
set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$BASE_DIR/Start SmartAttend Pi.sh"
AUTOSTART_DIR="${HOME}/.config/autostart"
AUTOSTART_FILE="${AUTOSTART_DIR}/SmartAttend.desktop"

mkdir -p "$AUTOSTART_DIR"

cat > "$AUTOSTART_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=SmartAttend
Comment=Start SmartAttend automatically on Raspberry Pi desktop login
Path=$BASE_DIR
Exec=/bin/bash "$LAUNCHER"
Icon=web-browser
Terminal=false
X-GNOME-Autostart-enabled=true
EOF

chmod +x "$LAUNCHER"
echo "SmartAttend desktop autostart enabled at: $AUTOSTART_FILE"
