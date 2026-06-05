#!/bin/bash
set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$BASE_DIR/Start SmartAttend Pi.sh"
DESKTOP_DIR="${HOME}/Desktop"
SHORTCUT_PATH="${DESKTOP_DIR}/SmartAttend.desktop"

mkdir -p "$DESKTOP_DIR"

cat > "$SHORTCUT_PATH" <<EOF
[Desktop Entry]
Type=Application
Name=SmartAttend
Comment=Start SmartAttend and open the web dashboard
Path=$BASE_DIR
Exec=/bin/bash "$LAUNCHER"
Icon=web-browser
Terminal=true
Categories=Education;
EOF

chmod +x "$SHORTCUT_PATH"
chmod +x "$LAUNCHER"
echo "Desktop shortcut created at: $SHORTCUT_PATH"
