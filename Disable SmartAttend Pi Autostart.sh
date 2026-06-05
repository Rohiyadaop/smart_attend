#!/bin/bash
set -e

AUTOSTART_FILE="${HOME}/.config/autostart/SmartAttend.desktop"

if [ -f "$AUTOSTART_FILE" ]; then
  rm -f "$AUTOSTART_FILE"
  echo "SmartAttend desktop autostart removed."
else
  echo "SmartAttend desktop autostart was not found."
fi
