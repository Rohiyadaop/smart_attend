"""
=============================================================================
  SmartAttend GPIO Indicator
=============================================================================
  Controls LED and buzzer on Raspberry Pi GPIO pins to give physical
  feedback when attendance is marked.

  CIRCUIT WIRING
  ─────────────────────────────────────────────────────────────────────────
  GREEN LED  (attendance marked):
    GPIO 17 (Pin 11) → 220Ω resistor → LED anode (+)
    LED cathode (−) → Ground (Pin 6)

  RED LED  (unknown face):
    GPIO 27 (Pin 13) → 220Ω resistor → LED anode (+)
    LED cathode (−) → Ground

  BUZZER (passive):
    GPIO 22 (Pin 15) → Buzzer + terminal
    Buzzer − terminal → Ground

  Falls back to no-op if RPi.GPIO is not available (dev machine).
=============================================================================
"""

import time
import logging
import threading

logger = logging.getLogger("smartattend.gpio")

# ─── GPIO pin assignments (BCM numbering) ─────────────────────────────────────
PIN_GREEN  = 17    # Green LED — attendance marked
PIN_RED    = 27    # Red LED   — unknown face
PIN_BUZZER = 22    # Passive buzzer


class GPIOIndicator:
    """Hardware indicator controller with graceful fallback."""

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._gpio    = None

        if enabled:
            try:
                import RPi.GPIO as GPIO
                self._gpio = GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                GPIO.setup(PIN_GREEN,  GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(PIN_RED,    GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(PIN_BUZZER, GPIO.OUT, initial=GPIO.LOW)
                logger.info("GPIO indicators initialised (BCM pins %d/%d/%d)",
                            PIN_GREEN, PIN_RED, PIN_BUZZER)
            except (ImportError, RuntimeError) as exc:
                logger.warning("GPIO not available (%s) — indicator disabled", exc)
                self._gpio    = None
                self._enabled = False

    # ── High-level events ──────────────────────────────────────────────────

    def on_attendance_marked(self, name: str):
        """Flash green LED + short beep when attendance is successfully marked."""
        logger.debug("Indicator: attendance marked (%s)", name)
        self._blink_async(PIN_GREEN, on_ms=500, beep=True, beeps=1)

    def on_duplicate(self, name: str):
        """Short amber-style blink (green, quick) for duplicate."""
        logger.debug("Indicator: duplicate (%s)", name)
        self._blink_async(PIN_GREEN, on_ms=150, off_ms=100, count=2)

    def on_unknown_face(self):
        """Red LED pulse for unknown face."""
        logger.debug("Indicator: unknown face")
        self._blink_async(PIN_RED, on_ms=300, count=1)

    def on_error(self):
        """Rapid red blink for system error."""
        self._blink_async(PIN_RED, on_ms=100, off_ms=100, count=3)

    def cleanup(self):
        if self._gpio:
            self._gpio.cleanup()
            logger.info("GPIO cleaned up")

    # ── Internal helpers ──────────────────────────────────────────────────

    def _blink_async(self, pin: int, on_ms: int = 300, off_ms: int = 200,
                     count: int = 1, beep: bool = False, beeps: int = 0):
        """Run blink/beep in a daemon thread to not block recognition."""
        if not self._enabled or self._gpio is None:
            return
        t = threading.Thread(
            target=self._blink,
            args=(pin, on_ms, off_ms, count, beep, beeps),
            daemon=True,
        )
        t.start()

    def _blink(self, pin, on_ms, off_ms, count, beep, beeps):
        GPIO = self._gpio
        try:
            for _ in range(count):
                GPIO.output(pin, GPIO.HIGH)
                time.sleep(on_ms / 1000)
                GPIO.output(pin, GPIO.LOW)
                time.sleep(off_ms / 1000)
            if beep and beeps > 0:
                for _ in range(beeps):
                    GPIO.output(PIN_BUZZER, GPIO.HIGH)
                    time.sleep(0.08)
                    GPIO.output(PIN_BUZZER, GPIO.LOW)
                    time.sleep(0.05)
        except Exception as exc:
            logger.error("GPIO blink error: %s", exc)
