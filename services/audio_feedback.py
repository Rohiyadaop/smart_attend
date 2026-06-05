from __future__ import annotations

import base64
import logging
import queue
import shutil
import subprocess
import sys
import threading
import time

logger = logging.getLogger("smartattend.audio")


class AttendanceAudioFeedback:
    """Plays local audio feedback without blocking the recognition loop."""

    def __init__(
        self,
        enabled: bool = True,
        beep_enabled: bool = True,
        prompt_cooldown_sec: float = 3.0,
    ):
        self._enabled = enabled
        self._beep_enabled = beep_enabled
        self._prompt_cooldown_sec = max(prompt_cooldown_sec, 0.5)
        self._queue: queue.Queue[tuple[str, bool]] = queue.Queue()
        self._prompt_cache: dict[str, tuple[str, float]] = {}

        if not enabled:
            return

        worker = threading.Thread(
            target=self._run,
            daemon=True,
            name="AttendanceAudioFeedback",
        )
        worker.start()

    def on_attendance_marked(self, name: str):
        if not self._enabled:
            return
        spoken_name = " ".join((name or "Student").split()) or "Student"
        self.clear_prompt_cache()
        self._queue.put((f"{spoken_name} attendance marked", True))

    def on_guidance(self, track_key: str, task_text: str):
        if not self._enabled or not task_text:
            return
        spoken_text = self._normalize_guidance(task_text)
        cache_key = track_key or spoken_text
        previous = self._prompt_cache.get(cache_key)
        now = time.time()
        if previous and previous[0] == spoken_text and (now - previous[1]) < self._prompt_cooldown_sec:
            return
        self._prompt_cache[cache_key] = (spoken_text, now)
        self._queue.put((spoken_text, False))

    def clear_prompt_cache(self, track_key: str | None = None):
        if track_key:
            self._prompt_cache.pop(track_key, None)
            return
        self._prompt_cache.clear()

    def _run(self):
        while True:
            phrase, play_beep = self._queue.get()
            try:
                if play_beep:
                    self._play_beep()
                self._speak(phrase)
            except Exception:
                logger.exception("Audio feedback failed for phrase: %s", phrase)
            finally:
                self._queue.task_done()

    def _play_beep(self):
        if not self._beep_enabled:
            return

        if sys.platform == "win32":
            try:
                import winsound

                winsound.Beep(1600, 180)
                return
            except Exception as exc:  # pragma: no cover - platform dependent
                logger.debug("winsound beep unavailable: %s", exc)

    def _speak(self, phrase: str):
        if sys.platform == "win32":
            self._run_command(self._windows_speech_command(phrase))
            return

        if shutil.which("say"):
            self._run_command(["say", phrase])
            return

        for command in ("spd-say", "espeak"):
            if shutil.which(command):
                self._run_command([command, phrase])
                return

        logger.debug("No speech backend available for audio feedback")

    @staticmethod
    def _normalize_guidance(task_text: str) -> str:
        normalized = " ".join(task_text.split())
        lowered = normalized.lower()
        if lowered.startswith("challenge: "):
            normalized = normalized.split(":", 1)[1].strip()
            lowered = normalized.lower()
        replacements = {
            "turn your head left or right": "Please turn your head left or right",
            "move closer to camera": "Please move closer to the camera",
            "align face for liveness": "Please align your face for liveness",
            "liveness below threshold": "Please turn your head left or right",
            "waiting in queue": "Please wait for liveness check",
        }
        return replacements.get(lowered, normalized)

    def _windows_speech_command(self, phrase: str) -> list[str]:
        escaped_phrase = phrase.replace("'", "''")
        script = (
            "$voice = New-Object -ComObject SAPI.SpVoice; "
            f"[void]$voice.Speak('{escaped_phrase}')"
        )
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        return [
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            encoded,
        ]

    def _run_command(self, command: list[str]):
        kwargs = {
            "check": False,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "timeout": 15,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        completed = subprocess.run(command, **kwargs)
        if completed.returncode != 0:
            logger.warning("Audio feedback command failed with exit code %s", completed.returncode)
