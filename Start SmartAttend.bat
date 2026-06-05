@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo SmartAttend could not find venv\Scripts\python.exe
  echo Please create the virtual environment first.
  pause
  exit /b 1
)

start "SmartAttend" /D "%~dp0" "venv\Scripts\pythonw.exe" "app.py"
exit /b 0
