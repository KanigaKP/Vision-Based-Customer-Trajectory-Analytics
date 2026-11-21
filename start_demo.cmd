@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo [ERROR] Project virtual environment ^(venv^) is missing in this directory.
  echo start_demo.cmd requires a release-local virtual environment at .\venv\Scripts\python.exe.
  echo.
  echo To set up the environment, run:
  echo   python -m venv venv
  echo   .\venv\Scripts\python.exe -m pip install -r requirements.txt
  echo   .\venv\Scripts\python.exe scripts\check_prerequisites.py
  echo   .\start_demo.cmd
  echo.
  pause
  exit /b 1
)

"venv\Scripts\python.exe" scripts\check_prerequisites.py --launcher
set PRE_CODE=%ERRORLEVEL%

if %PRE_CODE% EQU 2 (
  echo.
  echo [INFO] Project Streamlit server is ALREADY RUNNING and healthy on port 8501.
  echo [ACTION] Reusing existing server. No duplicate process will be launched.
  echo Opening browser at http://127.0.0.1:8501/?view=all ...
  start "" "http://127.0.0.1:8501/?view=workflow"
  pause
  exit /b 0
)

if %PRE_CODE% EQU 3 (
  echo.
  echo [ERROR] Port 8501 is occupied by an unrecognized process or unresponsive service.
  echo [ACTION] Launch aborted to prevent conflict.
  echo Please terminate the conflicting process or free port 8501 before restarting.
  pause
  exit /b 1
)

if %PRE_CODE% NEQ 0 (
  echo.
  echo [ERROR] Preflight checks failed ^(Code: %PRE_CODE%^).
  echo Please resolve the missing dependencies or configuration errors above.
  pause
  exit /b 1
)

echo.
echo [STARTING] Launching fresh Streamlit dashboard on port 8501...
echo Access URL: http://127.0.0.1:8501/?view=workflow
start "" "http://127.0.0.1:8501/?view=workflow"
"venv\Scripts\python.exe" -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
pause
