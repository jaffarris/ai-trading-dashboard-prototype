@echo off
cd /d "%~dp0"

echo Starting AI Trading Dashboard...
py -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo Python is not installed. Install Python 3.11 or newer, then run:
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

py -m streamlit run dashboard.py --server.headless false

pause
