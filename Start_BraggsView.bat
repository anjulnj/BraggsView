@echo off
REM =============================================================================
REM  BraggsView - one-click launcher (Windows)
REM  Double-click this file to open the XRD viewer in your browser.
REM
REM  What it does automatically (no programming needed):
REM    1. Finds Python 3 on this PC
REM    2. Creates a private app environment (stored in %LOCALAPPDATA%\BraggsView,
REM       OUTSIDE the OneDrive folder so it can't break across machines)
REM    3. Installs the required packages (one-time, needs internet)
REM    4. Starts the app and opens your browser
REM
REM  Close the black window to stop the app.
REM =============================================================================

title BraggsView Launcher
cd /d "%~dp0"

echo ======================================
echo   BraggsView Launcher (Windows)
echo ======================================
echo.

REM --- 1. Find Python 3 -------------------------------------------------------
set "PYTHON="
where python >nul 2>&1 && set "PYTHON=python"
if not defined PYTHON where py >nul 2>&1 && set "PYTHON=py"
if not defined PYTHON (
    echo Python 3 is not installed on this PC.
    echo.
    echo Install it ^(free^) from:  https://www.python.org/downloads/
    echo During install, tick "Add python.exe to PATH".
    echo Then double-click this file again.
    echo.
    pause
    exit /b 1
)
echo Using Python: %PYTHON%

REM Check version is new enough
%PYTHON% -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"
if errorlevel 1 (
    echo.
    echo Your Python is too old ^(needs 3.9 or newer^).
    echo Install a recent one from:  https://www.python.org/downloads/
    pause
    exit /b 1
)

REM --- 2. Create the app environment (machine-local, outside OneDrive) ---------
set "VENV_DIR=%LOCALAPPDATA%\BraggsView\venv"
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo.
    echo First run: setting up the app environment...
    echo ^(one-time, a few minutes, needs internet^)
    if not exist "%LOCALAPPDATA%\BraggsView" mkdir "%LOCALAPPDATA%\BraggsView"
    %PYTHON% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo Could not create the app environment.
        pause
        exit /b 1
    )
)
set "VPY=%VENV_DIR%\Scripts\python.exe"

REM --- 3. Install / refresh dependencies --------------------------------------
echo Checking packages ^(this is fast once done^)...
"%VPY%" -m pip install --quiet --upgrade pip
"%VPY%" -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo.
    echo Could not install the packages. Check your internet connection and try again.
    pause
    exit /b 1
)

REM --- 4. Launch --------------------------------------------------------------
echo.
echo Starting BraggsView... your browser will open automatically.
echo ^(Keep this window open. Close it to stop the app.^)
echo.
"%VPY%" -m streamlit run app.py --server.headless false

echo.
echo BraggsView has stopped.
pause
