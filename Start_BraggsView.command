#!/bin/bash
# =============================================================================
#  BraggsView - one-click launcher (macOS)
#  Double-click this file to open the XRD viewer in your browser.
#
#  What it does automatically (no programming needed):
#    1. Finds Python 3 on your Mac
#    2. Creates a private app environment (stored in ~/.braggsview/venv,
#       OUTSIDE the OneDrive folder so it can't break across machines)
#    3. Installs the required packages (one-time, needs internet)
#    4. Starts the app and opens your browser
#
#  Close the black window to stop the app.
# =============================================================================

cd "$(dirname "$0")"

echo "======================================"
echo "  BraggsView Launcher (macOS)"
echo "======================================"
echo ""

# --- 1. Find Python 3 -------------------------------------------------------
PYTHON=""
for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        PYTHON="$cand"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Python 3 is not installed on this Mac."
    echo ""
    echo "Install it (free) from:  https://www.python.org/downloads/"
    echo "During install, tick \"Add Python to PATH\"."
    echo "Then double-click this file again."
    echo ""
    read -n 1 -s -r -p "Press any key to close..."
    exit 1
fi

echo "Using Python: $("$PYTHON" --version 2>&1)"

# Check version is new enough
"$PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" || {
    echo ""
    echo "Your Python is too old (needs 3.9 or newer)."
    echo "Install a recent one from:  https://www.python.org/downloads/"
    read -n 1 -s -r -p "Press any key to close..."
    exit 1
}

# --- 2. Create the app environment (machine-local, outside OneDrive) ---------
VENV_DIR="$HOME/.braggsview/venv"
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo ""
    echo "First run: setting up the app environment..."
    echo "(one-time, a few minutes, needs internet)"
    mkdir -p "$VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo ""
        echo "Could not create the app environment."
        read -n 1 -s -r -p "Press any key to close..."
        exit 1
    fi
fi
VPY="$VENV_DIR/bin/python"

# --- 3. Install / refresh dependencies ---------------------------------------
echo "Checking packages (this is fast once done)..."
"$VPY" -m pip install --quiet --upgrade pip
"$VPY" -m pip install --quiet -r requirements.txt
if [ $? -ne 0 ]; then
    echo ""
    echo "Could not install the packages. Check your internet connection and try again."
    read -n 1 -s -r -p "Press any key to close..."
    exit 1
fi

# --- 4. Launch ---------------------------------------------------------------
echo ""
echo "Starting BraggsView... your browser will open automatically."
echo "(Keep this window open. Close it to stop the app.)"
echo ""
"$VPY" -m streamlit run app.py --server.headless false

echo ""
echo "BraggsView has stopped."
read -n 1 -s -r -p "Press any key to close..."
