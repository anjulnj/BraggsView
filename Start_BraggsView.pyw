"""
Start_BraggsView.pyw - One-click BraggsView launcher for Windows.

Double-click this file to open the XRD viewer in your browser.
It runs WITHOUT a terminal window: a small status window appears, and
closing it stops the app. All setup (Python env, packages) is automatic.

  - On Windows: double-click (uses pythonw, no console). Works as long as
    Python 3.9+ is installed from python.org.
  - On macOS/Linux: run with  python Start_BraggsView.pyw

The app environment is created OUTSIDE the OneDrive folder so it cannot
break when the folder is synced to another machine.

If something goes wrong, details are written to braggsview_error.log next
to this file.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Callable, Optional

APP_DIR = Path(__file__).resolve().parent
ERROR_LOG = APP_DIR / "braggsview_error.log"
SERVER_LOG = APP_DIR / "braggsview_server.log"

HAS_GUI = False
try:
    import tkinter as tk  # noqa: PLC0415
    HAS_GUI = True
except Exception:
    pass

# Running under pythonw (Windows) has no console -> use GUI.
USE_GUI = HAS_GUI and sys.stdout is None


def log_error(msg: str) -> None:
    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")
    except Exception:
        pass


def find_python() -> str:
    """Python interpreter that launched us (python.exe on Windows, not pythonw)."""
    exe = Path(sys.executable)
    if os.name == "nt" and exe.name.lower().startswith("pythonw"):
        alt = exe.with_name("python.exe")
        if alt.exists():
            return str(alt)
    return str(exe)


def get_venv_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "BraggsView" / "venv"
    return Path.home() / ".braggsview" / "venv"


def get_venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def run(cmd, status_cb=None) -> int:
    """Run a command, capture output, return exit code. 0 = success."""
    if status_cb:
        status_cb(" ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except Exception as e:
        log_error(f"Could not run {' '.join(cmd)}: {e}")
        return 1
    if proc.returncode != 0:
        log_error(f"{' '.join(cmd)} exited {proc.returncode}: {(proc.stderr or proc.stdout or '').strip()}")
    return proc.returncode


def check_python_version(python: str, status_cb) -> bool:
    try:
        vp = subprocess.run(
            [python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True,
        )
        if vp.returncode != 0:
            log_error(f"Python check failed: {vp.stderr}")
            status_cb("Python 3.9+ needed - install from python.org")
            return False
        major, minor = (int(p) for p in vp.stdout.strip().split("."))
        if (major, minor) < (3, 9):
            status_cb("Python too old (needs 3.9+). Install from python.org")
            return False
        return True
    except Exception as e:
        log_error(f"Could not run python: {e}")
        status_cb("Could not run Python. Install it from python.org")
        return False


def launch(status_cb: Callable[[str], None]) -> int:
    """Full launcher flow. Returns 0 on success, non-zero on failure."""
    python = find_python()

    if not check_python_version(python, status_cb):
        return 1

    venv = get_venv_dir()
    venv_python = get_venv_python(venv)

    if not venv_python.exists():
        status_cb("First run: setting up environment (a few minutes)...")
        try:
            venv.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if run([python, "-m", "venv", str(venv)], status_cb) != 0:
            status_cb("Could not create the app environment")
            return 1

    status_cb("Checking packages (fast once done)...")
    run([str(venv_python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], status_cb)
    if run([str(venv_python), "-m", "pip", "install", "--quiet", "-r", str(APP_DIR / "requirements.txt")], status_cb) != 0:
        status_cb("Could not install packages (internet needed)")
        return 1

    status_cb("Starting app... your browser will open")
    logf = open(SERVER_LOG, "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [str(venv_python), "-m", "streamlit", "run", str(APP_DIR / "app.py"),
             "--server.headless", "false"],
            stdout=logf, stderr=logf, cwd=str(APP_DIR),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as e:
        logf.close()
        log_error(f"Could not start streamlit: {e}")
        status_cb("Could not start the app")
        return 1

    def open_browser() -> None:
        for _ in range(120):
            if proc.poll() is not None:
                return
            try:
                import urllib.request  # noqa: PLC0415
                urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=1)
                webbrowser.open("http://localhost:8501")
                return
            except Exception:
                time.sleep(0.5)

    threading.Thread(target=open_browser, daemon=True).start()
    return 0


def run_gui() -> None:
    """GUI mode (pythonw): status window with a Stop button."""
    root = tk.Tk()
    root.title("BraggsView")
    root.resizable(False, False)
    frame = tk.Frame(root, padx=28, pady=24)
    frame.pack()
    tk.Label(frame, text="BraggsView", font=("Segoe UI", 16, "bold")).pack()
    status_var = tk.StringVar(value="Starting...")
    tk.Label(frame, textvariable=status_var, justify="left", font=("Segoe UI", 11)).pack(pady=(12, 16))

    proc_holder = {"proc": None}
    result_holder = {"code": None}

    def on_close() -> None:
        p = proc_holder["proc"]
        if p is not None and p.poll() is None:
            p.terminate()
        root.destroy()

    tk.Button(frame, text="Stop BraggsView", command=on_close,
              padx=18, pady=4, font=("Segoe UI", 11)).pack()
    root.protocol("WM_DELETE_WINDOW", on_close)

    def status(msg: str) -> None:
        root.after(0, lambda: status_var.set(msg))

    def work() -> None:
        code = launch(status)
        result_holder["code"] = code
        if code != 0:
            status("Error - see braggsview_error.log")
        else:
            status("BraggsView is running in your browser.\nClose this window to stop it.")

    threading.Thread(target=work, daemon=True).start()
    root.mainloop()


def run_console() -> int:
    print("======================================")
    print("  BraggsView Launcher")
    print("======================================")

    def status(msg: str) -> None:
        print("  " + msg)

    code = launch(status)
    if code != 0:
        print()
        print(f"Something went wrong. See {ERROR_LOG}")
        input("Press Enter to close...")
    return code


if __name__ == "__main__":
    if USE_GUI:
        run_gui()
    else:
        sys.exit(run_console())
