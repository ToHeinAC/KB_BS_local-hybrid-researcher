"""Password-gated launcher for the Hybrid Researcher app via Cloudflare Tunnel."""

import os
import subprocess
import time
from datetime import datetime

import psutil
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Hybrid Researcher Launcher",
    page_icon="🧠",
    layout="centered",
)

# --- Configuration -----------------------------------------------------------

ADMIN_PASSWORD = os.getenv("LAUNCHER_PASSWORD", "changeme123")
APP_PORT = 8511
LAUNCHER_PORT = 8522
# Must match server.baseUrlPath in .streamlit/config.toml and the nginx path.
APP_BASE_PATH = "brain"
PROJECT_DIR = "/home/he/ai/dev/langgraph/KB_BS_local-hybrid-researcher"
APP_COMMAND = "uv run streamlit run src/ui/app.py --server.port 8511 --server.headless true"
APP_LOG = "/tmp/hybrid_researcher_app.log"

# Tunnel URL file paths (written by start-quick-tunnels.sh)
LAUNCHER_URL_FILE = "/tmp/hybrid-launcher-url.txt"
APP_URL_FILE = "/tmp/hybrid-app-url.txt"


# --- Tunnel URL helpers -------------------------------------------------------

def get_tunnel_urls() -> tuple[str, str]:
    """Read current tunnel URLs from env vars or saved files.

    The app's URLs carry APP_BASE_PATH: the app sets baseUrlPath in
    .streamlit/config.toml, so it answers at /brain/ and 404s at its root. The
    launcher itself is served at the root (start-launcher.sh overrides the
    inherited config), so its own URL is untouched.
    """
    launcher_url = os.getenv("LAUNCHER_URL", f"http://localhost:{LAUNCHER_PORT}")
    app_url = os.getenv("MAIN_APP_URL", f"http://localhost:{APP_PORT}/{APP_BASE_PATH}/")

    for path, setter in [(LAUNCHER_URL_FILE, "launcher"), (APP_URL_FILE, "app")]:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    url = f.read().strip()
                if url.startswith("https://"):
                    if setter == "launcher":
                        launcher_url = url
                    else:
                        # The tunnel points at the app's port root; the app lives
                        # one level in.
                        app_url = f"{url.rstrip('/')}/{APP_BASE_PATH}/"
            except OSError:
                pass

    return launcher_url, app_url


LAUNCHER_URL, MAIN_APP_URL = get_tunnel_urls()
IS_QUICK_TUNNEL = "trycloudflare.com" in LAUNCHER_URL


# --- Authentication -----------------------------------------------------------

def check_password() -> bool:
    """Return True if the user entered the correct password."""

    def _on_submit():
        if st.session_state["password"] == ADMIN_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=_on_submit, key="password")
        return False

    if not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=_on_submit, key="password")
        st.error("Password incorrect")
        return False

    return True


# --- Process helpers ----------------------------------------------------------

def get_process_on_port(port: int):
    """Return the psutil.Process listening on *port*, or None."""
    try:
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception as exc:
        st.error(f"Error checking port: {exc}")
    return None


def get_app_status() -> dict:
    """Return a status dict for the main research app."""
    proc = get_process_on_port(APP_PORT)
    if proc is None:
        return {"running": False}
    try:
        return {
            "running": True,
            "pid": proc.pid,
            "cpu": proc.cpu_percent(interval=0.1),
            "memory_mb": proc.memory_info().rss / 1024 / 1024,
            "cmdline": " ".join(proc.cmdline()) if proc.cmdline() else "N/A",
        }
    except Exception:
        return {"running": True, "pid": proc.pid}


def kill_process_on_port(port: int) -> bool:
    """Kill the process on *port*. Refuses ports < 1024 or the launcher port."""
    if port < 1024:
        st.error(f"Refusing to kill port {port} (system port)")
        return False
    if port == LAUNCHER_PORT:
        st.error(f"Refusing to kill launcher port {port}")
        return False
    try:
        result = subprocess.run(
            f"lsof -ti:{port}",
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            for pid in result.stdout.strip().split("\n"):
                try:
                    subprocess.run(f"kill -9 {pid}", shell=True, check=True)
                    st.success(f"Killed process {pid} on port {port}")
                except Exception as exc:
                    st.error(f"Error killing {pid}: {exc}")
            return True
        st.warning(f"No process found on port {port}")
        return False
    except Exception as exc:
        st.error(f"Error: {exc}")
        return False


def start_app() -> bool:
    """Start the main research app in the background."""
    try:
        proc = get_process_on_port(APP_PORT)
        if proc:
            st.warning(f"Process already running on port {APP_PORT} (PID: {proc.pid})")
            return False

        st.info(f"Starting app: {APP_COMMAND}")
        subprocess.Popen(
            f"cd {PROJECT_DIR} && nohup {APP_COMMAND} > {APP_LOG} 2>&1 &",
            shell=True,
            start_new_session=True,
        )
        time.sleep(3)

        proc = get_process_on_port(APP_PORT)
        if proc:
            st.success(f"App started (PID: {proc.pid})")
            return True
        st.error("Failed to start app - no process detected")
        return False
    except Exception as exc:
        st.error(f"Error starting app: {exc}")
        return False


# --- UI -----------------------------------------------------------------------

st.title("Hybrid Researcher Launcher")
st.markdown("### Remote Control Panel")

if IS_QUICK_TUNNEL:
    st.warning("Using Quick Tunnels (URLs are temporary)")

st.caption("Start, stop, and monitor the research application")
st.markdown("---")

# Auth gate
if not check_password():
    st.stop()

st.success("Authenticated")

status = get_app_status()

# Tunnel URLs
with st.expander("Current Tunnel URLs", expanded=IS_QUICK_TUNNEL):
    st.markdown("**Access from anywhere:**")

    c1, c2 = st.columns([3, 1])
    with c1:
        st.code(f"Launcher: {LAUNCHER_URL}", language="text")
    with c2:
        st.link_button("Open", LAUNCHER_URL, use_container_width=True)

    c3, c4 = st.columns([3, 1])
    with c3:
        st.code(f"Main App: {MAIN_APP_URL}", language="text")
    with c4:
        if status.get("running"):
            st.link_button("Open", MAIN_APP_URL, use_container_width=True, type="primary")
        else:
            st.button("Start First", disabled=True, use_container_width=True)

    if IS_QUICK_TUNNEL:
        st.info(
            "Quick Tunnel URLs change on each restart. "
            "Bookmark the current URLs for this session."
        )

st.markdown("---")

# Status
st.subheader("Application Status")

if status["running"]:
    st.success(f"App is RUNNING (PID: {status['pid']})")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("CPU Usage", f"{status.get('cpu', 'N/A')}%")
    with c2:
        st.metric("Memory Usage", f"{status.get('memory_mb', 0):.1f} MB")

    if "cmdline" in status:
        with st.expander("Process Details"):
            st.code(status["cmdline"], language="bash")

    col_open, _ = st.columns([2, 1])
    with col_open:
        st.link_button(
            "Open Hybrid Researcher App",
            MAIN_APP_URL,
            use_container_width=True,
            type="primary",
        )
    st.caption(f"Direct URL: {MAIN_APP_URL}")
else:
    st.warning("App is NOT running")

st.markdown("---")

# Controls
st.subheader("Controls")
c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Start App", type="primary", disabled=status["running"], use_container_width=True):
        with st.spinner("Starting..."):
            if start_app():
                time.sleep(2)
                st.rerun()

with c2:
    if st.button("Restart App", disabled=not status["running"], use_container_width=True):
        with st.spinner("Restarting..."):
            kill_process_on_port(APP_PORT)
            time.sleep(2)
            if start_app():
                time.sleep(2)
                st.rerun()

with c3:
    if st.button("Stop App", disabled=not status["running"], use_container_width=True):
        with st.spinner("Stopping..."):
            if kill_process_on_port(APP_PORT):
                time.sleep(2)
                st.rerun()

st.markdown("---")

# Logs
st.subheader("Application Logs")
if st.button("Refresh Logs"):
    st.rerun()

try:
    if os.path.exists(APP_LOG):
        with open(APP_LOG) as f:
            lines = f.read().split("\n")
        st.text_area("Recent logs (last 50 lines)", "\n".join(lines[-50:]), height=300)
    else:
        st.info("No log file found yet")
except Exception as exc:
    st.error(f"Error reading logs: {exc}")

# Footer
st.markdown("---")

with st.expander("How This Works"):
    st.markdown(
        f"""
**Workflow:**
1. Access this launcher via Cloudflare Tunnel
2. After login, start the main Hybrid Researcher app
3. Click "Open Hybrid Researcher App" to access it
4. Use "Stop App" to shut it down when finished

**URLs:**
- **Launcher**: `{LAUNCHER_URL}` (this page, port {LAUNCHER_PORT})
- **Main App**: `{MAIN_APP_URL}` (the researcher, port {APP_PORT})

Both URLs are routed through separate Cloudflare quick tunnels.
"""
    )

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
