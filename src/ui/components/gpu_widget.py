"""Live GPU stats widget via Tornado route injection.

Injects an ``_api/gpu`` handler into Streamlit's own Tornado server, under
Streamlit's configured base path (see ``base_path``). The sidebar renders an
``st.components.v1.html()`` snippet whose JS fetches ``./_api/gpu`` every 1 s.
Because the iframe inherits the parent page's origin (``allow-same-origin``) and
its base URL, this works over SSH tunnels, behind the reverse proxy's
``/brain/`` path, and by any other remote access method — no cross-origin or
sandbox issues.

Tornado's I/O loop runs independently of Streamlit's script-runner thread,
so updates keep flowing even while ``graph.stream()`` blocks for 30 s+.

The live Tornado ``Application`` is discovered via ``gc.get_objects()`` since
Streamlit ≥1.53 removed ``Server.get_current()``.
"""

import gc
import json
import logging
import subprocess
import time

import streamlit as st
import streamlit.components.v1 as components

from src.ui.components.base_path import base_path

logger = logging.getLogger(__name__)

# Module-level timing state (safe for single-user local app).
# Use a mutable dict so all readers — including the Tornado handler defined
# inside _inject_gpu_route() — always see the current values without relying
# on global-variable rebinding, which can be stale across closure boundaries.
_timer: dict = {"start": None, "end": None}


def set_research_start() -> None:
    _timer["start"] = time.monotonic()
    _timer["end"] = None


def set_research_end() -> None:
    if _timer["start"] is not None:
        _timer["end"] = time.monotonic()


def reset_research_timer() -> None:
    _timer["start"] = None
    _timer["end"] = None


# ---------------------------------------------------------------------------
# GPU stats (unchanged)
# ---------------------------------------------------------------------------

def _get_gpu_stats() -> list[dict]:
    """Query nvidia-smi. Returns [] on failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,fan.speed,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                gpus.append({
                    "name": parts[0],
                    "fan": parts[1],
                    "temp": parts[2],
                    "util": parts[3],
                })
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


# ---------------------------------------------------------------------------
# Tornado handler + route injection (gc-based discovery)
# ---------------------------------------------------------------------------

def _inject_gpu_route() -> bool:
    """Inject ``/_api/gpu`` into Streamlit's Tornado app. Returns success.

    Uses ``gc.get_objects()`` to find the live ``tornado.web.Application``
    instance, since Streamlit ≥1.53 removed ``Server.get_current()``.
    """
    try:
        import tornado.web

        # Find the live Tornado Application via gc
        apps = [
            obj for obj in gc.get_objects()
            if type(obj) is tornado.web.Application
        ]
        if not apps:
            logger.debug("No tornado.web.Application found via gc")
            return False
        tornado_app = apps[0]

        route_path = f"{base_path()}/_api/gpu"

        # Guard against double-registration.
        # add_handlers() writes to default_router, so check there.
        for rule in tornado_app.default_router.rules:
            target = getattr(rule, "target", None)
            if target is None:
                continue
            for sub_rule in getattr(target, "rules", []):
                matcher = getattr(sub_rule, "matcher", None)
                if matcher and hasattr(matcher, "regex"):
                    if route_path in matcher.regex.pattern:
                        return True  # already registered

        class GPUStatsHandler(tornado.web.RequestHandler):
            """Serves live GPU stats as JSON."""

            def set_default_headers(self):
                self.set_header("Content-Type", "application/json")
                self.set_header("Cache-Control", "no-store")

            def get(self):
                gpus = _get_gpu_stats()
                elapsed = None
                is_running = False
                start = _timer["start"]
                if start is not None:
                    end = _timer["end"] if _timer["end"] is not None else time.monotonic()
                    elapsed = int(end - start)
                    is_running = _timer["end"] is None
                self.write(json.dumps({"gpus": gpus, "elapsed": elapsed, "is_running": is_running}))

        tornado_app.add_handlers(".*", [(route_path, GPUStatsHandler)])
        logger.info("Injected %s Tornado route for GPU widget", route_path)
        return True

    except Exception:
        logger.debug("Could not inject GPU Tornado route", exc_info=True)
        return False


@st.cache_resource
def _ensure_gpu_route() -> bool:
    """One-time injection (cached across reruns). Returns True if route is live."""
    if not _get_gpu_stats():
        return False  # no GPU available
    return _inject_gpu_route()


# ---------------------------------------------------------------------------
# HTML/JS template (fetches _api/gpu relative to the page)
# ---------------------------------------------------------------------------

def _gpu_html(model: str) -> str:
    """Return the GPU widget HTML with the LLM model name embedded."""
    return f"""\
<div id="gpu-stats" style="font-family:monospace; font-size:13px; color:#ddd; white-space:nowrap;">
  Lade GPU...
</div>
<script>
function fetchGPU() {{
  // Relative, not "/_api/gpu": behind the reverse proxy the page is served at
  // /brain/, and a root-absolute URL would escape that prefix. This srcdoc
  // iframe resolves relative URLs against the parent page.
  fetch("./_api/gpu")
    .then(r => r.json())
    .then(data => {{
      const gpus = data.gpus || [];
      if (!gpus.length) return;
      let html = gpus.map(g => {{
        let name = g.name.replace("NVIDIA GeForce ", "").padEnd(10);
        let t = parseInt(g.temp);
        let u = parseInt(g.util);
        let tCol = t >= 80 ? "#ff4b4b" : t >= 70 ? "#ffa421" : "#21c354";
        let uCol = u >= 80 ? "#ff4b4b" : u >= 50 ? "#ffa421" : "#21c354";
        let fan = String(g.fan).padStart(2);
        let tmp = String(t).padStart(2);
        let load = String(u).padStart(3);
        return name
          + " <span style='color:" + tCol + "'>" + tmp + "&deg;C</span>"
          + "|Fan:" + fan + "%"
          + "|<span style='color:" + uCol + "'>Load:" + load + "%</span>";
      }}).join("<br>");
      html += "<br><span style='color:#aaa'>llm: {model}</span>";
      if (data.elapsed !== null && data.elapsed !== undefined) {{
        let eCol = data.is_running ? "#21c354" : "#aaa";
        let dots = data.is_running ? "..." : "";
        html += "<br><span style='color:" + eCol + "'>t: " + data.elapsed + "s" + dots + "</span>";
      }}
      document.getElementById("gpu-stats").innerHTML = html;
    }})
    .catch(() => {{}});
}}
fetchGPU();
setInterval(fetchGPU, 1000);
</script>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_gpu_sidebar() -> None:
    """Render live GPU widget in sidebar via Tornado route injection."""
    if not _ensure_gpu_route():
        return  # no GPU or injection failed
    from src.config import settings
    st.sidebar.markdown("**GPU**")
    with st.sidebar:
        components.html(_gpu_html(settings.ollama_model), height=85, scrolling=False)
