"""Live GPU stats widget via Tornado route injection.

Injects a ``/_api/gpu`` handler into Streamlit's own Tornado server.
The sidebar renders an ``st.components.v1.html()`` snippet whose JS fetches
the relative URL ``/_api/gpu`` every 1 s.  Because the iframe inherits the
parent page's origin (``allow-same-origin``), this works over SSH tunnels and
any remote access method — no cross-origin or sandbox issues.

Tornado's I/O loop runs independently of Streamlit's script-runner thread,
so updates keep flowing even while ``graph.stream()`` blocks for 30 s+.

The live Tornado ``Application`` is discovered via ``gc.get_objects()`` since
Streamlit ≥1.53 removed ``Server.get_current()``.
"""

import gc
import json
import logging
import subprocess

import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)


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

        # Guard against double-registration.
        # add_handlers() writes to default_router, so check there.
        for rule in tornado_app.default_router.rules:
            target = getattr(rule, "target", None)
            if target is None:
                continue
            for sub_rule in getattr(target, "rules", []):
                matcher = getattr(sub_rule, "matcher", None)
                if matcher and hasattr(matcher, "regex"):
                    if "/_api/gpu" in matcher.regex.pattern:
                        return True  # already registered

        class GPUStatsHandler(tornado.web.RequestHandler):
            """Serves live GPU stats as JSON."""

            def set_default_headers(self):
                self.set_header("Content-Type", "application/json")
                self.set_header("Cache-Control", "no-store")

            def get(self):
                self.write(json.dumps(_get_gpu_stats()))

        tornado_app.add_handlers(".*", [(r"/_api/gpu", GPUStatsHandler)])
        logger.info("Injected /_api/gpu Tornado route for GPU widget")
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
# HTML/JS template (fetches relative /_api/gpu)
# ---------------------------------------------------------------------------

_GPU_HTML = """\
<div id="gpu-stats" style="font-family:monospace; font-size:13px; color:#ddd; white-space:nowrap;">
  Lade GPU...
</div>
<script>
function fetchGPU() {
  fetch("/_api/gpu")
    .then(r => r.json())
    .then(gpus => {
      if (!gpus.length) return;
      let html = gpus.map(g => {
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
      }).join("<br>");
      document.getElementById("gpu-stats").innerHTML = html;
    })
    .catch(() => {});
}
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
    st.sidebar.markdown("**GPU**")
    with st.sidebar:
        components.html(_GPU_HTML, height=50, scrolling=False)
