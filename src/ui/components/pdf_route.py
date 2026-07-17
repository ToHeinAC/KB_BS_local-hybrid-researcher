"""PDF serving via Tornado route injection.

Injects an ``_api/pdf`` handler into Streamlit's Tornado server so that
PDF links in the citation list open in the browser instead of being blocked
by the ``file://`` security restriction.

The route is registered under Streamlit's base path (see ``base_path``); the
citation links in ``src/agents/tools.py`` are relative and resolve to it.

Uses the same gc-based Tornado discovery pattern as ``gpu_widget.py``.
"""

import gc
import logging
from pathlib import Path
from urllib.parse import unquote

import streamlit as st

from src.ui.components.base_path import base_path

logger = logging.getLogger(__name__)

# Resolved kb/ base directory (absolute) — used for path-traversal validation
_KB_BASE: Path | None = None


def _get_kb_base() -> Path:
    """Return the resolved absolute path to the kb/ directory."""
    global _KB_BASE
    if _KB_BASE is None:
        from src.config import settings
        _KB_BASE = Path(settings.chromadb_path).parent.resolve()
    return _KB_BASE


def _inject_pdf_route() -> bool:
    """Inject the PDF route into Streamlit's Tornado app. Returns success."""
    try:
        import tornado.web

        apps = [
            obj for obj in gc.get_objects()
            if type(obj) is tornado.web.Application
        ]
        if not apps:
            logger.debug("No tornado.web.Application found via gc")
            return False
        tornado_app = apps[0]

        route_path = f"{base_path()}/_api/pdf"

        # Double-injection guard
        for rule in tornado_app.default_router.rules:
            target = getattr(rule, "target", None)
            if target is None:
                continue
            for sub_rule in getattr(target, "rules", []):
                matcher = getattr(sub_rule, "matcher", None)
                if matcher and hasattr(matcher, "regex"):
                    if route_path in matcher.regex.pattern:
                        return True  # already registered

        kb_base = _get_kb_base()

        class PDFHandler(tornado.web.RequestHandler):
            """Serves PDF files from the kb/ directory."""

            def get(self):
                raw_path = self.get_argument("path", default="")
                if not raw_path:
                    self.set_status(400)
                    self.write("Missing 'path' parameter")
                    return

                decoded = unquote(raw_path)
                requested = Path(decoded).resolve()

                # Security: must be under kb/ directory
                if not str(requested).startswith(str(kb_base)):
                    self.set_status(403)
                    self.write("Access denied: path outside kb/ directory")
                    return

                if not requested.exists() or not requested.is_file():
                    self.set_status(404)
                    self.write("File not found")
                    return

                self.set_header("Content-Type", "application/pdf")
                self.set_header("Content-Disposition", "inline")
                self.set_header("Cache-Control", "public, max-age=3600")
                with open(requested, "rb") as f:
                    self.write(f.read())

        tornado_app.add_handlers(".*", [(route_path, PDFHandler)])
        logger.info("Injected %s Tornado route for PDF serving", route_path)
        return True

    except Exception:
        logger.debug("Could not inject PDF Tornado route", exc_info=True)
        return False


@st.cache_resource
def ensure_pdf_route() -> bool:
    """One-time injection (cached across reruns). Returns True if route is live."""
    return _inject_pdf_route()
