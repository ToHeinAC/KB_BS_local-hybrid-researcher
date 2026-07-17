"""Streamlit's configured base path, as a URL prefix.

This app serves two routes of its own — ``_api/gpu`` and ``_api/pdf`` — by
injecting them into Streamlit's Tornado app. Streamlit prefixes its *own* routes
with ``server.baseUrlPath``, but an injected handler would stay at the server
root, out of reach of the reverse proxy's ``location /brain/``.

Both injectors read the prefix from here so they cannot drift from
``.streamlit/config.toml`` or from each other. The browser side stays *relative*
(``./_api/gpu``, ``_api/pdf?path=``) and resolves against the page, which the
proxy always serves at ``/brain/``.
"""

import streamlit as st


def base_path() -> str:
    """Return ``""`` or ``"/brain"`` — never a trailing slash."""
    base = (st.get_option("server.baseUrlPath") or "").strip("/")
    return f"/{base}" if base else ""
