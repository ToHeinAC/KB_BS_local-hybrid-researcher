"""Simple GUI login gate for the Streamlit app.

A self-contained, role-free authentication layer inspired by the sibling
project ``KB_BS_local-wiki-he``. Credentials live in a gitignored JSON store
(``data/users.json``) seeded on first run with salted PBKDF2 password hashes.

Auth state is kept in a top-level ``st.session_state["auth_user"]`` key (NOT on
the ``SessionState`` dataclass) so that ``reset_session_state()`` — triggered by
the "Free GPU & Reset" button and "Neue Recherche starten" — does not log the
user out.
"""

import hashlib
import hmac
import json
import logging
import os
from pathlib import Path

import streamlit as st

from src.ui.state import reset_session_state

logger = logging.getLogger(__name__)

# Seed-only plaintext defaults (hashed immediately on seeding; never stored raw).
_DEFAULT_USERS = {
    "T. Hein": "#BrAIn1",
    "Gast": "2026_BrAIn",
}

# Repo-relative store path: src/ui/auth.py -> parents[2] == repo root.
USERS_JSON_PATH = Path(__file__).resolve().parents[2] / "data" / "users.json"

_PBKDF2_ITERATIONS = 200_000
_SESSION_KEY = "auth_user"


def _hash(password: str, salt: bytes) -> str:
    """Return the hex PBKDF2-HMAC-SHA256 hash of ``password`` with ``salt``."""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    ).hex()


def _load() -> dict:
    """Load the user store, returning ``{"users": {}}`` if unreadable."""
    try:
        with open(USERS_JSON_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"users": {}}


def ensure_seeded() -> None:
    """Create ``data/users.json`` with hashed default users if it is missing.

    Idempotent: a no-op when the file already exists, so password changes made
    later survive.
    """
    if USERS_JSON_PATH.exists():
        return
    users: dict[str, dict] = {}
    for name, password in _DEFAULT_USERS.items():
        salt = os.urandom(16)
        users[name] = {"salt": salt.hex(), "pw_hash": _hash(password, salt)}
    USERS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"users": users}, f, indent=2, ensure_ascii=False)
    logger.info("Seeded user store at %s with %d users", USERS_JSON_PATH, len(users))


def verify(username: str, password: str) -> bool:
    """Return True if ``password`` matches the stored hash for ``username``."""
    user = _load().get("users", {}).get(username)
    if not user:
        return False
    try:
        salt = bytes.fromhex(user["salt"])
        expected = user["pw_hash"]
    except (KeyError, ValueError):
        return False
    return hmac.compare_digest(_hash(password, salt), expected)


def is_authenticated() -> bool:
    """Return whether the current session is logged in."""
    return bool(st.session_state.get(_SESSION_KEY))


def current_user() -> str | None:
    """Return the logged-in username, or None."""
    return st.session_state.get(_SESSION_KEY)


def logout() -> None:
    """Clear auth + session state and rerun to the login screen."""
    st.session_state.pop(_SESSION_KEY, None)
    reset_session_state()
    st.rerun()


def render_login() -> None:
    """Render a centered login form. Call only when not authenticated."""
    _left, mid, _right = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("## :brain: BrAIn")
        st.caption("Bitte anmelden, um fortzufahren.")
        with st.form("login_form"):
            username = st.text_input("Benutzername")
            password = st.text_input("Passwort", type="password")
            submitted = st.form_submit_button(
                "Anmelden", type="primary", use_container_width=True
            )
        if submitted:
            if verify(username, password):
                st.session_state[_SESSION_KEY] = username
                st.rerun()
            else:
                st.error("Benutzername oder Passwort ungültig.")
