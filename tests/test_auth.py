"""Tests for the GUI login gate (src/ui/auth.py)."""

import json

import pytest

from src.ui import auth


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the user store at a temp file for each test."""
    path = tmp_path / "users.json"
    monkeypatch.setattr(auth, "USERS_JSON_PATH", path)
    return path


def test_ensure_seeded_creates_both_users(store):
    assert not store.exists()
    auth.ensure_seeded()
    assert store.exists()
    data = json.loads(store.read_text(encoding="utf-8"))
    assert set(data["users"]) == {"T. Hein", "Gast"}


def test_ensure_seeded_is_idempotent(store):
    auth.ensure_seeded()
    first = store.read_text(encoding="utf-8")
    auth.ensure_seeded()  # second call must not overwrite
    assert store.read_text(encoding="utf-8") == first


def test_verify_correct_credentials(store):
    auth.ensure_seeded()
    assert auth.verify("T. Hein", "#BrAIn1")
    assert auth.verify("Gast", "2026_BrAIn")


def test_verify_wrong_password(store):
    auth.ensure_seeded()
    assert not auth.verify("T. Hein", "wrong")
    assert not auth.verify("Gast", "#BrAIn1")


def test_verify_unknown_user(store):
    auth.ensure_seeded()
    assert not auth.verify("Nobody", "#BrAIn1")


def test_passwords_stored_as_hash_not_plaintext(store):
    auth.ensure_seeded()
    raw = store.read_text(encoding="utf-8")
    assert "#BrAIn1" not in raw
    assert "2026_BrAIn" not in raw


def test_users_have_distinct_salts(store):
    auth.ensure_seeded()
    data = json.loads(store.read_text(encoding="utf-8"))
    salts = {u["salt"] for u in data["users"].values()}
    assert len(salts) == len(data["users"])
