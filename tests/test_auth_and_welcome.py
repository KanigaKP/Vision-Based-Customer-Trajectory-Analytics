"""
Unit tests for Persistent Authentication and Welcome Page Access Control.
========================================================================
Verifies:
1. PBKDF2-HMAC-SHA256 password hashing with random salt and constant-time comparison.
2. Duplicate username detection.
3. Correct and incorrect password authentication.
4. Protected view access barrier enforcement for unauthenticated guests.
5. Free guest access to Retail Analysis Workflow / Sample Shops.
"""

import os
import pytest
from pathlib import Path
from dashboard.auth import (
    create_account,
    authenticate_user,
    hash_password,
    verify_password,
    is_view_protected,
    ACCOUNTS_FILE,
)


def test_password_hashing_and_verification():
    """Verify PBKDF2 salting and constant-time verification."""
    password = "DemonstrationPassword2026!"
    salt_hex, hash_hex = hash_password(password)

    assert len(salt_hex) == 32  # 16 bytes in hex
    assert len(hash_hex) == 64  # 32 bytes in hex (SHA-256)
    assert verify_password(password, salt_hex, hash_hex) is True
    assert verify_password("WrongPassword!", salt_hex, hash_hex) is False
    assert verify_password("", salt_hex, hash_hex) is False


def test_account_creation_and_duplicate_prevention(tmp_path, monkeypatch):
    """Verify persistent account creation and duplicate rejection."""
    fake_accounts_file = str(tmp_path / "test_accounts.json")
    monkeypatch.setattr("dashboard.auth.ACCOUNTS_FILE", fake_accounts_file)
    monkeypatch.setattr("dashboard.auth.AUTH_DIR", str(tmp_path))

    # Initial creation
    ok, msg = create_account("examiner_lead", "MasterPass2026!")
    assert ok is True
    assert "created successfully" in msg.lower()

    # Duplicate check (case-insensitive)
    ok_dup, msg_dup = create_account("EXAMINER_LEAD", "DifferentPass!")
    assert ok_dup is False
    assert "already exists" in msg_dup.lower()

    # Authenticate with correct credentials
    ok_auth, msg_auth, user = authenticate_user("examiner_lead", "MasterPass2026!")
    assert ok_auth is True
    assert user is not None
    assert user["username"] == "examiner_lead"

    # Authenticate with incorrect credentials
    ok_bad, msg_bad, _ = authenticate_user("examiner_lead", "IncorrectPass!")
    assert ok_bad is False
    assert "incorrect password" in msg_bad.lower()

    # Non-existent user
    ok_none, msg_none, _ = authenticate_user("nonexistent_user", "AnyPass")
    assert ok_none is False
    assert "not found" in msg_none.lower()


def test_protected_view_definitions():
    """Verify protected view detection."""
    assert is_view_protected("New Analysis") is True
    assert is_view_protected("Analyze Indoor Video") is True
    assert is_view_protected("Spatial Analysis Studio") is True
    assert is_view_protected("new") is True
    assert is_view_protected("upload") is True
    assert is_view_protected("studio") is True

    # Public views
    assert is_view_protected("Welcome") is False
    assert is_view_protected("Retail Analysis Workflow") is False
    assert is_view_protected("Overview") is False
    assert is_view_protected("Sample Library") is False
    assert is_view_protected("workflow") is False
    assert is_view_protected("welcome") is False


def test_session_token_lifecycle_and_isolation(tmp_path, monkeypatch):
    """Verify session token creation, disk storage, and clean sign-out."""
    import streamlit as st
    from dashboard.auth import sign_in, sign_out, get_current_user, is_authenticated, _load_active_sessions

    fake_sessions_file = str(tmp_path / "test_active_sessions.json")
    monkeypatch.setattr("dashboard.auth.ACTIVE_SESSIONS_FILE", fake_sessions_file)
    monkeypatch.setattr("dashboard.auth.AUTH_DIR", str(tmp_path))

    # Clear any previous session state
    st.session_state.clear()
    assert is_authenticated() is False
    assert get_current_user() is None

    # Sign in
    token = sign_in("examiner_isolated")
    assert token is not None
    assert len(token) == 32
    assert is_authenticated() is True
    assert get_current_user() == "examiner_isolated"

    # Verify active sessions storage on disk
    active_sessions = _load_active_sessions()
    assert token in active_sessions
    assert active_sessions[token]["username"] == "examiner_isolated"

    # Simulate separate incognito session by clearing in-memory state and token request
    st.session_state.clear()
    # Without cookie or token, it must remain guest
    assert get_current_user() is None
    assert is_authenticated() is False

    # Simulate restoring token in the original browser
    st.session_state["auth_session_token"] = token
    assert get_current_user() == "examiner_isolated"
    assert is_authenticated() is True

    # Sign out
    sign_out()
    assert is_authenticated() is False
    assert get_current_user() is None
    active_after = _load_active_sessions()
    assert token not in active_after
