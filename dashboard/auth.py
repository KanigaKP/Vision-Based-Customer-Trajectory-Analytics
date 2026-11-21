"""
Authentication and local account management for Customer Trajectory Analytics.
=============================================================================
Provides persistent, offline local user accounts using PBKDF2-HMAC-SHA256 password hashing.
Invariants:
1. Strictly local & offline: accounts stored in `outputs/.auth/accounts.json`.
2. Credentials kept out of logs and run artifacts.
3. Passwords salted with 16 random bytes and hashed with 100,000 iterations.
4. Duplicate username handling and case-insensitive uniqueness.
5. Session state integration with Streamlit (`authenticated_user`).
"""

import os
import json
import hmac
import hashlib
import secrets
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
import streamlit as st

AUTH_DIR = os.path.join("outputs", ".auth")
ACCOUNTS_FILE = os.path.join(AUTH_DIR, "accounts.json")
ACTIVE_SESSIONS_FILE = os.path.join(AUTH_DIR, "active_sessions.json")
LEGACY_SESSION_FILE = os.path.join(AUTH_DIR, "active_session.json")


def _get_accounts_file_path() -> str:
    return ACCOUNTS_FILE


def load_accounts() -> Dict[str, Any]:
    """Load accounts dictionary from local disk."""
    if not os.path.exists(ACCOUNTS_FILE):
        return {}
    try:
        with open(ACCOUNTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_accounts(accounts: Dict[str, Any]) -> bool:
    """Save accounts dictionary to local disk securely."""
    try:
        os.makedirs(AUTH_DIR, exist_ok=True)
        with open(ACCOUNTS_FILE, "w", encoding="utf-8") as f:
            json.dump(accounts, f, indent=2)
        return True
    except Exception:
        return False


def _load_active_sessions() -> Dict[str, Any]:
    """Load session token dictionary from disk."""
    if not os.path.exists(ACTIVE_SESSIONS_FILE):
        return {}
    try:
        with open(ACTIVE_SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_active_sessions(sessions: Dict[str, Any]) -> bool:
    """Save session token dictionary to disk."""
    try:
        os.makedirs(AUTH_DIR, exist_ok=True)
        with open(ACTIVE_SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2)
        return True
    except Exception:
        return False


def _get_token_from_request() -> Optional[str]:
    """Extract session token from session state, cookies, or query parameters."""
    token = st.session_state.get("auth_session_token")
    if token:
        return str(token)

    # Check cookies
    try:
        if hasattr(st, "context") and hasattr(st.context, "cookies"):
            cookie_val = st.context.cookies.get("cta_session_token")
            if cookie_val:
                return str(cookie_val)
    except Exception:
        pass

    # Check query params
    try:
        if hasattr(st, "query_params") and "session_token" in st.query_params:
            return str(st.query_params.get("session_token"))
    except Exception:
        pass

    return None


def hash_password(password: str, salt_bytes: Optional[bytes] = None) -> Tuple[str, str]:
    """Salt and hash a password with PBKDF2-HMAC-SHA256 (100,000 iterations)."""
    if salt_bytes is None:
        salt_bytes = os.urandom(16)
    hash_bytes = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 100_000)
    return salt_bytes.hex(), hash_bytes.hex()


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    """Constant-time verification of candidate password against salt and hash."""
    try:
        salt_bytes = bytes.fromhex(salt_hex)
        cand_hash_bytes = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 100_000)
        return hmac.compare_digest(cand_hash_bytes.hex(), hash_hex)
    except Exception:
        return False


def create_account(username: str, password: str, role: str = "evaluator") -> Tuple[bool, str]:
    """Create a new local account with salted PBKDF2 password hash."""
    clean_user = username.strip()
    if not clean_user:
        return False, "Username cannot be blank."
    if len(clean_user) < 3:
        return False, "Username must be at least 3 characters long."
    if not password:
        return False, "Password cannot be blank."
    if len(password) < 4:
        return False, "Password must be at least 4 characters long."

    accounts = load_accounts()
    lookup_key = clean_user.lower()
    if lookup_key in [k.lower() for k in accounts.keys()]:
        return False, f"Username '{clean_user}' already exists. Please sign in or choose another username."

    salt_hex, hash_hex = hash_password(password)
    accounts[clean_user] = {
        "username": clean_user,
        "salt": salt_hex,
        "password_hash": hash_hex,
        "role": role,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if save_accounts(accounts):
        return True, f"Account '{clean_user}' created successfully! You can now sign in."
    else:
        return False, "Failed to persist account file. Please check local disk permissions."


def authenticate_user(username: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Authenticate user credentials against local accounts store."""
    clean_user = username.strip()
    if not clean_user or not password:
        return False, "Please enter both username and password.", None

    accounts = load_accounts()
    matched_entry = None
    for acc_user, acc_data in accounts.items():
        if acc_user.lower() == clean_user.lower():
            matched_entry = acc_data
            break

    if not matched_entry:
        return False, f"User '{clean_user}' not found. Please check spelling or create an account.", None

    salt_hex = matched_entry.get("salt", "")
    hash_hex = matched_entry.get("password_hash", "")
    if verify_password(password, salt_hex, hash_hex):
        return True, "Authentication successful.", matched_entry
    else:
        return False, "Incorrect password. Please try again.", None


def get_current_user() -> Optional[str]:
    """
    Retrieve currently authenticated username for THIS browser session.
    Isolates sessions so incognito windows and separate visitors remain guests.
    """
    # 1. In-memory session state for this WebSocket
    user = st.session_state.get("authenticated_user")
    if user:
        return str(user)

    # 2. Token lookup from request cookie or query parameter
    token = _get_token_from_request()
    if token:
        sessions = _load_active_sessions()
        if token in sessions:
            username = sessions[token].get("username")
            if username:
                st.session_state["authenticated_user"] = username
                st.session_state["auth_session_token"] = token
                return str(username)

    return None


def is_authenticated() -> bool:
    """Check if current session has an authenticated user."""
    return bool(get_current_user())


def sign_in(username: str) -> str:
    """Mark current session as authenticated, generate a session token, and isolate per browser."""
    clean = username.strip()
    token = secrets.token_hex(16)
    st.session_state["authenticated_user"] = clean
    st.session_state["auth_session_token"] = token

    sessions = _load_active_sessions()
    sessions[token] = {
        "username": clean,
        "signed_in_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_active_sessions(sessions)

    # Clean up legacy single-session file if it exists
    if os.path.exists(LEGACY_SESSION_FILE):
        try:
            os.remove(LEGACY_SESSION_FILE)
        except Exception:
            pass

    # Inject client-side cookie so reloads in the same browser retain authentication
    try:
        import streamlit.components.v1 as components
        components.html(
            f"""
            <script>
            document.cookie = "cta_session_token={token}; path=/; max-age=86400; SameSite=Lax";
            </script>
            """,
            height=0,
            width=0,
        )
    except Exception:
        pass

    return token


def sign_out() -> None:
    """Clear session authentication, invalidate session token, and clear cookies."""
    st.session_state["authenticated_user"] = None
    token = st.session_state.pop("auth_session_token", None)
    if "auth_user_role" in st.session_state:
        del st.session_state["auth_user_role"]

    if not token:
        token = _get_token_from_request()

    if token:
        sessions = _load_active_sessions()
        sessions.pop(token, None)
        _save_active_sessions(sessions)

    if os.path.exists(LEGACY_SESSION_FILE):
        try:
            os.remove(LEGACY_SESSION_FILE)
        except Exception:
            pass

    # Clear client cookie
    try:
        import streamlit.components.v1 as components
        components.html(
            """
            <script>
            document.cookie = "cta_session_token=; path=/; max-age=0; SameSite=Lax";
            </script>
            """,
            height=0,
            width=0,
        )
    except Exception:
        pass


# Views that require authentication when accessed directly
PROTECTED_VIEWS = {
    "New Analysis",
    "Analyze Indoor Video",
    "Spatial Analysis Studio",
}

PROTECTED_QUERY_PARAMS = {
    "new",
    "upload",
    "studio",
}


def is_view_protected(view_name_or_query: str) -> bool:
    """Determine whether a view name or query parameter represents a protected feature."""
    val = (view_name_or_query or "").strip().lower()
    if val in [v.lower() for v in PROTECTED_VIEWS]:
        return True
    if val in PROTECTED_QUERY_PARAMS:
        return True
    return False
