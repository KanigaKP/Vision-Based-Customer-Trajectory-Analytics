"""
Welcome & Animated Sign-In View for Customer Trajectory Analytics.
===================================================================
1. Impressive retail-themed 3D animation with shop floor, customer trajectories,
   and a softly highlighted display zone rendered natively without markup leak.
2. Compact Sign In / Create Account form in a readable frosted-glass panel.
3. Sidebar, guest links, decorative badges, signed-out Deploy control, and extra panels hidden.
4. Static fallback for reduced motion and responsive layout on mobile.
5. Persistent local authentication via PBKDF2-HMAC-SHA256 password hashing.
"""

import os
import html
import streamlit as st
import streamlit.components.v1 as components

from dashboard.auth import (
    create_account,
    authenticate_user,
    is_authenticated,
    get_current_user,
    sign_in,
    sign_out,
)


def get_retail_3d_animation_html() -> str:
    """Generate self-contained HTML/CSS/SVG document for the animated 3D retail perspective floor."""
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: radial-gradient(circle at 75% 25%, #132742 0%, #090F1D 55%, #050811 100%);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 16px;
}

.retail-scene-viewport {
    position: relative;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 16px;
    box-shadow: inset 0 0 70px rgba(0, 0, 0, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

/* Ambient soft depth lights */
.ambient-glow {
    position: absolute;
    border-radius: 50%;
    filter: blur(65px);
    pointer-events: none;
    opacity: 0.45;
}
.ambient-teal {
    top: 6%;
    right: 14%;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, #0D9488 0%, transparent 70%);
    animation: floatGlow 10s ease-in-out infinite alternate;
}
.ambient-amber {
    bottom: 8%;
    left: 14%;
    width: 240px;
    height: 240px;
    background: radial-gradient(circle, #D97706 0%, transparent 70%);
    animation: floatGlow 12s ease-in-out infinite alternate-reverse;
}

@keyframes floatGlow {
    0% { transform: translateY(0px) scale(1); }
    100% { transform: translateY(-22px) scale(1.12); }
}

/* 3D Perspective Stage */
.retail-stage-3d {
    perspective: 850px;
    perspective-origin: 50% 40%;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Store Floor Plane with isometric tilt */
.store-floor-plane {
    position: relative;
    width: 620px;
    height: 420px;
    transform: rotateX(56deg) rotateZ(-26deg);
    transform-style: preserve-3d;
    background: rgba(15, 23, 42, 0.78);
    border: 2px solid rgba(45, 212, 191, 0.35);
    box-shadow: 0 35px 70px rgba(0, 0, 0, 0.8), inset 0 0 50px rgba(15, 118, 110, 0.2);
    border-radius: 12px;
}

/* Grid tile pattern */
.floor-grid-lines {
    position: absolute;
    inset: 0;
    background-size: 38px 38px;
    background-image:
        linear-gradient(to right, rgba(255, 255, 255, 0.05) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(255, 255, 255, 0.05) 1px, transparent 1px);
    border-radius: 10px;
}

/* Retail fixtures */
.store-fixture {
    position: absolute;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
    border: 1px solid rgba(148, 163, 184, 0.3);
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.6);
}
.fixture-label {
    font-size: 0.58rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    color: #94A3B8;
    transform: rotateZ(26deg);
}

.fixture-wall-left {
    left: 25px;
    top: 50px;
    width: 65px;
    height: 190px;
    border-left: 3px solid #0EA5E9;
}
.fixture-shelf-island {
    left: 175px;
    top: 85px;
    width: 125px;
    height: 70px;
    border-top: 3px solid #6366F1;
}
.fixture-checkout {
    right: 35px;
    bottom: 40px;
    width: 100px;
    height: 55px;
    border-right: 3px solid #10B981;
}

/* Highlighted Display Zone Pedestal */
.highlighted-display-zone {
    position: absolute;
    left: 350px;
    top: 165px;
    width: 110px;
    height: 95px;
    transform-style: preserve-3d;
}
.display-pedestal {
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.28) 0%, rgba(217, 119, 6, 0.15) 100%);
    border: 2px solid #F59E0B;
    border-radius: 8px;
    box-shadow: 0 0 24px rgba(245, 158, 11, 0.55), inset 0 0 16px rgba(245, 158, 11, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    animation: softGlow 3s ease-in-out infinite alternate;
}
.pedestal-tag {
    font-size: 0.62rem;
    font-weight: 850;
    letter-spacing: 0.06em;
    color: #FDE68A;
    transform: rotateZ(26deg);
    text-align: center;
    padding: 2px 4px;
    text-shadow: 0 0 8px rgba(245, 158, 11, 0.8);
}
.pulse-ring {
    position: absolute;
    inset: -12px;
    border: 1.5px dashed rgba(245, 158, 11, 0.65);
    border-radius: 14px;
    animation: pulseRing 4s linear infinite;
}

@keyframes softGlow {
    0% { box-shadow: 0 0 16px rgba(245, 158, 11, 0.35), inset 0 0 10px rgba(245, 158, 11, 0.25); }
    100% { box-shadow: 0 0 32px rgba(245, 158, 11, 0.7), inset 0 0 20px rgba(245, 158, 11, 0.45); }
}
@keyframes pulseRing {
    0% { transform: scale(0.96) rotate(0deg); opacity: 0.4; }
    50% { transform: scale(1.05) rotate(180deg); opacity: 0.9; }
    100% { transform: scale(0.96) rotate(360deg); opacity: 0.4; }
}

/* Trajectories SVG */
.trajectories-svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    overflow: visible;
}
.trajectory-line {
    stroke-dasharray: 10, 6;
    animation: dashMove 6s linear infinite;
}
.traj-2 {
    animation-duration: 8s;
}
.traj-3 {
    animation-duration: 10s;
}
@keyframes dashMove {
    to { stroke-dashoffset: -96; }
}

/* Shopper Pulse Dots */
.shopper-dot {
    animation: moveShopper1 6s cubic-bezier(0.4, 0, 0.2, 1) infinite;
}
.dot-2 {
    animation: moveShopper2 7.5s cubic-bezier(0.4, 0, 0.2, 1) infinite 1.2s;
}

@keyframes moveShopper1 {
    0%   { cx: 110px; cy: 480px; opacity: 0; }
    12%  { opacity: 1; }
    50%  { cx: 250px; cy: 310px; }
    85%  { cx: 395px; cy: 215px; opacity: 1; }
    96%  { cx: 405px; cy: 210px; opacity: 0; }
    100% { cx: 405px; cy: 210px; opacity: 0; }
}
@keyframes moveShopper2 {
    0%   { cx: 135px; cy: 190px; opacity: 0; }
    12%  { opacity: 1; }
    45%  { cx: 280px; cy: 220px; }
    80%  { cx: 395px; cy: 215px; opacity: 1; }
    95%  { cx: 405px; cy: 210px; opacity: 0; }
    100% { cx: 405px; cy: 210px; opacity: 0; }
}

/* Reduced Motion Fallback */
@media (prefers-reduced-motion: reduce) {
    .ambient-glow, .display-pedestal, .pulse-ring, .trajectory-line, .shopper-dot {
        animation: none !important;
    }
    .trajectory-line { stroke-dasharray: none !important; }
}

/* Mobile adjustments */
@media (max-width: 600px) {
    .store-floor-plane {
        width: 320px;
        height: 230px;
        transform: rotateX(56deg) rotateZ(-26deg) scale(0.62);
    }
}
</style>
</head>
<body>
<div class="retail-scene-viewport">
    <div class="ambient-glow ambient-teal"></div>
    <div class="ambient-glow ambient-amber"></div>

    <div class="retail-stage-3d">
        <div class="store-floor-plane">
            <div class="floor-grid-lines"></div>

            <div class="store-fixture fixture-wall-left">
                <span class="fixture-label">WALL DISPLAY</span>
            </div>
            <div class="store-fixture fixture-shelf-island">
                <span class="fixture-label">MERCHANDISE AISLE</span>
            </div>
            <div class="store-fixture fixture-checkout">
                <span class="fixture-label">CHECKOUT</span>
            </div>

            <div class="highlighted-display-zone">
                <div class="pulse-ring"></div>
                <div class="display-pedestal">
                    <span class="pedestal-tag">Area to Review</span>
                </div>
            </div>

            <svg class="trajectories-svg" viewBox="0 0 620 420" preserveAspectRatio="none">
                <defs>
                    <linearGradient id="gTeal" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#2DD4BF" stop-opacity="0.2" />
                        <stop offset="60%" stop-color="#2DD4BF" stop-opacity="0.9" />
                        <stop offset="100%" stop-color="#14B8A6" stop-opacity="1" />
                    </linearGradient>
                    <linearGradient id="gAmber" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#F59E0B" stop-opacity="0.2" />
                        <stop offset="70%" stop-color="#F59E0B" stop-opacity="0.95" />
                        <stop offset="100%" stop-color="#FBBF24" stop-opacity="1" />
                    </linearGradient>
                    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur stdDeviation="3" result="blur" />
                        <feComposite in="SourceGraphic" in2="blur" operator="over" />
                    </filter>
                </defs>

                <path class="trajectory-line traj-1"
                      d="M 110 480 Q 195 385 235 320 T 345 240 T 395 212"
                      fill="none" stroke="url(#gTeal)" stroke-width="3.5" filter="url(#glow)" />

                <path class="trajectory-line traj-2"
                      d="M 135 190 Q 215 180 280 220 T 395 215"
                      fill="none" stroke="url(#gAmber)" stroke-width="3" filter="url(#glow)" />

                <path class="trajectory-line traj-3"
                      d="M 400 220 Q 480 240 530 305 T 580 395"
                      fill="none" stroke="url(#gTeal)" stroke-width="2.5" stroke-dasharray="6,4" />

                <circle class="shopper-dot dot-1" r="5.5" fill="#5EEAD4" filter="url(#glow)" />
                <circle class="shopper-dot dot-2" r="5.5" fill="#FDE68A" filter="url(#glow)" />
            </svg>
        </div>
    </div>
</div>
</body>
</html>"""


def render_welcome_view(reports_dir: str = "outputs/reports") -> None:
    """Render the impressive animated sign-in screen as the default landing page for signed-out visitors."""
    # Apply full-page dark retail backdrop and hide sidebar and header controls
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, [data-testid="stHeader"], [data-testid="stAppDeployButton"], .stDeployButton {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 75% 25%, #132742 0%, #090F1D 55%, #050811 100%) !important;
            background-attachment: fixed !important;
        }
        .block-container { max-width: 1280px !important; padding-top: 1.8rem !important; }

        /* Readable frosted glass panel for authentication form */
        [data-testid="stForm"] {
            background: rgba(15, 23, 42, 0.88) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border: 1px solid rgba(255, 255, 255, 0.16) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6) !important;
        }
        [data-testid="stForm"] label, [data-testid="stForm"] p {
            color: #F1F5F9 !important;
            font-weight: 650 !important;
        }
        [data-testid="stForm"] input {
            background: rgba(30, 41, 59, 0.95) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.22) !important;
            border-radius: 8px !important;
        }
        [data-testid="stForm"] input:focus {
            border-color: #2DD4BF !important;
            box-shadow: 0 0 0 2px rgba(45, 212, 191, 0.3) !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            background: transparent !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.15) !important;
            gap: 0.5rem !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            color: #94A3B8 !important;
            background: transparent !important;
            border-radius: 8px 8px 0 0 !important;
        }
        [data-testid="stTabs"] [aria-selected="true"] {
            color: #2DD4BF !important;
            background: rgba(45, 212, 191, 0.12) !important;
            border-bottom: 2px solid #2DD4BF !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: transparent !important;
            border-color: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Main Grid Layout: 3D Retail Animation alongside Frosted-Glass Sign-In Panel
    col_visual, col_form = st.columns([1.25, 0.95], gap="large")

    with col_visual:
        # Project branding: clean typography without decorative badges
        st.markdown(
            """
            <div style="margin-bottom: 0.85rem;">
                <h1 style="color:#FFFFFF !important; font-size:clamp(1.85rem, 3.0vw, 2.45rem); font-weight:800;
                           letter-spacing:-0.03em; line-height:1.2; margin:0 0 0.5rem 0;">
                    Vision-Based Customer Trajectory Analytics
                </h1>
                <p style="color:#94A3B8; font-size:0.98rem; line-height:1.55; max-width:560px; margin:0 0 0.5rem 0;">
                    Transform standard overhead security camera recordings into customer circulation heatmaps,
                    shelf dwell measurements, and evidence-backed display placement recommendations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3D Animation Component embedded cleanly via iframe (no markdown code-block escaping)
        components.html(get_retail_3d_animation_html(), height=460, scrolling=False)

    with col_form:
        # Frosted-Glass Card for Authentication
        st.markdown(
            """
            <div style="margin-top:0.4rem; margin-bottom:0.8rem;">
                <h2 style="color:#FFFFFF !important; font-size:1.35rem; font-weight:800; margin:0;">
                    Sign in to your store workspace
                </h2>
                <div style="color:#94A3B8; font-size:0.85rem; margin-top:2px;">
                    Enter your local credentials or create an account
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            tab_signin, tab_signup = st.tabs(["🔑 Sign In", "📝 Create Account"])

            # -------------------------------------------------------------
            # SIGN IN TAB
            # -------------------------------------------------------------
            with tab_signin:
                with st.form("signin_form", clear_on_submit=False):
                    login_user = st.text_input("Username", placeholder="e.g. store_owner", key="inp_signin_user")
                    login_pass = st.text_input("Password", type="password", placeholder="Enter your password", key="inp_signin_pass")
                    submitted_login = st.form_submit_button("Sign In ➔", type="primary", use_container_width=True)

                    if submitted_login:
                        if not login_user or not login_pass:
                            st.error("Please enter both username and password.")
                        else:
                            ok, msg, user_data = authenticate_user(login_user, login_pass)
                            if ok:
                                sign_in(login_user)
                                st.session_state["pending_navigation_view"] = "Home"
                                st.session_state["effective_nav_view"] = "Home"
                                st.query_params["view"] = "home"
                                st.rerun()
                            else:
                                st.error(msg)

            # -------------------------------------------------------------
            # CREATE ACCOUNT TAB
            # -------------------------------------------------------------
            with tab_signup:
                with st.form("signup_form", clear_on_submit=False):
                    new_user = st.text_input("Choose Username", placeholder="e.g. retail_manager", key="inp_signup_user")
                    new_pass = st.text_input("Create Password", type="password", placeholder="At least 4 characters", key="inp_signup_pass")
                    confirm_pass = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="inp_signup_confirm")
                    submitted_signup = st.form_submit_button("Create Account ➔", type="secondary", use_container_width=True)

                    if submitted_signup:
                        if not new_user or not new_pass:
                            st.error("Username and password cannot be empty.")
                        elif new_pass != confirm_pass:
                            st.error("Passwords do not match. Please re-enter identical passwords.")
                        else:
                            ok, msg = create_account(new_user, new_pass)
                            if ok:
                                st.success(f"✅ {msg}")
                                st.info("Account created. Switch to the **Sign In** tab to enter your workspace.")
                            else:
                                st.error(f"❌ {msg}")


def render_protected_access_barrier(view_name: str) -> None:
    """Render a clean access barrier when an unauthenticated visitor directly enters a protected feature."""
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, [data-testid="stHeader"], [data-testid="stAppDeployButton"], .stDeployButton {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
                    border: 1px solid #475569; border-radius: 14px; padding: 1.8rem 2rem;
                    color: #FFFFFF; margin: 2rem auto; max-width: 640px; box-shadow: 0 10px 30px rgba(0,0,0,0.4);">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="background:#EF4444; color:#fff; font-size:0.72rem; font-weight:800; padding:2px 8px; border-radius:4px;">
                    SIGN IN REQUIRED
                </span>
                <span style="color:#94A3B8; font-size:0.82rem; font-weight:700;">Protected Feature Access · Authentication Required</span>
            </div>
            <h2 style="color:#FFFFFF !important; margin:0.25rem 0 0.5rem 0; font-size:1.5rem; font-weight:800;">
                🔒 Sign in to access {html.escape(view_name)}
            </h2>
            <div style="color:#CBD5E1; font-size:0.92rem; line-height:1.55; margin-bottom:1.2rem;">
                Custom video uploads, camera calibration, and analysis pipeline runs require an authenticated local account.
                Please sign in or create an account to continue.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_b1, c_b2, c_b3 = st.columns([1, 2, 1])
    with c_b2:
        if st.button("🔑 Go to Sign In Page ➔", type="primary", use_container_width=True, key="btn_barrier_go_login"):
            st.session_state["pending_navigation_view"] = "Welcome"
            st.session_state["effective_nav_view"] = "Welcome"
            st.query_params["view"] = "welcome"
            st.rerun()
