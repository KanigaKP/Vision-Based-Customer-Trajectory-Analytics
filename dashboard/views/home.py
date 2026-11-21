"""
Home View for Customer Trajectory Analytics.
=============================================
Presents exactly three primary actions after login:
1. View Sample Shops — open saved results immediately (Teal accent).
2. Test Sample Shops — rerun analysis with matching camera profile (Violet accent).
3. Upload a Recording — select video, calibrate, configure zones, and analyze (Amber accent).
"""

import html
import streamlit as st
from dashboard.auth import get_current_user


def render_home_view() -> None:
    """Render the simplified 3-action Home dashboard for authenticated shop owners."""
    user = get_current_user() or "Store Owner"

    # Executive Hero Banner
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #172554 100%);
                    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px;
                    padding: 1.6rem 2rem; color: #FFFFFF; margin-bottom: 1.5rem;
                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.4rem;">
                <div style="font-size:0.75rem; font-weight:800; letter-spacing:0.12em; color:#2DD4BF; text-transform:uppercase;">
                    Retail Store Analytics · Control Center
                </div>
                <div style="background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15);
                            border-radius:9999px; padding:3px 12px; font-size:0.75rem; color:#E2E8F0; font-weight:700;">
                    👤 Signed in as {html.escape(user)}
                </div>
            </div>
            <h1 style="color:#FFFFFF !important; font-size:clamp(1.7rem, 2.8vw, 2.3rem); font-weight:800;
                       letter-spacing:-0.03em; margin:0.25rem 0 0.5rem 0;">
                Store Traffic & Display Optimization
            </h1>
            <p style="color:#CBD5E1; font-size:0.98rem; line-height:1.6; margin:0; max-width:820px;">
                Understand how customers navigate your store, see where they linger, and identify high-attention
                display locations to boost merchandise visibility. Select what you would like to do:
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # EXACTLY THREE PRIMARY CARDS
    col_view, col_test, col_upload = st.columns(3, gap="medium")

    # -------------------------------------------------------------
    # CARD 1: View Sample Shops (Teal Accent)
    # -------------------------------------------------------------
    with col_view:
        with st.container(border=True):
            st.markdown(
                """
                <div style="min-height: 180px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                        <span style="background:rgba(13,148,136,0.15); color:#0D9488; border:1px solid #99F6E4;
                                     font-size:0.7rem; font-weight:800; padding:3px 8px; border-radius:6px; letter-spacing:0.06em;">
                            SAVED RESULTS
                        </span>
                        <span style="font-size:1.4rem;">🏪</span>
                    </div>
                    <h3 style="color:#0F172A; font-size:1.25rem; font-weight:800; margin:0.2rem 0 0.4rem 0;">
                        View Sample Shops
                    </h3>
                    <p style="color:#475569; font-size:0.88rem; line-height:1.55; margin:0 0 0.6rem 0;">
                        Explore saved findings and supporting footage.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("View Sample Shops ➔", key="home_btn_view_samples", type="primary", use_container_width=True):
                st.session_state["sample_action_mode"] = "view"
                st.session_state["pending_navigation_view"] = "Sample Shops"
                st.session_state["effective_nav_view"] = "Sample Shops"
                st.session_state.pop("_last_seen_q_run_id", None)
                st.session_state.pop("_last_seen_q_stage", None)
                st.query_params.clear()
                st.query_params["view"] = "samples"
                st.rerun()

    # -------------------------------------------------------------
    # CARD 2: Test Sample Shops (Violet Accent)
    # -------------------------------------------------------------
    with col_test:
        with st.container(border=True):
            st.markdown(
                """
                <div style="min-height: 180px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                        <span style="background:rgba(124,58,237,0.12); color:#7C3AED; border:1px solid #DDD6FE;
                                     font-size:0.7rem; font-weight:800; padding:3px 8px; border-radius:6px; letter-spacing:0.06em;">
                            PIPELINE EXECUTION
                        </span>
                        <span style="font-size:1.4rem;">🧪</span>
                    </div>
                    <h3 style="color:#0F172A; font-size:1.25rem; font-weight:800; margin:0.2rem 0 0.4rem 0;">
                        Test Sample Shops
                    </h3>
                    <p style="color:#475569; font-size:0.88rem; line-height:1.55; margin:0 0 0.4rem 0;">
                        Run the full analysis on a bundled shop recording.
                    </p>
                    <div style="font-size:0.78rem; color:#64748B; background:#F8FAFC; border-radius:6px; padding:4px 8px; margin-bottom:0.6rem;">
                        ℹ️ Creates a new result from the same recording while keeping original sample results available.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Test Sample Shops ➔", key="home_btn_test_samples", type="secondary", use_container_width=True):
                st.session_state["sample_action_mode"] = "test"
                st.session_state["pending_navigation_view"] = "Sample Shops"
                st.session_state["effective_nav_view"] = "Sample Shops"
                st.session_state.pop("_last_seen_q_run_id", None)
                st.session_state.pop("_last_seen_q_stage", None)
                st.query_params.clear()
                st.query_params["view"] = "samples"
                st.rerun()

    # -------------------------------------------------------------
    # CARD 3: Upload a Recording (Amber Accent)
    # -------------------------------------------------------------
    with col_upload:
        with st.container(border=True):
            st.markdown(
                """
                <div style="min-height: 180px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                        <span style="background:rgba(217,119,6,0.12); color:#D97706; border:1px solid #FDE68A;
                                     font-size:0.7rem; font-weight:800; padding:3px 8px; border-radius:6px; letter-spacing:0.06em;">
                            CUSTOM VIDEO
                        </span>
                        <span style="font-size:1.4rem;">📹</span>
                    </div>
                    <h3 style="color:#0F172A; font-size:1.25rem; font-weight:800; margin:0.2rem 0 0.4rem 0;">
                        Upload a Recording
                    </h3>
                    <p style="color:#475569; font-size:0.88rem; line-height:1.55; margin:0 0 0.6rem 0;">
                        Analyse your own recording after confirming its camera mapping and zones.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Upload a Recording ➔", key="home_btn_upload_recording", type="secondary", use_container_width=True):
                st.session_state["workflow_source_mode"] = "upload"
                st.session_state["workflow_active_task"] = "recording"
                st.session_state["workflow_active_stage"] = 1
                st.session_state["workflow_landing_action"] = "upload"
                st.session_state["workflow_landing_action_choice"] = "📹 Upload a Recording"
                st.session_state["pending_navigation_view"] = "Retail Analysis"
                st.session_state["effective_nav_view"] = "Retail Analysis"
                st.session_state.pop("_last_seen_q_run_id", None)
                st.session_state.pop("_last_seen_q_stage", None)
                st.query_params.clear()
                st.query_params["view"] = "workflow"
                st.query_params["task"] = "recording"
                st.query_params["stage"] = "1"
                st.rerun()

    # Bottom Quick Reference
    st.markdown(
        """
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px;
                    padding:1rem 1.4rem; margin-top:1.5rem; display:flex; justify-content:space-between;
                    align-items:center; flex-wrap:wrap; gap:0.75rem; font-size:0.85rem; color:#64748B;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-weight:750; color:#0F172A;">Bundled Retail Environments:</span>
                <span>🛒 Supermarket Aisle</span> • <span>🛋️ Home Décor Boutique</span> • <span>💎 Jewelry Store</span>
            </div>
            <div style="font-size:0.8rem; color:#94A3B8;">
                Offline YOLO11 + ByteTrack AI Engine · 100% Local Processing
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
