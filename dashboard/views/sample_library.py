"""
Sample Shop Library & Reference Store Selection View.
======================================================
Preserves the workspace overview as the shop-selection screen for both sample actions:
- View Sample Shops: opens saved results immediately ("View Results ➔").
- Test Sample Shops: runs the existing analysis again with matching camera setup and zones ("Run Analysis ➔").

Customer-Facing Store Names:
1. Supermarket Aisle (MERL Shopping Dataset)
2. Home Décor Boutique (PRCV Site 03)
3. Jewelry Store (PRCV Site 02)
"""

import os
import json
import html
import streamlit as st
from typing import Dict, Any, Optional

from dashboard.run_adapter import (
    StandardizedRunData,
    discover_standardized_runs,
    format_observed_value,
)
from dashboard.views.main_workflow import VERIFIED_EXAMPLES, LocalVideoWrapper


def render_sample_library_view(reports_dir: str = "outputs/reports") -> None:
    """Render the simplified customer-facing Sample Shop Library."""
    # Entry mode: 'view' (default) or 'test', support query parameter override
    q_mode = st.query_params.get("mode")
    if q_mode in ("view", "test"):
        action_mode = q_mode
        st.session_state["sample_action_mode"] = q_mode
    else:
        action_mode = st.session_state.get("sample_action_mode", "view")
    is_test_mode = (action_mode == "test")

    # Discover runs from disk
    runs = discover_standardized_runs(reports_dir)
    runs_by_id = {r.run_id: r for r in runs}

    # Top Navigation Row with Contextual Back Button
    c_back, c_space = st.columns([1.5, 4.5])
    with c_back:
        if st.button("⬅️ Back to Home", key="btn_samples_back_home", type="secondary", use_container_width=True):
            st.session_state["pending_navigation_view"] = "Home"
            st.session_state["effective_nav_view"] = "Home"
            st.query_params["view"] = "home"
            st.rerun()

    # Dynamic Mode Hero Banner
    if is_test_mode:
        banner_badge = "TEST PIPELINE · FRESH RUN"
        banner_color = "#7C3AED"
        banner_border = "#DDD6FE"
        banner_bg = "linear-gradient(135deg, #1E1B4B 0%, #2E1065 50%, #0F172A 100%)"
        headline = "Test Sample Shops: Select a Store to Run Analysis"
        subtext = (
            "Choose a bundled retail store below to rerun AI person detection, tracking, and zone dwell analysis "
            "with its pre-calibrated camera mapping. This executes a fresh pipeline run while preserving your original sample results."
        )
        btn_label = "Run Analysis ➔"
        btn_type = "secondary"
    else:
        banner_badge = "SAVED RESULTS · BENCHMARK STORES"
        banner_color = "#0D9488"
        banner_border = "#99F6E4"
        banner_bg = "linear-gradient(135deg, #0F172A 0%, #134E4A 60%, #0F2E2E 100%)"
        headline = "View Sample Shops: Select a Store to View Results"
        subtext = (
            "Explore verified customer foot-traffic trajectories, shelf dwell measurements, and evidence-backed "
            "display placement recommendations across our 3 curated retail benchmark environments."
        )
        btn_label = "View Results ➔"
        btn_type = "primary"

    st.markdown(
        f"""
        <div style="background:{banner_bg}; border:1px solid rgba(255,255,255,0.12);
                    border-radius:14px; padding:1.4rem 1.8rem; color:#FFFFFF; margin:0.8rem 0 1.5rem 0;
                    box-shadow:0 6px 20px rgba(0,0,0,0.12);">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-bottom:0.4rem;">
                <span style="background:rgba(255,255,255,0.1); border:1px solid {banner_border}; color:#FFFFFF;
                             font-size:0.7rem; font-weight:800; padding:2px 10px; border-radius:9999px; letter-spacing:0.08em;">
                    {banner_badge}
                </span>
                <span style="font-size:0.75rem; color:#CBD5E1;">
                    3 Curated Retail Environments Available
                </span>
            </div>
            <h1 style="color:#FFFFFF !important; font-size:1.85rem; font-weight:800; margin:0.2rem 0 0.4rem 0; letter-spacing:-0.02em;">
                {headline}
            </h1>
            <p style="color:#E2E8F0; font-size:0.94rem; line-height:1.55; margin:0; max-width:860px;">
                {subtext}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Curated Shop Catalog Definitions
    curated_shops = [
        {
            "key": "merl",
            "name": "Supermarket Aisle",
            "icon": "🛒",
            "badge": "Grocery & Supermarket",
            "canonical_run_id": "upload_ca30f3e8",
            "fallback_id": "uploaded_runs/upload_ca30f3e8/run_summary.json",
            "rep_image": "outputs/reports/uploaded_runs/upload_ca30f3e8/run_tracking_frame.png",
            "camera_coverage": (
                "Ceiling-mounted angled security camera overlooking a snack and beverage shelf aisle "
                "leading toward the store checkout counter."
            ),
            "about_recording": (
                "Stationary retail surveillance clip (42.2 seconds, 30 FPS, 960×540 px) capturing authentic "
                "shopper browsing trajectories and shelf stops."
            ),
            "what_it_shows": (
                "Highlights shopper dwell along grocery shelves (Shelf A recorded 10.2s dwell). "
                "Note: Camera coverage is boundary-truncated; customer paths do not measure storewide sales conversion."
            ),
            "technical_id": "MERL Shopping Dataset (10_1_crop.mp4) · Run: upload_ca30f3e8",
            "profile_path": "configs/merl_scene_profile.json",
            "video_path": "data/merl/10_1_crop.mp4",
        },
        {
            "key": "site03",
            "name": "Home Décor Boutique",
            "icon": "🛋️",
            "badge": "Lifestyle & Gift Boutique",
            "canonical_run_id": "upload_6f969f54",
            "fallback_id": "uploaded_runs/upload_38d7afe1/run_summary.json",
            "rep_image": "outputs/reports/uploaded_runs/upload_38d7afe1/run_tracking_frame.png",
            "camera_coverage": (
                "Ceiling overhead camera monitoring perimeter wall display shelving, center merchandise tables, "
                "and customer walkways."
            ),
            "about_recording": (
                "Stationary overhead camera clip (26.7 seconds, 15 FPS, 896×672 px) tracking customer circulation "
                "among display tables and side shelving."
            ),
            "what_it_shows": (
                "Shows shopper dwell across merchandising tables and perimeter shelves (Left Wall Display recorded 1.6s dwell). "
                "Note: Metric scale is uncalibrated; findings are descriptive traffic observations."
            ),
            "technical_id": "PRCV Multiview Retail Site 03 (C0001020) · Run: upload_6f969f54",
            "profile_path": "configs/prcv_site03_scene_profile.json",
            "video_path": "data/prcv_site03_boutique.mp4",
        },
        {
            "key": "site02",
            "name": "Jewelry Store",
            "icon": "💎",
            "badge": "Luxury Jewelry & Watches",
            "canonical_run_id": "upload_site02_jewelry",
            "fallback_id": "uploaded_runs/upload_site02_jewelry/run_summary.json",
            "rep_image": "outputs/reports/uploaded_runs/upload_site02_jewelry/run_tracking_frame.png",
            "camera_coverage": (
                "Overhead security camera overlooking illuminated glass showcases and the customer browsing aisle."
            ),
            "about_recording": (
                "Stationary boutique recording (20.0 seconds, 15 FPS, 896×672 px) capturing multi-zone "
                "browsing engagement across display counters."
            ),
            "what_it_shows": (
                "Compares customer attention between showcases (Rear Showcase recorded 2.8s vs Front Showcase 0.6s). "
                "Important Note: Candidate was selected at zone level; within-zone hotspot was Not evaluated."
            ),
            "technical_id": "PRCV Multiview Retail Site 02 · Run: upload_site02_jewelry",
            "profile_path": "configs/prcv_site02_scene_profile.json",
            "video_path": "data/prcv_site02_jewelry.mp4",
        },
    ]

    col1, col2, col3 = st.columns(3, gap="medium")
    cols = [col1, col2, col3]

    for col, shop in zip(cols, curated_shops):
        with col:
            with st.container(border=True):
                # Header with icon and category badge
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                        <span style="font-size:1.15rem; font-weight:800; color:#0F172A;">
                            {shop['icon']} {shop['name']}
                        </span>
                        <span style="background:#F1F5F9; color:#475569; font-size:0.68rem; font-weight:750;
                                     padding:2px 8px; border-radius:4px; border:1px solid #E2E8F0;">
                            {shop['badge']}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Representative Image from this shop's own recording
                rep_img = shop["rep_image"]
                if not os.path.isfile(rep_img):
                    # Check alternate paths if needed
                    alt_path = os.path.join("outputs", "reports", "uploaded_runs", shop["canonical_run_id"], "run_tracking_frame.png")
                    if os.path.isfile(alt_path):
                        rep_img = alt_path

                if os.path.isfile(rep_img):
                    st.image(rep_img, caption=f"Footage: {shop['name']}", use_container_width=True)
                else:
                    st.caption("Representative recording frame available in run outputs.")

                # Camera Coverage
                st.markdown(f"**Camera View:** {shop['camera_coverage']}")

                # About this recording
                st.markdown(f"**About this recording:** {shop['about_recording']}")

                # What this recording can show
                st.markdown(
                    f"<div style='background:#F8FAFC; border-left:3px solid {banner_color}; padding:6px 10px; "
                    f"border-radius:4px; font-size:0.83rem; color:#334155; margin:0.5rem 0; line-height:1.45;'>"
                    f"<strong>What this recording can show:</strong> {shop['what_it_shows']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Technical Details Collapsed Expander
                with st.expander("🔬 Technical Details", expanded=False):
                    st.caption(f"**Dataset Identifier:** `{shop['technical_id']}`")
                    st.caption(f"**Camera Setup:** `{shop['profile_path']}`")
                    st.caption(f"**Video File:** `{shop['video_path']}`")

                # Action Button: View Results or Run Analysis
                st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)
                if is_test_mode:
                    if st.button(
                        f"🧪 Test {shop['name']} ➔",
                        key=f"btn_action_test_{shop['key']}",
                        type="secondary",
                        use_container_width=True,
                    ):
                        # Carry matching camera setup and zones forward to Analysis task
                        ex_info = VERIFIED_EXAMPLES.get(shop["key"])
                        if ex_info and os.path.exists(ex_info["profile_path"]):
                            with open(ex_info["profile_path"], "r", encoding="utf-8") as fp:
                                st.session_state["workflow_scene_profile"] = json.load(fp)
                        if ex_info and os.path.exists(ex_info["video_path"]):
                            st.session_state["workflow_uploaded_file"] = LocalVideoWrapper(ex_info["video_path"])
                        st.session_state["workflow_selected_example"] = shop["key"]
                        st.session_state["workflow_source_mode"] = "example"
                        st.session_state["workflow_geometry_confirmed"] = True
                        st.session_state["workflow_active_task"] = "analysis"
                        st.session_state["workflow_active_stage"] = 5
                        st.session_state["pending_navigation_view"] = "Retail Analysis"
                        st.session_state["effective_nav_view"] = "Retail Analysis"
                        st.session_state.pop("authoritative_run_id", None)
                        st.session_state.pop("_last_seen_q_run_id", None)
                        st.session_state.pop("_last_seen_q_stage", None)
                        st.session_state.pop("analysis_error", None)
                        st.session_state["is_analyzing"] = False
                        st.query_params.clear()
                        st.query_params["view"] = "workflow"
                        st.query_params["task"] = "analysis"
                        st.query_params["stage"] = "5"
                        st.rerun()
                else:
                    if st.button(
                        f"📊 {btn_label}",
                        key=f"btn_action_view_{shop['key']}",
                        type="primary",
                        use_container_width=True,
                    ):
                        # Open saved results immediately
                        st.session_state["authoritative_run_id"] = shop["canonical_run_id"]
                        st.session_state["_last_seen_q_run_id"] = shop["canonical_run_id"]
                        st.session_state["_last_seen_q_stage"] = 7
                        st.session_state["workflow_selected_example"] = shop["key"]
                        ex_info = VERIFIED_EXAMPLES.get(shop["key"])
                        if ex_info and os.path.exists(ex_info["profile_path"]):
                            with open(ex_info["profile_path"], "r", encoding="utf-8") as fp:
                                st.session_state["workflow_scene_profile"] = json.load(fp)
                        if ex_info and os.path.exists(ex_info["video_path"]):
                            st.session_state["workflow_uploaded_file"] = LocalVideoWrapper(ex_info["video_path"])
                        st.session_state["workflow_active_task"] = "results"
                        st.session_state["workflow_active_stage"] = 7
                        st.session_state["pending_navigation_view"] = "Retail Analysis"
                        st.session_state["effective_nav_view"] = "Retail Analysis"
                        st.query_params.clear()
                        st.query_params["view"] = "workflow"
                        st.query_params["task"] = "results"
                        st.query_params["stage"] = "7"
                        st.query_params["run_id"] = shop["canonical_run_id"]
                        st.rerun()


def render_overview_view(reports_dir: str = "outputs/reports") -> None:
    """Redirect or alias Overview to Sample Shop Library per Requirement 3."""
    render_sample_library_view(reports_dir)
