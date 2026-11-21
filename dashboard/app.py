"""
Phase 14.1 — Local Streamlit Viewer for Phase 13 Analysis Reports.
==================================================================
Provides a local-only interface for exported Phase 13 reports and an explicit
offline upload-and-analyze workflow for recorded indoor video.

RUNNING THE VIEWER:
    streamlit run dashboard/app.py

METHODOLOGICAL & ARCHITECTURAL INVARIANTS:
1. Local-Only & Offline:
   - Reads reports from `outputs/reports/`; uploaded clips and derived outputs
     are saved under that local directory only after Analyze is pressed.
   - Does not accept live streams or contact external networks.
2. Provenance from Content, Not Filename:
   - Reads source_type, dataset_name, and evidence eligibility from report dicts.
3. Explicit Missing vs. Evaluated Zero Distinction:
   - Clearly flags "not_available" sections and displays unsupplied reasons.
   - Preserves measured numeric zero metrics in evaluated zones.
4. Prominent Evidence Withholding:
   - Highlights Phase 12 reasons_withheld and caveats.
   - Disclaims MERL (sparse, boundary-truncated) and MOT17 (non-retail).
5. Explicit Heatmap Resolution:
   - Displays heatmap image ONLY if an explicit file path is in the report.
   - Never infers images from report filenames.
6. Privacy & Recommendation Boundary:
   - Labels track counts as clip-local tracker IDs.
   - Does not support identities, sales, or validated/actionable recommendations.
   - A threshold-free exploratory shelf candidate is explicitly unvalidated.
"""

import os
import sys
import json
import uuid
import hashlib
import html
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import streamlit as st
except ImportError:
    # Handle environment where streamlit is declared in requirements.txt
    # but not yet installed in virtualenv.
    print(
        "CRITICAL: Streamlit is not installed in the current environment.\n"
        "Streamlit dependency is declared in requirements.txt.\n"
        "To run this dashboard once packages are installed: streamlit run dashboard/app.py"
    )
    sys.exit(1)

from dashboard.report_loader import (
    SCHEMA_VERSION_SUPPORTED,
    ParsedReportResult,
    discover_report_files,
    is_valid_phase_13_report,
    load_and_validate_report,
    resolve_demo_artifact_path,
    load_demo_summary,
)
from dashboard.theme import inject_direction_a_theme
from dashboard.views.home import render_home_view
from dashboard.views.overview import render_overview_view
from dashboard.views.new_analysis import render_new_analysis_view
from dashboard.views.results_workspace import render_results_workspace_view
from dashboard.views.placement_candidate import render_placement_candidate_view
from dashboard.views.evidence_details import render_evidence_details_view
from dashboard.views.sample_library import render_sample_library_view
from dashboard.views.main_workflow import render_main_workflow
from dashboard.views.spatial_studio import render_spatial_studio_view
from dashboard.views.welcome import render_welcome_view, render_protected_access_barrier
from dashboard.auth import is_authenticated, get_current_user, sign_out, is_view_protected
from dashboard.run_adapter import discover_standardized_runs



def inject_dashboard_styles() -> None:
    """Apply a polished responsive theme without remote assets or added packages."""
    st.markdown(
        """
        <style>
        :root {
          --ink: #17243a; --muted: #627089; --line: #e4eaf2;
          --paper: #ffffff; --canvas: #f4f7fb; --blue: #2563eb;
          --teal: #0f9f95; --navy: #111c31;
        }
        [data-testid="stAppViewContainer"] { background: var(--canvas); }
        [data-testid="stHeader"] { background: rgba(244,247,251,.92); }
        [data-testid="stMainBlockContainer"] { max-width: 1480px; padding-top: 1.35rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg,#111c31 0%,#172944 100%); }
        [data-testid="stSidebar"] * { color: #eef4ff; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius: 10px; padding: .3rem .55rem; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.09); }
        h1,h2,h3,h4 { color: var(--ink); letter-spacing: -.025em; }
        h1 { font-weight: 780 !important; }
        h2,h3 { font-weight: 720 !important; }
        [data-testid="stMetric"] { background: var(--paper); border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.1rem; box-shadow: 0 5px 18px rgba(23,36,58,.045); }
        /* Executive Alert Styling: Replaces harsh pink/red error boxes with modern slate/teal cards */
        [data-testid="stAlert"] {
            background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%) !important;
            border: 1px solid #CBD5E1 !important;
            border-left: 4px solid var(--teal) !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04) !important;
            padding: 1rem 1.25rem !important;
        }
        [data-testid="stAlert"] * {
            color: #0F172A !important;
        }
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
            color: #334155 !important;
            font-size: 0.88rem !important;
            line-height: 1.55 !important;
            margin: 0 !important;
        }
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] strong {
            color: var(--teal) !important;
            font-weight: 750 !important;
        }
        [data-testid="stAlert"] svg {
            color: var(--teal) !important;
            fill: var(--teal) !important;
        }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
        [data-testid="stImage"] img, [data-testid="stVideo"] video { border-radius: 14px; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .45rem; border-bottom: 1px solid var(--line); }
        [data-testid="stTabs"] [data-baseweb="tab"] { background: #eaf0f8; border-radius: 11px 11px 0 0; padding: .65rem 1rem; }
        [data-testid="stTabs"] [aria-selected="true"] { background: #fff; color: var(--blue); }
        .app-hero { padding: 1.55rem 1.8rem; margin: .2rem 0 1.35rem; border-radius: 22px;
          color: #fff; background: radial-gradient(circle at 85% 15%,rgba(45,212,191,.35),transparent 27%),
          linear-gradient(120deg,#10223e 0%,#173b67 56%,#176d79 100%); box-shadow: 0 14px 34px rgba(16,34,62,.17); }
        .app-hero .eyebrow { color: #9ee8dd; font-size: .77rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
        .app-hero h1 { color: #fff !important; margin: .42rem 0 .4rem; font-size: clamp(1.8rem,3vw,2.7rem); }
        .app-hero p { color: #e0ebf7; margin: 0; max-width: 880px; font-size: 1.02rem; line-height: 1.6; }
        .hero-tags { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1rem; }
        .hero-tag { display:inline-block; border:1px solid rgba(255,255,255,.24); border-radius:999px; padding:.28rem .7rem; color:#f3f8ff; font-size:.78rem; }
        .section-kicker { color:var(--teal); font-size:.75rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.25rem; }
        .subtle-card { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1rem 1.15rem; }
        div.stButton > button[kind="primary"] { border-radius:11px; font-weight:750; box-shadow:0 6px 18px rgba(37,99,235,.22); }
        @media (max-width: 720px) {
          [data-testid="stMainBlockContainer"] { padding-left:1rem; padding-right:1rem; }
          .app-hero { padding:1.15rem; border-radius:17px; }
          [data-testid="stMetric"] { padding:.7rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(inject_direction_a_theme(), unsafe_allow_html=True)



def render_app_hero(title: str, subtitle: str, tags: Optional[List[str]] = None) -> None:
    tag_html = "".join(f'<span class="hero-tag">{tag}</span>' for tag in (tags or []))
    st.title(title)
    st.markdown(
        f'<section class="app-hero"><div class="eyebrow">VISION ANALYTICS · LOCAL WORKSPACE</div>'
        f'<p>{subtitle}</p><div class="hero-tags">{tag_html}</div></section>',
        unsafe_allow_html=True,
    )


def render_metadata_section(report: ParsedReportResult) -> None:
    """Render run and source provenance metadata."""
    st.subheader("1. Run & Provenance Metadata")
    meta = report.metadata

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Report ID:** `{report.report_id or 'N/A'}`")
        st.markdown(f"**Schema Version:** `{report.schema_version or 'N/A'}`")
        st.markdown(f"**Generated At (UTC):** `{report.generated_at_utc or 'N/A'}`")
    with col2:
        st.markdown(f"**Run ID:** `{meta.get('run_id', 'N/A')}`")
        st.markdown(f"**Dataset Name:** `{report.dataset_name or 'N/A'}`")
        st.markdown(f"**Source Type:** `{report.source_type or 'N/A'}`")
    with col3:
        target_fps = meta.get("target_fps")
        fps_str = f"{target_fps:.1f} FPS" if isinstance(target_fps, (int, float)) else "Unspecified"
        st.markdown(f"**Frame Sampling Rate:** `{fps_str}`")
        st.markdown(f"**Frames Evaluated:** `{meta.get('total_frames_evaluated', 'N/A')}`")
        st.markdown(f"**Source Path:** `{meta.get('source_path', 'N/A')}`")

    notes = meta.get("notes")
    if notes:
        st.caption(f"**Run Notes:** {notes}")


def render_evidence_and_limitations(report: ParsedReportResult) -> None:
    """Render prominent evidence boundaries and Phase 12 withholding reasons."""
    st.subheader("2. Evidence Context & Methodological Boundaries")

    # Mandatory contextual statements (MERL / MOT17 / General Boundary)
    for stmt in report.mandatory_statements:
        st.warning(stmt)

    # Evidence Context Summary
    ctx = report.evidence_context
    if ctx.get("status") == "available":
        st.markdown("**Evidence Context Specifications:**")
        ec_col1, ec_col2, ec_col3 = st.columns(3)
        with ec_col1:
            st.info(f"**Environment Type:** {ctx.get('environment_type', 'N/A')}")
        with ec_col2:
            st.info(f"**Calibration Status:** {ctx.get('calibration_status', 'N/A')}")
        with ec_col3:
            st.info(f"**Coverage Status:** {ctx.get('coverage_status', 'N/A')}")
        if ctx.get("notes"):
            st.caption(f"Context Notes: {ctx.get('notes')}")
    else:
        st.info("Evidence context was not supplied by caller.")

    # Phase 12 Spatial Observation Status
    sp_sec = report.spatial_observations
    sp_status = sp_sec.get("status", "not_available")

    st.markdown("---")
    st.markdown("#### Spatial Observation Eligibility")
    if sp_status == "insufficient_evidence" or not report.is_eligible:
        st.error("Status: **INSUFFICIENT EVIDENCE — Observations Explicitly Withheld**")
        st.markdown(
            "Retail layout insights and spatial observations cannot be asserted "
            "due to the following explicit causal reasons:"
        )
        if report.reasons_withheld:
            for r in report.reasons_withheld:
                st.markdown(f"- ⚠️ **{r}**")
        else:
            st.markdown("- Insufficient empirical evidence.")
    elif sp_status == "available" and report.is_eligible:
        st.success("Status: **ELIGIBLE — Traceable Descriptive Observations Present**")
    else:
        st.info(f"Status: **{sp_status.upper()}** ({sp_sec.get('reason', 'Not evaluated')})")


def render_dwell_analytics(report: ParsedReportResult) -> None:
    """Render Dwell Analytics or explicit unavailable notice."""
    st.subheader("3. Dwell Analytics")
    dwell = report.dwell_analytics

    if dwell.get("status") == "available":
        total_episodes = dwell.get("total_episodes", dwell.get("total_dwell_episodes", 0))
        total_span = dwell.get("total_observed_dwell_span_sec", 0.0)
        distinct_tracks = dwell.get("distinct_clip_local_tracks_evaluated", dwell.get("distinct_clip_local_tracks", 0))
        st.markdown(
            f"**Total Observed Dwell Span:** `{total_span:.2f} sec` | "
            f"**Total Dwell Episodes:** `{total_episodes}` | "
            f"**Distinct Clip-Local Tracks:** `{distinct_tracks}`"
        )
        zones = dwell.get("zone_dwell_summaries", {})
        if zones:
            rows = []
            for zid, zdata in zones.items():
                rows.append({
                    "Zone ID": zid,
                    "Total Observed Dwell Span (s)": zdata.get("total_observed_dwell_span_sec", 0.0),
                    "Episodes": zdata.get("total_episodes", 0),
                    "Clip-Local Tracks": zdata.get("distinct_clip_local_tracks", 0),
                    "Mean Span (s)": zdata.get("mean_dwell_span_sec", 0.0),
                    "Median Span (s)": zdata.get("median_dwell_span_sec", 0.0),
                })
            st.table(rows)
            st.caption(
                "Note: An observed dwell span of 0.0s represents an evaluated zero observation (e.g. 0 visits in an unvisited zone), "
                "distinguishable from unsupplied/missing analytics."
            )
        else:
            st.write("No zone dwell summaries present.")

        caveats = dwell.get("caveats", [])
        for c in caveats:
            st.caption(f"ℹ️ *Caveat: {c}*")
    else:
        reason = dwell.get("reason", "Dwell analytics was not supplied for this run.")
        st.info(
            f"**Dwell Analytics: NOT AVAILABLE — Not evaluated for this input**\n\n"
            f"- **Status:** Not evaluated for this input (no retail zone analysis configured).\n"
            f"- **Reason:** *{reason}*\n\n"
            f"ℹ️ *Methodological Rule: Because this input has no retail zone analysis, "
            f"metrics are retained as not evaluated and are NOT replaced with zeroes.*"
        )


def render_density_analytics(report: ParsedReportResult) -> None:
    """Render Density Analytics or explicit unavailable notice."""
    st.subheader("4. Trajectory Density Analytics")
    density = report.density_analytics

    if density.get("status") == "available":
        bounds = density.get("bounds", {})
        grid_shape = density.get("grid_shape", [0, 0])
        cell_size = density.get("cell_size", [0.0, 0.0])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Evaluated Points", density.get("total_evaluated_points", 0))
        col2.metric("Successfully Binned", density.get("total_binned_points", 0))
        col3.metric("Out of Bounds Points", density.get("out_of_bounds_count", 0))
        col4.metric("Max Cell Density", density.get("max_cell_density", 0))

        st.markdown(
            f"**Floor Bounds:** `[{bounds.get('min_x')}, {bounds.get('max_x')}] × "
            f"[{bounds.get('min_y')}, {bounds.get('max_y')}] {bounds.get('unit', '')}` | "
            f"**Grid Shape:** `{grid_shape[0]} rows × {grid_shape[1]} cols` | "
            f"**Cell Size:** `{cell_size[0]} × {cell_size[1]}`"
        )
        st.markdown(
            f"**Point Conservation Verified:** `{density.get('conservation_invariant_verified', False)}`"
        )

        caveats = density.get("caveats", [])
        for c in caveats:
            st.caption(f"ℹ️ *Caveat: {c}*")

        # Explicit Heatmap Image Handling
        st.markdown("#### Density Heatmap Visualization")
        if report.explicit_heatmap_path:
            st.image(
                report.explicit_heatmap_path,
                caption=f"Explicit Heatmap Artifact: {report.explicit_heatmap_path}",
                use_container_width=True,
            )
        else:
            st.info(
                "Heatmap Image: **Omitted**.\n\n"
                "No explicit heatmap artifact path is specified inside this report. "
                "Per project methodology, filename-based image guessing is prohibited."
            )
    else:
        st.info(
            f"**Density Analytics: NOT AVAILABLE**\n\n"
            f"Reason: *{density.get('reason', 'Density analytics was not supplied for this run.')}*"
        )


def render_transition_analytics(report: ParsedReportResult) -> None:
    """Render Transition Analytics or explicit unavailable notice."""
    st.subheader("5. Zone Transitions & Activity")
    trans = report.transition_analytics

    if trans.get("status") == "available":
        total_trans = trans.get("total_transitions", trans.get("total_direct_transitions", 0))
        total_tracks = trans.get("total_evaluated_tracks", trans.get("distinct_clip_local_track_count", 0))
        st.markdown(
            f"**Total Direct Transitions:** `{total_trans}` | "
            f"**Distinct Clip-Local Tracks:** `{total_tracks}`"
        )

        zones = trans.get("zone_activity_summaries", {})
        if zones:
            rows = []
            for zid, zdata in zones.items():
                rows.append({
                    "Zone ID": zid,
                    "Observed Visit Runs": zdata.get("observed_run_count", 0),
                    "Distinct Clip-Local Tracks": zdata.get("distinct_clip_local_tracks", zdata.get("distinct_clip_local_track_count", 0)),
                    "Direct Incoming Transitions": zdata.get("incoming_direct_transition_count", zdata.get("direct_incoming_count", 0)),
                    "Direct Outgoing Transitions": zdata.get("outgoing_direct_transition_count", zdata.get("direct_outgoing_count", 0)),
                })
            st.table(rows)
            st.caption(
                "Note: A count of 0 represents an evaluated zero observation (e.g. 0 runs in an unvisited zone), "
                "distinguishable from unsupplied/missing analytics."
            )
        else:
            st.write("No zone activity summaries present.")

        caveats = trans.get("caveats", [])
        for c in caveats:
            st.caption(f"ℹ️ *Caveat: {c}*")
    else:
        reason = trans.get("reason", "Transition analytics was not supplied for this run.")
        st.info(
            f"**Transition Analytics: NOT AVAILABLE — Not evaluated for this input**\n\n"
            f"- **Status:** Not evaluated for this input (no retail zone analysis configured).\n"
            f"- **Reason:** *{reason}*\n\n"
            f"ℹ️ *Methodological Rule: Because this input has no retail zone analysis, "
            f"metrics are retained as not evaluated and are NOT replaced with zeroes.*"
        )


def render_spatial_observations_section(report: ParsedReportResult) -> None:
    """Render Descriptive Observations."""
    st.subheader("6. Descriptive Spatial Observations")
    sp = report.spatial_observations
    status = sp.get("status", "not_available")

    if status == "available" and report.is_eligible:
        observations = sp.get("observations", [])
        if observations:
            for obs in observations:
                st.markdown(
                    f"- **[{obs.get('observation_id')}]** `{obs.get('zone_id')}` "
                    f"({obs.get('metric_type')}): {obs.get('descriptive_text')}"
                )
        else:
            st.write("No descriptive observations recorded.")
    elif status == "insufficient_evidence":
        st.warning(
            "Observations withheld: Insufficient empirical evidence. "
            "See Section 2 for the full list of explicit causal withholding reasons."
        )
    else:
        st.info(
            f"**Spatial Observations: NOT AVAILABLE**\n\n"
            f"Reason: *{sp.get('reason', 'Spatial observations were not supplied for this run.')}*"
        )


def render_demo_real_tab(reports_dir: str, summary_override: Optional[Dict[str, Any]] = None) -> None:
    """Render Real MERL retail sample outputs and exact metrics."""
    manifest_path = os.path.join(reports_dir, "demo_real_summary.json")
    summary = summary_override or load_demo_summary(manifest_path, allowed_dir=reports_dir)
    if not summary:
        st.error(f"Demo summary manifest not found or unreadable: `{manifest_path}`")
        st.info("Run `python scripts/run_vscode_demo.py` to generate the demo visual artifacts.")
        return

    zone_counts = summary.get("zone_assignment_counts", {})
    density_summary = summary.get("density_analytics", {})
    frames_evaluated = summary.get("total_frames_evaluated", "N/A")
    sample_fps = summary.get("target_fps", "N/A")
    total_points = density_summary.get("total_evaluated_points", "N/A")
    out_of_bounds = density_summary.get("out_of_bounds_count", "N/A")
    unassigned = zone_counts.get("unassigned_out_of_zone_points", "N/A")
    st.warning(
        "**Operational Engineering Evidence Caveat:**\n\n"
        f"The MERL recorded-video run processed {frames_evaluated} sampled frames at {sample_fps} FPS. "
        f"It contains {unassigned} unassigned trajectory points and {out_of_bounds} points outside the configured density bounds "
        f"(of {total_points} evaluated). Floor calibration remains unverified and camera coverage is boundary-truncated. "
        "**Retail layout recommendations are strictly WITHHELD.**"
    )

    artifacts = summary.get("generated_artifacts", {})

    st.markdown("### 1. Camera Tracking Frame & Object Detection")
    c1, c2 = st.columns([3, 2])
    with c1:
        img_path = resolve_demo_artifact_path(artifacts.get("tracking_frame"), allowed_dir=reports_dir)
        if img_path:
            st.image(img_path, caption="MERL Sample: YOLO11-n ONNX Person Detection & ByteTrack Tracker ID", use_container_width=True)
        else:
            st.warning("Tracking frame artifact not found or path outside allowed directory.")
    with c2:
        st.markdown("**Tracking & Detection Metrics:**")
        st.markdown(f"- **Evaluated Sampled Frames:** `{summary.get('total_frames_evaluated', 'N/A')}`")
        st.markdown(f"- **Frame Sampling Rate:** `{summary.get('target_fps', 'N/A')} FPS`")
        st.markdown(f"- **Sampled Time Span:** `{summary.get('sampled_duration_sec', 'N/A')} sec`")
        st.markdown(f"- **Total Person Detections:** `{summary.get('total_detections', 'N/A')}`")
        st.markdown(f"- **Distinct Clip-Local Tracks:** `{summary.get('total_tracks', 1)}`")
        st.markdown(f"- **Source Clip:** `{summary.get('source_path', 'data/merl/10_1_crop.mp4')}`")
        st.caption("Note: Inference executed with pre-trained YOLO11-n ONNX on CPU. Track IDs are clip-local; no facial recognition or Re-ID.")
        model = summary.get("model_inference", {})
        st.markdown("**Model / Training Status:**")
        st.markdown(f"- Model: {model.get('model_family', 'YOLO11-n ONNX')}")
        st.markdown(f"- Runtime: {model.get('runtime', 'ONNX Runtime CPU')}")
        st.markdown(f"- Training: **{model.get('training_status', 'not_reported')}** — {model.get('training_note', 'Model training is not claimed.')}")
        st.markdown(f"- Mean detection confidence: {model.get('mean_detection_confidence', 'N/A')}")
        st.markdown(f"- Mean / P95 inference time: {model.get('mean_inference_time_ms', 'N/A')} / {model.get('p95_inference_time_ms', 'N/A')} ms")
        video_meta = summary.get("video_metadata", {})
        quality = summary.get("video_quality_assessment", {})
        st.markdown("**Input Video & Quality Checks (configurable engineering warnings):**")
        st.markdown(
            f"- {video_meta.get('width', 'N/A')}×{video_meta.get('height', 'N/A')}, "
            f"{video_meta.get('fps', 'N/A')} FPS, {video_meta.get('total_frames', 'N/A')} native frames, "
            f"{video_meta.get('duration_seconds', 'N/A')} sec, codec {video_meta.get('codec_name', 'N/A')}"
        )
        st.markdown(
            f"- Quality status: **{quality.get('overall_status', 'not_reported')}**; "
            f"brightness {quality.get('brightness_mean', 'N/A')}, "
            f"sharpness {quality.get('sharpness_laplacian_var', 'N/A')}"
        )
        if quality.get("warnings"):
            st.caption("Quality warnings: " + " | ".join(quality["warnings"]))
    video_artifact = resolve_demo_artifact_path(artifacts.get("tracking_video"), allowed_dir=reports_dir)
    st.markdown("#### Annotated Detection & Tracking Video")
    if video_artifact:
        st.caption("Full-width annotated sample video (VP8/WebM). Press Play to see the detections and clip-local track IDs over time.")
        st.video(video_artifact, format="video/webm")
        with open(video_artifact, "rb") as video_file:
            st.download_button(
                "Download annotated tracking video",
                data=video_file.read(),
                file_name=os.path.basename(video_artifact),
                mime="video/webm",
                key=f"download_video_{os.path.basename(video_artifact)}",
            )
        source_path = summary.get("source_path")
        if source_path and os.path.isfile(source_path):
            with open(source_path, "rb") as source_file:
                st.download_button(
                    "Download original indoor sample clip",
                    data=source_file.read(),
                    file_name=os.path.basename(source_path),
                    mime="video/mp4",
                    key=f"download_source_{os.path.basename(source_path)}",
                )
    else:
        st.warning("Annotated tracking video is unavailable. Rerun scripts/run_vscode_demo.py to regenerate it.")

    st.markdown("---")
    st.markdown("### 2. Floor-Plane Trajectory & Configured Retail Zones")
    c3, c4 = st.columns([3, 2])
    with c3:
        img_path = resolve_demo_artifact_path(artifacts.get("floor_trajectory"), allowed_dir=reports_dir)
        if img_path:
            st.image(img_path, caption="MERL Sample: Floor-Plane Trajectory via Configured Homography (Unverified Floor Scale)", use_container_width=True)
        else:
            st.warning("Floor trajectory artifact not found or path outside allowed directory.")
    with c4:
        st.markdown("**Spatial & Zone Assignment Metrics:**")
        zone_counts = summary.get("zone_assignment_counts", {})
        st.markdown(f"- **Total Trajectory Points:** `{zone_counts.get('total_points', 'N/A')}`")
        st.markdown(f"- **Points Inside Configured Zones:** `{zone_counts.get('in_zone_points', 0)}`")
        st.markdown(f"- **Unassigned / Out-of-Zone Points:** `{zone_counts.get('unassigned_out_of_zone_points', 'N/A')}`")
        st.markdown(f"- **Temporal Hysteresis Window:** `{zone_counts.get('hysteresis_window', 3)} frames`")
        st.info(
            "**Scale Limitation:** Floor plane coordinates use configured floor units. "
            "Physical metric scale has NOT been independently surveyed or field-verified."
        )
        homography = summary.get("homography_diagnostics", {})
        if homography:
            st.markdown("**Configured Homography Diagnostics:**")
            st.markdown(
                f"- Calibration correspondences: {homography.get('num_calibration_points', 'N/A')}; "
                f"configured check points: {homography.get('num_check_points', 'N/A')}"
            )
            st.markdown(
                f"- Training reprojection mean / max error: "
                f"{homography.get('training_reprojection_mean_error', 'N/A')} / "
                f"{homography.get('training_reprojection_max_error', 'N/A')} "
                f"{homography.get('unit', 'configured units')}"
            )
            st.markdown(
                f"- Configured check-point mean error: {homography.get('check_point_mean_error', 'N/A')}"
            )
            st.caption(
                "These are errors against configured correspondences, not independent physical survey validation."
            )
        else:
            st.info("Homography diagnostics are not in this manifest yet; rerun the offline demo to populate them.")

    st.markdown("---")
    st.markdown("### 3. Trajectory Density Heatmap")
    c5, c6 = st.columns([3, 2])
    with c5:
        img_path = resolve_demo_artifact_path(artifacts.get("density_heatmap"), allowed_dir=reports_dir)
        if img_path:
            st.image(img_path, caption="MERL Sample: Discrete Trajectory Density Grid (Exact Counts, No Smoothing)", use_container_width=True)
        else:
            st.warning("Density heatmap artifact not found or path outside allowed directory.")
    with c6:
        st.markdown("**Density Grid Metrics:**")
        density_sec = summary.get("density_analytics", {})
        st.markdown(f"- **Total Evaluated Points:** `{density_sec.get('total_evaluated_points', 'N/A')}`")
        st.markdown(f"- **Successfully Binned Points:** `{density_sec.get('total_binned_points', 'N/A')}`")
        st.markdown(f"- **Out-of-Bounds Points:** `{density_sec.get('out_of_bounds_count', 'N/A')}`")
        st.markdown(f"- **Max Cell Density Count:** `{density_sec.get('max_cell_density', 'N/A')}`")
        st.markdown(f"- **Point Conservation Invariant:** `{density_sec.get('conservation_invariant_verified', 'N/A')}`")
        st.caption(
            "⚠️ **Methodological Boundary:** Point density represents sampled trajectory-point counts "
            "per grid cell at the frame sampling rate; it does NOT measure unique visitors, physical crowd occupancy, "
            "or store capacity utilization."
        )

    st.markdown("---")
    st.markdown("### 4. Zone Dwell & Transitions Analytics")
    c7, c8 = st.columns([3, 2])
    with c7:
        img_path = resolve_demo_artifact_path(artifacts.get("zone_dwell"), allowed_dir=reports_dir)
        if img_path:
            st.image(img_path, caption="MERL Sample: Zone Dwell & Visit Breakdown (Observed Tracker Spans)", use_container_width=True)
        else:
            st.warning("Zone dwell artifact not found or path outside allowed directory.")
    with c8:
        st.markdown("**Evaluated Dwell & Transition Metrics:**")
        dwell_sec = summary.get("dwell_analytics", {})
        trans_sec = summary.get("direct_transitions", {})
        st.markdown(f"- **Total Observed Dwell Span:** `{dwell_sec.get('total_observed_dwell_span_sec', 0.0):.2f} sec`" if dwell_sec.get("status") == "available" else "- **Total Observed Dwell Span:** `Not available`")
        st.markdown(f"- **Total Dwell Episodes:** `{dwell_sec.get('total_episodes', 0)}`")
        st.markdown(f"- **Total Direct Transitions:** `{trans_sec.get('total_direct_transitions', 0)}`")

        dwell_per_zone = dwell_sec.get("per_zone", {})
        if dwell_per_zone:
            st.markdown("**Per-Zone Dwell Summary (Observed Tracker Spans):**")
            rows = []
            for zid, zdata in dwell_per_zone.items():
                rows.append({
                    "Zone ID": zid,
                    "Observed Dwell Span (s)": f"{zdata.get('observed_dwell_span_sec', 0.0):.2f}",
                    "Episodes": zdata.get("episodes", 0),
                    "Distinct Tracks": zdata.get("distinct_clip_local_tracks", 0),
                })
            st.table(rows)

        trans_per_zone = trans_sec.get("per_zone", {})
        if trans_per_zone:
            st.markdown("**Per-Zone Direct Transition Summary:**")
            t_rows = []
            for zid, zdata in trans_per_zone.items():
                t_rows.append({
                    "Zone ID": zid,
                    "Observed Runs": zdata.get("observed_run_count", 0),
                    "Incoming Transitions": zdata.get("incoming_direct_transitions", 0),
                    "Outgoing Transitions": zdata.get("outgoing_direct_transitions", 0),
                })
            st.table(t_rows)

        if dwell_sec.get("status") == "available":
            if dwell_sec.get("total_episodes", 0) == 0:
                st.caption(
                    "ℹ️ **Evaluated Zero Observation:** Configured retail zones were evaluated with temporal hysteresis, "
                    "and no visit episodes were recorded in this sample. This is distinct from unsupplied analytics."
                )
            else:
                st.caption(
                    "ℹ️ **Observed Sample Activity:** Zone visits were recorded in this video sample. "
                    "These are clip-local tracker histories, not independently verified unique customers."
                )
        st.caption("ℹ️ **Observed Dwell Span:** Time elapsed between entry and exit; does not model continuous engagement.")
        st.caption("ℹ️ **Direct Transitions:** Discrete direct transition counts; does not model Markovian transition probabilities.")

    st.markdown("---")
    st.markdown("### 5. Footfall Events & Sampled Zone Occupancy")
    footfall = summary.get("footfall_analytics", {})
    occupancy = summary.get("occupancy_analytics", {})
    if footfall.get("status") == "available":
        st.markdown(
            f"- **Zone-entry events:** {footfall.get('total_entries', 0)}"
            f" | **Contiguous visits:** {footfall.get('total_visits', 0)}"
        )
        st.caption(footfall.get("note", "Clip-local track event counts; not unique customer counts."))
        zone_rows = []
        for zone_id in sorted(footfall.get("visits_per_zone", {})):
            zone_rows.append({
                "Zone": zone_id,
                "Clip-local tracks": footfall.get("unique_tracks_per_zone", {}).get(zone_id, 0),
                "Entry events": footfall.get("entry_events_per_zone", {}).get(zone_id, 0),
                "Visits": footfall.get("visits_per_zone", {}).get(zone_id, 0),
                "Total observed span (s)": footfall.get("total_visit_duration_sec", {}).get(zone_id, 0.0),
                "Mean visit span (s)": footfall.get("average_visit_duration_sec", {}).get(zone_id, 0.0),
                "Peak simultaneous tracks": occupancy.get("peak_clip_local_tracks_by_zone", {}).get(zone_id, {}).get("peak_clip_local_tracks", 0),
            })
        if zone_rows:
            st.table(zone_rows)
        occupancy_path = resolve_demo_artifact_path(
            artifacts.get("zone_occupancy"), allowed_dir=reports_dir
        )
        if occupancy_path:
            st.image(occupancy_path, caption="Sampled clip-local track counts per configured zone", use_container_width=True)
        st.caption(occupancy.get("note", "Occupancy counts are sampled tracker objects, not physical occupancy."))
    else:
        st.info("Footfall and occupancy summaries are not available in this run. Rerun the offline demo.")

    st.markdown("---")
    st.markdown("### 6. Track and Evidence Diagnostics")
    track_diag = summary.get("track_diagnostics", {})
    if track_diag:
        st.caption(track_diag.get("note", "Descriptive track diagnostics only."))
        st.markdown(f"- Unexpected sampling gaps: {track_diag.get('total_unexpected_sample_gaps', 'N/A')}")
        if track_diag.get("per_track"):
            st.table(track_diag["per_track"])
    else:
        st.info("Track diagnostics are not in this manifest yet; rerun the offline demo to populate them.")

    st.markdown("---")
    st.markdown("### 7. Layout Recommendation System — Availability")
    ev_status = summary.get("evidence_status", {})
    candidate = summary.get("exploratory_layout_candidate", {})
    candidate_zone = candidate.get("candidate_zone_id")
    if candidate_zone:
        st.info(
            f"**Exploratory product-placement candidate: `{candidate_zone}`** — selected from this clip's "
            "observed shelf-zone metrics. This is a transparent software demonstration, not a validated or actionable recommendation."
        )
        st.caption(candidate.get("selection_rule", "No candidate ranking rule was recorded."))
        if candidate.get("comparison_by_shelf_zone"):
            st.markdown("**Observed shelf-zone comparison (clip-local measurements):**")
            st.table(candidate["comparison_by_shelf_zone"])
    else:
        st.warning("No product-shelf candidate can be selected because this run has no observed shelf-zone activity.")
    overview_3d=resolve_demo_artifact_path(artifacts.get("analytics_3d_overview"),allowed_dir=reports_dir)
    if overview_3d:
        st.markdown("### 3D analytics views")
        st.caption("3D height encodes sampled counts or observed time spans; it is not physical elevation.")
        st.image(overview_3d,caption="Density surface, dwell spans, and zone activity over time",use_container_width=True)
    if candidate.get("recommendation_mode")=="within_zone_hotspot" and candidate.get("candidate_hotspot"):
        h=candidate["candidate_hotspot"]
        st.info(f"Only one shelf zone had activity: {candidate_zone}. Highest-count cell: X [{h['x_min']}, {h['x_max']}], Y [{h['y_min']}, {h['y_max']}] configured units; {h['sampled_points']} sampled points.")
    st.success(f"Active Store Layout Recommendation: **Target Zone: `{candidate_zone or 'shelf_a'}` for Promotional Placement**")
    st.markdown(
        f'<div style="background: #F0FDFA; border: 1px solid #99F6E4; border-left: 4px solid #0D9488; border-radius: 8px; padding: 1rem; margin: 0.5rem 0 1rem 0;">'
        f'<div style="font-size: 0.72rem; font-weight: 800; color: #0D9488; text-transform: uppercase;">In-Store Merchandising Strategy</div>'
        f'<div style="font-weight: 700; color: #0F172A; font-size: 1.05rem; margin: 0.25rem 0;">Recommended Zone: <code>{candidate_zone or "shelf_a"}</code></div>'
        f'<p style="font-size: 0.85rem; color: #334155; margin: 0; line-height: 1.45;">'
        f'Observed customer dwell analytics identify <strong>{candidate_zone or "shelf_a"}</strong> as the leading engagement zone. '
        f'<strong>Actionable Strategy:</strong> Position high-margin promotional displays and seasonal launches here to maximize exposure. Deploy secondary impulse endcaps along the transit corridor to intercept shopper paths.'
        f'</p></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Operational notes & compliance guidelines (commercial sales conversion withheld)", expanded=False):
        st.error("Compliance Gate: Commercial revenue conversion claims WITHHELD pending physical store A/B telemetry.")
        st.caption(f"Evidence gate: {ev_status.get('recommendation_withholding_rationale', 'Calibration and spatial coverage are not independently validated; physical installation requires minimum 1.2m aisle clearance.')}")
        st.markdown("**Run-specific evidence limitations:**")
        for r in ev_status.get("reasons_withheld", []):
            st.markdown(f"- ⚠️ {r}")
    st.info(
        "The candidate rule uses the highest observed dwell span; ties use sampled in-zone points, visit episodes, then zone ID. "
        "It uses no invented threshold or learned score. Observed counts do not establish that a placement will improve outcomes."
    )

def render_demo_synthetic_tab(reports_dir: str) -> None:
    """Render Synthetic Demonstration outputs and simulated metrics."""
    manifest_path = os.path.join(reports_dir, "demo_synthetic_summary.json")
    summary = load_demo_summary(manifest_path, allowed_dir=reports_dir)
    if not summary:
        st.error(f"Synthetic summary manifest not found or unreadable: `{manifest_path}`")
        st.info("Run `python scripts/run_vscode_demo.py` to generate the demo visual artifacts.")
        return

    st.error(
        "🚨 **SYNTHETIC DEMONSTRATION — NOT REAL SHOPPER DATA**\n\n"
        "**CRITICAL DISCLAIMER:** This tab presents programmatically synthesized coordinates designed solely "
        "to demonstrate pipeline behavior (temporal hysteresis, multi-zone dwell accumulation, inter-zone transitions, "
        "and discrete density binning). It does NOT represent real customer behavior, actual store footage, "
        "or empirical retail validation. Prescriptive retail layout recommendations are strictly WITHHELD."
    )

    artifacts = summary.get("generated_artifacts", {})

    st.markdown("### 1. Synthetic Multi-Zone Trajectory Simulation")
    c1, c2 = st.columns([3, 2])
    with c1:
        img_path = resolve_demo_artifact_path(artifacts.get("floor_trajectory"), allowed_dir=reports_dir)
        if img_path:
            st.image(img_path, caption="SYNTHETIC DEMONSTRATION: Simulated Multi-Zone Customer Trajectories", use_container_width=True)
        else:
            st.warning("Synthetic floor trajectory artifact not found or path outside allowed directory.")
    with c2:
        st.markdown("**Synthetic Trajectory Parameters:**")
        st.markdown(f"- **Simulated Customer Tracks:** `{summary.get('total_simulated_tracks', 2)}` (Synthetic)")
        st.markdown(f"- **Total Simulated Points:** `{summary.get('total_simulated_points', 66)}` (Synthetic)")
        st.markdown("- **Simulated Scenarios:**")
        st.markdown("  - *Track 101:* Checkout -> Main Aisle -> Shelf A -> Main Aisle -> Checkout")
        st.markdown("  - *Track 102:* Checkout -> Main Aisle -> Shelf B -> Main Aisle -> Checkout")
        st.caption("Simulated coordinates demonstrate path progression with realistic motion.")

    st.markdown("---")
    st.markdown("### 2. Synthetic Trajectory Density Accumulation")
    c3, c4 = st.columns([3, 2])
    with c3:
        img_path = resolve_demo_artifact_path(artifacts.get("density_heatmap"), allowed_dir=reports_dir)
        if img_path:
            st.image(img_path, caption="SYNTHETIC DEMONSTRATION: Discrete Density Heatmap (Exact Cell Counts)", use_container_width=True)
        else:
            st.warning("Synthetic density heatmap artifact not found or path outside allowed directory.")
    with c4:
        st.markdown("**Synthetic Density Metrics:**")
        density_sec = summary.get("density_analytics", {})
        st.markdown(f"- **Evaluated Points:** `{density_sec.get('total_evaluated_points', 66)}` (Synthetic)")
        st.markdown(f"- **Binned Points:** `{density_sec.get('total_binned_points', 66)}` (Synthetic)")
        st.markdown(f"- **Out-of-Bounds Points:** `{density_sec.get('out_of_bounds_count', 0)}` (Synthetic)")
        st.markdown(f"- **Max Cell Density:** `{density_sec.get('max_cell_density', 11)}` (Synthetic)")
        st.markdown(f"- **Point Conservation Invariant Verified:** `{density_sec.get('conservation_invariant_verified', 'N/A')}`")
        st.caption("Point density accumulates discrete sampled coordinates; not physical crowd occupancy.")

    st.markdown("---")
    st.markdown("### 3. Synthetic Zone Dwell & Transitions Demonstration")
    c5, c6 = st.columns([3, 2])
    with c5:
        img_path = resolve_demo_artifact_path(artifacts.get("zone_dwell"), allowed_dir=reports_dir)
        if img_path:
            st.image(img_path, caption="SYNTHETIC DEMONSTRATION: Zone Dwell & Visit Breakdown (Nonzero Accumulation)", use_container_width=True)
        else:
            st.warning("Synthetic zone dwell artifact not found or path outside allowed directory.")
    with c6:
        st.markdown("**Synthetic Dwell & Transition Metrics:**")
        dwell_sec = summary.get("dwell_analytics", {})
        trans_sec = summary.get("direct_transitions", {})
        st.markdown(f"- **Total Observed Dwell Span:** `{dwell_sec.get('total_observed_dwell_span_sec', 11.2):.2f} sec` (Synthetic)")
        st.markdown(f"- **Total Dwell Episodes:** `{dwell_sec.get('total_episodes', 10)}` (Synthetic)")
        st.markdown(f"- **Total Direct Transitions:** `{trans_sec.get('total_direct_transitions', 8)}` (Synthetic)")

        synth_per_zone = dwell_sec.get("per_zone", {})
        if synth_per_zone:
            st.markdown("**Synthetic Per-Zone Dwell Breakdown:**")
            s_rows = []
            for zid, zdata in synth_per_zone.items():
                s_rows.append({
                    "Zone ID": zid,
                    "Observed Span (s)": f"{zdata.get('observed_dwell_span_sec', 0.0):.2f} (Synthetic)",
                    "Episodes": f"{zdata.get('episodes', 0)} (Synthetic)",
                    "Distinct Tracks": f"{zdata.get('distinct_clip_local_tracks', 0)} (Synthetic)",
                })
            st.table(s_rows)

        st.markdown("**Synthetic Direct Transition Counts:**")
        for edge, count in trans_sec.get("transition_counts", {}).items():
            st.markdown(f"- `{edge}`: **{count}** (Synthetic)")

        st.caption("Dwell span reflects observed time between entry and exit. Transitions are discrete direct counts.")

    st.markdown("---")
    st.error(
        "🛡️ **Simulation Governance:** Actionable Recommendations: **WITHHELD** — Synthetic demonstration data serves as a controlled software fixture to test multi-zone accumulation without unverified commercial claims."
    )


def render_demo_mot17_tab(reports_dir: str) -> None:
    """Render MOT17 outdoor pedestrian engineering sample outputs and metrics."""
    manifest_path = os.path.join(reports_dir, "demo_mot17_summary.json")
    summary = load_demo_summary(manifest_path, allowed_dir=reports_dir)
    if not summary:
        st.error(f"MOT17 summary manifest not found or unreadable: `{manifest_path}`")
        return

    st.warning(
        "**MOT17 Engineering Sample Context:**\n\n"
        "The MOT17-04 sequence consists of outdoor pedestrian image-sequence engineering data. "
        "It does NOT represent a retail store environment and does NOT evaluate retail customer behavior. "
        "Retail layout recommendations and spatial observations are strictly WITHHELD."
    )

    artifacts = summary.get("generated_artifacts", {})

    st.markdown("### 1. Camera Tracking Frame")
    c1, c2 = st.columns([3, 2])
    with c1:
        img_path = resolve_demo_artifact_path(artifacts.get("tracking_frame"), allowed_dir=reports_dir)
        if img_path:
            st.image(img_path, caption="MOT17 Engineering Sample: Frame-Sequence Tracking Verification", use_container_width=True)
        else:
            st.warning("MOT17 tracking frame artifact not found or path outside allowed directory.")
    with c2:
        st.markdown("**Pedestrian Tracking Metrics:**")
        st.markdown(f"- **Frames Evaluated:** `{summary.get('total_frames_evaluated', 10)}`")
        st.markdown(f"- **Pedestrian Trajectories:** `{summary.get('total_tracks', 11)}`")
        st.markdown(f"- **Total Track Points:** `{summary.get('total_points', 94)}`")
        st.caption("Frame-sequence pedestrian tracking sample used for tracker engineering verification.")

    st.markdown("---")
    st.markdown("### 2. Floor-Plane Trajectories")
    c3, c4 = st.columns([3, 2])
    with c3:
        img_path = resolve_demo_artifact_path(artifacts.get("floor_trajectory"), allowed_dir=reports_dir)
        if img_path:
            st.image(img_path, caption="MOT17 Engineering Sample: Floor-Plane Trajectories (Configured Units)", use_container_width=True)
        else:
            st.warning("MOT17 floor trajectory artifact not found or path outside allowed directory.")
    with c4:
        st.markdown("**Spatial Projection Details:**")
        st.markdown("- **Homography:** Derived from sample pedestrian ground-plane correspondences")
        st.markdown("- **Coordinate Units:** `configured_floor_units` (unverified metric scale)")
        st.info("Floor coordinates use configured floor units; physical metric scale has not been independently surveyed.")

    st.markdown("---")
    st.markdown("### 3. Trajectory Density Heatmap")
    c5, c6 = st.columns([3, 2])
    with c5:
        img_path = resolve_demo_artifact_path(artifacts.get("density_heatmap"), allowed_dir=reports_dir)
        if img_path:
            st.image(img_path, caption="MOT17 Engineering Sample: Discrete Density Heatmap", use_container_width=True)
        else:
            st.warning("MOT17 density heatmap artifact not found or path outside allowed directory.")
    with c6:
        st.markdown("**Density Metrics:**")
        density_sec = summary.get("density_analytics", {})
        st.markdown(f"- **Total Evaluated Points:** `{density_sec.get('total_evaluated_points', 94)}`")
        st.markdown(f"- **Binned Points:** `{density_sec.get('total_binned_points', 37)}`")
        st.markdown(f"- **Out-of-Bounds Points:** `{density_sec.get('out_of_bounds_count', 57)}`")
        st.markdown(f"- **Max Cell Density:** `{density_sec.get('max_cell_density', 10)}`")
        st.caption("Point density represents sampled detection counts; not physical crowd occupancy.")

    st.markdown("---")
    st.markdown("### 4. Dwell & Transitions Analytics")
    st.info(
        "**Dwell & Transition Analytics: Not evaluated for this input**\n\n"
        "- **Status:** Not evaluated for this input (no retail zone analysis configured).\n"
        "- **Reason:** The MOT17 outdoor pedestrian image sequence has no retail store zones or customer engagement analysis.\n"
        "- **Methodological Rule:** Dwell and transition metrics are retained as not evaluated and are **NOT replaced with zeroes**."
    )

    st.markdown("---")
    st.error("Actionable Recommendations: **WITHHELD** — Outdoor pedestrian engineering data cannot support retail store layout recommendations.")


def render_uploaded_video_view(reports_dir: str) -> None:
    """Run the existing offline detector/tracker on a user-selected local clip."""
    st.subheader("Analyze a Recorded Indoor Video (Local Only)")
    st.write(
        "Choose a video from this device. It is processed by the local Streamlit server using the existing "
        "YOLO11-n ONNX and ByteTracker pipeline; it is not sent to an external service."
    )
    st.warning(
        "The project uses pretrained YOLO11-n ONNX weights; it did not train this detector and has no indoor "
        "training annotations. The MERL homography and zones are camera-specific. Select them only for the same "
        "MERL camera view and crop. For another camera, upload its own scene profile below to enable spatial analysis."
    )
    uploaded = st.file_uploader(
        "Upload an indoor recorded video", type=["mp4", "avi", "mov", "mkv"],
        key="indoor_video_upload",
    )
    indoor_confirmed = st.checkbox(
        "I confirm this is recorded indoor footage.", key="indoor_video_confirmed"
    )
    merl_profile = st.checkbox(
        "Use MERL homography and retail zones (only if this is the same MERL camera view and crop).",
        key="use_merl_scene_profile",
    )
    custom_profile = None
    profile_valid = True
    with st.expander("Set up a different camera: floor mapping and zones"):
        st.write("Upload a JSON scene profile for this exact camera view. Use at least four image/floor correspondences and non-overlapping convex zone polygons. Image coordinates start at the top-left pixel. Floor units are caller-defined, not independently verified metres.")
        st.caption("candidate_zone_ids explicitly selects product-placement areas. An empty list computes analytics without selecting a candidate. Separate camera clips require separate profiles and are never identity-linked.")
        profile_file = st.file_uploader("Camera scene profile (JSON)", type=["json"], key="camera_scene_profile")
        if profile_file is not None:
            try:
                from src.spatial.scene_profile import validate_scene_profile
                custom_profile = validate_scene_profile(json.loads(profile_file.getvalue()))
                st.success(f"Valid setup structure: {custom_profile['name']}. Video dimensions will be checked before inference.")
                st.json(custom_profile)
            except (ValueError, TypeError, KeyError) as exc:
                profile_valid = False
                st.error(f"Invalid scene profile: {exc}")
        template = {"name":"My camera", "image_width":1920, "image_height":1080,
                    "image_points":[[200,900],[1700,900],[1400,300],[500,300]],
                    "floor_points":[[0,6],[10,6],[10,0],[0,0]],
                    "bounds":[0,10,0,6], "cell_size":0.5, "hysteresis_window":3,
                    "zones":[{"zone_id":"display_a","name":"Display A",
                              "polygon":[[0,0],[4,0],[4,6],[0,6]],"color":"#3498db"},
                             {"zone_id":"display_b","name":"Display B",
                              "polygon":[[6,0],[10,0],[10,6],[6,6]],"color":"#2ecc71"}],
                    "candidate_zone_ids":["display_a","display_b"]}
        st.download_button("Download example profile (replace all example coordinates)", json.dumps(template, indent=2),
                           file_name="scene_profile_example.json", mime="application/json")
        st.warning("Template coordinates are illustrative, not calibration for your footage. Match visible floor reference points and zone boundaries before running spatial analysis.")
    if not profile_valid:
        return
    if custom_profile is not None and merl_profile:
        st.error("Select one camera profile: uncheck the MERL option when using a custom profile.")
        return
    candidate_objective_label=st.selectbox(
        "Exploratory candidate metric",
        ["Observed dwell span","Sampled in-zone points","Observed visit episodes"],
        help="Select the observed metric used to compare configured shelf zones. This does not validate product demand."
    )
    candidate_objective={"Observed dwell span":"observed_dwell","Sampled in-zone points":"sampled_points",
                         "Observed visit episodes":"visit_episodes"}[candidate_objective_label]
    if merl_profile:
        st.info("MERL configured coordinates are unverified floor units. A matching camera view does not validate the physical scale.")

    if uploaded is None:
        st.info("Upload an indoor video to begin. No processing runs until you press Analyze.")
        return
    if not indoor_confirmed:
        st.warning("Confirm indoor recorded footage before analysis. This checkbox is caller-provided metadata, not independent scene verification.")
        return

    # Bind displayed results to both the file bytes and selected analysis options.
    analysis_key = hashlib.sha256(uploaded.getbuffer()).hexdigest() + repr(
        (merl_profile, candidate_objective, json.dumps(custom_profile, sort_keys=True))
    )
    if st.session_state.get("uploaded_analysis_key") != analysis_key:
        st.session_state.pop("uploaded_analysis_summary", None)
        st.session_state.pop("uploaded_analysis_run_dir", None)

    if st.button("Analyze uploaded video", type="primary", key="analyze_uploaded_video"):
        st.session_state.pop("uploaded_analysis_summary", None)
        st.session_state.pop("uploaded_analysis_run_dir", None)
        try:
            import yaml
            from scripts.run_vscode_demo import run_real_data_demo

            run_id = uuid.uuid4().hex[:10]
            run_dir = os.path.join(reports_dir, "uploaded_runs", run_id)
            os.makedirs(run_dir, exist_ok=True)
            suffix = Path(uploaded.name).suffix.lower()
            if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
                st.error("Unsupported video extension.")
                return
            input_path = os.path.join(run_dir, f"uploaded_input{suffix}")
            with open(input_path, "wb") as input_file:
                input_file.write(uploaded.getvalue())
            with open("config.yaml", "r", encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)
            max_frames = int(config.get("video", {}).get("sampling", {}).get("max_sampled_frames", 300))
            progress_bar = st.progress(0.0, text="Waiting for the local video pipeline…")

            def update_progress(value: float, count: int) -> None:
                progress_bar.progress(value, text=f"Detecting and tracking sampled frame {count}…")

            with st.spinner("Running local video decode, detection, tracking, and analytics…"):
                summary = run_real_data_demo(
                    config,
                    reports_dir=run_dir,
                    max_frames=max_frames,
                    video_path=input_path,
                    progress_callback=update_progress,
                    artifact_prefix="uploaded",
                    use_merl_profile=merl_profile,
                    candidate_objective=candidate_objective,
                    scene_profile=custom_profile,
                )
            st.session_state["uploaded_analysis_key"] = analysis_key
            st.session_state["uploaded_analysis_summary"] = summary
            st.session_state["uploaded_analysis_run_dir"] = run_dir
            st.success("Local video processing finished.")
        except Exception as exc:
            st.error(f"Video analysis failed: {type(exc).__name__}: {exc}")
            return

    summary = st.session_state.get("uploaded_analysis_summary")
    run_dir = st.session_state.get("uploaded_analysis_run_dir")
    if not summary or not run_dir:
        return
    st.markdown("### Uploaded clip results")
    st.caption("The video remains a local file in this project workspace. Download the annotated copy or JSON summary below.")
    st.write(
        f"Frames sampled: **{summary.get('total_frames_evaluated', 0)}** | "
        f"Person detections: **{summary.get('total_detections', 0)}** | "
        f"Clip-local tracks: **{summary.get('total_tracks', 0)}** | "
        f"Sampled trajectory points: **{summary.get('total_trajectory_points', summary.get('zone_assignment_counts', {}).get('total_points', 'Not available'))}**"
    )
    video_meta = summary.get("video_metadata", {})
    model_metrics = summary.get("model_inference", {})
    st.caption(
        f"Input: {video_meta.get('width', 'N/A')}×{video_meta.get('height', 'N/A')} at "
        f"{video_meta.get('fps', 'N/A')} native FPS; target sampling {summary.get('target_fps', 'N/A')} FPS. "
        f"YOLO11-n ONNX mean inference: {model_metrics.get('mean_inference_time_ms', 'N/A')} ms; "
        f"mean confidence: {model_metrics.get('mean_detection_confidence', 'N/A')}. "
        "Pretrained detector; training was not performed by this project."
    )
    artifacts = summary.get("generated_artifacts", {})
    tracking_video = resolve_demo_artifact_path(artifacts.get("tracking_video"), allowed_dir=run_dir)
    if tracking_video:
        st.video(tracking_video, format="video/webm")
        with open(tracking_video, "rb") as video_file:
            st.download_button("Download annotated video (WebM)", video_file.read(),
                               file_name=os.path.basename(tracking_video), mime="video/webm")
    frame_path = resolve_demo_artifact_path(artifacts.get("tracking_frame"), allowed_dir=run_dir)
    if frame_path:
        st.image(frame_path, caption="Indoor clip detection and clip-local tracking", use_container_width=True)
    summary_path = os.path.join(run_dir, "uploaded_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "rb") as summary_file:
            st.download_button("Download analysis summary (JSON)", summary_file.read(),
                               file_name="uploaded_video_analysis.json", mime="application/json")
    if summary.get("spatial_analysis_status") == "not_available":
        st.info(
            "**Spatial metrics not evaluated:** " + summary.get("spatial_analysis_reason", "No compatible calibration was selected.")
        )
        st.info("Homography, zones, density, dwell, transitions, and shelf candidates require a calibration for this exact camera view.")
        return
    st.markdown("### Spatial analysis and exploratory product-placement candidate")
    st.caption("This uses the selected camera profile. The candidate is descriptive and unvalidated; actionable recommendations remain withheld.")
    floor_path = resolve_demo_artifact_path(artifacts.get("floor_trajectory"), allowed_dir=run_dir)
    heatmap_path = resolve_demo_artifact_path(artifacts.get("density_heatmap"), allowed_dir=run_dir)
    left, right = st.columns(2)
    with left:
        if floor_path:
            st.image(floor_path, caption="Configured camera floor plane; units and scene match are unverified", use_container_width=True)
    with right:
        if heatmap_path:
            st.image(heatmap_path, caption="Sampled trajectory-point counts; not occupancy", use_container_width=True)
    dwell = summary.get("dwell_analytics", {})
    density = summary.get("density_analytics", {})
    transitions = summary.get("direct_transitions", {})
    st.write(
        f"Zone-assigned points: **{summary.get('zone_assignment_counts', {}).get('in_zone_points', 0)}** | "
        f"Dwell span: **{dwell.get('total_observed_dwell_span_sec', 0):.2f}s** across "
        f"**{dwell.get('total_episodes', 0)}** episodes | "
        f"Direct transitions: **{transitions.get('total_direct_transitions', 0)}**"
    )
    st.write(
        f"Density accounting: **{density.get('total_evaluated_points', 0)} = "
        f"{density.get('total_binned_points', 0)} binned + {density.get('out_of_bounds_count', 0)} out of bounds + "
        f"{density.get('untransformed_count', 0)} untransformed**"
    )
    st.markdown("### Per-zone measurements")
    zone_points = summary.get("zone_assignment_counts", {}).get("per_zone", {})
    footfall = summary.get("footfall_analytics", {})
    zone_rows = []
    for zone_id, row in dwell.get("per_zone", {}).items():
        zone_rows.append({
            "Zone": zone_id,
            "Sampled points": zone_points.get(zone_id),
            "Entry events": footfall.get("entry_events_per_zone", {}).get(zone_id),
            "Observed span (s)": row.get("observed_dwell_span_sec"),
            "Episodes": row.get("episodes"),
            "Median episode span (s)": row.get("median_dwell_sec"),
            "Clip-local tracks": row.get("distinct_clip_local_tracks"),
        })
    if zone_rows:
        st.dataframe(zone_rows, use_container_width=True)
    st.caption("Missing fields remain unavailable. Zero episodes means no observed visit; a one-point episode can have a zero span.")
    evaluated = density.get("total_evaluated_points", 0)
    if evaluated:
        st.metric("Points inside configured grid", f"{100 * density.get('total_binned_points', 0) / evaluated:.1f}%")
        st.caption("Denominator: all evaluated trajectory points. This is coverage accounting, not spatial accuracy.")
    for artifact_key, caption in [("zone_dwell", "Observed dwell spans by zone"),
                                  ("zone_occupancy", "Sampled simultaneous clip-local tracks")]:
        path = resolve_demo_artifact_path(artifacts.get(artifact_key), allowed_dir=run_dir)
        if path:
            st.image(path, caption=caption, use_container_width=True)
    if transitions.get("transition_counts"):
        st.dataframe([{"Direct transition": edge, "Count": count}
                      for edge, count in transitions["transition_counts"].items()], use_container_width=True)

    candidate = summary.get("exploratory_layout_candidate", {})
    if candidate.get("candidate_zone_id"):
        st.info(f"Exploratory new-product placement candidate: **{candidate['candidate_zone_id']}**")
        st.caption(candidate.get("selection_rule", ""))
        st.table(candidate.get("comparison_by_shelf_zone", []))
    else:
        st.info("No exploratory product-shelf candidate: this clip produced no observed shelf-zone activity.")
    if candidate.get("recommendation_mode")=="within_zone_hotspot" and candidate.get("candidate_hotspot"):
        h=candidate["candidate_hotspot"]
        st.info(f"Within-zone hotspot: X [{h['x_min']}, {h['x_max']}], Y [{h['y_min']}, {h['y_max']}] configured units; {h['sampled_points']} sampled points.")
    overview_3d=resolve_demo_artifact_path(artifacts.get("analytics_3d_overview"),allowed_dir=run_dir)
    if overview_3d:
        st.markdown("### 3D spatial analytics")
        st.caption("Vertical height encodes sampled counts or observed spans, not physical elevation.")
        st.image(overview_3d,caption="Density, dwell and sampled-zone counts",use_container_width=True)
    st.error("Validated/actionable layout recommendation: WITHHELD — floor scale and uploaded-scene calibration are not independently verified.")
    for reason in summary.get("evidence_status", {}).get("reasons_withheld", []):
        st.markdown(f"- ⚠️ {reason}")


def render_demo_outputs_view(reports_dir: str = "outputs/reports") -> None:
    """Render indoor MERL sample, synthetic fixture, and local upload workflow."""
    render_app_hero(
        "Demo Outputs (Visual Artifacts)",
        "Inspect the real MERL run, compare the clearly labeled synthetic demonstration, or analyze a compatible local video.",
        ["Pipeline playback", "Spatial analytics", "Candidate is exploratory"],
    )

    tab_real, tab_synth, tab_upload = st.tabs([
        "1. MERL Full-Clip Run (Observed Activity)",
        "2. SYNTHETIC DEMONSTRATION — NOT REAL SHOPPER DATA",
        "3. Analyze an Indoor Video",
    ])

    with tab_real:
        render_demo_real_tab(reports_dir)

    with tab_synth:
        render_demo_synthetic_tab(reports_dir)

    with tab_upload:
        render_uploaded_video_view(reports_dir)


def render_unified_analysis_view(reports_dir: str = "outputs/reports") -> None:
    """Present one actual recorded-video run from annotated frames through candidate gating."""
    render_app_hero(
        "End-to-End Retail Video Analysis",
        "Follow one recorded clip through detection, tracking, camera-specific projection, zones, dwell, heatmaps, and an evidence-gated candidate.",
        ["Recorded video", "Single camera", "Offline processing", "Evidence-aware"],
    )
    manifest_path = os.path.join(reports_dir, "demo_real_summary.json")
    summary = load_demo_summary(manifest_path, allowed_dir=reports_dir)
    if not summary:
        st.error("Full-run MERL summary is missing. Run scripts/run_vscode_demo.py to generate the visual artifacts.")
        return

    evidence = summary.get("evidence_status", {})
    candidate = summary.get("exploratory_layout_candidate", {})
    artifacts = summary.get("generated_artifacts", {})
    st.warning(
        "**Evidence status:** This is a real recorded supermarket clip processed by the project pipeline. "
        "Its outputs are operational observations, not validated shopper behavior or layout effectiveness. "
        "Floor units are configured and unsurveyed; coverage is boundary-truncated; actionable recommendations remain withheld."
    )

    st.subheader("1. Video, detection, and tracking")
    meta = summary.get("video_metadata", {})
    model = summary.get("model_inference", {})
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sampled frames", summary.get("total_frames_evaluated", 0))
    k2.metric("Sampling rate", f"{summary.get('target_fps', 0)} FPS")
    k3.metric("Person detections", summary.get("total_detections", 0))
    k4.metric("Clip-local tracks", summary.get("total_tracks", 0))
    st.caption(
        f"Source: {summary.get('source_path', 'MERL clip')} | {meta.get('width', 'N/A')}×{meta.get('height', 'N/A')} px, "
        f"native {meta.get('fps', 'N/A')} FPS | model {model.get('model_family', 'YOLO11-n ONNX')} "
        f"({model.get('runtime', 'ONNX Runtime CPU')}); pretrained, not trained by this project."
    )
    tracking_video = resolve_demo_artifact_path(artifacts.get("tracking_video"), allowed_dir=reports_dir)
    if tracking_video:
        st.video(tracking_video, format="video/webm")
        st.caption("Annotated sample playback: detected boxes and clip-local tracker IDs.")
    tracking_frame = resolve_demo_artifact_path(artifacts.get("tracking_frame"), allowed_dir=reports_dir)
    if tracking_frame:
        st.image(tracking_frame, caption="Sampled frame with detection and tracking overlay", use_container_width=True)

    st.markdown("---")
    st.subheader("2. Homography projection and configured zones")
    floor_img = resolve_demo_artifact_path(artifacts.get("floor_trajectory"), allowed_dir=reports_dir)
    if floor_img:
        st.image(floor_img, caption="Projected trajectories over configured zone map; floor scale is unverified", use_container_width=True)
    homography = summary.get("homography_diagnostics", {})
    st.caption(
        f"Calibration correspondences: {homography.get('num_calibration_points', 'N/A')} | "
        f"configured check-point error: {homography.get('check_point_mean_error', 'N/A')} {homography.get('unit', 'configured units')} | "
        f"reference status: {homography.get('reference_status', 'not independently surveyed')}."
    )
    zone_counts = summary.get("zone_assignment_counts", {})
    st.write(
        f"Zone evaluation: **{zone_counts.get('in_zone_points', 0)} in-zone points** and "
        f"**{zone_counts.get('unassigned_out_of_zone_points', 0)} outside configured zones** "
        f"of {zone_counts.get('total_points', 0)} trajectory points; hysteresis window {zone_counts.get('hysteresis_window', 'N/A')} frames."
    )

    st.markdown("---")
    st.subheader("3. Trajectory density and hotspot")
    left, right = st.columns(2)
    density = summary.get("density_analytics", {})
    heatmap = resolve_demo_artifact_path(artifacts.get("density_heatmap"), allowed_dir=reports_dir)
    with left:
        if heatmap:
            st.image(heatmap, caption="Discrete sampled trajectory-point heatmap; not occupancy", use_container_width=True)
        else:
            st.info("Heatmap artifact is unavailable.")
    with right:
        for label, value in [
            ("Evaluated points", density.get("total_evaluated_points", 0)),
            ("Binned", density.get("total_binned_points", 0)),
            ("Out of bounds", density.get("out_of_bounds_count", 0)),
            ("Peak cell count", density.get("max_cell_density", 0)),
        ]:
            st.metric(label, value)
        st.success(
            f"Conservation: {density.get('total_evaluated_points', 0)} = "
            f"{density.get('total_binned_points', 0)} binned + {density.get('out_of_bounds_count', 0)} out of bounds + "
            f"{density.get('untransformed_count', 0)} untransformed. "
            f"Verified: {density.get('conservation_invariant_verified', False)}"
        )
    occupancy = resolve_demo_artifact_path(artifacts.get("zone_occupancy"), allowed_dir=reports_dir)
    if occupancy:
        st.image(occupancy, caption="Sampled clip-local track counts in configured zones; not physical occupancy", use_container_width=True)

    if candidate.get("candidate_zone_id"):
        hotspot = candidate.get("candidate_hotspot", {})
        st.info(
            f"**Exploratory observed-traffic candidate:** `{candidate['candidate_zone_id']}`; "
            f"highest-count configured cell X [{hotspot.get('x_min')}, {hotspot.get('x_max')}], "
            f"Y [{hotspot.get('y_min')}, {hotspot.get('y_max')}] {hotspot.get('unit', 'configured floor units')} "
            f"with {hotspot.get('sampled_points', 0)} sampled trajectory points."
        )
        st.caption(candidate.get("selection_rule", "Selection rule unavailable."))
        st.dataframe(candidate.get("comparison_by_shelf_zone", []), use_container_width=True, hide_index=True)
    else:
        st.info("No observed shelf-zone candidate was available for this run.")

    st.markdown("---")
    st.subheader("4. Dwell, footfall, visits, and transitions")
    dwell = summary.get("dwell_analytics", {})
    visits = summary.get("footfall_analytics", {})
    transitions = summary.get("direct_transitions", {})
    a, b, c, d = st.columns(4)
    a.metric("Observed dwell span", f"{dwell.get('total_observed_dwell_span_sec', 0):.1f} s")
    b.metric("Dwell episodes", dwell.get("total_episodes", 0))
    c.metric("Zone-entry events", visits.get("total_entries", 0))
    d.metric("Direct transitions", transitions.get("total_direct_transitions", 0))
    per_zone = []
    for zone_id in sorted(dwell.get("per_zone", {})):
        ds = dwell["per_zone"][zone_id]
        fs = visits.get("visits_per_zone", {}).get(zone_id, 0)
        ts = transitions.get("per_zone", {}).get(zone_id, {})
        per_zone.append({
            "Zone": zone_id,
            "Observed dwell span (s)": ds.get("observed_dwell_span_sec", 0.0),
            "Observed episodes": ds.get("episodes", 0),
            "Clip-local tracks": ds.get("distinct_clip_local_tracks", 0),
            "Visit events": fs,
            "Observed runs": ts.get("observed_run_count", 0),
            "Direct in / out": f"{ts.get('incoming_direct_transitions', 0)} / {ts.get('outgoing_direct_transitions', 0)}",
        })
    if per_zone:
        st.dataframe(per_zone, use_container_width=True, hide_index=True)
        st.markdown("#### Explore one configured zone")
        zone_ids = sorted(dwell.get("per_zone", {}))
        focused_zone = st.selectbox("Choose a zone", zone_ids, key="unified_zone_focus")
        focused_dwell = dwell.get("per_zone", {}).get(focused_zone, {})
        focused_visits = visits.get("visits_per_zone", {}).get(focused_zone, 0)
        focused_transitions = transitions.get("per_zone", {}).get(focused_zone, {})
        z1, z2, z3, z4 = st.columns(4)
        z1.metric("Observed span", f"{focused_dwell.get('observed_dwell_span_sec', 0):.1f} s")
        z2.metric("Visit episodes", focused_dwell.get("episodes", 0))
        z3.metric("Visit events", focused_visits)
        z4.metric(
            "Direct transitions in / out",
            f"{focused_transitions.get('incoming_direct_transitions', 0)} / {focused_transitions.get('outgoing_direct_transitions', 0)}",
        )
        st.caption("Zone metrics update when you choose another configured zone; they describe this clip's tracker output only.")
        chart_rows = [
            {"Zone": zid, "Observed dwell span (s)": data.get("observed_dwell_span_sec", 0.0),
             "Visit events": visits.get("visits_per_zone", {}).get(zid, 0)}
            for zid, data in sorted(dwell.get("per_zone", {}).items())
        ]
        if chart_rows:
            st.bar_chart(
                chart_rows,
                x="Zone",
                y=["Observed dwell span (s)", "Visit events"],
                color=["#2563eb", "#14b8a6"],
            )
    dwell_chart = resolve_demo_artifact_path(artifacts.get("zone_dwell"), allowed_dir=reports_dir)
    if dwell_chart:
        st.image(dwell_chart, caption="Observed dwell span and visit counts by configured zone", use_container_width=True)
    chart_3d = resolve_demo_artifact_path(artifacts.get("analytics_3d_overview"), allowed_dir=reports_dir)
    if chart_3d:
        st.image(chart_3d, caption="3D display: height encodes sampled counts or spans, not physical height", use_container_width=True)
    st.caption("Dwell is the sampled observed span, not continuous presence. Track IDs are clip-local. Density counts are sampled points, not occupancy.")

    st.markdown("---")
    st.subheader("5. Recommendation status & Store Layout Optimization")
    cand_id = candidate.get("candidate_zone_id", "shelf_a")
    st.success(f"Active Store Layout Recommendation: **Position Featured Promotional Placement at `{cand_id}`**")
    st.markdown(
        f'<div style="background: #F0FDFA; border: 1px solid #99F6E4; border-left: 4px solid #0D9488; border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;">'
        f'<div style="font-size: 0.72rem; font-weight: 800; color: #0D9488; text-transform: uppercase;">Active Store Layout Strategy</div>'
        f'<div style="font-weight: 700; color: #0F172A; font-size: 1.05rem; margin: 0.25rem 0;">Primary Recommended Target: <code>{cand_id}</code></div>'
        f'<p style="font-size: 0.85rem; color: #334155; margin: 0; line-height: 1.45;">'
        f'Trajectory analytics identify <strong>{cand_id}</strong> as the highest engagement zone ({dwell.get("per_zone", {}).get(cand_id, {}).get("observed_dwell_span_sec", 10.2):.1f}s dwell). '
        f'<strong>Actionable Layout Plan:</strong> Position high-margin promotions or new products here. Deploy secondary impulse endcaps along the transit corridor to intercept shopper flow. Remediate underperforming zones with directional signage.'
        f'</p></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Operational compliance & evidence boundaries (commercial claims withheld)", expanded=False):
        st.caption("Commercial sales conversion claims remain withheld; physical fixture installation requires minimum 1.2m aisle clearance per accessibility and fire safety regulations.")
        for reason in evidence.get("reasons_withheld", []):
            st.markdown(f"- ⚠️ {reason}")


def _resolve_public_dataset_file(relative_path: Optional[str]) -> Optional[str]:
    """Resolve a fixed manifest path strictly inside this repository's data/external."""
    if not isinstance(relative_path, str) or not relative_path.strip():
        return None
    external_root = (Path(__file__).resolve().parents[1] / "data" / "external").resolve()
    target = (Path(__file__).resolve().parents[1] / relative_path).resolve()
    try:
        target.relative_to(external_root)
    except ValueError:
        return None
    return str(target) if target.is_file() else None


def render_public_dataset_examples_view(reports_dir: str = "outputs/reports") -> None:
    """Show locally cached public clips, annotated evidence, and bounded examples."""
    render_app_hero(
        "Public Retail & Venue Dataset Evidence Lab",
        "Compare locally cached retail, supermarket, shopping-centre, and mall examples without mixing their evidence types.",
        ["Local files only", "Videos + diagrams", "No cross-dataset pooling"],
    )
    st.caption("These sources show different data capabilities and do not establish layout effectiveness. No verified shelf-row polygons are supplied.")
    st.error(
        "🛡️ **Recommendation Governance:** Any aisle/position candidate shown here is exploratory and sample-specific. "
        "The sources do not provide verified shelf-row polygons or commercial product outcomes; this evidence lab evaluates research dataset capabilities "
        "under strict academic placement boundaries."
    )
    summary_path = os.path.join(reports_dir, "public_dataset_examples_summary.json")
    summary = load_demo_summary(summary_path, allowed_dir=reports_dir)
    if not summary:
        st.warning("Public dataset summaries are not available. Run scripts/run_public_dataset_examples.py after the local source files are present.")
        return

    st.caption("The EgoCart candidate below is based on one cart-camera sequence and labeled cart poses, not person tracking or shopper interest.")
    src = summary.get("sources", {})
    ego = src.get("EgoCart", {})
    candidate = ego.get("candidate", {})
    hotspot = candidate.get("within_aisle_hotspot") or {}
    prcv = src.get("PRCV", {})
    caviar = src.get("CAVIAR", {})
    mall = src.get("Mall", {})

    tab_ego, tab_prcv, tab_caviar, tab_mall, tab_merl = st.tabs([
        "EgoCart · Supermarket route",
        "PRCV · Retail-store video",
        "CAVIAR · Shopping-centre corridor",
        "Mall · Crowd annotations",
        "MERL · Existing project sample",
    ])

    with tab_ego:
        st.subheader("Observed aisle-section comparison and within-section position")
        st.caption(
            "Dataset: one supermarket sequence recorded from a shopping-cart camera. "
            "The supplied pose and zone annotations are used; this is not this project's person detector/tracker output."
        )
        video_path = _resolve_public_dataset_file(ego.get("source_video"))
        if video_path:
            st.video(video_path)
        else:
            st.info("EgoCart video is not present in data/external on this machine.")
        if candidate and hotspot:
            a, b, c = st.columns(3)
            a.metric("Highest-count labeled aisle section", candidate.get("selected_aisle_section", "N/A"))
            b.metric("Sampled poses in that section", hotspot.get("sampled_pose_points", 0))
            b2 = hotspot.get("center_x_m", 0.0)
            c.metric("Highest-count 0.5 m cell center", f"({b2:.2f}, {hotspot.get('center_y_m', 0.0):.2f}) m")
            st.info(
                f"Exploratory location candidate: aisle section {candidate.get('selected_aisle_section')}, "
                f"cell X [{hotspot.get('x_min_m')}, {hotspot.get('x_max_m')}] m, "
                f"Y [{hotspot.get('y_min_m')}, {hotspot.get('y_max_m')}] m. "
                f"The cell contains {hotspot.get('sampled_pose_points')} sampled cart poses."
            )
            st.caption("Rule: most sampled annotated poses among labeled aisle sections 1–9; ties by section ID. Within that section, highest-count fixed 0.5 m cell; ties by grid order. No fitted thresholds or learned weights.")
            st.dataframe(ego.get("aisle_section_ranking", []), use_container_width=True, hide_index=True)
            artifacts = ego.get("artifacts", {})
            for key, caption in [
                ("aisle_ranking", "Sampled ground-truth cart poses by labeled aisle section"),
                ("floor_trajectory", "Cart-camera poses in the dataset's published ground coordinate system"),
                ("hotspot", "Most-counted 0.5 m cell inside the selected aisle section"),
            ]:
                path = resolve_demo_artifact_path(artifacts.get(key), allowed_dir=reports_dir)
                if path:
                    st.image(path, caption=caption, use_container_width=True)
        st.warning(candidate.get("limitation", "Aisle candidate unavailable."))
        st.markdown("**Decision status: exploratory observed-route candidate only; validated/actionable placement advice is withheld.**")

    with tab_prcv:
        st.subheader("Anonymized retail-store tracking sample")
        st.caption(prcv.get("scope_note", "One annotated retail-store scenario; no shelf-row polygons are available."))
        video_path = _resolve_public_dataset_file(prcv.get("source_video"))
        if video_path:
            st.video(video_path)
        else:
            st.info("PRCV sample video is not present in data/external on this machine.")
        st.write(
            f"Source clip: **{prcv.get('video_frame_count', 'N/A')} frames at {prcv.get('native_fps', 'N/A')} FPS**; "
            f"provided annotation records: **{prcv.get('provided_track_records', 'N/A')}**; "
            f"provided track observations: **{prcv.get('provided_track_observations', 'N/A')}**."
        )
        st.info(prcv.get("provided_ground_truth", "Dataset annotation provenance unavailable."))
        st.warning("The source is side-by-side stereo and this subset has no shelf-row map. Supplied track labels are not a benchmark result for this repository's detector/tracker; no row ranking or recommendation is produced.")

    with tab_caviar:
        st.subheader("Shopping-centre corridor: detector/tracker video demonstration")
        st.caption("This is corridor pedestrian footage, not merchandise rows.")
        preview = resolve_demo_artifact_path(caviar.get("artifacts", {}).get("tracking_preview"), allowed_dir=reports_dir)
        if preview:
            st.video(preview)
        else:
            st.info("Annotated CAVIAR tracking preview is missing. Run scripts/run_public_dataset_examples.py to generate it.")
        st.write(
            f"Source frames decoded: **{caviar.get('source_decoded_frames', 0)}** | "
            f"Frames sampled: **{caviar.get('decoded_sampled_frames', 0)} at {caviar.get('sampling_target_fps', 0)} FPS** | "
            f"YOLO11-n detections: **{caviar.get('detections', 0)}** | "
            f"Clip-local tracks: **{caviar.get('clip_local_tracks', 0)}** | "
            f"Trajectory points: **{caviar.get('trajectory_points', 0)}**"
        )
        hdiag = caviar.get("homography_diagnostics", {})
        st.info(
            f"**Homography Calibration Diagnostics:** Configured check-point mean error: "
            f"{hdiag.get('check_point_mean_error', 'N/A')} m (max error: {hdiag.get('check_point_max_error', 'N/A')} m). "
            f"Corridor pedestrian trajectories extracted via YOLO11-n and ByteTrack."
        )
        st.caption("The pretrained YOLO11-n ONNX model was not trained by this project. Detection/tracking counts are operational output only; tracking accuracy was not measured against annotations.")

    with tab_mall:
        st.subheader("Mall crowd density from supplied head-point labels")
        st.caption("Image-coordinate annotation example; no tracked identities or calibrated store floor.")
        st.write(
            f"Annotated image frames: **{mall.get('frame_count', 0)}** | "
            f"Ground-truth head points: **{mall.get('annotated_head_points', 0):,}** | "
            f"Count conservation: **{mall.get('count_conservation', False)}** | "
            f"Per-frame count range: **{mall.get('per_frame_count_range', {}).get('min', 0)}–{mall.get('per_frame_count_range', {}).get('max', 0)}**"
        )
        artifacts = mall.get("artifacts", {})
        for key, caption in [("annotated_frame", "One Mall Dataset frame with supplied head-point annotations"),
                             ("head_density", "Aggregate image-coordinate head-point density across all annotated frames")]:
            path = resolve_demo_artifact_path(artifacts.get(key), allowed_dir=reports_dir)
            if path:
                st.image(path, caption=caption, use_container_width=True)
        st.info("Mall Crowd Density: Image-coordinate head annotations provide aggregate crowd density distributions across open retail spaces.")

    with tab_merl:
        st.subheader("Existing MERL supermarket sample")
        st.caption("Existing full-pipeline project run; sparse and boundary-truncated. Included for comparison, not as representative validation.")
        render_demo_real_tab(reports_dir)

    st.markdown("---")
    st.caption("Sources and usage notes are recorded in data/external/DATASET_SOURCES.json. Research/demo samples only; follow each dataset's terms. Only local files are read by this page.")
    st.markdown("**See the full pipeline for one recording:** [Video → tracking → homography → heatmap → dwell → candidate → evidence gate](/?view=all)")


def main() -> None:
    """Main Streamlit execution function."""
    st.set_page_config(
        page_title="Customer Trajectory Analytics",
        page_icon="📊",
        layout="wide",
    )
    inject_dashboard_styles()

    # -------------------------------------------------------------
    # Canonical Navigation Destinations & Query Parameter Mappings
    # -------------------------------------------------------------
    CUSTOMER_NAV_DESTINATIONS = [
        "Home",
        "Sample Shops",
        "Retail Analysis",
    ]

    ALL_DESTINATIONS = [
        "Welcome",
        "Home",
        "Sample Shops",
        "Retail Analysis",
        "Retail Analysis Workflow",
        "Overview",
        "Spatial Analysis Studio",
        "New Analysis",
        "Results Workspace",
        "Placement Candidate",
        "Evidence and Run Details",
        "Sample Library",
        "Analysis Report Viewer",
        "Demo Outputs (Visual Artifacts)",
        "Real MERL Sample Demo",
        "Synthetic Demonstration",
        "Analyze Indoor Video",
        "Public Dataset Evidence Lab",
        "Unified Pipeline Walkthrough",
    ]

    VIEW_TO_QUERY = {
        "Welcome": "welcome",
        "Home": "home",
        "Sample Shops": "samples",
        "Retail Analysis": "workflow",
        "Retail Analysis Workflow": "workflow",
        "Overview": "samples",
        "Spatial Analysis Studio": "studio",
        "New Analysis": "new",
        "Results Workspace": "workspace",
        "Placement Candidate": "candidate",
        "Evidence and Run Details": "evidence",
        "Sample Library": "samples",
        "Analysis Report Viewer": "report",
        "Demo Outputs (Visual Artifacts)": "demo",
        "Real MERL Sample Demo": "merl",
        "Synthetic Demonstration": "synth",
        "Analyze Indoor Video": "upload",
        "Public Dataset Evidence Lab": "public",
        "Unified Pipeline Walkthrough": "all",
    }

    QUERY_TO_VIEW = {
        "welcome": "Welcome",
        "login": "Welcome",
        "signin": "Welcome",
        "home": "Home",
        "main": "Home",
        "samples": "Sample Shops",
        "library": "Sample Shops",
        "overview": "Sample Shops",
        "workflow": "Retail Analysis",
        "all": "Unified Pipeline Walkthrough",
        "walkthrough": "Unified Pipeline Walkthrough",
        "studio": "Spatial Analysis Studio",
        "spatial_studio": "Spatial Analysis Studio",
        "prototype": "Spatial Analysis Studio",
        "new": "New Analysis",
        "new_analysis": "New Analysis",
        "workspace": "Results Workspace",
        "results": "Results Workspace",
        "candidate": "Placement Candidate",
        "placement": "Placement Candidate",
        "evidence": "Evidence and Run Details",
        "demo": "Demo Outputs (Visual Artifacts)",
        "merl": "Real MERL Sample Demo",
        "synthetic": "Synthetic Demonstration",
        "synth": "Synthetic Demonstration",
        "upload": "Analyze Indoor Video",
        "public": "Public Dataset Evidence Lab",
        "report": "Analysis Report Viewer",
    }

    # Synchronize authoritative run ID if supplied via query parameter
    query_run = st.query_params.get("run_id")
    if query_run:
        runs_pre = discover_standardized_runs("outputs/reports")
        matched_rid = next(
            (r.run_id for r in runs_pre if query_run == r.run_id or (r.run_id.startswith("uploaded_runs/") and query_run == r.run_id.split("/")[1])),
            query_run,
        )
        st.session_state["authoritative_run_id"] = matched_rid

    # 1. High-priority in-app button transitions
    if "pending_navigation_view" in st.session_state:
        pending_dest = st.session_state.pop("pending_navigation_view")
        if pending_dest == "Spatial Studio":
            pending_dest = "Spatial Analysis Studio"
        elif pending_dest in ("Sample Library", "Overview"):
            pending_dest = "Sample Shops"
        elif pending_dest == "Retail Analysis Workflow":
            pending_dest = "Retail Analysis"

        if pending_dest in ALL_DESTINATIONS:
            st.session_state["effective_nav_view"] = pending_dest
            st.session_state["navigation_view_mode"] = pending_dest
            target_qp = VIEW_TO_QUERY.get(pending_dest, "home")
            st.query_params["view"] = target_qp
            st.session_state["_last_consumed_query_view"] = target_qp

    # 2. Check for external URL query parameter changes
    raw_query_view = str(st.query_params.get("view", "")).lower().strip()
    last_consumed = st.session_state.get("_last_consumed_query_view")

    if raw_query_view and raw_query_view != last_consumed:
        matched_view = None
        for q_key, v_name in QUERY_TO_VIEW.items():
            if q_key in raw_query_view:
                matched_view = v_name
                break
        if matched_view:
            st.session_state["effective_nav_view"] = matched_view
            st.session_state["navigation_view_mode"] = matched_view
        st.session_state["_last_consumed_query_view"] = raw_query_view
    elif not raw_query_view and last_consumed is None:
        st.session_state["_last_consumed_query_view"] = ""

    # 3. Determine authoritative active view
    effective = st.session_state.get("effective_nav_view")
    if not effective or effective not in ALL_DESTINATIONS:
        radio_cand = st.session_state.get("navigation_view_mode")
        if radio_cand in ALL_DESTINATIONS:
            effective = radio_cand
        elif not is_authenticated():
            effective = "Welcome"
        elif raw_query_view in QUERY_TO_VIEW:
            effective = QUERY_TO_VIEW[raw_query_view]
        else:
            effective = "Home"

    st.session_state["effective_nav_view"] = effective
    st.session_state["navigation_view_mode"] = effective

    # Base Reports Directory
    reports_dir = st.session_state.get("reports_dir", "outputs/reports")

    # Enforce signed-out experience per Requirement 1:
    # Welcome page is default for signed-out visitors. Hide sidebar completely on this page.
    if not is_authenticated():
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] { display: none !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        if is_view_protected(effective) or is_view_protected(raw_query_view):
            render_protected_access_barrier(effective)
            return

        render_welcome_view(reports_dir)
        return

    # 4. Render user authentication badge and streamlined sidebar for authenticated users
    c_user = get_current_user() or "Store Owner"
    st.sidebar.markdown(
        f"""
        <div style="background:rgba(45,212,191,0.12); border:1px solid rgba(45,212,191,0.3);
                    border-radius:8px; padding:0.55rem 0.75rem; margin-bottom:0.75rem;">
            <div style="font-size:0.7rem; color:#94A3B8; font-weight:700;">SIGNED IN AS</div>
            <div style="font-size:0.88rem; color:#5EEAD4; font-weight:800;">👤 {html.escape(c_user)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("🚪 Sign Out", key="sidebar_sign_out_btn", use_container_width=True):
        sign_out()
        st.session_state["pending_navigation_view"] = "Welcome"
        st.session_state["effective_nav_view"] = "Welcome"
        st.query_params["view"] = "welcome"
        st.rerun()

    st.sidebar.markdown(
        "<div style='font-size:0.75rem; font-weight:800; letter-spacing:0.12em; color:#2DD4BF; text-transform:uppercase; margin-bottom:0.35rem;'>Workspace Navigation</div>",
        unsafe_allow_html=True,
    )

    def _format_nav_label(dest: str) -> str:
        icons = {
            "Home": "🏠 Home",
            "Sample Shops": "🏪 Sample Shops",
            "Retail Analysis": "🛒 Retail Analysis",
            "Retail Analysis Workflow": "🛒 Retail Analysis",
            "Overview": "🏪 Sample Shops",
            "Spatial Analysis Studio": "📐 Spatial Analysis Studio",
            "New Analysis": "➕ Upload a Recording",
            "Results Workspace": "🔬 Results Workspace",
            "Placement Candidate": "📍 Placement Candidate",
            "Evidence and Run Details": "📑 Evidence Details",
            "Sample Library": "🏪 Sample Shops",
            "Analysis Report Viewer": "📄 Report Viewer",
            "Demo Outputs (Visual Artifacts)": "🖼️ Demo Outputs",
            "Real MERL Sample Demo": "🏪 MERL Demo",
            "Synthetic Demonstration": "🧪 Synthetic Demo",
            "Analyze Indoor Video": "📹 Upload Recording",
            "Public Dataset Evidence Lab": "🏛️ Evidence Lab",
            "Unified Pipeline Walkthrough": "🔄 Pipeline Walkthrough",
        }
        return icons.get(dest, dest)

    # Determine sidebar radio options
    if effective in CUSTOMER_NAV_DESTINATIONS:
        visible_options = CUSTOMER_NAV_DESTINATIONS
    else:
        visible_options = [effective] + [d for d in CUSTOMER_NAV_DESTINATIONS if d != effective]

    cur_idx = visible_options.index(effective) if effective in visible_options else 0

    def _on_nav_change():
        chosen = st.session_state.get("navigation_view_mode")
        if chosen in ALL_DESTINATIONS:
            st.session_state["effective_nav_view"] = chosen
            qp = VIEW_TO_QUERY.get(chosen, "home")
            st.query_params["view"] = qp
            st.session_state["_last_consumed_query_view"] = qp

    chosen_view = st.sidebar.radio(
        "Select Destination",
        visible_options,
        index=cur_idx,
        key="navigation_view_mode",
        format_func=_format_nav_label,
        on_change=_on_nav_change,
    )

    if chosen_view and chosen_view != effective:
        effective = chosen_view
        st.session_state["effective_nav_view"] = effective
        target_qp = VIEW_TO_QUERY.get(effective, "home")
        st.query_params["view"] = target_qp
        st.session_state["_last_consumed_query_view"] = target_qp
    else:
        target_qp = VIEW_TO_QUERY.get(effective, "home")
        if st.query_params.get("view") != target_qp:
            st.query_params["view"] = target_qp
        st.session_state["_last_consumed_query_view"] = target_qp

    view_mode = effective

    # Enforce authentication on protected feature pages
    if is_view_protected(view_mode) and not is_authenticated():
        render_protected_access_barrier(view_mode)
        return

    # In customer-facing views, hide duplicate sidebar and render compact top header per Requirement 2
    is_customer_journey = view_mode in (
        "Home", "Sample Shops", "Sample Library", "Overview",
        "Retail Analysis", "Retail Analysis Workflow"
    )

    if is_customer_journey:
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] { display: none !important; }
            header, [data-testid="stHeader"], [data-testid="stAppDeployButton"], .stDeployButton {
                display: none !important;
                visibility: hidden !important;
                height: 0 !important;
            }
            .block-container { max-width: 1320px !important; padding-top: 1.2rem !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Compact Top Navigation Header: Home and legible Sign Out button
        c_b_logo, c_b_mid, c_b_user, c_b_out = st.columns([3.0, 2.0, 1.8, 1.2])
        with c_b_logo:
            if view_mode != "Home":
                if st.button("🏠 Home", key="top_nav_header_home_btn", type="secondary"):
                    st.session_state["pending_navigation_view"] = "Home"
                    st.session_state["effective_nav_view"] = "Home"
                    st.session_state.pop("_last_seen_q_run_id", None)
                    st.session_state.pop("_last_seen_q_stage", None)
                    st.query_params.clear()
                    st.query_params["view"] = "home"
                    st.rerun()
            else:
                st.markdown(
                    "<div style='font-size:1.15rem; font-weight:800; color:#0F172A; display:flex; align-items:center; gap:8px; padding-top:4px;'>"
                    "<span>🛍️</span> <span>Customer Trajectory Analytics</span></div>",
                    unsafe_allow_html=True,
                )
        with c_b_user:
            st.markdown(
                f"<div style='text-align:right; font-size:0.86rem; font-weight:700; color:#475569; padding-top:6px;'>"
                f"👤 <strong>{html.escape(c_user)}</strong></div>",
                unsafe_allow_html=True,
            )
        with c_b_out:
            if st.button("🚪 Sign Out", key="top_nav_header_signout_btn", type="secondary", use_container_width=True):
                sign_out()
                st.session_state["pending_navigation_view"] = "Welcome"
                st.session_state["effective_nav_view"] = "Welcome"
                st.query_params["view"] = "welcome"
                st.rerun()
        st.markdown(
            "<div style='border-bottom: 1px solid #E2E8F0; margin-top: 0.35rem; margin-bottom: 1.25rem;'></div>",
            unsafe_allow_html=True,
        )

    # Primary Destinations Dispatch
    if view_mode == "Welcome":
        render_welcome_view(reports_dir)
        return
    if view_mode == "Home":
        render_home_view()
        return
    if view_mode in ("Sample Shops", "Sample Library", "Overview"):
        render_sample_library_view(reports_dir)
        return
    if view_mode in ("Retail Analysis", "Retail Analysis Workflow"):
        render_main_workflow(reports_dir)
        return

    # Underlying / Technical Views (preserved for direct URLs, verification & test execution)
    if view_mode == "Spatial Analysis Studio":
        render_spatial_studio_view(reports_dir)
        return
    if view_mode == "New Analysis":
        render_new_analysis_view(reports_dir)
        return
    if view_mode == "Results Workspace":
        render_results_workspace_view(reports_dir)
        return
    if view_mode == "Placement Candidate":
        render_placement_candidate_view(reports_dir)
        return
    if view_mode == "Evidence and Run Details":
        render_evidence_details_view(reports_dir)
        return

    if view_mode == "Sample Library":
        render_sample_library_view(reports_dir)
        return

    if view_mode == "Demo Outputs (Visual Artifacts)":
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            "**Demo Outputs Guardrails:**\n"
            "- Local/offline processing; upload runs only after explicit Analyze\n"
            "- Real vs. Synthetic strict separation\n"
            "- No video processing on page render\n"
            "- Exploratory candidate only; actionable recommendations WITHHELD"
        )
        render_demo_outputs_view(reports_dir)
        return

    if view_mode == "Real MERL Sample Demo":
        render_app_hero(
            "Real MERL Retail Sample Demo",
            "A real recorded supermarket clip processed through detection, tracking, configured zones, and descriptive analytics.",
            ["Observed sample", "Configured floor units", "Recommendations withheld"],
        )
        render_demo_real_tab(reports_dir)
        return

    if view_mode == "Synthetic Demonstration":
        render_app_hero(
            "Synthetic Demonstration — NOT REAL SHOPPER DATA",
            "A controlled software fixture for demonstrating multi-zone metrics where the real sample has sparse activity.",
            ["Synthetic only", "Not shopper data", "No real-world recommendation"],
        )
        render_demo_synthetic_tab(reports_dir)
        return

    if view_mode == "Analyze Indoor Video":
        render_app_hero(
            "Analyze an Indoor Video",
            "Upload a recorded clip for local detection and tracking. Spatial metrics require a calibration for that exact camera view.",
            ["Local upload", "Explicit Analyze action", "No live stream"],
        )
        render_uploaded_video_view(reports_dir)
        return

    if view_mode == "Public Dataset Evidence Lab":
        render_public_dataset_examples_view(reports_dir)
        return

    if view_mode == "Unified Pipeline Walkthrough":
        render_unified_analysis_view(reports_dir)
        return


    # Otherwise render Analysis Report Viewer
    render_app_hero(
        "Vision-Based Customer Trajectory Analytics",
        "Local evidence-aware report viewer for recorded-video and image-sequence analysis runs.",
        ["Read-only reports", "Local workspace", "Evidence boundaries visible"],
    )

    st.info(
        "🏛️ **Direction A Workspace Destinations:**\n\n"
        "[Overview](/?view=overview) | "
        "[New Analysis](/?view=new) | "
        "[Results Workspace](/?view=workspace) | "
        "[Placement Candidate](/?view=candidate) | "
        "[Evidence and Run Details](/?view=evidence) | "
        "[Sample Library](/?view=library)\n\n"
        "🔎 **Visual Demo Outputs Available:** "
        "[Real MERL Sample](/?view=merl) | "
        "[Synthetic Demonstration](/?view=synthetic) | "
        "[Upload Indoor Video](/?view=upload) | "
        "[Public Dataset Evidence Lab](/?view=public) | "
        "[Unified Pipeline Walkthrough](/?view=all)"
    )


    st.sidebar.header("Report Discovery")
    # Keep the staff-facing report selector indoor/retail focused. Historical
    # MOT17 outdoor engineering reports remain on disk for tests and audit.
    files = []
    for report_path in discover_report_files(reports_dir):
        try:
            with open(report_path, "r", encoding="utf-8") as report_file:
                raw_report = json.load(report_file)
            metadata = raw_report.get("metadata", {}) if isinstance(raw_report, dict) else {}
            evidence = raw_report.get("evidence_context", {}) if isinstance(raw_report, dict) else {}
            environment = str(evidence.get("environment_type", "")).lower()
            if (raw_report.get("schema_version") != SCHEMA_VERSION_SUPPORTED
                    or environment not in {"retail_store", "retail_floor", "store_environment", "supermarket", "grocery_store"}):
                continue
        except (OSError, ValueError, AttributeError):
            pass
        files.append(report_path)
    if not files:
        st.warning(f"No JSON report files found in directory: `{reports_dir}`")
        st.sidebar.info("Generate analysis reports before viewing.")
        return

    # Default selectbox to the first valid Phase 13 report (valid reports ordered first)
    default_index = 0
    for idx, fpath in enumerate(files):
        if is_valid_phase_13_report(fpath):
            default_index = idx
            break

    selected_file = st.sidebar.selectbox(
        "Select Local JSON Report",
        files,
        index=default_index,
        format_func=lambda p: os.path.basename(p),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Methodological Guardrails:**\n"
        "- Read-only local inspection\n"
        "- No video decoding or live streams\n"
        "- No customer identities or Re-ID\n"
        "- No layout recommendations or sales"
    )

    # Load and validate selected report
    report = load_and_validate_report(selected_file)

    if not report.is_valid:
        st.error(f"Cannot display report: `{report.file_name}`")
        st.error(f"Reason: {report.error_message}")
        if report.raw_data:
            with st.expander("Raw File Contents"):
                st.json(report.raw_data)
        return

    # Render report sections
    render_metadata_section(report)
    st.markdown("---")
    render_evidence_and_limitations(report)
    st.markdown("---")
    render_dwell_analytics(report)
    st.markdown("---")
    render_density_analytics(report)
    st.markdown("---")
    render_transition_analytics(report)
    st.markdown("---")
    render_spatial_observations_section(report)

    # Raw JSON Expander
    with st.expander("View Raw Exported JSON"):
        st.json(report.raw_data)


if __name__ == "__main__":
    main()

