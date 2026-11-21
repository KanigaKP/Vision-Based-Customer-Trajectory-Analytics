"""
Direction A Destination 3: Results Workspace View.
==================================================
Provides a unified, authoritative analysis workspace for the selected run across 5 tabs:
  Tab 1: Video & Tracking Frame (playback, frame fallback, download)
  Tab 2: Floor Trajectories & Zone Geometry (orthogonal SVG floor plan + trajectory plot)
  Tab 3: Discrete Point Density (discrete binned heatmap, out-of-bounds metrics)
  Tab 4: Observed Dwell Spans (per-zone dwell table, preserving zero vs unavailable)
  Tab 5: Direct Transitions (direct transition counts and matrix)
"""

import os
from dashboard.views.run_journey import render_run_journey
import streamlit as st
from typing import List, Optional

from dashboard.run_adapter import (
    StandardizedRunData,
    discover_standardized_runs,
    format_observed_value,
    load_run_observation_timeline,
    render_synchronized_player,
)
from dashboard.theme import (
    render_metric_card_html,
    render_withheld_banner_html,
    generate_floor_plan_svg,
    generate_isometric_floor_plan_svg,
    ZONE_COLORS,
    COLOR_TEAL_ACCENT,
    COLOR_BORDER,
)


def render_results_workspace_view(reports_dir: str = "outputs/reports") -> None:
    """Render the Direction A Results Workspace."""
    runs = discover_standardized_runs(reports_dir)
    if not runs:
        st.warning(f"No runs found in `{reports_dir}`. Please run an analysis or generate demo summaries.")
        return

    # Synchronize authoritative run selection across workspace
    run_ids = [r.run_id for r in runs]
    query_run_id = st.query_params.get("run_id")
    if query_run_id:
        matched = next((rid for rid in run_ids if query_run_id == rid or (rid.startswith("uploaded_runs/") and query_run_id == rid.split("/")[1])), None)
        if matched and st.session_state.get("workspace_applied_query") != matched:
            st.session_state["authoritative_run_id"] = matched
            st.session_state["workspace_active_run_selector"] = matched
            st.session_state["workspace_applied_query"] = matched

    current_run_id = st.session_state.get("authoritative_run_id")
    if current_run_id not in run_ids:
        current_run_id = run_ids[0]
        st.session_state["authoritative_run_id"] = current_run_id

    # If widget state has a value, sync it bidirectionally
    if "workspace_active_run_selector" in st.session_state and st.session_state["workspace_active_run_selector"] in run_ids and not query_run_id:
        current_run_id = st.session_state["workspace_active_run_selector"]
        st.session_state["authoritative_run_id"] = current_run_id

    default_idx = run_ids.index(current_run_id)

    # Run Selector Header
    c_sel, c_actions = st.columns([3.5, 1.5])
    with c_sel:
        def _on_run_change() -> None:
            st.session_state["authoritative_run_id"] = st.session_state["workspace_active_run_selector"]
            st.query_params["run_id"] = st.session_state["authoritative_run_id"]
            st.session_state["workspace_applied_query"] = st.session_state["authoritative_run_id"]
            # Clear previous run's playback/selection state
            for k in list(st.session_state.keys()):
                if k.startswith("sync_sample_") or k.startswith("sync_track_"):
                    st.session_state.pop(k, None)

        selected_run_id = st.selectbox(
            "Authoritative Selected Run",
            options=run_ids,
            index=default_idx,
            format_func=lambda rid: next((r.display_name for r in runs if r.run_id == rid), rid),
            key="workspace_active_run_selector",
            on_change=_on_run_change,
        )

    run = next(r for r in runs if r.run_id == selected_run_id)

    with c_actions:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("🔍 View Candidate Analysis ➔", use_container_width=True):
            st.session_state["pending_navigation_view"] = "Placement Candidate"
            st.rerun()

    # Compact Run Header Card
    badge_class = "dir-a-badge-synthetic" if run.is_synthetic else ("dir-a-badge-neutral" if not run.is_retail else "dir-a-badge-verified")
    badge_text = "SYNTHETIC FIXTURE" if run.is_synthetic else ("UNKNOWN PROVENANCE" if run.is_retail is None else ("NON-RETAIL SEQUENCE" if run.is_retail is False else "OBSERVED RETAIL DATA"))

    st.markdown(
        f'<div class="dir-a-card" style="margin-top:0.3rem; margin-bottom:0.5rem; padding:0.85rem 1.1rem;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.4rem;">'
        f'<div>'
        f'<span class="dir-a-badge {badge_class}">{badge_text}</span>'
        f'<span class="dir-a-badge dir-a-badge-withheld" style="margin-left:0.4rem;">STATUS: {run.evidence_status}</span>'
        f'<h2 style="margin:0.25rem 0 0.15rem 0; font-size:1.25rem; color:#0F172A;">{run.display_name}</h2>'
        f'<span style="font-size:0.8rem; color:#64748B;">'
        f'Dataset: <code>{run.dataset_name}</code> | Source: <code>{run.source_type}</code> | '
        f'Lineage: <code>{run.model_family}</code> on <code>{run.model_runtime}</code>'
        f'</span>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<span style="font-size:0.82rem; font-weight:700; color:#334155;">'
        f'Source: {run.source_frame_count or "N/A"} frames {f"@ {run.source_fps:.1f} FPS" if run.source_fps is not None else ""} | '
        f'Sampled: {run.sampled_frame_count or run.total_frames or "N/A"} frames {f"@ {run.sampling_fps:.1f} FPS" if run.sampling_fps is not None else ""}'
        f'</span><br/>'
        f'<span style="font-size:0.8rem; color:#64748B;">Clip Tracks: {format_observed_value(run.total_tracks)} | Detections: {format_observed_value(run.total_detections)}</span>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Concise Evidence Status & Accessible Expandable Limitations
    st.caption(
        f"🔒 **Evidence Status: {run.evidence_status}** — Single-camera recording with uncalibrated floor scale. "
        "All actionable layout recommendations remain strictly withheld."
    )
    if run.reasons_withheld or run.caveats:
        with st.expander(f"⚠️ View Detailed Evidence Boundaries & Methodological Limitations ({run.evidence_status})", expanded=False):
            items = run.reasons_withheld + (run.caveats or [])
            li_items = "".join(f"<li>{r}</li>" for r in items)
            st.markdown(f"<ul style='margin:0; padding-left:1.2rem; color:#334155; font-size:0.85rem; line-height:1.55;'>{li_items}</ul>", unsafe_allow_html=True)

    render_run_journey(run)

    # 5 Workspace Tabs (Mobile-friendly responsive analytical sections)
    tab_video, tab_floor, tab_density, tab_dwell, tab_transitions = st.tabs([
        "1. Video & Tracking Frame",
        "2. Floor Trajectories & Zones",
        "3. Discrete Point Density",
        "4. Observed Dwell Spans",
        "5. Direct Transitions",
    ])

    # -------------------------------------------------------------
    # TAB 1: Video & Tracking Frame
    # -------------------------------------------------------------
    with tab_video:
        video_path = run.artifacts.get("tracking_video")
        frame_path = run.artifacts.get("tracking_frame")
        timeline_data = load_run_observation_timeline(run)

        if timeline_data and run.has_observation_timeline and run.has_geometry:
            st.subheader("Frame-Aligned Video & Trajectory Explorer")
            st.caption(
                "Interactive time-aligned exploration linking video playback, tracked bounding boxes, "
                "and synchronized orthogonal floor trajectories."
            )
            sample_query = st.query_params.get("sample")
            default_sample = int(sample_query) if sample_query and sample_query.isdigit() else 0
            seek_sample = st.session_state.get(f"sync_sample_{run.run_id}", default_sample)
            sel_zone = st.session_state.get("authoritative_selected_zone", run.candidate_zone)
            render_synchronized_player(
                run,
                timeline_data=timeline_data,
                seek_sample=seek_sample,
                selected_zone_id=sel_zone,
                key=f"sync_player_{run.run_id}",
            )
            if video_path and os.path.isfile(video_path):
                with open(video_path, "rb") as vf:
                    st.download_button(
                        "📥 Download Annotated Tracking Video (WebM)",
                        vf.read(),
                        file_name=os.path.basename(video_path),
                        mime="video/webm",
                        use_container_width=True,
                    )
        else:
            st.subheader("Video Playback and Tracking Overlay")
            st.caption("Displays detected bounding boxes and clip-local tracker IDs.")
            st.info("ℹ️ Frame-aligned exploration is unavailable because this run lacks a usable timeline or floor geometry.")

            if video_path and os.path.isfile(video_path):
                st.video(video_path, format="video/webm")
                with open(video_path, "rb") as vf:
                    st.download_button(
                        "📥 Download Annotated Tracking Video (WebM)",
                        vf.read(),
                        file_name=os.path.basename(video_path),
                        mime="video/webm",
                        use_container_width=True,
                    )
            elif frame_path and os.path.isfile(frame_path):
                st.image(frame_path, caption="Annotated tracking frame (Video playback unavailable or omitted)", use_container_width=True)
                with open(frame_path, "rb") as ff:
                    st.download_button(
                        "📥 Download Annotated Tracking Frame (PNG)",
                        ff.read(),
                        file_name=os.path.basename(frame_path),
                        mime="image/png",
                    )
            else:
                st.info("No video playback or tracking frame artifact is available for this run.")

        st.caption(
            "⚠️ **Methodological Boundary:** Tracker IDs are strictly clip-local. "
            "No cross-camera re-identification, facial recognition, or biometric identification is performed."
        )

    # -------------------------------------------------------------
    # TAB 2: Floor Trajectories & Zones
    # -------------------------------------------------------------
    with tab_floor:
        st.subheader("Floor Plane Trajectories & Zone Geometry")
        if run.bounds:
            st.caption(f"Planar projection into configured floor units (X [{run.bounds[0]:.1f}..{run.bounds[1]:.1f}] × Y [{run.bounds[2]:.1f}..{run.bounds[3]:.1f}]).")
        else:
            st.caption("Planar projection maps detections into configured floor units.")

        c_svg, c_plot = st.columns([1.1, 1.0])
        with c_svg:
            st.markdown("##### Floor Plan Geometry")
            workspace_floor_mode = st.radio(
                "Perspective Mode",
                ["Orthogonal 2D Plan", "Isometric 3D Projection (2D Floor Basis)"],
                horizontal=True,
                key="workspace_floor_mode_choice",
                label_visibility="collapsed",
            )
            if run.has_geometry and run.bounds and run.zones:
                if "Isometric" in workspace_floor_mode:
                    svg_html = generate_isometric_floor_plan_svg(
                        bounds=run.bounds,
                        zones=run.zones,
                        zone_colors=run.zone_colors,
                        zone_names=run.zone_names,
                        candidate_hotspot=run.candidate_hotspot_cell,
                        candidate_zone_id=run.candidate_zone,
                    )
                    if svg_html:
                        st.markdown(svg_html, unsafe_allow_html=True)
                        st.caption("Isometric projection derived strictly from 2D planar floor geometry. Extruded heights are illustrative; not a 3D sensor capture.")
                    else:
                        st.info("Floor Plan Geometry: Cannot render isometric SVG for the provided coordinate bounds.")
                else:
                    svg_html = generate_floor_plan_svg(
                        bounds=run.bounds,
                        zones=run.zones,
                        zone_colors=run.zone_colors,
                        zone_names=run.zone_names,
                        candidate_hotspot=run.candidate_hotspot_cell,
                        candidate_zone_id=run.candidate_zone,
                    )
                    if svg_html:
                        st.markdown(svg_html, unsafe_allow_html=True)
                        st.caption("Orthogonal floor plane diagram (configured floor units).")
                    else:
                        st.info("Floor Plan Geometry: Cannot render SVG for the provided coordinate bounds.")
            else:
                st.info("Floor Plan Geometry: Not available for this run. No scene profile or calibrated zone bounds were recorded.")

        with c_plot:
            st.markdown("##### Generated Trajectory Artifact")
            traj_img = run.artifacts.get("floor_trajectory")
            if traj_img and os.path.isfile(traj_img):
                st.image(traj_img, caption="Projected floor trajectory plot", use_container_width=True)
            else:
                st.info("Trajectory plot artifact not present on disk.")

        # Per-Zone point counts table
        st.markdown("##### Zone Assignment Counts")
        if run.zone_assignment_counts:
            z_rows = []
            for zid, cnt in run.zone_assignment_counts.items():
                z_color = run.zone_colors.get(zid, ZONE_COLORS.get(zid, "#64748B")) if run.zone_colors else ZONE_COLORS.get(zid, "#64748B")
                z_display = run.zone_names.get(zid, zid.replace("_", " ").title()) if run.zone_names else zid.replace("_", " ").title()
                z_rows.append({"Zone ID": zid, "Name": z_display, "Color": z_color, "Sampled Points": cnt})
            st.dataframe(z_rows, use_container_width=True)
        else:
            st.caption("No zone point assignments recorded for this run.")

        st.info(
            "📐 **Floor Observation Regions vs. Elevated Display Fixtures:**\n"
            "- Coordinate projections reflect 2D floor surface ground contacts from customer foot points.\n"
            "- Configured zone polygons represent floor aisle areas bordering fixtures, NOT elevated display countertops or checkout surfaces.\n"
            "- Foot-point proximity to a display fixture does NOT measure countertop touchpoints, vertical reach, or fixture installation feasibility.\n"
            "- Coordinate scale operates in configured floor units; independent physical metric scale is unverified."
        )

    # -------------------------------------------------------------
    # TAB 3: Discrete Point Density
    # -------------------------------------------------------------
    with tab_density:
        st.subheader("Discrete Trajectory Point Density")
        cell_desc = f"(cell size: {run.cell_size} configured units)" if run.cell_size else "across configured floor bounds"
        st.caption(f"Discrete grid binning {cell_desc}. Image rendering displays bins using nearest-neighbour pixel sampling.")

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(render_metric_card_html("Evaluated Points", format_observed_value(run.total_points, missing_text="Not evaluated")), unsafe_allow_html=True)
        with d2:
            st.markdown(render_metric_card_html("Binned Points", format_observed_value(run.binned_points, missing_text="Not evaluated")), unsafe_allow_html=True)
        with d3:
            st.markdown(render_metric_card_html("Out of Bounds", format_observed_value(run.out_of_bounds_points, missing_text="Not evaluated")), unsafe_allow_html=True)
        with d4:
            st.markdown(render_metric_card_html("Max Cell Density", format_observed_value(run.max_density, missing_text="Not evaluated")), unsafe_allow_html=True)

        max_d_str = f"{run.max_density}" if run.max_density is not None else "N/A"
        st.markdown(
            f'<div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:0.6rem 0.8rem; margin:0.6rem 0 1rem 0;">'
            f'<div style="font-size:0.75rem; font-weight:700; color:#334155; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.35rem;">'
            f'Colormap Palette & Discrete Normalization:</div>'
            f'<div style="font-size:0.83rem; color:#475569; line-height:1.45;">'
            f'Rendered via matplotlib <code>viridis</code> colormap (nearest-neighbour discrete cell binning). '
            f'Dark purple represents <strong>0 sampled trajectory points</strong>; cyan/green indicates moderate density; '
            f'bright yellow indicates the peak cell count (<strong>{max_d_str} sampled points</strong> for this run). '
            f'The vertical colorbar integrated into the artifact reflects the exact continuous min-to-max scaling.'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        heatmap_img = run.artifacts.get("density_heatmap")
        if heatmap_img and os.path.isfile(heatmap_img):
            st.image(heatmap_img, caption="Discrete density heatmap artifact", use_container_width=True)
        else:
            st.info("Heatmap Image: **Omitted** (No explicit heatmap file path exists for this run).")

        st.caption(
            "⚠️ **Methodological Boundary:** Density counts represent discrete sampled trajectory points, "
            "not physical crowd occupancy, continuous presence, or product demand."
        )


    # -------------------------------------------------------------
    # TAB 4: Observed Dwell Spans
    # -------------------------------------------------------------
    with tab_dwell:
        st.subheader("Observed Zone Dwell Spans")
        st.caption("Temporal spans between sampled detections across configured zone polygons.")

        if run.dwell_status != "available":
            st.warning(f"Dwell Analytics: **Not Evaluated** ({run.dwell_status}). Indoor retail zone polygons are not applicable to this dataset.")
        else:
            dw1, dw2 = st.columns(2)
            with dw1:
                dwell_str = format_observed_value(run.total_dwell_sec, "s", missing_text="0.0 s")
                st.markdown(render_metric_card_html("Total Dwell Span", dwell_str, "Sum of observed spans"), unsafe_allow_html=True)
            with dw2:
                ep_str = format_observed_value(run.total_episodes, missing_text="0")
                st.markdown(render_metric_card_html("Total Dwell Episodes", ep_str, "Zone entry-to-exit runs"), unsafe_allow_html=True)

            if run.zone_dwell:
                st.markdown("##### Per-Zone Observed Dwell Breakdown")
                dwell_rows = []
                for zid, zdata in run.zone_dwell.items():
                    dwell_rows.append({
                        "Zone": zid,
                        "Observed Dwell Span (s)": f"{float(zdata.get('observed_dwell_span_sec', 0.0)):.2f}",
                        "Episodes": zdata.get("episodes", 0),
                        "Distinct Clip Tracks": zdata.get("distinct_clip_local_tracks", 0),
                    })
                st.table(dwell_rows)

            dwell_plot = run.artifacts.get("zone_dwell")
            if dwell_plot and os.path.isfile(dwell_plot):
                st.image(dwell_plot, caption="Zone dwell summary chart", use_container_width=True)

            if run.total_dwell_sec == 0.0 and run.total_episodes == 0:
                st.caption("ℹ️ Evaluated Zero Observation: The target person tracks remained outside configured retail zone boundaries during this run.")

        st.caption(
            "⚠️ **Methodological Boundary:** Dwell spans measure observed durations between sampled frames; "
            "they do not imply continuous presence or engaged product shopping."
        )

    # -------------------------------------------------------------
    # TAB 5: Direct Transitions
    # -------------------------------------------------------------
    with tab_transitions:
        st.subheader("Direct Inter-Zone Transitions")
        st.caption("Consecutive zone sequence counts (A → B) without intermediate zone visits.")

        if run.transition_status != "available":
            st.warning("Transition Analytics: **Not Evaluated** (Not applicable to this non-retail sequence).")
        else:
            t1, t2 = st.columns(2)
            with t1:
                tr_str = format_observed_value(run.total_transitions, missing_text="0")
                st.markdown(render_metric_card_html("Total Direct Transitions", tr_str, "Inter-zone sequence counts"), unsafe_allow_html=True)
            with t2:
                edge_count = len(run.direct_transitions)
                st.markdown(render_metric_card_html("Active Directed Edges", str(edge_count), "Observed transition paths"), unsafe_allow_html=True)

            if run.direct_transitions:
                st.markdown("##### Direct Transition Counts")
                st.dataframe(
                    [{"Transition Edge": k, "Count": v} for k, v in run.direct_transitions.items()],
                    use_container_width=True,
                )
            else:
                st.caption("Zero direct transitions observed between configured zones.")

            if run.transition_matrix:
                with st.expander("View Full Transition Matrix"):
                    st.json(run.transition_matrix)

        st.caption(
            "⚠️ **Methodological Boundary:** Direct transition counts represent observed sequence events; "
            "they do not model Markovian transition probabilities or behavioral intent."
        )
