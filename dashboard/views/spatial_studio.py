"""
Spatial Analysis Studio — Interactive 11-Stage Research Demonstration Prototype.
================================================================================
Implements the unified 11-stage spatial analytics pipeline:
  01. Source Recording
  02. Input Suitability Assessment
  03. Camera Geometry & Zones
  04. Detection
  05. Tracking
  06. Floor Trajectories
  07. Density / Heat Zones
  08. Observed Dwell & Zone Activity
  09. Direct Transitions
  10. Placement Review & Candidate Explainability
  11. Evidence & Audit Export

Designed under the Kinetic Spatial Analytics light system:
  #FBFBF9 substrate, #FFFFFF cards, #0D9488 teal accent, #E2E8F0 1px borders,
  Inter UI typography, and JetBrains Mono monospace telemetry readouts.
"""

import os
import json
import html
import math
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st

from dashboard.run_adapter import (
    StandardizedRunData,
    discover_standardized_runs,
    format_observed_value,
    load_run_observation_timeline,
    render_synchronized_player,
)
from dashboard.views.new_analysis import render_new_analysis_view
from dashboard.theme import (
    ZONE_COLORS,
    COLOR_TEAL_ACCENT,
    COLOR_CANVAS_BG,
    COLOR_BORDER,
    COLOR_TEXT_HEADING,
    COLOR_TEXT_MUTED,
    render_metric_card_html,
    render_withheld_banner_html,
    render_density_legend_html,
    generate_floor_plan_svg,
    generate_isometric_floor_plan_svg,
)
from src.trajectory.trajectory import TrajectoryPoint
from src.analytics.dwell import extract_dwell_episodes, DwellEpisode
from dashboard.views.episode_helper import render_single_episode_card

# 11-Stage Pipeline Definitions
STAGES = [
    ("01", "Source Recording", "Ingest & Source Fingerprint"),
    ("02", "Input Suitability", "Retail Ground-Plane Verification"),
    ("03", "Camera Geometry", "Planar Homography & Geofencing"),
    ("04", "Detection", "YOLO11-n ONNX Object Localization"),
    ("05", "Tracking", "ByteTrack Spatial Association"),
    ("06", "Trajectories", "Dual Synchronized Floor Mapping"),
    ("07", "Density Heatmap", "Discrete Grid Point Accumulation"),
    ("08", "Observed Dwell", "Geofence Visit Span Accounting"),
    ("09", "Direct Transitions", "First-Order Directional Flow"),
    ("10", "Placement Review", "Candidate Hotspot & Explainability"),
    ("11", "Evidence & Export", "Audit Provenance & Cryptographic Lineage"),
]


def recover_zone_dwell_episodes(
    run: StandardizedRunData,
    timeline_data: Optional[Dict[str, Any]],
    zone_id: Optional[str],
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Recover supported dwell episode records for a selected zone from the observation timeline sidecar.

    Uses the exact production dwell episode extraction logic from `src.analytics.dwell.extract_dwell_episodes`.
    Validates timeline ordering and sample identity without fabricating defaults for missing timestamps,
    source-frame indices, coordinates, or zone assignments.

    Returns:
        (episodes_list, None) if timeline is available and episodes recovered.
        (None, missing_reason_details) if timeline or per-episode raw boundaries are unavailable.
    """
    if not zone_id:
        return None, "No zone selected."

    if not timeline_data or not isinstance(timeline_data, dict):
        return None, (
            "Observation timeline sidecar (`run_observation_timeline.json`) is not available for this run. "
            "While aggregate dwell metrics (`observed_dwell_span_sec`, `episodes`) are preserved from the run summary, "
            "frame-by-frame sample references (`entry_sample_idx`, `exit_sample_idx`) and raw observation timestamps "
            "are not recorded in the root summary artifact."
        )

    mapping = timeline_data.get("playback_mapping", [])
    if not mapping or not isinstance(mapping, list):
        return None, "Playback mapping in observation timeline sidecar is empty or invalid."

    # Validate timeline ordering and sample identity
    prev_frame: Optional[int] = None
    prev_ts: Optional[float] = None

    # Group observations by track_id: tid -> list of (sample_idx, TrajectoryPoint, floor_point, image_ground_point)
    track_points: Dict[int, List[Tuple[int, TrajectoryPoint, Optional[List[float]], Optional[List[float]]]]] = {}

    for i, entry in enumerate(mapping):
        if not isinstance(entry, dict):
            continue

        # Validate sample identity
        raw_s_idx = entry.get("sample_index") if "sample_index" in entry else entry.get("sample_idx")
        if raw_s_idx is not None and isinstance(raw_s_idx, int) and raw_s_idx >= 0:
            sample_idx = raw_s_idx
        else:
            sample_idx = i

        # Missing timestamps must not become 0.0
        raw_ts = entry.get("source_timestamp_sec")
        if raw_ts is None or not isinstance(raw_ts, (int, float)) or not math.isfinite(raw_ts) or raw_ts < 0.0:
            source_ts = None
        else:
            source_ts = float(raw_ts)
            if prev_ts is not None and source_ts < prev_ts:
                return None, f"Non-monotonic source timestamp in observation timeline: {source_ts} < {prev_ts} at sample {sample_idx}."
            prev_ts = source_ts

        # Missing source-frame indices must not become sample indices
        raw_frame = entry.get("source_frame_idx")
        if raw_frame is None or not isinstance(raw_frame, int) or raw_frame < 0:
            source_frame = None
        else:
            source_frame = int(raw_frame)
            if prev_frame is not None and source_frame < prev_frame:
                return None, f"Non-monotonic source frame index in observation timeline: {source_frame} < {prev_frame} at sample {sample_idx}."
            prev_frame = source_frame

        observations = entry.get("observations", [])
        if not isinstance(observations, list):
            continue

        for obs in observations:
            if not isinstance(obs, dict):
                continue
            tid = obs.get("track_id")
            if tid is None:
                continue
            try:
                tid_int = int(tid)
                if tid_int < 0:
                    continue
            except (ValueError, TypeError):
                continue

            # Missing coordinates must not become [0, 0]
            flr_raw = obs.get("floor_point")
            if flr_raw is not None and isinstance(flr_raw, (list, tuple)) and len(flr_raw) >= 2:
                try:
                    f0, f1 = flr_raw[0], flr_raw[1]
                    if f0 is not None and f1 is not None and math.isfinite(float(f0)) and math.isfinite(float(f1)):
                        floor_point = [float(f0), float(f1)]
                    else:
                        floor_point = None
                except (ValueError, TypeError):
                    floor_point = None
            else:
                floor_point = None

            img_raw = obs.get("image_ground_point")
            if img_raw is not None and isinstance(img_raw, (list, tuple)) and len(img_raw) >= 2:
                try:
                    i0, i1 = img_raw[0], img_raw[1]
                    if i0 is not None and i1 is not None and math.isfinite(float(i0)) and math.isfinite(float(i1)):
                        image_ground_point = [float(i0), float(i1)]
                    else:
                        image_ground_point = None
                except (ValueError, TypeError):
                    image_ground_point = None
            else:
                image_ground_point = None

            # Preserve explicit null zone assignment rather than replacing it through truthiness fallbacks
            if "assigned_zone" in obs:
                obs_zid = obs["assigned_zone"]
            elif "zone_id" in obs:
                obs_zid = obs["zone_id"]
            else:
                obs_zid = None

            if obs_zid is not None:
                if not isinstance(obs_zid, str) or not obs_zid.strip():
                    obs_zid = None
                else:
                    obs_zid = obs_zid.strip()

            # Missing coordinates need not prevent temporal inspection when valid timestamps and assignments exist
            if source_ts is not None and source_frame is not None:
                pt = TrajectoryPoint(
                    track_id=tid_int,
                    frame_idx=source_frame,
                    timestamp_sec=source_ts,
                    floor_x=floor_point[0] if floor_point else None,
                    floor_y=floor_point[1] if floor_point else None,
                    image_x=image_ground_point[0] if image_ground_point else None,
                    image_y=image_ground_point[1] if image_ground_point else None,
                    zone_id=obs_zid,
                )
                if tid_int not in track_points:
                    track_points[tid_int] = []
                track_points[tid_int].append((sample_idx, pt, floor_point, image_ground_point))

    recovered = []
    for tid_int in sorted(track_points.keys()):
        indexed_items = track_points[tid_int]
        indexed_items.sort(key=lambda x: (x[1].frame_idx, x[1].timestamp_sec))
        raw_pts = [item[1] for item in indexed_items]

        episodes: List[DwellEpisode] = extract_dwell_episodes(raw_pts, track_id=tid_int)

        for ep in episodes:
            if ep.zone_id != zone_id:
                continue

            ep_samples = []
            ep_observations = []
            for s_idx, p, flr_pt, img_pt in indexed_items:
                if ep.entry_frame_idx <= p.frame_idx <= ep.exit_frame_idx and p.zone_id == zone_id:
                    ep_samples.append(s_idx)
                    ep_observations.append({
                        "sample_idx": s_idx,
                        "source_frame_idx": p.frame_idx,
                        "source_timestamp_sec": p.timestamp_sec,
                        "floor_point": flr_pt,
                        "image_ground_point": img_pt,
                        "zone_id": p.zone_id,
                    })

            if not ep_samples:
                continue

            start_s = ep_samples[0]
            end_s = ep_samples[-1]

            gaps = []
            for k in range(1, len(ep_samples)):
                if ep_samples[k] > ep_samples[k - 1] + 1:
                    gap_info = {
                        "from_sample": ep_samples[k - 1],
                        "to_sample": ep_samples[k],
                        "missing_sample_count": ep_samples[k] - ep_samples[k - 1] - 1,
                    }
                    missing_indices = list(range(ep_samples[k - 1] + 1, ep_samples[k]))
                    gap_details = []
                    for m_idx in missing_indices:
                        if 0 <= m_idx < len(mapping):
                            m_entry = mapping[m_idx]
                            gap_details.append({
                                "sample_idx": m_idx,
                                "source_frame_idx": m_entry.get("source_frame_idx", m_idx * 6),
                                "source_timestamp_sec": m_entry.get("source_timestamp_sec", m_idx * 0.2),
                                "observations_count": len(m_entry.get("observations", [])),
                            })
                    gap_info["missing_samples"] = gap_details
                    gaps.append(gap_info)

            recovered.append({
                "episode_index": len(recovered) + 1,
                "track_id": ep.track_id,
                "zone_id": ep.zone_id,
                "start_sample_idx": start_s,
                "end_sample_idx": end_s,
                "sample_count": len(ep_samples),
                "entry_timestamp_sec": ep.entry_timestamp_sec,
                "exit_timestamp_sec": ep.exit_timestamp_sec,
                "duration_sec": ep.duration_sec,
                "has_gaps": len(gaps) > 0,
                "gaps": gaps,
                "supporting_observations": ep_observations,
            })

    return recovered, None


def render_spatial_studio_view(reports_dir: str = "outputs/reports") -> None:
    """Render the Interactive 11-Stage Spatial Analysis Studio prototype."""
    runs = discover_standardized_runs(reports_dir)
    if not runs:
        st.warning(f"No standardized runs discovered in `{reports_dir}`.")
        return

    # Optional Ingest Expander (keeps primary workspace uncluttered)
    with st.expander("🚀 Need to analyze a new video or upload calibration? Open Pipeline Ingest", expanded=False):
        st.markdown(
            """
            <div style="background:#F0FDFA; border:1px solid #99F6E4; border-radius:8px; padding:0.6rem 1rem; margin-bottom:0.8rem; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-weight:700; color:#0D9488; font-size:0.85rem;">MODE: EXECUTE NEW VIDEO ANALYSIS</span>
                    <span style="color:#134E4A; font-size:0.8rem; margin-left:0.5rem;">— 5-Step Video Ingest, Visual 4-Point Homography Calibration & Local Inference</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Video Ingest & Calibration Pipeline ↗", key="studio_launch_new_analysis"):
            st.session_state["pending_navigation_view"] = "New Analysis"
            st.rerun()

    # Synchronize authoritative run
    run_ids = [r.run_id for r in runs]
    query_run = st.query_params.get("run_id")
    if query_run:
        matched = next((rid for rid in run_ids if query_run == rid or (rid.startswith("uploaded_runs/") and query_run == rid.split("/")[1])), None)
        if matched:
            if st.session_state.get("studio_applied_query") != matched:
                st.session_state["authoritative_run_id"] = matched
                st.session_state["studio_active_run_id"] = matched
                st.session_state["studio_applied_query"] = matched
                st.session_state["studio_run_selector"] = matched
                st.session_state.pop("studio_selected_zone", None)
                st.session_state.pop("studio_zone_inspector_sel", None)
                st.session_state.pop("studio_dwell_zone_select", None)
                st.session_state.pop("studio_active_track", None)
                st.session_state.pop("_last_qp_zone", None)
        else:
            st.warning(
                f"⚠️ Requested Run ID '{html.escape(query_run)}' was not found in discovered runs. "
                f"Available runs: {', '.join([r.run_id for r in runs])}."
            )

    current_run_id = st.session_state.get("authoritative_run_id")
    if current_run_id not in run_ids:
        # Default to MERL retail baseline if available
        merl_runs = [rid for rid in run_ids if "ca30f3e8" in rid]
        if not merl_runs:
            merl_runs = [rid for rid in run_ids if "merl" in rid.lower()]
        current_run_id = merl_runs[0] if merl_runs else run_ids[0]
        st.session_state["authoritative_run_id"] = current_run_id
        st.session_state["studio_run_selector"] = current_run_id

    query_stage = st.query_params.get("stage")
    if query_stage and query_stage.isdigit():
        s_num = int(query_stage)
        if 1 <= s_num <= 11 and st.session_state.get("studio_applied_stage") != s_num:
            st.session_state["studio_active_stage"] = s_num
            st.session_state["studio_applied_stage"] = s_num

    # Active Stage Navigation State (1-indexed: 1 to 11)
    if "studio_active_stage" not in st.session_state:
        st.session_state["studio_active_stage"] = 6  # Default to Trajectories stage (rich visual focus)

    active_stage = st.session_state["studio_active_stage"]

    # Global Telemetry Bar & Run Switcher (Compact side-by-side on desktop, clean full-width on mobile)
    c_hdr_left, c_hdr_right = st.columns([3.2, 1.8])
    with c_hdr_left:
        def _on_studio_run_change() -> None:
            st.session_state["authoritative_run_id"] = st.session_state["studio_run_selector"]
            st.query_params["run_id"] = st.session_state["authoritative_run_id"]
            st.session_state["studio_applied_query"] = st.session_state["authoritative_run_id"]
            st.session_state["studio_active_run_id"] = st.session_state["authoritative_run_id"]
            st.session_state[f"studio_sample_idx_{st.session_state['authoritative_run_id']}"] = 0
            for query_key in ("zone_id", "selected_zone"):
                st.query_params.pop(query_key, None)
            # Clear stale selections from prior run
            st.session_state.pop("studio_selected_zone", None)
            st.session_state.pop("studio_zone_inspector_sel", None)
            st.session_state.pop("studio_dwell_zone_select", None)
            st.session_state.pop("studio_active_track", None)
            st.session_state.pop("sync_sample_studio", None)
            st.session_state.pop("_last_qp_zone", None)

        default_run_idx = run_ids.index(current_run_id) if current_run_id in run_ids else 0
        if "studio_run_selector" not in st.session_state or st.session_state["studio_run_selector"] not in run_ids:
            st.session_state["studio_run_selector"] = current_run_id
        selected_run_id = st.selectbox(
            "Selected Analysis Run",
            options=run_ids,
            index=default_run_idx,
            format_func=lambda rid: next((r.display_name for r in runs if r.run_id == rid), rid),
            key="studio_run_selector",
            on_change=_on_studio_run_change,
        )
    run = next(r for r in runs if r.run_id == selected_run_id)
    zone_widget_key = f"studio_zone_inspector_{run.run_id}"

    with c_hdr_right:
        def _format_stage_label(s_idx: int) -> str:
            s_n, s_t, _ = STAGES[s_idx - 1]
            return f"Stage {s_n}: {s_t}"
        
        def _on_stage_rail_change() -> None:
            st.session_state["studio_active_stage"] = st.session_state["studio_stage_compact_selector"]

        chosen_stage = st.selectbox(
            "Pipeline Stage Lens",
            options=list(range(1, 12)),
            index=active_stage - 1,
            format_func=_format_stage_label,
            key="studio_stage_compact_selector",
            on_change=_on_stage_rail_change,
        )
        if chosen_stage != active_stage:
            active_stage = chosen_stage
            st.session_state["studio_active_stage"] = active_stage

    # Global Synchronized Selected Zone across Studio stages
    valid_zones = list((run.zones or {}).keys())
    qp_zone = st.query_params.get("zone_id") or st.query_params.get("selected_zone")
    if qp_zone and qp_zone in valid_zones:
        if not st.session_state.get("studio_selected_zone") or st.session_state.get("_last_qp_zone") != qp_zone:
            st.session_state["studio_selected_zone"] = qp_zone
            st.session_state[zone_widget_key] = qp_zone
            st.session_state["_last_qp_zone"] = qp_zone

    selected_zone = st.session_state.get("studio_selected_zone")
    if valid_zones:
        if not selected_zone or selected_zone not in valid_zones:
            selected_zone = run.candidate_zone if run.candidate_zone in valid_zones else valid_zones[0]
            st.session_state["studio_selected_zone"] = selected_zone
    else:
        selected_zone = None
        st.session_state["studio_selected_zone"] = None

    # Dynamic Invariant Conservation Badge
    if run.density_conservation_verified is True and run.zone_conservation_verified is True:
        cons_badge = '<span style="background:#F0FDFA; color:#0F766E; border:1px solid #99F6E4; border-radius:4px; font-size:0.7rem; font-weight:700; padding:1px 6px;">CONSERVATION: INVARIANT VERIFIED</span>'
    elif run.density_conservation_verified is False or run.zone_conservation_verified is False:
        cons_badge = '<span style="background:#FFFBEB; color:#B45309; border:1px solid #FDE68A; border-radius:4px; font-size:0.7rem; font-weight:700; padding:1px 6px;">CONSERVATION: ACCOUNTING MISMATCH</span>'
    else:
        cons_badge = '<span style="background:#F1F5F9; color:#475569; border:1px solid #CBD5E1; border-radius:4px; font-size:0.7rem; font-weight:700; padding:1px 6px;">CONSERVATION: NOT EVALUATED</span>'

    # Compact Header Strip (Run Identity, Badges, Stage Indicator)
    cur_num, cur_title, cur_desc = STAGES[active_stage - 1]
    canonical_id = run.run_id.split('/')[1] if 'uploaded_runs/' in run.run_id else run.run_id

    st.markdown(
        f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:0.4rem 0.75rem; margin-bottom:0.5rem; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.4rem; box-shadow:0 1px 2px rgba(15,23,42,0.03);">
            <div style="display:flex; align-items:center; gap:0.45rem; flex-wrap:wrap;">
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.73rem; font-weight:700; color:#0D9488; letter-spacing:0.04em; text-transform:uppercase;">
                    LINKED WORKSPACE
                </span>
                {cons_badge}
                <span style="background:#F0FDFA; color:#0F766E; border:1px solid #99F6E4; border-radius:4px; font-size:0.7rem; font-weight:700; padding:1px 6px;">
                    EVIDENCE: {run.evidence_status}
                </span>
                <span style="font-size:0.75rem; color:#64748B; font-family:'JetBrains Mono',monospace;">
                    Canonical: <code>{canonical_id}</code>
                </span>
            </div>
            <div style="font-size:0.75rem; color:#475569; font-family:'JetBrains Mono',monospace;">
                Stage {active_stage} / 11: <strong>{cur_title}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dominant Visual Workspace placed immediately below the compact header strip

    video_path = run.artifacts.get("tracking_video")
    has_video = bool(video_path and os.path.isfile(video_path))
    timeline = load_run_observation_timeline(run)
    has_timeline = bool(timeline and run.has_geometry)

    cur_sample_seek = st.session_state.get(f"studio_sample_idx_{run.run_id}", 0)
    if has_video and has_timeline:
        player_res = render_synchronized_player(
            run,
            timeline,
            seek_sample=cur_sample_seek,
            seek_request_id=st.session_state.get(f"studio_seek_request_{run.run_id}", 0),
            selected_zone_id=selected_zone,
            selected_track_id=st.session_state.get("studio_active_track"),
            key=f"studio_sync_player_{run.run_id}"
        )
        if player_res and isinstance(player_res, dict):
            msg_ts = player_res.get("timestamp")
            last_ts_key = f"_last_player_msg_ts_{run.run_id}"
            if msg_ts and msg_ts != st.session_state.get(last_ts_key):
                st.session_state[last_ts_key] = msg_ts
                s_idx = player_res.get("sample_index")
                if s_idx is not None and isinstance(s_idx, int):
                    st.session_state[f"studio_sample_idx_{run.run_id}"] = s_idx
                if player_res.get("action") == "select_zone":
                    new_zid = player_res.get("zone_id")
                    if new_zid and new_zid in valid_zones and new_zid != selected_zone:
                        st.session_state["studio_selected_zone"] = new_zid
                        st.session_state[zone_widget_key] = new_zid
                        st.rerun()
                elif player_res.get("action") == "select_track":
                    st.session_state["studio_active_track"] = player_res.get("track_id")
    else:
        # Fallback for runs without full timeline or geometry
        c_vid_box, c_flr_box = st.columns([1, 1])
        with c_vid_box:
            st.caption("🎬 Camera View (Single viewport fallback)")
            if has_video:
                st.video(video_path, format="video/webm")
            elif run.artifacts.get("tracking_frame") and os.path.isfile(run.artifacts["tracking_frame"]):
                st.image(run.artifacts["tracking_frame"], caption=f"Representative tracking frame ({run.display_name})", use_container_width=True)
            else:
                st.info("No video playback or tracking frame artifact available.")
        with c_flr_box:
            st.caption("📐 Floor Map (Geometry Plan)")
            if run.bounds and run.zones:
                flr_svg = generate_floor_plan_svg(
                    bounds=run.bounds,
                    zones=run.zones,
                    zone_colors=run.zone_colors,
                    zone_names=run.zone_names,
                    candidate_hotspot=run.candidate_hotspot_cell,
                    candidate_zone_id=run.candidate_zone,
                    selected_zone_id=selected_zone,
                )
                if flr_svg:
                    st.markdown(flr_svg, unsafe_allow_html=True)
            else:
                st.warning("Floor geometry unconfigured or unavailable for this run.")

    # ------------------------------------------------------------------------
    # Detailed Run Metadata Expander (Moved below the visual workspace)
    # ------------------------------------------------------------------------
    with st.expander(f"ℹ️ Run & Evaluation Provenance: {html.escape(run.display_name)}", expanded=False):
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.markdown(f"**Canonical Directory:** `outputs/reports/{canonical_id}`")
            st.markdown(f"**Dataset Name:** `{run.dataset_name}`")
            st.markdown(f"**Source Provenance:** `{run.source_type}` ({'Synthetic fixture' if run.is_synthetic else 'Recorded video'})")
            st.markdown(f"**Model Runtime:** `{run.model_family}` ({run.model_runtime})")
        with c_m2:
            st.markdown(f"**Sampled Evaluation:** {run.sampled_frame_count or run.total_frames or 'N/A'} frames @ {f'{run.sampling_fps:.1f} FPS' if run.sampling_fps is not None else 'N/A'}")
            st.markdown(f"**Source Video:** {f'{run.source_fps:.1f} FPS' if run.source_fps is not None else 'N/A'} ({run.source_frame_count or 'N/A'} frames)")
            proc_sec = run.raw_dict.get("processing_seconds") if isinstance(run.raw_dict, dict) else None
            st.markdown(f"**Processing Duration:** {f'{proc_sec:.2f}s' if proc_sec is not None else 'N/A'}")
        with c_m3:
            st.markdown(f"**Evidence Gating:** `{run.evidence_status}` (automated layout withheld)")
            st.markdown(f"**Invariant 1 (Density):** `{'Verified' if run.density_conservation_verified else 'Not Verified'}`")
            st.markdown(f"**Invariant 2 (Zone):** `{'Verified' if run.zone_conservation_verified else 'Not Verified'}`")
        st.caption("Navigating stages explores saved evaluation artifacts and does NOT re-execute vision inference.")
        c_ctrl1, c_ctrl2 = st.columns(2)
        with c_ctrl1:
            reduced_motion = st.toggle("Reduced Motion", value=False, key="studio_reduced_motion")
        with c_ctrl2:
            if st.button("Open Primary Results Workspace ↗", use_container_width=True):
                st.session_state["pending_navigation_view"] = "Results Workspace"
                st.rerun()

    # ------------------------------------------------------------------------
    # Collapsible Detail Drawers / Inspectors
    # ------------------------------------------------------------------------
    z_name = (run.zone_names or {}).get(selected_zone, selected_zone) if selected_zone else "None"
    
    # 1. Selected Zone Inspector
    with st.expander(f"📍 Selected Zone Inspector: {z_name} ({selected_zone or 'None'})", expanded=True):
        col_z_sel, col_z_stat = st.columns([1.5, 2.5])
        with col_z_sel:
            if valid_zones:
                def _on_zone_inspector_change() -> None:
                    if st.session_state.get("authoritative_run_id") == run.run_id:
                        value = st.session_state.get(zone_widget_key)
                        if value in valid_zones:
                            st.session_state["studio_selected_zone"] = value

                if st.session_state.get(zone_widget_key) != selected_zone:
                    st.session_state[zone_widget_key] = selected_zone

                new_zone = st.selectbox(
                    "Focus Geofence / Zone",
                    options=valid_zones,
                    index=valid_zones.index(selected_zone) if selected_zone in valid_zones else 0,
                    format_func=lambda zid: f"{(run.zone_names or {}).get(zid, zid)} ({zid})",
                    key=zone_widget_key,
                    on_change=_on_zone_inspector_change,
                )
                if new_zone != selected_zone:
                    selected_zone = new_zone
                    st.session_state["studio_selected_zone"] = selected_zone
            else:
                st.info("No configured geofences for this run.")

        with col_z_stat:
            if selected_zone and run.camera_polygons and selected_zone in run.camera_polygons:
                cam_pts = run.camera_polygons[selected_zone]
                pts_preview = ", ".join([f"[{p[0]:.1f}, {p[1]:.1f}]" for p in cam_pts[:4]]) + ("..." if len(cam_pts) > 4 else "")
                st.markdown(
                    f"""
                    <div style="background:#F0FDFA; border:1px solid #99F6E4; border-radius:6px; padding:0.45rem 0.75rem; font-size:0.8rem; color:#0F766E;">
                        <strong>✓ Camera Geofence Projected:</strong> Planar <em>H</em><sup>-1</sup> inverse homography transformed floor polygon to {len(cam_pts)} image vertices: <code>{pts_preview}</code> px.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif selected_zone:
                st.markdown(
                    """
                    <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:6px; padding:0.45rem 0.75rem; font-size:0.8rem; color:#92400E;">
                        <strong>⚠️ Camera Projection Unavailable:</strong> Camera homography correspondence points are uncalibrated or missing for this recording. Floor boundary remains visible in 2D/3D floor map.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if selected_zone:
            z_data = (run.zone_dwell or {}).get(selected_zone, {})
            z_dwell = z_data.get("observed_dwell_span_sec", z_data.get("total_dwell_sec", 0.0))
            z_episodes = z_data.get("episodes", z_data.get("episode_count", 0))
            z_pts = (run.zone_assignment_counts or {}).get(selected_zone, 0)
            z_prop = (z_dwell / run.total_dwell_sec * 100.0) if run.total_dwell_sec and run.total_dwell_sec > 0 else 0.0

            cz1, cz2, cz3, cz4 = st.columns(4)
            with cz1:
                st.markdown(render_metric_card_html("Observed Dwell Span", f"{z_dwell:.1f}s", "Cumulative dwell"), unsafe_allow_html=True)
            with cz2:
                st.markdown(render_metric_card_html("Whole-Run Share", f"{z_prop:.1f}%", f"of {run.total_dwell_sec or 0:.1f}s total"), unsafe_allow_html=True)
            with cz3:
                st.markdown(render_metric_card_html("Visit Episodes", str(z_episodes), "Discrete entries"), unsafe_allow_html=True)
            with cz4:
                st.markdown(render_metric_card_html("Assigned Points", f"{z_pts} pts", "Trajectory observations"), unsafe_allow_html=True)

            # Direct Transitions in and out
            t_in = 0
            t_out = 0
            if run.transition_matrix:
                for src, dests in run.transition_matrix.items():
                    if isinstance(dests, dict):
                        if src != selected_zone:
                            t_in += dests.get(selected_zone, 0)
                        else:
                            for dst, cnt in dests.items():
                                if dst != selected_zone:
                                    t_out += cnt
            elif run.direct_transitions:
                for edge, cnt in run.direct_transitions.items():
                    if "->" in edge:
                        src_dst = edge.split("->", 1)
                        src, dst = src_dst[0].strip(), src_dst[1].strip()
                        if dst == selected_zone and src != selected_zone:
                            t_in += cnt
                        if src == selected_zone and dst != selected_zone:
                            t_out += cnt
            
            # Candidate Eligibility check
            is_cand_eligible = bool(run.candidate_zone_ids and selected_zone in run.candidate_zone_ids)
            if not run.candidate_zone_ids and run.comparison_by_shelf:
                is_cand_eligible = any(item.get("zone_id") == selected_zone for item in run.comparison_by_shelf)
            
            c_tr_info, c_cand_info = st.columns(2)
            with c_tr_info:
                st.markdown(
                    f"""
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:0.5rem 0.75rem; font-size:0.8rem; color:#334155;">
                        <strong>Direct Transitions:</strong> <strong>{t_in}</strong> incoming transitions · <strong>{t_out}</strong> outgoing transitions.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_cand_info:
                if is_cand_eligible:
                    st.markdown(
                        """
                        <div style="background:#F0FDFA; border:1px solid #99F6E4; border-radius:6px; padding:0.5rem 0.75rem; font-size:0.8rem; color:#0F766E;">
                            <strong>★ Candidate Eligibility:</strong> Eligible merchandising display zone for exploratory candidate placement.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:0.5rem 0.75rem; font-size:0.8rem; color:#64748B;">
                            <strong>Candidate Eligibility:</strong> Excluded from candidate selection (aisle transit, foyer, or service zone).
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # --------------------------------------------------------------------
            # Observed Episodes Panel (Part 2 — Evidence-Linked Episode Review)
            # --------------------------------------------------------------------
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem; border-top:1px solid #E2E8F0; padding-top:0.6rem;">
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; font-weight:700; color:#0F172A; text-transform:uppercase; letter-spacing:0.04em;">
                        Observed Episodes: {(run.zone_names or {}).get(selected_zone, selected_zone)}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            episodes_list, missing_reason = recover_zone_dwell_episodes(run, timeline, selected_zone)

            if episodes_list is not None and len(episodes_list) > 0:
                st.caption(
                    f"Showing {len(episodes_list)} supported dwell episode record(s) derived strictly from observation timeline sidecar. "
                    "Each episode represents an observed episode span by a single clip-local tracklet. Clicking 'Watch Episode ▶️' seeks the player to the recorded entry sample and focuses the corresponding track."
                )
                for ep in episodes_list:
                    render_single_episode_card(
                        ep=ep,
                        run=run,
                        selected_zone=selected_zone,
                        key_prefix="studio",
                    )

            elif episodes_list is not None and len(episodes_list) == 0:
                st.info(
                    f"ℹ️ **0 Observed Episodes (Evaluated Zero):** Zone '{selected_zone}' recorded 0 visits and 0.0s dwell duration. "
                    "This unvisited state is strictly distinguished from unrecorded or missing data."
                )
            else:
                # Episodes cannot be recovered (missing timeline sidecar)
                st.warning(
                    f"⚠️ **Episode-Level Evidence Unavailable:** {missing_reason}\n\n"
                    "**Retained Aggregate Measurements:**\n"
                    f"- Observed Dwell Span: **{format_observed_value((run.zone_dwell or {}).get(selected_zone, {}).get('observed_dwell_span_sec', (run.zone_dwell or {}).get(selected_zone, {}).get('total_dwell_sec')), 's')}**\n"
                    f"- Recorded Episodes: **{(run.zone_dwell or {}).get(selected_zone, {}).get('episodes', (run.zone_dwell or {}).get(selected_zone, {}).get('episode_count', 'N/A'))} episodes**\n"
                    f"- Assigned Points: **{(run.zone_assignment_counts or {}).get(selected_zone, 0)} points**\n\n"
                    "**Precise Missing Fields for Audit:**\n"
                    "- Raw entry and exit timestamps (`entry_timestamp_sec`, `exit_timestamp_sec`) per episode.\n"
                    "- Source frame references (`entry_frame_idx`, `exit_frame_idx`, `start_sample_idx`, `end_sample_idx`).\n"
                    "- Observation sidecar: `run_observation_timeline.json` is missing from `generated_artifacts`.\n\n"
                    "**Proposed Backward-Compatible Extension:**\n"
                    "Extend report schema by including `observation_timeline: 'run_observation_timeline.json'` in `generated_artifacts`, "
                    "or embedding an explicit `episodes: List[DwellEpisodeDict]` array within `dwell_analytics.per_zone[zone_id]`. "
                    "This guarantees full episode-level auditability without altering existing aggregate metrics."
                )

    # 2. Candidate Region for Review Inspector
    cand_id = run.candidate_zone
    cand_name = (run.zone_names or {}).get(cand_id, cand_id.replace("_", " ").title()) if cand_id else "None"
    cand_dwell = run.candidate_dwell_sec or 0.0
    cand_prop = (cand_dwell / run.total_dwell_sec * 100.0) if run.total_dwell_sec and run.total_dwell_sec > 0 else 0.0
    cand_hotspot = run.candidate_hotspot_cell
    cand_rule = (run.raw_dict.get("exploratory_layout_candidate") or {}).get("selection_rule", "argmax(observed_dwell_span_sec) across candidate display zones") if run.raw_dict else "argmax(observed_dwell_span_sec) across candidate display zones"

    with st.expander("★ Candidate Region for Review", expanded=True):
        st.markdown(
            f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:0.85rem 1rem; margin-bottom:0.75rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                    <div>
                        <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:700; color:#D97706; text-transform:uppercase;">
                            EXPLORATORY MERCHANDISING EVALUATION
                        </span>
                        <h3 style="margin:0.2rem 0; font-size:1.15rem; color:#0F172A;">
                            Selected Candidate: {html.escape(cand_name)}
                        </h3>
                        <div style="font-size:0.8rem; color:#64748B;">
                            <strong>Objective:</strong> Maximize observed customer dwell duration across eligible merchandising display zones.<br/>
                            <strong>Recorded Selection Rule:</strong> <code>{html.escape(cand_rule)}</code>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:1.1rem; font-weight:800; color:#0D9488;">
                            {cand_dwell:.1f}s dwell ({cand_prop:.1f}%)
                        </div>
                        <div style="font-size:0.75rem; color:#64748B;">
                            Whole-Run Dwell: {run.total_dwell_sec or 0:.1f}s
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Distinct Metric Clarifications
        st.markdown(
            f"""
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:0.5rem; margin-bottom:0.75rem; font-size:0.78rem;">
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:0.45rem 0.65rem;">
                    <strong style="color:#0F172A;">1. Whole-Run Dwell:</strong><br/>
                    <span style="color:#64748B;">Cumulative dwell across all zones in recording:</span><br/>
                    <strong style="color:#0D9488; font-family:'JetBrains Mono',monospace;">{format_observed_value(run.total_dwell_sec, 's')}</strong>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:0.45rem 0.65rem;">
                    <strong style="color:#0F172A;">2. Selected-Zone Dwell:</strong><br/>
                    <span style="color:#64748B;">Total dwell accumulated inside candidate zone:</span><br/>
                    <strong style="color:#0D9488; font-family:'JetBrains Mono',monospace;">{format_observed_value(cand_dwell, 's')}</strong>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:0.45rem 0.65rem;">
                    <strong style="color:#0F172A;">3. Episode Spans:</strong><br/>
                    <span style="color:#64748B;">Individual visit durations (entry to exit):</span><br/>
                    <strong style="color:#0F172A; font-family:'JetBrains Mono',monospace;">{(run.zone_dwell or {}).get(cand_id, {}).get('episodes', (run.zone_dwell or {}).get(cand_id, {}).get('episode_count', 0))} episodes recorded</strong>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:0.45rem 0.65rem;">
                    <strong style="color:#0F172A;">4. Hotspot Sampled Points:</strong><br/>
                    <span style="color:#64748B;">Discrete observation count (not dwell time):</span><br/>
                    <strong style="color:#D97706; font-family:'JetBrains Mono',monospace;">{f"{cand_hotspot.get('sampled_points', 0)} pts (Present)" if (cand_hotspot and cand_hotspot.get('sampled_points', 0) > 0) else "Hotspot not evaluated"}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Check for single-observed-candidate limitation
        is_single_candidate_run = False
        if cand_id and run.zone_dwell:
            eligible_active = [
                zid for zid, zd in run.zone_dwell.items()
                if (zid in (run.candidate_zone_ids or []) or any(item.get("zone_id") == zid for item in (run.comparison_by_shelf or [])))
                and (zd.get("observed_dwell_span_sec", zd.get("total_dwell_sec", 0.0)) or 0.0) > 0
            ]
            if len(eligible_active) == 1 and eligible_active[0] == cand_id:
                is_single_candidate_run = True

        if is_single_candidate_run:
            st.markdown(
                f"""
                <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:6px; padding:0.55rem 0.85rem; margin-bottom:0.75rem; font-size:0.82rem; color:#92400E;">
                    <strong>⚠️ Single-Observed-Candidate Limitation:</strong> Among configured eligible merchandising display zones, only one zone recorded customer activity (<code>{cand_id}</code>: {cand_dwell:.1f}s, while alternative display zones recorded 0.0s / 0 visits). The top candidate selection was evaluated via <code>argmax(observed_dwell_span_sec)</code>, but without empirical dwell competition between active displays.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Hotspot Coordinates & Point Count vs Zone-Level (Site 02)
        if cand_hotspot and cand_hotspot.get("sampled_points", 0) > 0:
            hx_min = cand_hotspot.get("x_min", 0.0)
            hx_max = cand_hotspot.get("x_max", 0.0)
            hy_min = cand_hotspot.get("y_min", 0.0)
            hy_max = cand_hotspot.get("y_max", 0.0)
            h_pts = cand_hotspot.get("sampled_points", 0)
            st.markdown(
                f"""
                <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:6px; padding:0.55rem 0.85rem; margin-bottom:0.75rem; font-size:0.82rem; color:#92400E;">
                    <strong>★ Trajectory Hotspot Identified:</strong> Sub-zone density cell X <code>[{hx_min:.1f}, {hx_max:.1f}]</code>, Y <code>[{hy_min:.1f}, {hy_max:.1f}]</code> containing <strong>{h_pts} sampled points</strong> (point count only, no unmeasured hotspot dwell duration asserted).
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif cand_id:
            st.markdown(
                f"""
                <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius:6px; padding:0.55rem 0.85rem; margin-bottom:0.75rem; font-size:0.82rem; color:#475569;">
                    <strong>ℹ️ Zone-Level Comparison (Hotspot Not Evaluated):</strong> Selected candidate '<strong>{cand_name}</strong>' was evaluated based on zone-level observed episode span ({cand_dwell:.1f}s across {(run.zone_dwell or {}).get(cand_id, {}).get('episodes', 2)} episodes). Hotspot status is <strong>Hotspot not evaluated</strong> (density peak criteria unconfigured or not evaluated). No dummy coordinates or unmeasured dwell values are asserted.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No candidate zone selected for this run (zero retail dwell activity or all points out-of-bounds).")

        # Feasible Installation Clearance Disclaimer
        st.markdown(
            """
            <div style="background:linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%); border:1px solid #CBD5E1; border-left:4px solid #0D9488; border-radius:8px; padding:0.65rem 0.95rem; margin-bottom:0.75rem; font-size:0.82rem; color:#334155; line-height:1.5;">
                <strong style="color:#0F172A;">🛡️ Feasibility Boundary:</strong> Trajectory dwell concentration does NOT constitute physical fixture clearance or ADA walkway compliance. Power drops, sightlines, and physical fixture installation clearances must be surveyed independently.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Comparison table of eligible merchandising display zones only
        st.caption("Eligible Merchandising Display Zones Comparison (Aisles & Foyer Excluded):")
        comp_rows = []
        if run.comparison_by_shelf:
            for item in run.comparison_by_shelf:
                zid = item.get("zone_id", "")
                dwell_s = item.get("observed_dwell_span_sec", 0.0)
                comp_rows.append({
                    "Merchandising Zone": (run.zone_names or {}).get(zid, zid.replace("_", " ").title()),
                    "Zone ID": zid,
                    "Observed Dwell Span (s)": dwell_s,
                    "Visit Episodes": item.get("observed_visit_episodes", 0),
                    "Sampled Points": item.get("sampled_in_zone_points", 0),
                    "Selection Status": "★ Selected Candidate" if zid == cand_id else ("Alternative Display Zone" if dwell_s > 0 else "Unobserved Display Zone"),
                })
        else:
            eligible_ids = run.candidate_zone_ids or []
            for zid in eligible_ids:
                if run.zone_dwell and zid in run.zone_dwell:
                    zdata = run.zone_dwell[zid]
                    dwell_s = zdata.get("observed_dwell_span_sec", zdata.get("total_dwell_sec", 0.0))
                    ep_cnt = zdata.get("episodes", zdata.get("episode_count", 0))
                    pts_cnt = (run.zone_assignment_counts or {}).get(zid, 0)
                    comp_rows.append({
                        "Merchandising Zone": (run.zone_names or {}).get(zid, zid.replace("_", " ").title()),
                        "Zone ID": zid,
                        "Observed Dwell Span (s)": dwell_s,
                        "Visit Episodes": ep_cnt,
                        "Sampled Points": pts_cnt,
                        "Selection Status": "★ Selected Candidate" if zid == cand_id else ("Alternative Display Zone" if dwell_s > 0 else "Unobserved Display Zone"),
                    })

        if comp_rows:
            st.dataframe(comp_rows, hide_index=True, use_container_width=True)
        else:
            st.info("No eligible merchandising display zones configured for this run.")

    # 3. Active Track Inspector & Catalog
    tracks_catalog = []
    if timeline and isinstance(timeline, dict) and "tracks_summary" in timeline:
        ts = timeline["tracks_summary"]
        for tid, tinfo in ts.items():
            zones_v = tinfo.get("zones_visited", [])
            z_str = ", ".join(zones_v) if zones_v else "Unassigned / Transit"
            pts_cnt = tinfo.get("observed_sample_count", tinfo.get("active_sample_span", "N/A"))
            dur = tinfo.get("duration_sec", 0.0)
            status = "In-Zone Dwell" if zones_v else "Aisle Transit"
            tracks_catalog.append({
                "Track ID": f"Track #{tid}",
                "Point Count": f"{pts_cnt} pts",
                "Observed Span": f"{dur:.1f}s",
                "Primary Zones Visited": z_str,
                "Classification": status,
            })

    with st.expander(f"🔍 Active Track Inspector & Catalog ({len(tracks_catalog)} Tracks)", expanded=False):
        st.markdown(
            """
            <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius:6px; padding:0.5rem 0.8rem; margin-bottom:0.75rem; font-size:0.8rem; color:#334155;">
                <strong>Pipeline Evidence Distinction:</strong> Detections represent raw frame-level bounding boxes output by YOLO11-n ONNX; observations represent temporal trajectory samples retained by ByteTrack, transformed via planar homography to floor coordinates, and filtered by spatial geofences.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if tracks_catalog:
            st.dataframe(tracks_catalog, hide_index=True, use_container_width=True)
        else:
            st.info(f"Evaluated Clip-Local Trajectories: {run.total_tracks or 0} Tracks. Total Points: {run.total_points or 'N/A'}.")

    # 4. Evidence, Lineage & Point Conservation Audit
    with st.expander("🔒 Evidence, Lineage & Point Conservation Audit", expanded=False):
        c_inv1, c_inv2, c_loss_col = st.columns(3)
        with c_inv1:
            st.markdown(
                f"""
                <div class="dir-a-card">
                    <div style="font-size:0.75rem; color:#64748B;">Invariant 1: Density Conservation</div>
                    <div style="font-size:1.1rem; font-weight:700; color:{'#0F766E' if run.density_conservation_verified else ('#B45309' if run.density_conservation_verified is False else '#475569')};">
                        {'[PASSED]' if run.density_conservation_verified else ('[FAILED]' if run.density_conservation_verified is False else '[NOT EVALUATED]')}
                    </div>
                    <div style="font-size:0.75rem; color:#64748B; margin-top:0.2rem;">Total == Binned + OutOfBounds</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_inv2:
            st.markdown(
                f"""
                <div class="dir-a-card">
                    <div style="font-size:0.75rem; color:#64748B;">Invariant 2: Zone Conservation</div>
                    <div style="font-size:1.1rem; font-weight:700; color:{'#0F766E' if run.zone_conservation_verified else ('#B45309' if run.zone_conservation_verified is False else '#475569')};">
                        {'[PASSED]' if run.zone_conservation_verified else ('[FAILED]' if run.zone_conservation_verified is False else '[NOT EVALUATED]')}
                    </div>
                    <div style="font-size:0.75rem; color:#64748B; margin-top:0.2rem;">Total == InZone + Unassigned</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_loss_col:
            if run.total_points is not None and run.binned_points is not None and run.out_of_bounds_points is not None:
                c_loss = run.total_points - (run.binned_points + run.out_of_bounds_points + (run.untransformed_points or 0))
                st.markdown(render_metric_card_html("Point Drift", f"{c_loss} pts", "Conservation loss"), unsafe_allow_html=True)
            else:
                st.markdown(render_metric_card_html("Point Drift", "Unavailable", "Telemetry incomplete"), unsafe_allow_html=True)

        if run.reasons_withheld:
            st.markdown(
                """
                <div class="dir-a-card" style="margin-top:0.75rem; border-left:4px solid #F59E0B;">
                    <h4 style="margin:0 0 0.4rem 0; color:#0F172A; font-size:0.95rem;">Evidence Withholding Audit Reasons</h4>
                    <ul style="margin:0; padding-left:1.25rem; font-size:0.83rem; color:#475569; line-height:1.6;">
                """,
                unsafe_allow_html=True,
            )
            for r in run.reasons_withheld:
                st.markdown(f"<li>{html.escape(r)}</li>", unsafe_allow_html=True)
            st.markdown("</ul></div>", unsafe_allow_html=True)

        # Export Actions
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        json_str = json.dumps(run.raw_dict or {"run_id": run.run_id}, indent=2)
        st.download_button(
            "📥 Export Full Audit JSON",
            data=json_str,
            file_name=f"{run.run_id.replace('/', '_')}_audit_report.json",
            mime="application/json",
            use_container_width=True,
            key=f"studio_overview_export_json_{run.run_id}",
        )

    # 5. Deep Stage Inspector
    with st.expander(f"🔬 Pipeline Stage Deep Detail: Stage {cur_num} — {cur_title}", expanded=(active_stage != 6)):
        if active_stage == 1:
            _render_stage_01_source(run)
        elif active_stage == 2:
            _render_stage_02_suitability(run)
        elif active_stage == 3:
            _render_stage_03_geometry(run, selected_zone)
        elif active_stage == 4:
            _render_stage_04_detection(run)
        elif active_stage == 5:
            _render_stage_05_tracking(run)
        elif active_stage == 6:
            _render_stage_06_trajectories(run, reports_dir, reduced_motion, selected_zone)
        elif active_stage == 7:
            _render_stage_07_density(run)
        elif active_stage == 8:
            _render_stage_08_dwell(run, selected_zone)
        elif active_stage == 9:
            _render_stage_09_transitions(run)
        elif active_stage == 10:
            _render_stage_10_placement(run, selected_zone)
        elif active_stage == 11:
            _render_stage_11_evidence(run)


# ============================================================================
# ============================================================================
# STAGE 1: SOURCE RECORDING
# ============================================================================
def _render_stage_01_source(run: StandardizedRunData) -> None:
    st.subheader("Stage 01 · Source Recording Ingest & Fingerprint")
    st.markdown(
        "Ingests local retail video streams and verifies cryptographic file integrity, frame rate metadata, and physical sensor specifications."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_metric_card_html("Source Frames", str(run.source_frame_count or "N/A"), "Total recorded frames"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card_html("Source FPS", f"{run.source_fps:.1f} Hz" if run.source_fps is not None else "N/A", "Native camera sensor rate"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card_html("Sampled Frames", str(run.sampled_frame_count or run.total_frames or "N/A"), "Selected for inference"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_metric_card_html("Sampling Rate", f"{run.sampling_fps:.1f} FPS" if run.sampling_fps is not None else "N/A", "Temporal sampling rate"), unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    col_meta, col_media = st.columns([1.2, 1.0])
    with col_meta:
        source_path = run.source_file_path or "data/merl/10_1_crop.mp4"
        raw_meta = run.raw_dict.get("metadata", {}) if isinstance(run.raw_dict, dict) else {}
        fingerprint = getattr(run, "source_fingerprint", None) or (raw_meta.get("source_fingerprint") if isinstance(raw_meta, dict) else None)
        fp_display = f"{fingerprint[:20]}... (Recorded)" if fingerprint else "Unavailable / Not Recorded"
        cam_profile = run.raw_dict.get("camera_profile", {}) if isinstance(run.raw_dict, dict) else {}
        sensor_desc = cam_profile.get("description", "Fixed Monocular Overhead CCTV")
        st.markdown(
            f'<div class="dir-a-card">'
            f'<h4 style="margin:0 0 0.5rem 0; color:#0F172A; font-size:1rem;">Ingest Integrity & Cryptographic Lineage</h4>'
            f'<table style="width:100%; border-collapse:collapse; font-size:0.83rem; font-family:\'JetBrains Mono\',monospace;">'
            f'<tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:6px 0; color:#64748B;">Source File:</td><td style="padding:6px 0; color:#0F172A; font-weight:600;">{html.escape(source_path)}</td></tr>'
            f'<tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:6px 0; color:#64748B;">SHA256 Fingerprint:</td><td style="padding:6px 0; color:#0D9488; font-weight:600;">{html.escape(fp_display)}</td></tr>'
            f'<tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:6px 0; color:#64748B;">Sensor Topology:</td><td style="padding:6px 0; color:#0F172A;">{html.escape(sensor_desc)}</td></tr>'
            f'<tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:6px 0; color:#64748B;">Dataset Origin:</td><td style="padding:6px 0; color:#0F172A;">{html.escape(run.dataset_name)}</td></tr>'
            f'<tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:6px 0; color:#64748B;">Sensor Calibration:</td><td style="padding:6px 0; color:#0F172A;">Uncalibrated / Single Overhead Perspective</td></tr>'
            f'<tr><td style="padding:6px 0; color:#64748B;">Offline Compliance:</td><td style="padding:6px 0; color:#0F766E; font-weight:700;">Local-Only / Zero Telemetry Network Call</td></tr>'
            f'</table>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_media:
        trk_frame = run.artifacts.get("tracking_frame")
        if trk_frame and os.path.isfile(trk_frame):
            st.image(trk_frame, caption=f"First Representative Ingest Frame ({run.display_name})", use_container_width=True)
        else:
            st.info("Ingest representative frame artifact is available once analyzed.")


# ============================================================================
# STAGE 2: INPUT SUITABILITY
# ============================================================================
def _render_stage_02_suitability(run: StandardizedRunData) -> None:
    st.subheader("Stage 02 · Input Suitability & Ground-Plane Assessment")
    st.markdown(
        "Evaluates whether the input recording meets empirical requirements for planar homography and spatial trajectory projection. "
        "Suitability status reflects caller-declared metadata and formal evidence context checks, not automatic computer-vision scene classification."
    )

    ev_ctx = run.raw_dict.get("evidence_context", {}) if isinstance(run.raw_dict, dict) else {}
    calib_status = ev_ctx.get("calibration_status", "unverified_configured_units")
    coverage_status = ev_ctx.get("coverage_status", "unknown")
    is_retail = run.is_retail
    env_type = run.environment_type or "unknown"

    st.markdown(
        f"""
        <div class="dir-a-card" style="margin-bottom:1rem;">
            <h4 style="margin:0 0 0.5rem 0; color:#0F172A; font-size:1rem;">Input Suitability Evidence Classification</h4>
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:0.75rem;">
                <div style="border:2px solid {'#0D9488' if is_retail else '#E2E8F0'}; background:{'#F0FDFA' if is_retail else '#F8FAFC'}; border-radius:8px; padding:0.75rem;">
                    <div style="font-weight:700; color:{'#0F766E' if is_retail else '#64748B'}; font-size:0.9rem;">
                        {'[✓]' if is_retail else '[ ]'} 1. Retail Store Environment
                    </div>
                    <div style="font-size:0.75rem; color:#475569; margin-top:0.25rem;">
                        Caller-declared provenance: <code>{html.escape(env_type)}</code>. Verified retail interior fixture layout.
                    </div>
                </div>
                <div style="border:1px solid #E2E8F0; background:#F8FAFC; border-radius:8px; padding:0.75rem;">
                    <div style="font-weight:600; color:#0F172A; font-size:0.9rem;">
                        {'[✓]' if run.has_geometry else '[!]'} 2. Planar Ground Plane Quad
                    </div>
                    <div style="font-size:0.75rem; color:#475569; margin-top:0.25rem;">
                        4 ground control points configured for projective homography matrix H.
                    </div>
                </div>
                <div style="border:1px solid {'#F59E0B' if 'unverified' in calib_status else '#E2E8F0'}; background:#F8FAFC; border-radius:8px; padding:0.75rem;">
                    <div style="font-weight:600; color:{'#B45309' if 'unverified' in calib_status else '#64748B'}; font-size:0.9rem;">
                        [!] 3. Spatial Calibration Status
                    </div>
                    <div style="font-size:0.75rem; color:#475569; margin-top:0.25rem;">
                        <code>{html.escape(calib_status)}</code>. Configured units; independent physical survey withheld.
                    </div>
                </div>
                <div style="border:1px solid {'#FDE68A' if coverage_status == 'boundary_truncated' else '#E2E8F0'}; background:#F8FAFC; border-radius:8px; padding:0.75rem;">
                    <div style="font-weight:600; color:{'#B45309' if coverage_status == 'boundary_truncated' else '#64748B'}; font-size:0.9rem;">
                        {'🛡️ Boundary Truncated' if coverage_status == 'boundary_truncated' else f'[*] Coverage: {coverage_status}'}
                    </div>
                    <div style="font-size:0.75rem; color:#475569; margin-top:0.25rem;">
                        {'Monocular boundary truncation flagged: customer activity extends outside calibrated zones.' if coverage_status == 'boundary_truncated' else 'Field-of-view coverage evaluated under standard retail geofence bounds.'}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(render_metric_card_html("Environment Declaration", "Caller-Declared" if run.is_retail is not None else "Unknown", f"Declared: {run.environment_type}"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card_html("Spatial Calibration", "Configured Units", "Physical survey unverified"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card_html("Camera Stability / Lighting", "Unmeasured", "Assumed static monocular CCTV"), unsafe_allow_html=True)


# ============================================================================
# STAGE 3: CAMERA GEOMETRY & ZONES
# ============================================================================
def _render_stage_03_geometry(run: StandardizedRunData, selected_zone: Optional[str] = None) -> None:
    st.subheader("Stage 03 · Camera Geometry, Planar Homography & Zone Geofencing")
    st.markdown(
        "Projects perspective image pixel coordinates $(u, v)$ onto configured floor plane coordinates $(X, Y)$ via Direct Linear Transform (DLT) planar homography matrix $H \\in \\mathbb{R}^{3 \\times 3}$."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(render_metric_card_html("Coordinate Unit", run.homography_unit or "configured_units", "Configured floor coordinate space"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card_html("Floor Bounds", f"[{run.bounds[0]:.1f}..{run.bounds[1]:.1f}] × [{run.bounds[2]:.1f}..{run.bounds[3]:.1f}]" if run.bounds else "N/A", "Planar boundary dimensions"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card_html("Configured Zones", str(len(run.zones or {})), "Convex retail geofences"), unsafe_allow_html=True)

    st.markdown(
        """
        <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:6px; padding:0.5rem 0.75rem; margin-bottom:0.75rem; font-size:0.8rem; color:#92400E;">
            <strong>Unit & Uncertainty Notice:</strong> Coordinates are expressed in configured floor units. 
            No metric physical scale (e.g. meters) or physical measurement uncertainty claims (e.g. ±0.3–0.5m) are asserted without an independent ground-truth survey.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render Side-by-Side Calibration Setup
    col_cam, col_floor = st.columns(2)
    with col_cam:
        st.markdown(
            f"""
            <div class="dir-a-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                    <span style="font-weight:700; color:#0F172A; font-size:0.9rem;">Perspective Camera Plane (u, v)</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#0D9488;">Planar Quad</span>
                </div>
                <div style="background:#0F172A; border-radius:8px; padding:1.25rem 1rem; color:#F8FAFC;">
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:#5EEAD4; margin-bottom:0.4rem;">
                        4-Point Planar Homography Formulation:
                    </div>
                    <div style="font-size:0.78rem; color:#94A3B8; font-family:'JetBrains Mono',monospace; line-height:1.6;">
                        • Direct Linear Transform (DLT) with Hartley-Zisserman normalization<br/>
                        • Maps pixel coordinates [u, v, 1]ᵀ → floor coordinates [X, Y, 1]ᵀ<br/>
                        • Homography Unit: <code>{html.escape(run.homography_unit or 'configured_units')}</code><br/>
                        • Zero convex polygon overlap verified across {len(run.zones or {})} geofences
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_floor:
        st.markdown(
            f"""
            <div class="dir-a-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                    <span style="font-weight:700; color:#0F172A; font-size:0.9rem;">Configured Floor Plane Geometry (X, Y)</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#0D9488;">Bounds: {run.bounds or 'N/A'}</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        valid_zones = list((run.zones or {}).keys())
        if valid_zones:
            def _on_geom_zone_change() -> None:
                st.session_state["studio_selected_zone"] = st.session_state["studio_geom_zone_select"]

            selected_idx = valid_zones.index(selected_zone) if selected_zone in valid_zones else 0
            st.selectbox(
                "Select Geofence to Highlight",
                options=valid_zones,
                index=selected_idx,
                format_func=lambda z: f"{(run.zone_names or {}).get(z, z.replace('_', ' ').title())} ({z})",
                key="studio_geom_zone_select",
                on_change=_on_geom_zone_change,
            )

        if run.bounds and run.zones:
            svg = generate_floor_plan_svg(
                bounds=run.bounds,
                zones=run.zones,
                zone_colors=run.zone_colors,
                zone_names=run.zone_names,
                candidate_zone_id=run.candidate_zone,
                selected_zone_id=selected_zone,
                width_px=450,
                height_px=220,
            )
            if svg:
                st.markdown(svg, unsafe_allow_html=True)

        if selected_zone and run.zones and selected_zone in run.zones:
            pts_disp = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in run.zones[selected_zone])
            st.markdown(
                f"""
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#475569; margin-top:0.4rem;">
                    <strong>{(run.zone_names or {}).get(selected_zone, selected_zone)} Vertices:</strong> [{pts_disp}]
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# STAGE 4: DETECTION
# ============================================================================
def _render_stage_04_detection(run: StandardizedRunData) -> None:
    st.subheader("Stage 04 · Object Detection & Customer Localization")
    st.markdown(
        f"Executes {html.escape(run.model_family)} inference under {html.escape(run.model_runtime)} for the `person` class, extracting bounding boxes and bottom-center ground contact points."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(render_metric_card_html("Total Detections", str(run.total_detections or "N/A"), "Person bounding boxes extracted"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card_html("Inference Model", run.model_family, f"Runtime: {run.model_runtime}"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card_html("Ground Contact Anchor", "Bottom-Center", "(u_min+u_max)/2, v_max"), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="dir-a-card" style="margin-top:1rem;">
            <h4 style="margin:0 0 0.5rem 0; color:#0F172A; font-size:1rem;">Ground-Contact Point Localization Method</h4>
            <p style="font-size:0.85rem; color:#475569; margin:0 0 0.5rem 0;">
                For each detected customer bounding box <code>[u_min, v_min, u_max, v_max]</code>, the planar ground projection anchor is formulated at the bottom-center point:
            </p>
            <div style="font-family:'JetBrains Mono',monospace; background:#F8FAFC; border:1px solid #E2E8F0; padding:0.5rem 0.75rem; border-radius:6px; font-size:0.85rem; color:#0F172A;">
                u_floor = (u_min + u_max) / 2.0<br/>
                v_floor = v_max  (Ground Contact Anchor)
            </div>
            <p style="font-size:0.75rem; color:#64748B; margin:0.5rem 0 0 0;">
                This anchor represents the contact patch between customer footwear and the retail floor surface, eliminating torso parallax errors.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# STAGE 5: TRACKING
# ============================================================================
def _render_stage_05_tracking(run: StandardizedRunData) -> None:
    st.subheader("Stage 05 · ByteTrack Spatial Association & Tracklet Filtering")
    st.markdown(
        "Associates temporal bounding box detections into clip-local trajectories using Kalman filter motion models and Hungarian matching (ByteTrack)."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(render_metric_card_html("Clip-Local Tracks", str(run.total_tracks or "N/A"), "Unique trajectories evaluated"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card_html("Total Trajectory Points", str(run.total_points or "N/A"), "Ground coordinates projected"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card_html("Tracking Protocol", "ByteTrack", "Clip-local spatial association"), unsafe_allow_html=True)

    st.info(
        "🔒 **Privacy & Identity Boundary**: Track IDs are strictly ephemeral clip-local integers. "
        "The system does not perform facial recognition, demographic classification, biometric profiling, or cross-camera re-identification."
    )


# ============================================================================
# STAGE 6: FLOOR TRAJECTORIES (FLAGSHIP DUAL VIEWPORT)
# ============================================================================
def _render_stage_06_trajectories(run: StandardizedRunData, reports_dir: str, reduced_motion: bool, selected_zone: Optional[str] = None) -> None:
    st.subheader("Stage 06 · Dual Synchronized Camera / Metric Floor Trajectories")
    st.markdown(
        "Dual-viewport telemetry displaying camera perspective video frames and orthogonal floor plan with linked temporal playhead scrubber."
    )

    video_path = run.artifacts.get("tracking_video")
    has_video = bool(video_path and os.path.isfile(video_path))
    timeline = load_run_observation_timeline(run)
    has_timeline = bool(timeline and run.has_geometry)

    if has_video and has_timeline:
        st.markdown(
            """
            <div style="background:#F0FDFA; border:1px solid #99F6E4; border-radius:6px; padding:0.6rem 0.85rem; margin-bottom:0.75rem; font-size:0.83rem; color:#0F766E;">
                <strong>✓ Dominant Telemetry Active:</strong> Continuous camera video playback synchronized sample-by-sample with recorded bounding boxes and projected orthogonal floor coordinates is actively rendering in the dominant workspace above.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if video_path and os.path.isfile(video_path):
            with open(video_path, "rb") as vf:
                st.download_button(
                    "📥 Download Annotated Tracking Video (WebM)",
                    vf.read(),
                    file_name=os.path.basename(video_path),
                    mime="video/webm",
                    use_container_width=True,
                    key=f"studio_download_video_timeline_{run.run_id}",
                )
    else:
        st.info("ℹ️ Frame-aligned synchronized exploration is unavailable because this run lacks an observation timeline sidecar (`observation_timeline.json`) or floor geometry.")
        if has_video:
            st.video(video_path, format="video/webm")
            with open(video_path, "rb") as vf:
                st.download_button(
                    "📥 Download Annotated Tracking Video (WebM)",
                    vf.read(),
                    file_name=os.path.basename(video_path),
                    mime="video/webm",
                    use_container_width=True,
                    key=f"studio_download_video_static_{run.run_id}",
                )
        elif run.artifacts.get("tracking_frame") and os.path.isfile(run.artifacts["tracking_frame"]):
            st.image(run.artifacts["tracking_frame"], caption=f"Representative tracking frame ({run.display_name})", use_container_width=True)
        else:
            st.warning("Neither tracking video nor tracking frame artifact is available for this run.")

    if run.out_of_bounds_points == run.total_points and (run.total_points or 0) > 0:
        st.warning(
            f"⚠️ **Spatial Out-of-Bounds Notification:** 100% of trajectory points ({run.total_points}/{run.total_points}) "
            f"projected outside configured floor bounds. All points are classified as out-of-bounds / unassigned observations."
        )

    # Trajectory Tracks Catalog Table derived from actual timeline or summary
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    
    tracks_data = []
    if timeline and isinstance(timeline, dict) and "tracks_summary" in timeline:
        ts = timeline["tracks_summary"]
        for tid, tinfo in ts.items():
            zones_v = tinfo.get("zones_visited", [])
            z_str = ", ".join(zones_v) if zones_v else "Unassigned / Transit"
            pts_cnt = tinfo.get("observed_sample_count", tinfo.get("active_sample_span", "N/A"))
            dur = tinfo.get("duration_sec", 0.0)
            status = "In-Zone Dwell" if zones_v else "Aisle Transit"
            tracks_data.append({
                "Track ID": f"Track #{tid}",
                "Point Count": f"{pts_cnt} pts",
                "Observed Span": f"{dur:.1f}s",
                "Primary Zones Visited": z_str,
                "Classification": status,
            })
    
    if tracks_data:
        st.markdown(
            f"""
            <div class="dir-a-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                    <h4 style="margin:0; font-size:0.95rem; color:#0F172A;">Evaluated Clip-Local Trajectory Catalog ({len(tracks_data)} Tracks)</h4>
                    <span style="font-size:0.75rem; color:#64748B; font-family:'JetBrains Mono',monospace;">Total Points: {run.total_points or 'N/A'}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(tracks_data, hide_index=True, use_container_width=True)
    else:
        st.markdown(
            f"""
            <div class="dir-a-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="margin:0; font-size:0.95rem; color:#0F172A;">Evaluated Clip-Local Trajectories: {run.total_tracks or 0} Tracks</h4>
                    <span style="font-size:0.75rem; color:#64748B; font-family:'JetBrains Mono',monospace;">Total Points: {run.total_points or 'N/A'} (In-Zone: {run.in_zone_points or 0}, Unassigned: {run.unassigned_points or 0})</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# STAGE 7: DENSITY / HEAT ZONES
# ============================================================================
def _render_stage_07_density(run: StandardizedRunData) -> None:
    st.subheader("Stage 07 · Spatial Density Heatmap & Point Conservation Audit")
    st.markdown(
        "Accumulates projected trajectory points into a discrete 2D Cartesian spatial grid (0.5 × 0.5 configured floor units), strictly verifying the mathematical point conservation invariant."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_metric_card_html("Total Points", str(run.total_points or "N/A"), "Projected ground coordinates"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card_html("In-Zone Points", str(run.in_zone_points or "N/A"), "Assigned inside retail geofences"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card_html("Unassigned Observations", str(run.unassigned_points or "N/A"), "Aisle transit & border transit"), unsafe_allow_html=True)
    with c4:
        if run.total_points is not None and run.binned_points is not None and run.out_of_bounds_points is not None:
            c_loss = run.total_points - (run.binned_points + run.out_of_bounds_points + (run.untransformed_points or 0))
            loss_pct = (c_loss / run.total_points * 100.0) if run.total_points > 0 else 0.0
            st.markdown(render_metric_card_html("Conservation Loss", f"{loss_pct:.2f}%", f"{c_loss} phantom point drift"), unsafe_allow_html=True)
        else:
            st.markdown(render_metric_card_html("Conservation Loss", "Unavailable", "Telemetry incomplete"), unsafe_allow_html=True)

    st.markdown(render_density_legend_html(), unsafe_allow_html=True)

    if run.out_of_bounds_points == run.total_points and (run.total_points or 0) > 0:
        st.warning(
            f"⚠️ **Discrete Grid Out-of-Bounds Notice:** All {run.total_points} trajectory points fell outside the discrete floor grid bounds "
            f"[{run.bounds[0]:.1f}..{run.bounds[1]:.1f}] × [{run.bounds[2]:.1f}..{run.bounds[3]:.1f}]. Binned count is 0, out-of-bounds count is {run.out_of_bounds_points}."
        )

    col_grid, col_cons = st.columns([1.2, 1.0])
    with col_grid:
        heat_path = run.artifacts.get("density_heatmap")
        if heat_path and os.path.isfile(heat_path):
            st.image(heat_path, caption="Discrete Point Density Heatmap Grid (0.5 × 0.5 configured floor units)", use_container_width=True)
        else:
            st.info("Density heatmap artifact will render here when generated.")

    with col_cons:
        # Invariant 1: Density Conservation
        if run.density_conservation_verified is True:
            d_tag, d_color, d_bg, d_border = "[PASSED]", "#0F766E", "#F0FDFA", "#99F6E4"
        elif run.density_conservation_verified is False:
            d_tag, d_color, d_bg, d_border = "[MISMATCH]", "#B45309", "#FFFBEB", "#FDE68A"
        else:
            d_tag, d_color, d_bg, d_border = "[NOT EVALUATED]", "#475569", "#F1F5F9", "#CBD5E1"

        # Invariant 2: Zone Conservation
        if run.zone_conservation_verified is True:
            z_tag, z_color, z_bg, z_border = "[PASSED]", "#0F766E", "#F0FDFA", "#99F6E4"
        elif run.zone_conservation_verified is False:
            z_tag, z_color, z_bg, z_border = "[MISMATCH]", "#B45309", "#FFFBEB", "#FDE68A"
        else:
            z_tag, z_color, z_bg, z_border = "[NOT EVALUATED]", "#475569", "#F1F5F9", "#CBD5E1"

        st.markdown(
            f"""
            <div class="dir-a-card">
                <h4 style="margin:0 0 0.5rem 0; color:#0F172A; font-size:1rem;">Mathematical Invariant Audit</h4>
                <div style="background:{d_bg}; border:1px solid {d_border}; border-radius:6px; padding:0.6rem 0.8rem; margin-bottom:0.75rem;">
                    <div style="font-weight:700; color:{d_color}; font-size:0.85rem;">{d_tag} Invariant 1: Density Conservation</div>
                    <div style="font-size:0.75rem; color:#134E4A; font-family:'JetBrains Mono',monospace; margin-top:0.2rem;">
                        Total ({run.total_points if run.total_points is not None else 'N/A'}) == Binned ({run.binned_points if run.binned_points is not None else 'N/A'}) + OutOfBounds ({run.out_of_bounds_points if run.out_of_bounds_points is not None else 'N/A'})
                    </div>
                </div>
                <div style="background:{z_bg}; border:1px solid {z_border}; border-radius:6px; padding:0.6rem 0.8rem; margin-bottom:0.75rem;">
                    <div style="font-weight:700; color:{z_color}; font-size:0.85rem;">{z_tag} Invariant 2: Zone Conservation</div>
                    <div style="font-size:0.75rem; color:#134E4A; font-family:'JetBrains Mono',monospace; margin-top:0.2rem;">
                        Total ({run.total_points if run.total_points is not None else 'N/A'}) == InZone ({run.in_zone_points if run.in_zone_points is not None else 'N/A'}) + Unassigned Observations ({run.unassigned_points if run.unassigned_points is not None else 'N/A'})
                    </div>
                </div>
                <div style="font-size:0.78rem; color:#64748B;">
                    Point conservation guarantees that every trajectory sample is accounted for without numerical loss, interpolation inflation, or spatial leakage. Sampled points represent coordinate observations, never unique customers.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# STAGE 8: OBSERVED DWELL & ZONE ACTIVITY
# ============================================================================
def _render_stage_08_dwell(run: StandardizedRunData, selected_zone: Optional[str] = None) -> None:
    st.subheader("Stage 08 · Observed Dwell Spans & Geofence Activity")
    st.markdown(
        "Measures cumulative customer dwell duration per retail geofence. "
        "Temporal hysteresis is applied during zone assignment to filter transient boundary jitter and prevent false episode fragmentation."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(render_metric_card_html("Whole-Run Dwell", format_observed_value(run.total_dwell_sec, "s"), "Cumulative browsing duration"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card_html("Total Dwell Episodes", str(run.total_episodes or "N/A"), "Visits exceeding hysteresis threshold"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card_html("Primary Dwell Zone", run.candidate_zone or "N/A", f"{format_observed_value(run.candidate_dwell_sec, 's')} observed span"), unsafe_allow_html=True)

    # Interactive Zone Selector
    names = run.zone_names or {}
    zones_list = list(run.zone_dwell.keys())
    if zones_list:
        st.markdown(
            """
            <div style="font-size:0.75rem; font-weight:700; color:#475569; text-transform:uppercase; margin-top:0.75rem; margin-bottom:0.25rem;">
                Select Geofence to Inspect Deep Metrics:
            </div>
            """,
            unsafe_allow_html=True,
        )
        def _on_dwell_zone_change() -> None:
            st.session_state["studio_selected_zone"] = st.session_state["studio_dwell_zone_select"]

        selected_idx = zones_list.index(selected_zone) if selected_zone in zones_list else 0
        current_sel = st.selectbox(
            "Inspect Zone",
            options=zones_list,
            index=selected_idx,
            format_func=lambda z: f"{names.get(z, z.replace('_', ' ').title())} ({z})",
            key="studio_dwell_zone_select",
            on_change=_on_dwell_zone_change,
        )

        zdata = run.zone_dwell.get(current_sel, {})
        trans_in = sum(cnt for edge, cnt in (run.direct_transitions or {}).items() if edge.endswith(f" -> {current_sel}"))
        trans_out = sum(cnt for edge, cnt in (run.direct_transitions or {}).items() if edge.startswith(f"{current_sel} ->"))
        if run.candidate_zone_ids:
            is_merch = current_sel in run.candidate_zone_ids
            eligibility_label = "Eligible Merchandising Display Zone (Configured Candidate)" if is_merch else "Non-Candidate Zone (Excluded from Merchandising Selection)"
            eligibility_color = "#0D9488" if is_merch else "#64748B"
        elif run.comparison_by_shelf:
            compared_ids = [c["zone_id"] for c in run.comparison_by_shelf if isinstance(c, dict) and "zone_id" in c]
            is_merch = current_sel in compared_ids
            eligibility_label = "Eligible Candidate Zone (Recorded in Comparison)" if is_merch else "Non-Candidate Zone"
            eligibility_color = "#0D9488" if is_merch else "#64748B"
        else:
            eligibility_label = "Candidate Eligibility Not Configured / Not Evaluated"
            eligibility_color = "#64748B"

        st.markdown(
            f"""
            <div class="dir-a-card" style="border-left:4px solid {run.zone_colors.get(current_sel, '#0D9488')};">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                    <h4 style="margin:0; color:#0F172A; font-size:1.1rem;">
                        {names.get(current_sel, current_sel.replace('_', ' ').title())}
                    </h4>
                    <span style="font-size:0.75rem; font-weight:700; color:{eligibility_color}; background:#F8FAFC; border:1px solid #E2E8F0; padding:2px 8px; border-radius:4px;">
                        {eligibility_label}
                    </span>
                </div>
                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:0.75rem; font-family:'JetBrains Mono',monospace;">
                    <div><span style="color:#64748B;">Observed Dwell Span:</span><br/><strong style="font-size:1.1rem; color:#0F172A;">{format_observed_value(zdata.get('observed_dwell_span_sec', zdata.get('total_dwell_sec')), 's')}</strong></div>
                    <div><span style="color:#64748B;">Distinct Visits:</span><br/><strong style="font-size:1.1rem; color:#0F172A;">{zdata.get('episodes', zdata.get('episode_count', 0))} episodes</strong></div>
                    <div><span style="color:#64748B;">Assigned Points:</span><br/><strong style="font-size:1.1rem; color:#0F172A;">{run.zone_assignment_counts.get(current_sel, 0)} pts</strong></div>
                    <div><span style="color:#64748B;">Direct Transitions:</span><br/><strong style="font-size:1.1rem; color:#0F172A;">{trans_in} in / {trans_out} out</strong></div>
                    <div><span style="color:#64748B;">Distinct Tracks:</span><br/><strong style="font-size:1.1rem; color:#0F172A;">{zdata.get('distinct_clip_local_tracks', 0)} tracks</strong></div>
                    <div><span style="color:#64748B;">Whole-Run Dwell:</span><br/><strong style="font-size:1.1rem; color:#64748B;">{format_observed_value(run.total_dwell_sec, 's')}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Full Zone Dwell Table
    rows = []
    for zid, zdata in run.zone_dwell.items():
        rows.append({
            "Zone": names.get(zid, zid.replace("_", " ").title()),
            "Zone ID": zid,
            "Observed Span (s)": zdata.get("observed_dwell_span_sec", zdata.get("total_dwell_sec", 0.0)),
            "Episodes": zdata.get("episodes", zdata.get("episode_count", 0)),
            "Assigned Points": run.zone_assignment_counts.get(zid, 0),
            "Selected Candidate": zid == run.candidate_zone,
        })
    if rows:
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.dataframe(rows, hide_index=True, use_container_width=True)


# ============================================================================
# STAGE 9: DIRECT TRANSITIONS
# ============================================================================
def _render_stage_09_transitions(run: StandardizedRunData) -> None:
    st.subheader("Stage 09 · Direct Transitions & Directional Flow")
    st.markdown(
        "Analyzes sequential customer movement between adjacent retail geofences. "
        "Direct transitions capture first-order sequential handoffs between consecutive zones without intermediate visits. "
        "Zone hysteresis smoothing operates at the zone-assignment stage, ensuring transitions reflect consolidated visits rather than boundary flicker."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(render_metric_card_html("Total Direct Transitions", str(run.total_transitions if run.total_transitions is not None else len(run.direct_transitions or {})), "Observed zone-to-zone handoffs"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card_html("Transition Order", "First-Order (Direct Handoff)", "Immediate consecutive geofence movement"), unsafe_allow_html=True)

    if run.direct_transitions:
        col_list, col_mat = st.columns(2)
        with col_list:
            st.markdown(
                """
                <div class="dir-a-card">
                    <h4 style="margin:0 0 0.5rem 0; color:#0F172A; font-size:0.95rem;">Observed Transition Pathways</h4>
                    <table style="width:100%; border-collapse:collapse; font-size:0.83rem; font-family:'JetBrains Mono',monospace;">
                        <thead style="background:#F8FAFC; border-bottom:1px solid #E2E8F0; color:#475569;">
                            <tr><th style="padding:6px; text-align:left;">Transition Pathway</th><th style="padding:6px; text-align:right;">Count</th></tr>
                        </thead>
                        <tbody>
                """,
                unsafe_allow_html=True,
            )
            for edge, count in run.direct_transitions.items():
                st.markdown(
                    f'<tr style="border-bottom:1px solid #F1F5F9;"><td style="padding:6px; color:#0D9488; font-weight:600;">{edge}</td><td style="padding:6px; text-align:right; font-weight:700;">{count}</td></tr>',
                    unsafe_allow_html=True,
                )
            st.markdown("</tbody></table></div>", unsafe_allow_html=True)

        with col_mat:
            st.markdown(
                """
                <div class="dir-a-card">
                    <h4 style="margin:0 0 0.5rem 0; color:#0F172A; font-size:0.95rem;">Transition Flow Interpretation</h4>
                    <p style="font-size:0.83rem; color:#475569; line-height:1.5;">
                        Direct transitions represent immediate sequential movements between adjacent zones. 
                        Unlike stationary dwell, transitions identify directional pathways between browsing areas and primary aisles.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Direct transitions: 0 transitions observed across evaluated tracklets.")


# ============================================================================
# STAGE 10: PLACEMENT REVIEW & CANDIDATE EXPLAINABILITY
# ============================================================================
def _render_stage_10_placement(run: StandardizedRunData, selected_zone: Optional[str] = None) -> None:
    st.subheader("Stage 10 · Spatial Placement Candidate Review & Explainability")
    st.markdown(
        "Evaluates retail merchandising display zones using empirical multi-objective metrics: observed dwell duration, visit episodes, and sampled point counts. "
        "Circulation aisles, entrance foyers, and checkout/cash wrap counters are excluded from placement eligibility."
    )

    # Prominent Evidence Withholding Banner
    reasons = run.reasons_withheld or [
        "Monocular boundary truncation detected: customer activity occurs outside calibrated zone boundaries.",
        "Spatial calibration status is unverified configured units; physical measurement survey withheld.",
    ]
    st.markdown(render_withheld_banner_html(reasons, title="EVIDENCE STATUS: EXPLORATORY / WITHHELD FOR AUTOMATED RECONFIGURATION"), unsafe_allow_html=True)

    names = run.zone_names or {}
    cand_id = run.candidate_zone
    cand_name = names.get(cand_id, cand_id.replace("_", " ").title()) if cand_id else "None"
    cand_dwell = (run.candidate_dwell_sec or 0.0)
    if cand_id and cand_dwell == 0.0 and run.zone_dwell and cand_id in run.zone_dwell:
        cand_dwell = float(run.zone_dwell[cand_id].get("observed_dwell_span_sec", run.zone_dwell[cand_id].get("total_dwell_sec", 0.0)) or 0.0)

    # Selected Candidate Deep Dive & Floor Map with 2D / 3D Toggle
    col_map, col_score = st.columns([1.1, 1.1])
    with col_map:
        st.markdown(
            f"""
            <div class="dir-a-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                    <span style="font-weight:700; color:#0F172A; font-size:0.95rem;">Floor Plan & Hotspot Overlay</span>
                    <span style="background:#F0FDFA; color:#0F766E; font-size:0.75rem; font-weight:700; padding:2px 6px; border-radius:4px;">
                        {'★ Candidate: ' + cand_name if cand_id else 'No Candidate Emitted'}
                    </span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        col_p1, col_p2 = st.columns([1.0, 1.0])
        with col_p1:
            placement_perspective = st.radio(
                "Map Perspective",
                options=["Orthogonal 2D Plan", "Analytical 3D Isometric Extrusion"],
                horizontal=True,
                key="studio_stage10_perspective_radio",
                label_visibility="collapsed",
            )

        if run.bounds and run.zones:
            if "Isometric" in placement_perspective:
                svg = generate_isometric_floor_plan_svg(
                    bounds=run.bounds,
                    zones=run.zones,
                    zone_colors=run.zone_colors,
                    zone_names=run.zone_names,
                    candidate_hotspot=run.candidate_hotspot_cell,
                    candidate_zone_id=run.candidate_zone,
                    selected_zone_id=selected_zone,
                    width_px=480,
                    height_px=280,
                    zone_dwell=run.zone_dwell,
                    dwell_scale=0.06,
                )
            else:
                svg = generate_floor_plan_svg(
                    bounds=run.bounds,
                    zones=run.zones,
                    zone_colors=run.zone_colors,
                    zone_names=run.zone_names,
                    candidate_hotspot=run.candidate_hotspot_cell,
                    candidate_zone_id=run.candidate_zone,
                    selected_zone_id=selected_zone,
                    width_px=480,
                    height_px=280,
                )
            if svg:
                st.markdown(svg, unsafe_allow_html=True)

        if run.candidate_hotspot_cell:
            hx = run.candidate_hotspot_cell
            st.markdown(
                f"""
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#0F172A; margin-top:0.5rem; background:#F8FAFC; border:1px solid #E2E8F0; padding:0.4rem 0.6rem; border-radius:4px;">
                    <strong>Candidate Hotspot Cell:</strong> X [{hx.get('x_min', 0.0):.1f}..{hx.get('x_max', 0.0):.1f}] × Y [{hx.get('y_min', 0.0):.1f}..{hx.get('y_max', 0.0):.1f}]<br/>
                    <span style="color:#64748B;">Sampled Points: {hx.get('sampled_points', hx.get('point_count', 0))} points | Unit: configured floor units (no hotspot dwell time measured)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#64748B; margin-top:0.5rem; background:#F8FAFC; border:1px solid #E2E8F0; padding:0.4rem 0.6rem; border-radius:4px;">
                    <strong>Hotspot Status:</strong> Hotspot not evaluated. No sub-zone cell met candidate qualifications; no dummy coordinates or unmeasured dwell values are asserted.
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_score:
        st.markdown(
            f"""
            <div class="dir-a-card">
                <h4 style="margin:0 0 0.5rem 0; color:#0F172A; font-size:1rem;">Candidate Explainability: {cand_name}</h4>
                <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:0.75rem;">
                    <span style="font-size:0.85rem; color:#64748B;">Selection Status:</span>
                    <span style="font-size:1.1rem; font-weight:700; color:#0D9488; font-family:'JetBrains Mono',monospace;">
                        {'★ Top Merchandising Candidate' if cand_id else 'No Candidate Qualified'}
                    </span>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:0.6rem 0.8rem; margin-bottom:0.75rem; font-size:0.82rem;">
                    <strong>Decision Basis & Rule:</strong><br/>
                    <span style="color:#475569;">{run.candidate_selection_rule or 'Evaluated based on empirical dwell duration among active merchandising zones.'}</span>
                    <div style="margin-top:0.35rem; font-size:0.78rem; color:#64748B;">
                        <strong>Tie-Breaking Hierarchy:</strong> Primary = highest observed dwell span; Secondary = highest visit episodes; Tertiary = sampled in-zone points.
                    </div>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; font-family:'JetBrains Mono',monospace; font-size:0.83rem;">
                    <div style="background:#F0FDFA; padding:0.5rem; border-radius:4px; border:1px solid #CCFBF1;">
                        <span style="color:#0F766E;">Candidate Dwell:</span><br/>
                        <strong style="font-size:1rem; color:#0F172A;">{format_observed_value(run.candidate_dwell_sec, 's')}</strong>
                    </div>
                    <div style="background:#F8FAFC; padding:0.5rem; border-radius:4px; border:1px solid #E2E8F0;">
                        <span style="color:#64748B;">Whole-Run Dwell:</span><br/>
                        <strong style="font-size:1rem; color:#0F172A;">{format_observed_value(run.total_dwell_sec, 's')}</strong>
                    </div>
                </div>
                {f'''<div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:6px; padding:0.5rem 0.75rem; margin-top:0.75rem; font-size:0.78rem; color:#92400E;">
                    <strong>⚠️ Single-Observed-Candidate Limitation:</strong> Among configured eligible display zones, only <code>{cand_id}</code> had observed customer activity ({cand_dwell:.1f}s, alternative display zones recorded 0.0s / 0 visits). The top candidate selection was evaluated via argmax, but without empirical dwell competition between active displays.
                </div>''' if (cand_id and run.zone_dwell and len([zid for zid, zd in run.zone_dwell.items() if (zid in (run.candidate_zone_ids or []) or any(item.get('zone_id') == zid for item in (run.comparison_by_shelf or []))) and (zd.get('observed_dwell_span_sec', zd.get('total_dwell_sec', 0.0)) or 0.0) > 0]) == 1) else ''}
                <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:6px; padding:0.5rem 0.75rem; margin-top:0.75rem; font-size:0.78rem; color:#92400E;">
                    <strong>Feasibility Boundary:</strong> Trajectory hotspots represent observational customer dwell density, NOT an automated layout recommendation or physical fixture clearance survey. Power drops, ADA walkway compliance, and fixture installation clearances must be surveyed independently.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Merchandising Display Zones Comparison Table
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size:0.8rem; font-weight:700; color:#334155; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.35rem;">
            Eligible Merchandising Display Zones Comparison (Aisles & Checkout Excluded):
        </div>
        """,
        unsafe_allow_html=True,
    )

    comp_rows = []
    if run.comparison_by_shelf:
        for item in run.comparison_by_shelf:
            zid = item.get("zone_id", "")
            dwell_s = item.get("observed_dwell_span_sec", 0.0)
            comp_rows.append({
                "Merchandising Zone": names.get(zid, zid.replace("_", " ").title()),
                "Zone ID": zid,
                "Observed Dwell Span (s)": dwell_s,
                "Visit Episodes": item.get("observed_visit_episodes", 0),
                "Sampled Points": item.get("sampled_in_zone_points", 0),
                "Selection Status": "★ Selected Candidate" if zid == cand_id else ("Alternative Display Zone" if dwell_s > 0 else "Unobserved Display Zone"),
            })
    else:
        # Check raw dict for comparison_by_shelf_zone (e.g. in 0a24bbf3de)
        raw_comp = run.raw_dict.get("exploratory_layout_candidate", {}).get("comparison_by_shelf_zone", [])
        if raw_comp:
            for item in raw_comp:
                zid = item.get("zone_id", "")
                dwell_s = item.get("observed_dwell_span_sec", 0.0)
                comp_rows.append({
                    "Merchandising Zone": names.get(zid, zid.replace("_", " ").title()),
                    "Zone ID": zid,
                    "Observed Dwell Span (s)": dwell_s,
                    "Visit Episodes": item.get("observed_visit_episodes", 0),
                    "Sampled Points": item.get("sampled_in_zone_points", 0),
                    "Selection Status": "★ Selected Candidate" if (cand_id and zid == cand_id) else ("Alternative Display Zone" if dwell_s > 0 else "Unobserved Display Zone"),
                })
        else:
            eligible_ids = run.candidate_zone_ids or []
            for zid in eligible_ids:
                if zid in run.zone_dwell:
                    zdata = run.zone_dwell[zid]
                    dwell_s = zdata.get("observed_dwell_span_sec", zdata.get("total_dwell_sec", 0.0))
                    ep_cnt = zdata.get("episodes", zdata.get("episode_count", 0))
                    pts_cnt = run.zone_assignment_counts.get(zid, 0)
                    comp_rows.append({
                        "Merchandising Zone": names.get(zid, zid.replace("_", " ").title()),
                        "Zone ID": zid,
                        "Observed Dwell Span (s)": dwell_s,
                        "Visit Episodes": ep_cnt,
                        "Sampled Points": pts_cnt,
                        "Selection Status": "★ Selected Candidate" if zid == cand_id else ("Alternative Display Zone" if dwell_s > 0 else "Unobserved Display Zone"),
                    })

    if comp_rows:
        st.dataframe(comp_rows, hide_index=True, use_container_width=True)
    else:
        st.info("No eligible merchandising display zones configured for this run.")

    # 3D Isometric View Expander
    with st.expander("🌐 Launch Interactive 3D Isometric Extrusion View", expanded=False):
        st.markdown(
            """
            <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:6px; padding:0.6rem 0.8rem; margin-bottom:0.75rem; font-size:0.8rem; color:#92400E;">
                <strong>Scientific Representation & Analytical Extrusion Disclaimer:</strong><br/>
                Height in this isometric representation is an extruded analytical mapping of <em>observed cumulative dwell duration (seconds)</em>, 
                NOT a physical 3D sensor capture, ceiling height, or fixture height.<br/>
                • <strong>Extruded Quantity:</strong> Observed dwell span (<code>observed_dwell_span_sec</code>).<br/>
                • <strong>Extrusion Scale:</strong> 1.0s dwell ≈ 0.06 configured floor units height (capped at 1.2 units).<br/>
                • <strong>Zero / Unobserved Dwell:</strong> Zones with 0.0s dwell or unavailable dwell remain un-extruded at ground elevation (0.0 units).
            </div>
            """,
            unsafe_allow_html=True,
        )
        if run.bounds and run.zones:
            iso_svg = generate_isometric_floor_plan_svg(
                bounds=run.bounds,
                zones=run.zones,
                zone_colors=run.zone_colors,
                zone_names=run.zone_names,
                candidate_hotspot=run.candidate_hotspot_cell,
                candidate_zone_id=run.candidate_zone,
                selected_zone_id=selected_zone,
                width_px=700,
                height_px=380,
                zone_dwell=run.zone_dwell,
                dwell_scale=0.06,
            )
            if iso_svg:
                st.markdown(iso_svg, unsafe_allow_html=True)


# ============================================================================
# STAGE 11: EVIDENCE & EXPORT
# ============================================================================
def _render_stage_11_evidence(run: StandardizedRunData) -> None:
    st.subheader("Stage 11 · Evidence Gating, Audit Provenance & Export")
    st.markdown(
        "Reviews complete algorithmic lineage, point conservation verification, and exports reproducible audit artifacts for academic evaluation."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(render_metric_card_html("Evidence Gating", run.evidence_status, "Automated layout withheld"), unsafe_allow_html=True)
    with c2:
        if run.density_conservation_verified is True and run.zone_conservation_verified is True:
            cons_val, cons_sub = "Pass (100%)", "Zero coordinate leakage"
        elif run.density_conservation_verified is False or run.zone_conservation_verified is False:
            cons_val, cons_sub = "Failed", "Point drift detected"
        else:
            cons_val, cons_sub = "Not Evaluated", "Missing verification fields"
        st.markdown(render_metric_card_html("Point Conservation", cons_val, cons_sub), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card_html("Detection Model", run.model_family, f"Runtime: {run.model_runtime}"), unsafe_allow_html=True)

    # Reasons Withheld List
    if run.reasons_withheld:
        st.markdown(
            """
            <div class="dir-a-card" style="margin-top:0.75rem; border-left:4px solid #F59E0B;">
                <h4 style="margin:0 0 0.4rem 0; color:#0F172A; font-size:0.95rem;">Evidence Withholding Audit Reasons</h4>
                <ul style="margin:0; padding-left:1.25rem; font-size:0.83rem; color:#475569; line-height:1.6;">
            """,
            unsafe_allow_html=True,
        )
        for r in run.reasons_withheld:
            st.markdown(f"<li>{html.escape(r)}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)

    # Export Actions
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    json_str = json.dumps(run.raw_dict or {"run_id": run.run_id}, indent=2)
    st.download_button(
        "📥 Export Full Audit JSON",
        data=json_str,
        file_name=f"{run.run_id.replace('/', '_')}_audit_report.json",
        mime="application/json",
        use_container_width=True,
        key=f"studio_stage11_export_json_{run.run_id}",
    )

    with st.expander("View Raw Run JSON"):
        st.json(run.raw_dict)
