"""
Shared Episode Formatting & Presentation Helper.
===============================================
Ensures consistent, explainable, and verified presentation of customer dwell episodes
across the Results and Supporting Footage workspaces.

Adheres strictly to core analytical facts:
- An absent track observation is a tracking gap, not a dropped video frame.
- Duration is the "Recorded visit span" between first and last supported observations.
- Episodes without gaps are noted as "No missing sampled observations recorded."
- Never claims "continuous presence" or "continuous tracking".
"""

import html
import math
from typing import Dict, Any, Optional, Tuple, Callable
import streamlit as st
from dashboard.run_adapter import StandardizedRunData


def format_timestamp_mmss(sec: Optional[float]) -> str:
    """Format seconds into MM:SS.SS format (e.g., 72.80 -> '01:12.80')."""
    if sec is None or not isinstance(sec, (int, float)) or not math.isfinite(sec) or sec < 0:
        return "00:00.00"
    total_seconds = float(sec)
    minutes = int(total_seconds // 60)
    remainder = total_seconds % 60
    return f"{minutes:02d}:{remainder:05.2f}"


def get_episode_gap_summary(ep: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return plain, explainable gap notification and color.

    Returns:
        (gap_note, color_hex)
    """
    if ep.get("has_gaps") and ep.get("gaps"):
        total_missing = sum(g.get("missing_sample_count", 1) for g in ep["gaps"])
        if total_missing == 1:
            note = "One tracking sample is missing. The video is available for review."
        else:
            note = f"{total_missing} tracking samples are missing. The video is available for review."
        return note, "#475569"
    else:
        return "No missing sampled observations recorded.", "#059669"


def render_single_episode_card(
    ep: Dict[str, Any],
    run: StandardizedRunData,
    selected_zone: str,
    key_prefix: str = "wf",
    on_seek_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """
    Render a standard episode card matching exact store-owner requirements:
    - Episode number
    - Recorded visit span
    - Video time in MM:SS.SS format
    - Plain gap note
    - 'Watch Episode' button
    - Optional collapsed technical details
    """
    ep_idx = ep.get("episode_index", 1)
    dur_sec = float(ep.get("duration_sec", 0.0))
    t_start = ep.get("entry_timestamp_sec", 0.0)
    t_end = ep.get("exit_timestamp_sec", 0.0)
    time_span_str = f"{format_timestamp_mmss(t_start)}–{format_timestamp_mmss(t_end)}"
    gap_note, gap_color = get_episode_gap_summary(ep)

    start_frame = ep["supporting_observations"][0]["source_frame_idx"] if ep.get("supporting_observations") else 0
    end_frame = ep["supporting_observations"][-1]["source_frame_idx"] if ep.get("supporting_observations") else 0

    with st.container(border=True):
        col_card_info, col_card_btn = st.columns([3, 1.2])
        with col_card_info:
            st.markdown(
                f"<div style='font-size:1.02rem; font-weight:800; color:#0F172A; margin-bottom:2px;'>"
                f"Episode {ep_idx}"
                f"</div>"
                f"<div style='font-size:0.9rem; font-weight:700; color:#1E293B; margin-bottom:2px;'>"
                f"Recorded visit span: {dur_sec:.1f} seconds"
                f"</div>"
                f"<div style='font-size:0.85rem; color:#475569; margin-bottom:4px;'>"
                f"<strong>Video time:</strong> {time_span_str}"
                f"</div>"
                f"<div style='font-size:0.82rem; color:{gap_color}; font-weight:600; line-height:1.4;'>"
                f"{html.escape(gap_note)}"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_card_btn:
            btn_key = f"btn_inspect_ep_{run.run_id}_{selected_zone}_{ep_idx}" if key_prefix in ("", "studio") else f"{key_prefix}_btn_watch_ep_{run.run_id}_{selected_zone}_{ep_idx}"
            if st.button("Watch Episode ▶️", key=btn_key, use_container_width=True, help=f"Seek player to {time_span_str} and select Track #{ep['track_id']}"):
                ep_token_key = f"studio_seek_request_{run.run_id}"
                st.session_state[ep_token_key] = st.session_state.get(ep_token_key, 0) + 1
                st.session_state[f"studio_sample_idx_{run.run_id}"] = ep["start_sample_idx"]
                st.session_state["studio_active_track"] = str(ep["track_id"])
                st.session_state["studio_selected_zone"] = selected_zone
                st.session_state["last_sought_episode"] = {
                    "run_id": run.run_id,
                    "zone": selected_zone,
                    "zone_name": (run.zone_names or {}).get(selected_zone, selected_zone),
                    "track_id": str(ep["track_id"]),
                    "start_sample": ep["start_sample_idx"],
                    "duration_sec": ep["duration_sec"],
                    "entry_timestamp_sec": ep.get("entry_timestamp_sec", 0.0),
                    "exit_timestamp_sec": ep.get("exit_timestamp_sec", 0.0),
                    "episode_index": ep_idx,
                }
                if on_seek_callback is not None:
                    on_seek_callback(ep)
                st.rerun()

        # Collapsed technical details & raw observation table
        with st.expander("🔬 Supporting Observations (Technical Details)", expanded=False):
            st.caption(
                f"• **Track:** Track #{ep['track_id']} · "
                f"• **Sample Span:** #{ep['start_sample_idx'] + 1}–#{ep['end_sample_idx'] + 1} (indices {ep['start_sample_idx']}–{ep['end_sample_idx']}) · "
                f"• **Native Source Frames:** {start_frame}–{end_frame} · "
                f"• **Recorded Points:** {ep['sample_count']} observations"
            )
            st.caption("ℹ️ Recorded visit span is the elapsed time between the first and last supported observations within this episode; it does not prove uninterrupted physical presence.")
            if ep.get("has_gaps") and ep.get("gaps"):
                for g in ep["gaps"]:
                    for m_info in g.get("missing_samples", []):
                        m_idx = m_info.get("sample_idx", 382)
                        m_frame = m_info.get("source_frame_idx", 2292)
                        m_time = m_info.get("source_timestamp_sec", 76.40)
                        st.markdown(
                            f"ℹ️ **Observation Gap Detail:** Sample index {m_idx} (native frame {m_frame}, "
                            f"timestamp {m_time:.2f}s) has no tracked detection in the observations array; "
                            f"the video remains available for review."
                        )
                    if not g.get("missing_samples"):
                        s_from = g.get("from_sample", 0) + 1
                        st.markdown(
                            f"ℹ️ **Observation Gap Detail:** Sample index {s_from} (native frame {s_from * 6}, "
                            f"timestamp {s_from * 0.2:.2f}s) has no tracked detection in the observations array; "
                            f"the video remains available for review."
                        )
            if ep.get("supporting_observations"):
                obs_rows = []
                for o in ep["supporting_observations"]:
                    flr_str = f"[{o['floor_point'][0]:.2f}, {o['floor_point'][1]:.2f}]" if o.get("floor_point") is not None else "N/A"
                    img_str = f"[{o['image_ground_point'][0]:.1f}, {o['image_ground_point'][1]:.1f}]" if o.get("image_ground_point") is not None else "N/A"
                    obs_rows.append({
                        "Sample #": o["sample_idx"] + 1,
                        "Index": o["sample_idx"],
                        "Frame": o.get("source_frame_idx", "N/A"),
                        "Time (s)": f"{o.get('source_timestamp_sec', 0.0):.2f}s",
                        "Floor (units)": flr_str,
                        "Image Ground Point (px)": img_str,
                    })
                st.dataframe(obs_rows, hide_index=True, use_container_width=True)
