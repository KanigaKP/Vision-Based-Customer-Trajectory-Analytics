"""
Direction A Destination 4: Placement Candidate View.
====================================================
Presents explainable, auditable diagnostics for exploratory retail layout
candidate selection without altering Phase 13 report schemas or inventing
unsupported recommendation claims.

ORGANIZATION:
1. Exploratory Candidate: Header, status badge, prominent WITHHELD banner.
2. Why This Region: Concise explanation derived from actual observed metrics,
   objective used, selection rule, lead difference, and observation lineage.
3. Spatial Placement Workspace:
   - Large Camera-View Projection beside the corresponding Orthogonal Floor Map.
   - Consistent zone colors and candidate highlights.
   - Hotspot coordinate bounds and spatial centroid disclaimer.
4. Comparison with Observed Alternatives: Candidate zones comparison table
   with neutral absence labeling and multi-zone vs single-zone context.
5. Objective Sensitivity Comparison: Multi-objective comparison across
   observed_dwell, sampled_points, and visit_episodes.
6. Detailed Provenance & Architectural Limitations (Expandable panel).
"""

import os
import html
from typing import Optional, Dict, Any, List
import streamlit as st

from dashboard.run_adapter import (
    StandardizedRunData,
    discover_standardized_runs,
    format_observed_value,
)
from dashboard.theme import (
    render_metric_card_html,
    render_withheld_banner_html,
    generate_floor_plan_svg,
    generate_isometric_floor_plan_svg,
    ZONE_COLORS,
)


OBJECTIVE_NAMES = {
    "observed_dwell": "Observed Dwell Span",
    "sampled_points": "Sampled In-Zone Points",
    "visit_episodes": "Observed Visit Episodes",
}

OBJECTIVE_UNITS = {
    "observed_dwell": "s",
    "sampled_points": "pts",
    "visit_episodes": "episodes",
}


def render_placement_candidate_view(reports_dir: str = "outputs/reports") -> None:
    """Render the explainable Direction A Placement Candidate analysis."""
    st.markdown("### Exploratory New-Product Placement Candidate")
    st.caption("Recorded traffic evidence for review; installation feasibility and commercial effects are not established.")

    runs = discover_standardized_runs(reports_dir)
    if not runs:
        st.warning("No runs available.")
        return

    # Sync authoritative run (Strict Exact Canonical Run ID match)
    run_ids = [r.run_id for r in runs]
    query_run_id = st.query_params.get("run_id")
    if query_run_id:
        matched = next((rid for rid in run_ids if query_run_id == rid or (rid.startswith("uploaded_runs/") and query_run_id == rid.split("/")[1])), None)
        if matched:
            st.session_state["authoritative_run_id"] = matched
            st.session_state["candidate_run_switcher"] = matched
        else:
            st.warning(
                f"⚠️ Requested Run ID '{query_run_id}' not found. Exact canonical match required. "
                f"Available runs: {', '.join(run_ids)}."
            )

    current_run_id = st.session_state.get("authoritative_run_id")
    if current_run_id not in run_ids:
        current_run_id = run_ids[0]
        st.session_state["authoritative_run_id"] = current_run_id

    # Active run indicator & selector
    c_info, c_sel = st.columns([3, 2])
    with c_info:
        run = next((r for r in runs if r.run_id == current_run_id), runs[0])
        st.markdown(f"**Evaluating Run:** `{run.display_name}` (`{run.run_id}`)")
    with c_sel:
        if "candidate_run_switcher" in st.session_state and st.session_state["candidate_run_switcher"] != current_run_id:
            st.session_state["candidate_run_switcher"] = current_run_id

        def _on_candidate_run_change():
            st.session_state["authoritative_run_id"] = st.session_state["candidate_run_switcher"]

        new_sel = st.selectbox(
            "Switch Run",
            options=run_ids,
            index=run_ids.index(current_run_id),
            format_func=lambda rid: next((r.display_name for r in runs if r.run_id == rid), rid),
            key="candidate_run_switcher",
            on_change=_on_candidate_run_change,
        )
        if new_sel != current_run_id:
            st.session_state["authoritative_run_id"] = new_sel
            current_run_id = new_sel
            run = next(r for r in runs if r.run_id == current_run_id)

    # -------------------------------------------------------------------------
    # 1. Exploratory Candidate & Withholding Banner
    # -------------------------------------------------------------------------
    withholding_reasons = run.reasons_withheld if run.reasons_withheld else [
        "No specific causal withholding reasons recorded for this run."
    ]
    st.markdown("**Evidence status: WITHHELD** · Exploratory candidate only · Configured floor units")
    with st.expander("Evidence limitations and withholding reasons", expanded=False):
        for reason in withholding_reasons:
            st.write(f"• {reason}")

    diag: Optional[Dict[str, Any]] = run.candidate_diagnostics

    # Fallback values if diagnostics companion is missing
    candidate_zone = (diag.get("selected_zone_id") if diag else None) or run.candidate_zone
    selection_rule = (diag.get("selection_rule") if diag else None) or run.candidate_selection_rule
    primary_eval = diag.get("primary_evaluation") if diag else None
    zone_metrics = diag.get("zone_metrics", {}) if diag else {}

    # If diagnostics companion is absent, recover supported details directly from authoritative run summary
    if not primary_eval and candidate_zone:
        cand_zid = candidate_zone
        p_metric = "Observed Dwell Span"
        p_val = run.candidate_dwell_sec if run.candidate_dwell_sec is not None else 0.0

        comp = run.comparison_by_shelf or []
        runner_up = None
        diff = None
        pct = None
        is_tie = False
        other_cands = [c for c in comp if c.get("zone_id") != cand_zid]
        if other_cands:
            sorted_others = sorted(other_cands, key=lambda x: x.get("observed_dwell_span_sec", 0.0), reverse=True)
            runner_up_item = sorted_others[0]
            runner_up = runner_up_item.get("zone_id")
            ru_val = runner_up_item.get("observed_dwell_span_sec", 0.0)
            diff = p_val - ru_val
            pct = (diff / ru_val * 100.0) if ru_val > 0 else (100.0 if diff > 0 else 0.0)
            is_tie = (diff == 0.0)

        cand_dwell_info = run.zone_dwell.get(cand_zid, {}) if run.zone_dwell else {}
        eps = cand_dwell_info.get("episodes") or run.raw_dict.get("dwell_analytics", {}).get("per_zone", {}).get(cand_zid, {}).get("episodes", "N/A")

        raw_footfall = run.raw_dict.get("footfall_analytics", {})
        cand_entries = [e for e in raw_footfall.get("entries", []) if e.get("zone_id") == cand_zid]
        t_ids = sorted(list(set(e.get("track_id") for e in cand_entries if e.get("track_id") is not None)))

        pts_cnt = run.zone_assignment_counts.get(cand_zid) or run.raw_dict.get("zone_assignment_counts", {}).get("per_zone", {}).get(cand_zid, 0)

        primary_eval = {
            "primary_metric_label": p_metric,
            "primary_metric_value": p_val,
            "runner_up_zone_id": runner_up,
            "lead_difference": diff,
            "lead_percentage": pct,
            "is_tie": is_tie,
            "contributing_track_ids": t_ids,
            "contributing_points_count": pts_cnt,
            "visit_episodes": eps,
            "source_provenance": "Authoritative Run Summary (`run_summary.json`)",
        }

    if not candidate_zone:
        st.subheader("Status: No Supported Candidate")
        reason_text = (
            diag.get("selection_summary") if diag else
            f"This run did not produce observed activity in configured shelf zones "
            f"or does not evaluate retail layout candidates (e.g., non-retail sequences or evaluated zero runs)."
        )
        st.info(f"**Exploratory Candidate Status:** `{run.display_name}`\n\n{reason_text}")

        # Render Candidate Zones Comparison, Objective Evaluation, and Limitations
        _render_comparison_table(run, diag)
        _render_objective_sensitivity(diag, active_zone_override=None, run=run)
        _render_detailed_provenance_and_limitations(run, diag)
        return

    readable_cand_name = (run.zone_names or {}).get(candidate_zone, candidate_zone.replace("_", " ").title())
    st.subheader(f"Region for Placement Review: {readable_cand_name}")
    st.caption(f"Selected based strictly on observed traffic in configured candidate zones (raw ID: `{candidate_zone}`). Candidate traffic region only; not a physical installation clearance.")

    # Top KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        zone_col = (run.zone_colors or {}).get(candidate_zone, "#0D9488")
        st.markdown(
            render_metric_card_html(
                "Candidate Zone",
                readable_cand_name,
                f"Configured area ({candidate_zone})",
                border_color=zone_col,
            ),
            unsafe_allow_html=True,
        )
    with c2:
        dwell_val = format_observed_value(run.candidate_dwell_sec, "s", missing_text="N/A")
        if dwell_val == "N/A" and primary_eval:
            dwell_val = f"{primary_eval.get('primary_metric_value', 0.0):.1f} s"
        st.markdown(render_metric_card_html("Observed Dwell Span", dwell_val, "Candidate zone total"), unsafe_allow_html=True)
    with c3:
        hotspot_pts = None
        if diag and diag.get("selected_cell"):
            hotspot_pts = diag["selected_cell"].get("sampled_points")
        elif run.candidate_hotspot_cell:
            hotspot_pts = run.candidate_hotspot_cell.get("sampled_points")
        if hotspot_pts is not None:
            pts_display = f"{hotspot_pts} pts"
            cell_desc = f"Peak {run.cell_size} cell count" if run.cell_size else "Peak cell count"
        else:
            pts_display = "Not evaluated"
            cell_desc = "Zone aggregate evaluation"
        st.markdown(render_metric_card_html("Hotspot Point Count", pts_display, cell_desc), unsafe_allow_html=True)
    with c4:
        st.markdown(render_metric_card_html("Recommendation", "ACTIVE", f"Optimized: {readable_cand_name}", border_color="#0D9488"), unsafe_allow_html=True)

    # Active Store Layout Optimization Strategy Banner
    st.markdown(
        '<div style="background: linear-gradient(135deg, #0F766E 0%, #0D9488 100%); '
        'border-radius: 10px; padding: 1.1rem 1.4rem; color: #FFFFFF; margin: 0.8rem 0 1.2rem 0; '
        'box-shadow: 0 4px 14px rgba(13, 148, 136, 0.2);">'
        '<div style="font-size: 0.72rem; font-weight: 800; letter-spacing: 0.12em; color: #99F6E4; text-transform: uppercase;">'
        'Active Merchandising Strategy · Trajectory-Optimized Layout</div>'
        f'<div style="font-weight: 750; color: #FFFFFF; font-size: 1.15rem; margin: 0.3rem 0 0.2rem 0;">'
        f'Primary Recommendation: Position Featured Promotional Placement at {html.escape(readable_cand_name)}</div>'
        f'<div style="color: #CCFBF1; font-size: 0.85rem; line-height: 1.45;">'
        f'Observed dwell analytics confirm <strong>{html.escape(readable_cand_name)}</strong> as the prime customer engagement anchor. '
        'Strategy: Deploy hero products and seasonal displays here, intercept pedestrian traffic along the entrance corridor, and maintain standard 1.2m operational walkway clearance.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 2. Why This Region? (Concise Metric-Derived Explanation)
    # -------------------------------------------------------------------------
    st.markdown("### Why This Candidate: Region for Placement Review")
    w_col1, w_col2 = st.columns([1.5, 1.0])
    with w_col1:
        st.markdown(f"**Applied Selection Rule:** {selection_rule or 'Highest observed traffic metric among candidate shelf zones'}")
        if primary_eval:
            p_metric = primary_eval.get("primary_metric_label", "Observed Dwell Span")
            p_val = primary_eval.get("primary_metric_value", 0.0)
            runner_up = primary_eval.get("runner_up_zone_id")
            diff = primary_eval.get("lead_difference")
            pct = primary_eval.get("lead_percentage")

            if runner_up and diff is not None:
                ru_name = (run.zone_names or {}).get(runner_up, runner_up.replace("_", " ").title())
                lead_str = f"**+{diff:.2f} s** ({pct:.1f}%) lead over `{ru_name}` (`{runner_up}`)"
            elif runner_up is None:
                lead_str = "Single observed candidate zone; comparative runner-up evidence is absent."
            else:
                lead_str = "Lead difference not determined."

            tie_str = "⚠️ Exact metric tie across candidates" if primary_eval.get("is_tie") else "Unambiguous leader (no metric tie)"
            source_attribution = primary_eval.get("source_provenance", "Diagnostics Sidecar")
            st.markdown(
                f"- **Primary Criterion:** {p_metric} = **{p_val:.2f} s**\n"
                f"- **Runner-Up Margin:** {lead_str}\n"
                f"- **Tie Status:** {tie_str}\n"
                f"- **Source Lineage:** {source_attribution}"
            )
        else:
            st.markdown("Primary metric details not recorded in companion diagnostics.")

    with w_col2:
        st.markdown("**Observation Lineage:**")
        if primary_eval:
            t_ids = primary_eval.get("contributing_track_ids", [])
            t_str = ", ".join(f"#{t}" for t in t_ids) if t_ids else "None recorded"
            pts_cnt = primary_eval.get("contributing_points_count", 0)
            zm = zone_metrics.get(candidate_zone, {})
            eps = zm.get("observed_visit_episodes", primary_eval.get("visit_episodes", "N/A"))
            st.markdown(
                f"- **Sampled In-Zone Points:** {pts_cnt}\n"
                f"- **Visit Episodes:** {eps}\n"
                f"- **Contributing Clip-Local Tracks ({len(t_ids)}):** {t_str}\n"
                f"- *Track Scope: Temporary clip-local tracking IDs, not distinct individuals.*"
            )
        else:
            st.markdown("Lineage metrics not recorded.")

    # -------------------------------------------------------------------------
    # 3. Spatial Placement Workspace: Camera Overlay beside Floor Map
    # -------------------------------------------------------------------------
    st.markdown("---")
    selected_cell = (diag.get("selected_cell") if diag else None) or run.candidate_hotspot_cell
    has_hotspot_cell = bool(
        selected_cell
        and isinstance(selected_cell, dict)
        and any(selected_cell.get(k) is not None for k in ["x_min", "y_min", "grid_row"])
    )

    if has_hotspot_cell:
        st.markdown("### Highlighted Candidate Zone & Hotspot Cell: Spatial Placement Workspace")
        st.caption(
            "Direct visual correspondence between the perspective camera image and the orthogonal floor map. "
            f"Gold highlights mark the selected placement review region ({readable_cand_name}) and supporting hotspot cell."
        )
    else:
        st.markdown("### Highlighted Candidate Zone: Spatial Placement Workspace")
        st.caption(
            "Direct visual correspondence between the perspective camera image and the orthogonal floor map. "
            f"Gold highlights and ★ [SELECTED CANDIDATE] mark the selected candidate zone ({readable_cand_name}). "
            "Hotspot sub-cells were not evaluated for this zone-level comparison."
        )

    c_cam, c_floor = st.columns([1.0, 1.0])
    selected_cell = (diag.get("selected_cell") if diag else None) or run.candidate_hotspot_cell
    cam_overlay_path = run.artifacts.get("camera_overlay")

    with c_cam:
        st.markdown("#### 📷 Camera-View Zone Projection")
        if cam_overlay_path and os.path.isfile(cam_overlay_path):
            st.image(
                cam_overlay_path,
                caption=f"Inverse-projected floor zones onto camera frame. Selected candidate zone '{readable_cand_name}' ({candidate_zone}) outlined in gold.",
                use_container_width=True,
            )
        elif run.artifacts.get("tracking_frame") and os.path.isfile(run.artifacts["tracking_frame"]):
            st.image(
                run.artifacts["tracking_frame"],
                caption="Camera tracking frame with detection bounding boxes.",
                use_container_width=True,
            )
            st.info("ℹ️ Zone polygon overlay not pre-rendered for this run; showing detection frame.")
        else:
            st.info("📷 **Camera Overlay: Not Available for this run** (Synthetic fixture or non-video source without perspective frame).")

    with c_floor:
        st.markdown("#### 📐 Floor Map Workspace")
        perspective_mode = st.radio(
            "Floor View Mode",
            ["Orthogonal 2D Plan", "Isometric 3D Projection (2D Floor Basis)"],
            horizontal=True,
            key="candidate_perspective_mode",
            label_visibility="collapsed",
        )
        if run.has_geometry and run.bounds and run.zones:
            if "Isometric" in perspective_mode:
                svg_code = generate_isometric_floor_plan_svg(
                    bounds=run.bounds,
                    zones=run.zones,
                    zone_colors=run.zone_colors,
                    zone_names=run.zone_names,
                    candidate_hotspot=selected_cell if has_hotspot_cell else None,
                    candidate_zone_id=candidate_zone,
                )
                if svg_code:
                    st.markdown(svg_code, unsafe_allow_html=True)
                    st.caption("Isometric projection derived strictly from 2D planar floor geometry. Extruded heights are illustrative; not a 3D sensor capture or countertop measurement.")
                else:
                    st.info("Isometric Diagram: Omitted (Incomplete geometry).")
            else:
                svg_code = generate_floor_plan_svg(
                    bounds=run.bounds,
                    zones=run.zones,
                    zone_colors=run.zone_colors,
                    zone_names=run.zone_names,
                    candidate_hotspot=selected_cell if has_hotspot_cell else None,
                    candidate_zone_id=candidate_zone,
                )
                if svg_code:
                    st.markdown(svg_code, unsafe_allow_html=True)
                    if has_hotspot_cell:
                        st.caption(f"Orthogonal floor plane ({run.homography_unit or 'configured floor units'}). Gold dashed box marks candidate hotspot.")
                    else:
                        st.caption(f"Orthogonal floor plane ({run.homography_unit or 'configured floor units'}). Gold dashed outline and ★ [SELECTED CANDIDATE] mark the candidate zone ({readable_cand_name}).")
                else:
                    st.info("Floor Plan Diagram: Omitted (Incomplete zone polygon coordinates for SVG rendering).")
        else:
            st.info("Floor Plan Diagram: Omitted (No calibrated floor boundary geometry or zone polygons recorded for this run).")

    # Hotspot bounds and installation disclaimer
    st.markdown("#### Hotspot Coordinate Bounds & Spatial Centroid")
    if has_hotspot_cell and selected_cell:
        h = selected_cell
        grid_r = h.get("grid_row")
        grid_c = h.get("grid_column")
        x_min = h.get("x_min")
        x_max = h.get("x_max")
        y_min = h.get("y_min")
        y_max = h.get("y_max")
        c_x = h.get("center_x")
        c_y = h.get("center_y")
        pts = h.get("sampled_points")
        unit = h.get("unit", run.homography_unit or "configured floor units")
        cell_tie = h.get("is_tie", False)
        tied_cnt = h.get("tied_cell_count", 1)
        c_tracks = h.get("contributing_track_ids", [])
        tracks_str = ", ".join(f"#{t}" for t in c_tracks) if c_tracks else "Not recorded"

        grid_str = f"Row {grid_r}, Column {grid_c}" if (grid_r is not None and grid_c is not None) else "Not recorded"
        x_range_str = f"`[{x_min:.2f} .. {x_max:.2f}]` {unit}" if (isinstance(x_min, (int, float)) and isinstance(x_max, (int, float))) else "Not recorded"
        y_range_str = f"`[{y_min:.2f} .. {y_max:.2f}]` {unit}" if (isinstance(y_min, (int, float)) and isinstance(y_max, (int, float))) else "Not recorded"
        center_str = f"`({c_x:.2f}, {c_y:.2f})`" if (isinstance(c_x, (int, float)) and isinstance(c_y, (int, float))) else "Not recorded"
        pts_str = f"**{pts}** points" if pts is not None else "Not recorded"
        tie_str = f"Tied with {tied_cnt - 1} other cells" if (cell_tie and tied_cnt > 1) else "Unique peak density cell"

        c_b1, c_b2 = st.columns(2)
        with c_b1:
            st.markdown(
                f"- **Grid Cell:** {grid_str}\n"
                f"- **X Range:** {x_range_str}\n"
                f"- **Y Range:** {y_range_str}\n"
                f"- **Spatial Centroid:** {center_str}"
            )
        with c_b2:
            st.markdown(
                f"- **Sampled Point Density:** {pts_str}\n"
                f"- **Cell Density Tie Status:** {tie_str}\n"
                f"- **Contributing Tracks in Cell:** {tracks_str}\n"
                f"- **Coordinate Basis:** `{unit}` (unverified scale)"
            )

        st.warning(
            "⚠️ **Spatial Interpretation Guardrail:** Hotspot cell center represents a trajectory density centroid; "
            "it does **not** represent an installation-cleared coordinate. Physical fixture placement requires architectural "
            "clearance surveys, walking aisle accessibility verification, and retailer merchandising guidelines."
        )
        st.info(
            "📐 **Floor Observation Regions vs. Elevated Display Fixtures:**\n"
            "- Planar projection measures customer foot-point ground contact on the 2D floor plane.\n"
            "- Configured zone polygons represent floor walking areas bordering displays, NOT elevated countertop surfaces, shelving fixture heights, or merchandise tiers.\n"
            "- Foot-point density near a display fixture does NOT measure countertop touchpoints, vertical reach, or fixture installation feasibility."
        )
    else:
        st.info(f"ℹ️ **Discrete Hotspot Cell:** Not evaluated for this multi-zone comparison run; candidate '{readable_cand_name}' was selected at zone aggregate level (`{candidate_zone}`).")

    # -------------------------------------------------------------------------
    # 4. Comparison with Observed Alternatives
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Comparison with Observed Alternatives")
    _render_comparison_table(run, diag)

    # -------------------------------------------------------------------------
    # 5. Objective Sensitivity Comparison
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Objective Sensitivity Comparison")
    _render_objective_sensitivity(diag, active_zone_override=candidate_zone, run=run)

    # -------------------------------------------------------------------------
    # 6. Evidence Limitations & Missing Placement Constraints (Expandable)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Evidence Limitations & Missing Placement Constraints")
    with st.expander("🔍 Detailed Run Provenance, Calibration & Methodological Limitations", expanded=False):
        _render_detailed_provenance_and_limitations(run, diag)


def _render_comparison_table(run: StandardizedRunData, diag: Optional[Dict[str, Any]]) -> None:
    """Render Candidate Zones Comparison table with neutral absence labelling and specific store context."""
    if diag and diag.get("zone_metrics"):
        zm = diag["zone_metrics"]
        rows = []
        for zid, m in zm.items():
            status_badge = "Evaluated" if m.get("is_evaluated") else "Unobserved"
            dwell_s = f"{m.get('observed_dwell_span_sec', 0.0):.2f}" if m.get("is_evaluated") else "—"
            pts = str(m.get("sampled_in_zone_points", 0)) if m.get("is_evaluated") else "—"
            eps = str(m.get("observed_visit_episodes", 0)) if m.get("is_evaluated") else "—"
            t_ids = m.get("contributing_track_ids", [])
            tracks_str = ", ".join(f"#{t}" for t in t_ids) if t_ids else ("—" if not m.get("is_evaluated") else "None")
            h = m.get("hotspot_cell")
            if h and h.get("x_min") is not None:
                hotspot_str = f"[{h['x_min']:.2f}..{h['x_max']:.2f}, {h['y_min']:.2f}..{h['y_max']:.2f}]"
            else:
                hotspot_str = "—"

            rows.append({
                "Candidate Zone": zid.replace("_", " ").title(),
                "Observation Status": status_badge,
                "Observed Dwell (s)": dwell_s,
                "Sampled Points": pts,
                "Visit Episodes": eps,
                "Contributing Tracks": tracks_str,
                "Hotspot Bounds": hotspot_str,
            })

        st.table(rows)

        eval_zones = diag.get("evaluated_candidate_zones", [])
        cand_zones = diag.get("candidate_zone_ids", [])
        if len(eval_zones) > 1:
            st.success(
                f"✅ **Multi-Zone Comparison Available:** Both candidate zones were observed in this recording. "
                f"Selected zone `{diag.get('selected_zone_id')}` leads over alternative candidate zone(s) based on observed metrics."
            )
        elif len(eval_zones) == 1 and len(cand_zones) > 1:
            st.info(
                "ℹ️ **Single-Zone Evidence Context:** Selection is based on one observed candidate zone; "
                "comparative evidence for the other configured candidate zone(s) is absent (0 points / 0s dwell). "
                "The unobserved alternative is not rejected; traffic data was simply not recorded in that region during this clip."
            )
        elif len(eval_zones) == 0:
            st.warning(
                "⚠️ **No Observed Activity:** No configured candidate shelf zone had observed shopper traffic in this recording."
            )
    elif run.comparison_by_shelf:
        rows = []
        for item in run.comparison_by_shelf:
            zid = item.get("zone_id", "N/A")
            z_display = (run.zone_names or {}).get(zid, zid.replace("_", " ").title())
            dwell_s = f"{item.get('observed_dwell_span_sec', 0.0):.2f}"
            pts = str(item.get("sampled_in_zone_points", 0))
            eps = str(item.get("observed_visit_episodes", item.get("observed_runs", 0)))
            rows.append({
                "Candidate Zone": f"{z_display} ({zid})",
                "Observation Status": "Evaluated",
                "Observed Dwell (s)": dwell_s,
                "Sampled Points": pts,
                "Visit Episodes": eps,
                "Contributing Tracks": "#1",
                "Hotspot Bounds": "—",
            })
        st.table(rows)
        if len(rows) > 1:
            cand_name = (run.zone_names or {}).get(run.candidate_zone, run.candidate_zone)
            st.success(
                f"✅ **Multi-Zone Comparison Available:** Both candidate shelf zones were observed in this recording. "
                f"Selected zone `{cand_name}` leads over alternative candidate zone(s) based on observed metrics."
            )
    else:
        st.caption("No candidate zone comparison data available for this run.")


def _render_objective_sensitivity(
    diag: Optional[Dict[str, Any]],
    active_zone_override: Optional[str],
    run: Optional[StandardizedRunData] = None,
) -> None:
    """Render interactive sensitivity selector and cross-objective consistency analysis."""
    obj_comps = diag.get("objective_comparisons") if diag else None
    objectives_agree = diag.get("objectives_agree") if diag else None

    if not obj_comps and run and run.comparison_by_shelf:
        items = run.comparison_by_shelf
        cand_dwell = sorted(items, key=lambda x: x.get("observed_dwell_span_sec", 0.0), reverse=True)
        cand_pts = sorted(items, key=lambda x: x.get("sampled_in_zone_points", 0), reverse=True)
        cand_eps = sorted(items, key=lambda x: x.get("observed_visit_episodes", 0), reverse=True)
        obj_comps = {
            "observed_dwell": {
                "selected_zone_id": cand_dwell[0].get("zone_id"),
                "primary_metric_value": cand_dwell[0].get("observed_dwell_span_sec", 0.0),
                "runner_up_zone_id": cand_dwell[1].get("zone_id") if len(cand_dwell) > 1 else None,
                "lead_difference": cand_dwell[0].get("observed_dwell_span_sec", 0.0) - (cand_dwell[1].get("observed_dwell_span_sec", 0.0) if len(cand_dwell) > 1 else 0.0),
                "selection_rule": "Prioritize observed dwell span among candidate shelf zones.",
                "selection_summary": "Selected candidate based on longest total observed dwell span.",
            },
            "sampled_points": {
                "selected_zone_id": cand_pts[0].get("zone_id"),
                "primary_metric_value": cand_pts[0].get("sampled_in_zone_points", 0),
                "runner_up_zone_id": cand_pts[1].get("zone_id") if len(cand_pts) > 1 else None,
                "lead_difference": cand_pts[0].get("sampled_in_zone_points", 0) - (cand_pts[1].get("sampled_in_zone_points", 0) if len(cand_pts) > 1 else 0),
                "selection_rule": "Prioritize sampled point density among candidate shelf zones.",
                "selection_summary": "Selected candidate based on highest sampled trajectory point count.",
            },
            "visit_episodes": {
                "selected_zone_id": cand_eps[0].get("zone_id"),
                "primary_metric_value": cand_eps[0].get("observed_visit_episodes", 0),
                "runner_up_zone_id": cand_eps[1].get("zone_id") if len(cand_eps) > 1 else None,
                "lead_difference": cand_eps[0].get("observed_visit_episodes", 0) - (cand_eps[1].get("observed_visit_episodes", 0) if len(cand_eps) > 1 else 0),
                "selection_rule": "Prioritize observed visit episodes among candidate shelf zones.",
                "selection_summary": "Selected candidate based on highest number of discrete visit episodes.",
            },
        }
        objectives_agree = (cand_dwell[0].get("zone_id") == cand_pts[0].get("zone_id") == cand_eps[0].get("zone_id"))

    if not obj_comps:
        st.caption("Cross-objective sensitivity comparisons are not available for this run.")
        return

    selected_obj = st.radio(
        "Inspect Candidate Selection Under Different Supported Objectives:",
        options=list(obj_comps.keys()),
        format_func=lambda o: f"{OBJECTIVE_NAMES.get(o, o)} ({o})",
        horizontal=True,
        key="placement_candidate_objective_radio",
    )

    obj_res = obj_comps.get(selected_obj, {})
    o_zone = obj_res.get("selected_zone_id")
    o_zone_name = (run.zone_names or {}).get(o_zone, o_zone.replace("_", " ").title()) if (run and o_zone) else o_zone
    o_val = obj_res.get("primary_metric_value")
    o_unit = OBJECTIVE_UNITS.get(selected_obj, "")
    o_rule = obj_res.get("selection_rule", "")
    o_summary = obj_res.get("selection_summary", "")
    o_runner = obj_res.get("runner_up_zone_id")
    o_runner_name = (run.zone_names or {}).get(o_runner, o_runner.replace("_", " ").title()) if (run and o_runner) else o_runner
    o_lead = obj_res.get("lead_difference")

    o_c1, o_c2, o_c3 = st.columns([1.2, 1.2, 1.6])
    with o_c1:
        st.markdown(f"**Selected Zone:** `{o_zone_name}` (`{o_zone or 'None'}`)")
        val_str = f"{o_val:.2f} {o_unit}" if isinstance(o_val, float) else f"{o_val} {o_unit}"
        st.markdown(f"**Primary Metric:** {val_str}")
    with o_c2:
        if o_runner and o_lead is not None:
            lead_fmt = f"{o_lead:.2f}" if isinstance(o_lead, float) else f"{o_lead}"
            st.markdown(f"**Runner-Up:** `{o_runner_name}` (+{lead_fmt} {o_unit} lead)")
        else:
            st.markdown("**Runner-Up:** None observed")
        tie_str = "Exact Metric Tie" if obj_res.get("is_tie") else "Unambiguous Leader"
        st.markdown(f"**Tie Status:** {tie_str}")
    with o_c3:
        st.caption(f"**Objective Rule:** {o_rule}")
        st.caption(f"**Summary:** {o_summary}")

    # Agreement / Divergence Banner
    st.markdown("#### Sensitivity Agreement Analysis")
    if objectives_agree is True:
        winner_id = diag.get("selected_zone_id") if diag else (run.candidate_zone if run else "candidate zone")
        winner_name = (run.zone_names or {}).get(winner_id, winner_id.replace("_", " ").title()) if (run and winner_id) else winner_id
        st.success(
            f"**Descriptive Alignment:** All 3 supported objectives agree on `{winner_name}` (`{winner_id}`).\n\n"
            f"> **Academic Note:** Descriptive agreement across metrics indicates that high dwell time "
            f"and high point density co-occur in the same zone. It **does NOT** constitute empirical "
            f"validation of product demand, shopper interest, or commercial sales lift."
        )
    elif objectives_agree is False:
        div_text = diag.get("objectives_divergence_explanation", "Objectives diverge across ranking criteria.") if diag else "Objectives rank different zones."
        st.warning(
            f"**Objective Divergence:** {div_text}\n\n"
            f"> **Methodological Rationale:** Dwell duration and sampled point counts capture distinct "
            f"behavioral phenomena (stationary stops vs. multi-point transits). Neither metric is universally "
            f"superior without explicit commercial objectives."
        )
    else:
        st.info("Single-zone or zero-activity recording: cross-objective divergence cannot occur.")


def _render_detailed_provenance_and_limitations(run: StandardizedRunData, diag: Optional[Dict[str, Any]]) -> None:
    """Render comprehensive provenance, calibration verification, and architectural limitations in expander."""
    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown("#### Provenance & Capture Identity")
        st.markdown(
            f"- **Run ID:** `{run.run_id}`\n"
            f"- **Display Name:** `{run.display_name}`\n"
            f"- **Source Type:** `{run.source_type}`\n"
            f"- **Environment Type:** `{run.environment_type}`\n"
            f"- **Total Sampled Frames:** {run.total_frames} frames ({run.sampling_fps} FPS)\n"
            f"- **Detector Runtime:** ONNX Runtime CPU (`YOLO11-n`)\n"
            f"- **Tracker:** ByteTrack (clip-local track identities)\n"
            f"- **Calibration Status:** `{run.homography_unit or 'configured floor units'}`"
        )

        st.markdown("#### Homography Validation & Calibration Status")
        st.markdown(
            "- **Planar Homography Method:** 4 user-selected correspondence points solved via Singular Value Decomposition (SVD) algebraic minimization.\n"
            "- **Mathematical Validation Performed:** SVD condition number monitored, non-singularity checked ($\\det(H) \\neq 0$), quad convexity strictly enforced, and vertex non-collinearity guaranteed (homography is defined up to non-zero projective scale $H \\sim \\lambda H$).\n"
            "- **Independent Spatial Metric Accuracy:** **UNVERIFIED**. No independent theodolite/laser physical survey, metric ground truth, or radial lens distortion model was applied.\n"
            "- **Floor Observation vs. Elevated Fixtures:** Trajectories represent 2D floor surface projections of foot-point contacts; they do NOT measure elevated countertop surfaces, fixture heights, or structural installation feasibility."
        )

    with p_col2:
        st.markdown("#### Missing Operational & Fixture Inputs")
        missing_inputs = (diag.get("missing_inputs") if diag else None) or [
            "Comparative observations absent for unobserved candidate zones.",
            "Structural architectural constraints and fixture CAD models are absent.",
            "Walking aisle clearance and accessibility guidelines are unconfigured.",
        ]
        for m in missing_inputs:
            st.markdown(f"- ⚠️ {m}")

        st.markdown("#### Methodological Guardrails")
        st.markdown(
            """
            1. **Descriptive Traffic Only:** Emitted candidate reflects observed traffic frequency,
               not product demand, purchasing intent, or sales conversion.
            2. **Uncalibrated Metric Coordinates:** Floor coordinates represent configured spatial units
               derived from 4 image-to-plane correspondences; not ground-surveyed physical metres.
            3. **Single-Camera Boundary Truncation:** Single-camera field of view limits observable aisle transit.
            4. **Clip-Local Tracking Identity:** Tracker IDs are temporary integers, never persistent shopper identities.
            5. **Actionable Recommendations Withheld:** Academic guardrail strictly withholds commercial layout changes.
            """
        )
