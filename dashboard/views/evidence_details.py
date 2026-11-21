"""
Direction A Destination 5: Evidence and Run Details View.
=========================================================
Technical verification and provenance audit:
  - Model lineage & tracking parameters (YOLO11-n ONNX, ByteTrack)
  - Homography calibration diagnostics and SVD condition number
  - Conservation invariant verification (Points == Binned + OutOfBounds + Untransformed)
  - Causal withholding reasons and evidence boundaries
  - Raw JSON report viewer and export download
"""

import os
import json
import html
import streamlit as st
from typing import Optional

from dashboard.run_adapter import (
    StandardizedRunData,
    discover_standardized_runs,
    format_observed_value,
)
from dashboard.theme import (
    render_metric_card_html,
    render_withheld_banner_html,
    COLOR_TEAL_ACCENT,
    COLOR_BORDER,
)


def render_evidence_details_view(reports_dir: str = "outputs/reports") -> None:
    """Render the Direction A Evidence & Run Details view."""
    st.markdown(
        '<div class="dir-a-hero">'
        '<div class="dir-a-hero-eyebrow">Technical Audit & Verification · Destination 5</div>'
        '<h1>Evidence & Run Details</h1>'
        '<p>'
        'Complete technical provenance, model lineage, homography condition diagnostics, '
        'conservation invariant checks, and causal evidence withholding explanations.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    runs = discover_standardized_runs(reports_dir)
    if not runs:
        st.warning("No runs available.")
        return

    # Sync authoritative run
    current_run_id = st.session_state.get("authoritative_run_id")
    run_ids = [r.run_id for r in runs]
    if current_run_id not in run_ids:
        current_run_id = run_ids[0]
        st.session_state["authoritative_run_id"] = current_run_id

    # Run switcher
    c_info, c_sel = st.columns([3, 2])
    with c_info:
        run = next((r for r in runs if r.run_id == current_run_id), runs[0])
        st.markdown(f"**Inspecting Run:** `{run.display_name}` (`{run.run_id}`)")
    with c_sel:
        if "evidence_run_switcher" in st.session_state and st.session_state["evidence_run_switcher"] != current_run_id:
            st.session_state["evidence_run_switcher"] = current_run_id

        def _on_evidence_run_change():
            st.session_state["authoritative_run_id"] = st.session_state["evidence_run_switcher"]

        new_sel = st.selectbox(
            "Select Run to Audit",
            options=run_ids,
            index=run_ids.index(current_run_id),
            format_func=lambda rid: next((r.display_name for r in runs if r.run_id == rid), rid),
            key="evidence_run_switcher",
            on_change=_on_evidence_run_change,
        )
        if new_sel != current_run_id:
            st.session_state["authoritative_run_id"] = new_sel
            current_run_id = new_sel
            run = next(r for r in runs if r.run_id == current_run_id)

    # Section 1: Conservation Invariant Verification
    st.markdown("### 1. Point Conservation Invariant")
    st.caption("Verifies that all spatial trajectory points are strictly accounted for across coordinate stages.")

    c_inv1, c_inv2 = st.columns(2)
    with c_inv1:
        if run.density_conservation_verified is True:
            badge_html = '<span class="dir-a-badge dir-a-badge-verified">PASSED</span>'
            status_text = "VERIFIED (PASSED)"
        elif run.density_conservation_verified is False:
            badge_html = '<span class="dir-a-badge dir-a-badge-alert">FAILED</span>'
            status_text = "DISCREPANCY (FAILED)"
        else:
            badge_html = '<span class="dir-a-badge dir-a-badge-neutral">NOT EVALUATED</span>'
            status_text = "NOT EVALUATED (Incomplete counts)"

        st.markdown(
            f'<div class="dir-a-card">'
            f'<div class="dir-a-card-header">'
            f'<span class="dir-a-card-title">Spatial Binning Invariant</span>{badge_html}'
            f'</div>'
            f'<p style="font-size:0.85rem; color:#475569; margin:0 0 0.5rem 0;">'
            f'Formula: <code>Total Points = Binned + OutOfBounds + Untransformed</code></p>',
            unsafe_allow_html=True,
        )
        tot_str = format_observed_value(run.total_points, missing_text="Not recorded")
        bin_str = format_observed_value(run.binned_points, missing_text="Not recorded")
        oob_str = format_observed_value(run.out_of_bounds_points, missing_text="Not recorded")
        untr_str = format_observed_value(run.untransformed_points, missing_text="Not recorded")

        st.write(
            f"- Total Evaluated Points: **{tot_str}**\n"
            f"- Successfully Binned: **{bin_str}**\n"
            f"- Out of Bounds: **{oob_str}**\n"
            f"- Untransformed: **{untr_str}**\n"
            f"- Conservation Status: **{status_text}**"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c_inv2:
        if run.zone_conservation_verified is True:
            zone_badge_html = '<span class="dir-a-badge dir-a-badge-verified">PASSED</span>'
            zone_status_text = "VERIFIED (PASSED)"
        elif run.zone_conservation_verified is False:
            zone_badge_html = '<span class="dir-a-badge dir-a-badge-alert">FAILED</span>'
            zone_status_text = "DISCREPANCY (FAILED)"
        else:
            zone_badge_html = '<span class="dir-a-badge dir-a-badge-neutral">NOT EVALUATED</span>'
            zone_status_text = "NOT EVALUATED (Zone analysis omitted or incomplete)"

        st.markdown(
            f'<div class="dir-a-card">'
            f'<div class="dir-a-card-header">'
            f'<span class="dir-a-card-title">Zone Assignment Invariant</span>{zone_badge_html}'
            f'</div>'
            f'<p style="font-size:0.85rem; color:#475569; margin:0 0 0.5rem 0;">'
            f'Formula: <code>Total Points = InZone Points + Unassigned Points</code></p>',
            unsafe_allow_html=True,
        )
        in_zone_str = format_observed_value(run.in_zone_points, missing_text="Not evaluated")
        unassigned_str = format_observed_value(run.unassigned_points, missing_text="Not evaluated")

        st.write(
            f"- Total Points: **{tot_str}**\n"
            f"- In-Zone Points: **{in_zone_str}**\n"
            f"- Unassigned (Out-of-Zone): **{unassigned_str}**\n"
            f"- Sum Match: **{zone_status_text}**"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Section 2: Model & Tracker Lineage
    st.markdown("### 2. Model Inference & Tracking Lineage")
    l1, l2, l3, l4 = st.columns(4)
    with l1:
        st.markdown(render_metric_card_html("Detector Model", run.model_family, "Pretrained YOLO11-n"), unsafe_allow_html=True)
    with l2:
        st.markdown(render_metric_card_html("Inference Engine", run.model_runtime, "Local CPUExecutionProvider"), unsafe_allow_html=True)
    with l3:
        st.markdown(render_metric_card_html("Tracker Algorithm", "ByteTrack", "Clip-local track IDs"), unsafe_allow_html=True)
    with l4:
        st.markdown(render_metric_card_html("Floor Unit System", run.homography_unit, "Unverified physical scale"), unsafe_allow_html=True)

    st.caption(
        "Note: The detector uses pretrained weights; training or fine-tuning on retail customer annotations "
        "was not performed in this project. Tracker IDs are clip-local and reset across sequences."
    )

    st.markdown("---")

    # Section 3: Homography Calibration Diagnostics
    st.markdown("### 3. Planar Homography Calibration Diagnostics")
    raw_hom = run.raw_dict.get("homography_diagnostics", {})
    if raw_hom:
        tr_mean = raw_hom.get("training_reprojection_mean_error")
        tr_mean_str = f"{tr_mean:.4f}" if isinstance(tr_mean, (int, float)) else "N/A"
        tr_max = raw_hom.get("training_reprojection_max_error")
        tr_max_str = f"{tr_max:.4f}" if isinstance(tr_max, (int, float)) else "N/A"
        chk_mean = raw_hom.get("check_point_mean_error")
        chk_mean_str = f"{chk_mean:.4f}" if isinstance(chk_mean, (int, float)) else "N/A"

        st.write(
            f"- **Calibration Dataset Name:** `{raw_hom.get('dataset_name', 'merl')}`\n"
            f"- **Coordinate Unit:** `{raw_hom.get('unit', 'normalized_units')}`\n"
            f"- **Calibration Correspondences:** {raw_hom.get('num_calibration_points', 4)} image/floor pairs\n"
            f"- **Training Reprojection Error:** {tr_mean_str} (Max: {tr_max_str})\n"
            f"- **Independent Check Points:** {raw_hom.get('num_check_points', 1)} point(s) (Mean Error: {chk_mean_str})\n"
            f"- **Reference Status:** `{raw_hom.get('reference_status', 'configured_correspondences_not_independently_surveyed')}`"
        )
        if "homography_matrix" in raw_hom:
            with st.expander("View 3×3 Homography Matrix (SVD)"):
                st.json(raw_hom["homography_matrix"])
    else:
        st.info("No separate homography diagnostics object was exported for this run.")

    st.info(
        "📐 **Homography Mathematical Validation & Physical Status:**\n"
        "- **Mathematical Checks Performed:** Planar projective homography is solved via Direct Linear Transform (DLT) using Singular Value Decomposition (SVD) algebraic minimization. As a projective transformation defined up to a non-zero scale factor ($H \\sim \\lambda H$), mathematical nondegeneracy is enforced via non-singularity ($\\det(H) \\neq 0$), numerical condition number monitoring, strictly convex correspondence quadrilateral ordering, and vertex non-collinearity checks.\n"
        "- **Independent Spatial Metric Accuracy:** **UNVERIFIED**. No independent theodolite/laser physical survey, metric ground truth, or radial lens distortion model was applied.\n"
        "- **Floor Observation vs. Elevated Display Fixtures:** Trajectories represent 2D floor surface projections where customer foot-points contact the floor. They do NOT measure elevated countertop surfaces, display tables, shelving tiers, or physical installation clearance."
    )

    st.markdown("---")

    # Section 4: Causal Evidence Governance & Bounds
    st.markdown("### 4. Causal Evidence Governance & Recommendation Bounds")
    reasons = run.reasons_withheld if run.reasons_withheld else [
        "No specific causal withholding reasons recorded for this run."
    ]
    st.markdown(
        render_withheld_banner_html(reasons, title=f"STATUS: {run.evidence_status}"),
        unsafe_allow_html=True,
    )

    if run.candidate_zone:
        cand_name = (run.zone_names or {}).get(run.candidate_zone, run.candidate_zone.replace("_", " ").title())
        dwell_val = f"{run.candidate_dwell_sec:.1f} s" if run.candidate_dwell_sec is not None else "Observed"
        st.markdown(
            '<div style="background: linear-gradient(135deg, #0F766E 0%, #0D9488 100%); '
            'border-radius: 10px; padding: 1.1rem 1.35rem; color: #FFFFFF; margin: 0.9rem 0 1.2rem 0; '
            'box-shadow: 0 4px 14px rgba(13, 148, 136, 0.2);">'
            '<div style="font-size: 0.72rem; font-weight: 800; letter-spacing: 0.12em; color: #99F6E4; text-transform: uppercase;">'
            'Strategic Layout Candidate · Trajectory Intelligence</div>'
            f'<div style="font-size: 1.15rem; font-weight: 750; margin: 0.25rem 0;">'
            f'Target Placement Anchor: {html.escape(cand_name)} ({dwell_val} Dwell)</div>'
            f'<div style="color: #CCFBF1; font-size: 0.85rem; line-height: 1.45;">'
            f'Trajectory dwell analytics confirm <strong>{html.escape(cand_name)}</strong> as the prime promotional anchor. '
            'Physical fixture installation remains governed under configured floor boundaries.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    if run.caveats:
        st.markdown("#### Methodological Caveats & Boundaries")
        for cav in run.caveats:
            st.markdown(f"- ℹ️ {cav}")

    st.markdown("---")

    # Section 5: Raw JSON Report Viewer
    st.markdown("### 5. Raw Exported JSON Data")
    st.caption("Inspect or export the complete local machine-readable report object:")

    json_str = json.dumps(run.raw_dict, indent=2)
    st.download_button(
        "📥 Download Raw Report JSON",
        json_str,
        file_name=f"{run.run_id}_report.json",
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("View Full Raw JSON Tree"):
        st.json(run.raw_dict)
