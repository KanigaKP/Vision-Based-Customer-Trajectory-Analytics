"""Explain one saved execution using its own recorded measurements."""
import streamlit as st
from dashboard.run_adapter import format_observed_value


def render_run_journey(run):
    with st.expander("Understand this run · Input → Tracking → Floor → Measurements → Candidate", expanded=True):
        st.caption(f"Every stage below belongs to {run.run_id}. Unavailable means the value was not recorded.")
        stages = [
            ("1 · Recording", run.sampled_frame_count if run.sampled_frame_count is not None else run.total_frames, "sampled frames", "Frames selected from this recording for processing."),
            ("2 · Detection & tracking", run.total_tracks, "clip-local tracks", f"{format_observed_value(run.total_detections)} detections. One track may contain observation gaps."),
            ("3 · Floor projection", run.total_points, "trajectory points", f"{format_observed_value(run.binned_points)} inside density bounds; {format_observed_value(run.out_of_bounds_points)} outside."),
            ("4 · Zone measurements", run.total_dwell_sec, "seconds observed span", f"{format_observed_value(run.total_episodes)} zone episodes; {format_observed_value(run.total_transitions)} direct transitions."),
        ]
        for column, (title, value, unit, explanation) in zip(st.columns(4), stages):
            with column:
                st.markdown(f"**{title}**")
                st.metric(unit, format_observed_value(value))
                st.caption(explanation)
        if run.total_points and run.out_of_bounds_points == run.total_points:
            st.error("Every projected point falls outside this run's floor bounds. Review the camera reference points against this recording and inspect the projected floor overlay before interpreting zone metrics. Detection results remain available below.")
        names = run.zone_names or {}
        if run.candidate_zone:
            name = names.get(run.candidate_zone, run.candidate_zone.replace("_", " ").title())
            st.markdown(f"**5 · Placement review — {name}**")
            st.write(f"Selected-zone observed span: {format_observed_value(run.candidate_dwell_sec, 's')}. Whole-run observed span: {format_observed_value(run.total_dwell_sec, 's')}.")
            st.caption(run.candidate_selection_rule or "Selection rule was not recorded for this run.")
        else:
            st.info("5 · Placement review — no candidate recorded for this run.")
        rows = []
        for zone, metrics in run.zone_dwell.items():
            rows.append({"Zone": names.get(zone, zone.replace("_", " ").title()), "Zone ID": zone,
                         "Assigned points": run.zone_assignment_counts.get(zone),
                         "Observed span (s)": metrics.get("observed_dwell_span_sec", metrics.get("total_dwell_sec")),
                         "Episodes": metrics.get("episodes", metrics.get("episode_count")),
                         "Selected candidate": zone == run.candidate_zone})
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption("Density counts sampled positions. Dwell sums observed episode spans. These measures have different units and are not interchangeable. The isometric view represents the same configured 2D floor; its height is illustrative.")
        with st.expander("Recorded input and output connections"):
            st.write("Source", run.source_file_path or "Not recorded")
            st.write("Coordinate units", run.homography_unit)
            st.json({key: path for key, path in run.artifacts.items() if path})


def render_pipeline_comparison(runs):
    """Compare artifacts by execution identity, keeping every column on one run."""
    import os
    with st.expander("Compare complete visual pipelines", expanded=False):
        st.caption("Choose up to three executions. Each column keeps its own camera, geometry, measurements and outputs. Different clip lengths and camera coverage affect the counts.")
        ids = [run.run_id for run in runs]
        labels = {run.run_id: run.display_name for run in runs}
        selected = st.multiselect("Runs to compare", ids, format_func=lambda value: labels[value], max_selections=3, key="pipeline_comparison_runs")
        if not selected:
            st.info("Select recordings above to align their visual outputs stage by stage.")
            return
        selected_runs = [next(run for run in runs if run.run_id == value) for value in selected]
        for column, run in zip(st.columns(len(selected_runs)), selected_runs):
            with column:
                st.markdown(f"**{run.display_name}**")
                st.caption(run.run_id)
                st.write(f"{format_observed_value(run.total_frames)} sampled frames · {format_observed_value(run.sampling_fps)} FPS")
        stages = [("Tracking · detected people and clip-local IDs", "tracking_frame"), ("Projection · trajectories in this floor model", "floor_trajectory"), ("Density · sampled positions per grid cell", "density_heatmap"), ("Dwell · observed zone spans", "zone_dwell")]
        for title, artifact_key in stages:
            st.markdown(f"**{title}**")
            for column, run in zip(st.columns(len(selected_runs)), selected_runs):
                with column:
                    path = run.artifacts.get(artifact_key)
                    if path and os.path.isfile(path):
                        st.image(path, caption=run.display_name, use_container_width=True)
                    else:
                        st.info("Artifact not recorded for this run.")
        for column, run in zip(st.columns(len(selected_runs)), selected_runs):
            with column:
                st.markdown("**Measured outcome**")
                st.write(f"Assigned points: {format_observed_value(run.in_zone_points)} / {format_observed_value(run.total_points)}")
                st.write(f"Whole-run span: {format_observed_value(run.total_dwell_sec, 's')} · Transitions: {format_observed_value(run.total_transitions)}")
                st.write(f"Candidate: {run.candidate_zone or 'Not recorded'} · Candidate span: {format_observed_value(run.candidate_dwell_sec, 's')}")
                st.caption(run.candidate_selection_rule or "Selection rule not recorded.")
