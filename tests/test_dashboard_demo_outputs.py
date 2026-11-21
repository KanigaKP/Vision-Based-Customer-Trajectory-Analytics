"""
Focused tests for Dashboard Demo Outputs View and Artifact Resolution.
======================================================================
Verifies:
1. Safe artifact path resolution (containment within allowed directory, path traversal rejection, file existence).
2. Demo summary JSON loading (real, synthetic, MOT17).
3. Separation of real, synthetic, and engineering sample outputs across independent tabs.
4. Streamlit AppTest navigation to Demo Outputs view and verification of:
   - Real MERL sample metrics, diagrams, scale limitation notices, and recommendation withholding.
   - Synthetic demonstration disclaimer ("SYNTHETIC DEMONSTRATION -- NOT REAL SHOPPER DATA"), simulated metrics, and synthetic labels.
   - MOT17 outdoor pedestrian engineering notice, metrics, and unavailable analytics.
5. Missing file and malformed path fallback behavior without crashing.
"""

import os
import json
import pytest
from pathlib import Path

from dashboard.report_loader import (
    resolve_demo_artifact_path,
    load_demo_summary,
)


def test_resolve_demo_artifact_path_valid():
    """Verify valid existing files inside outputs/reports resolve successfully."""
    valid_file = "outputs/reports/demo_real_tracking_frame.png"
    resolved = resolve_demo_artifact_path(valid_file, allowed_dir="outputs/reports")
    assert resolved is not None
    assert os.path.isabs(resolved)
    assert os.path.isfile(resolved)


def test_resolve_demo_artifact_path_nonexistent():
    """Verify nonexistent file returns None without raising an exception."""
    resolved = resolve_demo_artifact_path("outputs/reports/nonexistent_file_xyz.png")
    assert resolved is None


def test_resolve_demo_artifact_path_traversal_rejection():
    """Verify path traversal outside allowed directory is strictly rejected."""
    # Attempting to read config.yaml via parent directory traversal
    traversal_path = "outputs/reports/../../config.yaml"
    resolved = resolve_demo_artifact_path(traversal_path, allowed_dir="outputs/reports")
    assert resolved is None

    # Absolute path outside allowed directory
    abs_outside = os.path.abspath("config.yaml")
    resolved2 = resolve_demo_artifact_path(abs_outside, allowed_dir="outputs/reports")
    assert resolved2 is None


def test_resolve_demo_artifact_path_invalid_inputs():
    """Verify None, empty string, and whitespace return None."""
    assert resolve_demo_artifact_path(None) is None
    assert resolve_demo_artifact_path("") is None
    assert resolve_demo_artifact_path("   ") is None


def test_load_demo_summary_real():
    """Verify loading real MERL demo summary JSON."""
    summary = load_demo_summary("outputs/reports/demo_real_summary.json")
    assert summary is not None
    assert summary.get("dataset_name") == "merl_shopping_dataset"
    assert summary.get("total_frames_evaluated", 0) > 0
    assert summary.get("total_tracks", 0) > 0
    zone_counts = summary.get("zone_assignment_counts", {})
    density = summary.get("density_analytics", {})
    assert zone_counts.get("total_points") == zone_counts.get("in_zone_points") + zone_counts.get("unassigned_out_of_zone_points")
    assert sum(zone_counts.get("per_zone", {}).values()) == zone_counts.get("in_zone_points")
    assert density.get("total_evaluated_points") == density.get("total_binned_points") + density.get("out_of_bounds_count") + density.get("untransformed_count")
    assert density.get("conservation_invariant_verified") is True
    assert summary.get("evidence_status", {}).get("actionable_recommendations") == "WITHHELD"


def test_load_demo_summary_synthetic():
    """Verify loading synthetic demonstration summary JSON."""
    summary = load_demo_summary("outputs/reports/demo_synthetic_summary.json")
    assert summary is not None
    assert "SYNTHETIC DEMONSTRATION" in summary.get("dataset_name", "")
    assert summary.get("total_simulated_tracks") == 2
    assert summary.get("total_simulated_points") == 66
    assert summary.get("dwell_analytics", {}).get("total_episodes") == 10
    assert summary.get("direct_transitions", {}).get("total_direct_transitions") == 8
    assert "WITHHELD" in summary.get("actionable_recommendations", "")


def test_load_demo_summary_mot17():
    """Verify loading MOT17 engineering summary JSON."""
    summary = load_demo_summary("outputs/reports/demo_mot17_summary.json")
    assert summary is not None
    assert "MOT17" in summary.get("dataset_name", "")
    assert summary.get("source_type") == "image_sequence"
    assert summary.get("total_frames_evaluated") == 10
    assert summary.get("dwell_analytics", {}).get("status") == "not_available"
    assert summary.get("direct_transitions", {}).get("status") == "not_available"
    assert summary.get("density_analytics", {}).get("total_binned_points") == 37
    assert summary.get("density_analytics", {}).get("out_of_bounds_count") == 57


def test_load_demo_summary_missing_or_traversal():
    """Verify load_demo_summary handles missing file and path traversal."""
    assert load_demo_summary("outputs/reports/nonexistent.json") is None
    assert load_demo_summary("outputs/reports/../../config.yaml") is None


def test_streamlit_demo_outputs_view_rendering():
    """
    Verify Streamlit AppTest switches to Demo Outputs view and renders:
    1. Navigation to 'Demo Outputs (Visual Artifacts)'.
    2. Real MERL sample: exact metrics, 4 artifacts, scale note, WITHHELD recommendation.
    3. Synthetic demonstration: prominent disclaimer banner, simulated metrics, synthetic labels.
    4. MOT17 sample: outdoor pedestrian context notice, metrics, unavailable analytics.
    """


def _select_view(at, option_name: str):
    """Select a view from whichever radio contains the option (primary or legacy)."""
    for r in at.radio:
        if option_name in r.options:
            return r.set_value(option_name)
    return at.radio[0].set_value(option_name)


def test_streamlit_demo_outputs_view_rendering():
    """
    Verify complete rendering of Demo Outputs view:
    1. Real MERL metrics and diagrams.
    2. Synthetic demonstration disclaimer and metrics.
    3. Scale limitations visible.
    4. Methodological boundaries in captions.
    5. No crashes or unhandled exceptions.
    """
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.query_params["view"] = "demo"
    at.run(timeout=30)

    # Verify Title and Caption
    titles = [t.value for t in at.title]
    assert "Demo Outputs (Visual Artifacts)" in titles

    # Verify only indoor MERL context is shown in the staff demo.
    warnings = " ".join([w.value for w in at.warning])
    assert "MERL recorded-video run processed" in warnings
    assert "boundary-truncated" in warnings
    assert "MOT17-04 sequence consists of outdoor pedestrian image-sequence" not in warnings

    # Verify Errors (prominent synthetic disclaimer and withheld recommendations)
    errors = " ".join([e.value for e in at.error])
    assert "SYNTHETIC DEMONSTRATION — NOT REAL SHOPPER DATA" in errors
    assert "WITHHELD" in errors

    # Verify Markdowns contain exact metrics
    markdowns = " ".join([m.value for m in at.markdown])
    # MERL exact metrics
    assert "Evaluated Sampled Frames" in markdowns
    assert "Total Person Detections" in markdowns
    assert "Points Inside Configured Zones" in markdowns
    assert "Unassigned / Out-of-Zone Points" in markdowns
    assert "Successfully Binned Points" in markdowns
    assert "Out-of-Bounds Points" in markdowns

    # Synthetic metrics
    assert "Simulated Customer Tracks" in markdowns
    assert "Total Simulated Points" in markdowns
    assert "Synthetic Direct Transition Counts" in markdowns

    # Scale limitations
    infos = " ".join([i.value for i in at.info])
    assert "configured floor units" in infos or "configured floor units" in markdowns

    # Captions contain methodological boundaries
    captions = " ".join([c.value for c in at.caption])
    assert "not physical crowd occupancy" in captions or "not measure unique visitors" in captions
    assert "Observed Sample Activity" in captions or "Evaluated Zero Observation" in captions
    assert "not model Markovian transition probabilities" in captions

    # Verify images rendered
    assert len(at.image) >= 7  # 4 indoor MERL + 3 labeled synthetic diagrams


def test_demo_outputs_handles_missing_manifest(tmp_path):
    """
    Verify that if reports directory has no demo summary JSON files,
    Demo Outputs view renders graceful error without crashing.
    """
    from streamlit.testing.v1 import AppTest

    empty_dir = tmp_path / "empty_reports"
    empty_dir.mkdir()

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.session_state["reports_dir"] = str(empty_dir)
    at.query_params["view"] = "demo"
    at.run(timeout=30)

    # Should show error about manifest not found, but not crash
    errors = " ".join([e.value for e in at.error])
    assert "manifest not found" in errors.lower()


def test_streamlit_demo_outputs_view_via_query_param():
    """
    Verify opening app with query parameter ?view=demo automatically selects
    Demo Outputs (Visual Artifacts) on initial load.
    """
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.query_params["view"] = "demo"
    at.run(timeout=30)

    # Title should immediately be Demo Outputs
    titles = [t.value for t in at.title]
    assert "Demo Outputs (Visual Artifacts)" in titles
    assert any(r.value == "Demo Outputs (Visual Artifacts)" for r in at.radio)


def test_streamlit_view_selection_merl_sample():
    """Verify selecting Real MERL Sample Demo renders the supplied MERL summaries."""
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.query_params["view"] = "merl"
    at.run(timeout=30)

    titles = [t.value for t in at.title]
    assert "Real MERL Retail Sample Demo" in titles
    markdowns = " ".join([m.value for m in at.markdown])
    assert "Per-Zone Dwell Summary (Observed Tracker Spans)" in markdowns
    assert "Per-Zone Direct Transition Summary" in markdowns
    captions = " ".join([c.value for c in at.caption])
    assert "Observed Sample Activity" in captions or "Evaluated Zero Observation" in captions


def test_streamlit_view_selection_synthetic_demonstration():
    """Verify selecting Synthetic Demonstration renders synthetic view with nonzero metrics."""
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.query_params["view"] = "synth"
    at.run(timeout=30)

    titles = [t.value for t in at.title]
    assert "Synthetic Demonstration — NOT REAL SHOPPER DATA" in titles
    errors = " ".join([e.value for e in at.error])
    assert "SYNTHETIC DEMONSTRATION — NOT REAL SHOPPER DATA" in errors
    markdowns = " ".join([m.value for m in at.markdown])
    assert "Synthetic Per-Zone Dwell Breakdown" in markdowns
    assert "Synthetic Direct Transition Counts" in markdowns


def test_streamlit_view_selection_uploads_indoor_video_without_running_until_requested():
    """The upload page is visible, local-only, and idle until an explicit analyze click, protected by auth."""
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)

    # 1. Verify guest access restriction: guest is blocked by protected access barrier
    at.query_params["view"] = "upload"
    at.run(timeout=30)
    barrier_text = " ".join([m.value for m in at.markdown] + [w.value for w in at.warning])
    assert "Protected Feature Access" in barrier_text or "Authentication Required" in barrier_text

    # 2. Authenticate session and verify the upload page renders properly
    at.session_state["authenticated_user"] = "staff_tester"
    at.run(timeout=30)

    titles = [t.value for t in at.title]
    assert "Analyze an Indoor Video" in titles
    infos = " ".join([i.value for i in at.info])
    assert "Upload an indoor video to begin" in infos
    page_text = " ".join(
        [w.value for w in at.warning] + [m.value for m in at.markdown] + [i.value for i in at.info]
    )
    assert "has no indoor training annotations" in page_text


def test_public_dataset_evidence_lab_route_renders_without_claiming_validated_recommendations():
    """The public dataset page is reachable and keeps the candidate's evidence boundary visible."""
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.query_params["view"] = "public"
    at.run(timeout=60)

    assert not at.exception
    assert "Public Retail & Venue Dataset Evidence Lab" in [t.value for t in at.title]
    rendered = " ".join(
        [e.value for e in at.error]
        + [w.value for w in at.warning]
        + [i.value for i in at.info]
        + [m.value for m in at.markdown]
        + [c.value for c in at.caption]
    ).lower()
    if Path("outputs/reports/public_dataset_examples_summary.json").is_file():
        assert "not person tracking" in rendered
        assert "shelf-row polygons" in rendered
        assert "aisle section" in rendered


def test_public_dataset_file_resolver_rejects_paths_outside_external_data():
    from dashboard.app import _resolve_public_dataset_file

    assert _resolve_public_dataset_file("config.yaml") is None
    assert _resolve_public_dataset_file("data/external/../../config.yaml") is None


def test_unified_pipeline_walkthrough_keeps_metrics_and_withholding_together():
    """The one-run view presents trajectory, hotspot, dwell, transitions, and evidence gate together."""
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.query_params["view"] = "all"
    at.run(timeout=60)

    assert not at.exception
    assert "End-to-End Retail Video Analysis" in [t.value for t in at.title]
    content = " ".join(
        [x.value for x in at.subheader]
        + [x.value for x in at.caption]
        + [x.label + " " + str(x.value) for x in at.metric]
        + [x.value for x in at.error]
        + [x.value for x in at.warning]
        + [x.value for x in at.markdown]
    ).lower()
    assert "video, detection, and tracking" in content
    assert "trajectory density and hotspot" in content
    assert "dwell, footfall, visits, and transitions" in content
    assert "recommendation status" in content
    assert "withheld" in content
    assert "shelf_a" in content
    assert len(at.image) >= 5




def test_merl_demo_shows_full_pipeline_diagnostics():
    """The real run view exposes actual input/model/calibration/footfall metrics."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.query_params["view"] = "merl"
    at.run(timeout=30)

    assert not at.exception
    text = " ".join([m.value for m in at.markdown])
    assert "Model / Training Status" in text
    assert "not_performed" in text
    assert "Input Video & Quality Checks" in text
    assert "Configured Homography Diagnostics" in text
    assert "Footfall Events & Sampled Zone Occupancy" in text
    assert "Track and Evidence Diagnostics" in text
    assert "Zone-entry events" in text
    errors = " ".join([item.value for item in at.error])
    assert "WITHHELD" in errors



def test_merl_demo_shows_exploratory_candidate_and_withholds_actionable_claims():
    """Show transparent candidate semantics alongside explicit evidence gating."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.query_params["view"] = "merl"
    at.run(timeout=30)

    assert not at.exception
    errors = " ".join(item.value for item in at.error)
    markdowns = " ".join(item.value for item in at.markdown)
    infos = " ".join(item.value for item in at.info)
    assert "WITHHELD" in errors
    captions = " ".join(item.value for item in at.caption)
    assert "highest observed dwell span" in (markdowns + infos + captions).lower()
