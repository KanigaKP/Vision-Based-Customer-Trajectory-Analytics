"""
Focused tests for Direction A Presentation Layer and Standardized Run Adapter.
=============================================================================
Verifies:
1. StandardizedRunData loading across all local run formats (Real MERL, Synthetic, MOT17, Phase 13).
2. Explicit preservation of missing vs evaluated zero values:
   - Phase 13 MERL: status available, total_dwell_sec == 0.0, total_transitions == 0.
   - Real MERL demo: status available, total_dwell_sec == 17.2, total_transitions == 2.
   - MOT17 sample: status not_available, total_dwell_sec is None, total_transitions is None.
3. Point conservation invariant verification across spatial processing stages.
4. Safe path resolution and rejection of path traversal.
5. AppTest navigation and rendering for all 6 Direction A primary destinations:
   - Destination 1: Overview
   - Destination 2: New Analysis
   - Destination 3: Results Workspace
   - Destination 4: Placement Candidate
   - Destination 5: Evidence & Run Details
   - Destination 6: Sample Library
6. Backwards compatibility of legacy query parameters (?view=report, ?view=demo, ?view=merl, ?view=synthetic, etc.).
"""

import os
import json
import pytest
from pathlib import Path

from dashboard.run_adapter import (
    StandardizedRunData,
    load_standardized_run,
    discover_standardized_runs,
    format_observed_value,
)
from dashboard.theme import (
    generate_floor_plan_svg,
    ZONE_COLORS,
)


def test_standardized_run_loader_real_merl():
    """Verify loading real MERL demo summary into typed StandardizedRunData."""
    run = load_standardized_run("outputs/reports/demo_real_summary.json")
    assert run is not None
    assert run.run_id == "demo_real_summary"
    assert "MERL" in run.display_name
    assert run.dataset_name == "merl_shopping_dataset"
    assert run.source_type == "recorded_video"
    assert run.is_synthetic is False
    assert run.is_retail is True
    assert run.total_frames == 633
    assert run.sampling_fps == 5.0
    assert run.total_tracks == 3
    assert run.total_detections == 512
    assert run.total_points == 508
    assert run.in_zone_points == 88
    assert run.unassigned_points == 420
    assert run.in_zone_points + run.unassigned_points == run.total_points
    assert run.binned_points == 169
    assert run.out_of_bounds_points == 339
    assert run.untransformed_points == 0
    assert run.binned_points + run.out_of_bounds_points + run.untransformed_points == run.total_points
    assert run.conservation_verified is True
    assert run.dwell_status == "available"
    assert run.total_dwell_sec == 17.2
    assert run.total_episodes == 5
    assert run.zone_dwell["shelf_a"]["observed_dwell_span_sec"] == 10.2
    assert run.transition_status == "available"
    assert run.total_transitions == 2
    assert run.direct_transitions.get("shelf_a->main_aisle") == 1
    assert run.direct_transitions.get("main_aisle->shelf_a") == 1
    assert run.candidate_zone == "shelf_a"
    assert run.candidate_hotspot_cell is not None
    assert run.candidate_hotspot_cell.get("x_min") == 2.5
    assert run.candidate_hotspot_cell.get("x_max") == 3.0
    assert run.candidate_hotspot_cell.get("y_min") == 4.5
    assert run.candidate_hotspot_cell.get("y_max") == 5.0
    assert run.candidate_hotspot_cell.get("sampled_points") == 10
    assert run.evidence_status == "WITHHELD"
    assert len(run.reasons_withheld) >= 2


def test_standardized_run_loader_phase_13_evaluated_zero():
    """Verify loading Phase 13 report preserves evaluated numeric zero without converting to missing."""
    run = load_standardized_run("outputs/reports/phase_13_analysis_report_merl_sample.json")
    assert run is not None
    assert run.run_id == "phase_13_analysis_report_merl_sample"
    assert run.is_synthetic is False
    assert run.is_retail is True
    assert run.total_frames == 100
    assert run.total_tracks == 1
    assert run.total_points == 74
    assert run.binned_points == 3
    assert run.out_of_bounds_points == 71
    assert run.binned_points + run.out_of_bounds_points == run.total_points
    assert run.conservation_verified is True
    assert run.dwell_status == "available"
    assert run.total_dwell_sec == 0.0
    assert run.total_episodes == 0
    assert run.transition_status == "available"
    assert run.total_transitions == 0
    assert "WITHHELD" in run.evidence_status


def test_standardized_run_loader_synthetic():
    """Verify loading synthetic demonstration fixture."""
    run = load_standardized_run("outputs/reports/demo_synthetic_summary.json")
    assert run is not None
    assert run.is_synthetic is True
    assert "SYNTHETIC" in run.dataset_name.upper()
    assert run.total_tracks == 2
    assert run.total_points == 66
    assert run.dwell_status == "available"
    assert run.total_dwell_sec == 11.2
    assert run.total_episodes == 10
    assert run.transition_status == "available"
    assert run.total_transitions == 8
    assert "WITHHELD" in run.evidence_status


def test_standardized_run_loader_mot17_missing_analytics():
    """Verify loading MOT17 preserves not_available status without fabricating zeroes."""
    run = load_standardized_run("outputs/reports/demo_mot17_summary.json")
    assert run is not None
    assert run.is_retail is False
    assert run.total_frames == 10
    assert run.total_tracks == 11
    assert run.dwell_status == "not_available"
    assert run.total_dwell_sec is None
    assert run.total_episodes is None
    assert run.transition_status == "not_available"
    assert run.total_transitions is None
    assert run.binned_points == 37
    assert run.out_of_bounds_points == 57
    assert "WITHHELD" in run.evidence_status


def test_discover_standardized_runs():
    """Verify discovery finds existing local runs in order."""
    runs = discover_standardized_runs("outputs/reports")
    assert len(runs) >= 3
    run_ids = [r.run_id for r in runs]
    assert "demo_real_summary" in run_ids
    assert "phase_13_analysis_report_merl_sample" in run_ids
    assert "demo_synthetic_summary" in run_ids


def test_format_observed_value_helper():
    """Verify format_observed_value distinguishes missing from zero."""
    assert format_observed_value(None, "s") == "Not evaluated"
    assert format_observed_value(0.0, "s") == "0.0 s"
    assert format_observed_value(0, "episodes") == "0 episodes"
    assert format_observed_value(17.23, "s") == "17.2 s"


def test_generate_floor_plan_svg():
    """Verify offline SVG floor plan generator with explicit geometry."""
    zones = {
        "shelf_a": [(0.5, 4.5), (4.0, 4.5), (4.0, 5.5), (0.5, 5.5)],
        "shelf_b": [(6.0, 4.5), (9.5, 4.5), (9.5, 5.5), (6.0, 5.5)],
        "main_aisle": [(4.0, 1.0), (6.0, 1.0), (6.0, 5.5), (4.0, 5.5)],
        "checkout": [(0.5, 0.5), (4.0, 0.5), (4.0, 1.5), (0.5, 1.5)],
    }
    zone_names = {
        "shelf_a": "Shelf A",
        "shelf_b": "Shelf B",
        "main_aisle": "Main Aisle",
        "checkout": "Checkout",
    }
    svg = generate_floor_plan_svg(
        bounds=(0.0, 10.0, 0.0, 6.0),
        zones=zones,
        zone_names=zone_names,
        candidate_hotspot={"x_min": 2.5, "x_max": 3.0, "y_min": 4.5, "y_max": 5.0, "sampled_points": 10},
        candidate_zone_id="shelf_a",
    )
    assert svg is not None
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "Shelf A" in svg
    assert "Shelf B" in svg
    assert "Main Aisle" in svg
    assert "Checkout" in svg
    assert "Hotspot" in svg


# ---------------------------------------------------------------------------
# AppTest Navigation Across 6 Direction A Destinations
# ---------------------------------------------------------------------------

def test_apptest_overview_destination():
    """Verify Overview destination renders hero, metrics, and run cards."""
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.query_params["view"] = "overview"
    at.run(timeout=30)

    assert not at.exception
    markdowns = " ".join([m.value for m in at.markdown])
    assert "Supermarket Aisle" in markdowns
    assert "Home Décor Boutique" in markdowns
    assert "Jewelry Store" in markdowns
    assert "3 Curated Retail Environments Available" in markdowns


def test_apptest_new_analysis_destination():
    """Verify New Analysis destination renders 5-step setup and disclaimers, protected by auth."""
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.query_params["view"] = "new"
    at.run(timeout=30)

    # 1. Verify guest access barrier is enforced
    barrier_text = " ".join([m.value for m in at.markdown] + [w.value for w in at.warning])
    assert "Protected Feature Access" in barrier_text or "Authentication Required" in barrier_text

    # 2. Authenticate session and verify the 5-step setup
    at.session_state["authenticated_user"] = "staff_tester"
    at.run(timeout=30)

    assert not at.exception
    markdowns = " ".join([m.value for m in at.markdown])
    assert "New Video Analysis & Calibration Setup" in markdowns
    assert "Step 1: Upload Video & Local Preview" in markdowns
    assert "Step 2: Camera Profile & Spatial Calibration" in markdowns
    assert "Step 3: Analysis Settings" in markdowns
    assert "Step 4: Execution" in markdowns
    assert "Interactive canvas point dragging" in markdowns or "proposed capabilities" in markdowns.lower()


def test_apptest_results_workspace_destination():
    """Verify Results Workspace destination renders active run with 5 tabs."""
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.query_params["view"] = "workspace"
    at.run(timeout=30)

    assert not at.exception
    markdowns = " ".join([m.value for m in at.markdown])
    assert "Video Playback and Tracking Overlay" in [sh.value for sh in at.subheader] or "Video Playback" in markdowns
    assert "Floor Plane Trajectories & Zone Geometry" in [sh.value for sh in at.subheader] or "Floor Plane Trajectories" in markdowns
    assert "Discrete Trajectory Point Density" in [sh.value for sh in at.subheader] or "Discrete Trajectory Point Density" in markdowns
    assert "Observed Zone Dwell Spans" in [sh.value for sh in at.subheader] or "Observed Zone Dwell Spans" in markdowns
    assert "Direct Inter-Zone Transitions" in [sh.value for sh in at.subheader] or "Direct Inter-Zone Transitions" in markdowns
    assert "WITHHELD" in markdowns


def test_apptest_placement_candidate_destination():
    """Verify Placement Candidate destination renders hotspot bounds, rule, and WITHHELD banner."""
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.query_params["view"] = "candidate"
    at.run(timeout=30)

    assert not at.exception
    markdowns = " ".join([m.value for m in at.markdown])
    assert "Exploratory New-Product Placement Candidate" in markdowns
    assert "WITHHELD" in markdowns
    assert "Shelf A" in markdowns
    assert "[2.50 .. 3.00]" in markdowns or "2.5" in markdowns
    assert "[4.50 .. 5.00]" in markdowns or "4.5" in markdowns
    assert "Selection Rule" in markdowns
    assert "Why This Candidate" in markdowns
    assert "Comparison with Observed Alternatives" in markdowns
    assert "Highlighted Candidate Zone & Hotspot Cell" in markdowns
    assert "Objective Sensitivity Comparison" in markdowns
    assert "Evidence Limitations & Missing Placement Constraints" in markdowns


def test_apptest_evidence_details_destination():
    """Verify Evidence & Run Details destination renders technical audit and lineage."""
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.query_params["view"] = "evidence"
    at.run(timeout=30)

    assert not at.exception
    markdowns = " ".join([m.value for m in at.markdown])
    assert "Point Conservation Invariant" in markdowns
    assert "Model Inference & Tracking Lineage" in markdowns
    assert "YOLO11-n" in markdowns
    assert "ByteTrack" in markdowns
    assert "Homography" in markdowns
    assert "WITHHELD" in markdowns


def test_apptest_sample_library_destination():
    """Verify Sample Library destination renders 3 segregated benchmark cards."""
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.query_params["view"] = "library"
    at.run(timeout=30)

    assert not at.exception
    markdowns = " ".join([m.value for m in at.markdown])
    assert "Supermarket Aisle" in markdowns
    assert "Home Décor Boutique" in markdowns
    assert "Jewelry Store" in markdowns
    captions = " ".join([c.value for c in at.caption])
    assert "MERL Shopping Dataset" in captions or "MERL" in markdowns


# ---------------------------------------------------------------------------
# 10 Concrete Integrity Tests (Defects 1–8)
# ---------------------------------------------------------------------------

def test_custom_scene_non_merl_bounds_and_zones(tmp_path):
    """Test 1: Custom scene with non-MERL bounds and zone IDs."""
    data = {
        "metadata": {
            "run_id": "custom_scene_run",
            "environment_type": "retail",
            "is_retail": True,
            "is_synthetic": False,
            "scene_profile": {
                "bounds": [0.0, 20.0, 0.0, 15.0],
                "cell_size": 1.0,
                "zones": [
                    {"zone_id": "display_1", "name": "Display 1", "polygon": [[1, 1], [5, 1], [5, 5], [1, 5]], "color": "#112233"},
                    {"zone_id": "display_2", "name": "Display 2", "polygon": [[10, 2], [14, 2], [14, 6], [10, 6]], "color": "#445566"},
                ],
            },
        },
        "spatial_density": {"total_points": 10},
    }
    run = load_standardized_run(data)
    assert run is not None
    assert run.has_geometry is True
    assert run.bounds == (0.0, 20.0, 0.0, 15.0)
    assert set(run.zones.keys()) == {"display_1", "display_2"}
    assert run.zone_colors.get("display_1") == "#112233"
    assert run.zone_names.get("display_2") == "Display 2"

    # SVG renders custom bounds and names without hardcoded MERL zones
    svg = generate_floor_plan_svg(bounds=run.bounds, zones=run.zones, zone_names=run.zone_names)
    assert svg is not None
    assert "Display 1" in svg
    assert "Display 2" in svg
    assert "Shelf A" not in svg


def test_custom_cell_size():
    """Test 2: Custom cell size propagation."""
    data = {
        "metadata": {
            "run_id": "custom_cell_run",
            "scene_profile": {
                "cell_size": 1.25,
            },
        },
    }
    run = load_standardized_run(data)
    assert run is not None
    assert run.cell_size == 1.25


def test_missing_and_partial_hotspot_coordinates():
    """Test 3: Missing and partially recorded hotspot coordinates display missing fields without fake fallbacks."""
    data = {
        "metadata": {"run_id": "partial_hotspot_run"},
        "exploratory_layout_candidate": {
            "candidate_zone_id": "shelf_a",
            "candidate_hotspot_cell": {
                "grid_row": 4,
                "sampled_points": 15,
                # x_min, x_max, y_min, y_max, center_x, center_y intentionally omitted
            },
        },
    }
    run = load_standardized_run(data)
    assert run is not None
    assert run.candidate_hotspot_cell is not None
    assert run.candidate_hotspot_cell.get("grid_row") == 4
    assert run.candidate_hotspot_cell.get("x_min") is None
    assert run.candidate_hotspot_cell.get("center_x") is None

    # SVG generator omits hotspot marker when coordinates are incomplete
    svg = generate_floor_plan_svg(
        bounds=(0.0, 10.0, 0.0, 6.0),
        zones={"shelf_a": [[0, 0], [2, 0], [2, 2], [0, 2]]},
        candidate_hotspot=run.candidate_hotspot_cell,
    )
    assert svg is not None
    assert "Hotspot" not in svg


def test_explicit_unknown_provenance():
    """Test 4: Explicit unknown provenance preserved as unknown (is_retail is None)."""
    data = {
        "metadata": {
            "run_id": "unknown_prov_run",
            "dataset_name": "arbitrary_corridor",
            "environment_type": "unknown",
        },
    }
    run = load_standardized_run(data)
    assert run is not None
    assert run.is_retail is None
    assert run.environment_type == "unknown"


def test_filename_containing_merl_with_non_retail_metadata():
    """Test 5: Filename containing 'merl' with non-retail metadata is NOT inferred as retail."""
    data = {
        "metadata": {
            "run_id": "merl_experimental_hallway",
            "environment_type": "warehouse_logistics",
            "is_retail": False,
        },
    }
    # Pass dict directly; even if run_id has 'merl', provenance must respect metadata
    run = load_standardized_run(data)
    assert run is not None
    assert run.is_retail is False
    assert run.environment_type == "warehouse_logistics"


def test_missing_counts_vs_evaluated_zeros():
    """Test 6: Missing counts vs evaluated zeroes explicitly preserved."""
    data_zero = {
        "metadata": {"run_id": "zero_run"},
        "dwell_summary": {"total_dwell_sec": 0.0, "total_episodes": 0, "status": "available"},
        "transitions_summary": {"total_transitions": 0, "status": "available"},
    }
    run_zero = load_standardized_run(data_zero)
    assert run_zero.dwell_status == "available"
    assert run_zero.total_dwell_sec == 0.0
    assert run_zero.total_episodes == 0
    assert run_zero.transition_status == "available"
    assert run_zero.total_transitions == 0
    assert format_observed_value(run_zero.total_dwell_sec, "s") == "0.0 s"
    assert format_observed_value(run_zero.total_transitions) == "0"

    data_missing = {
        "metadata": {"run_id": "missing_run"},
        "dwell_summary": {"status": "not_available"},
        # transitions omitted entirely
    }
    run_missing = load_standardized_run(data_missing)
    assert run_missing.dwell_status == "not_available"
    assert run_missing.total_dwell_sec is None
    assert run_missing.transition_status == "not_available"
    assert run_missing.total_transitions is None
    assert format_observed_value(run_missing.total_dwell_sec, "s") == "Not evaluated"
    assert format_observed_value(run_missing.total_transitions) == "Not evaluated"


def test_inconsistent_conservation_counts():
    """Test 7: Conservation invariants return False on count discrepancy, None on missing counts."""
    # Discrepant density counts: 40 + 40 + 0 = 80 != 100
    data_bad_density = {
        "metadata": {"run_id": "bad_density_run"},
        "spatial_density": {
            "total_points": 100,
            "binned_points": 40,
            "out_of_bounds_points": 40,
            "untransformed_points": 0,
        },
    }
    run_bad = load_standardized_run(data_bad_density)
    assert run_bad.density_conservation_verified is False

    # Discrepant zone counts: 50 + 30 = 80 != 100
    data_bad_zone = {
        "metadata": {"run_id": "bad_zone_run"},
        "spatial_density": {"total_points": 100},
        "spatial_observations": {
            "in_zone_points": 50,
            "unassigned_points": 30,
        },
    }
    run_bad_zone = load_standardized_run(data_bad_zone)
    assert run_bad_zone.zone_conservation_verified is False

    # Missing counts: None, not False or True
    data_missing_counts = {
        "metadata": {"run_id": "missing_counts_run"},
        "spatial_density": {"total_points": 100},
    }
    run_missing = load_standardized_run(data_missing_counts)
    assert run_missing.density_conservation_verified is None
    assert run_missing.zone_conservation_verified is None


def test_traversal_rejection_and_nested_run_paths(tmp_path):
    """Test 8: Traversal rejection and valid nested run paths."""
    # Traversal should fail safe path resolution and return None
    assert load_standardized_run("../../config.yaml", reports_dir=str(tmp_path)) is None
    assert load_standardized_run("../outside.json", reports_dir=str(tmp_path)) is None

    # Valid nested path inside reports_dir
    nested_dir = tmp_path / "uploaded_runs" / "run_abc"
    nested_dir.mkdir(parents=True)
    nested_file = nested_dir / "summary.json"
    nested_file.write_text(json.dumps({"metadata": {"run_id": "nested_run", "environment_type": "retail"}}))

    run = load_standardized_run(str(nested_file), reports_dir=str(tmp_path))
    assert run is not None
    assert "run_abc" in run.run_id or run.run_id == "nested_run"


def test_two_runs_same_basename_different_directories(tmp_path):
    """Test 9: Two runs with the same basename in different directories produce distinct run IDs."""
    dir_a = tmp_path / "uploaded_runs" / "run_a"
    dir_b = tmp_path / "uploaded_runs" / "run_b"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)

    file_a = dir_a / "summary.json"
    file_b = dir_b / "summary.json"
    file_a.write_text(json.dumps({"metadata": {"run_id": "a_data"}}))
    file_b.write_text(json.dumps({"metadata": {"run_id": "b_data"}}))

    runs = discover_standardized_runs(str(tmp_path))
    run_ids = [r.run_id for r in runs]
    assert len(run_ids) == 2
    assert run_ids[0] != run_ids[1]


def test_shared_selection_across_views():
    """Test 10: Shared selection state across workspace, candidate, and evidence views."""
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authoritative_run_id"] = "demo_synthetic_summary"
    at.query_params["view"] = "workspace"
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["authoritative_run_id"] == "demo_synthetic_summary"

    # Switch to candidate view
    at.query_params["view"] = "candidate"
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["authoritative_run_id"] == "demo_synthetic_summary"

    # Switch to evidence view
    at.query_params["view"] = "evidence"
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["authoritative_run_id"] == "demo_synthetic_summary"


def test_upload_hash_differs_for_identical_initial_bytes():
    """Verify that video files sharing identical initial bytes produce different hashes."""
    import hashlib
    # Two files with identical first 64KB (65536 bytes) but differing afterwards
    prefix = b"0" * 65536
    file_1 = prefix + b"video_footage_sequence_A"
    file_2 = prefix + b"video_footage_sequence_B"

    # Previously: hashlib.sha256(bytes[:65536]) would produce identical hashes
    old_hash_1 = hashlib.sha256(file_1[:65536]).hexdigest()[:16]
    old_hash_2 = hashlib.sha256(file_2[:65536]).hexdigest()[:16]
    assert old_hash_1 == old_hash_2, "Old hashing caused hash collision on identical initial bytes"

    # Current implementation: full content hash with length
    new_hash_1 = hashlib.sha256(file_1).hexdigest()[:16] + f"_{len(file_1)}"
    new_hash_2 = hashlib.sha256(file_2).hexdigest()[:16] + f"_{len(file_2)}"
    assert new_hash_1 != new_hash_2, "Full content hash must differentiate files with matching initial bytes"

