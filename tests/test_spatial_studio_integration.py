"""
Integration tests for the Interactive Spatial Analysis Studio.
==============================================================
Verifies:
1. Authoritative run loading and metrics across curated runs:
   - MERL (upload_ca30f3e8): shelf_a 10.2s, whole-run 17.2s, hotspot X [2.5, 3.0], Y [4.5, 5.0], 10 pts.
   - Site 03 (upload_38d7afe1): side_display_left 1.6s, whole-run 1.8s, hotspot X [0.5, 1.0], Y [3.5, 4.0], 3 pts.
   - Site 02 (upload_site02_jewelry): showcase_display_back 2.8s, whole-run 13.8s, hotspot absent, 3 transitions.
   - Out-of-bounds run (0a24bbf3de): candidate None, 0 dwell, 479 unassigned observations, 100% out of bounds.
2. SVG Floor plan and Isometric extrusion generation:
   - Selected zone highlighting with teal stroke (#0D9488) and badge (★ [SELECTED GEOFENCE]).
   - Candidate zone highlighting with amber stroke (#D97706) and badge (★ [SELECTED CANDIDATE]).
   - Analytical extrusion scale legend present in isometric SVG.
   - Missing/zero dwell zones remain flat on ground plane (0.0 units).
3. Stage navigation and AppTest execution of Spatial Studio view.
"""

import os
import json
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.run_adapter import (
    StandardizedRunData,
    discover_standardized_runs,
    load_standardized_run,
    load_run_observation_timeline,
)
from dashboard.theme import (
    generate_floor_plan_svg,
    generate_isometric_floor_plan_svg,
)


def test_spatial_studio_authoritative_metrics():
    """Verify authoritative metrics across MERL, Site 03, Site 02, and Out-of-bounds runs."""
    runs = discover_standardized_runs("outputs/reports")
    run_map = {r.run_id: r for r in runs}

    # 1. MERL (upload_ca30f3e8)
    merl_id = next((rid for rid in run_map if "upload_ca30f3e8" in rid), None)
    assert merl_id is not None, "MERL upload_ca30f3e8 must be discovered"
    merl = run_map[merl_id]
    assert merl.candidate_zone == "shelf_a"
    assert merl.candidate_dwell_sec == 10.2
    assert merl.total_dwell_sec == 17.2
    assert merl.candidate_hotspot_cell is not None
    assert merl.candidate_hotspot_cell["x_min"] == 2.5
    assert merl.candidate_hotspot_cell["x_max"] == 3.0
    assert merl.candidate_hotspot_cell["y_min"] == 4.5
    assert merl.candidate_hotspot_cell["y_max"] == 5.0
    assert merl.candidate_hotspot_cell.get("sampled_points") == 10

    # 2. Site 03 (upload_38d7afe1)
    site03_id = next((rid for rid in run_map if "upload_38d7afe1" in rid), None)
    assert site03_id is not None, "Site 03 upload_38d7afe1 must be discovered"
    site03 = run_map[site03_id]
    assert site03.candidate_zone == "side_display_left"
    assert site03.candidate_dwell_sec == 1.6
    assert site03.total_dwell_sec == 1.8
    assert site03.candidate_hotspot_cell is not None
    assert site03.candidate_hotspot_cell["x_min"] == 0.5
    assert site03.candidate_hotspot_cell["x_max"] == 1.0
    assert site03.candidate_hotspot_cell["y_min"] == 3.5
    assert site03.candidate_hotspot_cell["y_max"] == 4.0
    assert site03.candidate_hotspot_cell.get("sampled_points") == 3

    # 3. Site 02 (upload_site02_jewelry)
    site02_id = next((rid for rid in run_map if "upload_site02_jewelry" in rid), None)
    assert site02_id is not None, "Site 02 upload_site02_jewelry must be discovered"
    site02 = run_map[site02_id]
    assert site02.candidate_zone == "showcase_display_back"
    assert site02.candidate_dwell_sec == 2.8
    assert site02.total_dwell_sec == 13.8
    assert site02.candidate_hotspot_cell is None
    assert site02.total_transitions == 3

    # 4. Out-of-bounds run (0a24bbf3de)
    oob_id = next((rid for rid in run_map if "0a24bbf3de" in rid), None)
    assert oob_id is not None, "Out-of-bounds run 0a24bbf3de must be discovered"
    oob = run_map[oob_id]
    assert oob.candidate_zone is None
    assert oob.candidate_dwell_sec is None
    assert oob.candidate_hotspot_cell is None
    assert oob.total_dwell_sec == 0.0
    assert oob.in_zone_points == 0
    assert oob.unassigned_points == 479
    assert oob.out_of_bounds_points == 479


def test_floor_plan_svg_selected_zone():
    """Verify generate_floor_plan_svg highlights selected geofences correctly."""
    bounds = (0.0, 10.0, 0.0, 6.0)
    zones = {
        "shelf_a": [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]],
        "shelf_b": [[5.0, 1.0], [7.0, 1.0], [7.0, 3.0], [5.0, 3.0]],
    }
    
    # Candidate zone shelf_a
    svg = generate_floor_plan_svg(
        bounds=bounds,
        zones=zones,
        candidate_zone_id="shelf_a",
        selected_zone_id="shelf_b",
    )
    assert svg is not None
    assert "★ [SELECTED CANDIDATE]" in svg
    assert "★ [SELECTED GEOFENCE]" in svg
    assert "#D97706" in svg  # Candidate amber stroke
    assert "#0D9488" in svg  # Selected teal stroke


def test_isometric_svg_dwell_extrusion_and_scale():
    """Verify generate_isometric_floor_plan_svg renders dwell scale legend and height extrusion."""
    bounds = (0.0, 10.0, 0.0, 6.0)
    zones = {
        "shelf_a": [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]],
        "shelf_b": [[5.0, 1.0], [7.0, 1.0], [7.0, 3.0], [5.0, 3.0]],
    }
    zone_dwell = {
        "shelf_a": {"observed_dwell_span_sec": 10.2},
        "shelf_b": {"observed_dwell_span_sec": 0.0},
    }

    svg = generate_isometric_floor_plan_svg(
        bounds=bounds,
        zones=zones,
        candidate_zone_id="shelf_a",
        selected_zone_id="shelf_b",
        zone_dwell=zone_dwell,
        dwell_scale=0.06,
    )
    assert svg is not None
    assert "DWELL EXTRUSION SCALE" in svg
    assert "1.0s dwell ≈ 0.06 floor units" in svg
    assert "★ CANDIDATE:" in svg
    assert "★ SELECTED:" in svg
    assert "(10.2s)" in svg
    assert "(0.0s)" in svg


def test_apptest_spatial_studio_view():
    """Verify AppTest successfully boots and renders Spatial Studio destination."""
    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.query_params["view"] = "spatial_studio"
    at.query_params["stage"] = "6"
    at.run()
    assert not at.exception, f"AppTest raised exception: {at.exception}"
