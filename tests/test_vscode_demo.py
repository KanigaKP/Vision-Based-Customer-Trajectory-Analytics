"""
Focused Tests for VS Code Demo Runner & Visual Artifact Generation.
===================================================================
Verifies:
1. `render_floor_trajectory_diagram` renders floor-plane trajectory diagrams
   with zone polygons and explicit unit labels.
2. `render_zone_dwell_barchart` renders dual bar charts with evaluated zeros
   and non-zero metrics without error.
3. `build_synthetic_demonstration` creates all 4 synthetic artifacts with the
   prominent disclaimer `SYNTHETIC DEMONSTRATION — NOT REAL SHOPPER DATA`
   and verifies that actionable recommendations are WITHHELD.
4. Point conservation invariant and direct transition semantics are preserved.
"""

import os
import json
import pytest
import yaml
from typing import Dict, Any

from scripts.run_vscode_demo import (
    render_floor_trajectory_diagram,
    render_zone_dwell_barchart,
    build_synthetic_demonstration,
    select_exploratory_product_zone_candidate,
)
from src.analytics.zone import Zone, ZoneConfig
from src.analytics.dwell import analyze_dwell
from src.trajectory.trajectory import Trajectory, TrajectoryPoint


@pytest.fixture
def repo_config() -> Dict[str, Any]:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_render_floor_trajectory_diagram(tmp_path, repo_config):
    """Verify floor trajectory diagram renders to PNG without exception."""
    zones, _ = ZoneConfig.load_from_config(repo_config, dataset_key="merl")

    trk = Trajectory(track_id=1)
    trk.add_point(TrajectoryPoint(1, 0, 0.0, 10.0, 10.0, floor_x=2.0, floor_y=2.0))
    trk.add_point(TrajectoryPoint(1, 1, 0.2, 20.0, 20.0, floor_x=3.0, floor_y=3.0))

    out_png = str(tmp_path / "test_floor_traj.png")
    result_path = render_floor_trajectory_diagram(
        zones=zones,
        trajectories=[trk],
        output_path=out_png,
        title="Test Trajectory Diagram",
        is_synthetic=True,
    )

    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 1000  # Non-trivial image file


def test_render_zone_dwell_barchart_evaluated_zeros(tmp_path):
    """Verify zone dwell barchart correctly handles evaluated zero metrics."""
    # Create empty dwell summary with 4 zones
    trk = Trajectory(track_id=1)
    # Outside all zones
    trk.add_point(TrajectoryPoint(1, 0, 0.0, 10.0, 10.0, floor_x=15.0, floor_y=15.0))

    known_zones = ["checkout", "main_aisle", "shelf_a", "shelf_b"]
    dwell_summary = analyze_dwell([trk], known_zone_ids=known_zones)
    assert dwell_summary.total_episodes == 0
    assert dwell_summary.total_dwell_sec == 0.0

    out_png = str(tmp_path / "test_dwell_zeros.png")
    result_path = render_zone_dwell_barchart(
        dwell_summary=dwell_summary,
        output_path=out_png,
        title="Evaluated Zeros Test",
        is_synthetic=False,
    )

    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 1000


def test_build_synthetic_demonstration_artifacts(tmp_path, repo_config):
    """Verify build_synthetic_demonstration exports all 4 artifacts with disclaimers."""
    reports_dir = str(tmp_path / "demo_reports")
    summary = build_synthetic_demonstration(repo_config, reports_dir=reports_dir)

    # 1. Check returned dictionary
    assert "SYNTHETIC DEMONSTRATION" in summary["dataset_name"]
    assert "WITHHELD" in summary["actionable_recommendations"]
    assert summary["density_analytics"]["conservation_invariant_verified"] is True
    assert summary["dwell_analytics"]["status"] == "available"
    assert summary["direct_transitions"]["status"] == "available"
    assert summary["direct_transitions"]["total_direct_transitions"] > 0

    # 2. Check exported artifact files on disk
    expected_files = [
        "demo_synthetic_floor_trajectory.png",
        "demo_synthetic_density_heatmap.png",
        "demo_synthetic_zone_dwell.png",
        "demo_synthetic_summary.json",
    ]
    for fname in expected_files:
        fpath = os.path.join(reports_dir, fname)
        assert os.path.exists(fpath), f"Missing artifact: {fname}"
        assert os.path.getsize(fpath) > 100

    # 3. Verify JSON content on disk
    with open(os.path.join(reports_dir, "demo_synthetic_summary.json"), "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert "SYNTHETIC DEMONSTRATION — NOT REAL SHOPPER DATA" in loaded_json["dataset_name"]
    assert "not Markovian probabilities" in loaded_json["direct_transitions"]["note"]
    assert "not physical crowd occupancy" in loaded_json["density_analytics"]["note"]


def test_exploratory_candidate_uses_observed_shelf_metrics_without_thresholds():
    candidate = select_exploratory_product_zone_candidate(
        ["shelf_a", "shelf_b", "main_aisle", "checkout"],
        {"shelf_a": 12, "shelf_b": 30},
        {
            "shelf_a": {"observed_dwell_span_sec": 5.0, "episodes": 2},
            "shelf_b": {"observed_dwell_span_sec": 2.0, "episodes": 1},
        },
    )
    assert candidate["candidate_zone_id"] == "shelf_a"
    assert candidate["status"] == "exploratory_unvalidated_candidate"
    assert candidate["actionable_recommendation"] == "WITHHELD"
    assert "No thresholds" in candidate["selection_rule"]
    assert candidate["recommendation_mode"] == "multi_zone_comparison"


def test_single_zone_candidate_marks_highest_count_grid_cell():
    from scripts.run_vscode_demo import compute_zone_hotspot
    h=compute_zone_hotspot([(1.1,2.1),(1.2,2.2),(1.4,2.4),(2.1,2.1)],(0,10,0,6),.5)
    c=select_exploratory_product_zone_candidate(["shelf_a","shelf_b"],{"shelf_a":3,"shelf_b":0},
        {"shelf_a":{"observed_dwell_span_sec":1.0,"episodes":1}},hotspot_by_zone={"shelf_a":h})
    assert c["recommendation_mode"]=="within_zone_hotspot"
    assert (c["candidate_hotspot"]["grid_row"],c["candidate_hotspot"]["grid_column"])==(4,2)
    assert c["candidate_hotspot"]["sampled_points"]==3
    assert c["actionable_recommendation"]=="WITHHELD"


def test_zone_candidate_can_prioritize_sampled_points():
    c=select_exploratory_product_zone_candidate(["shelf_a","shelf_b"],{"shelf_a":3,"shelf_b":20},
        {"shelf_a":{"observed_dwell_span_sec":8.0,"episodes":1},"shelf_b":{"observed_dwell_span_sec":2.0,"episodes":1}},
        objective="sampled_points")
    assert c["candidate_zone_id"]=="shelf_b"


def test_exploratory_candidate_is_absent_for_all_zero_shelf_metrics():
    candidate = select_exploratory_product_zone_candidate(
        ["shelf_a", "shelf_b"], {"shelf_a": 0, "shelf_b": 0}, {}
    )
    assert candidate["candidate_zone_id"] is None
    assert candidate["status"] == "no_observed_shelf_activity"
