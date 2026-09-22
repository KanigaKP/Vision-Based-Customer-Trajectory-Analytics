"""
Phase 6 — Unit Tests for Spatial Zone Analytics
===============================================
Validates Zone data structures, YAML configuration persistence, Point-in-Polygon (PIP)
testing, boundary hysteresis filtering, and multi-zone evaluation.
"""

import os
import pytest
from src.analytics.zone import Zone, ZoneConfig
from src.analytics.zone_evaluator import ZoneEvaluator
from src.trajectory.trajectory import Trajectory, TrajectoryPoint


def test_zone_creation_and_validation():
    """Test 1: Valid Zone creation and validation of invalid parameters."""
    # Valid zone
    z = Zone(
        zone_id="shelf_a",
        name="Shelf A",
        polygon=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        color="#3498db"
    )
    assert z.zone_id == "shelf_a"
    assert z.name == "Shelf A"
    assert len(z.polygon) == 4

    # Invalid zone_id
    with pytest.raises(ValueError, match="zone_id must be a non-empty string"):
        Zone(zone_id="", name="Invalid", polygon=[(0, 0), (1, 0), (1, 1)])

    # Less than 3 vertices
    with pytest.raises(ValueError, match="must contain at least 3 vertices"):
        Zone(zone_id="bad_poly", name="Bad Poly", polygon=[(0, 0), (1, 1)])

    # Invalid vertex format
    with pytest.raises(ValueError, match="coordinate pair"):
        Zone(zone_id="bad_pt", name="Bad Pt", polygon=[(0, 0, 0), (1, 1, 1), (2, 2, 2)])


def test_point_inside_zone():
    """Test 2: PIP test returns correct zone_id when point is inside polygon."""
    z1 = Zone("shelf_a", "Shelf A", [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)])
    evaluator = ZoneEvaluator(zones=[z1])

    # Point inside (2.5, 2.5)
    zone_id = evaluator.evaluate_point(2.5, 2.5)
    assert zone_id == "shelf_a"


def test_point_outside_zone():
    """Test 3: PIP test returns None when point is outside all defined zones."""
    z1 = Zone("shelf_a", "Shelf A", [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)])
    evaluator = ZoneEvaluator(zones=[z1])

    # Point outside (10.0, 10.0)
    zone_id = evaluator.evaluate_point(10.0, 10.0)
    assert zone_id is None


def test_boundary_hysteresis():
    """Test 4: Single-frame boundary noise does not trigger a false zone transition when N=3."""
    z1 = Zone("zone_a", "Zone A", [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    z2 = Zone("zone_b", "Zone B", [(5.0, 0.0), (9.0, 0.0), (9.0, 4.0), (5.0, 4.0)])
    evaluator = ZoneEvaluator(zones=[z1, z2], hysteresis_window=3)

    trk = Trajectory(track_id=1)
    # 3 points in zone_a, then 1 noisy point in zone_b, then back in zone_a
    coords = [
        (2.0, 2.0), (2.0, 2.0), (2.0, 2.0),
        (7.0, 2.0), # 1 frame boundary noise
        (2.0, 2.0)
    ]
    for idx, (x, y) in enumerate(coords):
        trk.add_point(TrajectoryPoint(
            track_id=1, frame_idx=idx, timestamp_sec=idx * 0.2,
            image_x=100.0, image_y=100.0, floor_x=x, floor_y=y
        ))

    assigned = evaluator.evaluate_trajectory(trk)
    # The 4th point (index 3) should still be assigned zone_a because noise was only 1 frame < N=3
    assert assigned[3] == "zone_a"
    assert assigned[4] == "zone_a"


def test_zone_transition_after_hysteresis():
    """Test 5: Zone transition is confirmed after N=3 consecutive frames in new candidate zone."""
    z1 = Zone("zone_a", "Zone A", [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    z2 = Zone("zone_b", "Zone B", [(5.0, 0.0), (9.0, 0.0), (9.0, 4.0), (5.0, 4.0)])
    evaluator = ZoneEvaluator(zones=[z1, z2], hysteresis_window=3)

    trk = Trajectory(track_id=2)
    # 2 points in zone_a, followed by 3 consecutive points in zone_b
    coords = [
        (2.0, 2.0), (2.0, 2.0),
        (7.0, 2.0), (7.0, 2.0), (7.0, 2.0)
    ]
    for idx, (x, y) in enumerate(coords):
        trk.add_point(TrajectoryPoint(
            track_id=2, frame_idx=idx, timestamp_sec=idx * 0.2,
            image_x=100.0, image_y=100.0, floor_x=x, floor_y=y
        ))

    assigned = evaluator.evaluate_trajectory(trk)
    assert assigned[0] == "zone_a"
    assert assigned[1] == "zone_a"
    # Transition happens on 3rd consecutive frame in zone_b (index 4)
    assert assigned[4] == "zone_b"


def test_multi_zone_evaluation():
    """Test 6: Evaluate trajectory passing through multiple non-overlapping zones."""
    shelf_a = Zone("shelf_a", "Shelf A", [(0.0, 0.0), (3.0, 0.0), (3.0, 5.0), (0.0, 5.0)])
    aisle = Zone("aisle", "Main Aisle", [(3.0, 0.0), (6.0, 0.0), (6.0, 5.0), (3.0, 5.0)])
    shelf_b = Zone("shelf_b", "Shelf B", [(6.0, 0.0), (9.0, 0.0), (9.0, 5.0), (6.0, 5.0)])

    evaluator = ZoneEvaluator(zones=[shelf_a, aisle, shelf_b], hysteresis_window=1)

    trk = Trajectory(track_id=3)
    coords = [(1.5, 2.5), (4.5, 2.5), (7.5, 2.5), (12.0, 12.0)]
    for idx, (x, y) in enumerate(coords):
        trk.add_point(TrajectoryPoint(
            track_id=3, frame_idx=idx, timestamp_sec=idx * 0.2,
            image_x=100.0, image_y=100.0, floor_x=x, floor_y=y
        ))

    assigned = evaluator.evaluate_trajectory(trk)
    assert assigned == ["shelf_a", "aisle", "shelf_b", None]


def test_zone_config_save_load(tmp_path):
    """Test 7: Save and load Zone definitions and hysteresis parameter to/from YAML."""
    z1 = Zone("shelf_a", "Shelf A", [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)], color="#3498db")
    z2 = Zone("checkout", "Checkout", [(3.0, 0.0), (5.0, 0.0), (5.0, 2.0), (3.0, 2.0)], color="#e74c3c")

    out_file = str(tmp_path / "test_zones_config.yaml")
    ZoneConfig.save_to_yaml(
        zones=[z1, z2],
        output_path=out_file,
        dataset_key="test_store",
        hysteresis_window=4
    )

    assert os.path.exists(out_file)

    loaded_zones, hysteresis = ZoneConfig.load_from_config(out_file, dataset_key="test_store")
    assert hysteresis == 4
    assert len(loaded_zones) == 2
    assert loaded_zones[0].zone_id == "shelf_a"
    assert loaded_zones[1].zone_id == "checkout"
    assert loaded_zones[0].color == "#3498db"
