"""
Unit & Integration Tests for Trajectory Module
==============================================
Tests TrajectoryPoint creation, bottom-center calculation, temporal ordering,
and trajectory extraction from Track objects.
"""

import pytest
from src.tracking.bytetrack import Track, TrackPoint, Detection
from src.trajectory import TrajectoryPoint, Trajectory, TrajectoryExtractor


def test_trajectory_point_creation():
    pt = TrajectoryPoint(
        track_id=1,
        frame_idx=10,
        timestamp_sec=2.0,
        image_x=150.0,
        image_y=300.0,
        floor_x=2.5,
        floor_y=8.0
    )
    assert pt.track_id == 1
    assert pt.frame_idx == 10
    assert pt.timestamp_sec == 2.0
    assert pt.image_x == 150.0
    assert pt.image_y == 300.0
    assert pt.floor_x == 2.5
    assert pt.floor_y == 8.0


def test_trajectory_temporal_ordering():
    traj = Trajectory(track_id=1)
    pt1 = TrajectoryPoint(track_id=1, frame_idx=1, timestamp_sec=0.0, image_x=10.0, image_y=20.0)
    pt2 = TrajectoryPoint(track_id=1, frame_idx=2, timestamp_sec=0.2, image_x=12.0, image_y=22.0)
    
    traj.add_point(pt1)
    traj.add_point(pt2)
    
    assert traj.start_frame == 1
    assert traj.end_frame == 2
    assert traj.duration_sec == 0.2
    assert traj.point_count == 2

    # Out-of-order frame index must raise ValueError
    pt_out_of_order = TrajectoryPoint(track_id=1, frame_idx=1, timestamp_sec=0.1, image_x=11.0, image_y=21.0)
    with pytest.raises(ValueError, match="Temporal ordering violation"):
        traj.add_point(pt_out_of_order)


def test_trajectory_distance_calculations():
    traj = Trajectory(track_id=1)
    p1 = TrajectoryPoint(1, 1, 0.0, 0.0, 0.0, floor_x=0.0, floor_y=0.0)
    p2 = TrajectoryPoint(1, 2, 0.2, 3.0, 4.0, floor_x=3.0, floor_y=4.0)  # 3-4-5 triangle -> dist 5
    
    traj.add_point(p1)
    traj.add_point(p2)
    
    assert traj.calculate_image_distance() == 5.0
    assert traj.calculate_floor_distance() == 5.0


def test_trajectory_extraction_from_track():
    Track.reset_id_counter(0)
    track = Track(bbox=[100.0, 100.0, 200.0, 300.0], confidence=0.90, frame_id=1, timestamp_sec=0.0)
    
    # Update track with 2nd frame
    det = Detection(bbox=[104.0, 102.0, 204.0, 302.0], confidence=0.92)
    track.update(det, frame_id=2, timestamp_sec=0.2)

    extracted_traj = TrajectoryExtractor.extract_from_track(track)
    
    assert extracted_traj.track_id == track.track_id
    assert extracted_traj.point_count == 2
    assert extracted_traj.points[0].image_x == 150.0  # (100+200)/2
    assert extracted_traj.points[0].image_y == 300.0  # y2
    assert extracted_traj.points[1].image_x == 154.0  # (104+204)/2
    assert extracted_traj.points[1].image_y == 302.0  # y2


def test_trajectory_missing_image_coordinates_non_fabricating():
    """Verify that missing image coordinates remain None, do not invent zeros, and do not affect floor distance or dwell."""
    pt1 = TrajectoryPoint(track_id=1, frame_idx=1, timestamp_sec=0.0, image_x=None, image_y=None, floor_x=0.0, floor_y=0.0, zone_id="zone_a")
    pt2 = TrajectoryPoint(track_id=1, frame_idx=2, timestamp_sec=1.0, image_x=None, image_y=None, floor_x=3.0, floor_y=4.0, zone_id="zone_a")
    
    traj = Trajectory(track_id=1)
    traj.add_point(pt1)
    traj.add_point(pt2)
    
    assert pt1.image_x is None
    assert pt1.image_y is None
    assert traj.calculate_image_distance() == 0.0
    assert traj.calculate_floor_distance() == 5.0
    assert len(traj.get_image_points()) == 0
    assert len(traj.get_floor_points()) == 2
    
    # Verify serialization does not convert None to 0.0
    d = pt1.to_dict()
    assert d["image_x"] is None
    assert d["image_y"] is None
    assert d["floor_x"] == 0.0
    assert d["floor_y"] == 0.0
    assert d["zone_id"] == "zone_a"

