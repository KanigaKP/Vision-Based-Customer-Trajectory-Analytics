"""
Unit & Integration Tests for Spatial Homography Module
=====================================================
Tests HomographyCalibrator input validations, insufficient points errors, collinearity handling,
point and trajectory transformations, and preservation of original image coordinates.
"""

import pytest
import numpy as np
import yaml

from src.trajectory import TrajectoryPoint, Trajectory
from src.spatial import HomographyCalibrator, HomographyDiagnostics


@pytest.fixture
def valid_calibration_points():
    """Fixture returning a known 4-point correspondence pair."""
    src = [
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 100.0],
        [0.0, 100.0]
    ]
    dst = [
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0]
    ]
    return src, dst


def test_insufficient_points_raises_error():
    src = [[0, 0], [10, 0], [10, 10]]  # Only 3 points!
    dst = [[0, 0], [1, 0], [1, 1]]
    with pytest.raises(ValueError, match="At least 4 correspondence points are required"):
        HomographyCalibrator(image_points=src, floor_points=dst)


def test_mismatched_points_length_raises_error():
    src = [[0, 0], [10, 0], [10, 10], [0, 10]]
    dst = [[0, 0], [1, 0], [1, 1]]
    with pytest.raises(ValueError, match="At least 4 correspondence points are required"):
        HomographyCalibrator(image_points=src, floor_points=dst)


def test_collinear_points_raises_error():
    # 4 points lying on the exact same line y = 2x
    src = [[0, 0], [1, 2], [2, 4], [3, 6]]
    dst = [[0, 0], [1, 1], [2, 2], [3, 3]]
    with pytest.raises(ValueError, match="collinear or degenerate"):
        HomographyCalibrator(image_points=src, floor_points=dst)


def test_valid_homography_point_transformation(valid_calibration_points):
    src, dst = valid_calibration_points
    calibrator = HomographyCalibrator(image_points=src, floor_points=dst, unit="metres")
    
    # Transform (50, 50) -> should map to (5.0, 5.0)
    fx, fy = calibrator.transform_point(50.0, 50.0)
    assert fx == pytest.approx(5.0, abs=1e-2)
    assert fy == pytest.approx(5.0, abs=1e-2)
    
    # Check diagnostics
    diag = calibrator.diagnostics
    assert isinstance(diag, HomographyDiagnostics)
    assert diag.num_calibration_points == 4
    assert diag.training_reprojection_mean_error < 1e-3


def test_homography_trajectory_transformation(valid_calibration_points):
    src, dst = valid_calibration_points
    calibrator = HomographyCalibrator(image_points=src, floor_points=dst, unit="metres")

    traj = Trajectory(track_id=42)
    p1 = TrajectoryPoint(track_id=42, frame_idx=1, timestamp_sec=0.0, image_x=0.0, image_y=0.0)
    p2 = TrajectoryPoint(track_id=42, frame_idx=2, timestamp_sec=0.2, image_x=100.0, image_y=100.0)
    traj.add_point(p1)
    traj.add_point(p2)

    transformed_traj = calibrator.transform_trajectory(traj)

    # Check track ID and timestamps preserved
    assert transformed_traj.track_id == 42
    assert transformed_traj.points[0].track_id == 42
    assert transformed_traj.points[0].frame_idx == 1
    assert transformed_traj.points[0].timestamp_sec == 0.0

    # Check original image coordinates preserved
    assert transformed_traj.points[0].image_x == 0.0
    assert transformed_traj.points[0].image_y == 0.0
    assert transformed_traj.points[1].image_x == 100.0
    assert transformed_traj.points[1].image_y == 100.0

    # Check transformed floor coordinates
    assert transformed_traj.points[0].floor_x == pytest.approx(0.0, abs=1e-2)
    assert transformed_traj.points[0].floor_y == pytest.approx(0.0, abs=1e-2)
    assert transformed_traj.points[1].floor_x == pytest.approx(10.0, abs=1e-2)
    assert transformed_traj.points[1].floor_y == pytest.approx(10.0, abs=1e-2)


def test_homography_trajectory_transformation_with_missing_image_coords(valid_calibration_points):
    src, dst = valid_calibration_points
    calibrator = HomographyCalibrator(image_points=src, floor_points=dst, unit="metres")

    traj = Trajectory(track_id=99)
    p1 = TrajectoryPoint(track_id=99, frame_idx=1, timestamp_sec=0.0, image_x=None, image_y=None, floor_x=5.5, floor_y=6.5)
    p2 = TrajectoryPoint(track_id=99, frame_idx=2, timestamp_sec=0.2, image_x=100.0, image_y=100.0)
    traj.add_point(p1)
    traj.add_point(p2)

    transformed_traj = calibrator.transform_trajectory(traj)

    assert transformed_traj.points[0].image_x is None
    assert transformed_traj.points[0].image_y is None
    # Pre-existing floor coords must be preserved without distortion
    assert transformed_traj.points[0].floor_x == 5.5
    assert transformed_traj.points[0].floor_y == 6.5
    # Valid image point transformed correctly
    assert transformed_traj.points[1].image_x == 100.0
    assert transformed_traj.points[1].image_y == 100.0
    assert transformed_traj.points[1].floor_x == pytest.approx(10.0, abs=1e-2)
    assert transformed_traj.points[1].floor_y == pytest.approx(10.0, abs=1e-2)



def test_homography_from_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    calibrator_mot = HomographyCalibrator.from_config(config, dataset_name="mot17_04")
    assert calibrator_mot.dataset_name == "mot17_04"
    assert calibrator_mot.unit == "metres"
    assert calibrator_mot.H.shape == (3, 3)

    calibrator_merl = HomographyCalibrator.from_config(config, dataset_name="merl")
    assert calibrator_merl.dataset_name == "merl"
    assert calibrator_merl.H.shape == (3, 3)
