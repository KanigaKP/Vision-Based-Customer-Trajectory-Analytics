"""
Phase 8.2 — Unit Tests for Independent Homography Validation
============================================================
Validates that:
1. Homography is estimated strictly from role='training' calibration points.
2. Calibration points with role='held_out' are evaluated independently and NEVER
   passed to the estimator.
3. Training reprojection error is kept distinct from out-of-sample held-out error.
4. Absence of held-out points returns an explicit INSUFFICIENT_EVIDENCE status.
5. Incomplete training sets (< 4 points) raise clear validation errors.
6. JSON serialization round-trips accurately.

All tests utilize synthetic geometric fixtures for software behavior verification.
They do NOT represent physical real-world accuracy claims.
"""

import math
import pytest
import numpy as np

from src.validation.ground_truth import (
    GroundTruthCalibrationPoint,
    GroundTruthSequence,
)
from src.validation.homography_validation import (
    HeldOutPointValidation,
    HomographyValidationResult,
    HomographyValidator,
    validate_homography,
)


@pytest.fixture
def synthetic_calibration_points():
    """
    Synthetic calibration fixture with a known scaling mapping:
    (image_x, image_y) -> (0.1 * image_x, 0.1 * image_y).
    Contains:
    - 4 training points (P1, P2, P3, P4)
    - 2 held-out points with exact reference floor points (CP1, CP2)
    - 1 held-out point with a deliberate 0.5m displacement (CP3)
    """
    return [
        GroundTruthCalibrationPoint("P1", 0.0, 0.0, 0.0, 0.0, role="training"),
        GroundTruthCalibrationPoint("P2", 100.0, 0.0, 10.0, 0.0, role="training"),
        GroundTruthCalibrationPoint("P3", 100.0, 100.0, 10.0, 10.0, role="training"),
        GroundTruthCalibrationPoint("P4", 0.0, 100.0, 0.0, 10.0, role="training"),
        # Held-out points (should map exactly to 0.1 * img)
        GroundTruthCalibrationPoint("CP1", 50.0, 50.0, 5.0, 5.0, role="held_out"),
        GroundTruthCalibrationPoint("CP2", 20.0, 80.0, 2.0, 8.0, role="held_out"),
        # Held-out point with reference offset of 0.5 in floor_y: (3.0, 3.5) vs true (3.0, 3.0)
        GroundTruthCalibrationPoint("CP3", 30.0, 30.0, 3.0, 3.5, role="held_out"),
    ]


def test_valid_training_and_held_out_evaluation(synthetic_calibration_points):
    """Test 1: Valid homography estimation from training points and evaluation of held-out points."""
    result = HomographyValidator.validate_calibration_points(
        synthetic_calibration_points,
        unit="metres",
        dataset_name="synthetic_grid",
    )

    # 1. Point counts
    assert result.training_point_count == 4
    assert result.held_out_point_count == 3
    assert result.evidence_status == "HELD_OUT_EVALUATED"
    assert len(result.held_out_results) == 3

    # 2. Separation of training reprojection diagnostics and held-out error
    assert result.training_reprojection_mean_error >= 0.0
    assert result.held_out_mean_error is not None
    assert result.held_out_max_error is not None

    # 3. Check individual held-out predictions
    res_cp1 = next(r for r in result.held_out_results if r.label == "CP1")
    assert res_cp1.predicted_floor_point == (5.0, 5.0)
    assert res_cp1.reference_floor_point == (5.0, 5.0)
    assert pytest.approx(res_cp1.error, abs=1e-3) == 0.0

    res_cp2 = next(r for r in result.held_out_results if r.label == "CP2")
    assert res_cp2.predicted_floor_point == (2.0, 8.0)
    assert res_cp2.reference_floor_point == (2.0, 8.0)
    assert pytest.approx(res_cp2.error, abs=1e-3) == 0.0

    # CP3 has 0.5m reference deviation in floor_y (predicted 3.0, 3.0 vs reference 3.0, 3.5)
    res_cp3 = next(r for r in result.held_out_results if r.label == "CP3")
    assert res_cp3.predicted_floor_point == (3.0, 3.0)
    assert res_cp3.reference_floor_point == (3.0, 3.5)
    assert pytest.approx(res_cp3.error, abs=1e-3) == 0.5

    # Aggregate held-out errors: [0.0, 0.0, 0.5] -> mean ~0.1667, max = 0.5
    assert pytest.approx(result.held_out_max_error, abs=1e-3) == 0.5
    assert pytest.approx(result.held_out_mean_error, abs=1e-3) == 0.1667


def test_held_out_independence_does_not_alter_fitted_matrix(synthetic_calibration_points):
    """
    Test 2: Modifying held-out reference coordinates changes validation error
    but does NOT alter the estimated homography matrix.
    """
    # Run 1: original held-out points
    res1 = HomographyValidator.validate_calibration_points(synthetic_calibration_points)

    # Run 2: same training points, but drastically modified held-out reference coordinates
    modified_points = [
        GroundTruthCalibrationPoint("P1", 0.0, 0.0, 0.0, 0.0, role="training"),
        GroundTruthCalibrationPoint("P2", 100.0, 0.0, 10.0, 0.0, role="training"),
        GroundTruthCalibrationPoint("P3", 100.0, 100.0, 10.0, 10.0, role="training"),
        GroundTruthCalibrationPoint("P4", 0.0, 100.0, 0.0, 10.0, role="training"),
        # Substantially displaced reference coordinates
        GroundTruthCalibrationPoint("CP1", 50.0, 50.0, 99.0, 99.0, role="held_out"),
        GroundTruthCalibrationPoint("CP2", 20.0, 80.0, 88.0, 88.0, role="held_out"),
    ]
    res2 = HomographyValidator.validate_calibration_points(modified_points)

    # Homography matrix MUST be strictly identical
    assert res1.homography_matrix is not None
    assert res2.homography_matrix is not None
    np.testing.assert_allclose(res1.homography_matrix, res2.homography_matrix, atol=1e-6)

    # But held-out error must reflect the coordinate change
    assert res1.held_out_mean_error != res2.held_out_mean_error
    assert res2.held_out_mean_error > 50.0


def test_insufficient_evidence_when_no_held_out_points():
    """Test 3: Training-only calibration input returns explicit INSUFFICIENT_EVIDENCE status."""
    training_only = [
        GroundTruthCalibrationPoint("P1", 0.0, 0.0, 0.0, 0.0, role="training"),
        GroundTruthCalibrationPoint("P2", 100.0, 0.0, 10.0, 0.0, role="training"),
        GroundTruthCalibrationPoint("P3", 100.0, 100.0, 10.0, 10.0, role="training"),
        GroundTruthCalibrationPoint("P4", 0.0, 100.0, 0.0, 10.0, role="training"),
    ]
    result = HomographyValidator.validate_calibration_points(training_only)

    assert result.training_point_count == 4
    assert result.held_out_point_count == 0
    assert result.evidence_status == "INSUFFICIENT_EVIDENCE"
    assert "No held-out calibration points available" in result.evidence_note
    assert result.held_out_mean_error is None
    assert result.held_out_max_error is None
    assert len(result.held_out_results) == 0
    # Training diagnostics remain populated
    assert result.training_reprojection_mean_error >= 0.0
    assert result.homography_matrix is not None


def test_fewer_than_four_training_points_raises_error():
    """Test 4: Less than 4 training points raises a clear ValueError."""
    few_training = [
        GroundTruthCalibrationPoint("P1", 0.0, 0.0, 0.0, 0.0, role="training"),
        GroundTruthCalibrationPoint("P2", 100.0, 0.0, 10.0, 0.0, role="training"),
        GroundTruthCalibrationPoint("P3", 100.0, 100.0, 10.0, 10.0, role="training"),
        # Only 3 training points!
        GroundTruthCalibrationPoint("CP1", 50.0, 50.0, 5.0, 5.0, role="held_out"),
    ]
    with pytest.raises(ValueError, match="At least 4 valid training calibration points are required"):
        HomographyValidator.validate_calibration_points(few_training)


def test_collinear_training_points_raises_error():
    """Test 5: Collinear training points produce a degenerate geometry ValueError."""
    collinear_training = [
        GroundTruthCalibrationPoint("P1", 0.0, 0.0, 0.0, 0.0, role="training"),
        GroundTruthCalibrationPoint("P2", 10.0, 20.0, 1.0, 2.0, role="training"),
        GroundTruthCalibrationPoint("P3", 20.0, 40.0, 2.0, 4.0, role="training"),
        GroundTruthCalibrationPoint("P4", 30.0, 60.0, 3.0, 6.0, role="training"),
        GroundTruthCalibrationPoint("CP1", 50.0, 50.0, 5.0, 5.0, role="held_out"),
    ]
    with pytest.raises(ValueError, match="collinear or degenerate"):
        HomographyValidator.validate_calibration_points(collinear_training)


def test_preservation_of_labels_and_metadata(synthetic_calibration_points):
    """Test 6: Point labels, units, and dataset names are preserved in validation results."""
    result = HomographyValidator.validate_calibration_points(
        synthetic_calibration_points,
        unit="grid_units",
        dataset_name="custom_scenario",
    )
    assert result.unit == "grid_units"
    assert result.dataset_name == "custom_scenario"

    labels = [r.label for r in result.held_out_results]
    assert labels == ["CP1", "CP2", "CP3"]
    for r in result.held_out_results:
        assert r.unit == "grid_units"


def test_serialization_round_trip(synthetic_calibration_points):
    """Test 7: HomographyValidationResult converts cleanly to/from dictionary."""
    result = HomographyValidator.validate_calibration_points(synthetic_calibration_points)
    data = result.to_dict()

    assert isinstance(data, dict)
    assert data["training_point_count"] == 4
    assert data["held_out_point_count"] == 3
    assert len(data["held_out_results"]) == 3

    reconstructed = HomographyValidationResult.from_dict(data)
    assert reconstructed.training_point_count == result.training_point_count
    assert reconstructed.held_out_point_count == result.held_out_point_count
    assert reconstructed.held_out_mean_error == result.held_out_mean_error
    assert len(reconstructed.held_out_results) == len(result.held_out_results)
    assert reconstructed.held_out_results[0].label == "CP1"


def test_validate_sequence_adapter():
    """Test 8: Validation works seamlessly from a GroundTruthSequence object."""
    calib_pts = [
        GroundTruthCalibrationPoint("P1", 0.0, 0.0, 0.0, 0.0, role="training"),
        GroundTruthCalibrationPoint("P2", 100.0, 0.0, 10.0, 0.0, role="training"),
        GroundTruthCalibrationPoint("P3", 100.0, 100.0, 10.0, 10.0, role="training"),
        GroundTruthCalibrationPoint("P4", 0.0, 100.0, 0.0, 10.0, role="training"),
        GroundTruthCalibrationPoint("CP1", 50.0, 50.0, 5.0, 5.0, role="held_out"),
    ]
    seq = GroundTruthSequence(
        schema_version="1.0",
        sequence_id="SEQ_TEST_01",
        source="synthetic",
        fps=5.0,
        calibration_points=calib_pts,
        metadata={"unit": "metres"},
    )

    # Validate via validator class method
    res1 = HomographyValidator.validate_sequence(seq)
    assert res1.dataset_name == "SEQ_TEST_01"
    assert res1.held_out_point_count == 1
    assert res1.evidence_status == "HELD_OUT_EVALUATED"

    # Validate via convenience function
    res2 = validate_homography(seq)
    assert res2.dataset_name == "SEQ_TEST_01"
    assert res2.held_out_point_count == 1


def test_unrounded_transformation_path_precision():
    """
    Test 9: Verifies that held-out error is computed from unrounded transformed coordinates.
    Under 2-decimal rounding (the old transform_point path), an image point at (123.4, 456.7)
    mapping to true floor (1.234, 4.567) would be rounded to (1.23, 4.57), producing an artificial
    rounding error of 0.005m (5mm). Under the unrounded vectorized transform_points path, the
    predicted coordinates preserve sub-centimeter decimals and the error is 0.0.
    """
    train_pts = [
        GroundTruthCalibrationPoint("P1", 0.0, 0.0, 0.0, 0.0, role="training"),
        GroundTruthCalibrationPoint("P2", 1000.0, 0.0, 10.0, 0.0, role="training"),
        GroundTruthCalibrationPoint("P3", 1000.0, 1000.0, 10.0, 10.0, role="training"),
        GroundTruthCalibrationPoint("P4", 0.0, 1000.0, 0.0, 10.0, role="training"),
    ]
    # Point with fractional coordinates requiring 3 decimal places
    held_out_pt = GroundTruthCalibrationPoint(
        "CP_frac",
        image_x=123.4,
        image_y=456.7,
        floor_x=1.234,
        floor_y=4.567,
        role="held_out",
    )

    result = HomographyValidator.validate_calibration_points(train_pts + [held_out_pt])
    res_pt = result.held_out_results[0]

    # 1. Predicted coordinates must come from unrounded transformation path
    assert pytest.approx(res_pt.predicted_floor_point[0], abs=1e-5) == 1.234
    assert pytest.approx(res_pt.predicted_floor_point[1], abs=1e-5) == 4.567
    assert res_pt.predicted_floor_point[0] != 1.23  # Would be 1.23 if transform_point() was used
    assert res_pt.predicted_floor_point[1] != 4.57  # Would be 4.57 if transform_point() was used

    # 2. Error must be calculated from unrounded coordinates (approx 0.0, not 0.005)
    assert pytest.approx(res_pt.error, abs=1e-5) == 0.0
    assert res_pt.error < 0.001  # If rounded, error would be 0.005

