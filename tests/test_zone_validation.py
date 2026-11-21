"""
Phase 8.3 — Unit Tests for Zone Assignment Agreement Validation
==============================================================
Validates that:
1. Zone label agreement is evaluated on explicitly aligned observation pairs.
2. An explicit id_mapping (gt_id -> track_id) is strictly required; automatic
   matching, Re-ID, or heuristic pairing is never performed.
3. Missing alignment is rejected without silent inference.
4. Duplicate alignment keys raise clear validation errors.
5. Outside areas are categorically represented as `None` in comparisons and confusion matrices.
6. Disagreements in both directions (zone vs zone, zone vs None, None vs zone) are recorded.
7. Empty observations yield an explicit INSUFFICIENT_EVIDENCE status.
8. JSON serialization and deserialization round-trips restore None keys accurately.

All tests utilize synthetic fixtures to verify evaluation mechanics.
They do NOT represent physical real-world accuracy claims.
"""

import json
import pytest

from src.trajectory.trajectory import TrajectoryPoint
from src.validation.ground_truth import GroundTruthPoint
from src.validation.zone_validation import (
    AlignedZoneObservation,
    ZoneDisagreement,
    ZoneValidationResult,
    ZoneValidator,
    validate_zone_assignments,
)


@pytest.fixture
def synthetic_ground_truth_and_predictions():
    """
    Synthetic fixture with:
    - 2 ground-truth IDs ("person_1", "person_2") explicitly mapped to tracker IDs (101, 102).
    - Frames 0 to 4 (5 frames each, 10 aligned pairs total).
    - 3 named zones ("shelf_a", "shelf_b", "main_aisle") plus outside (None).
    """
    id_mapping = {"person_1": 101, "person_2": 102}

    gt_points = [
        # person_1: shelf_a -> shelf_a -> main_aisle -> shelf_b -> None (outside)
        GroundTruthPoint("person_1", 0, 0.0, zone_id="shelf_a"),
        GroundTruthPoint("person_1", 1, 0.2, zone_id="shelf_a"),
        GroundTruthPoint("person_1", 2, 0.4, zone_id="main_aisle"),
        GroundTruthPoint("person_1", 3, 0.6, zone_id="shelf_b"),
        GroundTruthPoint("person_1", 4, 0.8, zone_id=None),
        # person_2: None (outside) -> main_aisle -> shelf_b -> shelf_b -> shelf_a
        GroundTruthPoint("person_2", 0, 0.0, zone_id=None),
        GroundTruthPoint("person_2", 1, 0.2, zone_id="main_aisle"),
        GroundTruthPoint("person_2", 2, 0.4, zone_id="shelf_b"),
        GroundTruthPoint("person_2", 3, 0.6, zone_id="shelf_b"),
        GroundTruthPoint("person_2", 4, 0.8, zone_id="shelf_a"),
    ]

    # Perfect predictions matching ground truth identically
    pred_points_perfect = [
        # track 101 (person_1)
        TrajectoryPoint(101, 0, 0.0, 100.0, 100.0, zone_id="shelf_a"),
        TrajectoryPoint(101, 1, 0.2, 100.0, 100.0, zone_id="shelf_a"),
        TrajectoryPoint(101, 2, 0.4, 150.0, 150.0, zone_id="main_aisle"),
        TrajectoryPoint(101, 3, 0.6, 200.0, 200.0, zone_id="shelf_b"),
        TrajectoryPoint(101, 4, 0.8, 250.0, 250.0, zone_id=None),
        # track 102 (person_2)
        TrajectoryPoint(102, 0, 0.0, 50.0, 50.0, zone_id=None),
        TrajectoryPoint(102, 1, 0.2, 150.0, 150.0, zone_id="main_aisle"),
        TrajectoryPoint(102, 2, 0.4, 200.0, 200.0, zone_id="shelf_b"),
        TrajectoryPoint(102, 3, 0.6, 200.0, 200.0, zone_id="shelf_b"),
        TrajectoryPoint(102, 4, 0.8, 100.0, 100.0, zone_id="shelf_a"),
    ]

    return gt_points, pred_points_perfect, id_mapping


def test_perfect_zone_agreement(synthetic_ground_truth_and_predictions):
    """Test 1: Identical zone labels produce 100% agreement and zero disagreements."""
    gt_points, pred_points, id_mapping = synthetic_ground_truth_and_predictions

    result = ZoneValidator.align_and_validate(
        expected_points=gt_points,
        predicted_points=pred_points,
        id_mapping=id_mapping,
        dataset_name="synthetic_aligned",
    )

    assert result.total_aligned_observations == 10
    assert result.matched_zone_count == 10
    assert result.mismatched_zone_count == 0
    assert result.agreement_rate == 1.0
    assert len(result.disagreements) == 0
    assert result.evidence_status == "ZONE_AGREEMENT_EVALUATED"
    assert "10 explicitly aligned observations" in result.evidence_note

    # Check that diagonal in confusion matrix equals expected counts
    for zone, count in result.expected_zone_counts.items():
        assert result.confusion_matrix[zone][zone] == count
        for other_zone in result.expected_zone_counts:
            if other_zone != zone:
                assert result.confusion_matrix[zone][other_zone] == 0


def test_expected_disagreements_and_none_transitions(synthetic_ground_truth_and_predictions):
    """
    Test 2: Validates detection of disagreements across named zones and None.
    Introduces 3 mismatches:
    - person_1, frame 1: expected 'shelf_a', predicted 'shelf_b' (zone vs zone)
    - person_1, frame 4: expected None, predicted 'shelf_a' (None vs zone)
    - person_2, frame 2: expected 'shelf_b', predicted None (zone vs None)
    """
    gt_points, _, id_mapping = synthetic_ground_truth_and_predictions

    pred_points_with_errors = [
        # track 101 (person_1)
        TrajectoryPoint(101, 0, 0.0, 100.0, 100.0, zone_id="shelf_a"),
        TrajectoryPoint(101, 1, 0.2, 100.0, 100.0, zone_id="shelf_b"),     # Disagreement: shelf_a -> shelf_b
        TrajectoryPoint(101, 2, 0.4, 150.0, 150.0, zone_id="main_aisle"),
        TrajectoryPoint(101, 3, 0.6, 200.0, 200.0, zone_id="shelf_b"),
        TrajectoryPoint(101, 4, 0.8, 250.0, 250.0, zone_id="shelf_a"),     # Disagreement: None -> shelf_a
        # track 102 (person_2)
        TrajectoryPoint(102, 0, 0.0, 50.0, 50.0, zone_id=None),
        TrajectoryPoint(102, 1, 0.2, 150.0, 150.0, zone_id="main_aisle"),
        TrajectoryPoint(102, 2, 0.4, 200.0, 200.0, zone_id=None),          # Disagreement: shelf_b -> None
        TrajectoryPoint(102, 3, 0.6, 200.0, 200.0, zone_id="shelf_b"),
        TrajectoryPoint(102, 4, 0.8, 100.0, 100.0, zone_id="shelf_a"),
    ]

    result = ZoneValidator.align_and_validate(
        expected_points=gt_points,
        predicted_points=pred_points_with_errors,
        id_mapping=id_mapping,
    )

    assert result.total_aligned_observations == 10
    assert result.matched_zone_count == 7
    assert result.mismatched_zone_count == 3
    assert result.agreement_rate == 0.70
    assert len(result.disagreements) == 3

    # Inspect disagreement records
    d0 = result.disagreements[0]
    assert d0.gt_id == "person_1"
    assert d0.track_id == 101
    assert d0.frame_idx == 1
    assert d0.expected_zone == "shelf_a"
    assert d0.predicted_zone == "shelf_b"

    d1 = result.disagreements[1]
    assert d1.gt_id == "person_1"
    assert d1.track_id == 101
    assert d1.frame_idx == 4
    assert d1.expected_zone is None
    assert d1.predicted_zone == "shelf_a"

    d2 = result.disagreements[2]
    assert d2.gt_id == "person_2"
    assert d2.track_id == 102
    assert d2.frame_idx == 2
    assert d2.expected_zone == "shelf_b"
    assert d2.predicted_zone is None

    # Check confusion matrix cell counts
    cm = result.confusion_matrix
    assert cm["shelf_a"]["shelf_b"] == 1
    assert cm[None]["shelf_a"] == 1
    assert cm["shelf_b"][None] == 1
    assert cm[None][None] == 1  # person_2, frame 0 matched None -> None


def test_confusion_matrix_structure_and_row_totals():
    """Test 3: Confusion matrix preserves None category and row/column sums match distributions."""
    obs = [
        AlignedZoneObservation("p1", 1, 0, 0.0, expected_zone="zone_1", predicted_zone="zone_1"),
        AlignedZoneObservation("p1", 1, 1, 0.1, expected_zone="zone_1", predicted_zone="zone_2"),
        AlignedZoneObservation("p1", 1, 2, 0.2, expected_zone="zone_2", predicted_zone=None),
        AlignedZoneObservation("p1", 1, 3, 0.3, expected_zone=None, predicted_zone=None),
        AlignedZoneObservation("p1", 1, 4, 0.4, expected_zone=None, predicted_zone="zone_1"),
    ]

    result = ZoneValidator.validate_observations(
        observations=obs,
        known_zones=["zone_1", "zone_2", "zone_3"],  # zone_3 unobserved
    )

    cm = result.confusion_matrix
    # All known zones plus observed zones plus None should be keys
    assert "zone_1" in cm
    assert "zone_2" in cm
    assert "zone_3" in cm
    assert None in cm

    # Row sums equal expected counts
    for exp_z, exp_count in result.expected_zone_counts.items():
        row_sum = sum(cm[exp_z].values())
        assert row_sum == exp_count

    # Column sums equal predicted counts
    for pred_z, pred_count in result.predicted_zone_counts.items():
        col_sum = sum(cm[exp_z][pred_z] for exp_z in cm)
        assert col_sum == pred_count


def test_duplicate_alignment_keys_rejected():
    """Test 4: Duplicate alignment keys in observations raise ValueError."""
    # Duplicate (gt_id, frame_idx) in validate_observations
    obs_dup_gt = [
        AlignedZoneObservation("p1", 1, 0, 0.0, "shelf_a", "shelf_a"),
        AlignedZoneObservation("p1", 2, 0, 0.0, "shelf_a", "shelf_a"),  # duplicate p1 at frame 0
    ]
    with pytest.raises(ValueError, match="Duplicate alignment key detected: gt_id='p1' at frame_idx 0"):
        ZoneValidator.validate_observations(obs_dup_gt)

    # Duplicate (track_id, frame_idx) in validate_observations
    obs_dup_track = [
        AlignedZoneObservation("p1", 1, 0, 0.0, "shelf_a", "shelf_a"),
        AlignedZoneObservation("p2", 1, 0, 0.0, "shelf_a", "shelf_a"),  # duplicate track 1 at frame 0
    ]
    with pytest.raises(ValueError, match="Duplicate alignment key detected: track_id=1 at frame_idx 0"):
        ZoneValidator.validate_observations(obs_dup_track)

    # Duplicate in align_and_validate for predicted points
    gt_pts = [GroundTruthPoint("p1", 0, 0.0, zone_id="shelf_a")]
    pred_pts_dup = [
        TrajectoryPoint(101, 0, 0.0, 10.0, 10.0, zone_id="shelf_a"),
        TrajectoryPoint(101, 0, 0.0, 12.0, 12.0, zone_id="shelf_a"),
    ]
    with pytest.raises(ValueError, match="Duplicate predicted observation detected"):
        ZoneValidator.align_and_validate(gt_pts, pred_pts_dup, id_mapping={"p1": 101})

    # Duplicate in align_and_validate for ground-truth points
    gt_pts_dup = [
        GroundTruthPoint("p1", 0, 0.0, zone_id="shelf_a"),
        GroundTruthPoint("p1", 0, 0.0, zone_id="shelf_b"),
    ]
    pred_pts = [TrajectoryPoint(101, 0, 0.0, 10.0, 10.0, zone_id="shelf_a")]
    with pytest.raises(ValueError, match="Duplicate ground-truth observation detected"):
        ZoneValidator.align_and_validate(gt_pts_dup, pred_pts, id_mapping={"p1": 101})


def test_missing_alignment_rejected_without_silent_inference():
    """Test 5: Missing tracker observation at a ground-truth frame raises ValueError."""
    gt_points = [
        GroundTruthPoint("person_1", 0, 0.0, zone_id="shelf_a"),
        GroundTruthPoint("person_1", 1, 0.2, zone_id="shelf_a"),  # Missing in predicted
    ]
    pred_points = [
        TrajectoryPoint(101, 0, 0.0, 100.0, 100.0, zone_id="shelf_a"),
        # Frame 1 is missing
    ]
    id_mapping = {"person_1": 101}

    # require_complete_alignment=True (default) must raise ValueError
    with pytest.raises(ValueError, match="Missing predicted observation for track_id=101"):
        ZoneValidator.align_and_validate(gt_points, pred_points, id_mapping)

    # require_complete_alignment=False skips missing frame
    res_partial = ZoneValidator.align_and_validate(
        gt_points, pred_points, id_mapping, require_complete_alignment=False
    )
    assert res_partial.total_aligned_observations == 1
    assert res_partial.matched_zone_count == 1


def test_id_mapping_validation():
    """Test 6: Invalid or unmapped id_mapping raises explicit ValueError."""
    gt_pts = [GroundTruthPoint("person_1", 0, 0.0, zone_id="shelf_a")]
    pred_pts = [TrajectoryPoint(101, 0, 0.0, 100.0, 100.0, zone_id="shelf_a")]

    # Empty dict
    with pytest.raises(ValueError, match="An explicit id_mapping dictionary"):
        ZoneValidator.align_and_validate(gt_pts, pred_pts, id_mapping={})

    # None
    with pytest.raises(ValueError, match="An explicit id_mapping dictionary"):
        ZoneValidator.align_and_validate(gt_pts, pred_pts, id_mapping=None)

    # Missing mapping for ground-truth identity
    with pytest.raises(ValueError, match="has no explicit tracker ID mapping"):
        ZoneValidator.align_and_validate(gt_pts, pred_pts, id_mapping={"unrelated_person": 999})


def test_insufficient_evidence_when_observations_empty():
    """Test 7: Empty observation list returns INSUFFICIENT_EVIDENCE status."""
    result = ZoneValidator.validate_observations([], dataset_name="empty_seq")

    assert result.total_aligned_observations == 0
    assert result.matched_zone_count == 0
    assert result.mismatched_zone_count == 0
    assert result.agreement_rate is None
    assert result.expected_zone_counts == {}
    assert result.predicted_zone_counts == {}
    assert result.confusion_matrix == {}
    assert len(result.disagreements) == 0
    assert result.evidence_status == "INSUFFICIENT_EVIDENCE"
    assert "No aligned zone observations provided" in result.evidence_note


def test_serialization_round_trip(synthetic_ground_truth_and_predictions):
    """
    Test 8: ZoneValidationResult converts to dict and back, accurately restoring None keys.
    """
    gt_points, _, id_mapping = synthetic_ground_truth_and_predictions

    pred_points = [
        TrajectoryPoint(101, 0, 0.0, 100.0, 100.0, zone_id="shelf_a"),
        TrajectoryPoint(101, 1, 0.2, 100.0, 100.0, zone_id="shelf_b"),  # Disagreement
        TrajectoryPoint(101, 2, 0.4, 150.0, 150.0, zone_id="main_aisle"),
        TrajectoryPoint(101, 3, 0.6, 200.0, 200.0, zone_id="shelf_b"),
        TrajectoryPoint(101, 4, 0.8, 250.0, 250.0, zone_id=None),
        TrajectoryPoint(102, 0, 0.0, 50.0, 50.0, zone_id=None),
        TrajectoryPoint(102, 1, 0.2, 150.0, 150.0, zone_id="main_aisle"),
        TrajectoryPoint(102, 2, 0.4, 200.0, 200.0, zone_id="shelf_b"),
        TrajectoryPoint(102, 3, 0.6, 200.0, 200.0, zone_id="shelf_b"),
        TrajectoryPoint(102, 4, 0.8, 100.0, 100.0, zone_id="shelf_a"),
    ]

    result = ZoneValidator.align_and_validate(gt_points, pred_points, id_mapping)
    data = result.to_dict()

    # JSON round trip simulation (dict -> JSON string -> dict -> from_dict)
    json_str = json.dumps(data)
    parsed_json = json.loads(json_str)
    reconstructed = ZoneValidationResult.from_dict(parsed_json)

    assert reconstructed.dataset_name == result.dataset_name
    assert reconstructed.total_aligned_observations == result.total_aligned_observations
    assert reconstructed.matched_zone_count == result.matched_zone_count
    assert reconstructed.mismatched_zone_count == result.mismatched_zone_count
    assert reconstructed.agreement_rate == result.agreement_rate
    assert reconstructed.evidence_status == result.evidence_status

    # None keys must be restored as Python None, not string 'null'
    assert None in reconstructed.expected_zone_counts
    assert reconstructed.expected_zone_counts[None] == result.expected_zone_counts[None]
    assert None in reconstructed.predicted_zone_counts
    assert reconstructed.predicted_zone_counts[None] == result.predicted_zone_counts[None]

    assert None in reconstructed.confusion_matrix
    assert None in reconstructed.confusion_matrix[None]
    assert reconstructed.confusion_matrix[None][None] == result.confusion_matrix[None][None]

    # Verify disagreement list reconstruction
    assert len(reconstructed.disagreements) == len(result.disagreements)
    d_orig = result.disagreements[0]
    d_recon = reconstructed.disagreements[0]
    assert d_recon.gt_id == d_orig.gt_id
    assert d_recon.track_id == d_orig.track_id
    assert d_recon.frame_idx == d_orig.frame_idx
    assert d_recon.expected_zone == d_orig.expected_zone
    assert d_recon.predicted_zone == d_orig.predicted_zone


def test_convenience_function_validate_zone_assignments():
    """Test 9: Convenience wrapper validate_zone_assignments produces identical results."""
    obs = [
        AlignedZoneObservation("p1", 1, 0, 0.0, "aisle", "aisle"),
        AlignedZoneObservation("p1", 1, 1, 0.1, "aisle", "promo"),
    ]
    res1 = ZoneValidator.validate_observations(obs, dataset_name="convenience_test")
    res2 = validate_zone_assignments(obs, dataset_name="convenience_test")

    assert res1.total_aligned_observations == res2.total_aligned_observations
    assert res1.agreement_rate == res2.agreement_rate
    assert res1.matched_zone_count == res2.matched_zone_count


def test_aligned_observation_dataclass_validation():
    """Test 10: AlignedZoneObservation enforces types, non-empty IDs, and computes is_match."""
    # Valid observation
    obs = AlignedZoneObservation(
        gt_id="person_1",
        track_id=10,
        frame_idx=5,
        timestamp_sec=1.0,
        expected_zone="zone_a",
        predicted_zone="zone_a",
    )
    assert obs.is_match is True

    # Valid observation matching None
    obs_none = AlignedZoneObservation(
        gt_id="person_1",
        track_id=10,
        frame_idx=6,
        timestamp_sec=1.2,
        expected_zone=None,
        predicted_zone=None,
    )
    assert obs_none.is_match is True

    # Mismatch with None
    obs_mismatch = AlignedZoneObservation(
        gt_id="person_1",
        track_id=10,
        frame_idx=7,
        timestamp_sec=1.4,
        expected_zone="zone_a",
        predicted_zone=None,
    )
    assert obs_mismatch.is_match is False

    # Invalid gt_id
    with pytest.raises(ValueError, match="gt_id must be a non-empty string"):
        AlignedZoneObservation("", 10, 0, 0.0, None, None)

    # Invalid track_id
    with pytest.raises(ValueError, match="track_id must be an integer"):
        AlignedZoneObservation("p1", "10", 0, 0.0, None, None)  # type: ignore

    with pytest.raises(ValueError, match="track_id must be an integer"):
        AlignedZoneObservation("p1", True, 0, 0.0, None, None)  # bool is not allowed

    # Invalid frame_idx
    with pytest.raises(ValueError, match="frame_idx must be a non-negative integer"):
        AlignedZoneObservation("p1", 10, -1, 0.0, None, None)

    # Invalid expected_zone
    with pytest.raises(ValueError, match="expected_zone must be a string or None"):
        AlignedZoneObservation("p1", 10, 0, 0.0, 123, None)  # type: ignore

    # Invalid predicted_zone
    with pytest.raises(ValueError, match="predicted_zone must be a string or None"):
        AlignedZoneObservation("p1", 10, 0, 0.0, None, 456)  # type: ignore


def test_aligned_observation_timestamp_validation():
    """
    Test 11: AlignedZoneObservation enforces finite, non-negative numeric timestamps.
    Covers:
    - valid zero and positive timestamps (int and float)
    - negative timestamp rejection
    - NaN rejection
    - infinite timestamp rejection (+inf and -inf)
    - non-numeric timestamp rejection
    - boolean timestamp rejection
    """
    # 1. Valid zero timestamp
    obs_zero = AlignedZoneObservation("p1", 1, 0, 0.0, None, None)
    assert obs_zero.timestamp_sec == 0.0
    assert isinstance(obs_zero.timestamp_sec, float)

    # 2. Valid positive integer timestamp converted to float
    obs_int = AlignedZoneObservation("p1", 1, 0, 5, None, None)
    assert obs_int.timestamp_sec == 5.0
    assert isinstance(obs_int.timestamp_sec, float)

    # 3. Valid positive float timestamp
    obs_pos = AlignedZoneObservation("p1", 1, 0, 12.345, None, None)
    assert obs_pos.timestamp_sec == 12.345
    assert isinstance(obs_pos.timestamp_sec, float)

    # 4. Negative timestamp rejection
    with pytest.raises(ValueError, match="timestamp_sec must be a non-negative finite number"):
        AlignedZoneObservation("p1", 1, 0, -0.001, None, None)

    with pytest.raises(ValueError, match="timestamp_sec must be a non-negative finite number"):
        AlignedZoneObservation("p1", 1, 0, -10.0, None, None)

    # 5. NaN rejection
    with pytest.raises(ValueError, match="timestamp_sec must be a non-negative finite number"):
        AlignedZoneObservation("p1", 1, 0, float("nan"), None, None)

    # 6. Infinite timestamp rejection
    with pytest.raises(ValueError, match="timestamp_sec must be a non-negative finite number"):
        AlignedZoneObservation("p1", 1, 0, float("inf"), None, None)

    with pytest.raises(ValueError, match="timestamp_sec must be a non-negative finite number"):
        AlignedZoneObservation("p1", 1, 0, float("-inf"), None, None)

    # 7. Non-numeric timestamp rejection
    with pytest.raises(ValueError, match="timestamp_sec must be a non-negative finite number"):
        AlignedZoneObservation("p1", 1, 0, "1.23", None, None)  # type: ignore

    with pytest.raises(ValueError, match="timestamp_sec must be a non-negative finite number"):
        AlignedZoneObservation("p1", 1, 0, None, None, None)  # type: ignore

    with pytest.raises(ValueError, match="timestamp_sec must be a non-negative finite number"):
        AlignedZoneObservation("p1", 1, 0, [1.0], None, None)  # type: ignore

    # 8. Boolean timestamp rejection
    with pytest.raises(ValueError, match="timestamp_sec must be a non-negative finite number"):
        AlignedZoneObservation("p1", 1, 0, True, None, None)  # type: ignore

    with pytest.raises(ValueError, match="timestamp_sec must be a non-negative finite number"):
        AlignedZoneObservation("p1", 1, 0, False, None, None)  # type: ignore
