"""
Phase 8.5 — Unit Tests for Track Fragmentation Analysis
======================================================
Validates that:
1. One ground-truth identity mapped to one tracker ID is correctly evaluated as unfragmented.
2. One ground-truth identity mapped to two tracker IDs is correctly marked fragmented.
3. Multiple ground-truth identities yield deterministic aggregate counts and fragmentation rates.
4. Frame coverage calculations avoid double-counting overlapping tracker frames.
5. Missing or incomplete explicit correspondence raises clear ValueError.
6. Unknown ground-truth IDs in correspondence raise clear ValueError.
7. Unknown tracker IDs in correspondence raise clear ValueError.
8. Duplicate tracker IDs inside a single correspondence mapping raise clear ValueError.
9. Ambiguous reuse of one tracker ID across multiple ground-truth identities is rejected.
10. Empty ground-truth observations return an explicit INSUFFICIENT_EVIDENCE status.
11. Serialization to and from dictionary/JSON preserves all fields, records, and metadata.
12. Convenience function and GroundTruthSequence adapter work seamlessly.
13. Multiple tracker input formats (Trajectory, TrajectoryPoint, dict) are handled consistently.
"""

import json
import pytest

from src.trajectory.trajectory import Trajectory, TrajectoryPoint
from src.validation.ground_truth import GroundTruthPoint, GroundTruthSequence
from src.validation.fragmentation import (
    PersonFragmentationRecord,
    TrackFragmentationResult,
    TrackFragmentationValidator,
    validate_track_fragmentation,
)


def _make_trajectory(track_id: int, frames: list) -> Trajectory:
    """Helper to build a Trajectory with specified frame indices."""
    traj = Trajectory(track_id=track_id)
    for f in frames:
        traj.add_point(
            TrajectoryPoint(
                track_id=track_id,
                frame_idx=f,
                timestamp_sec=round(f * 0.2, 3),
                image_x=100.0,
                image_y=100.0,
            )
        )
    return traj


def _make_gt_points(gt_id: str, frames: list) -> list:
    """Helper to build GroundTruthPoints with specified frame indices."""
    return [
        GroundTruthPoint(
            gt_id=gt_id,
            frame_idx=f,
            timestamp_sec=round(f * 0.2, 3),
        )
        for f in frames
    ]


def test_single_gt_mapped_to_single_tracker_id():
    """Test 1: One GT person mapped to one tracker track is unfragmented with full coverage."""
    gt_points = _make_gt_points("person_1", [0, 1, 2, 3, 4])
    trajectories = [_make_trajectory(101, [0, 1, 2, 3, 4])]
    correspondence = {"person_1": [101]}

    result = TrackFragmentationValidator.validate_fragmentation(
        gt_observations=gt_points,
        tracker_observations=trajectories,
        correspondence=correspondence,
        dataset_name="synthetic_unfragmented",
    )

    assert result.evidence_status == "FRAGMENTATION_EVALUATED"
    assert result.total_gt_identities == 1
    assert result.total_fragmented_identities == 0
    assert result.total_unfragmented_identities == 1
    assert result.total_observed_fragments == 1
    assert result.fragmentation_rate == 0.0
    assert result.mean_fragments_per_identity == 1.0
    assert result.overall_coverage_ratio == 1.0

    record = result.person_records[0]
    assert record.gt_id == "person_1"
    assert record.associated_track_ids == [101]
    assert record.fragment_count == 1
    assert record.is_fragmented is False
    assert record.gt_frame_count == 5
    assert record.mapped_tracker_frame_count == 5
    assert record.covered_frame_count == 5
    assert record.coverage_ratio == 1.0


def test_single_gt_mapped_to_two_tracker_ids_marked_fragmented():
    """Test 2: One GT person mapped to two tracker tracks is marked fragmented."""
    # GT active frames 0 to 9 (10 frames)
    gt_points = _make_gt_points("person_1", list(range(10)))
    # Track 101 covers 0..4, Track 102 covers 5..9
    traj101 = _make_trajectory(101, list(range(5)))
    traj102 = _make_trajectory(102, list(range(5, 10)))
    correspondence = {"person_1": [101, 102]}

    result = TrackFragmentationValidator.validate_fragmentation(
        gt_observations=gt_points,
        tracker_observations=[traj101, traj102],
        correspondence=correspondence,
        dataset_name="synthetic_fragmented",
    )

    assert result.evidence_status == "FRAGMENTATION_EVALUATED"
    assert result.total_gt_identities == 1
    assert result.total_fragmented_identities == 1
    assert result.total_unfragmented_identities == 0
    assert result.total_observed_fragments == 2
    assert result.fragmentation_rate == 1.0
    assert result.mean_fragments_per_identity == 2.0
    assert result.overall_coverage_ratio == 1.0

    record = result.person_records[0]
    assert record.gt_id == "person_1"
    assert record.associated_track_ids == [101, 102]
    assert record.fragment_count == 2
    assert record.is_fragmented is True
    assert record.gt_frame_count == 10
    assert record.mapped_tracker_frame_count == 10
    assert record.covered_frame_count == 10
    assert record.coverage_ratio == 1.0


def test_multiple_gt_identities_deterministic_aggregate_counts():
    """Test 3: Multiple GT persons yield deterministic aggregate fragmentation counts."""
    gt_p1 = _make_gt_points("person_1", [0, 1, 2, 3, 4])          # 5 frames
    gt_p2 = _make_gt_points("person_2", list(range(10)))          # 10 frames
    gt_p3 = _make_gt_points("person_3", list(range(10)))          # 10 frames
    all_gt = gt_p1 + gt_p2 + gt_p3

    trajs = [
        _make_trajectory(101, [0, 1, 2, 3, 4]),                  # p1 (1 track)
        _make_trajectory(102, [0, 1, 2, 3, 4]),                  # p2 (track 1)
        _make_trajectory(103, [5, 6, 7, 8, 9]),                  # p2 (track 2)
        _make_trajectory(104, [0, 1, 2]),                        # p3 (track 1)
        _make_trajectory(105, [3, 4, 5]),                        # p3 (track 2)
        _make_trajectory(106, [6, 7, 8, 9]),                     # p3 (track 3)
    ]

    correspondence = {
        "person_1": [101],
        "person_2": [102, 103],
        "person_3": [104, 105, 106],
    }

    result = TrackFragmentationValidator.validate_fragmentation(
        gt_observations=all_gt,
        tracker_observations=trajs,
        correspondence=correspondence,
    )

    assert result.total_gt_identities == 3
    assert result.total_fragmented_identities == 2  # p2 and p3
    assert result.total_unfragmented_identities == 1  # p1
    assert result.total_observed_fragments == 6     # 1 + 2 + 3
    assert pytest.approx(result.fragmentation_rate, abs=1e-4) == round(2 / 3, 4)
    assert result.mean_fragments_per_identity == 2.0
    assert result.overall_coverage_ratio == 1.0

    # Records must be deterministically sorted by gt_id
    assert [r.gt_id for r in result.person_records] == ["person_1", "person_2", "person_3"]
    assert [r.fragment_count for r in result.person_records] == [1, 2, 3]
    assert [r.is_fragmented for r in result.person_records] == [False, True, True]


def test_frame_coverage_avoids_double_counting_overlapping_tracker_frames():
    """
    Test 4: Overlapping frames between two tracks mapped to one person are NOT double-counted.
    GT active on frames 0..9 (10 frames).
    Track 101 on frames 0..6 (7 frames).
    Track 102 on frames 4..9 (6 frames).
    Overlap on frames 4, 5, 6 (3 frames).
    Total unique mapped frames = 10, NOT 13.
    """
    gt_points = _make_gt_points("person_1", list(range(10)))
    traj101 = _make_trajectory(101, list(range(7)))
    traj102 = _make_trajectory(102, list(range(4, 10)))
    correspondence = {"person_1": [101, 102]}

    result = TrackFragmentationValidator.validate_fragmentation(
        gt_observations=gt_points,
        tracker_observations=[traj101, traj102],
        correspondence=correspondence,
    )

    record = result.person_records[0]
    assert record.gt_frame_count == 10
    # Overlapping frames 4, 5, 6 must be counted only once
    assert record.mapped_tracker_frame_count == 10
    assert record.covered_frame_count == 10
    assert record.coverage_ratio == 1.0


def test_frame_coverage_with_untracked_gap():
    """Test 4b: Partial coverage with an untracked gap calculates coverage ratio accurately."""
    gt_points = _make_gt_points("person_1", list(range(10)))  # 10 frames: 0..9
    # Tracks cover 0..2 and 7..9. Gap at 3, 4, 5, 6 (4 frames missing)
    traj101 = _make_trajectory(101, [0, 1, 2])
    traj102 = _make_trajectory(102, [7, 8, 9])
    correspondence = {"person_1": [101, 102]}

    result = TrackFragmentationValidator.validate_fragmentation(
        gt_observations=gt_points,
        tracker_observations=[traj101, traj102],
        correspondence=correspondence,
    )

    record = result.person_records[0]
    assert record.gt_frame_count == 10
    assert record.mapped_tracker_frame_count == 6
    assert record.covered_frame_count == 6
    assert record.coverage_ratio == 0.60


def test_missing_or_incomplete_explicit_correspondence():
    """Test 5: Missing, empty, or incomplete correspondence raises ValueError."""
    gt_points = _make_gt_points("person_1", [0, 1]) + _make_gt_points("person_2", [0, 1])
    trajs = [_make_trajectory(101, [0, 1])]

    # None correspondence
    with pytest.raises(ValueError, match="An explicit correspondence mapping"):
        TrackFragmentationValidator.validate_fragmentation(gt_points, trajs, None)

    # Empty correspondence dict
    with pytest.raises(ValueError, match="An explicit correspondence mapping"):
        TrackFragmentationValidator.validate_fragmentation(gt_points, trajs, {})

    # Incomplete correspondence (person_2 missing from mapping)
    with pytest.raises(ValueError, match="Ground-truth ID 'person_2' has no explicit correspondence mapping"):
        TrackFragmentationValidator.validate_fragmentation(gt_points, trajs, {"person_1": [101]})


def test_unknown_gt_id_in_correspondence():
    """Test 6: Ground-truth ID in correspondence not present in GT observations raises ValueError."""
    gt_points = _make_gt_points("person_1", [0, 1])
    trajs = [_make_trajectory(101, [0, 1]), _make_trajectory(102, [0, 1])]
    correspondence = {"person_1": [101], "ghost_person": [102]}

    with pytest.raises(ValueError, match="Unknown ground-truth ID 'ghost_person' in correspondence"):
        TrackFragmentationValidator.validate_fragmentation(gt_points, trajs, correspondence)


def test_unknown_tracker_id_in_correspondence():
    """Test 7: Tracker ID in correspondence not present in tracker observations raises ValueError."""
    gt_points = _make_gt_points("person_1", [0, 1])
    trajs = [_make_trajectory(101, [0, 1])]
    correspondence = {"person_1": [101, 999]}  # 999 unknown

    with pytest.raises(ValueError, match="Unknown tracker ID 999 in correspondence"):
        TrackFragmentationValidator.validate_fragmentation(gt_points, trajs, correspondence)


def test_duplicate_tracker_id_inside_one_mapping():
    """Test 8: Duplicate tracker ID in a single person's mapping raises ValueError."""
    gt_points = _make_gt_points("person_1", [0, 1])
    trajs = [_make_trajectory(101, [0, 1])]
    correspondence = {"person_1": [101, 101]}  # Duplicate 101

    with pytest.raises(ValueError, match="Duplicate tracker ID found in correspondence mapping for gt_id 'person_1'"):
        TrackFragmentationValidator.validate_fragmentation(gt_points, trajs, correspondence)


def test_ambiguous_reuse_of_tracker_id_across_multiple_gt_ids():
    """Test 9: Assigning the same tracker ID to multiple GT identities raises ValueError."""
    gt_points = _make_gt_points("person_1", [0, 1]) + _make_gt_points("person_2", [0, 1])
    trajs = [_make_trajectory(101, [0, 1])]
    correspondence = {"person_1": [101], "person_2": [101]}  # Ambiguous reuse of 101

    with pytest.raises(ValueError, match="Ambiguous correspondence: tracker ID 101 is mapped to multiple"):
        TrackFragmentationValidator.validate_fragmentation(gt_points, trajs, correspondence)


def test_empty_ground_truth_returns_insufficient_evidence():
    """Test 10: Empty ground-truth input returns INSUFFICIENT_EVIDENCE status."""
    trajs = [_make_trajectory(101, [0, 1])]
    result = TrackFragmentationValidator.validate_fragmentation(
        gt_observations=[],
        tracker_observations=trajs,
        correspondence=None,
        dataset_name="empty_test",
    )

    assert result.evidence_status == "INSUFFICIENT_EVIDENCE"
    assert result.total_gt_identities == 0
    assert result.total_fragmented_identities == 0
    assert result.total_observed_fragments == 0
    assert result.fragmentation_rate is None
    assert result.mean_fragments_per_identity is None
    assert result.overall_coverage_ratio is None
    assert "No ground-truth observations provided" in result.evidence_note


def test_serialization_round_trip():
    """Test 11: TrackFragmentationResult serializes to dict/JSON and deserializes losslessly."""
    gt_p1 = _make_gt_points("person_1", [0, 1, 2])
    gt_p2 = _make_gt_points("person_2", [0, 1, 2, 3])
    trajs = [
        _make_trajectory(101, [0, 1, 2]),
        _make_trajectory(102, [0, 1]),
        _make_trajectory(103, [2, 3]),
    ]
    correspondence = {"person_1": [101], "person_2": [102, 103]}

    result = TrackFragmentationValidator.validate_fragmentation(
        gt_observations=gt_p1 + gt_p2,
        tracker_observations=trajs,
        correspondence=correspondence,
        dataset_name="json_seq",
    )

    data = result.to_dict()
    json_str = json.dumps(data)
    reconstructed = TrackFragmentationResult.from_dict(json.loads(json_str))

    assert reconstructed.dataset_name == result.dataset_name
    assert reconstructed.total_gt_identities == result.total_gt_identities
    assert reconstructed.total_fragmented_identities == result.total_fragmented_identities
    assert reconstructed.total_unfragmented_identities == result.total_unfragmented_identities
    assert reconstructed.total_observed_fragments == result.total_observed_fragments
    assert reconstructed.fragmentation_rate == result.fragmentation_rate
    assert reconstructed.mean_fragments_per_identity == result.mean_fragments_per_identity
    assert reconstructed.overall_coverage_ratio == result.overall_coverage_ratio
    assert reconstructed.evidence_status == result.evidence_status
    assert len(reconstructed.person_records) == 2

    r0 = reconstructed.person_records[0]
    assert r0.gt_id == "person_1"
    assert r0.associated_track_ids == [101]
    assert r0.is_fragmented is False

    r1 = reconstructed.person_records[1]
    assert r1.gt_id == "person_2"
    assert r1.associated_track_ids == [102, 103]
    assert r1.is_fragmented is True


def test_sequence_adapter_and_convenience_function():
    """Test 12: Sequence adapter and validate_track_fragmentation helper function."""
    gt_pts = _make_gt_points("person_1", [0, 1, 2])
    seq = GroundTruthSequence(
        schema_version="1.0",
        sequence_id="SEQ_FRAG_01",
        source="synthetic",
        fps=5.0,
        points=gt_pts,
    )
    trajs = [_make_trajectory(101, [0, 1, 2])]
    correspondence = {"person_1": [101]}

    res = validate_track_fragmentation(seq, trajs, correspondence)
    assert res.dataset_name == "SEQ_FRAG_01"
    assert res.total_gt_identities == 1
    assert res.total_fragmented_identities == 0
    assert res.evidence_status == "FRAGMENTATION_EVALUATED"


def test_tracker_input_formats():
    """Test 13: Multiple tracker input formats (Trajectory, TrajectoryPoint, dict) work identically."""
    gt_points = _make_gt_points("person_1", [0, 1, 2])
    correspondence = {"person_1": [101]}

    # Format 1: List[Trajectory]
    trajs = [_make_trajectory(101, [0, 1, 2])]
    res1 = TrackFragmentationValidator.validate_fragmentation(gt_points, trajs, correspondence)

    # Format 2: List[TrajectoryPoint]
    pts = trajs[0].points
    res2 = TrackFragmentationValidator.validate_fragmentation(gt_points, pts, correspondence)

    # Format 3: Dict[int, Set[int]]
    frame_dict = {101: {0, 1, 2}}
    res3 = TrackFragmentationValidator.validate_fragmentation(gt_points, frame_dict, correspondence)

    assert res1.total_gt_identities == res2.total_gt_identities == res3.total_gt_identities == 1
    assert res1.overall_coverage_ratio == res2.overall_coverage_ratio == res3.overall_coverage_ratio == 1.0
