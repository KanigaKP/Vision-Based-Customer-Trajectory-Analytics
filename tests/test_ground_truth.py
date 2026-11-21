"""
Phase 8.1 — Unit Tests for Ground-Truth Representation Schema
============================================================
Tests data models, validation constraints, temporal consistency checks,
training vs held-out calibration roles, and JSON round-trip serialization.

All fixtures are synthetic and intended strictly for schema verification.
They do NOT represent real-world physical validation evidence.
"""

import math
import os
import pytest

from src.validation.ground_truth import (
    GroundTruthPoint,
    GroundTruthEvent,
    GroundTruthCalibrationPoint,
    GroundTruthSequence,
    SUPPORTED_EVENT_TYPES,
    SUPPORTED_CALIBRATION_ROLES,
)


@pytest.fixture
def synthetic_sequence() -> GroundTruthSequence:
    """
    Synthetic fixture containing:
    - 2 distinct ground-truth identities (person_A, person_B)
    - Chronologically ordered frame/timestamp points
    - Image bounding boxes and image bottom-center points
    - Floor coordinates
    - Zone labels including an outside (None) state
    - Entry, transition, and exit events
    - Both training and held-out calibration points
    """
    calib_pts = [
        GroundTruthCalibrationPoint("P1", 100.0, 500.0, 0.0, 5.0, role="training"),
        GroundTruthCalibrationPoint("P2", 500.0, 500.0, 4.0, 5.0, role="training"),
        GroundTruthCalibrationPoint("P3", 500.0, 200.0, 4.0, 0.0, role="training"),
        GroundTruthCalibrationPoint("P4", 100.0, 200.0, 0.0, 0.0, role="training"),
        GroundTruthCalibrationPoint("CP1", 300.0, 350.0, 2.0, 2.5, role="held_out"),
        GroundTruthCalibrationPoint("CP2", 400.0, 400.0, 3.0, 3.5, role="held_out"),
    ]

    points = [
        # person_A observations
        GroundTruthPoint(
            gt_id="person_A",
            frame_idx=0,
            timestamp_sec=0.0,
            image_bbox=[100.0, 150.0, 140.0, 250.0],
            image_x=120.0,
            image_y=250.0,
            floor_x=1.0,
            floor_y=2.0,
            zone_id=None,  # Outside all defined zones
        ),
        GroundTruthPoint(
            gt_id="person_A",
            frame_idx=1,
            timestamp_sec=0.2,
            image_bbox=[105.0, 150.0, 145.0, 250.0],
            image_x=125.0,
            image_y=250.0,
            floor_x=1.2,
            floor_y=2.0,
            zone_id="zone_entrance",
        ),
        GroundTruthPoint(
            gt_id="person_A",
            frame_idx=2,
            timestamp_sec=0.4,
            image_bbox=[110.0, 150.0, 150.0, 250.0],
            image_x=130.0,
            image_y=250.0,
            floor_x=1.5,
            floor_y=2.2,
            zone_id="zone_entrance",
        ),
        GroundTruthPoint(
            gt_id="person_A",
            frame_idx=3,
            timestamp_sec=0.6,
            image_bbox=[120.0, 150.0, 160.0, 250.0],
            image_x=140.0,
            image_y=250.0,
            floor_x=2.5,
            floor_y=3.0,
            zone_id="zone_shelf",
        ),
        # person_B observations
        GroundTruthPoint(
            gt_id="person_B",
            frame_idx=1,
            timestamp_sec=0.2,
            image_bbox=[300.0, 200.0, 340.0, 300.0],
            image_x=320.0,
            image_y=300.0,
            floor_x=5.0,
            floor_y=4.0,
            zone_id="zone_reading",
        ),
        GroundTruthPoint(
            gt_id="person_B",
            frame_idx=2,
            timestamp_sec=0.4,
            image_bbox=[305.0, 200.0, 345.0, 300.0],
            image_x=325.0,
            image_y=300.0,
            floor_x=5.2,
            floor_y=4.0,
            zone_id="zone_reading",
        ),
        GroundTruthPoint(
            gt_id="person_B",
            frame_idx=3,
            timestamp_sec=0.6,
            image_bbox=[310.0, 200.0, 350.0, 300.0],
            image_x=330.0,
            image_y=300.0,
            floor_x=5.5,
            floor_y=4.2,
            zone_id=None,  # Exited to unassigned area
        ),
    ]

    events = [
        GroundTruthEvent(
            event_type="entry",
            gt_id="person_A",
            frame_idx=1,
            timestamp_sec=0.2,
            from_zone_id=None,
            to_zone_id="zone_entrance",
        ),
        GroundTruthEvent(
            event_type="transition",
            gt_id="person_A",
            frame_idx=3,
            timestamp_sec=0.6,
            from_zone_id="zone_entrance",
            to_zone_id="zone_shelf",
        ),
        GroundTruthEvent(
            event_type="exit",
            gt_id="person_B",
            frame_idx=3,
            timestamp_sec=0.6,
            from_zone_id="zone_reading",
            to_zone_id=None,
        ),
    ]

    return GroundTruthSequence(
        schema_version="1.0",
        sequence_id="SYNTHETIC_FIXTURE_SEQ01",
        source="synthetic_test_fixture",
        fps=5.0,
        points=points,
        events=events,
        calibration_points=calib_pts,
        metadata={"description": "Deterministic synthetic fixture for schema unit tests"},
    )


# ==============================================================================
# 1. Valid Construction of all Schema Records
# ==============================================================================

def test_valid_construction_of_all_schema_records():
    """Test 1: Valid construction of Point, Event, CalibrationPoint, and Sequence."""
    pt = GroundTruthPoint(
        gt_id="P1",
        frame_idx=0,
        timestamp_sec=0.0,
        image_bbox=[10.0, 20.0, 30.0, 40.0],
        image_x=20.0,
        image_y=40.0,
        floor_x=1.0,
        floor_y=2.0,
        zone_id="zone_A",
        visible=True,
    )
    assert pt.gt_id == "P1"
    assert pt.frame_idx == 0
    assert pt.timestamp_sec == 0.0
    assert pt.image_bbox == [10.0, 20.0, 30.0, 40.0]
    assert pt.visible is True

    ev = GroundTruthEvent(
        event_type="entry",
        gt_id="P1",
        frame_idx=5,
        timestamp_sec=1.0,
        from_zone_id=None,
        to_zone_id="zone_A",
    )
    assert ev.event_type == "entry"
    assert ev.to_zone_id == "zone_A"

    cp = GroundTruthCalibrationPoint(
        label="Ref1",
        image_x=100.0,
        image_y=200.0,
        floor_x=0.0,
        floor_y=0.0,
        role="training",
    )
    assert cp.label == "Ref1"
    assert cp.role == "training"

    seq = GroundTruthSequence(
        schema_version="1.0",
        sequence_id="SEQ01",
        source="unit_test",
        fps=10.0,
        points=[pt],
        events=[ev],
        calibration_points=[cp],
    )
    assert seq.sequence_id == "SEQ01"
    assert seq.fps == 10.0
    assert len(seq.points) == 1


# ==============================================================================
# 2. Validation of Empty IDs
# ==============================================================================

def test_validation_of_empty_ids():
    """Test 2: Empty, whitespace-only, or non-string IDs raise ValueError."""
    # Point gt_id
    with pytest.raises(ValueError, match="gt_id must be a non-empty string"):
        GroundTruthPoint(gt_id="", frame_idx=0, timestamp_sec=0.0)

    with pytest.raises(ValueError, match="gt_id must be a non-empty string"):
        GroundTruthPoint(gt_id="   ", frame_idx=0, timestamp_sec=0.0)

    # Event gt_id
    with pytest.raises(ValueError, match="gt_id must be a non-empty string"):
        GroundTruthEvent(event_type="entry", gt_id="", frame_idx=0, timestamp_sec=0.0, to_zone_id="Z1")

    # CalibrationPoint label
    with pytest.raises(ValueError, match="label must be a non-empty string"):
        GroundTruthCalibrationPoint(label="", image_x=0.0, image_y=0.0, floor_x=0.0, floor_y=0.0)

    # Sequence sequence_id and source
    with pytest.raises(ValueError, match="sequence_id must be a non-empty string"):
        GroundTruthSequence(schema_version="1.0", sequence_id="", source="test", fps=5.0)

    with pytest.raises(ValueError, match="source must be a non-empty string"):
        GroundTruthSequence(schema_version="1.0", sequence_id="S1", source="", fps=5.0)


# ==============================================================================
# 3. Validation of Negative Frame Indices
# ==============================================================================

def test_validation_of_negative_frame_indices():
    """Test 3: Negative or non-integer frame indices raise ValueError."""
    with pytest.raises(ValueError, match="frame_idx must be a non-negative integer"):
        GroundTruthPoint(gt_id="P1", frame_idx=-1, timestamp_sec=0.0)

    with pytest.raises(ValueError, match="frame_idx must be a non-negative integer"):
        GroundTruthPoint(gt_id="P1", frame_idx=True, timestamp_sec=0.0)  # bool is subclass of int

    with pytest.raises(ValueError, match="frame_idx must be a non-negative integer"):
        GroundTruthEvent(event_type="entry", gt_id="P1", frame_idx=-5, timestamp_sec=0.0, to_zone_id="Z1")


# ==============================================================================
# 4. Validation of Invalid Timestamps or FPS Values
# ==============================================================================

def test_validation_of_invalid_timestamps_or_fps():
    """Test 4: Negative, NaN, infinite timestamps, and non-positive/infinite FPS raise ValueError."""
    # Negative timestamp
    with pytest.raises(ValueError, match="timestamp_sec must be a non-negative finite number"):
        GroundTruthPoint(gt_id="P1", frame_idx=0, timestamp_sec=-0.5)

    # Infinite and NaN timestamps
    with pytest.raises(ValueError, match="timestamp_sec must be a non-negative finite number"):
        GroundTruthPoint(gt_id="P1", frame_idx=0, timestamp_sec=float("inf"))

    with pytest.raises(ValueError, match="timestamp_sec must be a non-negative finite number"):
        GroundTruthPoint(gt_id="P1", frame_idx=0, timestamp_sec=float("nan"))

    # FPS zero or negative
    with pytest.raises(ValueError, match="fps must be a positive finite number"):
        GroundTruthSequence(schema_version="1.0", sequence_id="S1", source="test", fps=0.0)

    with pytest.raises(ValueError, match="fps must be a positive finite number"):
        GroundTruthSequence(schema_version="1.0", sequence_id="S1", source="test", fps=-5.0)

    with pytest.raises(ValueError, match="fps must be a positive finite number"):
        GroundTruthSequence(schema_version="1.0", sequence_id="S1", source="test", fps=float("inf"))


# ==============================================================================
# 5. Validation of Malformed Bounding Boxes
# ==============================================================================

def test_validation_of_malformed_bounding_boxes():
    """Test 5: Bounding boxes with invalid lengths, non-finite values, or wrong types raise ValueError."""
    # Wrong length
    with pytest.raises(ValueError, match="must contain exactly four values"):
        GroundTruthPoint(gt_id="P1", frame_idx=0, timestamp_sec=0.0, image_bbox=[10.0, 20.0, 30.0])

    with pytest.raises(ValueError, match="must contain exactly four values"):
        GroundTruthPoint(gt_id="P1", frame_idx=0, timestamp_sec=0.0, image_bbox=[10.0, 20.0, 30.0, 40.0, 50.0])

    # Non-finite values
    with pytest.raises(ValueError, match="must be a finite number"):
        GroundTruthPoint(gt_id="P1", frame_idx=0, timestamp_sec=0.0, image_bbox=[10.0, float("nan"), 30.0, 40.0])

    with pytest.raises(ValueError, match="must be a finite number"):
        GroundTruthPoint(gt_id="P1", frame_idx=0, timestamp_sec=0.0, image_bbox=[10.0, 20.0, float("inf"), 40.0])


# ==============================================================================
# 6. Validation of Incomplete Image or Floor Coordinate Pairs
# ==============================================================================

def test_validation_of_incomplete_coordinate_pairs():
    """Test 6: Incomplete coordinate pairs (one None and one present) or non-finite values raise ValueError."""
    # Image coordinates: one present, one absent
    with pytest.raises(ValueError, match="image_x and image_y must either both be present or both be absent"):
        GroundTruthPoint(gt_id="P1", frame_idx=0, timestamp_sec=0.0, image_x=10.0, image_y=None)

    with pytest.raises(ValueError, match="image_x and image_y must either both be present or both be absent"):
        GroundTruthPoint(gt_id="P1", frame_idx=0, timestamp_sec=0.0, image_x=None, image_y=20.0)

    # Floor coordinates: one present, one absent
    with pytest.raises(ValueError, match="floor_x and floor_y must either both be present or both be absent"):
        GroundTruthPoint(gt_id="P1", frame_idx=0, timestamp_sec=0.0, floor_x=1.0, floor_y=None)

    with pytest.raises(ValueError, match="floor_x and floor_y must either both be present or both be absent"):
        GroundTruthPoint(gt_id="P1", frame_idx=0, timestamp_sec=0.0, floor_x=None, floor_y=2.0)

    # Non-finite coordinates
    with pytest.raises(ValueError, match="image_x and image_y must be finite numbers"):
        GroundTruthPoint(gt_id="P1", frame_idx=0, timestamp_sec=0.0, image_x=float("nan"), image_y=20.0)

    with pytest.raises(ValueError, match="floor_x and floor_y must be finite numbers"):
        GroundTruthPoint(gt_id="P1", frame_idx=0, timestamp_sec=0.0, floor_x=1.0, floor_y=float("inf"))


# ==============================================================================
# 7. Validation of Invalid Calibration-Point Roles
# ==============================================================================

def test_validation_of_invalid_calibration_point_roles():
    """Test 7: Roles other than 'training' or 'held_out' raise ValueError."""
    # Valid roles
    cp_train = GroundTruthCalibrationPoint("P1", 10.0, 20.0, 1.0, 2.0, role="training")
    assert cp_train.role == "training"

    cp_held = GroundTruthCalibrationPoint("CP1", 10.0, 20.0, 1.0, 2.0, role="held_out")
    assert cp_held.role == "held_out"

    # Invalid roles
    with pytest.raises(ValueError, match="role must be one of"):
        GroundTruthCalibrationPoint("P1", 10.0, 20.0, 1.0, 2.0, role="validation")

    with pytest.raises(ValueError, match="role must be one of"):
        GroundTruthCalibrationPoint("P1", 10.0, 20.0, 1.0, 2.0, role="test")


# ==============================================================================
# 8. JSON and Dictionary Round-Trip
# ==============================================================================

def test_json_and_dictionary_round_trip(synthetic_sequence):
    """Test 8: Full GroundTruthSequence correctly round-trips via to_dict/from_dict and to_json/from_json."""
    seq = synthetic_sequence

    # Dictionary round-trip
    d = seq.to_dict()
    seq_from_dict = GroundTruthSequence.from_dict(d)
    assert seq_from_dict.to_dict() == d

    # JSON round-trip
    json_str = seq.to_json()
    seq_from_json = GroundTruthSequence.from_json(json_str)
    assert seq_from_json.to_dict() == d

    # Verify key properties preserved
    assert seq_from_json.sequence_id == seq.sequence_id
    assert seq_from_json.fps == seq.fps
    assert len(seq_from_json.points) == len(seq.points)
    assert len(seq_from_json.events) == len(seq.events)
    assert len(seq_from_json.calibration_points) == len(seq.calibration_points)


# ==============================================================================
# 9. Preservation of Optional Fields and None Values
# ==============================================================================

def test_preservation_of_optional_fields_and_none_values():
    """Test 9: Minimal points/events with None values preserve exact field states during serialization."""
    pt_minimal = GroundTruthPoint(
        gt_id="P_minimal",
        frame_idx=0,
        timestamp_sec=0.0,
        # all optional fields default to None or True
    )
    d_pt = pt_minimal.to_dict()
    assert d_pt["image_bbox"] is None
    assert d_pt["image_x"] is None
    assert d_pt["image_y"] is None
    assert d_pt["floor_x"] is None
    assert d_pt["floor_y"] is None
    assert d_pt["zone_id"] is None
    assert d_pt["visible"] is True

    reconstructed_pt = GroundTruthPoint.from_dict(d_pt)
    assert reconstructed_pt.image_bbox is None
    assert reconstructed_pt.zone_id is None

    ev_entry = GroundTruthEvent("entry", "P_minimal", 0, 0.0, from_zone_id=None, to_zone_id="Z1")
    d_ev = ev_entry.to_dict()
    assert d_ev["from_zone_id"] is None
    assert d_ev["to_zone_id"] == "Z1"

    reconstructed_ev = GroundTruthEvent.from_dict(d_ev)
    assert reconstructed_ev.from_zone_id is None


# ==============================================================================
# 10. Rejection of Duplicate and Inconsistent Records
# ==============================================================================

def test_rejection_of_duplicate_or_inconsistent_records():
    """Test 10: Sequences reject duplicate observations, temporal ordering violations, and duplicate calibration labels."""
    # 1. Duplicate observation for same gt_id at same frame
    pt1 = GroundTruthPoint("person_A", frame_idx=5, timestamp_sec=1.0)
    pt2 = GroundTruthPoint("person_A", frame_idx=5, timestamp_sec=1.0)  # Duplicate!
    with pytest.raises(ValueError, match="Duplicate observation detected"):
        GroundTruthSequence(schema_version="1.0", sequence_id="S1", source="test", fps=5.0, points=[pt1, pt2])

    # 2. Temporal frame ordering violation for same gt_id
    pt_f10 = GroundTruthPoint("person_A", frame_idx=10, timestamp_sec=2.0)
    pt_f05 = GroundTruthPoint("person_A", frame_idx=5, timestamp_sec=2.2)  # Earlier frame after later frame!
    with pytest.raises(ValueError, match="Temporal frame ordering violation"):
        GroundTruthSequence(schema_version="1.0", sequence_id="S1", source="test", fps=5.0, points=[pt_f10, pt_f05])

    # 3. Temporal timestamp ordering violation for same gt_id
    pt_t2 = GroundTruthPoint("person_A", frame_idx=5, timestamp_sec=2.0)
    pt_t1 = GroundTruthPoint("person_A", frame_idx=6, timestamp_sec=1.5)  # Earlier timestamp after later timestamp!
    with pytest.raises(ValueError, match="Temporal timestamp ordering violation"):
        GroundTruthSequence(schema_version="1.0", sequence_id="S1", source="test", fps=5.0, points=[pt_t2, pt_t1])

    # 4. Duplicate calibration point label
    cp1 = GroundTruthCalibrationPoint("P1", 0.0, 0.0, 0.0, 0.0, role="training")
    cp2 = GroundTruthCalibrationPoint("P1", 10.0, 10.0, 1.0, 1.0, role="held_out")  # Duplicate label!
    with pytest.raises(ValueError, match="Duplicate calibration point label detected"):
        GroundTruthSequence(schema_version="1.0", sequence_id="S1", source="test", fps=5.0, calibration_points=[cp1, cp2])


# ==============================================================================
# 11. Preservation of Training versus Held-Out Calibration Points
# ==============================================================================

def test_preservation_of_training_versus_held_out_calibration_points(synthetic_sequence):
    """Test 11: Schema strictly preserves training and held-out calibration roles."""
    seq = synthetic_sequence

    train_pts = seq.get_training_calibration_points()
    held_out_pts = seq.get_held_out_calibration_points()

    assert len(train_pts) == 4
    assert len(held_out_pts) == 2

    # Verify roles are preserved
    for pt in train_pts:
        assert pt.role == "training"
    for pt in held_out_pts:
        assert pt.role == "held_out"

    # Verify labels match definitions
    assert [p.label for p in train_pts] == ["P1", "P2", "P3", "P4"]
    assert [p.label for p in held_out_pts] == ["CP1", "CP2"]


# ==============================================================================
# 12. Construction of a Complete Synthetic Sequence
# ==============================================================================

def test_construction_of_complete_synthetic_sequence(synthetic_sequence, tmp_path):
    """Test 12: Complete synthetic sequence query helpers and JSON file persistence."""
    seq = synthetic_sequence

    # Query helpers
    assert seq.get_gt_ids() == ["person_A", "person_B"]

    pts_A = seq.get_points_for_gt_id("person_A")
    assert len(pts_A) == 4
    assert pts_A[0].zone_id is None
    assert pts_A[1].zone_id == "zone_entrance"

    events_A = seq.get_events_for_gt_id("person_A")
    assert len(events_A) == 2
    assert events_A[0].event_type == "entry"
    assert events_A[1].event_type == "transition"

    # File persistence round-trip
    file_path = str(tmp_path / "synthetic_ground_truth.json")
    seq.save_to_json(file_path)

    assert os.path.exists(file_path)
    loaded_seq = GroundTruthSequence.load_from_json(file_path)

    assert loaded_seq.sequence_id == seq.sequence_id
    assert loaded_seq.get_gt_ids() == ["person_A", "person_B"]
    assert len(loaded_seq.points) == len(seq.points)
    assert len(loaded_seq.events) == len(seq.events)
    assert len(loaded_seq.calibration_points) == len(seq.calibration_points)
