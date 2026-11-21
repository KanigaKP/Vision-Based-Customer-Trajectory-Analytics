"""
Tests for Human Annotation Workbook Completeness Validator and Converter
==========================================================================
Verifies:
1. Rejection of pending review frames with actionable error messages.
2. Acceptance of reviewed-empty frames (REVIEWED_NO_PERSONS).
3. Rejection of status inconsistencies (empty status with persons, or present status with empty list).
4. Multi-person per frame support and identity uniqueness validation.
5. Coordinate validation and faithful preservation of unavailable ground contact.
6. Timestamp-to-frame alignment checks and monotonicity enforcement.
7. Disjoint segment boundary protection preventing cross-window episode bridging.
8. Production gap semantics (unassigned observations break transitions).

NOTE: All tests use synthetic test fixtures. They do NOT represent completed human evaluation.
"""

import hashlib
import json
import os
import subprocess
import sys
from typing import Dict, Any

import pytest

from src.validation.annotation_validator import (
    IncompleteAnnotationError,
    AnnotationValidationError,
    WorkbookValidationReport,
    SegmentManifest,
    SamplingManifest,
    MERL_PILOT_MANIFEST,
    PRCV_SITE03_PILOT_MANIFEST,
    REGISTERED_PILOT_MANIFESTS,
    TRUSTED_DATASET_PROFILES,
    get_trusted_dataset_zones,
    validate_annotation_workbook,
    convert_workbook_to_ground_truth_sequences,
)
from src.validation.ground_truth import GroundTruthPoint, GroundTruthEvent, GroundTruthSequence


def _create_minimal_synthetic_workbook(**overrides: Any) -> Dict[str, Any]:
    """Helper to create a valid base synthetic workbook with 2 frames."""
    base = {
        "workbook_version": "1.0",
        "dataset_name": "synthetic_test_aisle",
        "clip_path": "data/synthetic/test_clip.mp4",
        "source_fps": 30.0,
        "sampling_step": 6,
        "target_fps": 5.0,
        "evaluation_category": "software_unit_fixture",
        "overall_review_status": "REVIEWED_CONSENSUS",
        "segments": [
            {
                "segment_id": "seg_01",
                "name": "Test Segment 1",
                "frame_range": [0, 6],
                "frame_count": 2,
                "time_range_sec": [0.0, 0.2],
            }
        ],
        "frames": [
            {
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_NO_PERSONS",
                "annotator_notes": "Aisle empty",
                "persons": [],
            },
            {
                "frame_idx": 6,
                "timestamp_sec": 0.2,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "annotator_notes": "One shopper entered",
                "persons": [
                    {
                        "annotator_track_label": "P1",
                        "image_bbox": [100.0, 150.0, 200.0, 450.0],
                        "ground_contact_px": [150.0, 450.0],
                        "zone_id": "shelf_a",
                        "uncertainty": "certain",
                    }
                ],
            },
        ],
    }
    base.update(overrides)
    return base


# ==============================================================================
# 1. PENDING REJECTION TESTS
# ==============================================================================

def test_pending_frame_rejected_by_validator():
    """Verify that any workbook containing PENDING_REVIEW frames is rejected."""
    wb = _create_minimal_synthetic_workbook()
    wb["frames"][0]["frame_status"] = "PENDING_REVIEW"

    report = validate_annotation_workbook(wb)
    assert not report.is_valid
    assert 0 in report.pending_frames
    assert len(report.errors) >= 1
    assert any("PENDING_REVIEW" in err for err in report.errors)


def test_pending_frame_converter_raises_incomplete_error():
    """Verify that convert_workbook_to_ground_truth_sequences raises IncompleteAnnotationError."""
    wb = _create_minimal_synthetic_workbook()
    wb["frames"][1]["frame_status"] = "PENDING_REVIEW"

    with pytest.raises(IncompleteAnnotationError) as exc_info:
        convert_workbook_to_ground_truth_sequences(wb)

    assert "contains 1 pending frames" in str(exc_info.value)
    assert "First pending frame_idx: 6" in str(exc_info.value)


# ==============================================================================
# 2. REVIEWED-EMPTY ACCEPTANCE & INCONSISTENCY REJECTION TESTS
# ==============================================================================

def test_reviewed_empty_frame_accepted():
    """Verify that a completely empty reviewed frame is accepted and validated."""
    wb = _create_minimal_synthetic_workbook()
    # Both frames empty
    wb["frames"][1]["frame_status"] = "REVIEWED_NO_PERSONS"
    wb["frames"][1]["persons"] = []

    report = validate_annotation_workbook(wb)
    assert report.is_valid
    assert report.reviewed_empty_frames == 2
    assert report.reviewed_person_frames == 0
    assert report.total_person_observations == 0

    seqs = convert_workbook_to_ground_truth_sequences(wb)
    assert "seg_01" in seqs
    assert len(seqs["seg_01"].points) == 0
    assert len(seqs["seg_01"].events) == 0


def test_inconsistent_reviewed_empty_with_persons_rejected():
    """Verify rejection when frame is marked REVIEWED_NO_PERSONS but contains person observations."""
    wb = _create_minimal_synthetic_workbook()
    wb["frames"][0]["frame_status"] = "REVIEWED_NO_PERSONS"
    wb["frames"][0]["persons"] = [
        {
            "annotator_track_label": "P1",
            "image_bbox": [10.0, 10.0, 50.0, 50.0],
            "ground_contact_px": [30.0, 50.0],
            "zone_id": "shelf_a",
            "uncertainty": "certain",
        }
    ]

    report = validate_annotation_workbook(wb)
    assert not report.is_valid
    assert any("marked REVIEWED_NO_PERSONS but contains 1 person" in err for err in report.errors)


def test_inconsistent_reviewed_present_with_empty_persons_rejected():
    """Verify rejection when frame is marked REVIEWED_PERSONS_PRESENT but has empty persons list."""
    wb = _create_minimal_synthetic_workbook()
    wb["frames"][1]["frame_status"] = "REVIEWED_PERSONS_PRESENT"
    wb["frames"][1]["persons"] = []

    report = validate_annotation_workbook(wb)
    assert not report.is_valid
    assert any("marked REVIEWED_PERSONS_PRESENT but contains 0 person" in err for err in report.errors)


# ==============================================================================
# 3. MULTI-PERSON & IDENTITY UNIQUENESS TESTS
# ==============================================================================

def test_multiple_persons_per_frame_accepted():
    """Verify that multiple distinct persons in a single frame are accepted and converted."""
    wb = _create_minimal_synthetic_workbook()
    wb["frames"][1]["persons"] = [
        {
            "annotator_track_label": "P1",
            "image_bbox": [100.0, 150.0, 200.0, 450.0],
            "ground_contact_px": [150.0, 450.0],
            "zone_id": "shelf_a",
            "uncertainty": "certain",
        },
        {
            "annotator_track_label": "P2",
            "image_bbox": [300.0, 160.0, 400.0, 460.0],
            "ground_contact_px": [350.0, 460.0],
            "zone_id": "main_aisle",
            "uncertainty": "certain",
        },
    ]

    report = validate_annotation_workbook(wb)
    assert report.is_valid
    assert report.total_person_observations == 2

    seqs = convert_workbook_to_ground_truth_sequences(wb)
    seq = seqs["seg_01"]
    assert len(seq.points) == 2
    track_ids = {pt.gt_id for pt in seq.points}
    assert track_ids == {"P1", "P2"}


def test_duplicate_identities_in_same_frame_rejected():
    """Verify rejection when the same annotator_track_label appears multiple times in one frame."""
    wb = _create_minimal_synthetic_workbook()
    wb["frames"][1]["persons"] = [
        {
            "annotator_track_label": "P1",
            "image_bbox": [100.0, 150.0, 200.0, 450.0],
            "ground_contact_px": [150.0, 450.0],
            "zone_id": "shelf_a",
            "uncertainty": "certain",
        },
        {
            "annotator_track_label": "P1",  # Duplicate label in same frame!
            "image_bbox": [300.0, 160.0, 400.0, 460.0],
            "ground_contact_px": [350.0, 460.0],
            "zone_id": "main_aisle",
            "uncertainty": "certain",
        },
    ]

    report = validate_annotation_workbook(wb)
    assert not report.is_valid
    assert any("duplicate 'annotator_track_label' 'P1'" in err for err in report.errors)


# ==============================================================================
# 4. COORDINATE VALIDATION & PRESERVATION OF UNAVAILABLE GROUND CONTACT
# ==============================================================================

def test_invalid_bounding_box_rejected():
    """Verify rejection of inverted or non-finite bounding boxes."""
    wb = _create_minimal_synthetic_workbook()
    # x2 <= x1
    wb["frames"][1]["persons"][0]["image_bbox"] = [200.0, 150.0, 100.0, 450.0]

    report = validate_annotation_workbook(wb)
    assert not report.is_valid
    assert any("invalid image_bbox coordinates" in err for err in report.errors)


def test_unavailable_ground_contact_preserved_without_fabrication():
    """
    Verify that an occluded person with ground_contact_px=None is preserved faithfully
    with image_x=None, image_y=None rather than inventing fake coordinates.
    """
    wb = _create_minimal_synthetic_workbook()
    wb["frames"][1]["persons"][0]["ground_contact_px"] = None
    wb["frames"][1]["persons"][0]["uncertainty"] = "occluded"

    report = validate_annotation_workbook(wb)
    assert report.is_valid

    seqs = convert_workbook_to_ground_truth_sequences(wb)
    seq = seqs["seg_01"]
    assert len(seq.points) == 1
    pt = seq.points[0]
    assert pt.image_x is None
    assert pt.image_y is None
    assert pt.floor_x is None
    assert pt.floor_y is None
    assert pt.image_bbox == [100.0, 150.0, 200.0, 450.0]


# ==============================================================================
# 5. TIMESTAMP & FRAME ALIGNMENT TESTS
# ==============================================================================

def test_timestamp_divergence_rejected():
    """Verify rejection when timestamp_sec diverges from frame_idx / source_fps."""
    wb = _create_minimal_synthetic_workbook()
    # frame 6 at 30 fps should be 0.200s, not 0.850s
    wb["frames"][1]["timestamp_sec"] = 0.850

    report = validate_annotation_workbook(wb)
    assert not report.is_valid
    assert any("timestamp 0.85s deviates from expected 0.200s" in err for err in report.errors)


def test_non_monotonic_frame_index_rejected():
    """Verify rejection when frame indices are out of chronological order within a segment."""
    wb = _create_minimal_synthetic_workbook()
    wb["frames"][0]["frame_idx"] = 12
    wb["frames"][0]["timestamp_sec"] = 0.4
    wb["frames"][1]["frame_idx"] = 6
    wb["frames"][1]["timestamp_sec"] = 0.2

    report = validate_annotation_workbook(wb)
    assert not report.is_valid
    assert any("Non-monotonic frame sequence" in err for err in report.errors)


# ==============================================================================
# 6. DISJOINT SEGMENTS PREVENT CROSS-WINDOW BRIDGING
# ==============================================================================

def test_disjoint_segments_prevent_cross_window_bridging():
    """
    Verify that two disjoint segments (e.g. MERL Baseline frames 0..6 and Candidate frames 960..966)
    are converted into separate GroundTruthSequence instances and NEVER join episodes across the gap.
    """
    wb = {
        "workbook_version": "1.0",
        "dataset_name": "merl_diagnostic",
        "clip_path": "data/merl/10_1_crop.mp4",
        "source_fps": 30.0,
        "sampling_step": 6,
        "target_fps": 5.0,
        "evaluation_category": "development_diagnostic_evaluation",
        "overall_review_status": "REVIEWED_CONSENSUS",
        "segments": [
            {
                "segment_id": "merl_seg1_baseline",
                "name": "Segment 1: Baseline",
                "frame_range": [0, 6],
                "frame_count": 2,
                "time_range_sec": [0.0, 0.2],
            },
            {
                "segment_id": "merl_seg2_candidate",
                "name": "Segment 2: Candidate",
                "frame_range": [960, 966],
                "frame_count": 2,
                "time_range_sec": [32.0, 32.2],
            },
        ],
        "frames": [
            # Segment 1: Person P1 observed in shelf_a
            {
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "segment_id": "merl_seg1_baseline",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [
                    {
                        "annotator_track_label": "P1",
                        "image_bbox": [10.0, 10.0, 50.0, 50.0],
                        "ground_contact_px": [30.0, 50.0],
                        "zone_id": "shelf_a",
                        "uncertainty": "certain",
                    }
                ],
            },
            {
                "frame_idx": 6,
                "timestamp_sec": 0.2,
                "segment_id": "merl_seg1_baseline",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [
                    {
                        "annotator_track_label": "P1",
                        "image_bbox": [12.0, 10.0, 52.0, 50.0],
                        "ground_contact_px": [32.0, 50.0],
                        "zone_id": "shelf_a",
                        "uncertainty": "certain",
                    }
                ],
            },
            # Segment 2: Person P1 observed in shelf_a after 954 unreviewed frames
            {
                "frame_idx": 960,
                "timestamp_sec": 32.0,
                "segment_id": "merl_seg2_candidate",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [
                    {
                        "annotator_track_label": "P1",
                        "image_bbox": [20.0, 10.0, 60.0, 50.0],
                        "ground_contact_px": [40.0, 50.0],
                        "zone_id": "shelf_a",
                        "uncertainty": "certain",
                    }
                ],
            },
            {
                "frame_idx": 966,
                "timestamp_sec": 32.2,
                "segment_id": "merl_seg2_candidate",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [
                    {
                        "annotator_track_label": "P1",
                        "image_bbox": [22.0, 10.0, 62.0, 50.0],
                        "ground_contact_px": [42.0, 50.0],
                        "zone_id": "shelf_a",
                        "uncertainty": "certain",
                    }
                ],
            },
        ],
    }

    report = validate_annotation_workbook(wb)
    assert report.is_valid

    seqs = convert_workbook_to_ground_truth_sequences(wb)
    # Must yield two completely distinct sequences
    assert "merl_seg1_baseline" in seqs
    assert "merl_seg2_candidate" in seqs

    seq1 = seqs["merl_seg1_baseline"]
    seq2 = seqs["merl_seg2_candidate"]

    # Each sequence has 2 points
    assert len(seq1.points) == 2
    assert len(seq2.points) == 2

    # Verify that seq1 ends at frame 6 and seq2 starts at frame 960 without bridging
    assert seq1.points[-1].frame_idx == 6
    assert seq2.points[0].frame_idx == 960

    # No events span between frame 6 and frame 960
    assert all(ev.frame_idx <= 6 for ev in seq1.events)
    assert all(ev.frame_idx >= 960 for ev in seq2.events)


# ==============================================================================
# 7. PRODUCTION GAP & ZERO-BRIDGING SEMANTICS
# ==============================================================================

def test_production_zero_bridging_semantics():
    """
    Verify production zero-bridging semantics:
    An unassigned observation (zone_id = None) terminates active stay and prevents
    any direct transition to another zone (A -> None -> B yields ZERO direct transitions).
    """
    wb = {
        "workbook_version": "1.0",
        "dataset_name": "synthetic_transition_test",
        "clip_path": "data/synthetic/test.mp4",
        "source_fps": 30.0,
        "sampling_step": 6,
        "target_fps": 5.0,
        "overall_review_status": "REVIEWED_CONSENSUS",
        "frames": [
            # Frame 0: P1 in shelf_a
            {
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [{"annotator_track_label": "P1", "zone_id": "shelf_a"}],
            },
            # Frame 6: P1 in shelf_a
            {
                "frame_idx": 6,
                "timestamp_sec": 0.2,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [{"annotator_track_label": "P1", "zone_id": "shelf_a"}],
            },
            # Frame 12: P1 moves outside all retail zones (unassigned None)
            {
                "frame_idx": 12,
                "timestamp_sec": 0.4,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [{"annotator_track_label": "P1", "zone_id": None}],
            },
            # Frame 18: P1 enters shelf_b
            {
                "frame_idx": 18,
                "timestamp_sec": 0.6,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [{"annotator_track_label": "P1", "zone_id": "shelf_b"}],
            },
        ],
    }

    report = validate_annotation_workbook(wb)
    assert report.is_valid

    seqs = convert_workbook_to_ground_truth_sequences(wb)
    seq = seqs["seg_01"]

    # Verify events
    events = seq.events
    event_types = [e.event_type for e in events]

    # Should contain:
    # 1. Entry into shelf_a at frame 0
    # 2. Exit from shelf_a to None at frame 12
    # 3. Entry into shelf_b from None at frame 18
    # Must NOT contain a direct transition from shelf_a to shelf_b
    assert "transition" not in event_types
    assert event_types.count("entry") == 2
    assert event_types.count("exit") == 1

    exit_ev = next(e for e in events if e.event_type == "exit")
    assert exit_ev.from_zone_id == "shelf_a"
    assert exit_ev.to_zone_id is None
    assert exit_ev.frame_idx == 12

    entry_ev2 = [e for e in events if e.event_type == "entry"][1]
    assert entry_ev2.to_zone_id == "shelf_b"
    assert entry_ev2.from_zone_id is None
    assert entry_ev2.frame_idx == 18


def test_direct_transition_without_intervening_null():
    """Verify that a direct transition (A -> B without None) produces a transition event."""
    wb = {
        "workbook_version": "1.0",
        "dataset_name": "synthetic_transition_test",
        "clip_path": "data/synthetic/test.mp4",
        "source_fps": 30.0,
        "sampling_step": 6,
        "target_fps": 5.0,
        "overall_review_status": "REVIEWED_CONSENSUS",
        "frames": [
            # Frame 0: P1 in shelf_a
            {
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [{"annotator_track_label": "P1", "zone_id": "shelf_a"}],
            },
            # Frame 6: P1 steps directly into main_aisle
            {
                "frame_idx": 6,
                "timestamp_sec": 0.2,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [{"annotator_track_label": "P1", "zone_id": "main_aisle"}],
            },
        ],
    }

    report = validate_annotation_workbook(wb)
    assert report.is_valid

    seqs = convert_workbook_to_ground_truth_sequences(wb)
    seq = seqs["seg_01"]

    transitions = [e for e in seq.events if e.event_type == "transition"]
    assert len(transitions) == 1
    assert transitions[0].from_zone_id == "shelf_a"
    assert transitions[0].to_zone_id == "main_aisle"
    assert transitions[0].frame_idx == 6


# ==============================================================================
# 9. DEFECT REPAIR REGRESSION TESTS (DEFECTS 1 THROUGH 6)
# ==============================================================================

def test_defect_1_omitted_frames_rejected_against_manifest():
    """Defect 1: A 100-frame MERL workbook reduced to one reviewed frame must fail validation."""
    wb = {
        "workbook_version": "1.0",
        "dataset_name": "merl_shopping_aisle",
        "clip_path": "data/merl/10_1_crop.mp4",
        "source_fps": 30.0,
        "sampling_step": 6,
        "target_fps": 5.0,
        "overall_review_status": "REVIEWED_CONSENSUS",
        "segments": [
            {
                "segment_id": "merl_seg1_baseline",
                "name": "Segment 1: Negative Baseline",
                "frame_range": [0, 294],
                "sampling_step": 6,
            },
            {
                "segment_id": "merl_seg2_shelf_a_candidate",
                "name": "Segment 2: Candidate Zone shelf_a",
                "frame_range": [960, 1254],
                "sampling_step": 6,
            },
        ],
        # Intentionally reduced to only ONE frame!
        "frames": [
            {
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "segment_id": "merl_seg1_baseline",
                "frame_status": "REVIEWED_NO_PERSONS",
                "persons": [],
            }
        ],
    }

    # 1. Validation with explicit MERL_PILOT_MANIFEST
    report1 = validate_annotation_workbook(wb, manifest=MERL_PILOT_MANIFEST)
    assert not report1.is_valid
    assert any("Omitted expected frames" in err for err in report1.errors)
    assert any("merl_seg1_baseline" in err for err in report1.errors)
    assert any("merl_seg2_shelf_a_candidate" in err for err in report1.errors)

    # 2. Validation without explicit manifest (derived from declared segments)
    report2 = validate_annotation_workbook(wb)
    assert not report2.is_valid
    assert any("Omitted expected frames" in err for err in report2.errors)

    # 3. Conversion must raise AnnotationValidationError
    with pytest.raises(AnnotationValidationError) as exc_info:
        convert_workbook_to_ground_truth_sequences(wb, manifest=MERL_PILOT_MANIFEST)
    assert "Omitted expected frames" in str(exc_info.value)


def test_defect_2_undeclared_segment_id_rejected():
    """Defect 2: An undeclared segment_id passes in old implementation, now must be rejected."""
    wb = _create_minimal_synthetic_workbook()
    # Frame 6 declares an unknown segment_id
    wb["frames"][1]["segment_id"] = "undeclared_seg_xyz"

    report = validate_annotation_workbook(wb)
    assert not report.is_valid
    assert any("undeclared segment_id 'undeclared_seg_xyz'" in err for err in report.errors)
    assert any("Declared segments" in err for err in report.errors)


def test_defect_3_off_grid_frame_rejected():
    """Defect 3: An off-grid frame passes even with expected_sampling_step=6, now rejected."""
    wb = _create_minimal_synthetic_workbook()
    # Frame 5 is off-grid for step 6 (origin 0)
    wb["frames"][1]["frame_idx"] = 5
    wb["frames"][1]["timestamp_sec"] = round(5 / 30.0, 3)

    report = validate_annotation_workbook(wb, expected_sampling_step=6)
    assert not report.is_valid
    assert any("off-grid for sampling step 6" in err for err in report.errors)


def test_defect_4_malformed_timestamp_reports_error_without_typeerror():
    """Defect 4: Malformed timestamp raises TypeError instead of validation error."""
    # Test string timestamp
    wb_str = _create_minimal_synthetic_workbook()
    wb_str["frames"][1]["timestamp_sec"] = "0.200"

    report_str = validate_annotation_workbook(wb_str)
    assert not report_str.is_valid
    assert any("invalid 'timestamp_sec': '0.200'" in err for err in report_str.errors)

    # Test None timestamp
    wb_none = _create_minimal_synthetic_workbook()
    wb_none["frames"][1]["timestamp_sec"] = None

    report_none = validate_annotation_workbook(wb_none)
    assert not report_none.is_valid
    assert any("invalid 'timestamp_sec': None" in err for err in report_none.errors)

    # Test dict timestamp
    wb_dict = _create_minimal_synthetic_workbook()
    wb_dict["frames"][1]["timestamp_sec"] = {"time": 0.2}

    report_dict = validate_annotation_workbook(wb_dict)
    assert not report_dict.is_valid
    assert any("invalid 'timestamp_sec'" in err for err in report_dict.errors)

    # Test malformed frame_idx (e.g. dict or string or negative)
    wb_fidx = _create_minimal_synthetic_workbook()
    wb_fidx["frames"][0]["frame_idx"] = {"frame": 0}

    report_fidx = validate_annotation_workbook(wb_fidx)
    assert not report_fidx.is_valid
    assert any("invalid 'frame_idx'" in err for err in report_fidx.errors)


def test_defect_5_conversion_preserves_empty_frame_identities():
    """Defect 5: Conversion loses exact reviewed-empty frame identities."""
    wb = _create_minimal_synthetic_workbook()
    # Frame 0 is REVIEWED_NO_PERSONS, Frame 6 is REVIEWED_PERSONS_PRESENT
    wb["frames"][0]["frame_status"] = "REVIEWED_NO_PERSONS"
    wb["frames"][0]["persons"] = []

    seqs = convert_workbook_to_ground_truth_sequences(wb)
    seq = seqs["seg_01"]

    # 1. Query methods on sequence
    assert seq.get_empty_frame_indices() == [0]
    assert seq.is_frame_empty(0) is True
    assert seq.is_frame_empty(6) is False
    assert seq.is_frame_empty(12) is False

    assert seq.get_evaluated_frame_indices() == [0, 6]
    assert seq.is_frame_evaluated(0) is True
    assert seq.is_frame_evaluated(6) is True
    assert seq.is_frame_evaluated(12) is False

    # 2. Metadata presence
    assert seq.metadata is not None
    assert seq.metadata["empty_frame_indices"] == [0]
    assert seq.metadata["evaluated_frame_indices"] == [0, 6]
    assert len(seq.metadata["reviewed_frames"]) == 2

    # 3. Serialization round-trip (dict)
    seq_dict = seq.to_dict()
    seq_restored = GroundTruthSequence.from_dict(seq_dict)
    assert seq_restored.get_empty_frame_indices() == [0]
    assert seq_restored.is_frame_empty(0) is True
    assert seq_restored.is_frame_evaluated(0) is True

    # 4. Serialization round-trip (JSON string)
    json_str = seq.to_json()
    seq_json = GroundTruthSequence.from_json(json_str)
    assert seq_json.get_empty_frame_indices() == [0]
    assert seq_json.is_frame_empty(0) is True


def test_defect_6_conversion_and_serialization_preserves_uncertainty():
    """Defect 6: Conversion discards annotation uncertainty."""
    wb = {
        "workbook_version": "1.0",
        "dataset_name": "synthetic_uncertainty_test",
        "clip_path": "data/synthetic/test.mp4",
        "source_fps": 30.0,
        "sampling_step": 6,
        "target_fps": 5.0,
        "overall_review_status": "REVIEWED_CONSENSUS",
        "segments": [
            {
                "segment_id": "seg_01",
                "name": "Test Segment 1",
                "frame_range": [0, 18],
                "sampling_step": 6,
            }
        ],
        "frames": [
            {
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [
                    {
                        "annotator_track_label": "P1",
                        "image_bbox": [10.0, 20.0, 50.0, 100.0],
                        "ground_contact_px": [30.0, 100.0],
                        "uncertainty": "certain",
                    }
                ],
            },
            {
                "frame_idx": 6,
                "timestamp_sec": 0.2,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [
                    {
                        "annotator_track_label": "P1",
                        "image_bbox": [15.0, 20.0, 55.0, 100.0],
                        "ground_contact_px": None,
                        "uncertainty": "occluded",
                    }
                ],
            },
            {
                "frame_idx": 12,
                "timestamp_sec": 0.4,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [
                    {
                        "annotator_track_label": "P1",
                        "image_bbox": [0.0, 20.0, 30.0, 100.0],
                        "ground_contact_px": None,
                        "uncertainty": "truncated",
                    }
                ],
            },
            {
                "frame_idx": 18,
                "timestamp_sec": 0.6,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [
                    {
                        "annotator_track_label": "P2",
                        "image_bbox": [20.0, 20.0, 60.0, 100.0],
                        "ground_contact_px": [40.0, 100.0],
                        "uncertainty": "ambiguous_identity",
                    }
                ],
            },
        ],
    }

    seqs = convert_workbook_to_ground_truth_sequences(wb)
    seq = seqs["seg_01"]

    # Verify points retain exact uncertainty
    assert len(seq.points) == 4
    assert seq.points[0].uncertainty == "certain"
    assert seq.points[1].uncertainty == "occluded"
    assert seq.points[2].uncertainty == "truncated"
    assert seq.points[3].uncertainty == "ambiguous_identity"

    # Verify round-trip serialization preserves uncertainty
    seq_dict = seq.to_dict()
    assert seq_dict["points"][1]["uncertainty"] == "occluded"

    seq_restored = GroundTruthSequence.from_dict(seq_dict)
    assert seq_restored.points[1].uncertainty == "occluded"

    json_str = seq.to_json()
    seq_from_json = GroundTruthSequence.from_json(json_str)
    assert seq_from_json.points[2].uncertainty == "truncated"
    assert seq_from_json.points[3].uncertainty == "ambiguous_identity"


# ==============================================================================
# 10. CLOSURE REGRESSION TESTS: INDEPENDENT CONTRACT & ABSENCE-AWARE DERIVATION
# ==============================================================================

def test_truncated_frames_plus_truncated_segment_definition_rejected():
    """Verify that a MERL submission reduced to 1 frame and segment reduced to [0,0] is rejected."""
    wb = {
        "workbook_version": "1.0",
        "dataset_name": "merl_shopping_aisle",
        "clip_path": "data/merl/10_1_crop.mp4",
        "source_fps": 30.0,
        "sampling_step": 6,
        "target_fps": 5.0,
        "overall_review_status": "REVIEWED_CONSENSUS",
        # Submitter attempted to cheat manifest by reducing segment definition to [0, 0]
        "segments": [
            {
                "segment_id": "merl_seg1_baseline",
                "name": "Truncated Segment 1",
                "frame_range": [0, 0],
                "sampling_step": 6,
            }
        ],
        "frames": [
            {
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "segment_id": "merl_seg1_baseline",
                "frame_status": "REVIEWED_NO_PERSONS",
                "persons": [],
            }
        ],
    }

    # Validation must enforce trusted registered manifest MERL_PILOT_MANIFEST by default
    report = validate_annotation_workbook(wb)
    assert not report.is_valid
    assert not report.coverage_certified
    assert report.coverage_status == "failed"
    assert report.trusted_manifest_used == "merl_shopping_aisle"

    # Must catch submitted segment definition mismatch against trusted manifest
    assert any("frame_range [0, 0] does not match trusted manifest range [0, 294]" in e for e in report.errors)
    # Must catch missing second required segment
    assert any("missing required segment 'merl_seg2_shelf_a_candidate'" in e for e in report.errors)
    # Must catch omitted frames
    assert any("Omitted expected frames in segment 'merl_seg1_baseline'" in e for e in report.errors)

    with pytest.raises(AnnotationValidationError):
        convert_workbook_to_ground_truth_sequences(wb)


def test_recognized_pilot_default_validation_and_conversion_paths():
    """Verify default validation and conversion path for recognized pilot dataset (MERL)."""
    # Create complete synthetic 100-frame MERL workbook matching MERL_PILOT_MANIFEST
    frames = []
    # Segment 1: 0..294 step 6 (50 frames)
    for f_idx in range(0, 295, 6):
        frames.append({
            "frame_idx": f_idx,
            "timestamp_sec": round(f_idx / 30.0, 3),
            "segment_id": "merl_seg1_baseline",
            "frame_status": "REVIEWED_NO_PERSONS",
            "persons": [],
        })
    # Segment 2: 960..1254 step 6 (50 frames)
    for f_idx in range(960, 1255, 6):
        frames.append({
            "frame_idx": f_idx,
            "timestamp_sec": round(f_idx / 30.0, 3),
            "segment_id": "merl_seg2_shelf_a_candidate",
            "frame_status": "REVIEWED_NO_PERSONS",
            "persons": [],
        })

    wb = {
        "workbook_version": "1.0",
        "dataset_name": "merl_shopping_aisle",
        "clip_path": "data/merl/10_1_crop.mp4",
        "source_fps": 30.0,
        "sampling_step": 6,
        "target_fps": 5.0,
        "overall_review_status": "REVIEWED_CONSENSUS",
        "segments": [
            {
                "segment_id": "merl_seg1_baseline",
                "name": "Segment 1: Negative Baseline & Initial Aisle Traversal",
                "frame_range": [0, 294],
                "sampling_step": 6,
            },
            {
                "segment_id": "merl_seg2_shelf_a_candidate",
                "name": "Segment 2: Candidate Zone shelf_a Approach, Dwell & Depart",
                "frame_range": [960, 1254],
                "sampling_step": 6,
            },
        ],
        "frames": frames,
    }

    # Validated with NO explicit manifest argument: must resolve registered manifest
    report = validate_annotation_workbook(wb)
    assert report.is_valid
    assert report.schema_valid
    assert report.coverage_certified
    assert report.coverage_status == "certified"
    assert report.trusted_manifest_used == "merl_shopping_aisle"
    assert report.total_frames == 100
    assert report.reviewed_empty_frames == 100

    # Converted with NO explicit manifest argument
    seqs = convert_workbook_to_ground_truth_sequences(wb)
    assert len(seqs) == 2
    assert "merl_seg1_baseline" in seqs
    assert "merl_seg2_shelf_a_candidate" in seqs

    seq1 = seqs["merl_seg1_baseline"]
    assert seq1.metadata["coverage_certified"] is True
    assert seq1.metadata["coverage_status"] == "certified"
    assert seq1.metadata["trusted_manifest_used"] == "merl_shopping_aisle"
    assert len(seq1.get_empty_frame_indices()) == 50


def test_unknown_dataset_without_trusted_coverage_manifest():
    """Verify that unknown datasets distinguish schema validation from coverage certification."""
    wb_unknown = _create_minimal_synthetic_workbook(dataset_name="custom_unregistered_store")

    # 1. Standard validation without manifest: schema passes, but coverage is uncertified
    report = validate_annotation_workbook(wb_unknown)
    assert report.schema_valid is True
    assert report.coverage_certified is False
    assert report.coverage_status == "uncertified"
    assert report.trusted_manifest_used is None
    assert any("Coverage not certified" in w for w in report.warnings)

    # 2. Strict completeness certification request without manifest: fails validation
    report_strict = validate_annotation_workbook(wb_unknown, require_trusted_manifest=True)
    assert report_strict.is_valid is False
    assert report_strict.coverage_certified is False
    assert report_strict.coverage_status == "failed"
    assert any("Cannot certify completeness" in e for e in report_strict.errors)

    # Conversion with require_trusted_manifest=True raises AnnotationValidationError
    with pytest.raises(AnnotationValidationError) as exc:
        convert_workbook_to_ground_truth_sequences(wb_unknown, require_trusted_manifest=True)
    assert "Completeness certification failed" in str(exc.value)

    # 3. Supplying an explicit trusted manifest certifies coverage
    trusted_manifest = SamplingManifest(
        dataset_name="custom_unregistered_store",
        source_fps=30.0,
        segments=[
            SegmentManifest(
                segment_id="seg_01",
                start_frame=0,
                end_frame=6,
                sampling_step=6,
            )
        ],
    )
    report_cert = validate_annotation_workbook(wb_unknown, manifest=trusted_manifest)
    assert report_cert.is_valid is True
    assert report_cert.coverage_certified is True
    assert report_cert.coverage_status == "certified"
    assert report_cert.trusted_manifest_used == "custom_unregistered_store"


def test_a_reviewed_empty_b_produces_no_direct_transition():
    """
    Reproduction:
    Frame 0: P1 in zone A
    Frame 6: REVIEWED_NO_PERSONS
    Frame 12: P1 in zone B
    Current conversion emitted entry A followed by direct A->B.
    Must NOT emit that direct transition.
    """
    wb = {
        "workbook_version": "1.0",
        "dataset_name": "synthetic_empty_gap_test",
        "clip_path": "data/synthetic/test.mp4",
        "source_fps": 30.0,
        "sampling_step": 6,
        "target_fps": 5.0,
        "overall_review_status": "REVIEWED_CONSENSUS",
        "segments": [
            {
                "segment_id": "seg_01",
                "name": "Test Segment",
                "frame_range": [0, 12],
                "sampling_step": 6,
            }
        ],
        "frames": [
            # Frame 0: P1 in shelf_a
            {
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [
                    {
                        "annotator_track_label": "P1",
                        "image_bbox": [10.0, 20.0, 50.0, 100.0],
                        "ground_contact_px": [30.0, 100.0],
                        "zone_id": "shelf_a",
                    }
                ],
            },
            # Frame 6: Explicitly reviewed as empty (0 persons)
            {
                "frame_idx": 6,
                "timestamp_sec": 0.2,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_NO_PERSONS",
                "persons": [],
            },
            # Frame 12: P1 appears in shelf_b
            {
                "frame_idx": 12,
                "timestamp_sec": 0.4,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [
                    {
                        "annotator_track_label": "P1",
                        "image_bbox": [200.0, 20.0, 250.0, 100.0],
                        "ground_contact_px": [225.0, 100.0],
                        "zone_id": "shelf_b",
                    }
                ],
            },
        ],
    }

    report = validate_annotation_workbook(wb)
    assert report.is_valid

    seqs = convert_workbook_to_ground_truth_sequences(wb)
    seq = seqs["seg_01"]

    # 1. Exactly TWO points (observations); NO synthetic null-zone person point at frame 6!
    assert len(seq.points) == 2
    assert [p.frame_idx for p in seq.points] == [0, 12]

    # 2. NO direct transition shelf_a -> shelf_b emitted!
    transitions = [e for e in seq.events if e.event_type == "transition"]
    assert len(transitions) == 0

    # 3. TWO separate entries: entry into shelf_a at frame 0, and entry into shelf_b at frame 12
    entries = [e for e in seq.events if e.event_type == "entry"]
    assert len(entries) == 2
    assert entries[0].to_zone_id == "shelf_a"
    assert entries[0].frame_idx == 0
    assert entries[0].from_zone_id is None

    assert entries[1].to_zone_id == "shelf_b"
    assert entries[1].frame_idx == 12
    assert entries[1].from_zone_id is None


def test_a_reviewed_empty_a_does_not_imply_uninterrupted_occupancy():
    """
    Reproduction:
    Frame 0: P1 in zone A
    Frame 6: REVIEWED_NO_PERSONS
    Frame 12: P1 in zone A
    Must not imply uninterrupted occupancy (must emit two separate entries, not a continuous stay).
    """
    wb = {
        "workbook_version": "1.0",
        "dataset_name": "synthetic_empty_revisit_test",
        "clip_path": "data/synthetic/test.mp4",
        "source_fps": 30.0,
        "sampling_step": 6,
        "target_fps": 5.0,
        "overall_review_status": "REVIEWED_CONSENSUS",
        "segments": [
            {
                "segment_id": "seg_01",
                "name": "Test Segment",
                "frame_range": [0, 12],
                "sampling_step": 6,
            }
        ],
        "frames": [
            # Frame 0: P1 in shelf_a
            {
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [{"annotator_track_label": "P1", "zone_id": "shelf_a"}],
            },
            # Frame 6: Explicitly reviewed empty
            {
                "frame_idx": 6,
                "timestamp_sec": 0.2,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_NO_PERSONS",
                "persons": [],
            },
            # Frame 12: P1 re-appears in shelf_a
            {
                "frame_idx": 12,
                "timestamp_sec": 0.4,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [{"annotator_track_label": "P1", "zone_id": "shelf_a"}],
            },
        ],
    }

    report = validate_annotation_workbook(wb)
    assert report.is_valid

    seqs = convert_workbook_to_ground_truth_sequences(wb)
    seq = seqs["seg_01"]

    # Exactly 2 observations; no synthetic points
    assert len(seq.points) == 2

    # Two separate entry events into shelf_a (at frame 0 and frame 12)
    entries = [e for e in seq.events if e.event_type == "entry"]
    assert len(entries) == 2
    assert entries[0].frame_idx == 0 and entries[0].to_zone_id == "shelf_a"
    assert entries[1].frame_idx == 12 and entries[1].to_zone_id == "shelf_a"

    # Zero transitions
    assert len([e for e in seq.events if e.event_type == "transition"]) == 0


def test_existing_missing_observation_and_explicit_null_behavior_intact():
    """Verify that genuine missing-observation and explicit-null behavior remains intact."""
    # 1. Genuine missing observation: P1 is missing at frame 6 (P2 observed in shelf_c),
    # but the scene was NOT reviewed empty. P1's stay in shelf_a transitions to shelf_b.
    wb_missing = {
        "workbook_version": "1.0",
        "dataset_name": "synthetic_missing_test",
        "clip_path": "data/synthetic/test.mp4",
        "source_fps": 30.0,
        "sampling_step": 6,
        "target_fps": 5.0,
        "overall_review_status": "REVIEWED_CONSENSUS",
        "segments": [{"segment_id": "seg_01", "frame_range": [0, 12], "sampling_step": 6}],
        "frames": [
            {
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [{"annotator_track_label": "P1", "zone_id": "shelf_a"}],
            },
            {
                "frame_idx": 6,
                "timestamp_sec": 0.2,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [{"annotator_track_label": "P2", "zone_id": "shelf_c"}],
            },
            {
                "frame_idx": 12,
                "timestamp_sec": 0.4,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [{"annotator_track_label": "P1", "zone_id": "shelf_b"}],
            },
        ],
    }

    seq_missing = convert_workbook_to_ground_truth_sequences(wb_missing)["seg_01"]
    p1_transitions = [e for e in seq_missing.events if e.gt_id == "P1" and e.event_type == "transition"]
    assert len(p1_transitions) == 1
    assert p1_transitions[0].from_zone_id == "shelf_a"
    assert p1_transitions[0].to_zone_id == "shelf_b"

    # 2. Explicit null observation (P1 observed outside retail zones at frame 6):
    # Produces exit from shelf_a at frame 6, and entry to shelf_b at frame 12 (0 direct transitions).
    wb_null = {
        "workbook_version": "1.0",
        "dataset_name": "synthetic_explicit_null_test",
        "clip_path": "data/synthetic/test.mp4",
        "source_fps": 30.0,
        "sampling_step": 6,
        "target_fps": 5.0,
        "overall_review_status": "REVIEWED_CONSENSUS",
        "segments": [{"segment_id": "seg_01", "frame_range": [0, 12], "sampling_step": 6}],
        "frames": [
            {
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [{"annotator_track_label": "P1", "zone_id": "shelf_a"}],
            },
            {
                "frame_idx": 6,
                "timestamp_sec": 0.2,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [{"annotator_track_label": "P1", "zone_id": None}],
            },
            {
                "frame_idx": 12,
                "timestamp_sec": 0.4,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [{"annotator_track_label": "P1", "zone_id": "shelf_b"}],
            },
        ],
    }

    seq_null = convert_workbook_to_ground_truth_sequences(wb_null)["seg_01"]
    p1_events = [e for e in seq_null.events if e.gt_id == "P1"]
    assert not any(e.event_type == "transition" for e in p1_events)
    assert any(e.event_type == "exit" and e.from_zone_id == "shelf_a" and e.to_zone_id is None for e in p1_events)
    assert any(e.event_type == "entry" and e.from_zone_id is None and e.to_zone_id == "shelf_b" for e in p1_events)


def test_empty_frame_evidence_survives_serialization():
    """Verify that reviewed-empty frame identities survive to_dict, from_dict, to_json, from_json round-trips."""
    wb = {
        "workbook_version": "1.0",
        "dataset_name": "synthetic_empty_serial_test",
        "clip_path": "data/synthetic/test.mp4",
        "source_fps": 30.0,
        "sampling_step": 6,
        "target_fps": 5.0,
        "overall_review_status": "REVIEWED_CONSENSUS",
        "segments": [{"segment_id": "seg_01", "frame_range": [0, 12], "sampling_step": 6}],
        "frames": [
            {
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [{"annotator_track_label": "P1", "zone_id": "shelf_a"}],
            },
            {
                "frame_idx": 6,
                "timestamp_sec": 0.2,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_NO_PERSONS",
                "persons": [],
            },
            {
                "frame_idx": 12,
                "timestamp_sec": 0.4,
                "segment_id": "seg_01",
                "frame_status": "REVIEWED_PERSONS_PRESENT",
                "persons": [{"annotator_track_label": "P1", "zone_id": "shelf_b"}],
            },
        ],
    }

    seq = convert_workbook_to_ground_truth_sequences(wb)["seg_01"]

    assert seq.get_empty_frame_indices() == [6]
    assert seq.is_frame_empty(6) is True
    assert seq.is_frame_empty(0) is False
    assert seq.get_evaluated_frame_indices() == [0, 6, 12]

    # Round-trip via dict
    seq_dict = seq.to_dict()
    restored_dict = GroundTruthSequence.from_dict(seq_dict)
    assert restored_dict.get_empty_frame_indices() == [6]
    assert restored_dict.is_frame_empty(6) is True
    assert restored_dict.is_frame_empty(0) is False
    assert restored_dict.get_evaluated_frame_indices() == [0, 6, 12]

    # Round-trip via JSON
    json_str = seq.to_json()
    restored_json = GroundTruthSequence.from_json(json_str)
    assert restored_json.get_empty_frame_indices() == [6]
    assert restored_json.is_frame_empty(6) is True
    assert restored_json.is_frame_empty(0) is False
    assert restored_json.get_evaluated_frame_indices() == [0, 6, 12]


def test_site03_rejects_invalid_zone_names_in_synthetic_fixture(tmp_path):
    """
    Focused test: verify that Site 03 submission rejects invalid zone names
    (such as unverified 'apparel_rack_east', 'fitting_room_entry', or 'random_zone')
    against its authoritative scene profile.
    Uses clearly labelled synthetic fixture.
    """
    frames = []
    for f_idx in range(0, 298, 3):
        t = round(f_idx / 15.0, 3)
        if f_idx == 0:
            persons = [{
                "annotator_track_label": "P1",
                "image_bbox": [100.0, 100.0, 200.0, 300.0],
                "ground_contact_px": [150.0, 300.0],
                "zone_id": "apparel_rack_east",  # INVALID for Site 03
                "uncertainty": "certain",
            }]
            status = "REVIEWED_PERSONS_PRESENT"
        elif f_idx == 3:
            persons = [{
                "annotator_track_label": "P1",
                "image_bbox": [105.0, 100.0, 205.0, 300.0],
                "ground_contact_px": [155.0, 300.0],
                "zone_id": "fitting_room_entry",  # INVALID for Site 03
                "uncertainty": "certain",
            }]
            status = "REVIEWED_PERSONS_PRESENT"
        elif f_idx == 6:
            persons = [{
                "annotator_track_label": "P2",
                "image_bbox": [300.0, 200.0, 400.0, 450.0],
                "ground_contact_px": [350.0, 450.0],
                "zone_id": "unauthorized_counter",  # INVALID for Site 03
                "uncertainty": "certain",
            }]
            status = "REVIEWED_PERSONS_PRESENT"
        else:
            persons = []
            status = "REVIEWED_NO_PERSONS"

        frames.append({
            "frame_idx": f_idx,
            "timestamp_sec": t,
            "segment_id": "prcv_site03_seg1_boutique",
            "frame_status": status,
            "image_filename": f"prcv_site03_frame_{f_idx:06d}.png",
            "annotator_notes": "synthetic fixture",
            "persons": persons,
        })

    wb = {
        "workbook_version": "1.0",
        "dataset_name": "prcv_site03_boutique",
        "clip_path": "data/prcv_site03_boutique.mp4",
        "source_fps": 15.0,
        "sampling_step": 3,
        "target_fps": 5.0,
        "overall_review_status": "REVIEWED_CONSENSUS",
        "segments": [{
            "segment_id": "prcv_site03_seg1_boutique",
            "name": "Segment 1: Multi-Person Boutique Browsing & Crossing",
            "frame_range": [0, 297],
            "sampling_step": 3,
        }],
        "frames": frames,
    }

    report = validate_annotation_workbook(wb, require_trusted_manifest=True)
    assert report.is_valid is False
    assert any("apparel_rack_east" in e and "not in trusted profile allowed_zones" in e for e in report.errors)
    assert any("fitting_room_entry" in e and "not in trusted profile allowed_zones" in e for e in report.errors)
    assert any("unauthorized_counter" in e and "not in trusted profile allowed_zones" in e for e in report.errors)

    # Save to temp file and test CLI invocation
    fixture_path = tmp_path / "synthetic_site03_invalid.json"
    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(wb, f)

    cmd = [
        sys.executable,
        os.path.abspath("scripts/validate_annotation_submission.py"),
        "--workbook",
        str(fixture_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "REJECTED" in res.stdout
    assert "apparel_rack_east" in res.stdout or "fitting_room_entry" in res.stdout


def test_site03_accepts_configured_zone_names_in_synthetic_fixture(tmp_path):
    """
    Focused test: verify that Site 03 submission accepts valid configured zone names
    ('side_display_left', 'display_table_center', 'center_aisle', 'checkout_counter', and None)
    against its authoritative scene profile.
    Uses clearly labelled synthetic fixture.
    """
    frames = []
    zone_cycle = [
        "side_display_left",
        "display_table_center",
        "center_aisle",
        "checkout_counter",
        None,  # unassigned circulation area
    ]

    for i, f_idx in enumerate(range(0, 298, 3)):
        t = round(f_idx / 15.0, 3)
        if i < len(zone_cycle):
            zone = zone_cycle[i]
            persons = [{
                "annotator_track_label": "P1",
                "image_bbox": [100.0 + i * 10, 100.0, 200.0 + i * 10, 300.0],
                "ground_contact_px": [150.0 + i * 10, 300.0],
                "zone_id": zone,
                "uncertainty": "certain",
            }]
            status = "REVIEWED_PERSONS_PRESENT"
        else:
            persons = []
            status = "REVIEWED_NO_PERSONS"

        frames.append({
            "frame_idx": f_idx,
            "timestamp_sec": t,
            "segment_id": "prcv_site03_seg1_boutique",
            "frame_status": status,
            "image_filename": f"prcv_site03_frame_{f_idx:06d}.png",
            "annotator_notes": "synthetic fixture",
            "persons": persons,
        })

    wb = {
        "workbook_version": "1.0",
        "dataset_name": "prcv_site03_boutique",
        "clip_path": "data/prcv_site03_boutique.mp4",
        "source_fps": 15.0,
        "sampling_step": 3,
        "target_fps": 5.0,
        "overall_review_status": "REVIEWED_CONSENSUS",
        "segments": [{
            "segment_id": "prcv_site03_seg1_boutique",
            "name": "Segment 1: Multi-Person Boutique Browsing & Crossing",
            "frame_range": [0, 297],
            "sampling_step": 3,
        }],
        "frames": frames,
    }

    report = validate_annotation_workbook(wb, require_trusted_manifest=True)
    assert report.is_valid is True
    assert report.schema_valid is True
    assert report.coverage_certified is True
    assert report.coverage_status == "certified"
    assert len(report.errors) == 0
    assert report.trusted_zones_enforced == [
        "center_aisle",
        "checkout_counter",
        "display_table_center",
        "side_display_left",
    ]

    # Save to temp file and test CLI invocation
    fixture_path = tmp_path / "synthetic_site03_valid.json"
    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(wb, f)

    cmd = [
        sys.executable,
        os.path.abspath("scripts/validate_annotation_submission.py"),
        "--workbook",
        str(fixture_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "OVERALL VERIFICATION RESULT: PASSED" in res.stdout


def test_cli_does_not_trust_workbook_supplied_zone_list_override(tmp_path):
    """
    Focused test: verify that the CLI does NOT trust an allowed-zone list
    supplied by the submitted workbook itself (e.g. rogue 'target_retail_zones').
    Uses clearly labelled synthetic fixture.
    """
    frames = []
    for f_idx in range(0, 298, 3):
        t = round(f_idx / 15.0, 3)
        if f_idx == 0:
            persons = [{
                "annotator_track_label": "P1",
                "image_bbox": [100.0, 100.0, 200.0, 300.0],
                "ground_contact_px": [150.0, 300.0],
                "zone_id": "rogue_fabricated_zone",
                "uncertainty": "certain",
            }]
            status = "REVIEWED_PERSONS_PRESENT"
        else:
            persons = []
            status = "REVIEWED_NO_PERSONS"

        frames.append({
            "frame_idx": f_idx,
            "timestamp_sec": t,
            "segment_id": "prcv_site03_seg1_boutique",
            "frame_status": status,
            "image_filename": f"prcv_site03_frame_{f_idx:06d}.png",
            "annotator_notes": "synthetic fixture",
            "persons": persons,
        })

    wb = {
        "workbook_version": "1.0",
        "dataset_name": "prcv_site03_boutique",
        "clip_path": "data/prcv_site03_boutique.mp4",
        "source_fps": 15.0,
        "sampling_step": 3,
        "target_fps": 5.0,
        "overall_review_status": "REVIEWED_CONSENSUS",
        "target_retail_zones": ["rogue_fabricated_zone"],  # Attempt to override trusted zones
        "segments": [{
            "segment_id": "prcv_site03_seg1_boutique",
            "name": "Segment 1: Multi-Person Boutique Browsing & Crossing",
            "frame_range": [0, 297],
            "sampling_step": 3,
        }],
        "frames": frames,
    }

    report = validate_annotation_workbook(wb, require_trusted_manifest=True)
    assert report.is_valid is False
    assert any("rogue_fabricated_zone" in e for e in report.errors)
    assert any("declares unauthorized 'target_retail_zones'" in w for w in report.warnings)

    # Save to temp file and test CLI invocation
    fixture_path = tmp_path / "synthetic_site03_rogue.json"
    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(wb, f)

    cmd = [
        sys.executable,
        os.path.abspath("scripts/validate_annotation_submission.py"),
        "--workbook",
        str(fixture_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "REJECTED" in res.stdout
    assert "rogue_fabricated_zone" in res.stdout


def test_every_indexed_image_exists_and_matches_recorded_checksum():
    """
    Verification test: verify every indexed image across MERL and Site 03
    exists on disk and cryptographically matches its recorded SHA-256 checksum.
    """
    index_file = os.path.abspath("data/evaluation_pilot/frame_index.json")
    assert os.path.exists(index_file), f"Master index file missing: {index_file}"

    with open(index_file, "r", encoding="utf-8") as f:
        master = json.load(f)

    package_root = os.path.abspath(master.get("package_root", "data/evaluation_pilot"))
    assert os.path.isdir(package_root), f"Package root directory missing: {package_root}"

    datasets = master["datasets"]
    assert "merl_shopping_aisle" in datasets
    assert "prcv_site03_boutique" in datasets

    total_images_verified = 0

    for d_name, d_meta in datasets.items():
        ds_index_file = d_meta["frame_index_file_absolute"]
        assert os.path.exists(ds_index_file), f"Dataset index file missing: {ds_index_file}"

        with open(ds_index_file, "r", encoding="utf-8") as f:
            ds_index = json.load(f)

        assert ds_index["is_verified"] is True
        assert ds_index["missing_frames_count"] == 0
        frames = ds_index["frames"]
        assert len(frames) == 100, f"Expected 100 frames for {d_name}, got {len(frames)}"

        for entry in frames:
            abs_p = entry["absolute_path"]
            rel_p = entry["relative_path"]
            recorded_hash = entry["sha256"]
            recorded_bytes = entry["file_size_bytes"]

            # Verify relative_path joins correctly to package_root
            joined_p = os.path.abspath(os.path.join(package_root, rel_p))
            assert os.path.exists(joined_p), f"Indexed frame missing at relative_path: {joined_p}"

            # In origin workspace, verify absolute and joined paths coincide
            if os.path.exists(abs_p):
                try:
                    if os.path.samefile(os.path.dirname(abs_p), os.path.dirname(joined_p)):
                        assert os.path.samefile(abs_p, joined_p)
                except Exception:
                    pass

            # Cryptographic SHA-256 verification on local package frame
            with open(joined_p, "rb") as img_f:
                data = img_f.read()

            assert len(data) == recorded_bytes
            computed_hash = hashlib.sha256(data).hexdigest()
            assert computed_hash == recorded_hash, (
                f"Checksum mismatch for frame {entry['frame_idx']} in {d_name}!\n"
                f"Recorded: {recorded_hash}\nComputed: {computed_hash}"
            )

            total_images_verified += 1

    assert total_images_verified == 200, f"Expected 200 verified images, got {total_images_verified}"



