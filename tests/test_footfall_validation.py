"""
Phase 8.4 — Unit Tests for Footfall Event Validation
===================================================
Validates that:
1. Aggregate entry counts from Phase 7 FootfallSummary match GroundTruthEvent 'entry' counts.
2. Aggregate transition counts from Phase 7 FootfallSummary match GroundTruthEvent 'transition' counts.
3. Discrepancies in entry counts produce structured FootfallMetricDisagreement records.
4. Discrepancies in transition counts produce structured FootfallMetricDisagreement records.
5. Empty or non-entry/non-transition events return an explicit INSUFFICIENT_EVIDENCE status.
6. Invalid event types or incoherent fields are rejected by the existing GroundTruthEvent schema.
7. Output comparison ordering is deterministic.
8. JSON and dictionary serialization round-trips preserve all comparison records and metadata.
9. Explicit evidence notes and methodological caveats are present on all results.
10. Existing FootfallSummary outputs are consumed directly rather than recomputing from trajectories.
"""

import json
import pytest

from src.trajectory.trajectory import Trajectory, TrajectoryPoint
from src.analytics.footfall import (
    FootfallAnalyzer,
    FootfallSummary,
    ZoneEntryEvent,
    ZoneVisit,
)
from src.validation.ground_truth import (
    GroundTruthEvent,
    GroundTruthSequence,
)
from src.validation.footfall_validation import (
    EntryCountComparison,
    TransitionCountComparison,
    FootfallMetricDisagreement,
    FootfallValidationResult,
    FootfallValidator,
    validate_footfall,
)


def _make_trajectory(track_id: int, zone_sequence: list, start_frame: int = 0, fps: float = 5.0) -> Trajectory:
    """Helper to build a trajectory with pre-assigned zone_ids for testing."""
    traj = Trajectory(track_id=track_id)
    dt = 1.0 / fps
    for idx, z in enumerate(zone_sequence):
        f = start_frame + idx
        traj.add_point(
            TrajectoryPoint(
                track_id=track_id,
                frame_idx=f,
                timestamp_sec=round(f * dt, 3),
                image_x=100.0,
                image_y=200.0,
                floor_x=1.0,
                floor_y=2.0,
                zone_id=z,
            )
        )
    return traj


@pytest.fixture
def synthetic_footfall_scenario():
    """
    Deterministic synthetic scenario with:
    - 2 trajectories across 2 named zones ("shelf_a", "shelf_b") and outside (None).
    - Trajectory 1 (track 101): None -> shelf_a -> shelf_a -> shelf_b -> shelf_b -> None
      Entries: shelf_a (from None), shelf_b (from shelf_a)
      Transitions: shelf_a -> shelf_b
    - Trajectory 2 (track 102): None -> shelf_b -> shelf_b -> shelf_a -> shelf_a
      Entries: shelf_b (from None), shelf_a (from shelf_b)
      Transitions: shelf_b -> shelf_a
    - Run through existing FootfallAnalyzer to obtain a genuine FootfallSummary.
    - Matching ground-truth entry and transition events.
    """
    traj1 = _make_trajectory(101, [None, "shelf_a", "shelf_a", "shelf_b", "shelf_b", None])
    traj2 = _make_trajectory(102, [None, "shelf_b", "shelf_b", "shelf_a", "shelf_a"])

    analyzer = FootfallAnalyzer(known_zone_ids=["shelf_a", "shelf_b"])
    summary = analyzer.analyze([traj1, traj2])

    # Expected GroundTruthEvent records matching summary perfectly
    matching_events = [
        # Track 101 / person_1: entry shelf_a, transition shelf_a->shelf_b, entry shelf_b
        GroundTruthEvent("entry", "person_1", 1, 0.2, from_zone_id=None, to_zone_id="shelf_a"),
        GroundTruthEvent("transition", "person_1", 3, 0.6, from_zone_id="shelf_a", to_zone_id="shelf_b"),
        GroundTruthEvent("entry", "person_1", 3, 0.6, from_zone_id="shelf_a", to_zone_id="shelf_b"),
        # Track 102 / person_2: entry shelf_b, transition shelf_b->shelf_a, entry shelf_a
        GroundTruthEvent("entry", "person_2", 1, 0.2, from_zone_id=None, to_zone_id="shelf_b"),
        GroundTruthEvent("transition", "person_2", 3, 0.6, from_zone_id="shelf_b", to_zone_id="shelf_a"),
        GroundTruthEvent("entry", "person_2", 3, 0.6, from_zone_id="shelf_b", to_zone_id="shelf_a"),
    ]

    return summary, matching_events


def test_matching_entry_counts(synthetic_footfall_scenario):
    """Test 1: Perfectly matching ground-truth entry counts yield 100% entry consistency."""
    summary, matching_events = synthetic_footfall_scenario

    result = FootfallValidator.validate_summary(
        summary=summary,
        expected_events=matching_events,
        dataset_name="synthetic_perfect",
    )

    assert result.evidence_status == "FOOTFALL_COMPARISON_EVALUATED"
    assert result.expected_entry_counts == {"shelf_a": 2, "shelf_b": 2}
    assert result.predicted_entry_counts == {"shelf_a": 2, "shelf_b": 2}

    # Verify per-zone entry comparisons
    assert len(result.entry_comparisons) == 2
    for comp in result.entry_comparisons:
        assert comp.is_match is True
        assert comp.expected_count == 2
        assert comp.predicted_count == 2
        assert comp.difference == 0
        assert comp.absolute_difference == 0


def test_matching_transition_counts(synthetic_footfall_scenario):
    """Test 2: Perfectly matching ground-truth transition counts yield 100% transition consistency."""
    summary, matching_events = synthetic_footfall_scenario

    result = FootfallValidator.validate_summary(
        summary=summary,
        expected_events=matching_events,
        dataset_name="synthetic_perfect",
    )

    assert result.expected_transition_counts["shelf_a->shelf_b"] == 1
    assert result.predicted_transition_counts["shelf_a->shelf_b"] == 1
    assert result.expected_transition_counts["shelf_b->shelf_a"] == 1
    assert result.predicted_transition_counts["shelf_b->shelf_a"] == 1

    # Verify transition comparisons
    for comp in result.transition_comparisons:
        assert comp.is_match is True
        assert comp.difference == 0
        assert comp.absolute_difference == 0

    assert result.matched_metric_count == 4  # 2 entry metrics + 2 transition metrics
    assert result.mismatched_metric_count == 0
    assert result.metric_consistency_rate == 1.0
    assert len(result.disagreements) == 0


def test_detection_of_entry_count_mismatches(synthetic_footfall_scenario):
    """Test 3: Detection of entry-count discrepancies between GT and summary."""
    summary, matching_events = synthetic_footfall_scenario

    # Alter GT: person_2 never entered shelf_a in GT (expected shelf_a: 1 instead of 2)
    events_with_entry_error = [
        e for e in matching_events
        if not (e.event_type == "entry" and e.gt_id == "person_2" and e.to_zone_id == "shelf_a")
    ]

    result = FootfallValidator.validate_summary(
        summary=summary,
        expected_events=events_with_entry_error,
        dataset_name="synthetic_entry_mismatch",
    )

    assert result.mismatched_metric_count == 1
    assert result.metric_consistency_rate == 0.75  # 3 of 4 metrics match
    assert len(result.disagreements) == 1

    d = result.disagreements[0]
    assert d.metric_type == "entry_count"
    assert d.key == "shelf_a"
    assert d.expected_count == 1
    assert d.predicted_count == 2
    assert d.difference == 1       # predicted - expected
    assert d.absolute_difference == 1
    assert "Discrepancy" in d.evidence_note or "mismatch" in d.evidence_note


def test_detection_of_transition_count_mismatches(synthetic_footfall_scenario):
    """Test 4: Detection of transition-count discrepancies between GT and summary."""
    summary, matching_events = synthetic_footfall_scenario

    # Alter GT: omit the transition from shelf_b to shelf_a
    events_with_trans_error = [
        e for e in matching_events
        if not (e.event_type == "transition" and e.from_zone_id == "shelf_b" and e.to_zone_id == "shelf_a")
    ]

    result = FootfallValidator.validate_summary(
        summary=summary,
        expected_events=events_with_trans_error,
        dataset_name="synthetic_trans_mismatch",
    )

    assert result.mismatched_metric_count == 1
    assert result.metric_consistency_rate == 0.75
    assert len(result.disagreements) == 1

    d = result.disagreements[0]
    assert d.metric_type == "transition_count"
    assert d.key == "shelf_b->shelf_a"
    assert d.expected_count == 0
    assert d.predicted_count == 1
    assert d.difference == 1
    assert d.absolute_difference == 1


def test_empty_ground_truth_returns_insufficient_evidence(synthetic_footfall_scenario):
    """Test 5: Empty event list or list with no entry/transition events returns INSUFFICIENT_EVIDENCE."""
    summary, _ = synthetic_footfall_scenario

    # Completely empty event list
    res_empty = FootfallValidator.validate_summary(summary, [])
    assert res_empty.evidence_status == "INSUFFICIENT_EVIDENCE"
    assert res_empty.metric_consistency_rate is None
    assert res_empty.total_compared_metrics == 0
    assert res_empty.matched_metric_count == 0
    assert "No ground-truth entry or transition events provided" in res_empty.evidence_note

    # Event list containing only 'exit' events (unusable for entry/transition evaluation)
    exit_only = [
        GroundTruthEvent("exit", "p1", 5, 1.0, from_zone_id="shelf_a")
    ]
    res_exit = FootfallValidator.validate_summary(summary, exit_only)
    assert res_exit.evidence_status == "INSUFFICIENT_EVIDENCE"
    assert res_exit.metric_consistency_rate is None


def test_invalid_event_types_and_incoherent_fields_rejected():
    """Test 6: Existing schema rejects invalid event types and incoherent fields."""
    # Invalid event_type 'visit' (not supported by GroundTruthEvent schema)
    with pytest.raises(ValueError, match="event_type must be one of"):
        GroundTruthEvent("visit", "p1", 0, 0.0, to_zone_id="shelf_a")

    # Incoherent entry event (missing to_zone_id)
    with pytest.raises(ValueError, match="requires a non-empty 'to_zone_id'"):
        GroundTruthEvent("entry", "p1", 0, 0.0, to_zone_id="")

    # Incoherent transition event (identical from and to zones)
    with pytest.raises(ValueError, match="requires different from_zone_id and to_zone_id"):
        GroundTruthEvent("transition", "p1", 0, 0.0, from_zone_id="shelf_a", to_zone_id="shelf_a")

    # Type validation in FootfallValidator
    with pytest.raises(TypeError, match="summary must be a FootfallSummary"):
        FootfallValidator.validate_summary("invalid_summary", [])  # type: ignore

    with pytest.raises(TypeError, match="All items in expected_events must be GroundTruthEvent"):
        # Dummy FootfallSummary
        dummy_sum = FootfallSummary(
            unique_tracks_per_zone={},
            entry_events_per_zone={},
            visits_per_zone={},
            total_visit_duration_sec={},
            average_visit_duration_sec={},
            transition_matrix={},
            entries=[],
            visits=[],
        )
        FootfallValidator.validate_summary(dummy_sum, ["not_an_event"])  # type: ignore


def test_deterministic_comparison_output(synthetic_footfall_scenario):
    """Test 7: Ordering of comparisons and dictionary keys is strictly deterministic."""
    summary, matching_events = synthetic_footfall_scenario

    res1 = FootfallValidator.validate_summary(summary, matching_events)
    res2 = FootfallValidator.validate_summary(summary, matching_events)

    assert list(res1.expected_entry_counts.keys()) == list(res2.expected_entry_counts.keys())
    assert [c.zone_id for c in res1.entry_comparisons] == [c.zone_id for c in res2.entry_comparisons]
    assert [c.transition_key for c in res1.transition_comparisons] == [
        c.transition_key for c in res2.transition_comparisons
    ]
    assert res1.to_dict() == res2.to_dict()


def test_json_and_dictionary_round_trip(synthetic_footfall_scenario):
    """Test 8: FootfallValidationResult serializes to dict/JSON and deserializes identically."""
    summary, matching_events = synthetic_footfall_scenario

    # Introduce a mismatch so disagreements are also serialized
    mismatched_events = matching_events[:-1]
    result = FootfallValidator.validate_summary(summary, mismatched_events, dataset_name="json_test")

    data = result.to_dict()
    json_str = json.dumps(data)
    reloaded_dict = json.loads(json_str)
    reconstructed = FootfallValidationResult.from_dict(reloaded_dict)

    assert reconstructed.dataset_name == result.dataset_name
    assert reconstructed.total_compared_metrics == result.total_compared_metrics
    assert reconstructed.matched_metric_count == result.matched_metric_count
    assert reconstructed.mismatched_metric_count == result.mismatched_metric_count
    assert reconstructed.metric_consistency_rate == result.metric_consistency_rate
    assert reconstructed.evidence_status == result.evidence_status
    assert reconstructed.expected_entry_counts == result.expected_entry_counts
    assert reconstructed.predicted_entry_counts == result.predicted_entry_counts
    assert len(reconstructed.disagreements) == len(result.disagreements)
    if result.disagreements:
        assert reconstructed.disagreements[0].key == result.disagreements[0].key
        assert reconstructed.disagreements[0].difference == result.disagreements[0].difference


def test_explicit_evidence_notes_and_limitations(synthetic_footfall_scenario):
    """Test 9: Results clearly communicate evidence scope and methodology caveats."""
    summary, matching_events = synthetic_footfall_scenario
    result = FootfallValidator.validate_summary(summary, matching_events)

    assert result.evidence_status == "FOOTFALL_COMPARISON_EVALUATED"
    # Note must clearly state that this is descriptive consistency, not universal tracking accuracy
    assert "Descriptive event-count consistency" in result.evidence_note
    assert "does not claim universal tracking accuracy" in result.evidence_note


def test_consumes_existing_footfall_summary_without_recomputing():
    """
    Test 10: Confirms that FootfallValidator directly consumes FootfallSummary metrics
    without requiring or inspecting raw trajectories, PIP evaluation, or hysteresis.
    """
    # Create an artificial FootfallSummary with known custom numbers
    custom_summary = FootfallSummary(
        unique_tracks_per_zone={"checkout": 10, "entrance": 15},
        entry_events_per_zone={"checkout": 12, "entrance": 20},
        visits_per_zone={"checkout": 12, "entrance": 20},
        total_visit_duration_sec={"checkout": 120.0, "entrance": 200.0},
        average_visit_duration_sec={"checkout": 10.0, "entrance": 10.0},
        transition_matrix={
            "entrance": {"entrance": 0, "checkout": 8},
            "checkout": {"entrance": 3, "checkout": 0},
        },
        entries=[],
        visits=[],
    )

    # Supply GroundTruthEvents matching the custom summary exactly
    gt_events = [
        # 12 entries into checkout
        *[GroundTruthEvent("entry", f"gt_{i}", i, float(i), to_zone_id="checkout") for i in range(12)],
        # 20 entries into entrance
        *[GroundTruthEvent("entry", f"gt_{i}", i, float(i), to_zone_id="entrance") for i in range(20)],
        # 8 transitions entrance -> checkout
        *[
            GroundTruthEvent(
                "transition", f"gt_{i}", i, float(i), from_zone_id="entrance", to_zone_id="checkout"
            )
            for i in range(8)
        ],
        # 3 transitions checkout -> entrance
        *[
            GroundTruthEvent(
                "transition", f"gt_{i}", i, float(i), from_zone_id="checkout", to_zone_id="entrance"
            )
            for i in range(3)
        ],
    ]

    result = FootfallValidator.validate_summary(
        summary=custom_summary,
        expected_events=gt_events,
        dataset_name="direct_summary_test",
    )

    assert result.predicted_entry_counts["checkout"] == 12
    assert result.expected_entry_counts["checkout"] == 12
    assert result.predicted_entry_counts["entrance"] == 20
    assert result.expected_entry_counts["entrance"] == 20
    assert result.predicted_transition_counts["entrance->checkout"] == 8
    assert result.expected_transition_counts["entrance->checkout"] == 8
    assert result.predicted_transition_counts["checkout->entrance"] == 3
    assert result.expected_transition_counts["checkout->entrance"] == 3
    assert result.matched_metric_count == 4
    assert result.mismatched_metric_count == 0
    assert result.metric_consistency_rate == 1.0


def test_sequence_adapter_and_convenience_function(synthetic_footfall_scenario):
    """Test 11: Validates GroundTruthSequence integration and validate_footfall convenience helper."""
    summary, matching_events = synthetic_footfall_scenario

    seq = GroundTruthSequence(
        schema_version="1.0",
        sequence_id="SEQ_FOOTFALL_01",
        source="synthetic",
        fps=5.0,
        events=matching_events,
    )

    # 1. Via FootfallValidator.validate_sequence
    res1 = FootfallValidator.validate_sequence(summary, seq)
    assert res1.dataset_name == "SEQ_FOOTFALL_01"
    assert res1.metric_consistency_rate == 1.0

    # 2. Via validate_footfall with sequence
    res2 = validate_footfall(summary, seq)
    assert res2.dataset_name == "SEQ_FOOTFALL_01"
    assert res2.metric_consistency_rate == 1.0

    # 3. Via validate_footfall with list of events
    res3 = validate_footfall(summary, matching_events, dataset_name="event_list_test")
    assert res3.dataset_name == "event_list_test"
    assert res3.metric_consistency_rate == 1.0
