"""
Unit and integration tests for Phase 11.1: Zone Visit Sequences & Direct Transitions.
====================================================================================
Verifies:
1. Trajectory and point validation (non-monotonicity, non-finite values, malformed inputs).
2. Observed zone visit run extraction (single vs multi-point, timestamps, durations).
3. Direct zone-to-zone transitions (A -> B direct transitions).
4. Unassigned (None) observation chain-breaking (A -> None -> B yields 0 direct transitions).
5. Cycle and oscillation transitions (A -> B -> A).
6. Multi-track isolation (transitions are strictly per-track; never cross-track).
7. Matrix creation, known_zone_ids preservation, and zero-count retention.
8. Serialization to dictionary and JSON compatibility.
9. Immutability of caller trajectory objects.
10. Convenience functions and single vs batch inputs.
"""

import json
import pytest
from typing import List, Optional

from src.trajectory.trajectory import Trajectory, TrajectoryPoint
from src.analytics.transitions import (
    ZoneVisitRun,
    ObservedTrackSequence,
    ZoneActivitySummary,
    DirectTransitionSummary,
    TransitionAnalyzer,
    extract_zone_visit_runs,
    extract_track_sequence,
    compute_direct_transitions,
    compute_zone_activity_summaries,
)


def _make_point(
    track_id: int,
    frame_idx: int,
    timestamp_sec: float,
    zone_id: Optional[str] = None,
    image_x: float = 100.0,
    image_y: float = 200.0,
    floor_x: Optional[float] = 1.0,
    floor_y: Optional[float] = 2.0,
) -> TrajectoryPoint:
    """Helper to build a valid TrajectoryPoint."""
    return TrajectoryPoint(
        track_id=track_id,
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        image_x=image_x,
        image_y=image_y,
        floor_x=floor_x,
        floor_y=floor_y,
        zone_id=zone_id,
    )


def test_empty_trajectories_and_empty_points():
    """Verify handling of empty trajectory collections and empty point sequences."""
    # 1. Empty trajectories list
    summary = compute_direct_transitions([])
    assert summary.total_tracks == 0
    assert summary.total_transitions == 0
    assert summary.transition_counts == {}
    assert summary.transition_matrix == {}
    assert summary.track_sequences == []

    # 2. Trajectory with empty points
    empty_traj = Trajectory(track_id=1, points=[])
    seq = extract_track_sequence(empty_traj)
    assert seq.track_id == 1
    assert seq.runs == []
    assert seq.zone_sequence == []
    assert seq.direct_transitions == []
    assert seq.total_runs == 0
    assert seq.total_transitions == 0

    # 3. Trajectory with only None (unassigned) points
    none_pts = [
        _make_point(track_id=2, frame_idx=0, timestamp_sec=0.0, zone_id=None),
        _make_point(track_id=2, frame_idx=1, timestamp_sec=0.2, zone_id=None),
        _make_point(track_id=2, frame_idx=2, timestamp_sec=0.4, zone_id=None),
    ]
    none_traj = Trajectory(track_id=2, points=none_pts)
    seq_none = extract_track_sequence(none_traj)
    assert seq_none.runs == []
    assert seq_none.zone_sequence == []
    assert seq_none.direct_transitions == []


def test_single_zone_repeated_observations():
    """Consecutive points in the same zone form a single run with 0 transitions."""
    pts = [
        _make_point(track_id=10, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
        _make_point(track_id=10, frame_idx=1, timestamp_sec=0.2, zone_id="Zone_A"),
        _make_point(track_id=10, frame_idx=2, timestamp_sec=0.4, zone_id="Zone_A"),
        _make_point(track_id=10, frame_idx=3, timestamp_sec=0.6, zone_id="Zone_A"),
    ]
    traj = Trajectory(track_id=10, points=pts)
    seq = extract_track_sequence(traj)

    assert seq.total_runs == 1
    assert seq.zone_sequence == ["Zone_A"]
    assert seq.direct_transitions == []

    run = seq.runs[0]
    assert run.track_id == 10
    assert run.zone_id == "Zone_A"
    assert run.entry_frame_idx == 0
    assert run.exit_frame_idx == 3
    assert run.entry_timestamp_sec == 0.0
    assert run.exit_timestamp_sec == 0.6
    assert run.observation_count == 4
    assert run.duration_sec == pytest.approx(0.6)


def test_direct_transition_between_distinct_zones():
    """Direct transition A -> B creates two runs and exactly one direct transition."""
    pts = [
        _make_point(track_id=20, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
        _make_point(track_id=20, frame_idx=1, timestamp_sec=0.2, zone_id="Zone_A"),
        _make_point(track_id=20, frame_idx=2, timestamp_sec=0.4, zone_id="Zone_B"),
        _make_point(track_id=20, frame_idx=3, timestamp_sec=0.6, zone_id="Zone_B"),
        _make_point(track_id=20, frame_idx=4, timestamp_sec=0.8, zone_id="Zone_B"),
    ]
    traj = Trajectory(track_id=20, points=pts)
    seq = extract_track_sequence(traj)

    assert seq.total_runs == 2
    assert seq.zone_sequence == ["Zone_A", "Zone_B"]
    assert seq.direct_transitions == [("Zone_A", "Zone_B")]
    assert seq.total_transitions == 1

    run_a = seq.runs[0]
    assert run_a.zone_id == "Zone_A"
    assert run_a.entry_frame_idx == 0
    assert run_a.exit_frame_idx == 1
    assert run_a.observation_count == 2
    assert run_a.duration_sec == pytest.approx(0.2)

    run_b = seq.runs[1]
    assert run_b.zone_id == "Zone_B"
    assert run_b.entry_frame_idx == 2
    assert run_b.exit_frame_idx == 4
    assert run_b.observation_count == 3
    assert run_b.duration_sec == pytest.approx(0.4)

    summary = compute_direct_transitions(traj)
    assert summary.total_tracks == 1
    assert summary.total_transitions == 1
    assert summary.transition_counts == {("Zone_A", "Zone_B"): 1}
    assert summary.transition_matrix["Zone_A"]["Zone_B"] == 1
    assert summary.transition_matrix["Zone_B"]["Zone_A"] == 0
    assert summary.transition_matrix["Zone_A"]["Zone_A"] == 0


def test_direct_cycle_and_oscillation():
    """Direct A -> B -> A cycle creates 3 runs and 2 direct transitions."""
    pts = [
        _make_point(track_id=30, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
        _make_point(track_id=30, frame_idx=1, timestamp_sec=0.2, zone_id="Zone_B"),
        _make_point(track_id=30, frame_idx=2, timestamp_sec=0.4, zone_id="Zone_A"),
    ]
    traj = Trajectory(track_id=30, points=pts)
    seq = extract_track_sequence(traj)

    assert seq.total_runs == 3
    assert seq.zone_sequence == ["Zone_A", "Zone_B", "Zone_A"]
    assert seq.direct_transitions == [("Zone_A", "Zone_B"), ("Zone_B", "Zone_A")]
    assert seq.total_transitions == 2

    summary = compute_direct_transitions(traj)
    assert summary.total_transitions == 2
    assert summary.transition_counts == {
        ("Zone_A", "Zone_B"): 1,
        ("Zone_B", "Zone_A"): 1,
    }
    assert summary.transition_matrix["Zone_A"]["Zone_B"] == 1
    assert summary.transition_matrix["Zone_B"]["Zone_A"] == 1


def test_unassigned_observation_breaks_transition_chain():
    """Intervening unassigned (None) observation breaks the transition chain."""
    # Track visits Zone_A, exits to unassigned None, then enters Zone_B
    pts = [
        _make_point(track_id=40, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
        _make_point(track_id=40, frame_idx=1, timestamp_sec=0.2, zone_id="Zone_A"),
        _make_point(track_id=40, frame_idx=2, timestamp_sec=0.4, zone_id=None),
        _make_point(track_id=40, frame_idx=3, timestamp_sec=0.6, zone_id="Zone_B"),
        _make_point(track_id=40, frame_idx=4, timestamp_sec=0.8, zone_id="Zone_B"),
    ]
    traj = Trajectory(track_id=40, points=pts)
    seq = extract_track_sequence(traj)

    # 2 distinct runs are recorded
    assert seq.total_runs == 2
    assert seq.zone_sequence == ["Zone_A", "Zone_B"]
    # BUT 0 direct transitions because None intervened!
    assert seq.direct_transitions == []
    assert seq.total_transitions == 0

    summary = compute_direct_transitions(traj)
    assert summary.total_transitions == 0
    assert summary.transition_counts == {}
    assert summary.transition_matrix["Zone_A"]["Zone_B"] == 0
    assert summary.transition_matrix["Zone_B"]["Zone_A"] == 0


def test_unassigned_points_at_boundaries_and_reentrant_same_zone():
    """Points outside at start/end and multiple runs to same zone separated by None."""
    pts = [
        _make_point(track_id=50, frame_idx=0, timestamp_sec=0.0, zone_id=None),
        _make_point(track_id=50, frame_idx=1, timestamp_sec=0.2, zone_id="Zone_A"),
        _make_point(track_id=50, frame_idx=2, timestamp_sec=0.4, zone_id=None),
        _make_point(track_id=50, frame_idx=3, timestamp_sec=0.6, zone_id="Zone_A"),
        _make_point(track_id=50, frame_idx=4, timestamp_sec=0.8, zone_id=None),
    ]
    traj = Trajectory(track_id=50, points=pts)
    seq = extract_track_sequence(traj)

    # Two separate runs in Zone_A
    assert seq.total_runs == 2
    assert seq.zone_sequence == ["Zone_A", "Zone_A"]
    # No direct transitions (starts outside, leaves to outside, re-enters, exits to outside)
    assert seq.direct_transitions == []
    assert seq.total_transitions == 0


def test_mixed_direct_and_broken_transitions():
    """Trajectory with both direct transitions and None breaks."""
    # A -> B (direct) -> None (break) -> C -> D (direct)
    pts = [
        _make_point(track_id=60, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
        _make_point(track_id=60, frame_idx=1, timestamp_sec=0.2, zone_id="Zone_B"),
        _make_point(track_id=60, frame_idx=2, timestamp_sec=0.4, zone_id=None),
        _make_point(track_id=60, frame_idx=3, timestamp_sec=0.6, zone_id="Zone_C"),
        _make_point(track_id=60, frame_idx=4, timestamp_sec=0.8, zone_id="Zone_D"),
    ]
    traj = Trajectory(track_id=60, points=pts)
    seq = extract_track_sequence(traj)

    assert seq.total_runs == 4
    assert seq.zone_sequence == ["Zone_A", "Zone_B", "Zone_C", "Zone_D"]
    assert seq.direct_transitions == [("Zone_A", "Zone_B"), ("Zone_C", "Zone_D")]
    assert seq.total_transitions == 2


def test_multi_track_isolation_and_aggregation():
    """Transitions are strictly per-track and never bridge across different tracks."""
    # Track 100: A -> B
    pts_1 = [
        _make_point(track_id=100, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
        _make_point(track_id=100, frame_idx=1, timestamp_sec=0.2, zone_id="Zone_B"),
    ]
    # Track 101: B -> C
    pts_2 = [
        _make_point(track_id=101, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_B"),
        _make_point(track_id=101, frame_idx=1, timestamp_sec=0.2, zone_id="Zone_C"),
    ]
    # Track 102: A -> B (second occurrence of A -> B)
    pts_3 = [
        _make_point(track_id=102, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
        _make_point(track_id=102, frame_idx=1, timestamp_sec=0.2, zone_id="Zone_B"),
    ]

    trajs = [
        Trajectory(track_id=100, points=pts_1),
        Trajectory(track_id=101, points=pts_2),
        Trajectory(track_id=102, points=pts_3),
    ]

    summary = compute_direct_transitions(trajs)
    assert summary.total_tracks == 3
    assert summary.total_transitions == 3
    assert summary.transition_counts == {
        ("Zone_A", "Zone_B"): 2,
        ("Zone_B", "Zone_C"): 1,
    }
    assert summary.transition_matrix["Zone_A"]["Zone_B"] == 2
    assert summary.transition_matrix["Zone_B"]["Zone_C"] == 1
    assert summary.transition_matrix["Zone_C"]["Zone_A"] == 0


def test_known_zone_ids_config_preserves_zero_count_zones():
    """Providing known_zone_ids ensures unvisited zones are present in the transition matrix with 0s."""
    pts = [
        _make_point(track_id=70, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
        _make_point(track_id=70, frame_idx=1, timestamp_sec=0.2, zone_id="Zone_B"),
    ]
    traj = Trajectory(track_id=70, points=pts)

    known_zones = ["Zone_A", "Zone_B", "Zone_C", "Zone_D"]
    summary = compute_direct_transitions(traj, known_zone_ids=known_zones)

    assert summary.known_zone_ids == ["Zone_A", "Zone_B", "Zone_C", "Zone_D"]
    assert set(summary.transition_matrix.keys()) == set(known_zones)

    for z1 in known_zones:
        for z2 in known_zones:
            expected = 1 if (z1, z2) == ("Zone_A", "Zone_B") else 0
            assert summary.transition_matrix[z1][z2] == expected


def test_caller_trajectories_and_points_are_not_mutated():
    """Verify that transition extraction does not alter caller trajectory points or attributes."""
    pts = [
        _make_point(track_id=80, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
        _make_point(track_id=80, frame_idx=1, timestamp_sec=0.2, zone_id="Zone_B"),
    ]
    traj = Trajectory(track_id=80, points=pts)

    # Snapshot state
    pts_repr_before = [(p.track_id, p.frame_idx, p.timestamp_sec, p.zone_id) for p in traj.points]

    summary = compute_direct_transitions([traj])

    pts_repr_after = [(p.track_id, p.frame_idx, p.timestamp_sec, p.zone_id) for p in traj.points]
    assert pts_repr_before == pts_repr_after
    assert len(traj.points) == 2


def test_single_point_run_duration_is_zero():
    """A run with exactly one observation has duration_sec == 0.0s."""
    pts = [
        _make_point(track_id=90, frame_idx=5, timestamp_sec=1.0, zone_id="Zone_A"),
    ]
    traj = Trajectory(track_id=90, points=pts)
    runs = extract_zone_visit_runs(traj)

    assert len(runs) == 1
    assert runs[0].observation_count == 1
    assert runs[0].duration_sec == 0.0


def test_dataclass_and_serialization_round_trip():
    """Verify to_dict() outputs valid, JSON-serializable dictionaries for all structures."""
    pts = [
        _make_point(track_id=1, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
        _make_point(track_id=1, frame_idx=1, timestamp_sec=0.2, zone_id="Zone_B"),
    ]
    traj = Trajectory(track_id=1, points=pts)
    summary = compute_direct_transitions(traj)

    d = summary.to_dict()
    assert isinstance(d, dict)
    assert d["total_tracks"] == 1
    assert d["total_transitions"] == 1
    assert "Zone_A->Zone_B" in d["transition_counts"]
    assert d["transition_counts"]["Zone_A->Zone_B"] == 1
    assert "Zone_A" in d["transition_matrix"]
    assert len(d["track_sequences"]) == 1

    # Verify JSON serialization succeeds without error
    json_str = json.dumps(d)
    parsed = json.loads(json_str)
    assert parsed["total_transitions"] == 1


def test_temporal_and_schema_validation_rejections():
    """Verify strict validation and rejection of invalid timestamps, frames, and IDs."""
    # 1. Non-monotonic timestamp
    bad_pts_1 = [
        _make_point(track_id=1, frame_idx=0, timestamp_sec=1.0, zone_id="Zone_A"),
        _make_point(track_id=1, frame_idx=1, timestamp_sec=0.5, zone_id="Zone_A"),
    ]
    with pytest.raises(ValueError, match="Non-monotonic timestamp"):
        extract_track_sequence(Trajectory(track_id=1, points=bad_pts_1))

    # 2. Non-monotonic frame index
    bad_pts_2 = [
        _make_point(track_id=1, frame_idx=10, timestamp_sec=1.0, zone_id="Zone_A"),
        _make_point(track_id=1, frame_idx=5, timestamp_sec=2.0, zone_id="Zone_A"),
    ]
    with pytest.raises(ValueError, match="Non-monotonic frame index"):
        extract_track_sequence(Trajectory(track_id=1, points=bad_pts_2))

    # 3. Non-finite timestamp (NaN)
    bad_pts_3 = [
        _make_point(track_id=1, frame_idx=0, timestamp_sec=float("nan"), zone_id="Zone_A"),
    ]
    with pytest.raises(ValueError, match="timestamp_sec must be a finite number"):
        extract_track_sequence(Trajectory(track_id=1, points=bad_pts_3))

    # 4. Negative frame index
    bad_pts_4 = [
        _make_point(track_id=1, frame_idx=-1, timestamp_sec=0.0, zone_id="Zone_A"),
    ]
    with pytest.raises(ValueError, match="frame_idx must be non-negative"):
        extract_track_sequence(Trajectory(track_id=1, points=bad_pts_4))

    # 5. Empty zone string
    bad_pts_5 = [
        _make_point(track_id=1, frame_idx=0, timestamp_sec=0.0, zone_id="   "),
    ]
    with pytest.raises(ValueError, match="zone_id cannot be an empty or whitespace string"):
        extract_track_sequence(Trajectory(track_id=1, points=bad_pts_5))

    # 6. Point track_id mismatch
    bad_pts_6 = [
        _make_point(track_id=99, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
    ]
    with pytest.raises(ValueError, match="Point track_id.*does not match"):
        extract_track_sequence(Trajectory(track_id=1, points=bad_pts_6))

    # 7. Non-Trajectory input
    with pytest.raises(TypeError, match="Expected Trajectory instance"):
        extract_track_sequence("not_a_trajectory")  # type: ignore

    # 8. Duplicate track_id in batch
    t1 = Trajectory(track_id=5, points=[])
    t2 = Trajectory(track_id=5, points=[])
    with pytest.raises(ValueError, match="Duplicate track_id 5"):
        compute_direct_transitions([t1, t2])

    # 9. Invalid known_zone_ids
    with pytest.raises(TypeError, match="known_zone_ids must be a sequence of strings"):
        compute_direct_transitions([], known_zone_ids="Zone_A")  # type: ignore


def test_zone_visit_run_direct_instantiation_validation():
    """Verify validation in ZoneVisitRun __post_init__."""
    with pytest.raises(ValueError, match="Invalid track_id"):
        ZoneVisitRun(track_id=-1, zone_id="A", entry_frame_idx=0, exit_frame_idx=1, entry_timestamp_sec=0.0, exit_timestamp_sec=1.0, observation_count=2)

    with pytest.raises(ValueError, match="Invalid zone_id"):
        ZoneVisitRun(track_id=0, zone_id="", entry_frame_idx=0, exit_frame_idx=1, entry_timestamp_sec=0.0, exit_timestamp_sec=1.0, observation_count=2)

    with pytest.raises(ValueError, match="exit_frame_idx.*cannot precede entry_frame_idx"):
        ZoneVisitRun(track_id=0, zone_id="A", entry_frame_idx=5, exit_frame_idx=2, entry_timestamp_sec=0.0, exit_timestamp_sec=1.0, observation_count=2)

    with pytest.raises(ValueError, match="exit_timestamp_sec.*cannot precede entry_timestamp_sec"):
        ZoneVisitRun(track_id=0, zone_id="A", entry_frame_idx=0, exit_frame_idx=1, entry_timestamp_sec=2.0, exit_timestamp_sec=1.0, observation_count=2)

    with pytest.raises(ValueError, match="observation_count must be at least 1"):
        ZoneVisitRun(track_id=0, zone_id="A", entry_frame_idx=0, exit_frame_idx=1, entry_timestamp_sec=0.0, exit_timestamp_sec=1.0, observation_count=0)


def test_observed_track_sequence_direct_instantiation_validation():
    """Verify validation in ObservedTrackSequence __post_init__."""
    run = ZoneVisitRun(track_id=1, zone_id="A", entry_frame_idx=0, exit_frame_idx=0, entry_timestamp_sec=0.0, exit_timestamp_sec=0.0, observation_count=1)

    # Track ID mismatch between run and sequence
    with pytest.raises(ValueError, match="Run track_id.*does not match sequence track_id"):
        ObservedTrackSequence(track_id=2, runs=[run], zone_sequence=["A"], direct_transitions=[])

    # Direct transition with same zone (not distinct)
    with pytest.raises(ValueError, match="Direct transition must be between distinct zones"):
        ObservedTrackSequence(track_id=1, runs=[run], zone_sequence=["A"], direct_transitions=[("A", "A")])


# ==============================================================================
# Phase 11.2 — Descriptive Per-Zone Activity Summaries Tests
# ==============================================================================

def test_zone_activity_summary_unvisited_known_zone_distinguishable_from_missing():
    """Verify that unvisited known zones evaluate to 0 counts and are distinguishable from missing/unconfigured zones."""
    pts = [
        _make_point(track_id=1, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
        _make_point(track_id=1, frame_idx=1, timestamp_sec=0.2, zone_id="Zone_B"),
    ]
    traj = Trajectory(track_id=1, points=pts)
    known = ["Zone_A", "Zone_B", "Zone_Unvisited"]

    summary = compute_direct_transitions(traj, known_zone_ids=known)

    # 1. Unvisited known zone is present with distinct zero counts
    assert "Zone_Unvisited" in summary.zone_summaries
    unvisited_summary = summary.zone_summaries["Zone_Unvisited"]
    assert unvisited_summary.zone_id == "Zone_Unvisited"
    assert unvisited_summary.observed_run_count == 0
    assert unvisited_summary.distinct_track_count == 0
    assert unvisited_summary.direct_incoming_count == 0
    assert unvisited_summary.direct_outgoing_count == 0

    # 2. Unconfigured missing zone is NOT in zone_summaries
    assert "Zone_Missing" not in summary.zone_summaries

    # 3. Visited zones have positive counts
    assert summary.zone_summaries["Zone_A"].observed_run_count == 1
    assert summary.zone_summaries["Zone_A"].distinct_track_count == 1
    assert summary.zone_summaries["Zone_A"].direct_incoming_count == 0
    assert summary.zone_summaries["Zone_A"].direct_outgoing_count == 1

    assert summary.zone_summaries["Zone_B"].observed_run_count == 1
    assert summary.zone_summaries["Zone_B"].distinct_track_count == 1
    assert summary.zone_summaries["Zone_B"].direct_incoming_count == 1
    assert summary.zone_summaries["Zone_B"].direct_outgoing_count == 0


def test_zone_activity_summary_repeated_runs_same_track():
    """A single track re-entering the same zone has observed_run_count > 1 but distinct_track_count == 1."""
    # Track 42: Zone_A -> None -> Zone_A
    pts = [
        _make_point(track_id=42, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
        _make_point(track_id=42, frame_idx=1, timestamp_sec=0.2, zone_id=None),
        _make_point(track_id=42, frame_idx=2, timestamp_sec=0.4, zone_id="Zone_A"),
    ]
    traj = Trajectory(track_id=42, points=pts)
    summaries = compute_zone_activity_summaries(traj)

    assert "Zone_A" in summaries
    act = summaries["Zone_A"]
    assert act.observed_run_count == 2
    assert act.distinct_track_count == 1
    assert act.direct_incoming_count == 0  # Entered from None both times
    assert act.direct_outgoing_count == 0  # Exited to None


def test_zone_activity_summary_multiple_clip_local_tracks():
    """Multiple distinct clip-local tracks visiting a zone increment distinct_track_count."""
    pts_1 = [
        _make_point(track_id=10, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
    ]
    pts_2 = [
        _make_point(track_id=20, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
    ]
    pts_3 = [
        _make_point(track_id=30, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_B"),
    ]
    trajs = [
        Trajectory(track_id=10, points=pts_1),
        Trajectory(track_id=20, points=pts_2),
        Trajectory(track_id=30, points=pts_3),
    ]

    summaries = compute_zone_activity_summaries(trajs)
    assert summaries["Zone_A"].observed_run_count == 2
    assert summaries["Zone_A"].distinct_track_count == 2
    assert summaries["Zone_B"].observed_run_count == 1
    assert summaries["Zone_B"].distinct_track_count == 1


def test_zone_activity_summary_incoming_and_outgoing_direct_transitions():
    """Verify exact incoming and outgoing direct transition counts for a multi-zone cycle."""
    # Track 1: A -> B -> C -> A
    pts = [
        _make_point(track_id=1, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
        _make_point(track_id=1, frame_idx=1, timestamp_sec=0.2, zone_id="Zone_B"),
        _make_point(track_id=1, frame_idx=2, timestamp_sec=0.4, zone_id="Zone_C"),
        _make_point(track_id=1, frame_idx=3, timestamp_sec=0.6, zone_id="Zone_A"),
    ]
    traj = Trajectory(track_id=1, points=pts)
    summaries = compute_zone_activity_summaries(traj)

    # Zone A: runs=2, distinct_tracks=1, incoming=1 (from C), outgoing=1 (to B)
    assert summaries["Zone_A"].observed_run_count == 2
    assert summaries["Zone_A"].distinct_track_count == 1
    assert summaries["Zone_A"].direct_incoming_count == 1
    assert summaries["Zone_A"].direct_outgoing_count == 1

    # Zone B: runs=1, distinct_tracks=1, incoming=1 (from A), outgoing=1 (to C)
    assert summaries["Zone_B"].observed_run_count == 1
    assert summaries["Zone_B"].distinct_track_count == 1
    assert summaries["Zone_B"].direct_incoming_count == 1
    assert summaries["Zone_B"].direct_outgoing_count == 1

    # Zone C: runs=1, distinct_tracks=1, incoming=1 (from B), outgoing=1 (to A)
    assert summaries["Zone_C"].observed_run_count == 1
    assert summaries["Zone_C"].distinct_track_count == 1
    assert summaries["Zone_C"].direct_incoming_count == 1
    assert summaries["Zone_C"].direct_outgoing_count == 1


def test_zone_activity_summary_none_breaks_transition_chain():
    """None intervening between zones results in zero direct incoming and outgoing transitions."""
    # A -> None -> B
    pts = [
        _make_point(track_id=5, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
        _make_point(track_id=5, frame_idx=1, timestamp_sec=0.2, zone_id=None),
        _make_point(track_id=5, frame_idx=2, timestamp_sec=0.4, zone_id="Zone_B"),
    ]
    traj = Trajectory(track_id=5, points=pts)
    summaries = compute_zone_activity_summaries(traj)

    assert summaries["Zone_A"].observed_run_count == 1
    assert summaries["Zone_A"].distinct_track_count == 1
    assert summaries["Zone_A"].direct_incoming_count == 0
    assert summaries["Zone_A"].direct_outgoing_count == 0

    assert summaries["Zone_B"].observed_run_count == 1
    assert summaries["Zone_B"].distinct_track_count == 1
    assert summaries["Zone_B"].direct_incoming_count == 0
    assert summaries["Zone_B"].direct_outgoing_count == 0


def test_zone_activity_summary_convenience_function_signatures():
    """Verify compute_zone_activity_summaries works across single, batch, and DirectTransitionSummary inputs."""
    pts = [
        _make_point(track_id=1, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
        _make_point(track_id=1, frame_idx=1, timestamp_sec=0.2, zone_id="Zone_B"),
    ]
    traj = Trajectory(track_id=1, points=pts)

    # 1. From single Trajectory
    res1 = compute_zone_activity_summaries(traj)
    assert "Zone_A" in res1 and "Zone_B" in res1

    # 2. From Sequence[Trajectory]
    res2 = compute_zone_activity_summaries([traj])
    assert res1["Zone_A"] == res2["Zone_A"]

    # 3. From already computed DirectTransitionSummary
    summary = compute_direct_transitions(traj)
    res3 = compute_zone_activity_summaries(summary)
    assert res1["Zone_A"] == res3["Zone_A"]

    # 4. From already computed DirectTransitionSummary with extra known_zone_ids
    res4 = compute_zone_activity_summaries(summary, known_zone_ids=["Zone_A", "Zone_B", "Zone_C"])
    assert "Zone_C" in res4
    assert res4["Zone_C"].observed_run_count == 0


def test_zone_activity_summary_serialization_round_trip():
    """Verify ZoneActivitySummary and DirectTransitionSummary to_dict() serialization."""
    act = ZoneActivitySummary(
        zone_id="Zone_Checkout",
        observed_run_count=5,
        distinct_track_count=3,
        direct_incoming_count=2,
        direct_outgoing_count=1,
    )
    d = act.to_dict()
    assert d == {
        "zone_id": "Zone_Checkout",
        "observed_run_count": 5,
        "distinct_track_count": 3,
        "direct_incoming_count": 2,
        "direct_outgoing_count": 1,
    }

    # Verify nested serialization in DirectTransitionSummary
    pts = [
        _make_point(track_id=1, frame_idx=0, timestamp_sec=0.0, zone_id="Zone_A"),
        _make_point(track_id=1, frame_idx=1, timestamp_sec=0.2, zone_id="Zone_B"),
    ]
    summary = compute_direct_transitions(Trajectory(track_id=1, points=pts))
    sum_dict = summary.to_dict()
    assert "zone_summaries" in sum_dict
    assert "Zone_A" in sum_dict["zone_summaries"]
    assert sum_dict["zone_summaries"]["Zone_A"]["observed_run_count"] == 1

    # Verify JSON encoding succeeds
    json_str = json.dumps(sum_dict)
    loaded = json.loads(json_str)
    assert loaded["zone_summaries"]["Zone_A"]["direct_outgoing_count"] == 1


def test_zone_activity_summary_validation_rejections():
    """Verify that invalid field types or negative values in ZoneActivitySummary are rejected."""
    # Negative observed_run_count
    with pytest.raises(ValueError, match="observed_run_count must be a non-negative integer"):
        ZoneActivitySummary(zone_id="A", observed_run_count=-1, distinct_track_count=0, direct_incoming_count=0, direct_outgoing_count=0)

    # Boolean distinct_track_count
    with pytest.raises(ValueError, match="distinct_track_count must be a non-negative integer"):
        ZoneActivitySummary(zone_id="A", observed_run_count=0, distinct_track_count=True, direct_incoming_count=0, direct_outgoing_count=0)  # type: ignore

    # Negative direct_incoming_count
    with pytest.raises(ValueError, match="direct_incoming_count must be a non-negative integer"):
        ZoneActivitySummary(zone_id="A", observed_run_count=0, distinct_track_count=0, direct_incoming_count=-2, direct_outgoing_count=0)

    # Negative direct_outgoing_count
    with pytest.raises(ValueError, match="direct_outgoing_count must be a non-negative integer"):
        ZoneActivitySummary(zone_id="A", observed_run_count=0, distinct_track_count=0, direct_incoming_count=0, direct_outgoing_count=-1)

    # Empty zone_id
    with pytest.raises(ValueError, match="Invalid zone_id"):
        ZoneActivitySummary(zone_id="  ", observed_run_count=0, distinct_track_count=0, direct_incoming_count=0, direct_outgoing_count=0)

