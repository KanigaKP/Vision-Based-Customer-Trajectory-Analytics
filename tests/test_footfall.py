"""
Phase 7 — Unit Tests for Footfall Analytics
==========================================
Validates discrete zone entry events, contiguous zone visits, basic visit duration,
unique tracks per zone, per-frame occupancy, and zone-to-zone transitions.
"""

import pytest
from src.trajectory.trajectory import Trajectory, TrajectoryPoint
from src.analytics.footfall import (
    FootfallAnalyzer,
    ZoneEntryEvent,
    ZoneVisit,
    FootfallSummary
)


def _make_trajectory(track_id: int, zone_sequence: list, start_frame: int = 0, fps: float = 5.0) -> Trajectory:
    """Helper to build a trajectory with pre-assigned zone_ids for testing."""
    traj = Trajectory(track_id=track_id)
    dt = 1.0 / fps
    for idx, z in enumerate(zone_sequence):
        f = start_frame + idx
        traj.add_point(TrajectoryPoint(
            track_id=track_id,
            frame_idx=f,
            timestamp_sec=round(f * dt, 3),
            image_x=100.0,
            image_y=200.0,
            floor_x=1.0,
            floor_y=2.0,
            zone_id=z
        ))
    return traj


def test_zone_entry_event_detection():
    """Test 1: Zone entry event detection when transitioning into a zone."""
    analyzer = FootfallAnalyzer(known_zone_ids=["shelf_a", "shelf_b"])
    # Trajectory moves: None -> shelf_a -> shelf_a
    traj = _make_trajectory(1, [None, "shelf_a", "shelf_a"])

    entries, visits, trans = analyzer.analyze_trajectory(traj)

    assert len(entries) == 1
    e = entries[0]
    assert e.track_id == 1
    assert e.zone_id == "shelf_a"
    assert e.from_zone_id is None
    assert e.frame_idx == 1
    assert e.timestamp_sec == 0.2


def test_zone_exit_produces_no_entry():
    """Test 2: Exiting from a zone to None produces no new entry event."""
    analyzer = FootfallAnalyzer(known_zone_ids=["shelf_a"])
    # Trajectory moves: shelf_a -> shelf_a -> None -> None
    traj = _make_trajectory(1, ["shelf_a", "shelf_a", None, None])

    entries, visits, trans = analyzer.analyze_trajectory(traj)

    # Only 1 entry (at index 0 when entering shelf_a)
    assert len(entries) == 1
    assert entries[0].zone_id == "shelf_a"
    # Transition to None should close the visit, but NOT add an entry for None
    assert len(visits) == 1
    assert visits[0].zone_id == "shelf_a"
    assert visits[0].exit_frame_idx == 1


def test_re_entry_counted_separately():
    """Test 3: Re-entering a zone (A -> B -> A) counts distinct entries and visits."""
    analyzer = FootfallAnalyzer(known_zone_ids=["shelf_a", "shelf_b"])
    # Trajectory moves: shelf_a -> shelf_b -> shelf_a
    traj = _make_trajectory(1, ["shelf_a", "shelf_b", "shelf_a"])

    entries, visits, trans = analyzer.analyze_trajectory(traj)

    assert len(entries) == 3
    assert [e.zone_id for e in entries] == ["shelf_a", "shelf_b", "shelf_a"]
    assert len(visits) == 3
    assert [v.zone_id for v in visits] == ["shelf_a", "shelf_b", "shelf_a"]

    # Overall summary check
    summary = analyzer.analyze([traj])
    assert summary.entry_events_per_zone["shelf_a"] == 2
    assert summary.entry_events_per_zone["shelf_b"] == 1
    assert summary.visits_per_zone["shelf_a"] == 2
    assert summary.visits_per_zone["shelf_b"] == 1
    assert summary.unique_tracks_per_zone["shelf_a"] == 1
    assert summary.unique_tracks_per_zone["shelf_b"] == 1


def test_zone_visit_contiguous_interval():
    """Test 4: Contiguous stay in a zone produces exactly one visit."""
    analyzer = FootfallAnalyzer(known_zone_ids=["shelf_a"])
    # Trajectory stays in shelf_a for 5 consecutive points at 5 FPS (0.0s to 0.8s)
    traj = _make_trajectory(1, ["shelf_a", "shelf_a", "shelf_a", "shelf_a", "shelf_a"])

    entries, visits, trans = analyzer.analyze_trajectory(traj)

    assert len(visits) == 1
    v = visits[0]
    assert v.track_id == 1
    assert v.zone_id == "shelf_a"
    assert v.entry_frame_idx == 0
    assert v.exit_frame_idx == 4
    assert v.entry_timestamp_sec == 0.0
    assert v.exit_timestamp_sec == 0.8
    assert v.point_count == 5


def test_unique_track_count_per_zone():
    """Test 5: Multiple distinct tracks visiting a zone are counted correctly."""
    analyzer = FootfallAnalyzer(known_zone_ids=["shelf_a", "shelf_b"])

    # Track 1 visits shelf_a twice (A -> B -> A)
    t1 = _make_trajectory(10, ["shelf_a", "shelf_b", "shelf_a"])
    # Track 2 visits shelf_a once (A -> None)
    t2 = _make_trajectory(20, ["shelf_a", None])
    # Track 3 visits only shelf_b
    t3 = _make_trajectory(30, ["shelf_b", "shelf_b"])

    summary = analyzer.analyze([t1, t2, t3])

    # Distinct tracks in shelf_a: track 10 and track 20 => 2
    assert summary.unique_tracks_per_zone["shelf_a"] == 2
    # Distinct tracks in shelf_b: track 10 and track 30 => 2
    assert summary.unique_tracks_per_zone["shelf_b"] == 2

    # Entries in shelf_a: 2 from t1 + 1 from t2 => 3
    assert summary.entry_events_per_zone["shelf_a"] == 3


def test_occupancy_snapshot_at_frame():
    """Test 6: Per-frame occupancy snapshot returns count of active tracks per zone."""
    analyzer = FootfallAnalyzer(known_zone_ids=["shelf_a", "shelf_b"])

    # Frame 0: t1 is in shelf_a, t2 is in shelf_b, t3 is in shelf_a
    # Frame 1: t1 is in shelf_a, t2 is in None, t3 is in shelf_b
    t1 = _make_trajectory(1, ["shelf_a", "shelf_a"], start_frame=0)
    t2 = _make_trajectory(2, ["shelf_b", None], start_frame=0)
    t3 = _make_trajectory(3, ["shelf_a", "shelf_b"], start_frame=0)

    trajs = [t1, t2, t3]

    occ_f0 = analyzer.get_occupancy_snapshot(trajs, frame_idx=0)
    assert occ_f0["shelf_a"] == 2  # t1 and t3
    assert occ_f0["shelf_b"] == 1  # t2

    occ_f1 = analyzer.get_occupancy_snapshot(trajs, frame_idx=1)
    assert occ_f1["shelf_a"] == 1  # t1
    assert occ_f1["shelf_b"] == 1  # t3


def test_trajectory_with_no_zone_assignments():
    """Test 7: Trajectory entirely outside defined zones produces zero metrics."""
    analyzer = FootfallAnalyzer(known_zone_ids=["shelf_a", "shelf_b"])
    traj = _make_trajectory(1, [None, None, None])

    entries, visits, trans = analyzer.analyze_trajectory(traj)
    assert len(entries) == 0
    assert len(visits) == 0
    assert len(trans) == 0

    summary = analyzer.analyze([traj])
    assert summary.unique_tracks_per_zone["shelf_a"] == 0
    assert summary.unique_tracks_per_zone["shelf_b"] == 0
    assert summary.entry_events_per_zone["shelf_a"] == 0
    assert summary.visits_per_zone["shelf_a"] == 0
    assert summary.total_visit_duration_sec["shelf_a"] == 0.0


def test_single_point_trajectory():
    """Test 8: Single-point trajectory handles inside and outside cases gracefully."""
    analyzer = FootfallAnalyzer(known_zone_ids=["shelf_a"])

    # Single point inside zone
    traj_inside = _make_trajectory(1, ["shelf_a"])
    entries, visits, trans = analyzer.analyze_trajectory(traj_inside)
    assert len(entries) == 1
    assert entries[0].zone_id == "shelf_a"
    assert len(visits) == 1
    assert visits[0].point_count == 1
    assert visits[0].duration_sec == 0.0

    # Single point outside zone (None)
    traj_outside = _make_trajectory(2, [None])
    entries_out, visits_out, trans_out = analyzer.analyze_trajectory(traj_outside)
    assert len(entries_out) == 0
    assert len(visits_out) == 0


def test_basic_visit_duration():
    """Test 9: Basic visit duration calculations across multiple visits."""
    analyzer = FootfallAnalyzer(known_zone_ids=["shelf_a"])

    # Visit 1: 3 frames (0.0s to 0.4s) -> duration = 0.4s
    # Visit 2: 4 frames (1.0s to 1.6s) -> duration = 0.6s
    t1 = _make_trajectory(1, ["shelf_a", "shelf_a", "shelf_a"], start_frame=0)
    t2 = _make_trajectory(2, ["shelf_a", "shelf_a", "shelf_a", "shelf_a"], start_frame=5)

    summary = analyzer.analyze([t1, t2])

    assert summary.visits_per_zone["shelf_a"] == 2
    assert summary.total_visit_duration_sec["shelf_a"] == 1.0  # 0.4 + 0.6
    assert summary.average_visit_duration_sec["shelf_a"] == 0.5  # 1.0 / 2


def test_zone_to_zone_transition_matrix():
    """Test 10: Zone-to-zone transitions are aggregated into the transition matrix."""
    analyzer = FootfallAnalyzer(known_zone_ids=["shelf_a", "main_aisle", "checkout"])

    # Trajectory 1: None -> shelf_a -> shelf_a -> main_aisle -> main_aisle -> None
    t1 = _make_trajectory(1, [None, "shelf_a", "shelf_a", "main_aisle", "main_aisle", None])
    # Trajectory 2: main_aisle -> checkout
    t2 = _make_trajectory(2, ["main_aisle", "checkout"])

    summary = analyzer.analyze([t1, t2])

    assert summary.transition_matrix["shelf_a"]["main_aisle"] == 1
    assert summary.transition_matrix["main_aisle"]["checkout"] == 1
    assert summary.transition_matrix["shelf_a"]["checkout"] == 0
    assert summary.transition_matrix["checkout"]["shelf_a"] == 0


def test_occupancy_timeline():
    """Test 11: Occupancy timeline returns mapping across frames."""
    analyzer = FootfallAnalyzer(known_zone_ids=["shelf_a", "shelf_b"])
    t1 = _make_trajectory(1, ["shelf_a", "shelf_b"], start_frame=0)

    timeline = analyzer.get_occupancy_timeline([t1])
    assert 0 in timeline
    assert 1 in timeline
    assert timeline[0]["shelf_a"] == 1
    assert timeline[0]["shelf_b"] == 0
    assert timeline[1]["shelf_a"] == 0
    assert timeline[1]["shelf_b"] == 1
