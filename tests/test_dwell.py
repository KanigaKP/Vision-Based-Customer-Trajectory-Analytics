"""
Phase 9.1 — Unit & Integration Tests for Refined Dwell Analytics Foundation.
===========================================================================
Validates per-track, per-zone dwell episodes and basic temporal summaries:
1. Single visit in a zone.
2. Repeated visits to the same zone (separate episodes, not merged).
3. Transitions between zones (closing previous, opening new).
4. Outside/unknown-zone breaks (None breaks the episode; does not bridge across).
5. Multiple clip-local tracks (independent tracking, no re-identification).
6. Duration calculations (observed timestamps only, single-point 0.0s).
7. Invalid or non-monotonic timestamp handling (strict rejection with ValueError).
8. Empty trajectories and trajectories entirely outside defined zones.
9. DwellAnalyzer and convenience functions with known_zone_ids zero-padding.
10. Serialization round-trips via to_dict().
"""

import math
import pytest
from src.trajectory.trajectory import Trajectory, TrajectoryPoint
from src.analytics.dwell import (
    DwellEpisode,
    TrackDwellSummary,
    ZoneDwellSummary,
    ZoneSpanDistribution,
    DwellSummary,
    DwellAnalyzer,
    extract_dwell_episodes,
    analyze_dwell,
    compute_zone_span_distribution,
)


def _make_trajectory(
    track_id: int,
    zone_sequence: list,
    start_frame: int = 0,
    fps: float = 5.0,
    start_time_sec: float = 0.0,
) -> Trajectory:
    """Helper to construct a valid Trajectory with specified zone assignments."""
    traj = Trajectory(track_id=track_id)
    dt = 1.0 / fps
    for idx, z in enumerate(zone_sequence):
        f = start_frame + idx
        t = round(start_time_sec + idx * dt, 3)
        traj.add_point(
            TrajectoryPoint(
                track_id=track_id,
                frame_idx=f,
                timestamp_sec=t,
                image_x=100.0,
                image_y=200.0,
                floor_x=1.0,
                floor_y=2.0,
                zone_id=z,
            )
        )
    return traj


# ==============================================================================
# 1. Single Visit in a Zone
# ==============================================================================
def test_single_visit():
    """Test 1: Single continuous stay in a zone produces exactly one DwellEpisode."""
    traj = _make_trajectory(1, ["shelf_a", "shelf_a", "shelf_a", "shelf_a"], start_frame=10, fps=5.0)
    # 4 points at 5 FPS: f=10 (0.0s), f=11 (0.2s), f=12 (0.4s), f=13 (0.6s)
    episodes = extract_dwell_episodes(traj)

    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.track_id == 1
    assert ep.zone_id == "shelf_a"
    assert ep.entry_frame_idx == 10
    assert ep.exit_frame_idx == 13
    assert ep.entry_timestamp_sec == 0.0
    assert ep.exit_timestamp_sec == 0.6
    assert ep.duration_sec == 0.6
    assert ep.point_count == 4


# ==============================================================================
# 2. Repeated Visits to the Same Zone
# ==============================================================================
def test_repeated_visits_to_same_zone():
    """Test 2: Multiple separate visits to the same zone produce distinct episodes (never merged)."""
    # Sequence: shelf_a (2 frames) -> shelf_b (1 frame) -> shelf_a (2 frames)
    traj = _make_trajectory(1, ["shelf_a", "shelf_a", "shelf_b", "shelf_a", "shelf_a"], start_frame=0, fps=5.0)
    episodes = extract_dwell_episodes(traj)

    assert len(episodes) == 3
    assert [e.zone_id for e in episodes] == ["shelf_a", "shelf_b", "shelf_a"]

    # First visit to shelf_a: frames 0..1, duration 0.2s
    assert episodes[0].entry_frame_idx == 0
    assert episodes[0].exit_frame_idx == 1
    assert episodes[0].duration_sec == 0.2
    assert episodes[0].point_count == 2

    # Intermediate visit to shelf_b: frame 2, duration 0.0s
    assert episodes[1].entry_frame_idx == 2
    assert episodes[1].exit_frame_idx == 2
    assert episodes[1].duration_sec == 0.0
    assert episodes[1].point_count == 1

    # Second visit to shelf_a: frames 3..4, duration 0.2s
    assert episodes[2].entry_frame_idx == 3
    assert episodes[2].exit_frame_idx == 4
    assert episodes[2].duration_sec == 0.2
    assert episodes[2].point_count == 2

    # Per-track summary should have 2 episodes for shelf_a and 1 for shelf_b
    analyzer = DwellAnalyzer()
    track_summary = analyzer.analyze_track(traj)
    assert track_summary.total_episodes == 3
    assert track_summary.zone_episode_count["shelf_a"] == 2
    assert track_summary.zone_episode_count["shelf_b"] == 1
    assert track_summary.zone_dwell_sec["shelf_a"] == 0.4
    assert track_summary.zone_dwell_sec["shelf_b"] == 0.0
    assert track_summary.total_dwell_sec == 0.4


# ==============================================================================
# 3. Transitions Between Zones
# ==============================================================================
def test_transitions_between_zones():
    """Test 3: Direct transitions between zones close previous and start new episode."""
    # Sequence: zone_1 -> zone_2 -> zone_3
    traj = _make_trajectory(
        42,
        ["zone_1", "zone_1", "zone_2", "zone_2", "zone_2", "zone_3"],
        start_frame=100,
        fps=10.0,
    )
    # 10 FPS => dt = 0.1s
    episodes = extract_dwell_episodes(traj)

    assert len(episodes) == 3
    assert [e.zone_id for e in episodes] == ["zone_1", "zone_2", "zone_3"]

    # zone_1: f=100, 101 => 0.0s to 0.1s => 0.1s
    assert episodes[0].entry_frame_idx == 100
    assert episodes[0].exit_frame_idx == 101
    assert episodes[0].duration_sec == 0.1
    assert episodes[0].point_count == 2

    # zone_2: f=102, 103, 104 => 0.2s to 0.4s => 0.2s
    assert episodes[1].entry_frame_idx == 102
    assert episodes[1].exit_frame_idx == 104
    assert episodes[1].duration_sec == 0.2
    assert episodes[1].point_count == 3

    # zone_3: f=105 => 0.5s to 0.5s => 0.0s
    assert episodes[2].entry_frame_idx == 105
    assert episodes[2].exit_frame_idx == 105
    assert episodes[2].duration_sec == 0.0
    assert episodes[2].point_count == 1


# ==============================================================================
# 4. Outside / Unknown-Zone Breaks
# ==============================================================================
def test_outside_zone_breaks_episode():
    """Test 4: An observation outside defined zones (None) breaks an episode and does not bridge."""
    # Sequence: shelf_a (3 frames) -> None (2 frames) -> shelf_a (3 frames)
    traj = _make_trajectory(
        5,
        ["shelf_a", "shelf_a", "shelf_a", None, None, "shelf_a", "shelf_a", "shelf_a"],
        start_frame=0,
        fps=5.0,
    )
    # dt = 0.2s
    episodes = extract_dwell_episodes(traj)

    # Must produce 2 separate episodes for shelf_a, NOT one bridged episode
    assert len(episodes) == 2
    assert episodes[0].zone_id == "shelf_a"
    assert episodes[1].zone_id == "shelf_a"

    # Episode 1: f=0..2, t=0.0..0.4, dur=0.4
    assert episodes[0].entry_frame_idx == 0
    assert episodes[0].exit_frame_idx == 2
    assert episodes[0].entry_timestamp_sec == 0.0
    assert episodes[0].exit_timestamp_sec == 0.4
    assert episodes[0].duration_sec == 0.4
    assert episodes[0].point_count == 3

    # Episode 2: f=5..7, t=1.0..1.4, dur=0.4
    assert episodes[1].entry_frame_idx == 5
    assert episodes[1].exit_frame_idx == 7
    assert episodes[1].entry_timestamp_sec == 1.0
    assert episodes[1].exit_timestamp_sec == 1.4
    assert episodes[1].duration_sec == 0.4
    assert episodes[1].point_count == 3

    # Total observed dwell does NOT include the unassigned gap (t=0.4 to 1.0)
    analyzer = DwellAnalyzer()
    track_summary = analyzer.analyze_track(traj)
    assert track_summary.total_dwell_sec == 0.8  # 0.4 + 0.4, gap of 0.6s is excluded


# ==============================================================================
# 5. Multiple Clip-Local Tracks
# ==============================================================================
def test_multiple_clip_local_tracks():
    """Test 5: Multiple tracks are analyzed independently with no cross-track ID merging."""
    # Track 10: visits shelf_a (0.4s) then shelf_b (0.2s)
    t1 = _make_trajectory(10, ["shelf_a", "shelf_a", "shelf_a", "shelf_b", "shelf_b"], fps=5.0)
    # Track 20: visits shelf_a (0.2s) only
    t2 = _make_trajectory(20, ["shelf_a", "shelf_a"], fps=5.0)
    # Track 30: visits shelf_b (0.6s) only
    t3 = _make_trajectory(30, ["shelf_b", "shelf_b", "shelf_b", "shelf_b"], fps=5.0)

    analyzer = DwellAnalyzer(known_zone_ids=["shelf_a", "shelf_b", "checkout"])
    summary = analyzer.analyze([t1, t2, t3])

    # Per-track verification
    assert len(summary.by_track) == 3
    assert summary.by_track[10].track_id == 10
    assert summary.by_track[10].total_dwell_sec == 0.6  # 0.4 (shelf_a) + 0.2 (shelf_b)
    assert summary.by_track[10].zone_episode_count["shelf_a"] == 1
    assert summary.by_track[10].zone_episode_count["shelf_b"] == 1

    assert summary.by_track[20].track_id == 20
    assert summary.by_track[20].total_dwell_sec == 0.2
    assert summary.by_track[20].zone_episode_count["shelf_a"] == 1

    assert summary.by_track[30].track_id == 30
    assert summary.by_track[30].total_dwell_sec == 0.6
    assert summary.by_track[30].zone_episode_count["shelf_b"] == 1

    # Per-zone verification
    assert len(summary.by_zone) == 3  # shelf_a, shelf_b, checkout
    # shelf_a: track 10 (0.4s) + track 20 (0.2s) = 0.6s total, 2 episodes, 2 unique tracks
    za = summary.by_zone["shelf_a"]
    assert za.total_episodes == 2
    assert za.unique_tracks == 2
    assert za.total_dwell_sec == 0.6
    assert za.mean_dwell_sec == 0.3
    assert za.min_dwell_sec == 0.2
    assert za.max_dwell_sec == 0.4

    # shelf_b: track 10 (0.2s) + track 30 (0.6s) = 0.8s total, 2 episodes, 2 unique tracks
    zb = summary.by_zone["shelf_b"]
    assert zb.total_episodes == 2
    assert zb.unique_tracks == 2
    assert zb.total_dwell_sec == 0.8
    assert zb.mean_dwell_sec == 0.4
    assert zb.min_dwell_sec == 0.2
    assert zb.max_dwell_sec == 0.6

    # checkout: zero visits
    zc = summary.by_zone["checkout"]
    assert zc.total_episodes == 0
    assert zc.unique_tracks == 0
    assert zc.total_dwell_sec == 0.0
    assert zc.mean_dwell_sec == 0.0
    assert zc.min_dwell_sec == 0.0
    assert zc.max_dwell_sec == 0.0

    # Total episodes across all tracks
    assert summary.total_episodes == 4
    assert summary.total_dwell_sec == 1.4  # 0.6 + 0.8


# ==============================================================================
# 6. Duration Calculation & Sampling Semantics
# ==============================================================================
def test_duration_calculation_single_and_multi_point():
    """Test 6: Single-observation duration is 0.0s; multi-point is exit - entry without extrapolation."""
    # Single point episode
    traj_single = _make_trajectory(1, ["shelf_a"], start_frame=50, start_time_sec=10.0)
    eps_single = extract_dwell_episodes(traj_single)
    assert len(eps_single) == 1
    assert eps_single[0].entry_timestamp_sec == 10.0
    assert eps_single[0].exit_timestamp_sec == 10.0
    assert eps_single[0].duration_sec == 0.0
    assert eps_single[0].point_count == 1

    # Two point episode at dt = 0.5s
    traj_two = _make_trajectory(2, ["shelf_b", "shelf_b"], start_frame=0, fps=2.0, start_time_sec=1.0)
    eps_two = extract_dwell_episodes(traj_two)
    assert len(eps_two) == 1
    assert eps_two[0].entry_timestamp_sec == 1.0
    assert eps_two[0].exit_timestamp_sec == 1.5
    assert eps_two[0].duration_sec == 0.5
    assert eps_two[0].point_count == 2


# ==============================================================================
# 7. Invalid or Non-Monotonic Timestamp Handling
# ==============================================================================
def test_invalid_and_non_monotonic_timestamps():
    """Test 7: Malformed or non-monotonic time data fails clearly and raises ValueError."""
    # 7a. Non-monotonic timestamp (decreasing time)
    traj_non_mono = Trajectory(track_id=1)
    traj_non_mono.add_point(
        TrajectoryPoint(1, 0, 1.0, 10.0, 20.0, zone_id="shelf_a")
    )
    traj_non_mono.add_point(
        TrajectoryPoint(1, 1, 0.5, 10.0, 20.0, zone_id="shelf_a")
    )
    with pytest.raises(ValueError, match="Non-monotonic timestamp"):
        extract_dwell_episodes(traj_non_mono)

    # 7b. Negative timestamp
    traj_neg_time = Trajectory(track_id=1)
    traj_neg_time.add_point(
        TrajectoryPoint(1, 0, -0.5, 10.0, 20.0, zone_id="shelf_a")
    )
    with pytest.raises(ValueError, match="timestamp_sec must be non-negative"):
        extract_dwell_episodes(traj_neg_time)

    # 7c. NaN timestamp
    traj_nan_time = Trajectory(track_id=1)
    traj_nan_time.add_point(
        TrajectoryPoint(1, 0, float("nan"), 10.0, 20.0, zone_id="shelf_a")
    )
    with pytest.raises(ValueError, match="timestamp_sec must be a finite number"):
        extract_dwell_episodes(traj_nan_time)

    # 7d. Infinite timestamp
    traj_inf_time = Trajectory(track_id=1)
    traj_inf_time.add_point(
        TrajectoryPoint(1, 0, float("inf"), 10.0, 20.0, zone_id="shelf_a")
    )
    with pytest.raises(ValueError, match="timestamp_sec must be a finite number"):
        extract_dwell_episodes(traj_inf_time)

    # 7e. Boolean timestamp (disallowed)
    traj_bool_time = Trajectory(track_id=1)
    traj_bool_time.add_point(
        TrajectoryPoint(1, 0, True, 10.0, 20.0, zone_id="shelf_a")  # type: ignore
    )
    with pytest.raises(ValueError, match="timestamp_sec must be a finite number"):
        extract_dwell_episodes(traj_bool_time)

    # 7f. Negative frame_idx
    traj_neg_frame = Trajectory(track_id=1)
    traj_neg_frame.add_point(
        TrajectoryPoint(1, -1, 0.0, 10.0, 20.0, zone_id="shelf_a")
    )
    with pytest.raises(ValueError, match="frame_idx must be non-negative"):
        extract_dwell_episodes(traj_neg_frame)

    # 7g. Mismatched track_id in point
    traj_mismatch = Trajectory(track_id=1)
    traj_mismatch.add_point(
        TrajectoryPoint(2, 0, 0.0, 10.0, 20.0, zone_id="shelf_a")
    )
    with pytest.raises(ValueError, match="Point track_id .* does not match"):
        extract_dwell_episodes(traj_mismatch)

    # 7h. Invalid empty string zone_id
    traj_empty_zone = Trajectory(track_id=1)
    traj_empty_zone.add_point(
        TrajectoryPoint(1, 0, 0.0, 10.0, 20.0, zone_id="")
    )
    with pytest.raises(ValueError, match="zone_id cannot be an empty or whitespace string"):
        extract_dwell_episodes(traj_empty_zone)


# ==============================================================================
# 8. Empty & Unassigned Trajectory Edge Cases
# ==============================================================================
def test_empty_and_unassigned_trajectories():
    """Test 8: Trajectories with 0 points or entirely outside defined zones produce 0 episodes."""
    # 8a. Empty trajectory
    empty_traj = Trajectory(track_id=99)
    eps_empty = extract_dwell_episodes(empty_traj)
    assert eps_empty == []

    analyzer = DwellAnalyzer()
    track_summary = analyzer.analyze_track(empty_traj)
    assert track_summary.total_episodes == 0
    assert track_summary.total_dwell_sec == 0.0
    assert track_summary.episodes == []

    # 8b. Trajectory entirely outside defined zones (all None)
    outside_traj = _make_trajectory(100, [None, None, None, None])
    eps_outside = extract_dwell_episodes(outside_traj)
    assert eps_outside == []

    summary_outside = analyzer.analyze([outside_traj])
    assert summary_outside.total_episodes == 0
    assert summary_outside.total_dwell_sec == 0.0
    assert summary_outside.by_track[100].total_episodes == 0
    assert summary_outside.by_track[100].total_dwell_sec == 0.0


# ==============================================================================
# 9. Serialization Round-Trip
# ==============================================================================
def test_serialization_to_dict():
    """Test 9: All dwell dataclasses support complete JSON/dictionary serialization."""
    traj = _make_trajectory(7, ["zone_a", "zone_a", "zone_b"], fps=5.0)
    analyzer = DwellAnalyzer(known_zone_ids=["zone_a", "zone_b", "zone_c"])
    summary = analyzer.analyze([traj])

    # Convert to dict
    summary_dict = summary.to_dict()

    assert summary_dict["total_episodes"] == 2
    assert summary_dict["total_dwell_sec"] == 0.2  # zone_a (0.2s) + zone_b (0.0s)
    assert 7 in summary_dict["by_track"]
    assert "zone_a" in summary_dict["by_zone"]
    assert "zone_b" in summary_dict["by_zone"]
    assert "zone_c" in summary_dict["by_zone"]

    # Verify episode structure in dict
    ep0_dict = summary_dict["episodes"][0]
    assert ep0_dict["track_id"] == 7
    assert ep0_dict["zone_id"] == "zone_a"
    assert ep0_dict["duration_sec"] == 0.2
    assert ep0_dict["point_count"] == 2


# ==============================================================================
# 10. Convenience Function analyze_dwell
# ==============================================================================
def test_convenience_function():
    """Test 10: Convenience function analyze_dwell produces identical results to DwellAnalyzer."""
    t1 = _make_trajectory(1, ["shelf_a", "shelf_a"], fps=5.0)
    t2 = _make_trajectory(2, ["shelf_b", "shelf_b", "shelf_b"], fps=5.0)

    summary = analyze_dwell([t1, t2], known_zone_ids=["shelf_a", "shelf_b"])

    assert summary.total_episodes == 2
    assert summary.by_track[1].zone_dwell_sec["shelf_a"] == 0.2
    assert summary.by_track[2].zone_dwell_sec["shelf_b"] == 0.4
    assert summary.by_zone["shelf_a"].unique_tracks == 1
    assert summary.by_zone["shelf_b"].unique_tracks == 1


# ==============================================================================
# 11. Widely Separated Observations Joined as Observed Span
# ==============================================================================
def test_widely_separated_observations_same_zone_joined_as_span():
    """
    Test 11: Widely separated observations of the same track in the same zone
    without intervening breaks are joined into a single episode whose duration
    is the observed span (without applying an arbitrary gap threshold).
    """
    traj = Trajectory(track_id=5)
    # Point at frame 0 (t=0.0) and point at frame 100 (t=20.0), with no intervening points
    traj.add_point(TrajectoryPoint(5, 0, 0.0, 100.0, 200.0, zone_id="shelf_a"))
    traj.add_point(TrajectoryPoint(5, 100, 20.0, 100.0, 200.0, zone_id="shelf_a"))

    episodes = extract_dwell_episodes(traj)

    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.track_id == 5
    assert ep.zone_id == "shelf_a"
    assert ep.entry_frame_idx == 0
    assert ep.exit_frame_idx == 100
    assert ep.entry_timestamp_sec == 0.0
    assert ep.exit_timestamp_sec == 20.0
    assert ep.duration_sec == 20.0
    assert ep.point_count == 2


# ==============================================================================
# 12. Descriptive Span Distribution — Odd Count Median
# ==============================================================================
def test_zone_span_distribution_odd_count():
    """Test 12: Odd episode count yields middle element as deterministic median."""
    # 3 episodes in shelf_a with durations 0.2s, 0.4s, 0.8s (fps=5.0 => dt=0.2s)
    # Track 1: 2 frames => 0.2s
    t1 = _make_trajectory(1, ["shelf_a", "shelf_a"], start_frame=0, fps=5.0)
    # Track 2: 3 frames => 0.4s
    t2 = _make_trajectory(2, ["shelf_a", "shelf_a", "shelf_a"], start_frame=10, fps=5.0)
    # Track 3: 5 frames => 0.8s
    t3 = _make_trajectory(3, ["shelf_a", "shelf_a", "shelf_a", "shelf_a", "shelf_a"], start_frame=20, fps=5.0)

    summary = analyze_dwell([t1, t2, t3], known_zone_ids=["shelf_a"])
    za = summary.by_zone["shelf_a"]

    # Spans are [0.2, 0.4, 0.8]
    # Median is 0.4, Mean is (0.2 + 0.4 + 0.8)/3 = 0.467, Min is 0.2, Max is 0.8
    assert za.episode_count == 3
    assert za.total_episodes == 3
    assert za.min_dwell_sec == 0.2
    assert za.median_dwell_sec == 0.4
    assert za.mean_dwell_sec == 0.467
    assert za.max_dwell_sec == 0.8

    # Verify structured distribution property
    dist = za.distribution
    assert isinstance(dist, ZoneSpanDistribution)
    assert dist.episode_count == 3
    assert dist.min_span_sec == 0.2
    assert dist.median_span_sec == 0.4
    assert dist.mean_span_sec == 0.467
    assert dist.max_span_sec == 0.8


# ==============================================================================
# 13. Descriptive Span Distribution — Even Count Median
# ==============================================================================
def test_zone_span_distribution_even_count():
    """Test 13: Even episode count yields arithmetic mean of two middle elements."""
    # 4 episodes in shelf_b with spans 0.1s, 0.3s, 0.5s, 0.7s (at 10 FPS => dt=0.1s)
    t1 = _make_trajectory(1, ["shelf_b", "shelf_b"], start_frame=0, fps=10.0)             # dur: 0.1
    t2 = _make_trajectory(2, ["shelf_b", "shelf_b", "shelf_b", "shelf_b"], start_frame=10, fps=10.0) # dur: 0.3
    t3 = _make_trajectory(3, ["shelf_b"] * 6, start_frame=20, fps=10.0)                  # dur: 0.5
    t4 = _make_trajectory(4, ["shelf_b"] * 8, start_frame=30, fps=10.0)                  # dur: 0.7

    summary = analyze_dwell([t1, t2, t3, t4], known_zone_ids=["shelf_b"])
    zb = summary.by_zone["shelf_b"]

    # Spans are [0.1, 0.3, 0.5, 0.7]
    # Median is (0.3 + 0.5) / 2 = 0.4, Mean is (0.1 + 0.3 + 0.5 + 0.7) / 4 = 0.4
    assert zb.episode_count == 4
    assert zb.min_dwell_sec == 0.1
    assert zb.median_dwell_sec == 0.4
    assert zb.mean_dwell_sec == 0.4
    assert zb.max_dwell_sec == 0.7


# ==============================================================================
# 14. Descriptive Span Distribution — Empty Zone Group
# ==============================================================================
def test_zone_span_distribution_empty_group():
    """Test 14: Unvisited zone returns 0 episodes and 0.0 for all distribution metrics."""
    # Trajectory visits shelf_a only; shelf_empty has 0 visits
    t1 = _make_trajectory(1, ["shelf_a", "shelf_a"], fps=5.0)
    summary = analyze_dwell([t1], known_zone_ids=["shelf_a", "shelf_empty"])

    z_empty = summary.by_zone["shelf_empty"]
    assert z_empty.episode_count == 0
    assert z_empty.total_episodes == 0
    assert z_empty.unique_tracks == 0
    assert z_empty.total_dwell_sec == 0.0
    assert z_empty.min_dwell_sec == 0.0
    assert z_empty.median_dwell_sec == 0.0
    assert z_empty.mean_dwell_sec == 0.0
    assert z_empty.max_dwell_sec == 0.0
    assert z_empty.distribution.episode_count == 0
    assert z_empty.distribution.min_span_sec == 0.0
    assert z_empty.distribution.median_span_sec == 0.0
    assert z_empty.distribution.mean_span_sec == 0.0
    assert z_empty.distribution.max_span_sec == 0.0


# ==============================================================================
# 15. Single-Point Zero Spans Included in Distribution
# ==============================================================================
def test_zone_span_distribution_single_point_zero_spans():
    """Test 15: Single-point episodes (0.0s span) are included in distribution calculations."""
    # Episode 1: single point (span = 0.0s)
    # Episode 2: 3 points at 5 FPS (span = 0.4s)
    t1 = _make_trajectory(1, ["zone_x"], start_frame=0, fps=5.0)
    t2 = _make_trajectory(2, ["zone_x", "zone_x", "zone_x"], start_frame=10, fps=5.0)

    summary = analyze_dwell([t1, t2], known_zone_ids=["zone_x"])
    zx = summary.by_zone["zone_x"]

    # Spans: [0.0, 0.4] => median: 0.2, mean: 0.2, min: 0.0, max: 0.4
    assert zx.episode_count == 2
    assert zx.min_dwell_sec == 0.0
    assert zx.median_dwell_sec == 0.2
    assert zx.mean_dwell_sec == 0.2
    assert zx.max_dwell_sec == 0.4

    # Disambiguation between observed zero-span visit and unvisited zone:
    # Trajectory 3 visits zone_single with exactly 1 point (observed span = 0.0s)
    # zone_unvisited has no visits at all
    t3 = _make_trajectory(3, ["zone_single"], start_frame=20, fps=5.0)
    summary_disambig = analyze_dwell([t3], known_zone_ids=["zone_single", "zone_unvisited"])

    z_single = summary_disambig.by_zone["zone_single"]
    z_unvisited = summary_disambig.by_zone["zone_unvisited"]

    # Both yield 0.0s for span statistics, but episode_count distinguishes them unambiguously
    assert z_single.episode_count == 1
    assert z_single.total_episodes == 1
    assert len(z_single.episodes) == 1
    assert z_single.min_dwell_sec == 0.0
    assert z_single.median_dwell_sec == 0.0
    assert z_single.mean_dwell_sec == 0.0
    assert z_single.max_dwell_sec == 0.0
    assert z_single.distribution.episode_count == 1

    assert z_unvisited.episode_count == 0
    assert z_unvisited.total_episodes == 0
    assert len(z_unvisited.episodes) == 0
    assert z_unvisited.min_dwell_sec == 0.0
    assert z_unvisited.median_dwell_sec == 0.0
    assert z_unvisited.mean_dwell_sec == 0.0
    assert z_unvisited.max_dwell_sec == 0.0
    assert z_unvisited.distribution.episode_count == 0


# ==============================================================================
# 16. Standalone compute_zone_span_distribution & Serialization Round-Trip
# ==============================================================================
def test_standalone_distribution_and_serialization():
    """Test 16: Standalone helper computes distribution and to_dict includes distribution data."""
    # Construct synthetic DwellEpisode list directly
    episodes = [
        DwellEpisode(1, "shelf_c", 0, 1, 0.0, 0.2, 0.2, 2),
        DwellEpisode(2, "shelf_c", 10, 12, 2.0, 2.4, 0.4, 3),
        DwellEpisode(3, "shelf_c", 20, 23, 4.0, 4.6, 0.6, 4),
    ]

    dist = compute_zone_span_distribution(episodes)
    assert dist.episode_count == 3
    assert dist.min_span_sec == 0.2
    assert dist.median_span_sec == 0.4
    assert dist.mean_span_sec == 0.4
    assert dist.max_span_sec == 0.6

    # Test empty episode list
    empty_dist = compute_zone_span_distribution([])
    assert empty_dist.episode_count == 0
    assert empty_dist.median_span_sec == 0.0

    # Test serialization via ZoneDwellSummary.to_dict()
    traj = _make_trajectory(1, ["shelf_c", "shelf_c"], fps=5.0)
    summary = analyze_dwell([traj], known_zone_ids=["shelf_c"])
    d = summary.by_zone["shelf_c"].to_dict()

    assert "median_dwell_sec" in d
    assert "distribution" in d
    assert d["distribution"]["episode_count"] == 1
    assert d["distribution"]["median_span_sec"] == 0.2
    assert d["distribution"]["min_span_sec"] == 0.2
    assert d["distribution"]["max_span_sec"] == 0.2


def test_dwell_with_missing_image_coordinates():
    """Verify that points with missing image coordinates compute exact dwell without inventing zeros or raising errors."""
    traj = Trajectory(track_id=1)
    # 4 points at 5 FPS -> timestamps 0.0, 0.2, 0.4, 0.6 -> span 0.6s
    for idx, t in enumerate([0.0, 0.2, 0.4, 0.6]):
        traj.add_point(
            TrajectoryPoint(
                track_id=1,
                frame_idx=idx,
                timestamp_sec=t,
                image_x=None,
                image_y=None,
                floor_x=1.5,
                floor_y=2.5,
                zone_id="showcase_display_front",
            )
        )
    summary = analyze_dwell([traj], known_zone_ids=["showcase_display_front"])
    zone_summary = summary.by_zone["showcase_display_front"]
    assert zone_summary.total_episodes == 1
    assert len(zone_summary.episodes) == 1
    assert zone_summary.total_dwell_sec == 0.6
    assert zone_summary.unique_tracks == 1



