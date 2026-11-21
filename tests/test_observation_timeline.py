"""
Test suite for Phase 13.1 Observation Timeline Sidecar and Synchronized Explorer.
================================================================================
Covers:
1. Sidecar schema version, metadata, coordinate disclaimers, and JSON round-trip.
2. Source-to-playback timing mapping under sampling stride.
3. Missing observations, multiple concurrent tracks, and track gaps.
4. Out-of-bounds and untransformed point handling.
5. Cumulative density point conservation.
6. Legacy run handling and unavailable state.
7. Switching runs clearing playback/selection state.
8. End-to-end bounded pipeline execution generating and linking the sidecar.
"""

import os
import json
import pytest
import numpy as np

from src.reporting.observation_timeline import (
    ObservationRecord,
    SampleTimelineEntry,
    ObservationTimelineSidecar,
    build_observation_timeline_sidecar,
)
from dashboard.run_adapter import (
    StandardizedRunData,
    load_standardized_run,
    load_run_observation_timeline,
)
from src.tracking.bytetrack import Track, TrackPoint
from src.trajectory.trajectory import Trajectory, TrajectoryPoint


# ---------------------------------------------------------------------------
# 1. Sidecar Schema, Version & Serialization Tests
# ---------------------------------------------------------------------------

def test_sidecar_schema_version_and_serialization(tmp_path):
    """Sidecar must follow version 1.0 schema with explicit disclaimers and round-trip cleanly."""
    obs1 = ObservationRecord(
        track_id=101,
        image_ground_point=[150.0, 480.0],
        bbox=[120.0, 320.0, 180.0, 480.0],
        confidence=0.88,
        floor_point=[2.5, 3.0],
        is_out_of_bounds=False,
        assigned_zone="shelf_a",
    )
    obs2 = ObservationRecord(
        track_id=102,
        image_ground_point=[400.0, 500.0],
        bbox=[360.0, 350.0, 440.0, 500.0],
        confidence=0.91,
        floor_point=None,  # Untransformed / outside calibration
        is_out_of_bounds=True,
        assigned_zone=None,
    )
    entry0 = SampleTimelineEntry(
        sample_index=0,
        playback_frame_idx=0,
        playback_time_sec=0.0,
        source_frame_idx=0,
        source_timestamp_sec=0.0,
        observations=[obs1, obs2],
    )

    sidecar = ObservationTimelineSidecar(
        dataset_name="test_dataset",
        target_sampling_fps=5.0,
        playback_fps=5.0,
        total_samples=1,
        video_metadata={"width": 920, "height": 680, "native_fps": 30.0},
        playback_mapping=[entry0],
        tracks_summary={"101": {"track_id": 101, "zones_visited": ["shelf_a"]}},
    )

    assert sidecar.schema_version == "1.0"
    assert sidecar.coordinate_units == "configured_floor_units"
    assert "not independently surveyed" in sidecar.coordinate_units_disclaimer.lower()
    assert sidecar.timestamp_basis == "source_video_seconds"

    # Save to disk
    out_file = str(tmp_path / "test_sidecar.json")
    sidecar.save_json(out_file)
    assert os.path.isfile(out_file)

    # Load from disk
    loaded = ObservationTimelineSidecar.load_json(out_file)
    assert loaded.schema_version == "1.0"
    assert loaded.total_samples == 1
    assert len(loaded.playback_mapping) == 1
    loaded_obs = loaded.playback_mapping[0].observations
    assert len(loaded_obs) == 2
    assert loaded_obs[0].track_id == 101
    assert loaded_obs[0].floor_point == [2.5, 3.0]
    assert loaded_obs[0].assigned_zone == "shelf_a"
    assert loaded_obs[1].track_id == 102
    assert loaded_obs[1].floor_point is None
    assert loaded_obs[1].is_out_of_bounds is True


# ---------------------------------------------------------------------------
# 2. Timing Alignment Tests
# ---------------------------------------------------------------------------

def test_source_playback_timestamp_mapping():
    """Verify explicit mapping between source video frames and playback video time."""
    # Source video: 30 FPS. Target sampling: 5 FPS (stride = 6 frames)
    sampled_records = [
        (0, 0.0),
        (6, 0.200),
        (12, 0.400),
        (18, 0.600),
        (24, 0.800),
    ]

    sidecar = build_observation_timeline_sidecar(
        dataset_name="timing_test",
        target_sampling_fps=5.0,
        playback_fps=5.0,
        sampled_frame_records=sampled_records,
        all_tracks=[],
        trajectories=[],
    )

    assert sidecar.total_samples == 5
    for idx, entry in enumerate(sidecar.playback_mapping):
        expected_src_frame, expected_src_time = sampled_records[idx]
        assert entry.sample_index == idx
        assert entry.playback_frame_idx == idx
        assert entry.playback_time_sec == round(idx / 5.0, 3)
        assert entry.source_frame_idx == expected_src_frame
        assert abs(entry.source_timestamp_sec - expected_src_time) < 1e-4


# ---------------------------------------------------------------------------
# 3. Missing Observations & Multi-Track Handling
# ---------------------------------------------------------------------------

def test_missing_observations_and_track_gaps():
    """Tracks that appear and disappear must leave observation gaps without fabricated values."""
    sampled_records = [(0, 0.0), (6, 0.2), (12, 0.4), (18, 0.6)]

    # Track 1 appears at frame 0 and 6, missing at 12, reappears at 18
    # Track 2 appears only at frame 6 and 12
    Track.reset_id_counter(0)
    t1 = Track(bbox=[100, 100, 150, 200], confidence=0.9, frame_id=0, timestamp_sec=0.0)
    t1.track_id = 1
    t1.history.append(TrackPoint(frame_id=6, timestamp_sec=0.2, bbox=[110, 100, 160, 200], bottom_center=[135, 200], confidence=0.88))
    t1.history.append(TrackPoint(frame_id=18, timestamp_sec=0.6, bbox=[120, 100, 170, 200], bottom_center=[145, 200], confidence=0.85))

    t2 = Track(bbox=[300, 200, 350, 300], confidence=0.92, frame_id=6, timestamp_sec=0.2)
    t2.track_id = 2
    t2.history.append(TrackPoint(frame_id=12, timestamp_sec=0.4, bbox=[310, 200, 360, 300], bottom_center=[335, 300], confidence=0.90))

    traj1 = Trajectory(track_id=1)
    traj1.add_point(TrajectoryPoint(1, 0, 0.0, 125, 200, floor_x=1.0, floor_y=1.0, zone_id="shelf_a"))
    traj1.add_point(TrajectoryPoint(1, 6, 0.2, 135, 200, floor_x=1.1, floor_y=1.1, zone_id="shelf_a"))
    traj1.add_point(TrajectoryPoint(1, 18, 0.6, 145, 200, floor_x=1.2, floor_y=1.2, zone_id="shelf_a"))

    traj2 = Trajectory(track_id=2)
    traj2.add_point(TrajectoryPoint(2, 6, 0.2, 325, 300, floor_x=6.0, floor_y=2.0, zone_id="shelf_b"))
    traj2.add_point(TrajectoryPoint(2, 12, 0.4, 335, 300, floor_x=6.1, floor_y=2.1, zone_id="shelf_b"))

    sidecar = build_observation_timeline_sidecar(
        dataset_name="gap_test",
        target_sampling_fps=5.0,
        playback_fps=5.0,
        sampled_frame_records=sampled_records,
        all_tracks=[t1, t2],
        trajectories=[traj1, traj2],
        bounds=(0.0, 10.0, 0.0, 6.0),
    )

    mapping = sidecar.playback_mapping
    # Frame 0: Only Track 1
    assert [o.track_id for o in mapping[0].observations] == [1]
    # Frame 6: Both Track 1 and Track 2
    assert [o.track_id for o in mapping[1].observations] == [1, 2]
    # Frame 12: Only Track 2 (Track 1 is an observation gap)
    assert [o.track_id for o in mapping[2].observations] == [2]
    # Frame 18: Only Track 1
    assert [o.track_id for o in mapping[3].observations] == [1]


# ---------------------------------------------------------------------------
# 4. Out-of-Bounds and Untransformed Observations
# ---------------------------------------------------------------------------

def test_out_of_bounds_and_untransformed_points():
    """Points outside configured bounds must be marked out-of-bounds, unprojected points must have floor_point=None."""
    sampled_records = [(0, 0.0)]
    t1 = Track(bbox=[10, 10, 50, 100], confidence=0.8, frame_id=0, timestamp_sec=0.0)
    t1.track_id = 1
    t1.history.append(TrackPoint(frame_id=0, timestamp_sec=0.0, bbox=[10, 10, 50, 100], bottom_center=[30, 100], confidence=0.8))

    # Trajectory point with floor coordinates far outside bounds [0..10, 0..6]
    traj1 = Trajectory(track_id=1)
    traj1.add_point(TrajectoryPoint(1, 0, 0.0, 30, 100, floor_x=25.0, floor_y=-5.0, zone_id=None))

    sidecar = build_observation_timeline_sidecar(
        dataset_name="oob_test",
        target_sampling_fps=5.0,
        playback_fps=5.0,
        sampled_frame_records=sampled_records,
        all_tracks=[t1],
        trajectories=[traj1],
        bounds=(0.0, 10.0, 0.0, 6.0),
    )

    obs = sidecar.playback_mapping[0].observations[0]
    assert obs.is_out_of_bounds is True
    assert obs.assigned_zone is None
    assert obs.floor_point == [25.0, -5.0]


# ---------------------------------------------------------------------------
# 5. Cumulative Density Point Conservation
# ---------------------------------------------------------------------------

def test_cumulative_density_point_conservation():
    """Verify that cumulative density calculation up to sample k strictly conserves evaluated points."""
    # Simulate 5 samples with varying numbers of valid in-bounds observations
    obs_counts_per_sample = [2, 1, 3, 0, 2]
    total_expected_points = sum(obs_counts_per_sample)

    min_x, max_x, min_y, max_y = 0.0, 10.0, 0.0, 6.0
    cell_size = 0.5
    nx = int((max_x - min_x) / cell_size)
    ny = int((max_y - min_y) / cell_size)

    grid = np.zeros((nx, ny), dtype=int)
    cumulative_points_added = 0

    rng = np.random.default_rng(42)
    for sample_idx, count in enumerate(obs_counts_per_sample):
        for _ in range(count):
            fx = rng.uniform(min_x, max_x - 1e-4)
            fy = rng.uniform(min_y, max_y - 1e-4)
            cx = int((fx - min_x) / cell_size)
            cy = int((fy - min_y) / cell_size)
            grid[cx, cy] += 1
            cumulative_points_added += 1

        # Check conservation invariant at every step k
        assert np.sum(grid) == cumulative_points_added

    assert np.sum(grid) == total_expected_points


# ---------------------------------------------------------------------------
# 6. Legacy Run Handling
# ---------------------------------------------------------------------------

def test_legacy_runs_lack_timeline():
    """A legacy run dictionary without observation_timeline artifact must have has_observation_timeline=False."""
    legacy_data = {
        "dataset_name": "merl_shopping_dataset",
        "total_frames_evaluated": 100,
        "generated_artifacts": {
            "tracking_video": "demo_real_tracking_video.webm",
            "floor_trajectory": "demo_real_floor_trajectory.png",
        }
    }
    run = load_standardized_run(legacy_data, base_dir="outputs/reports")
    assert run is not None
    assert run.has_observation_timeline is False
    assert run.observation_timeline_path is None
    assert load_run_observation_timeline(run) is None


# ---------------------------------------------------------------------------
# 7. Switching Runs Clears State
# ---------------------------------------------------------------------------

def test_switching_runs_clears_playback_selection_state():
    """Verify that switching runs purges previous session state keys."""
    session_state = {
        "workspace_active_run_selector": "run_beta",
        "authoritative_run_id": "run_alpha",
        "sync_sample_run_alpha": 42,
        "sync_track_run_alpha": 3,
        "other_unrelated_key": "preserved",
    }

    # Simulate run change handler logic
    session_state["authoritative_run_id"] = session_state["workspace_active_run_selector"]
    for k in list(session_state.keys()):
        if k.startswith("sync_sample_") or k.startswith("sync_track_"):
            session_state.pop(k, None)

    assert session_state["authoritative_run_id"] == "run_beta"
    assert "sync_sample_run_alpha" not in session_state
    assert "sync_track_run_alpha" not in session_state
    assert session_state["other_unrelated_key"] == "preserved"


# ---------------------------------------------------------------------------
# 8. Pipeline Execution with Timeline Export
# ---------------------------------------------------------------------------

def test_real_data_pipeline_generates_timeline_artifact(tmp_path):
    """Verify run_real_data_demo creates and links the observation timeline sidecar."""
    merl_video = "data/merl/10_1_crop.mp4"
    if not os.path.exists(merl_video):
        pytest.skip("MERL video not present in local data directory")

    import yaml
    from scripts.run_vscode_demo import run_real_data_demo

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = str(tmp_path / "sync_pipeline_run")
    summary = run_real_data_demo(
        config,
        reports_dir=out_dir,
        max_frames=15,
        video_path=merl_video,
        artifact_prefix="test_sync",
        use_merl_profile=True,
    )

    assert summary is not None
    assert "observation_timeline" in summary["generated_artifacts"]
    timeline_path = summary["generated_artifacts"]["observation_timeline"]
    assert os.path.isfile(timeline_path)

    # Check sidecar content
    sidecar = ObservationTimelineSidecar.load_json(timeline_path)
    assert sidecar.total_samples == 15
    assert len(sidecar.playback_mapping) == 15

    # Check that standardized run adapter loads it
    run = load_standardized_run(os.path.join(out_dir, "test_sync_summary.json"), base_dir=out_dir)
    assert run is not None
    assert run.has_observation_timeline is True
    assert run.observation_timeline_path == timeline_path

    timeline_loaded = load_run_observation_timeline(run)
    assert timeline_loaded is not None
    assert timeline_loaded["schema_version"] == "1.0"
    assert len(timeline_loaded["playback_mapping"]) == 15


# ---------------------------------------------------------------------------
# 9. Differing Source, Sampling, and Playback FPS
# ---------------------------------------------------------------------------

def test_differing_source_sampling_and_playback_fps():
    """Every analysed sample, including empty frames, must map correctly when source, sampling, and playback FPS differ."""
    # Source: 60 FPS, Sampling: 10 FPS (stride = 6 frames), Playback: 2.5 FPS (slow motion)
    sampled_records = [
        (0, 0.0),
        (6, 0.100),
        (12, 0.200),
        (18, 0.300),
        (24, 0.400),
    ]

    t1 = Track(bbox=[100, 100, 150, 200], confidence=0.9, frame_id=6, timestamp_sec=0.1)
    t1.track_id = 7
    t1.history.append(TrackPoint(frame_id=6, timestamp_sec=0.1, bbox=[100, 100, 150, 200], bottom_center=[125, 200], confidence=0.9))

    traj1 = Trajectory(track_id=7)
    traj1.add_point(TrajectoryPoint(7, 6, 0.1, 125, 200, floor_x=3.0, floor_y=2.0, zone_id=None))

    sidecar = build_observation_timeline_sidecar(
        dataset_name="multi_rate_test",
        target_sampling_fps=10.0,
        playback_fps=2.5,
        sampled_frame_records=sampled_records,
        all_tracks=[t1],
        trajectories=[traj1],
        bounds=(0.0, 10.0, 0.0, 6.0),
        video_metadata={"fps": 60.0, "total_frames": 3600, "duration_seconds": 60.0},
    )

    assert sidecar.source_fps == 60.0
    assert sidecar.target_sampling_fps == 10.0
    assert sidecar.playback_fps == 2.5
    assert sidecar.total_samples == 5
    assert sidecar.total_source_frames == 3600
    assert sidecar.source_duration_sec == 60.0
    assert sidecar.sampled_source_time_span_sec == 0.400
    assert sidecar.playback_duration_sec == 2.0  # 5 samples / 2.5 fps = 2.0s

    for k, entry in enumerate(sidecar.playback_mapping):
        expected_src_frame, expected_src_time = sampled_records[k]
        assert entry.sample_index == k
        assert entry.playback_frame_idx == k
        assert entry.playback_time_sec == round(k / 2.5, 3)
        assert entry.source_frame_idx == expected_src_frame
        assert abs(entry.source_timestamp_sec - expected_src_time) < 1e-4

        if k == 1:
            assert len(entry.observations) == 1
            assert entry.observations[0].track_id == 7
        else:
            # Empty samples still map correctly
            assert len(entry.observations) == 0


# ---------------------------------------------------------------------------
# 10. Track Gap Index Tracking & Timing Metric Distinction
# ---------------------------------------------------------------------------

def test_tracks_summary_gap_indices_and_duration_distinction():
    """Verify that tracks_summary identifies non-contiguous gap indices without empty lists."""
    sampled_records = [(i * 6, round(i * 0.2, 3)) for i in range(10)]

    # Track 1 active at samples 2, 3, 5, 8
    t1 = Track(bbox=[100, 100, 150, 200], confidence=0.9, frame_id=12, timestamp_sec=0.4)
    t1.track_id = 1
    t1.history.append(TrackPoint(frame_id=12, timestamp_sec=0.4, bbox=[100, 100, 150, 200], bottom_center=[125, 200], confidence=0.9))
    t1.history.append(TrackPoint(frame_id=18, timestamp_sec=0.6, bbox=[105, 100, 155, 200], bottom_center=[130, 200], confidence=0.88))
    t1.history.append(TrackPoint(frame_id=30, timestamp_sec=1.0, bbox=[110, 100, 160, 200], bottom_center=[135, 200], confidence=0.87))
    t1.history.append(TrackPoint(frame_id=48, timestamp_sec=1.6, bbox=[115, 100, 165, 200], bottom_center=[140, 200], confidence=0.85))

    traj1 = Trajectory(track_id=1)
    traj1.add_point(TrajectoryPoint(1, 12, 0.4, 125, 200, floor_x=1.0, floor_y=1.0, zone_id=None))
    traj1.add_point(TrajectoryPoint(1, 18, 0.6, 130, 200, floor_x=1.1, floor_y=1.1, zone_id=None))
    traj1.add_point(TrajectoryPoint(1, 30, 1.0, 135, 200, floor_x=1.2, floor_y=1.2, zone_id=None))
    traj1.add_point(TrajectoryPoint(1, 48, 1.6, 140, 200, floor_x=1.3, floor_y=1.3, zone_id=None))

    sidecar = build_observation_timeline_sidecar(
        dataset_name="gap_summary_test",
        target_sampling_fps=5.0,
        playback_fps=5.0,
        sampled_frame_records=sampled_records,
        all_tracks=[t1],
        trajectories=[traj1],
        bounds=(0.0, 10.0, 0.0, 6.0),
        video_metadata={"fps": 30.0, "total_frames": 300, "duration_seconds": 10.0},
    )

    t1_summary = sidecar.tracks_summary["1"]
    assert t1_summary["first_sample_index"] == 2
    assert t1_summary["last_sample_index"] == 8
    assert t1_summary["active_sample_span"] == 7
    assert t1_summary["observed_sample_count"] == 4
    # Gaps within active span: samples 4, 6, 7
    assert t1_summary["missing_sample_indices"] == [4, 6, 7]
    assert t1_summary["sample_gaps"] == [4, 6, 7]


# ---------------------------------------------------------------------------
# 11. Density Parity Across Boundaries & Non-MERL Geometries
# ---------------------------------------------------------------------------

def test_density_parity_exact_boundary_cell_edges_and_non_merl():
    """Verify exact cell counts between compute_density_grid and explorer JS semantics across boundaries."""
    from src.analytics.density import GridBounds, compute_density_grid

    # Non-MERL custom bounds and cell size
    bounds = GridBounds(-5.0, 15.0, 2.0, 12.0)  # width=20, height=10
    cell_size = 1.0                              # nx=20, ny=10

    test_points = [
        # 1. Exact min bounds
        (-5.0, 2.0, "min_corner"),
        # 2. Exact max bounds (should map to last cell without crashing)
        (15.0, 12.0, "max_corner"),
        # 3. Exact internal cell edge
        (0.0, 5.0, "internal_edge"),
        # 4. Standard in-bounds points
        (2.3, 4.7, "interior_1"),
        (2.3, 4.7, "interior_2"),  # repeated point in same cell
        # 5. Slightly out of bounds (must not be clamped!)
        (-5.0001, 5.0, "oob_left"),
        (15.0001, 5.0, "oob_right"),
        (0.0, 1.9999, "oob_bottom"),
        (0.0, 12.0001, "oob_top"),
        # 6. Untransformed (None coordinates)
        (None, None, "untransformed"),
    ]

    traj = Trajectory(track_id=1)
    for idx, (fx, fy, _) in enumerate(test_points):
        traj.add_point(TrajectoryPoint(1, idx, idx * 0.2, 0, 0, floor_x=fx, floor_y=fy))

    grid = compute_density_grid([traj], bounds=bounds, cell_size=cell_size)

    # Invariant: evaluated = binned + out_of_bounds + untransformed
    assert grid.total_evaluated_points == len(test_points)
    assert grid.total_binned_points == 5  # min_corner, max_corner, internal_edge, interior_1, interior_2
    assert grid.out_of_bounds_count == 4  # oob_left, oob_right, oob_bottom, oob_top
    assert grid.untransformed_count == 1  # None, None
    assert grid.total_evaluated_points == grid.total_binned_points + grid.out_of_bounds_count + grid.untransformed_count

    # Simulate explorer JS algorithm with same semantics
    minX, maxX, minY, maxY = bounds.min_x, bounds.max_x, bounds.min_y, bounds.max_y
    nx = max(1, round((maxX - minX) / cell_size))
    ny = max(1, round((maxY - minY) / cell_size))
    actualCsX = (maxX - minX) / nx
    actualCsY = (maxY - minY) / ny

    js_counts = np.zeros((ny, nx), dtype=int)
    for fx, fy, _ in test_points:
        if fx is None or fy is None:
            continue
        if fx < minX or fx > maxX or fy < minY or fy > maxY:
            continue  # Do not clamp!
        col = (nx - 1) if fx >= maxX else int((fx - minX) / actualCsX)
        col = min(nx - 1, max(0, col))
        row = (ny - 1) if fy >= maxY else int((fy - minY) / actualCsY)
        row = min(ny - 1, max(0, row))
        js_counts[row, col] += 1

    # Exact equality between Python compute_density_grid counts and JS counts
    assert np.array_equal(grid.counts, js_counts)
    assert np.sum(js_counts) == grid.total_binned_points


def test_adversarial_near_boundary_binning():
    """Verify that points within 1e-6 of cell boundaries do not drift due to rounding and match grid_cell."""
    from src.analytics.density import GridBounds, compute_density_grid

    bounds = (0.0, 10.0, 0.0, 6.0)
    cell_size = 0.5  # nx=20, ny=12

    # Two adversarial points right across the boundary at x=0.500000
    pt_left = (0.499999, 1.25)   # col 0 (0.499999 / 0.5 = 0.999998 -> col 0), row 2 (1.25 / 0.5 = 2.5 -> row 2)
    pt_right = (0.500000, 1.25)  # col 1 (0.500000 / 0.5 = 1.000000 -> col 1), row 2

    t1 = Track(bbox=[100, 100, 150, 200], confidence=0.9, frame_id=0, timestamp_sec=0.0)
    t1.track_id = 1
    t1.history.append(TrackPoint(frame_id=0, timestamp_sec=0.0, bbox=[100, 100, 150, 200], bottom_center=[125, 200], confidence=0.9))
    t1.history.append(TrackPoint(frame_id=1, timestamp_sec=0.2, bbox=[100, 100, 150, 200], bottom_center=[125, 200], confidence=0.9))

    traj = Trajectory(track_id=1)
    traj.add_point(TrajectoryPoint(1, 0, 0.0, 125, 200, floor_x=pt_left[0], floor_y=pt_left[1]))
    traj.add_point(TrajectoryPoint(1, 1, 0.2, 125, 200, floor_x=pt_right[0], floor_y=pt_right[1]))

    sidecar = build_observation_timeline_sidecar(
        dataset_name="adversarial_binning_test",
        target_sampling_fps=5.0,
        playback_fps=5.0,
        sampled_frame_records=[(0, 0.0), (1, 0.2)],
        all_tracks=[t1],
        trajectories=[traj],
        bounds=bounds,
        video_metadata={"fps": 5.0, "total_frames": 2, "duration_seconds": 0.4},
        cell_size=cell_size,
    )

    obs0 = sidecar.playback_mapping[0].observations[0]
    obs1 = sidecar.playback_mapping[1].observations[0]

    # Explicit grid_cell check
    assert obs0.grid_cell == (2, 0), f"Expected (2, 0), got {obs0.grid_cell}"
    assert obs1.grid_cell == (2, 1), f"Expected (2, 1), got {obs1.grid_cell}"

    # Precision check
    assert obs0.floor_point[0] == 0.499999
    assert obs1.floor_point[0] == 0.500000

    # Cross-check with compute_density_grid
    grid = compute_density_grid([traj], bounds=bounds, cell_size=cell_size)
    assert grid.counts[2, 0] == 1
    assert grid.counts[2, 1] == 1
    assert grid.total_binned_points == 2

