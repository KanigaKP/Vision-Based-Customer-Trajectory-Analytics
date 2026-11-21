"""
Phase 8.6 — Unit Tests for Temporal Hysteresis Sensitivity Analysis
==================================================================
Validates that:
1. Hysteresis windows N=1 and larger N produce deterministic, visibly distinct metrics on noisy trajectories.
2. Entry counts and transitions are faithfully aggregated from FootfallSummary.
3. Multiple windows are evaluated and reported in sorted deterministic order.
4. Baseline window selection (default N=3, first window fallback, or caller-specified) and deltas are accurate.
5. Invalid hysteresis window values (empty, zero, negative, bool, non-int, duplicate) raise ValueError/TypeError.
6. Invalid baseline window selection is rejected.
7. Empty trajectory input produces an explicit INSUFFICIENT_EVIDENCE result without crashing.
8. Repeated runs on identical inputs produce strictly identical results.
9. Caller-owned Trajectory objects and points are NOT mutated across or after sensitivity runs.
10. Serialization to and from dictionary/JSON round-trips losslessly.
11. Existing ZoneEvaluator and FootfallAnalyzer production behavior remains intact.
"""

import json
import pytest

from src.analytics.zone import Zone, ZoneConfig
from src.analytics.zone_evaluator import ZoneEvaluator
from src.analytics.footfall import FootfallAnalyzer
from src.trajectory.trajectory import Trajectory, TrajectoryPoint
from src.validation.hysteresis_validation import (
    HysteresisWindowMetrics,
    HysteresisSensitivityResult,
    HysteresisSensitivityValidator,
    analyze_hysteresis_sensitivity,
)


@pytest.fixture
def synthetic_noisy_scenario():
    """
    Synthetic fixture with:
    - Zone A: [(0, 0), (4, 0), (4, 4), (0, 4)]
    - Zone B: [(5, 0), (9, 0), (9, 4), (5, 4)]
    - Trajectory 1: Starts in Zone A, has 1 frame of boundary noise in Zone B, returns to Zone A.
      Coordinates: [(2, 2), (2, 2), (7, 2), (2, 2), (2, 2)]
      Under N=1: enters A, transitions to B (entry B), transitions to A (entry A) -> 3 entries, 2 transitions.
      Under N=3: single noisy frame is filtered out -> 1 entry (A), 0 transitions.
    """
    z_a = Zone("zone_a", "Zone A", [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    z_b = Zone("zone_b", "Zone B", [(5.0, 0.0), (9.0, 0.0), (9.0, 4.0), (5.0, 4.0)])

    traj = Trajectory(track_id=1)
    coords = [(2.0, 2.0), (2.0, 2.0), (7.0, 2.0), (2.0, 2.0), (2.0, 2.0)]
    for idx, (x, y) in enumerate(coords):
        traj.add_point(
            TrajectoryPoint(
                track_id=1,
                frame_idx=idx,
                timestamp_sec=idx * 0.2,
                image_x=100.0,
                image_y=100.0,
                floor_x=x,
                floor_y=y,
                zone_id=None,
            )
        )

    return [z_a, z_b], [traj]


def test_n1_and_larger_n_produce_distinct_deterministic_results(synthetic_noisy_scenario):
    """Test 1: N=1 captures single-frame noise while N=3 filters it out deterministically."""
    zones, trajs = synthetic_noisy_scenario

    result = HysteresisSensitivityValidator.analyze_sensitivity(
        zones=zones,
        trajectories=trajs,
        windows=[1, 3],
        baseline_window=3,
        dataset_name="synthetic_noise_test",
    )

    assert result.evidence_status == "HYSTERESIS_SENSITIVITY_EVALUATED"
    assert result.evaluated_windows == [1, 3]
    assert len(result.window_metrics) == 2

    m_n1 = result.window_metrics[0]
    assert m_n1.window == 1
    # N=1 captures the 1-frame bounce: enters A, goes to B, returns to A
    assert m_n1.total_entry_events == 3
    assert m_n1.entry_events_per_zone == {"zone_a": 2, "zone_b": 1}
    assert m_n1.total_transitions == 2
    assert m_n1.total_zone_changes == 2

    m_n3 = result.window_metrics[1]
    assert m_n3.window == 3
    # N=3 filters out the 1-frame bounce: remains in A throughout
    assert m_n3.total_entry_events == 1
    assert m_n3.entry_events_per_zone == {"zone_a": 1, "zone_b": 0}
    assert m_n3.total_transitions == 0
    assert m_n3.total_zone_changes == 0


def test_metrics_aggregated_from_footfall_summary(synthetic_noisy_scenario):
    """Test 2: Entry counts and transitions are verified directly against FootfallSummary."""
    zones, trajs = synthetic_noisy_scenario

    result = HysteresisSensitivityValidator.analyze_sensitivity(
        zones=zones,
        trajectories=trajs,
        windows=[1],
    )

    m1 = result.window_metrics[0]
    assert m1.transition_matrix["zone_a"]["zone_b"] == 1
    assert m1.transition_matrix["zone_b"]["zone_a"] == 1
    assert m1.transition_matrix["zone_a"]["zone_a"] == 0
    assert m1.unique_tracks_per_zone == {"zone_a": 1, "zone_b": 1}


def test_multiple_windows_returned_in_sorted_order(synthetic_noisy_scenario):
    """Test 3: Windows passed out of order are evaluated and returned in ascending sorted order."""
    zones, trajs = synthetic_noisy_scenario

    # Pass unsorted windows
    result = HysteresisSensitivityValidator.analyze_sensitivity(
        zones=zones,
        trajectories=trajs,
        windows=[5, 1, 4, 2],
    )

    assert result.evaluated_windows == [1, 2, 4, 5]
    assert [m.window for m in result.window_metrics] == [1, 2, 4, 5]


def test_baseline_selection_and_deltas(synthetic_noisy_scenario):
    """Test 4: Default baseline (N=3 if present, else first) and delta calculations."""
    zones, trajs = synthetic_noisy_scenario

    # 1. Default baseline when 3 is present -> baseline = 3
    res_default_3 = HysteresisSensitivityValidator.analyze_sensitivity(
        zones=zones,
        trajectories=trajs,
        windows=[1, 2, 3, 5],
    )
    assert res_default_3.baseline_window == 3
    m1 = next(m for m in res_default_3.window_metrics if m.window == 1)
    m3 = next(m for m in res_default_3.window_metrics if m.window == 3)
    # Deltas relative to N=3:
    assert m1.delta_entry_events == m1.total_entry_events - m3.total_entry_events
    assert m1.delta_transitions == m1.total_transitions - m3.total_transitions
    assert m3.delta_entry_events == 0
    assert m3.delta_transitions == 0

    # 2. Default baseline when 3 is NOT present -> baseline = first window (4)
    res_no_3 = HysteresisSensitivityValidator.analyze_sensitivity(
        zones=zones,
        trajectories=trajs,
        windows=[4, 6],
    )
    assert res_no_3.baseline_window == 4

    # 3. Explicit baseline specified by caller
    res_explicit = HysteresisSensitivityValidator.analyze_sensitivity(
        zones=zones,
        trajectories=trajs,
        windows=[1, 2, 5],
        baseline_window=2,
    )
    assert res_explicit.baseline_window == 2


def test_invalid_windows_rejected(synthetic_noisy_scenario):
    """Test 5: Rejects empty, zero, negative, bool, non-int, and duplicate window inputs."""
    zones, trajs = synthetic_noisy_scenario

    # Empty list
    with pytest.raises(ValueError, match="windows sequence cannot be empty"):
        HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows=[])

    # Zero
    with pytest.raises(ValueError, match="must be a positive integer"):
        HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows=[0, 1])

    # Negative
    with pytest.raises(ValueError, match="must be a positive integer"):
        HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows=[-2, 3])

    # Bool (e.g. True)
    with pytest.raises(TypeError, match="must be an integer"):
        HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows=[True, 3])

    # Non-integer (float, string)
    with pytest.raises(TypeError, match="must be an integer"):
        HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows=[2.5, 3])

    with pytest.raises(TypeError, match="must be an integer"):
        HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows=["3"])

    # Duplicate values
    with pytest.raises(ValueError, match="Duplicate hysteresis window value found"):
        HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows=[1, 2, 2, 3])


def test_invalid_baseline_selection_rejected(synthetic_noisy_scenario):
    """Test 6: Baseline not in evaluated windows raises ValueError."""
    zones, trajs = synthetic_noisy_scenario

    with pytest.raises(ValueError, match="is not among evaluated windows"):
        HysteresisSensitivityValidator.analyze_sensitivity(
            zones=zones,
            trajectories=trajs,
            windows=[1, 2, 3],
            baseline_window=10,
        )


def test_empty_trajectories_handled_explicitly(synthetic_noisy_scenario):
    """Test 7: Empty trajectories list returns INSUFFICIENT_EVIDENCE status without crashing."""
    zones, _ = synthetic_noisy_scenario

    result = HysteresisSensitivityValidator.analyze_sensitivity(
        zones=zones,
        trajectories=[],
        windows=[1, 3],
        dataset_name="empty_seq",
    )

    assert result.evidence_status == "INSUFFICIENT_EVIDENCE"
    assert result.evaluated_windows == [1, 3]
    assert len(result.window_metrics) == 0
    assert "No trajectories provided" in result.evidence_note


def test_repeated_runs_produce_identical_results(synthetic_noisy_scenario):
    """Test 8: Running sensitivity analysis multiple times produces identical output."""
    zones, trajs = synthetic_noisy_scenario

    res1 = HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows=[1, 3])
    res2 = HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows=[1, 3])

    assert res1.to_dict() == res2.to_dict()


def test_caller_trajectories_not_mutated(synthetic_noisy_scenario):
    """
    Test 9: Critical mutation test.
    Confirms that ZoneEvaluator.evaluate_trajectory runs on clones and does NOT mutate
    the caller-owned TrajectoryPoint.zone_id attributes.
    """
    zones, trajs = synthetic_noisy_scenario
    traj = trajs[0]

    # Pre-set zone_ids to a known sentinel state
    for pt in traj.points:
        pt.zone_id = "caller_original_state"

    # Run sensitivity across several windows
    HysteresisSensitivityValidator.analyze_sensitivity(
        zones=zones,
        trajectories=trajs,
        windows=[1, 2, 3, 5],
    )

    # Caller's trajectory points must still have 'caller_original_state'
    for pt in traj.points:
        assert pt.zone_id == "caller_original_state"


def test_serialization_round_trip(synthetic_noisy_scenario):
    """Test 10: HysteresisSensitivityResult serializes and deserializes cleanly."""
    zones, trajs = synthetic_noisy_scenario

    result = HysteresisSensitivityValidator.analyze_sensitivity(
        zones=zones,
        trajectories=trajs,
        windows=[1, 3],
        baseline_window=3,
        dataset_name="json_test",
    )

    data = result.to_dict()
    json_str = json.dumps(data)
    reconstructed = HysteresisSensitivityResult.from_dict(json.loads(json_str))

    assert reconstructed.dataset_name == result.dataset_name
    assert reconstructed.baseline_window == result.baseline_window
    assert reconstructed.evaluated_windows == result.evaluated_windows
    assert reconstructed.evidence_status == result.evidence_status
    assert len(reconstructed.window_metrics) == 2

    m0 = reconstructed.window_metrics[0]
    assert m0.window == 1
    assert m0.total_entry_events == 3
    assert m0.delta_entry_events == 2


def test_production_behavior_unchanged(synthetic_noisy_scenario):
    """Test 11: Production ZoneEvaluator and FootfallAnalyzer continue to function normally."""
    zones, trajs = synthetic_noisy_scenario
    traj = trajs[0]

    # Direct production ZoneEvaluator run
    evaluator = ZoneEvaluator(zones=zones, hysteresis_window=3)
    assigned = evaluator.evaluate_trajectory(traj)
    assert assigned[2] == "zone_a"  # Noise filtered at N=3

    # Direct production FootfallAnalyzer run
    analyzer = FootfallAnalyzer(known_zone_ids=["zone_a", "zone_b"])
    summary = analyzer.analyze([traj])
    assert summary.entry_events_per_zone["zone_a"] == 1
    assert summary.entry_events_per_zone["zone_b"] == 0


def test_convenience_function_and_dict_config_input(synthetic_noisy_scenario):
    """Test 12: Convenience function analyze_hysteresis_sensitivity works with dict config."""
    zones, trajs = synthetic_noisy_scenario
    config_dict = {
        "zones": {
            "hysteresis_window": 3,
            "merl": [z.to_dict() for z in zones],
        }
    }

    res = analyze_hysteresis_sensitivity(
        zones=config_dict,
        trajectories=trajs,
        windows=[1, 3],
        dataset_name="convenience_test",
    )

    assert res.dataset_name == "convenience_test"
    assert res.evaluated_windows == [1, 3]
    assert len(res.window_metrics) == 2


def test_sequence_inputs_including_range_and_rejection_of_strings(synthetic_noisy_scenario):
    """
    Test 13: Verifies that arbitrary non-string Sequence[int] implementations (such as range)
    as well as tuples and sets are accepted, sorted deterministically, and that strings,
    bytes, empty sequences, and non-sequence containers are rejected.
    """
    zones, trajs = synthetic_noisy_scenario

    # 1. range object accepted and evaluated in order
    res_range = HysteresisSensitivityValidator.analyze_sensitivity(
        zones=zones,
        trajectories=trajs,
        windows=range(1, 4),
    )
    assert res_range.evaluated_windows == [1, 2, 3]
    assert [m.window for m in res_range.window_metrics] == [1, 2, 3]

    # 2. tuple input accepted and sorted deterministically
    res_tuple = HysteresisSensitivityValidator.analyze_sensitivity(
        zones=zones,
        trajectories=trajs,
        windows=(3, 1),
    )
    assert res_tuple.evaluated_windows == [1, 3]

    # 3. set input accepted and sorted deterministically
    res_set = HysteresisSensitivityValidator.analyze_sensitivity(
        zones=zones,
        trajectories=trajs,
        windows={4, 2},
    )
    assert res_set.evaluated_windows == [2, 4]

    # 4. Strings rejected even though they implement Sequence
    with pytest.raises(TypeError, match="windows must be a non-string sequence of integers"):
        HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows="123")

    # 5. Bytes and bytearray rejected even though they implement Sequence
    with pytest.raises(TypeError, match="windows must be a non-string sequence of integers"):
        HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows=b"\x01\x02\x03")

    with pytest.raises(TypeError, match="windows must be a non-string sequence of integers"):
        HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows=bytearray(b"\x01\x02"))

    # 6. Non-sequence scalar or dict rejected
    with pytest.raises(TypeError, match="windows must be a non-string sequence of integers"):
        HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows=5)

    with pytest.raises(TypeError, match="windows must be a non-string sequence of integers"):
        HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows={"1": 1})

    # 7. Empty range, tuple, and set rejected as empty sequences
    with pytest.raises(ValueError, match="windows sequence cannot be empty"):
        HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows=range(0))

    with pytest.raises(ValueError, match="windows sequence cannot be empty"):
        HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows=())

    with pytest.raises(ValueError, match="windows sequence cannot be empty"):
        HysteresisSensitivityValidator.analyze_sensitivity(zones, trajs, windows=set())

