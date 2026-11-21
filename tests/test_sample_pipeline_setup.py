"""
Focused Tests for Sample-Pipeline Setup & Configuration Consistency.
=====================================================================
Verifies:
1. `setup_merl_zone_evaluator` correctly obtains the MERL hysteresis window from
   `ZoneConfig.load_from_config(config, dataset_key="merl")`.
2. The returned hysteresis window is passed directly to `ZoneEvaluator` rather than
   being hard-coded.
3. Dynamically adapts when the configured hysteresis window changes (e.g. N=1, N=5, N=10).
4. Evaluates synthetic trajectories through PIP and hysteresis without requiring
   the MERL video container (`data/merl/10_1_crop.mp4`) or ONNX model weights (`yolo11n.onnx`).
"""

import os
import pytest
import yaml
from typing import Dict, Any

from scripts.run_sample_analysis import setup_merl_zone_evaluator
from src.analytics.zone import Zone, ZoneConfig
from src.analytics.zone_evaluator import ZoneEvaluator
from src.trajectory.trajectory import Trajectory, TrajectoryPoint


@pytest.fixture
def repo_config() -> Dict[str, Any]:
    """Load the project's actual config.yaml."""
    config_path = "config.yaml"
    assert os.path.exists(config_path), f"config.yaml not found at {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_setup_merl_zone_evaluator_uses_configured_hysteresis_from_config(repo_config):
    """
    Verify setup_merl_zone_evaluator loads zones and the exact hysteresis window
    declared in config.yaml, passing it to ZoneEvaluator.
    """
    expected_window = int(repo_config["zones"]["hysteresis_window"])
    assert expected_window == 3, "Sanity check: config.yaml specifies hysteresis_window: 3"

    evaluator, known_zone_ids, loaded_window = setup_merl_zone_evaluator(repo_config)

    # 1. Returned window matches config
    assert loaded_window == expected_window

    # 2. ZoneEvaluator uses the configured window (NOT hardcoded)
    assert isinstance(evaluator, ZoneEvaluator)
    assert evaluator.hysteresis_window == expected_window

    # 3. Known zones match MERL retail zones
    assert len(known_zone_ids) == 4
    for expected_zone in ["checkout", "main_aisle", "shelf_a", "shelf_b"]:
        assert expected_zone in known_zone_ids


@pytest.mark.parametrize("custom_hysteresis", [1, 2, 5, 8, 12])
def test_setup_merl_zone_evaluator_dynamically_respects_custom_hysteresis(custom_hysteresis):
    """
    Verify setup_merl_zone_evaluator does NOT hard-code any value:
    When config['zones']['hysteresis_window'] is modified, ZoneEvaluator
    strictly adopts the new configured value.
    """
    synthetic_config = {
        "zones": {
            "hysteresis_window": custom_hysteresis,
            "merl": [
                {
                    "zone_id": "shelf_left",
                    "name": "Left Shelf",
                    "polygon": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
                },
                {
                    "zone_id": "shelf_right",
                    "name": "Right Shelf",
                    "polygon": [[6.0, 0.0], [10.0, 0.0], [10.0, 4.0], [6.0, 4.0]],
                },
            ],
        }
    }

    evaluator, known_zone_ids, loaded_window = setup_merl_zone_evaluator(synthetic_config)

    assert loaded_window == custom_hysteresis
    assert evaluator.hysteresis_window == custom_hysteresis
    assert known_zone_ids == ["shelf_left", "shelf_right"]


def test_setup_merl_zone_evaluator_evaluates_synthetic_trajectory_without_assets(repo_config):
    """
    Verify that ZoneEvaluator instantiated via setup_merl_zone_evaluator functions
    on synthetic trajectory data without requiring the MERL video or ONNX model weights.
    """
    evaluator, known_zone_ids, window = setup_merl_zone_evaluator(repo_config)
    assert window == 3

    # Trajectory with points:
    # 1. [5.0, 0.5] inside checkout (checkout: x in [0, 10], y in [0, 1])
    # 2. [5.0, 0.5] inside checkout
    # 3. [5.0, 0.5] inside checkout -> confirms checkout
    # 4. [15.0, 15.0] outside all zones (boundary margin)
    trk = Trajectory(track_id=99)
    test_coords = [
        (5.0, 0.5),
        (5.0, 0.5),
        (5.0, 0.5),
        (15.0, 15.0),
        (15.0, 15.0),
        (15.0, 15.0),
    ]
    for idx, (x, y) in enumerate(test_coords):
        trk.add_point(TrajectoryPoint(
            track_id=99,
            frame_idx=idx,
            timestamp_sec=idx * 0.2,
            image_x=100.0,
            image_y=100.0,
            floor_x=x,
            floor_y=y,
        ))

    assigned = evaluator.evaluate_trajectory(trk)
    assert len(assigned) == 6
    # Points 0, 1, 2 inside checkout
    assert assigned[0] == "checkout"
    assert assigned[1] == "checkout"
    assert assigned[2] == "checkout"
    # Points 3, 4, 5 transition to None after hysteresis
    assert assigned[5] is None


def test_shop_switching_triplet_consistency():
    """
    Verify that each verified example shop defines consistent scene profiles,
    calibrated zones, and video dimensions without cross-shop contamination.
    """
    import json
    from dashboard.views.main_workflow import VERIFIED_EXAMPLES

    for key, info in VERIFIED_EXAMPLES.items():
        assert os.path.exists(info["profile_path"]), f"Profile missing: {info['profile_path']}"
        assert os.path.exists(info["video_path"]), f"Video missing: {info['video_path']}"

        with open(info["profile_path"], "r", encoding="utf-8") as f:
            profile = json.load(f)

        assert "zones" in profile and len(profile["zones"]) >= 3
        assert "bounds" in profile and len(profile["bounds"]) == 4

        # Verify dimension synchronization
        if key == "merl":
            assert profile["image_width"] == 920
            assert profile["image_height"] == 680
            zone_ids = [z.get("zone_id") for z in profile["zones"]]
            assert "shelf_a" in zone_ids
        elif key == "site03":
            assert profile["image_width"] == 896
            assert profile["image_height"] == 672
            zone_ids = [z.get("zone_id") for z in profile["zones"]]
            assert "side_display_left" in zone_ids
        elif key == "site02":
            assert profile["image_width"] == 896
            assert profile["image_height"] == 672
            zone_ids = [z.get("zone_id") for z in profile["zones"]]
            assert "showcase_display_front" in zone_ids


def test_episode_2_seek_and_gap_detail():
    """
    Verify Episode 2 of MERL run upload_ca30f3e8:
    - Starts at 01:12.80 (72.80s), Track #3, shelf_a
    - Identifies missing observation at sample 382 (frame 2292, timestamp 76.40s)
    - Provides exact concise gap note: 'One tracking sample is missing. The video is available for review.'
    """
    import json
    from dashboard.run_adapter import discover_standardized_runs
    from dashboard.views.spatial_studio import recover_zone_dwell_episodes
    from dashboard.views.episode_helper import get_episode_gap_summary, format_timestamp_mmss

    runs = discover_standardized_runs("outputs/reports")
    run = next((r for r in runs if "upload_ca30f3e8" in r.run_id), None)
    assert run is not None, "Saved run upload_ca30f3e8 must exist"

    timeline_path = "outputs/reports/uploaded_runs/upload_ca30f3e8/run_observation_timeline.json"
    assert os.path.exists(timeline_path), f"Timeline must exist: {timeline_path}"
    with open(timeline_path, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)

    episodes, err = recover_zone_dwell_episodes(run, timeline_data, "shelf_a")
    assert err is None
    assert len(episodes) == 3

    ep2 = episodes[1]
    assert ep2["episode_index"] == 2
    assert ep2["track_id"] == 3
    assert ep2["start_sample_idx"] == 364
    assert format_timestamp_mmss(ep2["entry_timestamp_sec"]) == "01:12.80"
    assert ep2["has_gaps"] is True
    assert len(ep2["gaps"]) == 1

    gap = ep2["gaps"][0]
    assert gap["missing_sample_count"] == 1
    assert gap["from_sample"] == 381
    assert gap["to_sample"] == 383
    assert "missing_samples" in gap
    assert len(gap["missing_samples"]) == 1
    missing = gap["missing_samples"][0]
    assert missing["sample_idx"] == 382
    assert missing["source_frame_idx"] == 2292
    assert abs(missing["source_timestamp_sec"] - 76.40) < 0.01

    gap_note, color = get_episode_gap_summary(ep2)
    assert gap_note == "One tracking sample is missing. The video is available for review."
    assert "continuous" not in gap_note.lower()
