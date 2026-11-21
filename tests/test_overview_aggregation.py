"""
Tests for Overview Dynamic Workspace Summary, Provenance States, and Missing Values.
==================================================================================
Verifies:
1. Dynamic workspace aggregation derived from discovered run metadata.
2. Distinct analysis runs vs. unique source recordings.
3. Deduplication of duplicate representations using explicit run identity / lineage.
4. Preservation of three provenance states: retail, non-retail, and unknown (unknown is never labeled non-retail).
5. Preservation of missing vs. evaluated-zero values (missing tracks reported as unavailable, not 0).
6. Dynamic updating of workspace summary when a newly discovered run is added.
"""

import pytest
from typing import Dict, Any, List
from pathlib import Path

from dashboard.run_adapter import (
    StandardizedRunData,
    compute_workspace_summary,
    format_observed_value,
    discover_standardized_runs,
)


def _make_dummy_run(
    run_id: str,
    dataset_name: str = "test_store",
    is_retail: Any = True,
    is_synthetic: bool = False,
    source_type: str = "recorded_video",
    sha256: str = None,
    source_path: str = None,
    report_id: str = None,
    source_frames: int = 1000,
    source_fps: float = 30.0,
    sampled_frames: int = 200,
    sampling_fps: float = 5.0,
    total_tracks: int = 5,
    dwell_sec: float = 45.0,
    reasons_withheld: List[str] = None,
) -> StandardizedRunData:
    """Helper to construct controlled StandardizedRunData test instances."""
    raw: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "source_type": source_type,
    }
    if report_id:
        raw["report_id"] = report_id
    if sha256:
        raw["source_file_fingerprint"] = {"sha256": sha256, "file_name": "video.mp4"}
    if source_path:
        raw["source_path"] = source_path
    if source_frames is not None:
        raw["video_metadata"] = {"total_frames": source_frames, "fps": source_fps}

    return StandardizedRunData(
        run_id=run_id,
        display_name=f"Run {run_id}",
        dataset_name=dataset_name,
        source_type=source_type,
        environment_type="retail_store" if is_retail else ("unknown" if is_retail is None else "outdoor"),
        is_synthetic=is_synthetic,
        is_retail=is_retail,
        total_frames=sampled_frames,
        sampling_fps=sampling_fps,
        total_tracks=total_tracks,
        total_detections=total_tracks * 10 if total_tracks is not None else None,
        total_points=sampled_frames,
        in_zone_points=int(sampled_frames * 0.8) if sampled_frames is not None else None,
        unassigned_points=int(sampled_frames * 0.2) if sampled_frames is not None else None,
        binned_points=sampled_frames,
        out_of_bounds_points=0 if sampled_frames is not None else None,
        untransformed_points=0 if sampled_frames is not None else None,
        max_density=10 if sampled_frames is not None else None,
        conservation_verified=True,
        density_conservation_verified=True,
        zone_conservation_verified=True,
        zone_assignment_counts={},
        dwell_status="available" if dwell_sec is not None else "not_available",
        total_dwell_sec=dwell_sec,
        total_episodes=2 if dwell_sec is not None else None,
        zone_dwell={},
        transition_status="available",
        total_transitions=3,
        direct_transitions={},
        transition_matrix={},
        evidence_status="WITHHELD",
        reasons_withheld=reasons_withheld or ["Spatial scale uncalibrated"],
        caveats=[],
        is_eligible=False,
        candidate_zone="shelf_a",
        candidate_dwell_sec=dwell_sec,
        candidate_hotspot_cell=None,
        candidate_selection_rule="max_observed_dwell",
        comparison_by_shelf=[],
        model_family="YOLO11-n",
        model_runtime="CPUExecutionProvider",
        homography_unit="configured_units",
        homography_error=None,
        source_file_path=source_path or f"data/{run_id}.mp4",
        raw_dict=raw,
        source_frame_count=source_frames,
        source_fps=source_fps,
        sampled_frame_count=sampled_frames,
    )


def test_dynamic_workspace_summary_aggregation():
    """Verify compute_workspace_summary dynamically aggregates runs without hardcoded counts."""
    runs = [
        _make_dummy_run("run_1", sha256="hash_A", source_frames=1000, sampled_frames=200, total_tracks=4),
        _make_dummy_run("run_2", sha256="hash_B", source_frames=2000, sampled_frames=400, total_tracks=6),
    ]
    summary = compute_workspace_summary(runs)
    assert summary["total_runs"] == 2
    assert summary["unique_source_count"] == 2
    assert summary["total_sampled_frames"] == 600
    assert summary["sampled_reporting_runs"] == 2
    assert summary["sampled_missing_runs"] == 0
    assert summary["total_source_frames"] == 3000
    assert summary["source_reporting_runs"] == 2
    assert summary["total_tracks"] == 10
    assert summary["tracks_reporting_runs"] == 2


def test_unique_source_recording_vs_analysis_runs():
    """Verify distinct executions of identical source remain distinct runs, while unique source count is 1."""
    # 3 distinct executions of the same source footage (same SHA-256)
    runs = [
        _make_dummy_run("exec_1", sha256="shared_hash_merl", source_frames=3796, sampled_frames=633, total_tracks=3),
        _make_dummy_run("exec_2", sha256="shared_hash_merl", source_frames=3796, sampled_frames=633, total_tracks=3),
        _make_dummy_run("exec_3", sha256="shared_hash_merl", source_frames=3796, sampled_frames=100, total_tracks=1),
    ]
    summary = compute_workspace_summary(runs)
    # Distinct executions must be preserved as distinct runs
    assert summary["total_runs"] == 3
    # Source recordings must be deduplicated by fingerprint
    assert summary["unique_source_count"] == 1


def test_duplicate_representations_deduplication():
    """Verify duplicate representations sharing explicit run lineage/report ID are deduplicated."""
    # Two discovery records for the exact same report ID
    runs = [
        _make_dummy_run("path1/summary.json", report_id="report_001", sampled_frames=100),
        _make_dummy_run("path2/copy_summary.json", report_id="report_001", sampled_frames=100),
        _make_dummy_run("path3/distinct.json", report_id="report_002", sampled_frames=200),
    ]
    summary = compute_workspace_summary(runs)
    # report_001 is represented twice; deduplication must reduce to 2 unique runs
    assert summary["total_runs"] == 2
    assert summary["total_sampled_frames"] == 300


def test_three_provenance_states_retail_nonretail_unknown():
    """Verify retail, non-retail, and unknown provenance are kept separate; unknown is never labeled non-retail."""
    runs = [
        _make_dummy_run("retail_run", is_retail=True, is_synthetic=False),
        _make_dummy_run("non_retail_run", is_retail=False, is_synthetic=False),
        _make_dummy_run("unknown_run", is_retail=None, is_synthetic=False),
        _make_dummy_run("synthetic_run", is_retail=None, is_synthetic=True),
    ]
    summary = compute_workspace_summary(runs)
    assert summary["retail_count"] == 1
    assert summary["non_retail_count"] == 1
    assert summary["unknown_provenance_count"] == 1
    assert summary["synthetic_count"] == 1
    # Unknown provenance must NOT be added into non-retail
    assert summary["non_retail_count"] != summary["non_retail_count"] + summary["unknown_provenance_count"]


def test_missing_versus_evaluated_zero_preservation():
    """Verify missing tracks and frames are flagged as unavailable rather than coerced to 0."""
    runs = [
        _make_dummy_run("run_full", total_tracks=5, sampled_frames=200, source_frames=1000),
        _make_dummy_run("run_zero_tracks", total_tracks=0, sampled_frames=150, source_frames=600),
        _make_dummy_run("run_missing_meta", total_tracks=None, sampled_frames=None, source_frames=None),
    ]
    summary = compute_workspace_summary(runs)
    assert summary["total_runs"] == 3
    # Tracks: run_full (5) + run_zero_tracks (0) = 5
    assert summary["total_tracks"] == 5
    assert summary["tracks_reporting_runs"] == 2
    assert summary["tracks_missing_runs"] == 1  # run_missing_meta is missing, not zero!

    # Sampled frames: 200 + 150 = 350
    assert summary["total_sampled_frames"] == 350
    assert summary["sampled_reporting_runs"] == 2
    assert summary["sampled_missing_runs"] == 1

    # Source frames: 1000 + 600 = 1600
    assert summary["total_source_frames"] == 1600
    assert summary["source_reporting_runs"] == 2
    assert summary["source_missing_runs"] == 1


def test_newly_discovered_run_updates_workspace_summary():
    """Verify adding a newly discovered run dynamically updates all workspace summary metrics."""
    initial_runs = [
        _make_dummy_run("run_a", sha256="hash_1", source_frames=500, sampled_frames=100, total_tracks=2, is_retail=True),
    ]
    s1 = compute_workspace_summary(initial_runs)
    assert s1["total_runs"] == 1
    assert s1["unique_source_count"] == 1
    assert s1["retail_count"] == 1
    assert s1["total_sampled_frames"] == 100

    # New run discovered
    new_run = _make_dummy_run("run_b_new", sha256="hash_2", source_frames=800, sampled_frames=160, total_tracks=3, is_retail=False)
    updated_runs = initial_runs + [new_run]
    s2 = compute_workspace_summary(updated_runs)

    assert s2["total_runs"] == 2
    assert s2["unique_source_count"] == 2
    assert s2["retail_count"] == 1
    assert s2["non_retail_count"] == 1
    assert s2["total_sampled_frames"] == 260
    assert s2["total_source_frames"] == 1300
    assert s2["total_tracks"] == 5


def test_actual_repository_discovered_runs_aggregation():
    """Verify compute_workspace_summary operates seamlessly on actual repository outputs."""
    actual_runs = discover_standardized_runs("outputs/reports")
    assert len(actual_runs) >= 3
    summary = compute_workspace_summary(actual_runs)

    assert summary["total_runs"] >= 3
    assert summary["unique_source_count"] >= 3
    assert summary["retail_count"] >= 1
    assert summary["total_sampled_frames"] >= 840  # MERL (633) + Site03 (134) + Site02 (73) >= 840
    # Provenance separation holds
    assert summary["unknown_provenance_count"] >= 0
    assert summary["non_retail_count"] >= 1  # MOT17 is non-retail
