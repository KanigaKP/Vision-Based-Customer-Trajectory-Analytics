"""
Focused tests for Phase 13.1 — Local Evidence-Aware Analysis Report Export.
=============================================================================
Verifies:
1. Versioned schema adherence (`schema_version = "1.0.0"`).
2. Required explicit run metadata validation and rejection of invalid fields.
3. Complete inputs serialization (Phase 9, 10, 11, 12).
4. Partial inputs serialization: missing inputs are marked 'not_available',
   never silently filled with zero.
5. Evaluated zero-count metrics are distinguishable from missing/unavailable data.
6. Serialization of Phase 12 insufficient-evidence results (MERL & MOT17 cases).
7. Privacy boundaries: no raw person histories or point lists are exposed;
   track counts are labeled as clip-local tracker IDs.
8. Methodological caveats for dwell spans, point density, and configured units.
9. Deterministic JSON serialization.
10. File export to established reports directory and round-trip readability.
"""

import os
import json
import pytest
from typing import Dict, Any

import numpy as np

from src.analytics.dwell import (
    DwellEpisode,
    TrackDwellSummary,
    ZoneDwellSummary,
    DwellSummary,
)
from src.analytics.density import (
    GridBounds,
    TrajectoryDensityGrid,
)
from src.analytics.transitions import (
    ZoneActivitySummary,
    DirectTransitionSummary,
)
from src.analytics.spatial_observations import (
    EvidenceContext,
    DescriptiveSpatialObservation,
    SpatialObservationResult,
    generate_spatial_observations,
)
from src.reporting.analysis_report import (
    SCHEMA_VERSION,
    ALLOWED_SOURCE_TYPES,
    AnalysisRunMetadata,
    AnalysisReport,
    build_analysis_report,
    export_analysis_report,
    generate_and_export_analysis_report,
)


@pytest.fixture
def mock_run_metadata() -> AnalysisRunMetadata:
    """Deterministic synthetic run metadata fixture."""
    return AnalysisRunMetadata(
        run_id="test_run_001",
        dataset_name="synthetic_retail_benchmark",
        source_type="recorded_video",
        target_fps=5.0,
        total_frames_evaluated=100,
        source_path="data/test/clip.mp4",
        notes="Synthetic test fixture for report export verification",
    )


@pytest.fixture
def mock_dwell_summary() -> DwellSummary:
    """Deterministic synthetic DwellSummary fixture."""
    ep1 = DwellEpisode(
        track_id=1,
        zone_id="shelf_a",
        entry_frame_idx=10,
        exit_frame_idx=30,
        entry_timestamp_sec=2.0,
        exit_timestamp_sec=6.0,
        duration_sec=4.0,
        point_count=5,
    )
    t1 = TrackDwellSummary(
        track_id=1,
        total_dwell_sec=4.0,
        total_episodes=1,
        zone_dwell_sec={"shelf_a": 4.0},
        zone_episode_count={"shelf_a": 1},
        episodes=[ep1],
    )
    za = ZoneDwellSummary(
        zone_id="shelf_a",
        total_dwell_sec=4.0,
        total_episodes=1,
        unique_tracks=1,
        mean_dwell_sec=4.0,
        median_dwell_sec=4.0,
        min_dwell_sec=4.0,
        max_dwell_sec=4.0,
        episodes=[ep1],
    )
    return DwellSummary(
        total_episodes=1,
        total_dwell_sec=4.0,
        by_track={1: t1},
        by_zone={"shelf_a": za},
        episodes=[ep1],
    )


@pytest.fixture
def mock_density_grid() -> TrajectoryDensityGrid:
    """Deterministic synthetic TrajectoryDensityGrid fixture."""
    bounds = GridBounds(min_x=0.0, max_x=10.0, min_y=0.0, max_y=6.0)
    counts = np.zeros((12, 20), dtype=np.int32)
    counts[5, 5] = 2
    return TrajectoryDensityGrid(
        bounds=bounds,
        counts=counts,
        num_bins_x=20,
        num_bins_y=12,
        cell_size_x=0.5,
        cell_size_y=0.5,
        total_evaluated_points=10,
        total_binned_points=2,
        out_of_bounds_count=8,
        untransformed_count=0,
    )


@pytest.fixture
def mock_transition_summary() -> DirectTransitionSummary:
    """Deterministic synthetic DirectTransitionSummary fixture."""
    zs_a = ZoneActivitySummary(
        zone_id="shelf_a",
        observed_run_count=1,
        distinct_track_count=1,
        direct_incoming_count=0,
        direct_outgoing_count=1,
    )
    zs_c = ZoneActivitySummary(
        zone_id="checkout",
        observed_run_count=1,
        distinct_track_count=1,
        direct_incoming_count=1,
        direct_outgoing_count=0,
    )
    return DirectTransitionSummary(
        total_tracks=1,
        total_transitions=1,
        known_zone_ids=["shelf_a", "checkout"],
        transition_counts={("shelf_a", "checkout"): 1},
        transition_matrix={
            "shelf_a": {"shelf_a": 0, "checkout": 1},
            "checkout": {"shelf_a": 0, "checkout": 0},
        },
        track_sequences=[],
        zone_summaries={"shelf_a": zs_a, "checkout": zs_c},
    )


def test_metadata_validation_rejections():
    """Verify that AnalysisRunMetadata rejects invalid or empty fields and disallowed source types."""
    with pytest.raises(ValueError, match="Invalid run_id"):
        AnalysisRunMetadata(run_id="", dataset_name="data", source_type="recorded_video")

    with pytest.raises(ValueError, match="Invalid dataset_name"):
        AnalysisRunMetadata(run_id="run1", dataset_name="   ", source_type="recorded_video")

    with pytest.raises(ValueError, match="Invalid source_type"):
        AnalysisRunMetadata(run_id="run1", dataset_name="data", source_type="")

    with pytest.raises(ValueError, match="Invalid source_type"):
        AnalysisRunMetadata(run_id="run1", dataset_name="data", source_type="   ")

    with pytest.raises(ValueError, match="Invalid target_fps"):
        AnalysisRunMetadata(run_id="run1", dataset_name="data", source_type="recorded_video", target_fps=-1.0)

    with pytest.raises(ValueError, match="Invalid total_frames_evaluated"):
        AnalysisRunMetadata(run_id="run1", dataset_name="data", source_type="recorded_video", total_frames_evaluated=-5)


def test_source_type_prohibits_live_streams_and_accepts_recorded_sources():
    """
    Verify that live_stream, RTSP, and live CCTV are rejected as outside project scope,
    while recorded_video, image_sequence, synthetic_fixture, and controlled_trial
    (meaning recorded project footage) are accepted.
    """
    assert "live_stream" not in ALLOWED_SOURCE_TYPES
    for prohibited in ("live_stream", "rtsp_stream", "live_cctv", "webcam_live", "rtmp", "video"):
        with pytest.raises(ValueError, match="Invalid source_type"):
            AnalysisRunMetadata(run_id="test", dataset_name="ds", source_type=prohibited)

    # All 4 permitted offline/recorded source types must be valid
    for allowed in ("recorded_video", "image_sequence", "synthetic_fixture", "controlled_trial"):
        assert allowed in ALLOWED_SOURCE_TYPES
        meta = AnalysisRunMetadata(run_id="test", dataset_name="ds", source_type=allowed)
        assert meta.source_type == allowed


def test_build_report_requires_valid_metadata():
    """Verify build_analysis_report raises TypeError if metadata is not AnalysisRunMetadata."""
    with pytest.raises(TypeError, match="metadata must be an AnalysisRunMetadata instance"):
        build_analysis_report(metadata={"run_id": "test"})  # type: ignore


def test_complete_inputs_report(
    mock_run_metadata,
    mock_dwell_summary,
    mock_density_grid,
    mock_transition_summary,
):
    """Verify building a report with all Phase 9-12 inputs populated."""
    ctx = EvidenceContext(
        environment_type="retail_store",
        calibration_status="independently_checked",
        coverage_status="representative",
        notes="Synthetic fixture: complete report check",
    )
    obs_res = generate_spatial_observations(
        evidence_context=ctx,
        dwell_summary=mock_dwell_summary,
        transition_summary=mock_transition_summary,
        density_grid=mock_density_grid,
    )

    report = build_analysis_report(
        metadata=mock_run_metadata,
        evidence_context=ctx,
        dwell_summary=mock_dwell_summary,
        density_grid=mock_density_grid,
        transition_summary=mock_transition_summary,
        observation_result=obs_res,
        generated_at_utc="2026-10-30T00:00:00Z",
    )

    d = report.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["report_id"] == "report_test_run_001"
    assert d["generated_at_utc"] == "2026-10-30T00:00:00Z"
    assert d["metadata"]["dataset_name"] == "synthetic_retail_benchmark"

    # All sections must be marked available
    assert d["evidence_context"]["status"] == "available"
    assert d["dwell_analytics"]["status"] == "available"
    assert d["density_analytics"]["status"] == "available"
    assert d["transition_analytics"]["status"] == "available"
    assert d["spatial_observations"]["status"] == "sufficient_evidence"
    assert d["spatial_observations"]["is_eligible"] is True
    assert len(d["spatial_observations"]["observations"]) > 0

    # Evidence summary
    assert d["evidence_summary"]["spatial_observation_eligibility"] == "sufficient_evidence"
    assert d["evidence_summary"]["classification"]["implemented"] is True
    assert d["evidence_summary"]["classification"]["quantitatively_validated"] is False


def test_partial_inputs_report_marks_missing_as_not_available(mock_run_metadata, mock_dwell_summary):
    """
    Verify that unsupplied sections (density, transitions, observations) are explicitly
    marked 'not_available', NEVER silently filled with zero or empty dicts.
    """
    report = build_analysis_report(
        metadata=mock_run_metadata,
        dwell_summary=mock_dwell_summary,
        density_grid=None,
        transition_summary=None,
        observation_result=None,
        evidence_context=None,
    )
    d = report.to_dict()

    # Dwell was supplied
    assert d["dwell_analytics"]["status"] == "available"
    assert d["dwell_analytics"]["total_episodes"] == 1

    # Others were NOT supplied
    assert d["density_analytics"]["status"] == "not_available"
    assert "reason" in d["density_analytics"]
    assert "total_evaluated_points" not in d["density_analytics"]

    assert d["transition_analytics"]["status"] == "not_available"
    assert "reason" in d["transition_analytics"]
    assert "total_transitions" not in d["transition_analytics"]

    assert d["spatial_observations"]["status"] == "not_available"
    assert "reason" in d["spatial_observations"]
    assert "observations" not in d["spatial_observations"]

    assert d["evidence_context"]["status"] == "not_available"
    assert "reason" in d["evidence_context"]


def test_missing_versus_zero_distinction(mock_run_metadata):
    """
    Verify that an evaluated zero count (e.g. zone with 0 runs) is distinguishable
    from an unsupplied/unavailable summary.
    """
    # 1. Supplied transition summary with zero activity in a known zone
    zs_empty = ZoneActivitySummary(
        zone_id="unvisited_aisle",
        observed_run_count=0,
        distinct_track_count=0,
        direct_incoming_count=0,
        direct_outgoing_count=0,
    )
    trans_summary = DirectTransitionSummary(
        total_tracks=0,
        total_transitions=0,
        known_zone_ids=["unvisited_aisle"],
        transition_counts={},
        transition_matrix={"unvisited_aisle": {"unvisited_aisle": 0}},
        track_sequences=[],
        zone_summaries={"unvisited_aisle": zs_empty},
    )

    report_supplied_zero = build_analysis_report(
        metadata=mock_run_metadata,
        transition_summary=trans_summary,
    )
    d1 = report_supplied_zero.to_dict()
    assert d1["transition_analytics"]["status"] == "available"
    assert d1["transition_analytics"]["zone_activity_summaries"]["unvisited_aisle"]["observed_run_count"] == 0

    # 2. Unsupplied transition summary
    report_unsupplied = build_analysis_report(
        metadata=mock_run_metadata,
        transition_summary=None,
    )
    d2 = report_unsupplied.to_dict()
    assert d2["transition_analytics"]["status"] == "not_available"
    assert "zone_activity_summaries" not in d2["transition_analytics"]


def test_merl_insufficient_evidence_serialization(
    mock_dwell_summary,
    mock_density_grid,
    mock_transition_summary,
):
    """
    Verify report serialization of the Phase 12 MERL insufficient-evidence result
    (boundary truncation and unverified calibration).
    """
    merl_meta = AnalysisRunMetadata(
        run_id="merl_run_100frames",
        dataset_name="merl_shopping_dataset",
        source_type="recorded_video",
        target_fps=5.0,
        total_frames_evaluated=100,
        source_path="data/merl/10_1_crop.mp4",
        notes="Synthetic fixture representing MERL 100-frame run metadata",
    )
    merl_ctx = EvidenceContext(
        environment_type="retail_store",
        calibration_status="unverified_configured_units",
        coverage_status="boundary_truncated",
        notes="Synthetic fixture: MERL boundary-truncated coverage",
    )
    obs_res = generate_spatial_observations(
        evidence_context=merl_ctx,
        dwell_summary=mock_dwell_summary,
        transition_summary=mock_transition_summary,
        density_grid=mock_density_grid,
    )
    assert not obs_res.is_eligible

    report = build_analysis_report(
        metadata=merl_meta,
        evidence_context=merl_ctx,
        dwell_summary=mock_dwell_summary,
        density_grid=mock_density_grid,
        transition_summary=mock_transition_summary,
        observation_result=obs_res,
    )
    d = report.to_dict()

    assert d["spatial_observations"]["status"] == "insufficient_evidence"
    assert d["spatial_observations"]["is_eligible"] is False
    assert d["spatial_observations"]["observations_count"] == 0
    assert len(d["spatial_observations"]["observations"]) == 0
    assert len(d["spatial_observations"]["reasons_withheld"]) >= 2
    assert any("boundary_truncated" in r for r in d["spatial_observations"]["reasons_withheld"])
    assert any("unverified or configured-only" in r for r in d["spatial_observations"]["reasons_withheld"])
    assert d["evidence_summary"]["spatial_observation_eligibility"] == "insufficient_evidence"


def test_mot17_insufficient_evidence_serialization(mock_transition_summary):
    """
    Verify report serialization of the Phase 12 MOT17 insufficient-evidence result
    (outdoor pedestrian environment and unverified calibration).
    """
    mot17_meta = AnalysisRunMetadata(
        run_id="mot17_04_run_10frames",
        dataset_name="mot17_04_frcnn",
        source_type="image_sequence",
        total_frames_evaluated=10,
        source_path="data/mot17/train/MOT17-04-FRCNN/img1",
        notes="Synthetic fixture representing MOT17 run metadata",
    )
    mot17_ctx = EvidenceContext(
        environment_type="pedestrian_outdoor",
        calibration_status="unverified_configured_units",
        coverage_status="representative",
        notes="Synthetic fixture: MOT17 non-retail outdoor scene",
    )
    obs_res = generate_spatial_observations(
        evidence_context=mot17_ctx,
        transition_summary=mock_transition_summary,
    )
    assert not obs_res.is_eligible

    report = build_analysis_report(
        metadata=mot17_meta,
        evidence_context=mot17_ctx,
        transition_summary=mock_transition_summary,
        observation_result=obs_res,
    )
    d = report.to_dict()

    assert d["spatial_observations"]["status"] == "insufficient_evidence"
    assert d["spatial_observations"]["is_eligible"] is False
    assert any("not a retail store environment" in r for r in d["spatial_observations"]["reasons_withheld"])
    assert any("unverified or configured-only" in r for r in d["spatial_observations"]["reasons_withheld"])


def test_privacy_boundaries_and_clip_local_track_labels(
    mock_run_metadata,
    mock_dwell_summary,
    mock_transition_summary,
):
    """
    Verify report does not expose raw person points, trajectory paths, or unaggregated
    tracking sequences, and labels counts as clip-local tracker IDs.
    """
    report = build_analysis_report(
        metadata=mock_run_metadata,
        dwell_summary=mock_dwell_summary,
        transition_summary=mock_transition_summary,
    )
    json_str = report.to_json()

    # Must NOT contain raw person identities or raw trajectory points
    assert "customer_identity" not in json_str
    assert "person_id" not in json_str
    assert "trajectory_points" not in json_str

    # Must label track counts as clip-local
    assert "distinct_clip_local_tracks" in json_str
    assert "distinct_clip_local_track_count" in json_str


def test_caveats_populated_in_all_available_sections(
    mock_run_metadata,
    mock_dwell_summary,
    mock_density_grid,
    mock_transition_summary,
):
    """Verify methodological caveats are attached to each analytics section."""
    report = build_analysis_report(
        metadata=mock_run_metadata,
        dwell_summary=mock_dwell_summary,
        density_grid=mock_density_grid,
        transition_summary=mock_transition_summary,
    )
    d = report.to_dict()

    # Dwell caveats
    dwell_caveats = " ".join(d["dwell_analytics"]["caveats"]).lower()
    assert "temporal spans" in dwell_caveats
    assert "not continuous" in dwell_caveats

    # Density caveats
    density_caveats = " ".join(d["density_analytics"]["caveats"]).lower()
    assert "frame sampling rate" in density_caveats
    assert "not measure unique visitors" in density_caveats
    assert "configured floor units" in density_caveats

    # Transition caveats
    trans_caveats = " ".join(d["transition_analytics"]["caveats"]).lower()
    assert "clip-local tracking histories" in trans_caveats
    assert "not unique human individuals" in trans_caveats


def test_deterministic_serialization(mock_run_metadata, mock_dwell_summary, mock_transition_summary):
    """Verify that multiple serialization calls produce byte-for-byte identical output."""
    report1 = build_analysis_report(
        metadata=mock_run_metadata,
        dwell_summary=mock_dwell_summary,
        transition_summary=mock_transition_summary,
        generated_at_utc="2026-10-30T08:00:00Z",
    )
    report2 = build_analysis_report(
        metadata=mock_run_metadata,
        dwell_summary=mock_dwell_summary,
        transition_summary=mock_transition_summary,
        generated_at_utc="2026-10-30T08:00:00Z",
    )
    assert report1.to_json() == report2.to_json()


def test_export_analysis_report_to_disk(tmp_path, mock_run_metadata, mock_dwell_summary):
    """Verify exporting an analysis report to a JSON file and reading it back."""
    target_json = str(tmp_path / "test_report.json")
    report = build_analysis_report(
        metadata=mock_run_metadata,
        dwell_summary=mock_dwell_summary,
        generated_at_utc="2026-10-30T08:00:00Z",
    )
    exported_path = export_analysis_report(report, output_path=target_json)
    assert exported_path == target_json
    assert os.path.exists(target_json)

    with open(target_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["report_id"] == "report_test_run_001"
    assert loaded["dwell_analytics"]["status"] == "available"
    assert loaded["density_analytics"]["status"] == "not_available"


def test_generate_and_export_convenience_function(tmp_path, mock_run_metadata):
    """Verify convenience function builds and exports in one step."""
    target_json = str(tmp_path / "convenience_report.json")
    exported_path = generate_and_export_analysis_report(
        metadata=mock_run_metadata,
        output_path=target_json,
        generated_at_utc="2026-10-30T08:00:00Z",
    )
    assert os.path.exists(exported_path)
    with open(exported_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["report_id"] == "report_test_run_001"
