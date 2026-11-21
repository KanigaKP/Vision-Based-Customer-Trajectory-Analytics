"""
Focused tests for Phase 12.1 — Evidence-Gated Spatial Observation Prototype.
=============================================================================
Verifies:
1. Rejection of missing evidence context with explicit withholding reasons.
2. Rejection of non-retail environments (e.g., MOT17 pedestrian outdoor scene).
3. Rejection of boundary-truncated coverage (e.g., MERL video sample pattern).
4. Rejection of uncalibrated spatial setups.
5. Successful generation of plain, traceable descriptive observations under eligible context.
6. Exact traceability from emitted observations to input metrics.
7. Verification that tracker IDs are never labeled as unique people.
8. Verification that no zone rankings, 'popular', or layout recommendations are emitted.
9. Presence of configured-coordinate-unit and tracker-fragmentation caveats.
10. Serialization round-trips and frozen dataclass immutability.
11. Handling of empty or unpopulated summary inputs.
"""

import pytest
from typing import Dict, Any

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
    evaluate_evidence_eligibility,
    SpatialObservationEngine,
    generate_spatial_observations,
)


@pytest.fixture
def mock_dwell_summary() -> DwellSummary:
    """Create a deterministic synthetic DwellSummary across two zones."""
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
    ep2 = DwellEpisode(
        track_id=2,
        zone_id="shelf_a",
        entry_frame_idx=40,
        exit_frame_idx=70,
        entry_timestamp_sec=8.0,
        exit_timestamp_sec=14.0,
        duration_sec=6.0,
        point_count=7,
    )
    ep3 = DwellEpisode(
        track_id=1,
        zone_id="checkout",
        entry_frame_idx=80,
        exit_frame_idx=100,
        entry_timestamp_sec=16.0,
        exit_timestamp_sec=20.0,
        duration_sec=4.0,
        point_count=5,
    )

    t1_summary = TrackDwellSummary(
        track_id=1,
        total_dwell_sec=8.0,
        total_episodes=2,
        zone_dwell_sec={"shelf_a": 4.0, "checkout": 4.0},
        zone_episode_count={"shelf_a": 1, "checkout": 1},
        episodes=[ep1, ep3],
    )
    t2_summary = TrackDwellSummary(
        track_id=2,
        total_dwell_sec=6.0,
        total_episodes=1,
        zone_dwell_sec={"shelf_a": 6.0},
        zone_episode_count={"shelf_a": 1},
        episodes=[ep2],
    )

    za_summary = ZoneDwellSummary(
        zone_id="shelf_a",
        total_dwell_sec=10.0,
        total_episodes=2,
        unique_tracks=2,
        mean_dwell_sec=5.0,
        median_dwell_sec=5.0,
        min_dwell_sec=4.0,
        max_dwell_sec=6.0,
        episodes=[ep1, ep2],
    )
    zc_summary = ZoneDwellSummary(
        zone_id="checkout",
        total_dwell_sec=4.0,
        total_episodes=1,
        unique_tracks=1,
        mean_dwell_sec=4.0,
        median_dwell_sec=4.0,
        min_dwell_sec=4.0,
        max_dwell_sec=4.0,
        episodes=[ep3],
    )

    return DwellSummary(
        total_episodes=3,
        total_dwell_sec=14.0,
        by_track={1: t1_summary, 2: t2_summary},
        by_zone={"shelf_a": za_summary, "checkout": zc_summary},
        episodes=[ep1, ep2, ep3],
    )


@pytest.fixture
def mock_transition_summary() -> DirectTransitionSummary:
    """Create a deterministic synthetic DirectTransitionSummary."""
    zs_a = ZoneActivitySummary(
        zone_id="shelf_a",
        observed_run_count=2,
        distinct_track_count=2,
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
        total_tracks=2,
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


@pytest.fixture
def mock_density_grid() -> TrajectoryDensityGrid:
    """Create a deterministic synthetic TrajectoryDensityGrid."""
    import numpy as np
    bounds = GridBounds(min_x=0.0, max_x=10.0, min_y=0.0, max_y=6.0)
    counts = np.zeros((12, 20), dtype=np.int32)
    counts[11, 19] = 3
    return TrajectoryDensityGrid(
        bounds=bounds,
        counts=counts,
        num_bins_x=20,
        num_bins_y=12,
        cell_size_x=0.5,
        cell_size_y=0.5,
        total_evaluated_points=74,
        total_binned_points=3,
        out_of_bounds_count=71,
        untransformed_count=0,
    )


def test_missing_evidence_context_withholds_observations():
    """Verify that passing None as evidence context immediately withholds observations."""
    engine = SpatialObservationEngine()
    result = engine.analyze(evidence_context=None)
    assert not result.is_eligible
    assert result.status == "insufficient_evidence"
    assert len(result.observations) == 0
    assert any("Evidence context is missing" in r for r in result.reasons_withheld)


def test_mot17_like_non_retail_environment_withholds_observations(mock_transition_summary):
    """
    Verify that an outdoor pedestrian environment with unverified calibration
    (synthetic fixture representing MOT17-04 metadata) is rejected with explicit
    reasons for both non-retail environment and unverified calibration.
    """
    ctx = EvidenceContext(
        environment_type="pedestrian_outdoor",
        calibration_status="unverified_configured_units",
        coverage_status="representative",
        notes="Synthetic fixture: MOT17-04 outdoor pedestrian tracking sample",
    )
    result = generate_spatial_observations(
        evidence_context=ctx,
        transition_summary=mock_transition_summary,
    )
    assert not result.is_eligible
    assert result.status == "insufficient_evidence"
    assert len(result.observations) == 0
    assert any("not a retail store environment" in r for r in result.reasons_withheld)
    assert any("unverified or configured-only coordinates" in r for r in result.reasons_withheld)


def test_merl_like_boundary_truncated_coverage_withholds_observations(
    mock_dwell_summary, mock_transition_summary, mock_density_grid
):
    """
    Verify that boundary-truncated coverage with unverified calibration
    (synthetic fixture representing local MERL run metadata) is explicitly
    withheld from retail spatial observation generation.
    """
    ctx = EvidenceContext(
        environment_type="retail_store",
        calibration_status="unverified_configured_units",
        coverage_status="boundary_truncated",
        notes="Synthetic fixture: MERL recorded video run with boundary truncation",
    )
    result = generate_spatial_observations(
        evidence_context=ctx,
        dwell_summary=mock_dwell_summary,
        transition_summary=mock_transition_summary,
        density_grid=mock_density_grid,
    )
    assert not result.is_eligible
    assert result.status == "insufficient_evidence"
    assert len(result.observations) == 0
    assert any("boundary_truncated" in r for r in result.reasons_withheld)
    assert any("unverified or configured-only coordinates" in r for r in result.reasons_withheld)


def test_uncalibrated_status_withholds_observations(mock_transition_summary):
    """Verify that an uncalibrated setup is rejected with explicit calibration notice."""
    ctx = EvidenceContext(
        environment_type="retail_store",
        calibration_status="uncalibrated",
        coverage_status="representative",
        notes="Synthetic fixture: Uncalibrated camera setup",
    )
    result = generate_spatial_observations(
        evidence_context=ctx,
        transition_summary=mock_transition_summary,
    )
    assert not result.is_eligible
    assert result.status == "insufficient_evidence"
    assert any("uncalibrated" in r for r in result.reasons_withheld)


def test_unverified_calibration_withholds_observations(mock_transition_summary):
    """
    Verify that unverified or configured-only calibration status is rejected even
    if environment is retail and coverage is representative.
    """
    ctx = EvidenceContext(
        environment_type="retail_store",
        calibration_status="unverified_configured_units",
        coverage_status="representative",
        notes="Synthetic fixture: Retail store with unverified configured calibration",
    )
    result = generate_spatial_observations(
        evidence_context=ctx,
        transition_summary=mock_transition_summary,
    )
    assert not result.is_eligible
    assert result.status == "insufficient_evidence"
    assert len(result.observations) == 0
    assert any("unverified or configured-only coordinates" in r for r in result.reasons_withheld)


def test_multiple_ineligibility_reasons_accumulate():
    """Verify that multiple disqualifications (non-retail AND boundary-truncated) are reported."""
    ctx = EvidenceContext(
        environment_type="pedestrian_outdoor",
        calibration_status="uncalibrated",
        coverage_status="boundary_truncated",
    )
    is_eligible, reasons = evaluate_evidence_eligibility(ctx)
    assert not is_eligible
    assert len(reasons) == 3
    assert any("not a retail store environment" in r for r in reasons)
    assert any("boundary_truncated" in r for r in reasons)
    assert any("uncalibrated" in r for r in reasons)


def test_eligible_evidence_context_produces_traceable_observations(
    mock_dwell_summary, mock_transition_summary, mock_density_grid
):
    """
    Verify that an eligible retail context generates plain, traceable descriptive
    observations for zone activity, zone dwell, direct transitions, and density grid.
    (Synthetic fixture with independently checked calibration).
    """
    ctx = EvidenceContext(
        environment_type="retail_store",
        calibration_status="independently_checked",
        coverage_status="representative",
        notes="Synthetic fixture: Controlled retail test sequence with independently checked calibration",
    )
    result = generate_spatial_observations(
        evidence_context=ctx,
        dwell_summary=mock_dwell_summary,
        transition_summary=mock_transition_summary,
        density_grid=mock_density_grid,
    )
    assert result.is_eligible
    assert result.status == "sufficient_evidence"
    assert len(result.reasons_withheld) == 0
    assert len(result.observations) > 0

    obs_types = {obs.observation_type for obs in result.observations}
    assert "zone_activity" in obs_types
    assert "zone_dwell_span" in obs_types
    assert "direct_transition" in obs_types
    assert "density_grid" in obs_types

    # Verify exact traceability of zone_activity observation
    shelf_a_act = next(
        obs for obs in result.observations
        if obs.observation_type == "zone_activity" and obs.zone_id == "shelf_a"
    )
    assert shelf_a_act.source_metrics["observed_run_count"] == 2
    assert shelf_a_act.source_metrics["distinct_track_count"] == 2
    assert shelf_a_act.source_metrics["direct_incoming_count"] == 0
    assert shelf_a_act.source_metrics["direct_outgoing_count"] == 1
    assert "2 observed visit run(s)" in shelf_a_act.summary_text
    assert "2 distinct clip-local track ID(s)" in shelf_a_act.summary_text

    # Verify exact traceability of zone_dwell observation
    shelf_a_dwell = next(
        obs for obs in result.observations
        if obs.observation_type == "zone_dwell_span" and obs.zone_id == "shelf_a"
    )
    assert shelf_a_dwell.source_metrics["total_episodes"] == 2
    assert shelf_a_dwell.source_metrics["total_dwell_sec"] == 10.0
    assert shelf_a_dwell.source_metrics["mean_dwell_sec"] == 5.0
    assert shelf_a_dwell.source_metrics["median_dwell_sec"] == 5.0

    # Verify exact traceability of transition observation
    trans_obs = next(
        obs for obs in result.observations
        if obs.observation_type == "direct_transition"
    )
    assert trans_obs.source_metrics["from_zone"] == "shelf_a"
    assert trans_obs.source_metrics["to_zone"] == "checkout"
    assert trans_obs.source_metrics["transition_count"] == 1

    # Verify exact traceability of density grid observation
    dens_obs = next(
        obs for obs in result.observations
        if obs.observation_type == "density_grid"
    )
    assert dens_obs.source_metrics["total_evaluated_points"] == 74
    assert dens_obs.source_metrics["total_binned_points"] == 3
    assert dens_obs.source_metrics["out_of_bounds_count"] == 71


def test_language_is_strictly_non_causal_and_includes_caveats(
    mock_dwell_summary, mock_transition_summary, mock_density_grid
):
    """
    Verify that emitted observations strictly avoid causal or evaluative language
    (no 'popular', 'top', 'customer preference', 'browsing', 'engaged') and include
    required caveats on tracking fragmentation and observed spans.
    (Synthetic fixture with independently checked calibration in configured units).
    """
    ctx = EvidenceContext(
        environment_type="retail_store",
        calibration_status="independently_checked",
        coverage_status="representative",
        notes="Synthetic fixture: Evaluates non-causal language and caveats",
    )
    result = generate_spatial_observations(
        evidence_context=ctx,
        dwell_summary=mock_dwell_summary,
        transition_summary=mock_transition_summary,
        density_grid=mock_density_grid,
    )
    forbidden_terms = [
        "popular", "best", "worst", "top", "rank", "underperforming",
        "engagement", "intent", "prefer", "browse", "conversion",
        "recommend", "relocate", "unique customer", "unique people",
    ]
    for obs in result.observations:
        text_lower = obs.summary_text.lower()
        for term in forbidden_terms:
            assert term not in text_lower, f"Forbidden term '{term}' found in observation: {obs.summary_text}"

        # Caveats must be populated
        assert len(obs.caveats) >= 1
        caveat_text = " ".join(obs.caveats).lower()
        # Must mention configured units when calibration is unverified metric
        if obs.observation_type in ("zone_activity", "zone_dwell_span", "density_grid"):
            assert "configured floor units" in caveat_text or "clip-local" in caveat_text


def test_empty_summaries_withholds_observations():
    """Verify that an eligible context with zero input data produces an insufficient evidence result."""
    ctx = EvidenceContext(
        environment_type="retail_store",
        calibration_status="independently_checked",
        coverage_status="representative",
        notes="Synthetic fixture: Zero input summaries",
    )
    result = generate_spatial_observations(
        evidence_context=ctx,
        dwell_summary=None,
        transition_summary=None,
        density_grid=None,
    )
    assert not result.is_eligible
    assert result.status == "insufficient_evidence"
    assert len(result.observations) == 0
    assert any("No populated trajectory observations" in r for r in result.reasons_withheld)


def test_evidence_context_validation_rejections():
    """Verify that invalid strings or empty fields in EvidenceContext raise ValueError."""
    with pytest.raises(ValueError, match="Invalid environment_type"):
        EvidenceContext(environment_type="", calibration_status="metric", coverage_status="rep")

    with pytest.raises(ValueError, match="Invalid calibration_status"):
        EvidenceContext(environment_type="retail", calibration_status="  ", coverage_status="rep")

    with pytest.raises(ValueError, match="Invalid coverage_status"):
        EvidenceContext(environment_type="retail", calibration_status="metric", coverage_status="")


def test_serialization_round_trip(mock_dwell_summary, mock_transition_summary):
    """Verify that SpatialObservationResult, EvidenceContext, and Observations serialize cleanly."""
    ctx = EvidenceContext(
        environment_type="retail_store",
        calibration_status="independently_checked",
        coverage_status="representative",
        notes="Synthetic fixture: Serialization round trip test",
    )
    result = generate_spatial_observations(
        evidence_context=ctx,
        dwell_summary=mock_dwell_summary,
        transition_summary=mock_transition_summary,
    )
    d = result.to_dict()
    assert d["is_eligible"] is True
    assert d["status"] == "sufficient_evidence"
    assert isinstance(d["observations"], list)
    assert len(d["observations"]) > 0
    assert d["evidence_context"]["environment_type"] == "retail_store"
    assert "input_diagnostics" in d


def test_frozen_dataclass_immutability():
    """Verify that EvidenceContext and DescriptiveSpatialObservation are immutable."""
    ctx = EvidenceContext(
        environment_type="retail_store",
        calibration_status="independently_checked",
        coverage_status="representative",
        notes="Synthetic fixture: Immutability check",
    )
    with pytest.raises(AttributeError):
        ctx.environment_type = "other"  # type: ignore

    obs = DescriptiveSpatialObservation(
        observation_id="TEST",
        observation_type="test",
        zone_id=None,
        summary_text="Text",
        source_metrics={"count": 1},
        caveats=("Caveat",),
    )
    with pytest.raises(AttributeError):
        obs.summary_text = "Mutated"  # type: ignore
