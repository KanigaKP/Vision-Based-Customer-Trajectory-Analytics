"""
Unit and integration tests for explainable candidate diagnostics.
==================================================================
Covers 10 core evaluation scenarios:
1. Two observed candidate zones with unambiguous leader and runner-up margin.
2. One observed candidate zone with transparent single-zone message.
3. No observations across any candidate zone (no-candidate state).
4. Exact metric ties across candidate zones.
5. Objective divergence (e.g. high dwell in zone A vs. high point count in zone B).
6. Multi-track vs single fragmented track lineage accounting.
7. Custom non-MERL bounds and zone identifiers.
8. Safe handling of missing hotspot cell data.
9. Tied candidate cells on peak count within zone.
10. Out-of-bounds heavy runs with point conservation integrity.
"""

import pytest
from typing import Dict, Any

from src.analytics.candidate_diagnostics import (
    CandidateDiagnosticsResult,
    CandidateZoneMetrics,
    ObjectiveEvaluation,
    compute_hotspot_with_track_lineage,
    evaluate_single_objective,
    generate_candidate_diagnostics,
    SUPPORTED_OBJECTIVES,
)


def _make_base_run_dict(
    candidate_zone_ids=None,
    zone_dwell=None,
    zone_points=None,
    total_eval=100,
    binned=100,
    oob=0,
    untransformed=0,
) -> Dict[str, Any]:
    """Helper to build test run dictionary matching Phase 13 / demo summary structure."""
    cand_ids = candidate_zone_ids if candidate_zone_ids is not None else ["shelf_a", "shelf_b"]
    z_dwell = zone_dwell or {}
    z_pts = zone_points or {}

    per_zone_dwell = {}
    for zid in cand_ids:
        if zid in z_dwell:
            per_zone_dwell[zid] = {
                "observed_dwell_span_sec": z_dwell[zid].get("dwell_sec", 0.0),
                "total_episodes": z_dwell[zid].get("episodes", 0),
                "distinct_tracks_count": z_dwell[zid].get("tracks", 0),
            }

    per_zone_counts = {}
    for zid in cand_ids:
        if zid in z_pts:
            per_zone_counts[zid] = z_pts[zid]

    return {
        "run_id": "test_run_001",
        "dataset_name": "synthetic_test_dataset",
        "scene_profile": {
            "bounds": [0.0, 10.0, 0.0, 6.0],
            "cell_size": 0.5,
            "zones": [
                {"zone_id": zid, "polygon": [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]}
                for zid in cand_ids
            ],
        },
        "density_analytics": {
            "bounds": {"min_x": 0.0, "max_x": 10.0, "min_y": 0.0, "max_y": 6.0},
            "cell_size": [0.5, 0.5],
            "total_evaluated_points": total_eval,
            "total_binned_points": binned,
            "out_of_bounds_count": oob,
            "untransformed_count": untransformed,
        },
        "dwell_analytics": {
            "status": "available",
            "per_zone": per_zone_dwell,
        },
        "zone_assignment_counts": {
            "per_zone": per_zone_counts,
        },
        "evidence_status": {
            "is_eligible": False,
            "actionable_recommendations": "WITHHELD",
            "reasons_withheld": ["Single-camera boundary truncation", "Uncalibrated metric floor units"],
        },
    }


# ---------------------------------------------------------------------------
# 1. Two Observed Candidate Zones
# ---------------------------------------------------------------------------

def test_two_observed_candidate_zones_leader_and_runner_up():
    """Verify that when 2 candidate zones have observations, proper lead margin and ranking are produced."""
    run_dict = _make_base_run_dict(
        candidate_zone_ids=["shelf_a", "shelf_b"],
        zone_dwell={
            "shelf_a": {"dwell_sec": 10.0, "episodes": 2, "tracks": 1},
            "shelf_b": {"dwell_sec": 4.0, "episodes": 1, "tracks": 1},
        },
        zone_points={"shelf_a": 50, "shelf_b": 20},
    )

    diag = generate_candidate_diagnostics(run_dict, primary_objective="observed_dwell")

    assert diag.status == "candidate_selected"
    assert diag.selected_zone_id == "shelf_a"
    assert len(diag.evaluated_candidate_zones) == 2
    assert len(diag.unobserved_candidate_zones) == 0

    eval_dwell = diag.objective_comparisons["observed_dwell"]
    assert eval_dwell.selected_zone_id == "shelf_a"
    assert eval_dwell.primary_metric_value == 10.0
    assert eval_dwell.runner_up_zone_id == "shelf_b"
    assert eval_dwell.lead_difference == 6.0
    assert eval_dwell.lead_percentage == 60.0
    assert not eval_dwell.is_tie


# ---------------------------------------------------------------------------
# 2. One Observed Candidate Zone (Single-Zone Evidence Context)
# ---------------------------------------------------------------------------

def test_one_observed_candidate_zone_transparent_neutral_message():
    """Verify that when only 1 candidate is observed, runner-up is None and unobserved zone is NOT called underperforming."""
    run_dict = _make_base_run_dict(
        candidate_zone_ids=["shelf_a", "shelf_b"],
        zone_dwell={
            "shelf_a": {"dwell_sec": 8.5, "episodes": 2, "tracks": 1},
        },
        zone_points={"shelf_a": 42},
    )

    diag = generate_candidate_diagnostics(run_dict, primary_objective="observed_dwell")

    assert diag.status == "candidate_selected"
    assert diag.selected_zone_id == "shelf_a"
    assert diag.evaluated_candidate_zones == ["shelf_a"]
    assert diag.unobserved_candidate_zones == ["shelf_b"]

    # Verify neutral labelling
    m_b = diag.zone_metrics["shelf_b"]
    assert not m_b.is_evaluated
    assert m_b.observed_dwell_span_sec is None
    assert m_b.sampled_in_zone_points is None

    # Check single-zone selection summary message
    eval_dwell = diag.objective_comparisons["observed_dwell"]
    assert eval_dwell.runner_up_zone_id is None
    assert eval_dwell.lead_difference is None
    assert "one observed candidate zone" in eval_dwell.selection_summary.lower()


# ---------------------------------------------------------------------------
# 3. No Observations Across Candidate Zones
# ---------------------------------------------------------------------------

def test_no_observations_across_any_candidate_zone():
    """Verify that when no candidate zone has observations, an explicit no-supported-candidate state is returned."""
    run_dict = _make_base_run_dict(
        candidate_zone_ids=["shelf_a", "shelf_b"],
        zone_dwell={},
        zone_points={},
    )

    diag = generate_candidate_diagnostics(run_dict)

    assert diag.status in ("no_supported_candidate", "no_observed_shelf_activity")
    assert diag.selected_zone_id is None
    assert diag.selected_cell is None
    assert len(diag.evaluated_candidate_zones) == 0
    assert len(diag.unobserved_candidate_zones) == 2
    assert "supporting observations" in diag.selection_summary.lower() or "absent" in diag.selection_summary.lower()


# ---------------------------------------------------------------------------
# 4. Exact Metric Ties
# ---------------------------------------------------------------------------

def test_exact_metric_tie_handling():
    """Verify that when candidates tie on primary metric, is_tie is True and all tied zones are recorded."""
    run_dict = _make_base_run_dict(
        candidate_zone_ids=["shelf_a", "shelf_b"],
        zone_dwell={
            "shelf_a": {"dwell_sec": 6.0, "episodes": 1, "tracks": 1},
            "shelf_b": {"dwell_sec": 6.0, "episodes": 1, "tracks": 1},
        },
        zone_points={"shelf_a": 30, "shelf_b": 30},
    )

    diag = generate_candidate_diagnostics(run_dict, primary_objective="observed_dwell")

    assert diag.status == "candidate_selected"
    assert diag.selected_zone_id in ["shelf_a", "shelf_b"]
    
    eval_dwell = diag.objective_comparisons["observed_dwell"]
    assert eval_dwell.is_tie is True
    assert set(eval_dwell.tied_zone_ids) == {"shelf_a", "shelf_b"}
    assert eval_dwell.lead_difference == 0.0
    assert eval_dwell.lead_percentage == 0.0


# ---------------------------------------------------------------------------
# 5. Objective Divergence
# ---------------------------------------------------------------------------

def test_objective_divergence_explanation():
    """Verify that when objectives select different zones, objectives_agree is False and divergence is explained."""
    run_dict = _make_base_run_dict(
        candidate_zone_ids=["shelf_a", "shelf_b"],
        zone_dwell={
            # shelf_a has higher dwell
            "shelf_a": {"dwell_sec": 12.0, "episodes": 1, "tracks": 1},
            "shelf_b": {"dwell_sec": 3.0, "episodes": 3, "tracks": 2},
        },
        zone_points={
            # shelf_b has higher points
            "shelf_a": 15,
            "shelf_b": 60,
        },
    )

    diag = generate_candidate_diagnostics(run_dict, primary_objective="observed_dwell")

    assert diag.objectives_agree is False
    assert "dwell" in diag.objectives_divergence_explanation.lower()

    # Dwell selects shelf_a
    assert diag.objective_comparisons["observed_dwell"].selected_zone_id == "shelf_a"
    # Sampled points selects shelf_b
    assert diag.objective_comparisons["sampled_points"].selected_zone_id == "shelf_b"
    # Visit episodes selects shelf_b
    assert diag.objective_comparisons["visit_episodes"].selected_zone_id == "shelf_b"


# ---------------------------------------------------------------------------
# 6. Multi-Track vs Single Fragmented Track Lineage
# ---------------------------------------------------------------------------

def test_multi_track_vs_single_track_lineage():
    """Verify that timeline lineage correctly records contributing track IDs."""
    timeline_dict = {
        "playback_mapping": [
            {
                "sample_index": 0,
                "observations": [
                    {"track_id": 1, "floor_point": [1.2, 1.2], "assigned_zone": "shelf_a", "is_out_of_bounds": False},
                    {"track_id": 2, "floor_point": [1.3, 1.3], "assigned_zone": "shelf_a", "is_out_of_bounds": False},
                ],
            },
            {
                "sample_index": 1,
                "observations": [
                    {"track_id": 3, "floor_point": [1.4, 1.4], "assigned_zone": "shelf_a", "is_out_of_bounds": False},
                ],
            },
        ]
    }

    run_dict = _make_base_run_dict(
        candidate_zone_ids=["shelf_a", "shelf_b"],
        zone_dwell={"shelf_a": {"dwell_sec": 5.0, "episodes": 1, "tracks": 3}},
        zone_points={"shelf_a": 3},
    )

    diag = generate_candidate_diagnostics(run_dict, timeline_dict=timeline_dict)

    assert diag.zone_metrics["shelf_a"].contributing_track_ids == [1, 2, 3]
    assert diag.primary_evaluation.contributing_track_ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# 7. Custom Non-MERL Bounds and Zone Identifiers
# ---------------------------------------------------------------------------

def test_custom_non_merl_bounds_and_zones():
    """Verify candidate evaluation with arbitrary custom bounds and non-MERL zone IDs."""
    run_dict = {
        "run_id": "custom_store_run",
        "dataset_name": "custom_store_camera",
        "scene_profile": {
            "bounds": [-10.0, 20.0, 0.0, 15.0],
            "cell_size": 1.0,
            "zones": [
                {"zone_id": "zone_promo_1", "polygon": [[0.0, 2.0], [5.0, 2.0], [5.0, 6.0], [0.0, 6.0]]},
                {"zone_id": "zone_beverage", "polygon": [[10.0, 2.0], [15.0, 2.0], [15.0, 6.0], [10.0, 6.0]]},
            ],
        },
        "dwell_analytics": {
            "status": "available",
            "per_zone": {
                "zone_promo_1": {"observed_dwell_span_sec": 14.2, "total_episodes": 4, "distinct_tracks_count": 2},
                "zone_beverage": {"observed_dwell_span_sec": 8.0, "total_episodes": 2, "distinct_tracks_count": 1},
            },
        },
        "zone_assignment_counts": {
            "per_zone": {
                "zone_promo_1": 71,
                "zone_beverage": 40,
            },
        },
    }

    diag = generate_candidate_diagnostics(run_dict)

    assert diag.candidate_zone_ids == ["zone_promo_1", "zone_beverage"]
    assert diag.selected_zone_id == "zone_promo_1"
    assert diag.grid_bounds == (-10.0, 20.0, 0.0, 15.0)
    assert diag.cell_size == 1.0


# ---------------------------------------------------------------------------
# 8. Missing Hotspot Data Handling
# ---------------------------------------------------------------------------

def test_missing_hotspot_data_safe_handling():
    """Verify that when timeline sidecar is absent, selected_cell is None and does not crash."""
    run_dict = _make_base_run_dict(
        candidate_zone_ids=["shelf_a", "shelf_b"],
        zone_dwell={"shelf_a": {"dwell_sec": 5.0, "episodes": 1, "tracks": 1}},
        zone_points={"shelf_a": 25},
    )

    diag = generate_candidate_diagnostics(run_dict, timeline_dict=None)

    assert diag.status == "candidate_selected"
    assert diag.selected_zone_id == "shelf_a"
    assert diag.selected_cell is None
    assert any("timeline sidecar is absent" in m.lower() for m in diag.missing_inputs)


# ---------------------------------------------------------------------------
# 9. Tied Candidate Cells on Peak Count
# ---------------------------------------------------------------------------

def test_tied_candidate_cells_within_zone():
    """Verify that when 2 cells in the same zone share the peak count, is_tie is True."""
    # Place 2 points in cell (1, 1) and 2 points in cell (1, 2)
    timeline_dict = {
        "playback_mapping": [
            {
                "sample_index": 0,
                "observations": [
                    {"track_id": 1, "floor_point": [0.25, 0.25], "assigned_zone": "shelf_a", "is_out_of_bounds": False},
                    {"track_id": 1, "floor_point": [0.25, 0.25], "assigned_zone": "shelf_a", "is_out_of_bounds": False},
                    {"track_id": 2, "floor_point": [0.75, 0.25], "assigned_zone": "shelf_a", "is_out_of_bounds": False},
                    {"track_id": 2, "floor_point": [0.75, 0.25], "assigned_zone": "shelf_a", "is_out_of_bounds": False},
                ],
            }
        ]
    }

    hotspot = compute_hotspot_with_track_lineage(
        zone_id="shelf_a",
        timeline_dict=timeline_dict,
        bounds=(0.0, 10.0, 0.0, 6.0),
        cell_size=0.5,
    )

    assert hotspot is not None
    assert hotspot["sampled_points"] == 2
    assert hotspot["is_tie"] is True
    assert hotspot["tied_cell_count"] >= 2


# ---------------------------------------------------------------------------
# 10. Out-of-Bounds Heavy Recordings Point Accounting
# ---------------------------------------------------------------------------

def test_out_of_bounds_heavy_run_accounting():
    """Verify that runs with high out-of-bounds counts preserve exact point conservation."""
    run_dict = _make_base_run_dict(
        candidate_zone_ids=["shelf_a", "shelf_b"],
        total_eval=34,
        binned=2,
        oob=32,
        untransformed=0,
    )

    diag = generate_candidate_diagnostics(run_dict)

    accounting = diag.whole_run_point_accounting
    assert accounting["total_points"] == 34
    assert accounting["binned_points"] == 2
    assert accounting["out_of_bounds_points"] == 32
    assert accounting["untransformed_points"] == 0
    assert accounting["conservation_verified"] is True
