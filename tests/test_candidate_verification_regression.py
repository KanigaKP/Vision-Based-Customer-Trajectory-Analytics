"""
Regression tests for candidate selection and verification logic.
================================================================
Verifies that:
1. A higher-dwell circulation aisle (e.g. browsing_aisle with 10.4s dwell)
   CANNOT replace the selected display candidate (e.g. showcase_display_back with 2.8s dwell).
2. The verification script strictly reads exploratory_layout_candidate.candidate_zone_id
   and candidate_metrics, forbidding any fallback to highest-dwell zone.
3. Missing or corrupted exploratory_layout_candidate blocks fail assertions with "Not verified".
4. Real benchmark run artifacts (upload_165c38c7, upload_7b4f16df, upload_6a838cf4)
   correctly reconcile against their saved exploratory candidate metrics.
5. Actionable commercial recommendations are strictly preserved as WITHHELD.
"""

import os
import json
import pytest
from pathlib import Path

from scratch.execute_final_acceptance import extract_selected_candidate_and_metrics
from dashboard.run_adapter import load_standardized_run


def test_higher_dwell_aisle_cannot_replace_selected_display_candidate():
    """
    Ensure that when an aisle has higher dwell than display fixtures,
    the verification logic reads the actual selected exploratory display candidate
    and does NOT substitute the circulation aisle.
    """
    summary_data = {
        "run_id": "test_jewelry_aisle_vs_display",
        "dwell_analytics": {
            "per_zone": {
                "browsing_aisle": {
                    "observed_dwell_span_sec": 10.4,
                    "episodes": 1,
                    "distinct_clip_local_tracks": 1
                },
                "showcase_display_back": {
                    "observed_dwell_span_sec": 2.8,
                    "episodes": 2,
                    "distinct_clip_local_tracks": 1
                },
                "showcase_display_front": {
                    "observed_dwell_span_sec": 0.6,
                    "episodes": 1,
                    "distinct_clip_local_tracks": 1
                }
            }
        },
        "exploratory_layout_candidate": {
            "status": "exploratory_unvalidated_candidate",
            "candidate_zone_id": "showcase_display_back",
            "candidate_metrics": {
                "zone_id": "showcase_display_back",
                "observed_dwell_span_sec": 2.8,
                "observed_visit_episodes": 2,
                "sampled_in_zone_points": 16
            },
            "comparison_by_shelf_zone": [
                {
                    "zone_id": "showcase_display_back",
                    "observed_dwell_span_sec": 2.8,
                    "observed_visit_episodes": 2
                },
                {
                    "zone_id": "showcase_display_front",
                    "observed_dwell_span_sec": 0.6,
                    "observed_visit_episodes": 1
                }
            ],
            "actionable_recommendation": "WITHHELD"
        }
    }

    zone_id, dwell_sec, episodes, act_rec = extract_selected_candidate_and_metrics(summary_data)

    # Assert that the display candidate was selected
    assert zone_id == "showcase_display_back"
    assert dwell_sec == 2.8
    assert episodes == 2
    assert act_rec == "WITHHELD"

    # Crucial assertion: the higher-dwell circulation aisle (10.4s) MUST NOT replace the display candidate
    assert zone_id != "browsing_aisle"
    assert dwell_sec != 10.4


def test_missing_exploratory_candidate_fails_assertion_no_fallback():
    """
    Ensure that if exploratory_layout_candidate is missing or corrupted,
    the extraction fails assertion rather than falling back to highest-dwell zone.
    """
    summary_without_candidate = {
        "run_id": "test_corrupt_run",
        "dwell_analytics": {
            "per_zone": {
                "browsing_aisle": {
                    "observed_dwell_span_sec": 15.0,
                    "episodes": 3
                }
            }
        }
    }

    with pytest.raises(AssertionError, match="Not verified"):
        extract_selected_candidate_and_metrics(summary_without_candidate)

    # Empty candidate zone
    summary_empty_candidate = {
        "run_id": "test_empty_zone_run",
        "exploratory_layout_candidate": {
            "candidate_zone_id": "",
            "candidate_metrics": {"observed_dwell_span_sec": 2.0, "observed_visit_episodes": 1}
        }
    }
    with pytest.raises(AssertionError, match="Not verified"):
        extract_selected_candidate_and_metrics(summary_empty_candidate)


def test_real_jewelry_run_artifact_reconciliation():
    """
    Reconcile the actual Jewelry Store benchmark run (upload_165c38c7) on disk:
    1. Selected candidate is showcase_display_back (2.8s, 2 episodes).
    2. Browsing aisle (10.4s) was not selected as candidate.
    3. Showcase comparison shows +2.2s difference (2.8s vs 0.6s).
    4. Actionable recommendation is WITHHELD.
    """
    summary_path = Path("outputs/reports/uploaded_runs/upload_165c38c7/run_summary.json")
    assert summary_path.is_file(), f"Artifact missing: {summary_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    zone_id, dwell_sec, episodes, act_rec = extract_selected_candidate_and_metrics(data)

    assert zone_id == "showcase_display_back"
    assert dwell_sec == 2.8
    assert episodes == 2
    assert act_rec == "WITHHELD"

    # Verify that browsing_aisle has 10.4s recorded in dwell_analytics
    zdwell = data.get("dwell_analytics", {}).get("per_zone", {})
    assert "browsing_aisle" in zdwell
    assert zdwell["browsing_aisle"]["observed_dwell_span_sec"] == 10.4
    assert zone_id != "browsing_aisle"

    # Verify showcase comparison
    expl = data.get("exploratory_layout_candidate", {})
    comp = expl.get("comparison_by_shelf_zone", [])
    assert len(comp) == 2
    rear = next(c for c in comp if c["zone_id"] == "showcase_display_back")
    front = next(c for c in comp if c["zone_id"] == "showcase_display_front")
    assert rear["observed_dwell_span_sec"] == 2.8
    assert front["observed_dwell_span_sec"] == 0.6
    diff = round(rear["observed_dwell_span_sec"] - front["observed_dwell_span_sec"], 1)
    assert diff == 2.2


def test_real_supermarket_and_boutique_artifact_reconciliation():
    """
    Reconcile Supermarket (upload_7b4f16df) and Boutique (upload_6a838cf4) runs on disk.
    """
    # 1. Supermarket Aisle
    merl_path = Path("outputs/reports/uploaded_runs/upload_7b4f16df/run_summary.json")
    assert merl_path.is_file(), f"Artifact missing: {merl_path}"
    with open(merl_path, "r", encoding="utf-8") as f:
        merl_data = json.load(f)

    m_zone, m_dwell, m_ep, m_rec = extract_selected_candidate_and_metrics(merl_data)
    assert m_zone == "shelf_a"
    assert m_dwell == 10.2
    assert m_ep == 3
    assert m_rec == "WITHHELD"

    # 2. Home Decor Boutique
    boutique_path = Path("outputs/reports/uploaded_runs/upload_6a838cf4/run_summary.json")
    assert boutique_path.is_file(), f"Artifact missing: {boutique_path}"
    with open(boutique_path, "r", encoding="utf-8") as f:
        boutique_data = json.load(f)

    b_zone, b_dwell, b_ep, b_rec = extract_selected_candidate_and_metrics(boutique_data)
    assert b_zone == "side_display_left"
    assert b_dwell == 1.6
    assert b_ep == 1
    assert b_rec == "WITHHELD"


def test_run_adapter_view_model_consistency():
    """
    Ensure load_standardized_run properly loads candidate_zone, candidate_dwell_sec,
    and evidence_status without substituting higher-dwell circulation zones.
    """
    run_jewelry = load_standardized_run("outputs/reports/uploaded_runs/upload_165c38c7/run_summary.json")
    assert run_jewelry is not None
    assert run_jewelry.candidate_zone == "showcase_display_back"
    assert run_jewelry.candidate_dwell_sec == 2.8
    assert run_jewelry.evidence_status == "WITHHELD"
    assert run_jewelry.candidate_zone != "browsing_aisle"
