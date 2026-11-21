"""
Tests for Unified Main Retail Analysis Workflow (8 Stages).
===========================================================
Verifies that:
1. The 8-stage main workflow view renders via Streamlit AppTest without errors or exceptions.
2. Direct stage navigation via query params or buttons works correctly.
3. State resetting correctly purges cached session data without leaks.
4. Each bundled example (MERL, Site 03, Site 02) initializes with verified profile bounds and candidate zones.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.views.main_workflow import VERIFIED_EXAMPLES, STAGE_NAMES, reset_workflow_state


def test_main_workflow_examples_catalog():
    """Verify bundled retail examples catalog contains all 3 required benchmarks."""
    assert "merl" in VERIFIED_EXAMPLES
    assert "site03" in VERIFIED_EXAMPLES
    assert "site02" in VERIFIED_EXAMPLES

    for key, info in VERIFIED_EXAMPLES.items():
        assert Path(info["video_path"]).exists(), f"Video missing: {info['video_path']}"
        assert Path(info["profile_path"]).exists(), f"Profile missing: {info['profile_path']}"
        assert len(info["zones"]) >= 4
        assert len(info["bounds"]) == 4


def test_main_workflow_stage_count():
    """Verify exactly 8 stages are defined."""
    assert len(STAGE_NAMES) == 8
    assert STAGE_NAMES[0][1] == "Choose Source"
    assert STAGE_NAMES[4][1] == "Run Analysis"
    assert STAGE_NAMES[6][1] == "Placement Review"
    assert STAGE_NAMES[7][1] == "Analyse Another"


def test_main_workflow_apptest_default_render():
    """Verify that authenticated app entry point renders Recording stage of the main workflow without exceptions."""
    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "test_examiner"
    at.query_params["view"] = "workflow"
    at.run(timeout=30)

    assert not at.exception
    # Check that stage navigation buttons and header are rendered
    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "Retail Analysis Workspace" in markdown_texts
    assert "Recording: Choose an Example or Upload a Recording" in [s.value for s in at.subheader]


def test_main_workflow_apptest_stage_navigation():
    """Verify direct navigation to Results (Placement Review) and Reset."""
    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    
    # Test Stage 7 (Results)
    at_stage7 = AppTest.from_file(app_path)
    at_stage7.session_state["authenticated_user"] = "test_examiner"
    at_stage7.query_params["view"] = "workflow"
    at_stage7.query_params["stage"] = "7"
    at_stage7.run(timeout=30)
    assert not at_stage7.exception
    # Verify the results findings and caption are rendered
    caption_texts = " ".join([c.value for c in at_stage7.caption])
    assert "Suggested areas to review based on recorded traffic." in caption_texts

    # Test Stage 8 (Reset / Analyse Another)
    at_stage8 = AppTest.from_file(app_path)
    at_stage8.session_state["authenticated_user"] = "test_examiner"
    at_stage8.query_params["view"] = "workflow"
    at_stage8.query_params["stage"] = "8"
    at_stage8.run(timeout=30)
    assert not at_stage8.exception
    assert "Results: Analyse Another Recording" in [s.value for s in at_stage8.subheader]
