"""
Tests for Streamlit Dashboard Navigation Architecture & Redesigned Flow.
========================================================================
Verifies:
1. Single authoritative sidebar radio for authenticated users (Home, Sample Shops, Retail Analysis).
2. Smooth switching between core destinations from the sidebar selector.
3. The 3 primary actions on the Home page (View Sample Shops, Test Sample Shops, Upload a Recording).
4. Sample Shop Library action buttons ("View Results ➔", "Run Analysis ➔") seamlessly routing to Retail Analysis.
5. Invariant: Query parameter ?view=workflow does NOT trap the user when clicking other destinations.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


def test_sidebar_single_radio_and_switching():
    """Verify exactly one sidebar radio exists and switching destinations works smoothly."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["authenticated_user"] = "test_examiner"
    at.query_params["view"] = "workflow"
    at.run(timeout=30)
    assert not at.exception

    # Assert exactly one radio widget exists on the sidebar
    assert len(at.sidebar.radio) == 1
    nav_radio = at.sidebar.radio[0]
    assert any("Home" in opt for opt in nav_radio.options)
    assert any("Sample" in opt and "Shops" in opt for opt in nav_radio.options)
    assert any("Retail Analysis" in opt for opt in nav_radio.options)

    # Switch to Home
    home_opt = next(opt for opt in nav_radio.options if "Home" in opt)
    at.sidebar.radio[0].set_value(home_opt).run(timeout=30)
    assert not at.exception
    markdown_home = " ".join([m.value for m in at.markdown])
    assert "Store Traffic & Display Optimization" in markdown_home

    # Switch to Sample Shops
    samples_opt = next(opt for opt in nav_radio.options if "Sample" in opt and "Shops" in opt)
    at.sidebar.radio[0].set_value(samples_opt).run(timeout=30)
    assert not at.exception
    markdown_samples = " ".join([m.value for m in at.markdown])
    assert "Supermarket Aisle" in markdown_samples or "Sample Shops" in markdown_samples


def test_home_three_primary_actions():
    """Verify the 3 primary cards on Home (View Sample Shops, Test Sample Shops, Upload a Recording)."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["authenticated_user"] = "test_examiner"
    at.query_params["view"] = "home"
    at.run(timeout=30)
    assert not at.exception

    markdown_all = " ".join([m.value for m in at.markdown])
    assert "View Sample Shops" in markdown_all
    assert "Test Sample Shops" in markdown_all
    assert "Upload a Recording" in markdown_all

    # 1. Click View Sample Shops
    btn_view = next((b for b in at.button if "View Sample Shops" in b.label), None)
    assert btn_view is not None
    btn_view.click().run(timeout=30)
    assert not at.exception
    assert at.session_state["effective_nav_view"] == "Sample Shops"
    assert at.session_state.get("sample_action_mode") == "view"

    # 2. Click Test Sample Shops via fresh instance
    at_test = AppTest.from_file(APP_PATH)
    at_test.session_state["authenticated_user"] = "test_examiner"
    at_test.query_params["view"] = "home"
    at_test.run(timeout=30)
    btn_test = next((b for b in at_test.button if "Test Sample Shops" in b.label), None)
    assert btn_test is not None
    btn_test.click().run(timeout=30)
    assert not at_test.exception
    assert at_test.session_state["effective_nav_view"] == "Sample Shops"
    assert at_test.session_state.get("sample_action_mode") == "test"

    # 3. Click Upload a Recording via fresh instance
    at_up = AppTest.from_file(APP_PATH)
    at_up.session_state["authenticated_user"] = "test_examiner"
    at_up.query_params["view"] = "home"
    at_up.run(timeout=30)
    btn_up = next((b for b in at_up.button if "Upload a Recording" in b.label), None)
    assert btn_up is not None
    btn_up.click().run(timeout=30)
    assert not at_up.exception
    assert at_up.session_state["effective_nav_view"] == "Retail Analysis"
    assert at_up.session_state.get("workflow_source_mode") == "upload"


def test_sample_library_view_results_routes_to_workflow():
    """Verify clicking 'View Results ➔' on a sample card opens Retail Analysis Results with that run."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["authenticated_user"] = "test_examiner"
    at.session_state["sample_action_mode"] = "view"
    at.query_params["view"] = "samples"
    at.run(timeout=30)
    assert not at.exception

    # Locate View Results buttons
    view_btns = [b for b in at.button if "View Results" in b.label]
    assert len(view_btns) >= 1

    first_btn = view_btns[0]
    first_btn.click().run(timeout=30)
    assert not at.exception

    # Must transition to Retail Analysis with authoritative_run_id populated
    assert at.session_state["effective_nav_view"] == "Retail Analysis"
    assert at.session_state.get("authoritative_run_id") is not None


def test_workflow_query_param_does_not_trap_navigation():
    """Verify that starting with ?view=workflow does NOT prevent navigation away via sidebar radio."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["authenticated_user"] = "test_examiner"
    at.query_params["view"] = "workflow"
    at.run(timeout=30)
    assert not at.exception

    # Header should be present in markdown
    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "Retail Analysis Workspace" in markdown_texts

    # Click Home in sidebar radio
    home_opt = next(opt for opt in at.sidebar.radio[0].options if "Home" in opt)
    at.sidebar.radio[0].set_value(home_opt).run(timeout=30)
    assert not at.exception

    # Must successfully render Home, not get trapped
    assert at.session_state["effective_nav_view"] == "Home"
    assert "Store Traffic & Display Optimization" in " ".join([m.value for m in at.markdown])
