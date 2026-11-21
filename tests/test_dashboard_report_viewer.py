"""
Focused tests for Phase 14.1 — Local Streamlit Viewer Report Loader & Backend.
=============================================================================
Verifies:
1. Discovery of local report files from outputs/reports/.
2. Graceful handling of non-existent directories and empty directories.
3. Loading and structural validation of valid Phase 13 reports (MERL, MOT17).
4. Reading source_type and dataset_name directly from content, not filenames.
5. Graceful handling of malformed JSON syntax and non-object root JSON.
6. Graceful rejection of unsupported or missing schema versions.
7. Accurate distinction between "not_available" sections and evaluated zero metrics.
8. Prominent extraction of Phase 12 reasons_withheld and caveats.
9. Contextual statement attachment for MERL (sparse/truncated) and MOT17 (non-retail).
10. Explicit heatmap artifact resolution: strictly requires explicit existing path,
    never guesses based on filename or report id.
11. Privacy boundaries: no raw customer identities or per-track coordinate arrays.
"""

import os
import json
import pytest
from typing import Dict, Any

from dashboard.report_loader import (
    SCHEMA_VERSION_SUPPORTED,
    MERL_CONTEXT_STATEMENT,
    MOT17_CONTEXT_STATEMENT,
    GENERAL_EVIDENCE_BOUNDARY,
    ParsedReportResult,
    discover_report_files,
    get_dataset_context_statements,
    resolve_explicit_heatmap_artifact,
    load_and_validate_report,
)


def test_discover_report_files_in_existing_directory():
    """Verify discovery finds existing JSON files in outputs/reports/."""
    files = discover_report_files("outputs/reports")
    assert isinstance(files, list)
    assert len(files) >= 2
    # Check that our sample reports are discovered
    basenames = [os.path.basename(f) for f in files]
    assert "phase_13_analysis_report_merl_sample.json" in basenames
    assert "phase_13_analysis_report_mot17_sample.json" in basenames


def test_default_report_selection_is_valid_phase13():
    """
    Verify that discover_report_files orders valid Phase 13 reports first,
    ensuring that the default selection (index 0) is a valid Phase 13 schema 1.0.0 report.
    """
    files = discover_report_files("outputs/reports")
    assert len(files) >= 1
    default_report = files[0]
    result = load_and_validate_report(default_report)
    assert result.is_valid is True
    assert result.schema_version == SCHEMA_VERSION_SUPPORTED
    assert "phase_13_analysis_report_merl_sample.json" in os.path.basename(default_report)


def test_discover_report_files_nonexistent_or_empty_directory(tmp_path):
    """Verify discover_report_files returns empty list for missing or empty directory."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    assert discover_report_files(str(empty_dir)) == []
    assert discover_report_files(str(tmp_path / "non_existent_dir")) == []


def test_load_and_validate_merl_sample_report():
    """
    Verify loading and parsing of the actual MERL Phase 13 sample report.
    Confirms reading source_type and evidence status from content, not filename.
    """
    merl_path = "outputs/reports/phase_13_analysis_report_merl_sample.json"
    result = load_and_validate_report(merl_path)

    assert result.is_valid is True
    assert result.error_message is None
    assert result.schema_version == SCHEMA_VERSION_SUPPORTED
    assert result.report_id == "phase_13_analysis_report_merl_sample"
    assert result.source_type == "recorded_video"
    assert result.dataset_name == "merl_shopping_dataset"

    # Evidence withholding verification
    assert result.is_eligible is False
    assert len(result.reasons_withheld) >= 2
    assert any("boundary_truncated" in r for r in result.reasons_withheld)
    assert any("unverified_configured_units" in r for r in result.reasons_withheld)

    # Mandatory contextual statement verification
    assert MERL_CONTEXT_STATEMENT in result.mandatory_statements
    assert GENERAL_EVIDENCE_BOUNDARY in result.mandatory_statements

    # Density, dwell, and transition sections are all evaluated and available
    assert result.density_analytics.get("status") == "available"
    assert result.dwell_analytics.get("status") == "available"
    assert result.dwell_analytics.get("total_episodes") == 0
    assert result.dwell_analytics.get("total_observed_dwell_span_sec") == 0.0
    assert result.transition_analytics.get("status") == "available"
    assert (
        result.transition_analytics.get("total_transitions") == 0
        or result.transition_analytics.get("total_direct_transitions") == 0
    )

    # Heatmap resolution: no explicit artifact reference inside report, so must be None
    assert result.explicit_heatmap_path is None


def test_load_and_validate_mot17_sample_report():
    """
    Verify loading and parsing of the actual MOT17 Phase 13 sample report.
    """
    mot17_path = "outputs/reports/phase_13_analysis_report_mot17_sample.json"
    result = load_and_validate_report(mot17_path)

    assert result.is_valid is True
    assert result.error_message is None
    assert result.schema_version == SCHEMA_VERSION_SUPPORTED
    assert result.report_id == "phase_13_analysis_report_mot17_sample"
    assert result.source_type == "image_sequence"
    assert result.dataset_name == "mot17_04_frcnn"

    # Evidence withholding verification
    assert result.is_eligible is False
    assert any("not a retail store environment" in r for r in result.reasons_withheld)

    # Contextual statement verification
    assert MOT17_CONTEXT_STATEMENT in result.mandatory_statements
    assert GENERAL_EVIDENCE_BOUNDARY in result.mandatory_statements


def test_load_malformed_json_syntax(tmp_path):
    """Verify malformed JSON syntax is caught gracefully without crashing."""
    bad_json = tmp_path / "bad_syntax.json"
    bad_json.write_text("{ unquoted_key: 123, invalid ", encoding="utf-8")

    result = load_and_validate_report(str(bad_json))
    assert result.is_valid is False
    assert "Malformed JSON syntax" in (result.error_message or "")


def test_load_non_object_root_json(tmp_path):
    """Verify JSON with a list root value is rejected gracefully."""
    array_json = tmp_path / "array.json"
    array_json.write_text("[1, 2, 3]", encoding="utf-8")

    result = load_and_validate_report(str(array_json))
    assert result.is_valid is False
    assert "Root JSON value is not an object/dictionary" in (result.error_message or "")


def test_load_unsupported_or_missing_schema_version(tmp_path):
    """Verify missing or unsupported schema version is rejected gracefully."""
    no_version_json = tmp_path / "no_version.json"
    no_version_json.write_text('{"report_id": "test"}', encoding="utf-8")

    result1 = load_and_validate_report(str(no_version_json))
    assert result1.is_valid is False
    assert "Unsupported or missing schema_version" in (result1.error_message or "")

    wrong_version_json = tmp_path / "wrong_version.json"
    wrong_version_json.write_text('{"schema_version": "9.9.9", "report_id": "test"}', encoding="utf-8")

    result2 = load_and_validate_report(str(wrong_version_json))
    assert result2.is_valid is False
    assert "Unsupported or missing schema_version: '9.9.9'" in (result2.error_message or "")


def test_file_not_found_handled_gracefully():
    """Verify non-existent file path returns is_valid=False without exception."""
    result = load_and_validate_report("outputs/reports/definitely_nonexistent_report_123.json")
    assert result.is_valid is False
    assert "File not found" in (result.error_message or "")


def test_missing_versus_zero_metric_preservation(tmp_path):
    """
    Verify report parsing preserves distinction between unavailable sections
    and evaluated zero metrics in available sections.
    """
    report_dict = {
        "schema_version": "1.0.0",
        "report_id": "zero_vs_missing_test",
        "metadata": {
            "run_id": "run_01",
            "dataset_name": "test_store",
            "source_type": "recorded_video",
        },
        "dwell_analytics": {
            "status": "not_available",
            "reason": "Dwell was not evaluated for this test clip.",
        },
        "transition_analytics": {
            "status": "available",
            "total_direct_transitions": 0,
            "distinct_clip_local_track_count": 0,
            "zone_activity_summaries": {
                "aisle_3": {
                    "observed_run_count": 0,
                    "distinct_clip_local_tracks": 0,
                    "incoming_direct_transition_count": 0,
                    "outgoing_direct_transition_count": 0,
                }
            },
        },
        "spatial_observations": {
            "status": "insufficient_evidence",
            "is_eligible": False,
            "reasons_withheld": ["Unverified calibration"],
        },
    }

    target = tmp_path / "zero_vs_missing.json"
    target.write_text(json.dumps(report_dict), encoding="utf-8")

    result = load_and_validate_report(str(target))
    assert result.is_valid is True

    # Dwell is missing with reason
    assert result.dwell_analytics["status"] == "not_available"
    assert "Dwell was not evaluated" in result.dwell_analytics["reason"]

    # Transition is available with explicit evaluated zero
    assert result.transition_analytics["status"] == "available"
    assert result.transition_analytics["total_direct_transitions"] == 0
    assert result.transition_analytics["zone_activity_summaries"]["aisle_3"]["observed_run_count"] == 0


def test_explicit_heatmap_artifact_resolution_never_guesses_filename(tmp_path):
    """
    Verify heatmap resolution:
    1. Returns None when report contains no explicit heatmap reference, even
       if a similarly named image exists in the same directory.
    2. Returns path only when report explicitly points to an existing file.
    3. Returns None when report explicitly points to a non-existent file.
    """
    # Create a dummy image file that has a name matching the report
    dummy_img = tmp_path / "phase_13_report_sample_heatmap.png"
    dummy_img.write_text("fake png bytes", encoding="utf-8")

    # 1. Report without explicit heatmap reference
    report_no_ref = {
        "schema_version": "1.0.0",
        "report_id": "phase_13_report_sample",
        "metadata": {"run_id": "r1", "dataset_name": "d1", "source_type": "recorded_video"},
        "density_analytics": {"status": "available"},
    }
    resolved = resolve_explicit_heatmap_artifact(report_no_ref, base_dir=str(tmp_path))
    assert resolved is None  # MUST NOT guess dummy_img based on name!

    # 2. Report with explicit heatmap reference pointing to existing image
    report_with_ref = {
        "schema_version": "1.0.0",
        "report_id": "phase_13_report_sample",
        "metadata": {"run_id": "r1", "dataset_name": "d1", "source_type": "recorded_video"},
        "density_analytics": {
            "status": "available",
            "heatmap_image_path": str(dummy_img),
        },
    }
    resolved2 = resolve_explicit_heatmap_artifact(report_with_ref)
    assert resolved2 == str(dummy_img)

    # 3. Report with explicit heatmap reference pointing to non-existent image
    report_bad_ref = {
        "schema_version": "1.0.0",
        "report_id": "phase_13_report_sample",
        "metadata": {"run_id": "r1", "dataset_name": "d1", "source_type": "recorded_video"},
        "density_analytics": {
            "status": "available",
            "heatmap_image_path": str(tmp_path / "does_not_exist.png"),
        },
    }
    resolved3 = resolve_explicit_heatmap_artifact(report_bad_ref)
    assert resolved3 is None


def test_privacy_boundaries_in_parsed_report(tmp_path):
    """
    Verify that parsed reports do not contain personal identities, raw trajectory
    coordinates, or sales metrics.
    """
    report_path = "outputs/reports/phase_13_analysis_report_merl_sample.json"
    result = load_and_validate_report(report_path)
    raw_str = json.dumps(result.raw_data)

    assert "customer_name" not in raw_str
    assert "person_id" not in raw_str
    assert "sales" not in raw_str
    assert "revenue" not in raw_str
    assert "trajectory_points" not in raw_str


def test_streamlit_app_runtime_smoke_test_merl_view():
    """
    Verify actual Streamlit app execution for MERL report:
    Confirms default selection, header rendering, mandatory MERL warning,
    Phase 12 gating banner, density metrics, omitted heatmap notice,
    and unavailable dwell/transitions notices.
    """
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.query_params["view"] = "report"
    at.run(timeout=30)

    # Confirm default report is MERL
    assert "merl" in at.selectbox[0].value.lower()

    # Subheaders rendered
    subheaders = [sh.value for sh in at.subheader]
    assert "1. Run & Provenance Metadata" in subheaders
    assert "2. Evidence Context & Methodological Boundaries" in subheaders
    assert "3. Dwell Analytics" in subheaders
    assert "4. Trajectory Density Analytics" in subheaders
    assert "5. Zone Transitions & Activity" in subheaders
    assert "6. Descriptive Spatial Observations" in subheaders

    # Mandatory MERL warning present
    warnings = " ".join([w.value for w in at.warning])
    assert "sparse, boundary-truncated" in warnings
    assert "retail layout recommendations" in warnings

    # Prominent Phase 12 gating banner
    errors = " ".join([e.value for e in at.error])
    assert "INSUFFICIENT EVIDENCE" in errors

    # Omitted heatmap notice present
    infos = " ".join([i.value for i in at.info])
    assert "Heatmap Image: **Omitted**" in infos

    # Evaluated zero metrics and notices present in markdown and captions
    markdowns = " ".join([m.value for m in at.markdown])
    assert "Total Observed Dwell Span" in markdowns
    assert "Total Direct Transitions" in markdowns

    captions = " ".join([c.value for c in at.caption])
    assert "evaluated zero observation" in captions


def test_streamlit_report_selector_excludes_outdoor_mot17_sample():
    """The staff-facing report selector stays focused on indoor retail reports."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.query_params["view"] = "report"
    at.run(timeout=30)

    options = " ".join(str(option).lower() for option in at.selectbox[0].options)
    assert "phase_13_analysis_report_merl_sample.json" in options
    assert "mot17" not in options


def test_streamlit_report_selector_omits_unclassified_legacy_reports():
    """Only schema-valid reports with explicit indoor retail context are selectable."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path)
    at.session_state["authenticated_user"] = "evaluator"
    at.query_params["view"] = "report"
    at.run(timeout=30)

    options = " ".join(str(option).lower() for option in at.selectbox[0].options)
    assert "phase_02" not in options
    assert "mot17" not in options

