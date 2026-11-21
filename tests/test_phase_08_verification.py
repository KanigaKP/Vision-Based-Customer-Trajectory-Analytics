"""
Unit & Integration Tests for Phase 8 Verification Runner
========================================================
Validates the non-I/O logic, test manifest integrity, command construction,
result classification, and evidence constraints for scripts/verify_phase_08.py.
"""

import os
import sys
import json
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from scripts.verify_phase_08 import (
    PHASE_08_MANIFEST,
    EVIDENCE_CLASSIFICATIONS,
    get_phase_08_manifest,
    get_manifest_paths,
    get_evidence_classifications,
    build_pytest_command,
    classify_result,
    run_phase_08_verification,
)


def test_manifest_contains_all_six_required_test_files():
    """Test 1: Manifest contains all six required Phase 8 verification test files."""
    expected_files = {
        "tests/test_ground_truth.py",
        "tests/test_homography_validation.py",
        "tests/test_zone_validation.py",
        "tests/test_footfall_validation.py",
        "tests/test_fragmentation.py",
        "tests/test_hysteresis_validation.py",
    }
    actual_files = set(get_manifest_paths())
    assert actual_files == expected_files, f"Manifest test files mismatch. Got: {actual_files}"
    assert len(PHASE_08_MANIFEST) == 6


def test_manifest_ordering_is_deterministic():
    """Test 2: Manifest ordering is deterministic and follows Phase 8.1 to 8.6 sequence."""
    manifest_1 = get_phase_08_manifest()
    manifest_2 = get_phase_08_manifest()
    assert manifest_1 == manifest_2

    expected_phases = ["8.1", "8.2", "8.3", "8.4", "8.5", "8.6"]
    actual_phases = [item["phase"] for item in manifest_1]
    assert actual_phases == expected_phases


def test_build_pytest_command_uses_python_executable():
    """Test 3: Runner builds a valid pytest command using sys.executable or custom python binary."""
    # Default uses sys.executable
    cmd_default = build_pytest_command()
    assert cmd_default[0] == sys.executable
    assert cmd_default[1:3] == ["-m", "pytest"]

    # Explicit custom executable
    cmd_custom = build_pytest_command(python_exe="custom_python_env")
    assert cmd_custom[0] == "custom_python_env"
    assert cmd_custom[1:3] == ["-m", "pytest"]


def test_build_pytest_command_includes_phase_08_test_files():
    """Test 4: Command includes all Phase 8 test manifest paths."""
    manifest_paths = get_manifest_paths()
    cmd = build_pytest_command()

    for path in manifest_paths:
        assert path in cmd, f"Expected manifest path '{path}' missing from pytest command: {cmd}"


def test_build_pytest_command_uses_no_cache_convention():
    """Test 5: Pytest command includes '-p no:cacheprovider' by convention."""
    cmd = build_pytest_command(no_cache=True)
    assert "-p" in cmd
    assert "no:cacheprovider" in cmd
    idx = cmd.index("-p")
    assert cmd[idx + 1] == "no:cacheprovider"


def test_successful_result_classification():
    """Test 6: Subprocess return code 0 is classified as passing."""
    assert classify_result(0) is True


def test_failing_result_classification():
    """Test 7: Non-zero subprocess return codes are classified as failing."""
    assert classify_result(1) is False
    assert classify_result(2) is False
    assert classify_result(130) is False
    assert classify_result(-1) is False


def test_evidence_summary_does_not_mark_synthetic_as_quantitatively_validated():
    """
    Test 8: Evidence summary marks all components as IMPLEMENTED and OPERATIONALLY VERIFIED,
    but strictly NOT QUANTITATIVELY VALIDATED without empirical real-world annotations.
    """
    classifications = get_evidence_classifications()
    assert len(classifications) == 6

    for item in classifications:
        assert item["implemented"] is True, f"{item['component']} should be IMPLEMENTED"
        assert item["operationally_verified"] is True, f"{item['component']} should be OPERATIONALLY VERIFIED"
        assert item["quantitatively_validated"] is False, (
            f"{item['component']} must NOT be claimed as QUANTITATIVELY VALIDATED "
            f"merely because synthetic tests pass."
        )
        assert len(item["limitation"]) > 10, f"{item['component']} missing explicit limitation"


def test_runner_does_not_include_future_phase_tests():
    """Test 9: Phase 8 manifest strictly excludes Phase 9 or later test paths and features."""
    manifest_paths = get_manifest_paths()
    prohibited_keywords = ["phase_09", "phase_10", "heatmap", "dashboard", "database", "recommendation", "reid"]

    for path in manifest_paths:
        for kw in prohibited_keywords:
            assert kw not in path.lower(), f"Prohibited future phase keyword '{kw}' found in manifest path '{path}'"


def test_dry_run_execution():
    """Test 10: Dry-run mode returns 0 without launching a subprocess."""
    code = run_phase_08_verification(dry_run=True)
    assert code == 0


def test_optional_output_file_writing(tmp_path):
    """Test 11: Structured output is written only when output_path is explicitly provided."""
    target_file = tmp_path / "test_summary.json"
    assert not target_file.exists()

    code = run_phase_08_verification(dry_run=True, output_path=str(target_file))
    assert code == 0
    assert target_file.exists()

    with open(target_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["phase"] == "8.7"
    assert data["all_passed"] is True
    assert len(data["manifest"]) == 6
    assert len(data["evidence_classifications"]) == 6


def test_phase_08_manifest_component_names_and_scope():
    """Test 12: Verifies Phase 8.5 naming and description accurately reflect implemented fragmentation scope."""
    manifest = get_phase_08_manifest()
    p85 = manifest[4]

    assert p85["phase"] == "8.5"
    assert p85["component"] == "Track Fragmentation Analysis"
    assert "fragmentation" in p85["description"].lower()
    assert "frame coverage" in p85["description"].lower()
    # Explicit non-scope keywords must not appear in description
    for non_scope_kw in ["duration", "gap", "identity transition", "re-id", "stitching"]:
        assert non_scope_kw not in p85["description"].lower(), f"Unexpected non-scope keyword '{non_scope_kw}' found"

    evidence = get_evidence_classifications()
    p85_ev = evidence[4]
    assert p85_ev["phase"] == "8.5"
    assert p85_ev["component"] == "Track Fragmentation Analysis"
    assert "duration" not in p85_ev["limitation"] or "does not compute duration" in p85_ev["limitation"]

