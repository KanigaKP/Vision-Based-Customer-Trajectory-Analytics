"""
Phase 8 Verification Runner — Validation Infrastructure & Methodology Audit
===========================================================================
Executes the Phase 8 validation test suite as a consolidated verification group:
1. Ground-Truth Schema & Reference Representation (Phase 8.1)
2. Independent Homography Held-Out Validation (Phase 8.2)
3. Zone Assignment Agreement Validation (Phase 8.3)
4. Footfall & Transition Event Validation (Phase 8.4)
5. Track Fragmentation Analysis (Phase 8.5)
6. Temporal Hysteresis Sensitivity Analysis (Phase 8.6)

Reports operational verification status, evidence classifications, and methodological boundaries.
Does NOT claim quantitative real-world validation without empirical annotated datasets.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import List, Dict, Any, Optional, Sequence


# Canonical Phase 8 Test Manifest (strictly ordered from 8.1 through 8.6)
PHASE_08_MANIFEST: List[Dict[str, str]] = [
    {
        "phase": "8.1",
        "component": "Ground-Truth Schema & Reference Representation",
        "module": "src/validation/ground_truth.py",
        "test_file": "tests/test_ground_truth.py",
        "description": "Validates schema structures (points, events, calibration points, sequences) and JSON serialization.",
    },
    {
        "phase": "8.2",
        "component": "Independent Homography Held-Out Validation",
        "module": "src/validation/homography_validation.py",
        "test_file": "tests/test_homography_validation.py",
        "description": "Evaluates homography transformation using held-out reference points separated from fitting points.",
    },
    {
        "phase": "8.3",
        "component": "Zone Assignment Agreement Validation",
        "module": "src/validation/zone_validation.py",
        "test_file": "tests/test_zone_validation.py",
        "description": "Measures point-by-point agreement and confusion matrix between Phase 6 assignments and reference zone labels.",
    },
    {
        "phase": "8.4",
        "component": "Footfall & Transition Event Validation",
        "module": "src/validation/footfall_validation.py",
        "test_file": "tests/test_footfall_validation.py",
        "description": "Compares Phase 7 FootfallSummary discrete entries and transitions against independently annotated events.",
    },
    {
        "phase": "8.5",
        "component": "Track Fragmentation Analysis",
        "module": "src/validation/fragmentation.py",
        "test_file": "tests/test_fragmentation.py",
        "description": "Quantifies observed fragmentation counts and overlap-safe frame coverage under explicit ground-truth correspondences.",
    },
    {
        "phase": "8.6",
        "component": "Temporal Hysteresis Sensitivity Analysis",
        "module": "src/validation/hysteresis_validation.py",
        "test_file": "tests/test_hysteresis_validation.py",
        "description": "Measures variation in entries, transitions, and boundary changes across varying hysteresis windows N in [1, 10].",
    },
]

# Evidence classification matrix strictly reflecting empirical status
EVIDENCE_CLASSIFICATIONS: List[Dict[str, Any]] = [
    {
        "phase": "8.1",
        "component": "Ground-Truth Schema & Reference Fixtures",
        "implemented": True,
        "operationally_verified": True,
        "quantitatively_validated": False,
        "evidence_available": "Synthetic points, events, calibration landmarks, and JSON round-trips.",
        "limitation": "Schema definition only; does not provide physical real-world camera annotations.",
    },
    {
        "phase": "8.2",
        "component": "Independent Homography Held-Out Validation",
        "implemented": True,
        "operationally_verified": True,
        "quantitatively_validated": False,
        "evidence_available": "Synthetic planar correspondences separating 4 fitting points from M >= 1 held-out check points.",
        "limitation": "Tested on synthetic coordinates; optical lens distortion and field-measured markers not yet evaluated.",
    },
    {
        "phase": "8.3",
        "component": "Zone Assignment Agreement Validation",
        "implemented": True,
        "operationally_verified": True,
        "quantitatively_validated": False,
        "evidence_available": "Synthetic aligned trajectories evaluating PIP agreement, confusion matrix, and unassigned/noise handling.",
        "limitation": "Verifies software agreement mechanics; does not evaluate retail customer occlusions or posture shifts.",
    },
    {
        "phase": "8.4",
        "component": "Footfall & Transition Event Validation",
        "implemented": True,
        "operationally_verified": True,
        "quantitatively_validated": False,
        "evidence_available": "Synthetic trajectory sequences comparing Phase 7 FootfallSummary counts against discrete reference events.",
        "limitation": "Validates aggregate event count comparison; does not perform frame-level temporal alignment or turnaround filtering.",
    },
    {
        "phase": "8.5",
        "component": "Track Fragmentation Analysis",
        "implemented": True,
        "operationally_verified": True,
        "quantitatively_validated": False,
        "evidence_available": "Synthetic multi-track fragments evaluating observed fragmentation counts, overlap-safe frame coverage, and coverage ratios under explicit correspondences.",
        "limitation": "Requires external person_id -> track_id mapping; does not compute duration metrics or temporal gaps; does NOT perform visual Re-ID, appearance embedding, or automatic trajectory stitching.",
    },
    {
        "phase": "8.6",
        "component": "Hysteresis Sensitivity Analysis",
        "implemented": True,
        "operationally_verified": True,
        "quantitatively_validated": False,
        "evidence_available": "Synthetic boundary-oscillating trajectories evaluated across hysteresis windows N in [1, 10] with baseline deltas.",
        "limitation": "Algorithmic parameter sweep only; does not declare any window optimal, nor replace empirical dwell ground truth.",
    },
]


def get_phase_08_manifest() -> List[Dict[str, str]]:
    """Return a deterministic list of Phase 8 validation components and their test files."""
    return [dict(item) for item in PHASE_08_MANIFEST]


def get_manifest_paths() -> List[str]:
    """Return the ordered list of test file paths in the Phase 8 manifest."""
    return [item["test_file"] for item in PHASE_08_MANIFEST]


def get_evidence_classifications() -> List[Dict[str, Any]]:
    """Return the consolidated evidence classification records."""
    return [dict(item) for item in EVIDENCE_CLASSIFICATIONS]


def build_pytest_command(
    test_files: Optional[Sequence[str]] = None,
    python_exe: Optional[str] = None,
    no_cache: bool = True,
    verbose: bool = False,
) -> List[str]:
    """
    Construct a valid pytest invocation command using the specified Python executable.
    
    Args:
        test_files: Specific test paths to invoke. Defaults to Phase 8 manifest paths.
        python_exe: Python interpreter binary path. Defaults to sys.executable.
        no_cache: Whether to include '-p no:cacheprovider'. Defaults to True.
        verbose: Whether to pass '-v' instead of '-q'. Defaults to False.

    Returns:
        List of command-line argument strings.
    """
    exe = python_exe or sys.executable
    cmd = [exe, "-m", "pytest"]
    if no_cache:
        cmd.extend(["-p", "no:cacheprovider"])
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    files = list(test_files) if test_files is not None else get_manifest_paths()
    cmd.extend(files)
    return cmd


def classify_result(returncode: int) -> bool:
    """
    Classify subprocess returncode into passing (True) or failing (False).
    Avoids brittle percentage parsing.
    """
    return returncode == 0


def format_text_summary(
    manifest: List[Dict[str, str]],
    classifications: List[Dict[str, Any]],
    pytest_command: str,
    passed: bool,
    returncode: int,
    stdout: str,
) -> str:
    """Format human-readable verification summary."""
    lines = []
    lines.append("=" * 80)
    lines.append("PHASE 8 -- COMPREHENSIVE VALIDATION VERIFICATION RUNNER")
    lines.append("=" * 80)
    lines.append(f"Pytest Command: {pytest_command}")
    lines.append(f"Subprocess Status: {'PASSED' if passed else 'FAILED'} (Exit Code: {returncode})")
    lines.append("")
    lines.append("-" * 80)
    lines.append("1. PHASE 8 COMPONENT MANIFEST & VERIFICATION TEST MAPPING")
    lines.append("-" * 80)
    lines.append(f"{'PHASE':<7} | {'COMPONENT':<45} | {'TEST FILE'}")
    lines.append("-" * 80)
    for item in manifest:
        lines.append(f"{item['phase']:<7} | {item['component']:<45} | {item['test_file']}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("2. EVIDENCE CLASSIFICATION MATRIX")
    lines.append("-" * 80)
    lines.append(f"{'COMPONENT':<38} | {'IMPL':<5} | {'OP VER':<7} | {'QUANT VAL':<9} | {'LIMITATION'}")
    lines.append("-" * 80)
    for c in classifications:
        impl_str = "YES" if c["implemented"] else "NO"
        op_str = "YES" if c["operationally_verified"] else "NO"
        quant_str = "YES" if c["quantitatively_validated"] else "NO"
        lines.append(f"{c['component']:<38} | {impl_str:<5} | {op_str:<7} | {quant_str:<9} | {c['limitation']}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("3. METHODOLOGICAL INVARIANTS & AUDIT CAVEATS")
    lines.append("-" * 80)
    lines.append("  * Synthetic fixtures verify software behavior and edge-case mechanics only.")
    lines.append("  * Homography training reprojection error is NOT independent validation.")
    lines.append("  * Inspecting a few video frames does NOT establish tracking accuracy.")
    lines.append("  * Fragmentation relies on explicit external mappings; does NOT perform Re-ID.")
    lines.append("  * Hysteresis sweeps evaluate sensitivity; they do NOT declare operational optimality.")
    lines.append("  * No real-world accuracy claims, sales uplift, or layout recommendations are made.")
    lines.append("  * Phase 8 does not implement Phase 9 (Refined Dwell / Temporal Analytics), Phase 10 (Trajectory Density / Heatmaps), Streamlit dashboards, or databases.")
    lines.append("")
    lines.append("-" * 80)
    lines.append("4. PYTEST EXECUTION OUTPUT")
    lines.append("-" * 80)
    lines.append(stdout if stdout else "(No stdout output)")
    lines.append("-" * 80)
    if passed:
        lines.append("OVERALL VERIFICATION RESULT: SUCCESS (All Phase 8 test suites passed)")
    else:
        lines.append("OVERALL VERIFICATION RESULT: FAILED (One or more Phase 8 tests failed)")
    lines.append("=" * 80)
    return "\n".join(lines)


def run_phase_08_verification(
    test_files: Optional[Sequence[str]] = None,
    python_exe: Optional[str] = None,
    verbose: bool = False,
    output_path: Optional[str] = None,
    as_json: bool = False,
    dry_run: bool = False,
) -> int:
    """
    Execute Phase 8 verification runner.
    
    Args:
        test_files: Optional custom list of test files.
        python_exe: Optional custom python binary.
        verbose: Verbose pytest execution.
        output_path: Optional path to write JSON summary (never written by default).
        as_json: Whether to print summary to stdout as JSON.
        dry_run: If True, do not run pytest subprocess; print command and return 0.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    manifest = get_phase_08_manifest()
    classifications = get_evidence_classifications()
    cmd = build_pytest_command(test_files=test_files, python_exe=python_exe, verbose=verbose)
    cmd_str = " ".join(cmd)

    if dry_run:
        passed = True
        returncode = 0
        stdout = "Dry run mode: pytest subprocess was not executed."
        stderr = ""
    else:
        # Execute pytest subprocess safely
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.abspath("."),
        )
        passed = classify_result(proc.returncode)
        returncode = proc.returncode
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip() if proc.stderr else ""

    summary = {
        "phase": "8.7",
        "title": "Phase 8 Verification Summary" + (" (Dry Run)" if dry_run else ""),
        "all_passed": passed,
        "returncode": returncode,
        "manifest": manifest,
        "evidence_classifications": classifications,
        "pytest_command": cmd_str,
        "stdout": stdout,
        "stderr": stderr,
    }

    # Write to optional output file only if explicitly requested
    if output_path is not None:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if as_json:
        print(json.dumps(summary, indent=2))
    else:
        text = format_text_summary(manifest, classifications, cmd_str, passed, returncode, stdout)
        print(text)

    return returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8 Verification Runner — Consolidated Verification and Methodology Audit."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print summary in JSON format to stdout.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file path to write structured JSON summary (not created by default).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Run pytest with verbose output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct and display the verification command without executing pytest.",
    )
    args = parser.parse_args()

    code = run_phase_08_verification(
        verbose=args.verbose,
        output_path=args.output,
        as_json=args.json,
        dry_run=args.dry_run,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
