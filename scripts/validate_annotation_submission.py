"""
Annotation Submission Validation CLI Tool
=========================================
Reproducible command-line utility to validate completed human annotation workbooks
against trusted SamplingManifest contracts and strict completeness invariants.

Enforces:
- require_trusted_manifest=True (independent sampling contract)
- Immediate rejection of any PENDING_REVIEW or unreviewed frames
- Detection of omitted, off-grid, or non-monotonic frames
- Verification of segment ranges and submitted segment definitions
- Coordinate and uncertainty validation

Usage:
    python scripts/validate_annotation_submission.py --workbook <path_to_workbook.json>
"""

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.validation.annotation_validator import (
    validate_annotation_workbook,
    WorkbookValidationReport,
    REGISTERED_PILOT_MANIFESTS,
    TRUSTED_DATASET_PROFILES,
    get_trusted_dataset_zones,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a human annotation workbook against trusted sampling manifests."
    )
    parser.add_argument(
        "--workbook",
        required=True,
        type=str,
        help="Path to the annotation workbook JSON file.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path to authoritative scene profile JSON (overrides default trusted dataset profile).",
    )
    parser.add_argument(
        "--require-trusted-manifest",
        action="store_true",
        default=True,
        help="Enforce independent trusted manifest contract (default: True).",
    )
    parser.add_argument(
        "--allow-uncertified",
        dest="require_trusted_manifest",
        action="store_false",
        help="Allow schema validation without independent coverage certification.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    wb_path = os.path.abspath(args.workbook)

    if not os.path.exists(wb_path):
        print(f"ERROR: Workbook file not found: {wb_path}", file=sys.stderr)
        sys.exit(2)

    try:
        with open(wb_path, "r", encoding="utf-8") as f:
            workbook = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to parse JSON in workbook {wb_path}: {e}", file=sys.stderr)
        sys.exit(2)

    dataset_name = workbook.get("dataset_name", "UNKNOWN")
    trusted_zones = get_trusted_dataset_zones(dataset_name, profile_path=args.profile)
    profile_source = args.profile or TRUSTED_DATASET_PROFILES.get(dataset_name, "None (unmapped)")

    print("=" * 80)
    print("ANNOTATION SUBMISSION VALIDATION RUNNER")
    print("=" * 80)
    print(f"Target File:               {wb_path}")
    print(f"Dataset Name:              {dataset_name}")
    print(f"Annotator ID:              {workbook.get('annotator_id', 'UNSPECIFIED')}")
    print(f"Review Status:             {workbook.get('overall_review_status', 'UNKNOWN')}")
    print(f"Authoritative Profile:     {profile_source}")
    print(f"Trusted Allowed Zones:     {trusted_zones if trusted_zones else 'NONE (unconstrained)'}")
    print(f"Enforce Trusted Manifest:  {args.require_trusted_manifest}")
    print("-" * 80)

    report: WorkbookValidationReport = validate_annotation_workbook(
        workbook=workbook,
        allowed_zones=trusted_zones,
        require_trusted_manifest=args.require_trusted_manifest,
        profile_path=args.profile,
    )

    print(f"Total Declared Frames:     {report.total_frames}")
    print(f"Reviewed Empty Frames:     {report.reviewed_empty_frames}")
    print(f"Reviewed Person Frames:    {report.reviewed_person_frames}")
    print(f"Pending Frames Count:      {len(report.pending_frames)}")
    print(f"Person Observations:       {report.total_person_observations}")
    print(f"Enforced Allowed Zones:    {report.trusted_zones_enforced or 'NONE'}")
    print("-" * 80)
    print(f"Schema Valid:              {'YES' if report.schema_valid else 'NO'}")
    print(f"Coverage Certified:        {'YES' if report.coverage_certified else 'NO'}")
    print(f"Coverage Status:           {report.coverage_status.upper()}")
    print(f"Trusted Manifest Used:     {report.trusted_manifest_used or 'NONE'}")
    print("-" * 80)

    if report.warnings:
        print(f"WARNINGS ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"  [WARN] {w}")
        print("-" * 80)

    if report.is_valid:
        print("OVERALL VERIFICATION RESULT: PASSED (Submission is complete and certified)")
        print("=" * 80)
        sys.exit(0)
    else:
        print(f"OVERALL VERIFICATION RESULT: REJECTED ({len(report.errors)} fatal error(s))")
        print("-" * 80)
        print("Fatal Validation Errors (sample up to 10):")
        for err in report.errors[:10]:
            print(f"  [FAIL] {err}")
        if len(report.errors) > 10:
            print(f"  ... and {len(report.errors) - 10} more error(s)")
        print("=" * 80)
        print("REJECTION NOTICE: This workbook contains incomplete/pending frames or sampling mismatches.")
        print("Scoring tools cannot ingest incomplete human annotations.")
        sys.exit(1)


if __name__ == "__main__":
    main()
