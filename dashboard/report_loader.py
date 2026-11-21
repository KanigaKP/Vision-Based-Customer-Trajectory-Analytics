"""
Phase 14.1 — Report Loader & Viewer Backend.
=============================================
Provides pure Python, dependency-free utilities to discover, validate,
and parse Phase 13 local JSON analysis reports for the Streamlit viewer.

ARCHITECTURAL & METHODOLOGICAL INVARIANTS:
1. Local File Discovery:
   - Discovers reports from the established ignored directory (outputs/reports/).
   - Validates parsed JSON structure and handles malformed or non-Phase-13
     files gracefully without raising unhandled exceptions or crashing.
2. Provenance from Content, Not Filename:
   - Reads run_id, dataset_name, source_type, and evidence eligibility directly
     from parsed report dictionary fields, NEVER inferring from filename.
3. Explicit Missing vs. Evaluated Zero Distinction:
   - Accurately distinguishes "not_available" sections (with unsupplied reasons)
     from evaluated sections containing numeric zero metrics.
4. Prominent Evidence Withholding & Contextual Limitations:
   - Extracts and formats Phase 12 insufficient-evidence reasons and caveats.
   - Enforces explicit contextual statements for MERL (sparse, boundary-truncated)
     and MOT17 (non-retail outdoor pedestrian sequence).
5. Explicit Heatmap Artifact Resolution (No Filename Guessing):
   - Only resolves a heatmap image if the report dictionary explicitly contains
     a valid, existing file reference (e.g. `heatmap_image_path`).
   - NEVER guesses or pairs images based on report filenames.
6. Privacy & Non-Prescriptive Boundaries:
   - Strictly read-only; no data writing, no live streaming, no external network calls.
   - Labels track counts as clip-local tracker IDs; no customer identities,
     raw coordinate histories, sales, rankings, or layout recommendations.
"""

import os
import glob
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple


SCHEMA_VERSION_SUPPORTED: str = "1.0.0"

MERL_CONTEXT_STATEMENT: str = (
    "MERL Dataset Context: The local MERL run is an operational engineering "
    "check on recorded video that produced a sparse, boundary-truncated trajectory "
    "density result (100 sampled frames, severe camera field-of-view cutoff). "
    "It does NOT support retail layout recommendations, zone rankings, "
    "or quantitative validation."
)

MOT17_CONTEXT_STATEMENT: str = (
    "MOT17 Dataset Context: The local MOT17-04 sequence is a non-retail outdoor "
    "pedestrian image-sequence engineering check. It does NOT represent a retail "
    "store environment and does NOT support retail layout recommendations or "
    "quantitative validation."
)

GENERAL_EVIDENCE_BOUNDARY: str = (
    "Methodological Boundary: A generated analysis report is a software artifact "
    "recording descriptive metrics and evidence gating status. Neither synthetic "
    "fixtures, sample image sequences, nor sparse operational runs support retail "
    "layout recommendations, zone rankings, customer intent claims, or quantitative "
    "accuracy assertions."
)


@dataclass(frozen=True)
class ParsedReportResult:
    """
    Result of discovering and validating a local analysis report file.

    Attributes:
        file_path: Absolute or relative path to the report file.
        file_name: Basename of the report file.
        is_valid: True if file is valid JSON conforming to supported Phase 13 schema.
        error_message: Descriptive message if is_valid is False.
        schema_version: Schema version string from the report, if present.
        report_id: Unique report identifier from report, if present.
        generated_at_utc: Timestamp when report was generated, if present.
        metadata: Caller-supplied AnalysisRunMetadata dict, if present.
        source_type: Explicit source_type from metadata (not filename).
        dataset_name: Explicit dataset_name from metadata (not filename).
        evidence_context: Parsed evidence_context section dict.
        dwell_analytics: Parsed dwell_analytics section dict.
        density_analytics: Parsed density_analytics section dict.
        transition_analytics: Parsed transition_analytics section dict.
        spatial_observations: Parsed spatial_observations section dict.
        evidence_summary: Parsed evidence_summary section dict.
        is_eligible: Boolean indicating if spatial observations were eligible.
        reasons_withheld: List of causal strings why observations were withheld.
        mandatory_statements: Contextual statements for MERL / MOT17 / evidence boundaries.
        explicit_heatmap_path: Path to heatmap image IF AND ONLY IF explicitly referenced and existing.
        raw_data: Complete parsed dictionary if valid JSON, else empty dict.
    """
    file_path: str
    file_name: str
    is_valid: bool
    error_message: Optional[str] = None
    schema_version: Optional[str] = None
    report_id: Optional[str] = None
    generated_at_utc: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_type: Optional[str] = None
    dataset_name: Optional[str] = None
    evidence_context: Dict[str, Any] = field(default_factory=dict)
    dwell_analytics: Dict[str, Any] = field(default_factory=dict)
    density_analytics: Dict[str, Any] = field(default_factory=dict)
    transition_analytics: Dict[str, Any] = field(default_factory=dict)
    spatial_observations: Dict[str, Any] = field(default_factory=dict)
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    is_eligible: bool = False
    reasons_withheld: List[str] = field(default_factory=list)
    mandatory_statements: List[str] = field(default_factory=list)
    explicit_heatmap_path: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)


def is_valid_phase_13_report(file_path: str) -> bool:
    """
    Check if a file exists, is valid JSON, and adheres to supported Phase 13 schema.
    """
    if not os.path.isfile(file_path):
        return False
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return isinstance(data, dict) and data.get("schema_version") == SCHEMA_VERSION_SUPPORTED
    except Exception:
        return False


def discover_report_files(reports_dir: str = "outputs/reports") -> List[str]:
    """
    Discover all JSON report files in the specified reports directory.

    Orders valid Phase 13 schema 1.0.0 reports first (prioritizing MERL sample
    as the primary default, followed by MOT17), followed by legacy/unsupported JSON files.

    Args:
        reports_dir: Path to directory containing report files.

    Returns:
        List of matching JSON file paths, with valid Phase 13 reports ordered first.
    """
    if not os.path.exists(reports_dir) or not os.path.isdir(reports_dir):
        return []
    pattern = os.path.join(reports_dir, "*.json")
    all_files = sorted(glob.glob(pattern))

    valid_phase13: List[str] = []
    legacy_files: List[str] = []

    for fpath in all_files:
        if is_valid_phase_13_report(fpath):
            valid_phase13.append(fpath)
        else:
            legacy_files.append(fpath)

    # Sort valid reports: MERL first (0), MOT17 second (1), then alphabetical (2)
    def _report_sort_priority(path: str) -> Tuple[int, str]:
        bname = os.path.basename(path).lower()
        if "merl" in bname:
            return (0, bname)
        if "mot17" in bname:
            return (1, bname)
        return (2, bname)

    valid_phase13.sort(key=_report_sort_priority)
    return valid_phase13 + legacy_files


def get_dataset_context_statements(
    dataset_name: Optional[str],
    environment_type: Optional[str] = None,
    coverage_status: Optional[str] = None,
) -> List[str]:
    """
    Determine applicable mandatory contextual limitations and evidence boundaries
    based on explicit metadata content (not filenames).

    Args:
        dataset_name: Dataset identifier string from metadata.
        environment_type: Environment type string from evidence context.
        coverage_status: Coverage status string from evidence context.

    Returns:
        List of contextual limitation statements.
    """
    statements: List[str] = [GENERAL_EVIDENCE_BOUNDARY]
    ds_lower = (dataset_name or "").lower()
    env_lower = (environment_type or "").lower()
    cov_lower = (coverage_status or "").lower()

    # MERL context detection
    if "merl" in ds_lower or cov_lower == "boundary_truncated":
        statements.append(MERL_CONTEXT_STATEMENT)

    # MOT17 / non-retail context detection
    if "mot17" in ds_lower or env_lower == "pedestrian_outdoor" or "outdoor" in env_lower:
        statements.append(MOT17_CONTEXT_STATEMENT)

    return statements


def resolve_explicit_heatmap_artifact(
    report_data: Dict[str, Any],
    base_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve a heatmap artifact image path IF AND ONLY IF explicitly referenced
    in the report data and present on disk.

    INVARIANT:
    NEVER infers or guesses image filenames from the report filename or dataset name.
    If no explicit key exists or the referenced file does not exist, returns None.

    Args:
        report_data: Parsed report dictionary.
        base_dir: Optional base directory to resolve relative paths against.

    Returns:
        Existing file path string, or None.
    """
    density_sec = report_data.get("density_analytics", {})
    metadata_sec = report_data.get("metadata", {})

    candidates: List[Any] = [
        density_sec.get("heatmap_image_path"),
        density_sec.get("heatmap_artifact"),
        density_sec.get("artifact_path"),
        metadata_sec.get("artifacts", {}).get("heatmap_path") if isinstance(metadata_sec.get("artifacts"), dict) else None,
    ]

    for cand in candidates:
        if isinstance(cand, str) and cand.strip():
            target = cand.strip()
            if os.path.isfile(target):
                return target
            if base_dir and not os.path.isabs(target):
                joined = os.path.join(base_dir, target)
                if os.path.isfile(joined):
                    return joined

    return None


def load_and_validate_report(
    file_path: str,
    base_dir: Optional[str] = None,
) -> ParsedReportResult:
    """
    Load a local JSON report file, validate its structure, and extract its
    sections and metadata gracefully without raising uncaught exceptions.

    Args:
        file_path: Path to the JSON file to parse.
        base_dir: Optional base directory for resolving relative file references.

    Returns:
        ParsedReportResult object with validation status and structured sections.
    """
    file_name = os.path.basename(file_path)

    # 1. Check file existence
    if not os.path.exists(file_path):
        return ParsedReportResult(
            file_path=file_path,
            file_name=file_name,
            is_valid=False,
            error_message=f"File not found: '{file_path}'",
        )

    # 2. Read and parse JSON
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return ParsedReportResult(
            file_path=file_path,
            file_name=file_name,
            is_valid=False,
            error_message=f"Malformed JSON syntax: {exc}",
        )
    except Exception as exc:
        return ParsedReportResult(
            file_path=file_name,
            file_name=file_name,
            is_valid=False,
            error_message=f"Failed to read file: {exc}",
        )

    # 3. Validate root object type
    if not isinstance(data, dict):
        return ParsedReportResult(
            file_path=file_path,
            file_name=file_name,
            is_valid=False,
            error_message="Root JSON value is not an object/dictionary.",
            raw_data={},
        )

    # 4. Check schema version
    schema_version = data.get("schema_version")
    if schema_version != SCHEMA_VERSION_SUPPORTED:
        return ParsedReportResult(
            file_path=file_path,
            file_name=file_name,
            is_valid=False,
            schema_version=str(schema_version) if schema_version is not None else None,
            error_message=(
                f"Unsupported or missing schema_version: {schema_version!r}. "
                f"Expected '{SCHEMA_VERSION_SUPPORTED}'."
            ),
            raw_data=data,
        )

    # 5. Extract core metadata
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    report_id = data.get("report_id")
    generated_at_utc = data.get("generated_at_utc")
    source_type = metadata.get("source_type")
    dataset_name = metadata.get("dataset_name")

    # 6. Extract sections
    evidence_context = data.get("evidence_context", {})
    if not isinstance(evidence_context, dict):
        evidence_context = {}

    dwell_analytics = data.get("dwell_analytics", {})
    if not isinstance(dwell_analytics, dict):
        dwell_analytics = {}

    density_analytics = data.get("density_analytics", {})
    if not isinstance(density_analytics, dict):
        density_analytics = {}

    transition_analytics = data.get("transition_analytics", {})
    if not isinstance(transition_analytics, dict):
        transition_analytics = {}

    spatial_observations = data.get("spatial_observations", {})
    if not isinstance(spatial_observations, dict):
        spatial_observations = {}

    evidence_summary = data.get("evidence_summary", {})
    if not isinstance(evidence_summary, dict):
        evidence_summary = {}

    # 7. Extract evidence eligibility and withheld reasons
    is_eligible = bool(spatial_observations.get("is_eligible", False))
    raw_reasons = spatial_observations.get("reasons_withheld", [])
    reasons_withheld = [str(r) for r in raw_reasons] if isinstance(raw_reasons, list) else []

    # 8. Extract mandatory context statements
    env_type = evidence_context.get("environment_type")
    cov_status = evidence_context.get("coverage_status")
    statements = get_dataset_context_statements(
        dataset_name=dataset_name,
        environment_type=env_type,
        coverage_status=cov_status,
    )

    # 9. Explicit heatmap resolution
    heatmap_path = resolve_explicit_heatmap_artifact(data, base_dir=base_dir)

    return ParsedReportResult(
        file_path=file_path,
        file_name=file_name,
        is_valid=True,
        schema_version=schema_version,
        report_id=report_id,
        generated_at_utc=generated_at_utc,
        metadata=metadata,
        source_type=source_type,
        dataset_name=dataset_name,
        evidence_context=evidence_context,
        dwell_analytics=dwell_analytics,
        density_analytics=density_analytics,
        transition_analytics=transition_analytics,
        spatial_observations=spatial_observations,
        evidence_summary=evidence_summary,
        is_eligible=is_eligible,
        reasons_withheld=reasons_withheld,
        mandatory_statements=statements,
        explicit_heatmap_path=heatmap_path,
        raw_data=data,
    )


def resolve_demo_artifact_path(
    artifact_path: Optional[str],
    allowed_dir: str = "outputs/reports",
) -> Optional[str]:
    """
    Validate and resolve an artifact image path against an allowed base directory.

    INVARIANTS:
    1. Returns None if artifact_path is None, empty, or not a string.
    2. Path Traversal Prevention:
       - Computes the normalized absolute path of the target file and allowed directory.
       - Confirms the target file resides strictly within allowed_dir or its subdirectories.
    3. File Existence:
       - Confirms os.path.isfile(target) is True.
    4. Never guesses or infers paths based on filename patterns.

    Args:
        artifact_path: File path string referenced in demo manifest.
        allowed_dir: Directory within which artifacts must strictly reside.

    Returns:
        Resolved existing file path string, or None if invalid/missing/out-of-bounds.
    """
    if not artifact_path or not isinstance(artifact_path, str):
        return None

    cleaned_path = artifact_path.strip()
    if not cleaned_path:
        return None

    abs_allowed_dir = os.path.abspath(allowed_dir)

    if not os.path.isabs(cleaned_path):
        if not os.path.isfile(os.path.abspath(cleaned_path)):
            cand = os.path.abspath(os.path.join(allowed_dir, cleaned_path))
            if os.path.isfile(cand):
                abs_target = cand
            else:
                abs_target = os.path.abspath(cleaned_path)
        else:
            abs_target = os.path.abspath(cleaned_path)
    else:
        abs_target = os.path.abspath(cleaned_path)

    # Path traversal check: abs_target must reside within abs_allowed_dir
    try:
        common = os.path.commonpath([abs_allowed_dir, abs_target])
        if os.path.normcase(common) != os.path.normcase(abs_allowed_dir):
            return None
    except ValueError:
        return None

    if os.path.isfile(abs_target):
        return abs_target

    return None


def load_demo_summary(
    summary_path: str,
    allowed_dir: str = "outputs/reports",
) -> Optional[Dict[str, Any]]:
    """
    Load and parse a local demo summary JSON file.

    Validates that the file exists, resides strictly within allowed_dir,
    and parses as a valid JSON dictionary.

    Args:
        summary_path: Path to demo summary JSON.
        allowed_dir: Directory within which summary must strictly reside.

    Returns:
        Parsed summary dictionary, or None if invalid or missing.
    """
    resolved_path = resolve_demo_artifact_path(summary_path, allowed_dir=allowed_dir)
    if not resolved_path:
        return None

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return None
    except Exception:
        return None

