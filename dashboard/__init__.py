"""
Dashboard package for Vision-Based Customer Trajectory Analytics.
Phase 14: Local Streamlit viewer for existing analysis reports.
"""

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
    resolve_demo_artifact_path,
    load_demo_summary,
)

__all__ = [
    "SCHEMA_VERSION_SUPPORTED",
    "MERL_CONTEXT_STATEMENT",
    "MOT17_CONTEXT_STATEMENT",
    "GENERAL_EVIDENCE_BOUNDARY",
    "ParsedReportResult",
    "discover_report_files",
    "get_dataset_context_statements",
    "resolve_explicit_heatmap_artifact",
    "load_and_validate_report",
    "resolve_demo_artifact_path",
    "load_demo_summary",
]
