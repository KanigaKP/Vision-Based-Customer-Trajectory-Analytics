"""
Phase 13.1 — Local Evidence-Aware Analysis Report Export.
=========================================================
Provides a reproducible way to serialize explicitly supplied analysis-run
metadata and available Phase 9-12 analytics results into a versioned local
JSON report.

METHODOLOGICAL & ARCHITECTURAL INVARIANTS:
1. Versioned & Deterministic Schema:
   - Exposes `SCHEMA_VERSION = "1.0.0"`.
   - Serializes into deterministic, strictly ordered dictionaries.
2. Explicit Caller Run & Provenance Metadata:
   - Requires an `AnalysisRunMetadata` instance with explicit run_id, dataset_name,
     and source_type.
   - Never infers provenance, evidence tier, or dataset origin from filenames,
     directory paths, or metric numbers.
3. Truthful Representation of Missing vs. Zero Data:
   - Includes only summaries actually supplied by the caller.
   - Missing Phase 9, 10, 11, or 12 inputs are marked explicitly as
     `{"status": "not_available", "reason": "..."}`, NEVER silently zero-filled.
   - Zero-activity zones that were evaluated are preserved as numeric 0,
     distinguishable from unsupplied analytics.
4. Privacy & Track Identity Boundaries:
   - Does NOT serialize raw person identities or expose per-track point histories.
   - All track counts are explicitly designated as clip-local tracker IDs,
     NOT unique physical customers.
5. Physical Dwell & Density Semantics:
   - Dwell metrics explicitly designate observed temporal spans (exit - entry)
     between sampled detections, NOT continuous physical dwell.
   - Density grids explicitly designate discrete sampled observation counts,
     NOT simultaneous crowd occupancy or unique visitors.
6. Local Report Export Only:
   - Writes generated JSON reports strictly to the established ignored output
     location (`outputs/reports/`).
   - No external databases, no cloud uploads, no PDF rendering, no Streamlit UI.
"""

import os
import json
import math
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

from src.analytics.dwell import DwellSummary, ZoneDwellSummary
from src.analytics.density import TrajectoryDensityGrid
from src.analytics.transitions import DirectTransitionSummary, ZoneActivitySummary
from src.analytics.spatial_observations import (
    EvidenceContext,
    SpatialObservationResult,
)


SCHEMA_VERSION: str = "1.0.0"

# Strictly permitted input source types (offline/recorded only; live streams are prohibited)
ALLOWED_SOURCE_TYPES: Tuple[str, ...] = (
    "recorded_video",
    "image_sequence",
    "synthetic_fixture",
    "controlled_trial",
)


def _is_finite_number(val: Any) -> bool:
    """Check if value is a finite int or float (not bool)."""
    if isinstance(val, bool):
        return False
    if isinstance(val, (int, float)):
        return math.isfinite(val)
    return False


@dataclass(frozen=True)
class AnalysisRunMetadata:
    """
    Explicit caller-supplied metadata defining the execution run context.

    Attributes:
        run_id: Unique or deterministic identifier for the analysis run.
        dataset_name: Name/label of the dataset or sequence (e.g. 'merl_video', 'mot17_04').
        source_type: Input source type, strictly restricted to recorded/offline sources:
            'recorded_video' (recorded video file), 'image_sequence' (sequential recorded image frames),
            'synthetic_fixture' (synthetic test fixture), or 'controlled_trial' (recorded project trial footage).
            Live streams (e.g. RTSP, live CCTV, live webcam) are strictly prohibited and rejected by validation.
        target_fps: Optional frame sampling rate (FPS) used for analysis.
        total_frames_evaluated: Optional count of frames evaluated.
        source_path: Optional path to the input video or directory.
        notes: Optional descriptive notes.
    """
    run_id: str
    dataset_name: str
    source_type: str
    target_fps: Optional[float] = None
    total_frames_evaluated: Optional[int] = None
    source_path: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError(f"Invalid run_id: {self.run_id!r}")
        if not isinstance(self.dataset_name, str) or not self.dataset_name.strip():
            raise ValueError(f"Invalid dataset_name: {self.dataset_name!r}")
        if not isinstance(self.source_type, str) or not self.source_type.strip():
            raise ValueError(f"Invalid source_type: {self.source_type!r}")
        if self.source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(
                f"Invalid source_type: {self.source_type!r}. "
                f"Allowed source types are: {', '.join(ALLOWED_SOURCE_TYPES)}. "
                "Live streams (e.g. RTSP, live CCTV) are outside project scope."
            )
        if self.target_fps is not None:
            if not _is_finite_number(self.target_fps) or self.target_fps <= 0.0:
                raise ValueError(f"Invalid target_fps: {self.target_fps!r}")
        if self.total_frames_evaluated is not None:
            if not isinstance(self.total_frames_evaluated, int) or isinstance(self.total_frames_evaluated, bool) or self.total_frames_evaluated < 0:
                raise ValueError(f"Invalid total_frames_evaluated: {self.total_frames_evaluated!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build_evidence_context_section(evidence_context: Optional[EvidenceContext]) -> Dict[str, Any]:
    """Serialize EvidenceContext or explicitly mark as not available."""
    if evidence_context is None:
        return {
            "status": "not_available",
            "reason": "Evidence context was not supplied by the caller.",
        }
    return {
        "status": "available",
        "environment_type": evidence_context.environment_type,
        "calibration_status": evidence_context.calibration_status,
        "coverage_status": evidence_context.coverage_status,
        "notes": evidence_context.notes,
    }


def _build_dwell_section(dwell_summary: Optional[DwellSummary]) -> Dict[str, Any]:
    """Serialize Phase 9 DwellSummary without exposing raw person histories."""
    if dwell_summary is None:
        return {
            "status": "not_available",
            "reason": "Dwell analytics summary was not supplied for this run.",
        }

    zone_dwell_summaries: Dict[str, Any] = {}
    for zone_id in sorted(dwell_summary.by_zone.keys()):
        zd: ZoneDwellSummary = dwell_summary.by_zone[zone_id]
        zone_dwell_summaries[zone_id] = {
            "zone_id": zd.zone_id,
            "total_observed_dwell_span_sec": zd.total_dwell_sec,
            "total_episodes": zd.total_episodes,
            "distinct_clip_local_tracks": zd.unique_tracks,
            "mean_dwell_span_sec": zd.mean_dwell_sec,
            "median_dwell_span_sec": zd.median_dwell_sec,
            "min_dwell_span_sec": zd.min_dwell_sec,
            "max_dwell_span_sec": zd.max_dwell_sec,
            "span_distribution": zd.distribution.to_dict(),
        }

    return {
        "status": "available",
        "total_episodes": dwell_summary.total_episodes,
        "total_observed_dwell_span_sec": dwell_summary.total_dwell_sec,
        "distinct_clip_local_tracks_evaluated": len(dwell_summary.by_track),
        "zone_dwell_summaries": zone_dwell_summaries,
        "caveats": [
            "Durations measure observed in-zone temporal spans (exit_time - entry_time) between discrete sampled detections, NOT continuous physically verified presence.",
            "Single-frame visits evaluate to 0.0s observed span by definition.",
            "distinct_clip_local_tracks measures session-local tracker IDs assigned by ByteTrack and may overstate unique visitors due to tracking fragmentation.",
        ],
    }


def _build_density_section(density_grid: Optional[TrajectoryDensityGrid]) -> Dict[str, Any]:
    """Serialize Phase 10 TrajectoryDensityGrid."""
    if density_grid is None:
        return {
            "status": "not_available",
            "reason": "Trajectory density grid was not supplied for this run.",
        }

    return {
        "status": "available",
        "bounds": {
            "min_x": density_grid.bounds.min_x,
            "max_x": density_grid.bounds.max_x,
            "min_y": density_grid.bounds.min_y,
            "max_y": density_grid.bounds.max_y,
            "unit": "configured_floor_units",
        },
        "grid_shape": [density_grid.num_bins_y, density_grid.num_bins_x],
        "cell_size": [density_grid.cell_size_x, density_grid.cell_size_y],
        "total_evaluated_points": density_grid.total_evaluated_points,
        "total_binned_points": density_grid.total_binned_points,
        "out_of_bounds_count": density_grid.out_of_bounds_count,
        "untransformed_count": density_grid.untransformed_count,
        "max_cell_density": density_grid.max_cell_count,
        "conservation_invariant_verified": density_grid.total_evaluated_points == (
            density_grid.total_binned_points + density_grid.out_of_bounds_count + density_grid.untransformed_count
        ),
        "caveats": [
            "Point density represents sampled trajectory-point counts per grid cell at the frame sampling rate; it does NOT measure unique visitors, physical occupancy, or dwell duration.",
            "Coordinates and bounds are expressed in configured floor units; physical metric scale has not been independently surveyed.",
        ],
    }


def _build_transition_section(transition_summary: Optional[DirectTransitionSummary]) -> Dict[str, Any]:
    """Serialize Phase 11 DirectTransitionSummary."""
    if transition_summary is None:
        return {
            "status": "not_available",
            "reason": "Zone transitions summary was not supplied for this run.",
        }

    zone_activity_summaries: Dict[str, Any] = {}
    for zone_id in sorted(transition_summary.zone_summaries.keys()):
        zs: ZoneActivitySummary = transition_summary.zone_summaries[zone_id]
        zone_activity_summaries[zone_id] = {
            "zone_id": zs.zone_id,
            "observed_run_count": zs.observed_run_count,
            "distinct_clip_local_track_count": zs.distinct_track_count,
            "direct_incoming_count": zs.direct_incoming_count,
            "direct_outgoing_count": zs.direct_outgoing_count,
        }

    formatted_transition_counts = {
        f"{from_z}->{to_z}": count
        for (from_z, to_z), count in sorted(transition_summary.transition_counts.items())
    }

    return {
        "status": "available",
        "total_evaluated_tracks": transition_summary.total_tracks,
        "total_transitions": transition_summary.total_transitions,
        "known_zone_ids": list(transition_summary.known_zone_ids),
        "zone_activity_summaries": zone_activity_summaries,
        "transition_counts": formatted_transition_counts,
        "transition_matrix": {
            z_from: dict(matrix_row)
            for z_from, matrix_row in sorted(transition_summary.transition_matrix.items())
        },
        "caveats": [
            "Tracker IDs represent clip-local tracking histories from ByteTrack, NOT unique human individuals; counts may be elevated due to tracking fragmentation.",
            "Direct transitions record only immediate consecutive movements between distinct defined zones; any intermediate unassigned (None) observation breaks the transition chain.",
            "Phase 11 outputs are direct transition counts; they do NOT model Markovian probabilities or transition matrices.",
        ],
    }


def _build_observation_section(observation_result: Optional[SpatialObservationResult]) -> Dict[str, Any]:
    """Serialize Phase 12 SpatialObservationResult."""
    if observation_result is None:
        return {
            "status": "not_available",
            "reason": "Spatial observation result was not supplied for this run.",
        }

    return {
        "status": observation_result.status,
        "is_eligible": observation_result.is_eligible,
        "observations_count": len(observation_result.observations),
        "observations": [obs.to_dict() for obs in observation_result.observations],
        "reasons_withheld": list(observation_result.reasons_withheld),
        "caveats": [
            "Spatial observations are plain descriptive statements of discrete sample counts where evidence is eligible.",
            "Where evidence is insufficient, observations are withheld with explicit causal reasons.",
            "No retail layout optimization recommendations, zone rankings, or customer intent claims are asserted.",
        ],
    }


def _build_evidence_summary(
    evidence_context: Optional[EvidenceContext],
    observation_result: Optional[SpatialObservationResult],
) -> Dict[str, Any]:
    """Construct top-level evidence summary."""
    return {
        "environment_type": evidence_context.environment_type if evidence_context else None,
        "calibration_status": evidence_context.calibration_status if evidence_context else None,
        "coverage_status": evidence_context.coverage_status if evidence_context else None,
        "spatial_observation_eligibility": observation_result.status if observation_result else "not_evaluated",
        "classification": {
            "implemented": True,
            "operationally_verified": True,
            "quantitatively_validated": False,
            "retail_performance_validated": False,
        },
        "disclaimer": (
            "This report is a software artifact recording descriptive metrics and evidence gating status. "
            "It does NOT establish empirical accuracy, physical ground truth, or retail layout performance."
        ),
    }


@dataclass(frozen=True)
class AnalysisReport:
    """
    Immutable representation of a versioned local analysis report.
    """
    schema_version: str
    report_id: str
    generated_at_utc: str
    metadata: AnalysisRunMetadata
    evidence_context_section: Dict[str, Any]
    dwell_section: Dict[str, Any]
    density_section: Dict[str, Any]
    transition_section: Dict[str, Any]
    observation_section: Dict[str, Any]
    evidence_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to deterministic dictionary representation."""
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "generated_at_utc": self.generated_at_utc,
            "metadata": self.metadata.to_dict(),
            "evidence_context": dict(self.evidence_context_section),
            "dwell_analytics": dict(self.dwell_section),
            "density_analytics": dict(self.density_section),
            "transition_analytics": dict(self.transition_section),
            "spatial_observations": dict(self.observation_section),
            "evidence_summary": dict(self.evidence_summary),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to deterministic JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def build_analysis_report(
    metadata: AnalysisRunMetadata,
    evidence_context: Optional[EvidenceContext] = None,
    dwell_summary: Optional[DwellSummary] = None,
    density_grid: Optional[TrajectoryDensityGrid] = None,
    transition_summary: Optional[DirectTransitionSummary] = None,
    observation_result: Optional[SpatialObservationResult] = None,
    report_id: Optional[str] = None,
    generated_at_utc: Optional[str] = None,
) -> AnalysisReport:
    """
    Construct an immutable versioned AnalysisReport from supplied inputs.

    Args:
        metadata: Required explicit run metadata.
        evidence_context: Optional caller-supplied evidence context.
        dwell_summary: Optional Phase 9 DwellSummary.
        density_grid: Optional Phase 10 TrajectoryDensityGrid.
        transition_summary: Optional Phase 11 DirectTransitionSummary.
        observation_result: Optional Phase 12 SpatialObservationResult.
        report_id: Optional custom report ID; defaults to 'report_{run_id}'.
        generated_at_utc: Optional ISO-8601 UTC timestamp string for deterministic testing.

    Returns:
        AnalysisReport instance.
    """
    if not isinstance(metadata, AnalysisRunMetadata):
        raise TypeError(f"metadata must be an AnalysisRunMetadata instance, got {type(metadata)!r}")

    effective_report_id = report_id or f"report_{metadata.run_id}"
    effective_timestamp = generated_at_utc or datetime.now(timezone.utc).isoformat()

    return AnalysisReport(
        schema_version=SCHEMA_VERSION,
        report_id=effective_report_id,
        generated_at_utc=effective_timestamp,
        metadata=metadata,
        evidence_context_section=_build_evidence_context_section(evidence_context),
        dwell_section=_build_dwell_section(dwell_summary),
        density_section=_build_density_section(density_grid),
        transition_section=_build_transition_section(transition_summary),
        observation_section=_build_observation_section(observation_result),
        evidence_summary=_build_evidence_summary(evidence_context, observation_result),
    )


def export_analysis_report(
    report: AnalysisReport,
    output_path: Optional[str] = None,
    reports_dir: str = "outputs/reports",
) -> str:
    """
    Export an AnalysisReport to a local JSON file in the established reports directory.

    Args:
        report: AnalysisReport instance to export.
        output_path: Optional exact output path. If None, saves to
            '{reports_dir}/{report.report_id}.json'.
        reports_dir: Directory where reports are saved if output_path is not specified.

    Returns:
        String path to the saved JSON file.
    """
    if not isinstance(report, AnalysisReport):
        raise TypeError(f"report must be an AnalysisReport instance, got {type(report)!r}")

    if output_path is None:
        target_path = os.path.join(reports_dir, f"{report.report_id}.json")
    else:
        target_path = output_path

    # Ensure parent directory exists
    parent_dir = os.path.dirname(target_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(report.to_json(indent=2))

    return target_path


def generate_and_export_analysis_report(
    metadata: AnalysisRunMetadata,
    output_path: Optional[str] = None,
    evidence_context: Optional[EvidenceContext] = None,
    dwell_summary: Optional[DwellSummary] = None,
    density_grid: Optional[TrajectoryDensityGrid] = None,
    transition_summary: Optional[DirectTransitionSummary] = None,
    observation_result: Optional[SpatialObservationResult] = None,
    report_id: Optional[str] = None,
    generated_at_utc: Optional[str] = None,
    reports_dir: str = "outputs/reports",
) -> str:
    """
    Convenience function: Build and export an analysis report in one call.
    """
    report = build_analysis_report(
        metadata=metadata,
        evidence_context=evidence_context,
        dwell_summary=dwell_summary,
        density_grid=density_grid,
        transition_summary=transition_summary,
        observation_result=observation_result,
        report_id=report_id,
        generated_at_utc=generated_at_utc,
    )
    return export_analysis_report(
        report=report,
        output_path=output_path,
        reports_dir=reports_dir,
    )
