"""
Phase 12.1 — Evidence-Gated, Explainable Spatial Observation Prototype.
========================================================================
Consumes existing Phase 9 dwell, Phase 10 density, and Phase 11 transition
summaries to emit plain, traceable descriptive observations where directly
supported by caller-supplied evidence context, or an explicit 'insufficient
evidence' result with detailed explanations.

METHODOLOGICAL & ARCHITECTURAL INVARIANTS:
1. Evidence-Gating via Explicit Caller Provenance:
   - Does NOT infer evidence tier or suitability from numeric values, thresholds,
     or filenames.
   - Requires explicit caller-provided EvidenceContext (environment type,
     calibration status, coverage status).
   - If coverage is boundary-truncated (e.g. MERL video run) or environment is
     non-retail (e.g. MOT17 pedestrian street), retail spatial observations are
     explicitly withheld with clear justification.
2. Pure Analytics Layer:
   - Does NOT rerun detection, tracking, homography, zone assignment, footfall,
     dwell, density, or transitions.
   - Reuses existing DwellSummary, TrajectoryDensityGrid, and DirectTransitionSummary.
3. Pure Descriptive & Traceable Observations:
   - Every emitted observation identifies its exact source metrics and values.
   - Non-causal: describes observed samples, not customer intent, preference,
     browsing, engagement, or conversion.
   - Tracker IDs are clip-local tracker IDs, NOT unique people.
   - Durations are observed spans between sampled detections, NOT continuous dwell.
   - Spatial coordinates are in configured floor units; physical scale is unverified
     unless explicitly confirmed by caller provenance.
4. Strictly Non-Prescriptive:
   - NO zone rankings or scores.
   - NO 'popular', 'high-engagement', or 'underperforming' labels.
   - NO store layout redesign recommendations or commercial claims.
   - NO invented numeric thresholds (e.g. no arbitrary dwell duration cutoffs).
5. No Phase 13 Spillover:
   - Pure in-memory deterministic dataclass models.
   - No JSON report writing to disk, no database persistence, no Streamlit UI.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Sequence

from src.analytics.dwell import DwellSummary, ZoneDwellSummary
from src.analytics.density import TrajectoryDensityGrid
from src.analytics.transitions import DirectTransitionSummary, ZoneActivitySummary


# Set of recognized retail environment designations (case-insensitive)
RECOGNIZED_RETAIL_ENVIRONMENTS = frozenset({
    "retail_store",
    "retail_floor",
    "store_environment",
    "supermarket",
    "grocery_store",
})

# Set of acceptable spatial coverage designations
RECOGNIZED_VALID_COVERAGE = frozenset({
    "representative",
    "full_coverage",
    "adequate_in_zone",
})

# Set of recognized independently checked calibration designations (case-insensitive)
RECOGNIZED_CHECKED_CALIBRATIONS = frozenset({
    "surveyed_metric",
    "independently_checked",
    "field_verified",
    "measured_floor_grid",
})


@dataclass(frozen=True)
class EvidenceContext:
    """
    Caller-supplied metadata establishing evidence provenance and suitability.

    Attributes:
        environment_type: Physical setting of the sequence (e.g., 'retail_store',
            'pedestrian_outdoor', 'synthetic_fixture', 'laboratory').
        calibration_status: Survey status of coordinates (e.g., 'surveyed_metric',
            'unverified_configured_units', 'uncalibrated').
        coverage_status: Spatial coverage characteristics (e.g., 'representative',
            'boundary_truncated', 'sparse_unrepresentative', 'unknown').
        notes: Optional descriptive notes regarding dataset or run origin.
    """
    environment_type: str
    calibration_status: str
    coverage_status: str
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.environment_type, str) or not self.environment_type.strip():
            raise ValueError(f"Invalid environment_type: {self.environment_type!r}")
        if not isinstance(self.calibration_status, str) or not self.calibration_status.strip():
            raise ValueError(f"Invalid calibration_status: {self.calibration_status!r}")
        if not isinstance(self.coverage_status, str) or not self.coverage_status.strip():
            raise ValueError(f"Invalid coverage_status: {self.coverage_status!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DescriptiveSpatialObservation:
    """
    A plain, factual, non-causal spatial observation directly traceable to input metrics.

    Attributes:
        observation_id: Deterministic identifier string (e.g. 'OBS_ZONE_ACTIVITY_shelf_a').
        observation_type: Category of observation ('zone_activity', 'zone_dwell_span',
            'direct_transition', 'density_grid').
        zone_id: Specific retail zone identifier, if observation is zone-specific.
        summary_text: Factual descriptive statement describing observed sample counts.
        source_metrics: Exact dictionary of metric names and numeric values from which
            this observation was derived.
        caveats: Explicit methodological caveats (e.g. track fragmentation, observed span,
            configured units).
    """
    observation_id: str
    observation_type: str
    zone_id: Optional[str]
    summary_text: str
    source_metrics: Dict[str, Any]
    caveats: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "observation_type": self.observation_type,
            "zone_id": self.zone_id,
            "summary_text": self.summary_text,
            "source_metrics": dict(self.source_metrics),
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True)
class SpatialObservationResult:
    """
    Deterministic result of evidence evaluation and spatial observation generation.

    Attributes:
        is_eligible: True if evidence context satisfies retail observation criteria.
        status: 'sufficient_evidence' if eligible, 'insufficient_evidence' otherwise.
        observations: Tuple of emitted DescriptiveSpatialObservation instances (empty if ineligible).
        reasons_withheld: Tuple of explicit explanation strings if evidence was insufficient.
        evidence_context: The EvidenceContext provided by caller, if any.
        input_diagnostics: Diagnostic summary of received Phase 9-11 inputs.
    """
    is_eligible: bool
    status: str
    observations: Tuple[DescriptiveSpatialObservation, ...]
    reasons_withheld: Tuple[str, ...]
    evidence_context: Optional[EvidenceContext]
    input_diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_eligible": self.is_eligible,
            "status": self.status,
            "observations": [obs.to_dict() for obs in self.observations],
            "reasons_withheld": list(self.reasons_withheld),
            "evidence_context": self.evidence_context.to_dict() if self.evidence_context else None,
            "input_diagnostics": dict(self.input_diagnostics),
        }


def evaluate_evidence_eligibility(context: Optional[EvidenceContext]) -> Tuple[bool, List[str]]:
    """
    Evaluate evidence context against retail observation prerequisites.

    Returns:
        Tuple of (is_eligible: bool, reasons_withheld: List[str]).
    """
    if context is None:
        return False, [
            "Evidence context is missing; provenance, environment type, and spatial coverage "
            "cannot be verified without explicit caller metadata."
        ]

    reasons: List[str] = []

    # 1. Environment Type Check
    env_norm = context.environment_type.lower().strip()
    if env_norm not in RECOGNIZED_RETAIL_ENVIRONMENTS:
        reasons.append(
            f"Environment type '{context.environment_type}' is not a retail store environment; "
            "retail spatial observations cannot be generated from non-retail settings "
            "(e.g. outdoor pedestrian footage)."
        )

    # 2. Coverage Status Check
    cov_norm = context.coverage_status.lower().strip()
    if cov_norm == "boundary_truncated":
        reasons.append(
            "Coverage status is 'boundary_truncated': spatial trajectories suffer from camera "
            "field-of-view boundary truncation; observations reflect this camera sub-region only and "
            "cannot support storewide layout reconfigurations."
        )
    elif cov_norm not in RECOGNIZED_VALID_COVERAGE:
        reasons.append(
            f"Coverage status '{context.coverage_status}' indicates trajectory observations "
            "are unrepresentative or unverified."
        )

    # 3. Calibration Status Check
    cal_norm = context.calibration_status.lower().strip()
    if cal_norm == "uncalibrated":
        reasons.append(
            "Calibration status is 'uncalibrated': spatial coordinates lack a defined planar "
            "homography or reference floor grid."
        )
    elif cal_norm in {"unverified_configured_units", "configured_only", "unverified"}:
        reasons.append(
            f"Calibration status is '{context.calibration_status}': spatial calibration has not been "
            "independently surveyed or field-verified; unverified or configured-only coordinates "
            "support relative zone comparisons in configured units, but cannot establish metric fixture dimensions, installation clearances, or actionable storewide layout recommendations."
        )
    elif cal_norm not in RECOGNIZED_CHECKED_CALIBRATIONS:
        reasons.append(
            f"Calibration status '{context.calibration_status}' is not recognized as an independently "
            "checked calibration reference."
        )

    is_eligible = len(reasons) == 0
    return is_eligible, reasons


class SpatialObservationEngine:
    """
    Deterministic analytics layer that evaluates evidence eligibility and emits
    traceable descriptive spatial observations from Phase 9-11 summaries.
    """

    def analyze(
        self,
        evidence_context: Optional[EvidenceContext],
        dwell_summary: Optional[DwellSummary] = None,
        transition_summary: Optional[DirectTransitionSummary] = None,
        density_grid: Optional[TrajectoryDensityGrid] = None,
    ) -> SpatialObservationResult:
        """
        Evaluate inputs and produce either descriptive spatial observations or
        an explicit insufficient evidence result.

        Args:
            evidence_context: Explicit caller-provided provenance metadata.
            dwell_summary: Pre-computed Phase 9 DwellSummary (optional).
            transition_summary: Pre-computed Phase 11 DirectTransitionSummary (optional).
            density_grid: Pre-computed Phase 10 TrajectoryDensityGrid (optional).

        Returns:
            SpatialObservationResult containing observations or withheld reasons.
        """
        # Collect diagnostic summary of inputs
        input_diagnostics: Dict[str, Any] = {
            "has_dwell_summary": dwell_summary is not None,
            "has_transition_summary": transition_summary is not None,
            "has_density_grid": density_grid is not None,
        }
        if dwell_summary is not None:
            input_diagnostics["dwell_total_episodes"] = dwell_summary.total_episodes
            input_diagnostics["dwell_total_tracks"] = len(dwell_summary.by_track)
        if transition_summary is not None:
            input_diagnostics["transition_total_tracks"] = transition_summary.total_tracks
            input_diagnostics["transition_total_transitions"] = transition_summary.total_transitions
            input_diagnostics["known_zones_count"] = len(transition_summary.known_zone_ids)
        if density_grid is not None:
            input_diagnostics["density_total_evaluated_points"] = density_grid.total_evaluated_points
            input_diagnostics["density_binned_points"] = density_grid.total_binned_points
            input_diagnostics["density_out_of_bounds_points"] = density_grid.out_of_bounds_count

        # Evaluate evidence eligibility
        is_eligible, reasons_withheld = evaluate_evidence_eligibility(evidence_context)

        # Check for presence of input data
        has_any_data = (
            (dwell_summary is not None and dwell_summary.total_episodes > 0) or
            (transition_summary is not None and (transition_summary.total_tracks > 0 or len(transition_summary.known_zone_ids) > 0)) or
            (density_grid is not None and density_grid.total_evaluated_points > 0)
        )
        if not has_any_data and is_eligible:
            is_eligible = False
            reasons_withheld.append(
                "No populated trajectory observations or analytics summaries were provided to evaluate."
            )

        if not is_eligible:
            return SpatialObservationResult(
                is_eligible=False,
                status="insufficient_evidence",
                observations=(),
                reasons_withheld=tuple(reasons_withheld),
                evidence_context=evidence_context,
                input_diagnostics=input_diagnostics,
            )

        # Generate traceable, plain descriptive observations
        observations: List[DescriptiveSpatialObservation] = []

        # Coordinate unit caveat determination
        unit_caveat = (
            "Coordinates and extents are expressed in configured floor units; "
            "physical metric scale has not been independently surveyed."
            if evidence_context and evidence_context.calibration_status.lower().strip() != "surveyed_metric"
            else "Coordinates are expressed in surveyed metric units."
        )

        track_caveat = (
            "Tracker IDs represent clip-local tracking histories assigned by ByteTrack, "
            "NOT unique individuals; counts may be elevated due to tracking fragmentation."
        )

        dwell_caveat = (
            "Durations measure observed in-zone temporal spans between discrete sampled detections, "
            "NOT verified continuous physical dwell."
        )

        # 1. Zone Activity Observations (Phase 11)
        if transition_summary is not None and transition_summary.zone_summaries:
            for zone_id in sorted(transition_summary.zone_summaries.keys()):
                zs: ZoneActivitySummary = transition_summary.zone_summaries[zone_id]
                source_metrics = {
                    "observed_run_count": zs.observed_run_count,
                    "distinct_track_count": zs.distinct_track_count,
                    "direct_incoming_count": zs.direct_incoming_count,
                    "direct_outgoing_count": zs.direct_outgoing_count,
                }
                summary_text = (
                    f"Zone '{zone_id}': {zs.observed_run_count} observed visit run(s) across "
                    f"{zs.distinct_track_count} distinct clip-local track ID(s); "
                    f"{zs.direct_incoming_count} direct incoming and "
                    f"{zs.direct_outgoing_count} direct outgoing transition(s)."
                )
                caveats = (track_caveat, unit_caveat)
                observations.append(
                    DescriptiveSpatialObservation(
                        observation_id=f"OBS_ZONE_ACTIVITY_{zone_id}",
                        observation_type="zone_activity",
                        zone_id=zone_id,
                        summary_text=summary_text,
                        source_metrics=source_metrics,
                        caveats=caveats,
                    )
                )

        # 2. Zone Dwell Observations (Phase 9)
        if dwell_summary is not None and dwell_summary.by_zone:
            for zone_id in sorted(dwell_summary.by_zone.keys()):
                zd: ZoneDwellSummary = dwell_summary.by_zone[zone_id]
                source_metrics = {
                    "total_dwell_sec": zd.total_dwell_sec,
                    "total_episodes": zd.total_episodes,
                    "unique_tracks": zd.unique_tracks,
                    "median_dwell_sec": zd.median_dwell_sec,
                    "mean_dwell_sec": zd.mean_dwell_sec,
                    "min_dwell_sec": zd.min_dwell_sec,
                    "max_dwell_sec": zd.max_dwell_sec,
                }
                summary_text = (
                    f"Zone '{zone_id}' dwell: {zd.total_episodes} episode(s) totaling "
                    f"{zd.total_dwell_sec:.1f}s observed span across {zd.unique_tracks} distinct track ID(s); "
                    f"median {zd.median_dwell_sec:.1f}s, mean {zd.mean_dwell_sec:.1f}s "
                    f"(min {zd.min_dwell_sec:.1f}s, max {zd.max_dwell_sec:.1f}s)."
                )
                caveats = (dwell_caveat, track_caveat, unit_caveat)
                observations.append(
                    DescriptiveSpatialObservation(
                        observation_id=f"OBS_ZONE_DWELL_{zone_id}",
                        observation_type="zone_dwell_span",
                        zone_id=zone_id,
                        summary_text=summary_text,
                        source_metrics=source_metrics,
                        caveats=caveats,
                    )
                )

        # 3. Direct Transition Pair Observations (Phase 11)
        if transition_summary is not None and transition_summary.transition_counts:
            for (from_z, to_z), count in sorted(transition_summary.transition_counts.items()):
                source_metrics = {
                    "from_zone": from_z,
                    "to_zone": to_z,
                    "transition_count": count,
                }
                summary_text = (
                    f"Direct transition '{from_z}' -> '{to_z}': {count} direct transition(s) "
                    "observed without intervening unassigned observations."
                )
                caveats = (
                    "Direct transitions record only immediate consecutive movement between distinct defined zones.",
                    "Any intermediate unassigned (None) observation breaks the transition chain.",
                )
                observations.append(
                    DescriptiveSpatialObservation(
                        observation_id=f"OBS_TRANSITION_{from_z}_{to_z}",
                        observation_type="direct_transition",
                        zone_id=None,
                        summary_text=summary_text,
                        source_metrics=source_metrics,
                        caveats=caveats,
                    )
                )

        # 4. Density Grid Observation (Phase 10)
        if density_grid is not None and density_grid.total_evaluated_points > 0:
            source_metrics = {
                "total_evaluated_points": density_grid.total_evaluated_points,
                "total_binned_points": density_grid.total_binned_points,
                "out_of_bounds_count": density_grid.out_of_bounds_count,
                "untransformed_count": density_grid.untransformed_count,
                "max_cell_count": density_grid.max_cell_count,
                "bounds_min_x": density_grid.bounds.min_x,
                "bounds_max_x": density_grid.bounds.max_x,
                "bounds_min_y": density_grid.bounds.min_y,
                "bounds_max_y": density_grid.bounds.max_y,
            }
            summary_text = (
                f"Spatial density grid: {density_grid.total_evaluated_points} evaluated point(s); "
                f"{density_grid.total_binned_points} binned, {density_grid.out_of_bounds_count} out-of-bounds "
                f"across bounds X=[{density_grid.bounds.min_x}, {density_grid.bounds.max_x}], "
                f"Y=[{density_grid.bounds.min_y}, {density_grid.bounds.max_y}]; "
                f"peak cell density is {density_grid.max_cell_count} observation(s)."
            )
            caveats = (
                "Point density accumulates discrete sampled detections per cell at sampling frame rate; "
                "it does NOT measure unique visitors or dwell duration.",
                unit_caveat,
            )
            observations.append(
                DescriptiveSpatialObservation(
                    observation_id="OBS_DENSITY_GRID",
                    observation_type="density_grid",
                    zone_id=None,
                    summary_text=summary_text,
                    source_metrics=source_metrics,
                    caveats=caveats,
                )
            )

        return SpatialObservationResult(
            is_eligible=True,
            status="sufficient_evidence",
            observations=tuple(observations),
            reasons_withheld=(),
            evidence_context=evidence_context,
            input_diagnostics=input_diagnostics,
        )


def generate_spatial_observations(
    evidence_context: Optional[EvidenceContext],
    dwell_summary: Optional[DwellSummary] = None,
    transition_summary: Optional[DirectTransitionSummary] = None,
    density_grid: Optional[TrajectoryDensityGrid] = None,
) -> SpatialObservationResult:
    """
    Convenience function: Evaluate evidence context and generate plain descriptive
    spatial observations from Phase 9-11 analytics summaries.
    """
    engine = SpatialObservationEngine()
    return engine.analyze(
        evidence_context=evidence_context,
        dwell_summary=dwell_summary,
        transition_summary=transition_summary,
        density_grid=density_grid,
    )
