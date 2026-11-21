"""
Analytics package for Vision-Based Customer Trajectory Analytics.
Phase 6: Spatial Zone Definition & Evaluation.
"""

from src.analytics.zone import Zone, ZoneConfig
from src.analytics.zone_evaluator import ZoneEvaluator
from src.analytics.footfall import (
    ZoneEntryEvent,
    ZoneVisit,
    FootfallSummary,
    FootfallAnalyzer,
)
from src.analytics.dwell import (
    DwellEpisode,
    TrackDwellSummary,
    ZoneDwellSummary,
    ZoneSpanDistribution,
    DwellSummary,
    DwellAnalyzer,
    extract_dwell_episodes,
    analyze_dwell,
    compute_zone_span_distribution,
)

from src.analytics.density import (
    GridBounds,
    TrajectoryDensityGrid,
    compute_density_grid,
    render_density_heatmap,
)

from src.analytics.transitions import (
    ZoneVisitRun,
    ObservedTrackSequence,
    ZoneActivitySummary,
    DirectTransitionSummary,
    TransitionAnalyzer,
    extract_zone_visit_runs,
    extract_track_sequence,
    compute_direct_transitions,
    compute_zone_activity_summaries,
)

from src.analytics.spatial_observations import (
    EvidenceContext,
    DescriptiveSpatialObservation,
    SpatialObservationResult,
    evaluate_evidence_eligibility,
    SpatialObservationEngine,
    generate_spatial_observations,
)

from src.analytics.candidate_diagnostics import (
    CandidateZoneMetrics,
    ObjectiveEvaluation,
    CandidateDiagnosticsResult,
    SUPPORTED_OBJECTIVES,
    OBJECTIVE_PRIMARY_METRIC,
    OBJECTIVE_METRIC_LABELS,
    compute_hotspot_with_track_lineage,
    evaluate_single_objective,
    generate_candidate_diagnostics,
)

__all__ = [
    "Zone",
    "ZoneConfig",
    "ZoneEvaluator",
    "ZoneEntryEvent",
    "ZoneVisit",
    "FootfallSummary",
    "FootfallAnalyzer",
    "DwellEpisode",
    "TrackDwellSummary",
    "ZoneDwellSummary",
    "ZoneSpanDistribution",
    "DwellSummary",
    "DwellAnalyzer",
    "extract_dwell_episodes",
    "analyze_dwell",
    "compute_zone_span_distribution",
    "GridBounds",
    "TrajectoryDensityGrid",
    "compute_density_grid",
    "render_density_heatmap",
    "ZoneVisitRun",
    "ObservedTrackSequence",
    "ZoneActivitySummary",
    "DirectTransitionSummary",
    "TransitionAnalyzer",
    "extract_zone_visit_runs",
    "extract_track_sequence",
    "compute_direct_transitions",
    "compute_zone_activity_summaries",
    "EvidenceContext",
    "DescriptiveSpatialObservation",
    "SpatialObservationResult",
    "evaluate_evidence_eligibility",
    "SpatialObservationEngine",
    "generate_spatial_observations",
    "CandidateZoneMetrics",
    "ObjectiveEvaluation",
    "CandidateDiagnosticsResult",
    "SUPPORTED_OBJECTIVES",
    "OBJECTIVE_PRIMARY_METRIC",
    "OBJECTIVE_METRIC_LABELS",
    "compute_hotspot_with_track_lineage",
    "evaluate_single_objective",
    "generate_candidate_diagnostics",
]



