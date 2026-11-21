"""
Validation Package
==================
Provides ground-truth data models, calibration reference schemas,
and validation sequence containers for evaluation and benchmarking.
"""

from src.validation.ground_truth import (
    GroundTruthPoint,
    GroundTruthEvent,
    GroundTruthCalibrationPoint,
    GroundTruthSequence,
    SUPPORTED_EVENT_TYPES,
    SUPPORTED_CALIBRATION_ROLES,
)
from src.validation.homography_validation import (
    HeldOutPointValidation,
    HomographyValidationResult,
    HomographyValidator,
    validate_homography,
)
from src.validation.zone_validation import (
    AlignedZoneObservation,
    ZoneDisagreement,
    ZoneValidationResult,
    ZoneValidator,
    validate_zone_assignments,
)
from src.validation.footfall_validation import (
    EntryCountComparison,
    TransitionCountComparison,
    FootfallMetricDisagreement,
    FootfallValidationResult,
    FootfallValidator,
    validate_footfall,
)
from src.validation.fragmentation import (
    PersonFragmentationRecord,
    TrackFragmentationResult,
    TrackFragmentationValidator,
    validate_track_fragmentation,
)
from src.validation.hysteresis_validation import (
    HysteresisWindowMetrics,
    HysteresisSensitivityResult,
    HysteresisSensitivityValidator,
    analyze_hysteresis_sensitivity,
)
from src.validation.annotation_validator import (
    IncompleteAnnotationError,
    AnnotationValidationError,
    WorkbookValidationReport,
    SegmentManifest,
    SamplingManifest,
    MERL_PILOT_MANIFEST,
    PRCV_SITE03_PILOT_MANIFEST,
    REGISTERED_PILOT_MANIFESTS,
    validate_annotation_workbook,
    convert_workbook_to_ground_truth_sequences,
)

__all__ = [
    "GroundTruthPoint",
    "GroundTruthEvent",
    "GroundTruthCalibrationPoint",
    "GroundTruthSequence",
    "SUPPORTED_EVENT_TYPES",
    "SUPPORTED_CALIBRATION_ROLES",
    "HeldOutPointValidation",
    "HomographyValidationResult",
    "HomographyValidator",
    "validate_homography",
    "AlignedZoneObservation",
    "ZoneDisagreement",
    "ZoneValidationResult",
    "ZoneValidator",
    "validate_zone_assignments",
    "EntryCountComparison",
    "TransitionCountComparison",
    "FootfallMetricDisagreement",
    "FootfallValidationResult",
    "FootfallValidator",
    "validate_footfall",
    "PersonFragmentationRecord",
    "TrackFragmentationResult",
    "TrackFragmentationValidator",
    "validate_track_fragmentation",
    "HysteresisWindowMetrics",
    "HysteresisSensitivityResult",
    "HysteresisSensitivityValidator",
    "analyze_hysteresis_sensitivity",
    "IncompleteAnnotationError",
    "AnnotationValidationError",
    "WorkbookValidationReport",
    "SegmentManifest",
    "SamplingManifest",
    "MERL_PILOT_MANIFEST",
    "PRCV_SITE03_PILOT_MANIFEST",
    "REGISTERED_PILOT_MANIFESTS",
    "validate_annotation_workbook",
    "convert_workbook_to_ground_truth_sequences",
]

