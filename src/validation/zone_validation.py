"""
Zone Assignment Agreement Validation Module
===========================================
Compares Phase 6 spatial zone assignments against independently supplied
ground-truth zone labels on explicitly aligned observation points.

Key Architectural Rules:
1. Explicit Correspondence Only:
   - Requires explicit mapping between ground-truth identities and tracker IDs.
   - Never infers identity, performs Re-ID, or rematches tracks by distance or heuristic.
   - Missing alignment is rejected explicitly; it is never silently inferred.
2. Zone Evaluation Decoupling:
   - Does NOT rerun Point-in-Polygon or hysteresis filtering.
   - Compares pre-existing zone labels without altering or tuning them.
3. Outside Area Representation:
   - Uses `None` consistently to denote exterior or unassigned store floor regions.
   - Preserves `None` as a valid categorical state in comparisons and confusion matrices.
4. Descriptive Statistic:
   - Reports "label agreement" rather than universal zone-classification accuracy.
   - Synthetic fixtures demonstrate verification logic only; no real-world accuracy is claimed.
"""

from dataclasses import dataclass, asdict
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

from src.trajectory.trajectory import TrajectoryPoint
from src.validation.ground_truth import GroundTruthPoint


def _is_finite_number(val: Any) -> bool:
    """Helper to check if a value is a finite int or float (excluding booleans)."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(val)


def _normalize_zone_key(k: Any) -> Optional[str]:
    """Helper to convert JSON 'null' string or None to None, preserving string zone names."""
    if k is None or k == "null":
        return None
    return str(k)


@dataclass
class AlignedZoneObservation:
    """
    Represents an explicitly aligned observation pair between ground truth and tracker.

    Attributes:
        gt_id: Canonical ground-truth individual identity string.
        track_id: Session-local tracker identifier integer.
        frame_idx: Video frame index.
        timestamp_sec: Video timestamp in seconds.
        expected_zone: Ground-truth zone label (or None if outside all defined zones).
        predicted_zone: Tracker-assigned zone label (or None if outside all defined zones).
    """
    gt_id: str
    track_id: int
    frame_idx: int
    timestamp_sec: float
    expected_zone: Optional[str]
    predicted_zone: Optional[str]

    def __post_init__(self):
        if not isinstance(self.gt_id, str) or not self.gt_id.strip():
            raise ValueError(f"gt_id must be a non-empty string. Got: {self.gt_id!r}")
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int):
            raise ValueError(f"track_id must be an integer. Got: {self.track_id!r}")
        if isinstance(self.frame_idx, bool) or not isinstance(self.frame_idx, int) or self.frame_idx < 0:
            raise ValueError(f"frame_idx must be a non-negative integer. Got: {self.frame_idx!r}")
        if not _is_finite_number(self.timestamp_sec) or self.timestamp_sec < 0.0:
            raise ValueError(f"timestamp_sec must be a non-negative finite number. Got: {self.timestamp_sec!r}")
        self.timestamp_sec = float(self.timestamp_sec)
        if self.expected_zone is not None and not isinstance(self.expected_zone, str):
            raise ValueError(f"expected_zone must be a string or None. Got: {self.expected_zone!r}")
        if self.predicted_zone is not None and not isinstance(self.predicted_zone, str):
            raise ValueError(f"predicted_zone must be a string or None. Got: {self.predicted_zone!r}")

    @property
    def is_match(self) -> bool:
        """Return True if expected and predicted zone labels match identically."""
        return self.expected_zone == self.predicted_zone

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "gt_id": self.gt_id,
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
            "timestamp_sec": self.timestamp_sec,
            "expected_zone": self.expected_zone,
            "predicted_zone": self.predicted_zone,
            "is_match": self.is_match,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlignedZoneObservation":
        """Reconstruct AlignedZoneObservation from dictionary."""
        return cls(
            gt_id=data["gt_id"],
            track_id=int(data["track_id"]),
            frame_idx=int(data["frame_idx"]),
            timestamp_sec=float(data["timestamp_sec"]),
            expected_zone=_normalize_zone_key(data.get("expected_zone")),
            predicted_zone=_normalize_zone_key(data.get("predicted_zone")),
        )


@dataclass
class ZoneDisagreement:
    """
    Detailed record of an observation where expected and predicted zone labels differ.

    Attributes:
        gt_id: Ground-truth identity string.
        track_id: Session-local tracker identifier.
        frame_idx: Frame index where disagreement occurred.
        timestamp_sec: Video timestamp in seconds.
        expected_zone: Expected ground-truth zone label (or None).
        predicted_zone: Predicted zone assignment (or None).
    """
    gt_id: str
    track_id: int
    frame_idx: int
    timestamp_sec: float
    expected_zone: Optional[str]
    predicted_zone: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "gt_id": self.gt_id,
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
            "timestamp_sec": self.timestamp_sec,
            "expected_zone": self.expected_zone,
            "predicted_zone": self.predicted_zone,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZoneDisagreement":
        """Reconstruct ZoneDisagreement from dictionary."""
        return cls(
            gt_id=data["gt_id"],
            track_id=int(data["track_id"]),
            frame_idx=int(data["frame_idx"]),
            timestamp_sec=float(data["timestamp_sec"]),
            expected_zone=_normalize_zone_key(data.get("expected_zone")),
            predicted_zone=_normalize_zone_key(data.get("predicted_zone")),
        )


@dataclass
class ZoneValidationResult:
    """
    Aggregate evaluation results comparing expected vs predicted zone labels.

    Attributes:
        total_aligned_observations: Count of evaluated observation pairs.
        matched_zone_count: Count of observations where labels matched.
        mismatched_zone_count: Count of observations where labels differed.
        agreement_rate: Descriptive proportion matched / total (or None if total == 0).
        expected_zone_counts: Distribution of expected zone occurrences.
        predicted_zone_counts: Distribution of predicted zone occurrences.
        confusion_matrix: 2D table keyed by expected_zone -> predicted_zone -> count.
        disagreements: List of specific ZoneDisagreement instances.
        evidence_status: 'ZONE_AGREEMENT_EVALUATED' or 'INSUFFICIENT_EVIDENCE'.
        evidence_note: Methodological note on evidence scope.
        dataset_name: Dataset identifier string.
    """
    total_aligned_observations: int
    matched_zone_count: int
    mismatched_zone_count: int
    agreement_rate: Optional[float]
    expected_zone_counts: Dict[Optional[str], int]
    predicted_zone_counts: Dict[Optional[str], int]
    confusion_matrix: Dict[Optional[str], Dict[Optional[str], int]]
    disagreements: List[ZoneDisagreement]
    evidence_status: str
    evidence_note: str
    dataset_name: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        # Convert Optional[str] keys in dicts to None/string for JSON serialization
        cm_serialized = {
            k_exp: {k_pred: count for k_pred, count in pred_dict.items()}
            for k_exp, pred_dict in self.confusion_matrix.items()
        }
        return {
            "dataset_name": self.dataset_name,
            "total_aligned_observations": self.total_aligned_observations,
            "matched_zone_count": self.matched_zone_count,
            "mismatched_zone_count": self.mismatched_zone_count,
            "agreement_rate": self.agreement_rate,
            "expected_zone_counts": dict(self.expected_zone_counts),
            "predicted_zone_counts": dict(self.predicted_zone_counts),
            "confusion_matrix": cm_serialized,
            "disagreements": [d.to_dict() for d in self.disagreements],
            "evidence_status": self.evidence_status,
            "evidence_note": self.evidence_note,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZoneValidationResult":
        """Reconstruct ZoneValidationResult from dictionary, restoring None keys."""
        # Restore None keys in expected_zone_counts
        raw_exp_counts = data.get("expected_zone_counts", {})
        expected_zone_counts = {
            _normalize_zone_key(k): int(v) for k, v in raw_exp_counts.items()
        }

        # Restore None keys in predicted_zone_counts
        raw_pred_counts = data.get("predicted_zone_counts", {})
        predicted_zone_counts = {
            _normalize_zone_key(k): int(v) for k, v in raw_pred_counts.items()
        }

        # Restore None keys in confusion_matrix
        raw_cm = data.get("confusion_matrix", {})
        confusion_matrix = {
            _normalize_zone_key(k_exp): {
                _normalize_zone_key(k_pred): int(v)
                for k_pred, v in pred_map.items()
            }
            for k_exp, pred_map in raw_cm.items()
        }

        disagreements = [
            ZoneDisagreement.from_dict(d) for d in data.get("disagreements", [])
        ]

        rate = data.get("agreement_rate")
        agreement_rate = float(rate) if rate is not None else None

        return cls(
            total_aligned_observations=data["total_aligned_observations"],
            matched_zone_count=data["matched_zone_count"],
            mismatched_zone_count=data["mismatched_zone_count"],
            agreement_rate=agreement_rate,
            expected_zone_counts=expected_zone_counts,
            predicted_zone_counts=predicted_zone_counts,
            confusion_matrix=confusion_matrix,
            disagreements=disagreements,
            evidence_status=data["evidence_status"],
            evidence_note=data["evidence_note"],
            dataset_name=data.get("dataset_name", "default"),
        )


class ZoneValidator:
    """
    Validates agreement between predicted spatial zone assignments and ground truth.
    Requires explicit correspondence between ground-truth identities and tracker IDs.
    """

    @staticmethod
    def validate_observations(
        observations: List[AlignedZoneObservation],
        dataset_name: str = "default",
        known_zones: Optional[List[str]] = None,
    ) -> ZoneValidationResult:
        """
        Evaluate label agreement across a collection of explicitly aligned observation pairs.

        Args:
            observations: List of AlignedZoneObservation instances.
            dataset_name: Identifier string for sequence or evaluation run.
            known_zones: Optional list of defined zone names to ensure matrix representation.

        Returns:
            ZoneValidationResult containing counts, agreement rate, confusion matrix,
            and disagreement details.

        Raises:
            ValueError: If duplicate alignment keys are detected.
        """
        if not isinstance(observations, list):
            raise TypeError(f"observations must be a list. Got: {type(observations).__name__}")

        # Check for duplicate alignment keys
        seen_gt_keys: Set[Tuple[str, int]] = set()
        seen_track_keys: Set[Tuple[int, int]] = set()

        for obs in observations:
            if not isinstance(obs, AlignedZoneObservation):
                raise TypeError(f"Expected AlignedZoneObservation, got: {type(obs).__name__}")

            gt_key = (obs.gt_id, obs.frame_idx)
            if gt_key in seen_gt_keys:
                raise ValueError(
                    f"Duplicate alignment key detected: gt_id='{obs.gt_id}' at frame_idx {obs.frame_idx}."
                )
            seen_gt_keys.add(gt_key)

            track_key = (obs.track_id, obs.frame_idx)
            if track_key in seen_track_keys:
                raise ValueError(
                    f"Duplicate alignment key detected: track_id={obs.track_id} at frame_idx {obs.frame_idx}."
                )
            seen_track_keys.add(track_key)

        # Handle empty observations explicitly
        if len(observations) == 0:
            return ZoneValidationResult(
                total_aligned_observations=0,
                matched_zone_count=0,
                mismatched_zone_count=0,
                agreement_rate=None,
                expected_zone_counts={},
                predicted_zone_counts={},
                confusion_matrix={},
                disagreements=[],
                evidence_status="INSUFFICIENT_EVIDENCE",
                evidence_note="No aligned zone observations provided. Zone label agreement cannot be evaluated.",
                dataset_name=dataset_name,
            )

        # Collect all categorical states (named zones + None for outside)
        all_zones_set: Set[Optional[str]] = set(known_zones) if known_zones else set()
        for obs in observations:
            all_zones_set.add(obs.expected_zone)
            all_zones_set.add(obs.predicted_zone)

        # Deterministic ordering: sorted strings, followed by None if present
        named_zones = sorted([z for z in all_zones_set if z is not None])
        ordered_zones: List[Optional[str]] = list(named_zones)
        if None in all_zones_set:
            ordered_zones.append(None)

        # Initialize counts and confusion matrix
        expected_counts: Dict[Optional[str], int] = {z: 0 for z in ordered_zones}
        predicted_counts: Dict[Optional[str], int] = {z: 0 for z in ordered_zones}
        confusion_matrix: Dict[Optional[str], Dict[Optional[str], int]] = {
            z_exp: {z_pred: 0 for z_pred in ordered_zones} for z_exp in ordered_zones
        }

        matched = 0
        mismatched = 0
        disagreements: List[ZoneDisagreement] = []

        for obs in observations:
            expected_counts[obs.expected_zone] += 1
            predicted_counts[obs.predicted_zone] += 1
            confusion_matrix[obs.expected_zone][obs.predicted_zone] += 1

            if obs.is_match:
                matched += 1
            else:
                mismatched += 1
                disagreements.append(
                    ZoneDisagreement(
                        gt_id=obs.gt_id,
                        track_id=obs.track_id,
                        frame_idx=obs.frame_idx,
                        timestamp_sec=obs.timestamp_sec,
                        expected_zone=obs.expected_zone,
                        predicted_zone=obs.predicted_zone,
                    )
                )

        total = len(observations)
        agreement_rate = round(matched / total, 4)

        return ZoneValidationResult(
            total_aligned_observations=total,
            matched_zone_count=matched,
            mismatched_zone_count=mismatched,
            agreement_rate=agreement_rate,
            expected_zone_counts=expected_counts,
            predicted_zone_counts=predicted_counts,
            confusion_matrix=confusion_matrix,
            disagreements=disagreements,
            evidence_status="ZONE_AGREEMENT_EVALUATED",
            evidence_note=(
                f"Evaluated {total} explicitly aligned observations across {len(ordered_zones)} "
                f"zone states. Note: Descriptive label agreement on supplied test data only; "
                f"does not prove universal physical tracking or zone accuracy."
            ),
            dataset_name=dataset_name,
        )

    @classmethod
    def align_and_validate(
        cls,
        expected_points: List[GroundTruthPoint],
        predicted_points: List[TrajectoryPoint],
        id_mapping: Dict[str, int],
        dataset_name: str = "default",
        known_zones: Optional[List[str]] = None,
        require_complete_alignment: bool = True,
    ) -> ZoneValidationResult:
        """
        Align expected GroundTruthPoints with predicted TrajectoryPoints via an explicit ID mapping,
        and evaluate zone label agreement.

        Args:
            expected_points: Ground-truth point observations.
            predicted_points: Tracker trajectory point observations.
            id_mapping: Explicit dictionary mapping gt_id (str) -> track_id (int).
            dataset_name: Identifier string.
            known_zones: Optional list of known zone labels.
            require_complete_alignment: If True, raises ValueError if any expected point lacks
                a corresponding predicted point at that frame index.

        Returns:
            ZoneValidationResult.

        Raises:
            ValueError: If id_mapping is empty/missing, if duplicate observations exist,
                or if expected observations lack a corresponding predicted point.
        """
        if not id_mapping or not isinstance(id_mapping, dict):
            raise ValueError(
                "An explicit id_mapping dictionary from ground-truth ID (str) to tracker ID (int) "
                "must be provided. Automatic ID matching, Re-ID, or heuristic pairing is not permitted."
            )

        # Build predicted lookup: (track_id, frame_idx) -> TrajectoryPoint
        predicted_map: Dict[Tuple[int, int], TrajectoryPoint] = {}
        for pt in predicted_points:
            key = (pt.track_id, pt.frame_idx)
            if key in predicted_map:
                raise ValueError(
                    f"Duplicate predicted observation detected for track_id={pt.track_id} at frame_idx {pt.frame_idx}."
                )
            predicted_map[key] = pt

        # Build aligned observations
        aligned_obs: List[AlignedZoneObservation] = []
        seen_expected: Set[Tuple[str, int]] = set()

        for gt_pt in expected_points:
            exp_key = (gt_pt.gt_id, gt_pt.frame_idx)
            if exp_key in seen_expected:
                raise ValueError(
                    f"Duplicate ground-truth observation detected for gt_id='{gt_pt.gt_id}' at frame_idx {gt_pt.frame_idx}."
                )
            seen_expected.add(exp_key)

            if gt_pt.gt_id not in id_mapping:
                raise ValueError(
                    f"Ground-truth ID '{gt_pt.gt_id}' has no explicit tracker ID mapping in id_mapping."
                )
            mapped_track_id = id_mapping[gt_pt.gt_id]

            pred_key = (mapped_track_id, gt_pt.frame_idx)
            if pred_key not in predicted_map:
                if require_complete_alignment:
                    raise ValueError(
                        f"Missing predicted observation for track_id={mapped_track_id} (gt_id='{gt_pt.gt_id}') "
                        f"at frame_idx {gt_pt.frame_idx}. Missing alignment is not silently inferred."
                    )
                continue

            pred_pt = predicted_map[pred_key]
            aligned_obs.append(
                AlignedZoneObservation(
                    gt_id=gt_pt.gt_id,
                    track_id=mapped_track_id,
                    frame_idx=gt_pt.frame_idx,
                    timestamp_sec=gt_pt.timestamp_sec,
                    expected_zone=gt_pt.zone_id,
                    predicted_zone=pred_pt.zone_id,
                )
            )

        return cls.validate_observations(
            observations=aligned_obs,
            dataset_name=dataset_name,
            known_zones=known_zones,
        )


def validate_zone_assignments(
    observations: List[AlignedZoneObservation],
    dataset_name: str = "default",
    known_zones: Optional[List[str]] = None,
) -> ZoneValidationResult:
    """
    Convenience function to perform zone label agreement validation on aligned observations.
    """
    return ZoneValidator.validate_observations(
        observations=observations,
        dataset_name=dataset_name,
        known_zones=known_zones,
    )
