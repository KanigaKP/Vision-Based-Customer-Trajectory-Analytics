"""
Track Fragmentation Analysis Module
===================================
Measures observed track fragmentation and frame coverage for tracked objects
under an explicit ground-truth-to-tracker correspondence mapping.

Architectural Rules & Methodological Boundaries:
1. Explicit Correspondence Only:
   - Requires caller-supplied explicit correspondence mapping: gt_id -> List[track_id].
   - Multiple tracker IDs mapped to a single ground-truth identity represent observed
     fragmentation under that correspondence.
   - Strictly prohibits automatic matching, rematching, Re-ID, appearance association,
     spatial proximity heuristics, or IoU overlap inference.
2. Incomplete or Ambiguous Mapping Rejection:
   - Rejects duplicate tracker IDs within a person's mapping.
   - Rejects ambiguous reuse of a single tracker ID across multiple ground-truth identities.
   - Rejects unknown ground-truth IDs and unknown tracker IDs.
3. Overlap-Safe Frame Coverage:
   - Evaluates unique frame sets per person and tracker to prevent artificial double-counting
     when multiple associated tracker tracks overlap in time.
4. Descriptive Validation Metric:
   - Reports 'observed fragmentation under supplied correspondence'.
   - Does NOT compute or claim MOTA, IDF1, or identity-switch accuracy without formal
     benchmark annotations and protocols.
   - Synthetic test fixtures demonstrate validation logic only and do not constitute
     real-world tracking verification.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import defaultdict

from src.trajectory.trajectory import Trajectory, TrajectoryPoint
from src.tracking.bytetrack import Track
from src.validation.ground_truth import GroundTruthPoint, GroundTruthSequence


@dataclass
class PersonFragmentationRecord:
    """
    Fragmentation and frame coverage record for a single ground-truth identity.

    Attributes:
        gt_id: Canonical ground-truth identity string.
        associated_track_ids: Sorted list of explicitly mapped tracker track IDs.
        fragment_count: Number of tracker tracks associated with this ground-truth person.
        is_fragmented: True if fragment_count > 1, False otherwise.
        gt_frame_count: Number of distinct frames where this identity is observed in ground truth.
        mapped_tracker_frame_count: Number of distinct frames covered by any associated tracker track
            (calculated as a set union to prevent double-counting across overlapping tracks).
        covered_frame_count: Number of distinct frames where both ground truth and at least one
            mapped tracker track are present (intersection).
        coverage_ratio: Proportion covered_frame_count / gt_frame_count (or None if gt_frame_count == 0).
    """
    gt_id: str
    associated_track_ids: List[int]
    fragment_count: int
    is_fragmented: bool
    gt_frame_count: int
    mapped_tracker_frame_count: int
    covered_frame_count: int
    coverage_ratio: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "gt_id": self.gt_id,
            "associated_track_ids": list(self.associated_track_ids),
            "fragment_count": self.fragment_count,
            "is_fragmented": self.is_fragmented,
            "gt_frame_count": self.gt_frame_count,
            "mapped_tracker_frame_count": self.mapped_tracker_frame_count,
            "covered_frame_count": self.covered_frame_count,
            "coverage_ratio": self.coverage_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonFragmentationRecord":
        """Reconstruct PersonFragmentationRecord from dictionary."""
        rate = data.get("coverage_ratio")
        return cls(
            gt_id=data["gt_id"],
            associated_track_ids=[int(x) for x in data["associated_track_ids"]],
            fragment_count=int(data["fragment_count"]),
            is_fragmented=bool(data["is_fragmented"]),
            gt_frame_count=int(data["gt_frame_count"]),
            mapped_tracker_frame_count=int(data["mapped_tracker_frame_count"]),
            covered_frame_count=int(data["covered_frame_count"]),
            coverage_ratio=float(rate) if rate is not None else None,
        )


@dataclass
class TrackFragmentationResult:
    """
    Aggregate evaluation results for track fragmentation under supplied correspondence.

    Attributes:
        dataset_name: Identifier string for sequence or evaluation run.
        person_records: List of PersonFragmentationRecord per evaluated ground-truth identity.
        total_gt_identities: Total evaluated ground-truth persons.
        total_fragmented_identities: Count of ground-truth persons with fragment_count > 1.
        total_unfragmented_identities: Count of ground-truth persons with fragment_count == 1.
        total_observed_fragments: Total tracker track IDs mapped across all evaluated persons.
        fragmentation_rate: Proportion total_fragmented_identities / total_gt_identities.
        mean_fragments_per_identity: Average fragment count per ground-truth identity.
        overall_coverage_ratio: Total covered frames across all persons / total ground-truth frames.
        evidence_status: 'FRAGMENTATION_EVALUATED' or 'INSUFFICIENT_EVIDENCE'.
        evidence_note: Methodological scope explanation.
    """
    dataset_name: str
    person_records: List[PersonFragmentationRecord]
    total_gt_identities: int
    total_fragmented_identities: int
    total_unfragmented_identities: int
    total_observed_fragments: int
    fragmentation_rate: Optional[float]
    mean_fragments_per_identity: Optional[float]
    overall_coverage_ratio: Optional[float]
    evidence_status: str
    evidence_note: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "person_records": [r.to_dict() for r in self.person_records],
            "total_gt_identities": self.total_gt_identities,
            "total_fragmented_identities": self.total_fragmented_identities,
            "total_unfragmented_identities": self.total_unfragmented_identities,
            "total_observed_fragments": self.total_observed_fragments,
            "fragmentation_rate": self.fragmentation_rate,
            "mean_fragments_per_identity": self.mean_fragments_per_identity,
            "overall_coverage_ratio": self.overall_coverage_ratio,
            "evidence_status": self.evidence_status,
            "evidence_note": self.evidence_note,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackFragmentationResult":
        """Reconstruct TrackFragmentationResult from dictionary."""
        frag_rate = data.get("fragmentation_rate")
        mean_frag = data.get("mean_fragments_per_identity")
        cov_rate = data.get("overall_coverage_ratio")
        return cls(
            dataset_name=data.get("dataset_name", "default"),
            person_records=[PersonFragmentationRecord.from_dict(r) for r in data.get("person_records", [])],
            total_gt_identities=int(data["total_gt_identities"]),
            total_fragmented_identities=int(data["total_fragmented_identities"]),
            total_unfragmented_identities=int(data["total_unfragmented_identities"]),
            total_observed_fragments=int(data["total_observed_fragments"]),
            fragmentation_rate=float(frag_rate) if frag_rate is not None else None,
            mean_fragments_per_identity=float(mean_frag) if mean_frag is not None else None,
            overall_coverage_ratio=float(cov_rate) if cov_rate is not None else None,
            evidence_status=data["evidence_status"],
            evidence_note=data["evidence_note"],
        )


class TrackFragmentationValidator:
    """
    Validates and measures observed track fragmentation under explicit correspondence.
    """

    @staticmethod
    def _extract_tracker_frames(
        tracker_observations: Union[List[Trajectory], List[TrajectoryPoint], List[Track], Dict[int, Set[int]]]
    ) -> Dict[int, Set[int]]:
        """
        Extract a dictionary mapping track_id (int) -> Set[frame_idx (int)] from supported tracker objects.
        """
        tracker_frames: Dict[int, Set[int]] = defaultdict(set)

        if isinstance(tracker_observations, dict):
            for t_id, frames in tracker_observations.items():
                if isinstance(t_id, bool) or not isinstance(t_id, int):
                    raise TypeError(f"Tracker ID must be an integer. Got: {t_id!r}")
                for f in frames:
                    if isinstance(f, bool) or not isinstance(f, int) or f < 0:
                        raise ValueError(f"Invalid frame index {f!r} for track_id {t_id}.")
                    tracker_frames[t_id].add(f)
            return dict(tracker_frames)

        if not isinstance(tracker_observations, list):
            raise TypeError(f"tracker_observations must be a list or dict. Got: {type(tracker_observations).__name__}")

        for item in tracker_observations:
            if isinstance(item, Trajectory):
                t_id = item.track_id
                for pt in item.points:
                    if pt.frame_idx < 0:
                        raise ValueError(f"Invalid frame index {pt.frame_idx} in Trajectory.")
                    tracker_frames[t_id].add(pt.frame_idx)
            elif isinstance(item, TrajectoryPoint):
                if item.frame_idx < 0:
                    raise ValueError(f"Invalid frame index {item.frame_idx} in TrajectoryPoint.")
                tracker_frames[item.track_id].add(item.frame_idx)
            elif isinstance(item, Track):
                t_id = item.track_id
                for tp in item.history:
                    if tp.frame_id < 0:
                        raise ValueError(f"Invalid frame index {tp.frame_id} in TrackPoint.")
                    tracker_frames[t_id].add(tp.frame_id)
            else:
                raise TypeError(
                    f"Unsupported tracker observation type in list: {type(item).__name__}. "
                    f"Expected Trajectory, TrajectoryPoint, or Track."
                )

        return dict(tracker_frames)

    @classmethod
    def validate_fragmentation(
        cls,
        gt_observations: Union[List[GroundTruthPoint], GroundTruthSequence],
        tracker_observations: Union[List[Trajectory], List[TrajectoryPoint], List[Track], Dict[int, Set[int]]],
        correspondence: Optional[Dict[str, List[int]]],
        dataset_name: str = "default",
    ) -> TrackFragmentationResult:
        """
        Analyze observed track fragmentation and frame coverage for ground-truth individuals
        under an explicit correspondence mapping.

        Args:
            gt_observations: GroundTruthSequence or list of GroundTruthPoint records.
            tracker_observations: Trajectories, trajectory points, tracks, or pre-indexed frames.
            correspondence: Explicit mapping from ground-truth identity (str) to list of tracker IDs (List[int]).
            dataset_name: Identifier string.

        Returns:
            TrackFragmentationResult with per-person records and aggregate metrics.

        Raises:
            TypeError: For incorrect argument types.
            ValueError: For missing correspondence, duplicate tracker IDs, ambiguous mappings,
                or unknown ground-truth/tracker IDs.
        """
        # 1. Normalize ground-truth points
        if isinstance(gt_observations, GroundTruthSequence):
            gt_points = gt_observations.points
            if dataset_name == "default":
                dataset_name = gt_observations.sequence_id
        elif isinstance(gt_observations, list):
            gt_points = gt_observations
        else:
            raise TypeError(
                f"gt_observations must be a GroundTruthSequence or List[GroundTruthPoint]. "
                f"Got: {type(gt_observations).__name__}"
            )

        for idx, pt in enumerate(gt_points):
            if not isinstance(pt, GroundTruthPoint):
                raise TypeError(f"Item at index {idx} in gt_observations is not a GroundTruthPoint.")

        # 2. Check empty ground truth
        if len(gt_points) == 0:
            return TrackFragmentationResult(
                dataset_name=dataset_name,
                person_records=[],
                total_gt_identities=0,
                total_fragmented_identities=0,
                total_unfragmented_identities=0,
                total_observed_fragments=0,
                fragmentation_rate=None,
                mean_fragments_per_identity=None,
                overall_coverage_ratio=None,
                evidence_status="INSUFFICIENT_EVIDENCE",
                evidence_note="No ground-truth observations provided. Track fragmentation cannot be evaluated.",
            )

        # 3. Validate correspondence existence
        if correspondence is None or not isinstance(correspondence, dict) or len(correspondence) == 0:
            raise ValueError(
                "An explicit correspondence mapping from ground-truth ID (str) to tracker ID(s) (List[int]) "
                "must be provided. Automatic identity matching or Re-ID is strictly prohibited."
            )

        # 4. Validate correspondence structure and reject duplicates / ambiguities
        seen_tracker_ids: Dict[int, str] = {}
        validated_correspondence: Dict[str, List[int]] = {}

        for gt_id, track_ids in correspondence.items():
            if not isinstance(gt_id, str) or not gt_id.strip():
                raise ValueError(f"Ground-truth ID in correspondence must be a non-empty string. Got: {gt_id!r}")
            if not isinstance(track_ids, (list, tuple, set)):
                raise TypeError(f"Correspondence track IDs for gt_id '{gt_id}' must be a list. Got: {type(track_ids).__name__}")
            if len(track_ids) == 0:
                raise ValueError(f"Correspondence list for gt_id '{gt_id}' cannot be empty.")

            # Validate each tracker ID
            id_list: List[int] = []
            for t_id in track_ids:
                if isinstance(t_id, bool) or not isinstance(t_id, int):
                    raise TypeError(f"Tracker ID must be an integer. Got: {t_id!r} in mapping for '{gt_id}'.")
                id_list.append(t_id)

            # Check duplicate tracker ID inside single mapping
            if len(id_list) != len(set(id_list)):
                raise ValueError(f"Duplicate tracker ID found in correspondence mapping for gt_id '{gt_id}': {id_list}")

            # Check ambiguous reuse of tracker ID across multiple ground-truth identities
            for t_id in id_list:
                if t_id in seen_tracker_ids:
                    raise ValueError(
                        f"Ambiguous correspondence: tracker ID {t_id} is mapped to multiple ground-truth "
                        f"identities: '{seen_tracker_ids[t_id]}' and '{gt_id}'."
                    )
                seen_tracker_ids[t_id] = gt_id

            validated_correspondence[gt_id] = sorted(id_list)

        # 5. Extract tracker frame map
        tracker_frames = cls._extract_tracker_frames(tracker_observations)

        # 6. Check unknown tracker IDs
        for t_id, owner_gt in seen_tracker_ids.items():
            if t_id not in tracker_frames:
                raise ValueError(
                    f"Unknown tracker ID {t_id} in correspondence for gt_id '{owner_gt}'; "
                    f"not found in tracker observations."
                )

        # 7. Group ground-truth frames by gt_id
        gt_frames_by_id: Dict[str, Set[int]] = defaultdict(set)
        for pt in gt_points:
            gt_frames_by_id[pt.gt_id].add(pt.frame_idx)

        # 8. Check for unknown ground-truth IDs
        for gt_id in validated_correspondence:
            if gt_id not in gt_frames_by_id:
                raise ValueError(
                    f"Unknown ground-truth ID '{gt_id}' in correspondence; not present in ground-truth observations."
                )

        for gt_id in gt_frames_by_id:
            if gt_id not in validated_correspondence:
                raise ValueError(
                    f"Ground-truth ID '{gt_id}' has no explicit correspondence mapping. "
                    f"Incomplete correspondence is not silently inferred."
                )

        # 9. Compute fragmentation and coverage metrics deterministically
        person_records: List[PersonFragmentationRecord] = []
        total_gt_frames_all = 0
        total_covered_frames_all = 0

        for gt_id in sorted(validated_correspondence.keys()):
            associated_t_ids = validated_correspondence[gt_id]
            frag_count = len(associated_t_ids)
            is_frag = (frag_count > 1)

            gt_f_set = gt_frames_by_id[gt_id]
            gt_f_count = len(gt_f_set)
            total_gt_frames_all += gt_f_count

            # Union of frames across all mapped tracker tracks (prevents double-counting overlapping tracks)
            mapped_tracker_f_set: Set[int] = set()
            for t_id in associated_t_ids:
                mapped_tracker_f_set.update(tracker_frames.get(t_id, set()))

            mapped_tracker_f_count = len(mapped_tracker_f_set)

            # Intersection between GT frames and mapped tracker frames
            covered_f_set = gt_f_set & mapped_tracker_f_set
            covered_f_count = len(covered_f_set)
            total_covered_frames_all += covered_f_count

            cov_ratio = round(covered_f_count / gt_f_count, 4) if gt_f_count > 0 else None

            person_records.append(
                PersonFragmentationRecord(
                    gt_id=gt_id,
                    associated_track_ids=associated_t_ids,
                    fragment_count=frag_count,
                    is_fragmented=is_frag,
                    gt_frame_count=gt_f_count,
                    mapped_tracker_frame_count=mapped_tracker_f_count,
                    covered_frame_count=covered_f_count,
                    coverage_ratio=cov_ratio,
                )
            )

        total_gt_identities = len(person_records)
        total_fragmented = sum(1 for r in person_records if r.is_fragmented)
        total_unfragmented = total_gt_identities - total_fragmented
        total_fragments = sum(r.fragment_count for r in person_records)

        frag_rate = round(total_fragmented / total_gt_identities, 4) if total_gt_identities > 0 else None
        mean_frags = round(total_fragments / total_gt_identities, 4) if total_gt_identities > 0 else None
        overall_cov = round(total_covered_frames_all / total_gt_frames_all, 4) if total_gt_frames_all > 0 else None

        return TrackFragmentationResult(
            dataset_name=dataset_name,
            person_records=person_records,
            total_gt_identities=total_gt_identities,
            total_fragmented_identities=total_fragmented,
            total_unfragmented_identities=total_unfragmented,
            total_observed_fragments=total_fragments,
            fragmentation_rate=frag_rate,
            mean_fragments_per_identity=mean_frags,
            overall_coverage_ratio=overall_cov,
            evidence_status="FRAGMENTATION_EVALUATED",
            evidence_note=(
                f"Evaluated observed track fragmentation across {total_gt_identities} ground-truth identities "
                f"under explicit correspondence ({total_fragmented} fragmented, {total_fragments} total fragments). "
                f"Note: Descriptive measurement under supplied correspondence only; does not perform automatic "
                f"identity matching or claim benchmark MOTA/IDF1 tracking accuracy."
            ),
        )


def validate_track_fragmentation(
    gt_observations: Union[List[GroundTruthPoint], GroundTruthSequence],
    tracker_observations: Union[List[Trajectory], List[TrajectoryPoint], List[Track], Dict[int, Set[int]]],
    correspondence: Optional[Dict[str, List[int]]],
    dataset_name: str = "default",
) -> TrackFragmentationResult:
    """
    Convenience function to analyze observed track fragmentation and frame coverage.
    """
    return TrackFragmentationValidator.validate_fragmentation(
        gt_observations=gt_observations,
        tracker_observations=tracker_observations,
        correspondence=correspondence,
        dataset_name=dataset_name,
    )
