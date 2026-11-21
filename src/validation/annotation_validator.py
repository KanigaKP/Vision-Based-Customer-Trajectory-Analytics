"""
Annotation Completeness Validator and Ground-Truth Converter Module
===================================================================
Provides strict schema validation, independent sampling manifest enforcement,
pending-annotation rejection, multi-person consistency checks, coordinate
integrity validation, absence-aware event derivation, and conversion from human
annotation workbooks into canonical GroundTruthSequence instances.

Key Architectural Invariants:
1. Separation of Concerns:
   - Human annotation workbooks are separate from evaluation scoring inputs.
   - Model predictions are NEVER used to populate human ground truth.
2. Strict Rejection of Incomplete Workbooks:
   - Any frame marked PENDING_REVIEW, PENDING_HUMAN_ANNOTATION, or PENDING_PILOT_ANNOTATION
     is strictly rejected.
   - Incomplete workbooks cannot be scored, treated as empty scenes, unassigned
     zones, or valid track identities.
3. Independent Sampling Contract (Manifest Enforcement):
   - For recognized pilot datasets (merl_shopping_aisle, prcv_site03_boutique),
     the trusted registered manifest is enforced unless an explicit trusted manifest
     is supplied by the caller. Submitted segment definitions are strictly validated
     against that trusted contract.
   - For other datasets, an explicit trusted manifest is required for completeness
     certification. Completeness is NEVER certified solely from a submission's own declarations.
   - Schema validation is explicitly distinguished from coverage certification.
4. Robust Type Safety:
   - Malformed types (e.g. string timestamps, dict frame indices) produce clear validation
     errors and never crash validator execution via unchecked comparisons, arithmetic, or hashing.
5. Absence-Aware Event Derivation (Break Continuity Across Empty Frames):
   - Confirmed absence (REVIEWED_NO_PERSONS) breaks event continuity. It prevents emitting
     false direct transitions (e.g. A -> reviewed-empty -> B yields NO direct A->B transition)
     and ensures A -> reviewed-empty -> A does not imply uninterrupted occupancy.
   - Continuity is broken without inventing synthetic null-zone person observations or exact
     departure timestamps. Genuine missing observations and explicit null observations
     remain distinct and preserved.
6. Segment Boundary Integrity:
   - Disjoint evaluation windows are tagged with distinct segment IDs.
   - Episodes and transitions are strictly prohibited from bridging across segment gaps.
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Any, Tuple, Set

from src.validation.ground_truth import (
    GroundTruthPoint,
    GroundTruthEvent,
    GroundTruthCalibrationPoint,
    GroundTruthSequence,
)


class IncompleteAnnotationError(ValueError):
    """Raised when an annotation workbook contains pending frames or incomplete reviews."""
    pass


class AnnotationValidationError(ValueError):
    """Raised when an annotation workbook contains malformed or inconsistent records."""
    pass


def _is_finite_number(val: Any) -> bool:
    """Check if value is a finite int or float and not a boolean."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(val)


@dataclass(frozen=True)
class SegmentManifest:
    """
    Explicit specification of an evaluation segment.

    Fields:
        segment_id: Unique segment identifier string.
        start_frame: Initial frame index of the segment (inclusive).
        end_frame: Terminal frame index of the segment (inclusive).
        sampling_step: Frame stride between consecutive samples within the segment.
        name: Optional human-readable description.
    """
    segment_id: str
    start_frame: int
    end_frame: int
    sampling_step: int
    name: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.segment_id, str) or not self.segment_id.strip():
            raise ValueError(f"segment_id must be a non-empty string. Got: {self.segment_id!r}")
        if isinstance(self.start_frame, bool) or not isinstance(self.start_frame, int) or self.start_frame < 0:
            raise ValueError(f"start_frame must be a non-negative int. Got: {self.start_frame!r}")
        if isinstance(self.end_frame, bool) or not isinstance(self.end_frame, int) or self.end_frame < self.start_frame:
            raise ValueError(f"end_frame must be an int >= start_frame. Got: {self.end_frame!r}")
        if isinstance(self.sampling_step, bool) or not isinstance(self.sampling_step, int) or self.sampling_step <= 0:
            raise ValueError(f"sampling_step must be a positive int. Got: {self.sampling_step!r}")

    @property
    def expected_frame_indices(self) -> List[int]:
        """Generate list of expected frame indices from start_frame to end_frame with sampling_step."""
        return list(range(self.start_frame, self.end_frame + 1, self.sampling_step))

    @property
    def expected_frame_count(self) -> int:
        return len(self.expected_frame_indices)


@dataclass(frozen=True)
class SamplingManifest:
    """
    Explicit manifest of expected sampling coverage across all segments for a dataset.
    """
    dataset_name: str
    source_fps: float
    segments: List[SegmentManifest]
    target_fps: Optional[float] = None

    def __post_init__(self):
        if not isinstance(self.dataset_name, str) or not self.dataset_name.strip():
            raise ValueError(f"dataset_name must be a non-empty string. Got: {self.dataset_name!r}")
        if not _is_finite_number(self.source_fps) or self.source_fps <= 0.0:
            raise ValueError(f"source_fps must be a positive finite number. Got: {self.source_fps!r}")
        if not isinstance(self.segments, list) or len(self.segments) == 0:
            raise ValueError("segments must be a non-empty list of SegmentManifest.")
        seen_sids: Set[str] = set()
        for s in self.segments:
            if not isinstance(s, SegmentManifest):
                raise TypeError(f"Expected SegmentManifest instance, got: {type(s).__name__}")
            if s.segment_id in seen_sids:
                raise ValueError(f"Duplicate segment_id '{s.segment_id}' in manifest.")
            seen_sids.add(s.segment_id)

    @property
    def segment_ids(self) -> List[str]:
        return [s.segment_id for s in self.segments]

    def get_segment(self, segment_id: str) -> Optional[SegmentManifest]:
        for s in self.segments:
            if s.segment_id == segment_id:
                return s
        return None

    @property
    def total_expected_frames(self) -> int:
        return sum(s.expected_frame_count for s in self.segments)

    @property
    def all_expected_frame_indices(self) -> List[int]:
        result: List[int] = []
        for s in self.segments:
            result.extend(s.expected_frame_indices)
        return result


# Standard Diagnostic Pilot Manifests
MERL_PILOT_MANIFEST = SamplingManifest(
    dataset_name="merl_shopping_aisle",
    source_fps=30.0,
    target_fps=5.0,
    segments=[
        SegmentManifest(
            segment_id="merl_seg1_baseline",
            name="Segment 1: Negative Baseline & Initial Aisle Traversal",
            start_frame=0,
            end_frame=294,
            sampling_step=6,
        ),
        SegmentManifest(
            segment_id="merl_seg2_shelf_a_candidate",
            name="Segment 2: Candidate Zone shelf_a Approach, Dwell & Depart",
            start_frame=960,
            end_frame=1254,
            sampling_step=6,
        ),
    ],
)

PRCV_SITE03_PILOT_MANIFEST = SamplingManifest(
    dataset_name="prcv_site03_boutique",
    source_fps=15.0,
    target_fps=5.0,
    segments=[
        SegmentManifest(
            segment_id="prcv_site03_seg1_boutique",
            name="Segment 1: Multi-Person Boutique Browsing & Crossing",
            start_frame=0,
            end_frame=297,
            sampling_step=3,
        ),
    ],
)

# Registry of trusted pilot manifests
REGISTERED_PILOT_MANIFESTS: Dict[str, SamplingManifest] = {
    "merl_shopping_aisle": MERL_PILOT_MANIFEST,
    "prcv_site03_boutique": PRCV_SITE03_PILOT_MANIFEST,
}

# Authoritative scene profile associations for recognized retail datasets
TRUSTED_DATASET_PROFILES: Dict[str, str] = {
    "merl_shopping_aisle": "configs/merl_scene_profile.json",
    "prcv_site03_boutique": "configs/prcv_site03_scene_profile.json",
}


def get_trusted_dataset_zones(
    dataset_name: Optional[str] = None,
    profile_path: Optional[str] = None,
) -> Optional[List[str]]:
    """
    Obtain authoritative allowed retail zone IDs from a trusted scene profile.

    Zone IDs are resolved from:
    1. An explicit profile_path if provided.
    2. The TRUSTED_DATASET_PROFILES registry for recognized pilot datasets.

    Validation Invariant:
    Never trusts an allowed-zone list declared solely by a submitted workbook.

    Returns:
        Optional[List[str]]: List of valid zone_id strings, or None if no trusted profile exists.
    """
    resolved_path = profile_path
    if resolved_path is None and dataset_name in TRUSTED_DATASET_PROFILES:
        resolved_path = TRUSTED_DATASET_PROFILES[dataset_name]

    if resolved_path is None:
        return None

    if not os.path.isabs(resolved_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidate = os.path.join(repo_root, resolved_path)
        if os.path.exists(candidate):
            resolved_path = candidate

    if not os.path.exists(resolved_path):
        return None

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        zones = data.get("zones", [])
        zone_ids = []
        for z in zones:
            if isinstance(z, dict) and "zone_id" in z:
                zone_ids.append(str(z["zone_id"]))
        return zone_ids if zone_ids else None
    except Exception:
        return None


@dataclass
class WorkbookValidationReport:
    """
    Summary report produced by validate_annotation_workbook.

    Attributes:
        is_valid: True if workbook passes all schema and requested completeness checks.
        total_frames: Total number of frames present in the workbook.
        reviewed_empty_frames: Number of frames explicitly reviewed with 0 persons.
        reviewed_person_frames: Number of frames explicitly reviewed with >= 1 person.
        pending_frames: List of frame indices that remain in a pending/unreviewed status.
        total_person_observations: Total individual person observations across all frames.
        errors: List of fatal validation error descriptions.
        warnings: List of non-fatal warning messages.
        schema_valid: True if all structural schema, format, and coordinate checks pass.
        coverage_certified: True if frame coverage has been verified against an independent trusted manifest.
        coverage_status: One of ('certified', 'uncertified', 'failed').
        trusted_manifest_used: Dataset name of the trusted manifest used, or None.
        trusted_zones_enforced: List of allowed zone IDs enforced from trusted profile, or None.
    """
    is_valid: bool
    total_frames: int
    reviewed_empty_frames: int
    reviewed_person_frames: int
    pending_frames: List[int] = field(default_factory=list)
    total_person_observations: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    schema_valid: bool = True
    coverage_certified: bool = False
    coverage_status: str = "uncertified"
    trusted_manifest_used: Optional[str] = None
    trusted_zones_enforced: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "schema_valid": self.schema_valid,
            "coverage_certified": self.coverage_certified,
            "coverage_status": self.coverage_status,
            "trusted_manifest_used": self.trusted_manifest_used,
            "trusted_zones_enforced": self.trusted_zones_enforced,
            "total_frames": self.total_frames,
            "reviewed_empty_frames": self.reviewed_empty_frames,
            "reviewed_person_frames": self.reviewed_person_frames,
            "pending_frames_count": len(self.pending_frames),
            "pending_frames": self.pending_frames,
            "total_person_observations": self.total_person_observations,
            "errors": self.errors,
            "warnings": self.warnings,
        }


ALLOWED_UNCERTAINTIES = {"certain", "occluded", "truncated", "ambiguous_identity"}
ALLOWED_FRAME_STATUSES = {"REVIEWED_NO_PERSONS", "REVIEWED_PERSONS_PRESENT"}
PENDING_STATUS_MARKERS = {
    "PENDING_REVIEW",
    "PENDING_HUMAN_ANNOTATION",
    "PENDING_PILOT_ANNOTATION",
}


def _resolve_manifest(
    workbook: Dict[str, Any],
    manifest: Optional[SamplingManifest],
) -> Tuple[Optional[SamplingManifest], bool]:
    """
    Resolve trusted SamplingManifest for coverage verification.

    Returns:
        Tuple[Optional[SamplingManifest], bool]:
            - active_manifest: The trusted SamplingManifest (or None if uncertified).
            - is_trusted: True if supplied by caller or registered pilot dataset; False otherwise.
    """
    if manifest is not None:
        return manifest, True

    dataset_name = workbook.get("dataset_name")
    if isinstance(dataset_name, str) and dataset_name in REGISTERED_PILOT_MANIFESTS:
        return REGISTERED_PILOT_MANIFESTS[dataset_name], True

    # For other/unregistered datasets, never build a trusted manifest from submission declarations alone.
    return None, False


def validate_annotation_workbook(
    workbook: Dict[str, Any],
    manifest: Optional[SamplingManifest] = None,
    allowed_zones: Optional[Sequence[str]] = None,
    expected_sampling_step: Optional[int] = None,
    require_trusted_manifest: bool = False,
    profile_path: Optional[str] = None,
) -> WorkbookValidationReport:
    """
    Validate a human annotation workbook for completeness and consistency.

    Enforces:
    1. Rejection of any frame marked with a pending status.
    2. Strict alignment of frame_status:
       - 'REVIEWED_NO_PERSONS' requires persons == []
       - 'REVIEWED_PERSONS_PRESENT' requires len(persons) >= 1
    3. Multi-person integrity within frames (unique track labels per frame).
    4. Coordinate validity (finite bounding boxes, x1 < x2, y1 < y2).
       Preserves unavailable ground contact as None for occluded/truncated persons.
    5. Frame and timestamp monotonic progression and alignment with source_fps.
    6. Independent sampling contract enforcement:
       - Recognized pilot datasets enforce registered manifest unless caller provides trusted manifest.
       - Validates submitted segment definitions against trusted manifest.
       - For other datasets, requires trusted manifest for completeness certification.
       - Catches omitted frames, unexpected frames, off-grid frames, and undeclared segment IDs.
    7. Robust type-safety: malformed values produce validation errors without crashing.

    Args:
        workbook: Dictionary deserialized from a workbook JSON file.
        manifest: Optional explicit SamplingManifest.
        allowed_zones: Optional list of approved retail zone identifier strings.
        expected_sampling_step: Optional integer check for expected step size.
        require_trusted_manifest: If True, fails validation if no trusted manifest is available.

    Returns:
        WorkbookValidationReport: Structured validation report.
    """
    errors: List[str] = []
    coverage_errors: List[str] = []
    warnings: List[str] = []
    pending_frames: List[int] = []

    # 1. Top-level workbook fields
    if not isinstance(workbook, dict):
        return WorkbookValidationReport(
            is_valid=False,
            total_frames=0,
            reviewed_empty_frames=0,
            reviewed_person_frames=0,
            schema_valid=False,
            coverage_certified=False,
            coverage_status="failed",
            errors=["Workbook root must be a dictionary."],
        )

    version = workbook.get("workbook_version")
    if not version or not isinstance(version, str):
        errors.append("Workbook missing valid 'workbook_version' string.")

    source_fps = workbook.get("source_fps")
    if not _is_finite_number(source_fps) or source_fps <= 0.0:
        errors.append(f"Invalid or missing 'source_fps': {source_fps!r}")
        source_fps = 30.0  # Fallback to prevent divide-by-zero in later checks
    else:
        source_fps = float(source_fps)

    frames_data = workbook.get("frames")
    if not isinstance(frames_data, list) or len(frames_data) == 0:
        return WorkbookValidationReport(
            is_valid=False,
            total_frames=0,
            reviewed_empty_frames=0,
            reviewed_person_frames=0,
            schema_valid=False,
            coverage_certified=False,
            coverage_status="failed",
            errors=["Workbook 'frames' must be a non-empty list."],
        )

    # 2. Resolve trusted manifest
    active_manifest, is_trusted = _resolve_manifest(workbook, manifest)
    dataset_name = str(workbook.get("dataset_name", "unknown_dataset"))

    # Resolve authoritative allowed retail zone IDs from trusted scene profile
    if allowed_zones is None:
        trusted_profile_zones = get_trusted_dataset_zones(dataset_name, profile_path=profile_path)
        if trusted_profile_zones is not None:
            allowed_zones = trusted_profile_zones

    # Audit workbook-declared target_retail_zones against trusted profile zones
    submitted_retail_zones = workbook.get("target_retail_zones")
    if isinstance(submitted_retail_zones, list) and allowed_zones is not None:
        allowed_zones_set = set(allowed_zones)
        unrecognized_zones = [z for z in submitted_retail_zones if z not in allowed_zones_set]
        if unrecognized_zones:
            warnings.append(
                f"Submitted workbook declares unauthorized 'target_retail_zones' {unrecognized_zones} "
                f"not found in authoritative scene profile {sorted(list(allowed_zones_set))}. "
                f"Validation strictly enforces trusted profile zones and rejects workbook overrides."
            )

    # Determine effective global sampling step if specified
    effective_step: Optional[int] = expected_sampling_step
    if effective_step is None:
        wb_step = workbook.get("sampling_step")
        if isinstance(wb_step, int) and not isinstance(wb_step, bool) and wb_step > 0:
            effective_step = wb_step

    declared_segment_ids: Optional[Set[str]] = None

    if active_manifest is not None:
        declared_segment_ids = set(active_manifest.segment_ids)

        # Validate submitted segments against the trusted manifest contract
        submitted_segments = workbook.get("segments")
        if isinstance(submitted_segments, list):
            submitted_by_id: Dict[str, Dict[str, Any]] = {}
            for s in submitted_segments:
                if isinstance(s, dict):
                    sid = s.get("segment_id") or s.get("name")
                    if sid:
                        submitted_by_id[str(sid)] = s

            for seg_m in active_manifest.segments:
                if seg_m.segment_id not in submitted_by_id:
                    coverage_errors.append(
                        f"Submitted segments definition missing required segment '{seg_m.segment_id}' from trusted manifest."
                    )
                else:
                    s_dict = submitted_by_id[seg_m.segment_id]
                    s_range = s_dict.get("frame_range")
                    expected_range = [seg_m.start_frame, seg_m.end_frame]
                    if s_range != expected_range:
                        coverage_errors.append(
                            f"Submitted segment '{seg_m.segment_id}' frame_range {s_range} does not match trusted manifest range {expected_range}."
                        )
                    s_step = s_dict.get("sampling_step")
                    if s_step is not None and s_step != seg_m.sampling_step:
                        coverage_errors.append(
                            f"Submitted segment '{seg_m.segment_id}' sampling_step {s_step} does not match trusted manifest step {seg_m.sampling_step}."
                        )

            # Check for undeclared extra segments in workbook definitions
            for sid_sub in submitted_by_id.keys():
                if sid_sub not in declared_segment_ids:
                    coverage_errors.append(
                        f"Submitted segment '{sid_sub}' is not defined in trusted manifest: {sorted(list(declared_segment_ids))}."
                    )
    else:
        # No trusted manifest available: read declared segments from submission for schema parsing
        segments_config = workbook.get("segments")
        if isinstance(segments_config, list):
            s_set: Set[str] = set()
            for seg in segments_config:
                if isinstance(seg, dict):
                    sid = seg.get("segment_id") or seg.get("name")
                    if isinstance(sid, str) and sid.strip():
                        s_set.add(str(sid))
            if s_set:
                declared_segment_ids = s_set

        if require_trusted_manifest:
            coverage_errors.append(
                f"Completeness certification failed: Dataset '{dataset_name}' requires an explicit trusted SamplingManifest. Cannot certify completeness from submission declarations alone."
            )
        else:
            warnings.append(
                f"Coverage not certified: Dataset '{dataset_name}' is not a recognized pilot dataset and no trusted SamplingManifest was supplied. Schema validation passed, but completeness cannot be certified from submission declarations alone."
            )

    total_frames = len(frames_data)
    reviewed_empty_count = 0
    reviewed_person_count = 0
    total_observations = 0

    seen_frame_indices: Set[int] = set()
    frames_by_segment: Dict[str, List[int]] = {}
    observed_segment_order: List[str] = []

    prev_frame_idx: Optional[int] = None
    prev_timestamp: Optional[float] = None
    current_segment_id: Optional[str] = None

    for idx, f_entry in enumerate(frames_data):
        if not isinstance(f_entry, dict):
            errors.append(f"Frame record at record index {idx} must be a dictionary.")
            continue

        f_idx = f_entry.get("frame_idx")
        t_sec = f_entry.get("timestamp_sec")
        status = f_entry.get("frame_status")
        seg_id = f_entry.get("segment_id")

        # 3. Validate frame_idx safely
        f_idx_valid = True
        if isinstance(f_idx, bool) or not isinstance(f_idx, int) or f_idx < 0:
            errors.append(
                f"Frame record at record index {idx} has invalid 'frame_idx': {f_idx!r} "
                f"(must be a non-negative integer)."
            )
            f_idx_valid = False
        else:
            if f_idx in seen_frame_indices:
                errors.append(f"Duplicate frame_idx {f_idx} detected at record index {idx}.")
            seen_frame_indices.add(f_idx)

        frame_identifier = f"Frame {f_idx}" if f_idx_valid else f"Record at index {idx}"

        # 4. Validate segment_id safely
        seg_id_str: Optional[str] = None
        if seg_id is not None:
            if not isinstance(seg_id, str) or not seg_id.strip():
                errors.append(f"{frame_identifier} has invalid 'segment_id': {seg_id!r} (must be non-empty string).")
            else:
                seg_id_str = str(seg_id)
                if declared_segment_ids is not None and seg_id_str not in declared_segment_ids:
                    errors.append(
                        f"{frame_identifier} has undeclared segment_id '{seg_id_str}'. "
                        f"Declared segments: {sorted(list(declared_segment_ids))}."
                    )
        else:
            if declared_segment_ids is not None:
                errors.append(
                    f"{frame_identifier} is missing required 'segment_id'. "
                    f"Declared segments: {sorted(list(declared_segment_ids))}."
                )

        if seg_id_str is not None:
            if seg_id_str not in frames_by_segment:
                frames_by_segment[seg_id_str] = []
                observed_segment_order.append(seg_id_str)
            if f_idx_valid:
                frames_by_segment[seg_id_str].append(f_idx)

        # 5. Check segment sequence transitions & monotonicity
        if seg_id_str != current_segment_id:
            if seg_id_str in observed_segment_order[:-1]:
                errors.append(
                    f"Segment interleaving violation at {frame_identifier}: segment '{seg_id_str}' "
                    f"reappears after previous segment '{current_segment_id}'."
                )
            current_segment_id = seg_id_str
            prev_frame_idx = None
            prev_timestamp = None

        if f_idx_valid:
            if prev_frame_idx is not None and f_idx <= prev_frame_idx:
                errors.append(
                    f"Non-monotonic frame sequence within segment '{current_segment_id}': "
                    f"frame {f_idx} appears after {prev_frame_idx}."
                )
            prev_frame_idx = f_idx

        # 6. Validate timestamp_sec safely
        t_sec_float: Optional[float] = None
        if not _is_finite_number(t_sec) or t_sec < 0.0:
            errors.append(
                f"{frame_identifier} has invalid 'timestamp_sec': {t_sec!r} "
                f"(must be a non-negative finite number)."
            )
        else:
            t_sec_float = float(t_sec)
            if f_idx_valid:
                expected_t = round(f_idx / source_fps, 3)
                if abs(t_sec_float - expected_t) > 0.005:
                    errors.append(
                        f"{frame_identifier} timestamp {t_sec_float}s deviates from expected {expected_t:.3f}s "
                        f"(based on source_fps={source_fps})."
                    )
            if prev_timestamp is not None and t_sec_float <= prev_timestamp:
                errors.append(
                    f"Non-monotonic timestamp sequence within segment '{current_segment_id}': "
                    f"timestamp {t_sec_float}s appears after {prev_timestamp}s."
                )
            prev_timestamp = t_sec_float

        # 7. Check sampling grid alignment
        if f_idx_valid:
            step_for_frame: Optional[int] = None
            origin_for_frame: int = 0
            if active_manifest is not None and seg_id_str is not None:
                seg_manifest = active_manifest.get_segment(seg_id_str)
                if seg_manifest is not None:
                    step_for_frame = seg_manifest.sampling_step
                    origin_for_frame = seg_manifest.start_frame

            if step_for_frame is None and effective_step is not None:
                step_for_frame = effective_step

            if step_for_frame is not None and step_for_frame > 1:
                if (f_idx - origin_for_frame) % step_for_frame != 0:
                    errors.append(
                        f"{frame_identifier} is off-grid for sampling step {step_for_frame} "
                        f"(segment '{seg_id_str}' origin={origin_for_frame})."
                    )

        # 8. Check review status
        if not isinstance(status, str) or status in PENDING_STATUS_MARKERS:
            if f_idx_valid:
                pending_frames.append(f_idx)
            errors.append(
                f"{frame_identifier} (timestamp {t_sec!r}) has pending review status: {status!r}. "
                f"Evaluation scoring requires complete human annotation."
            )
            continue

        if status not in ALLOWED_FRAME_STATUSES:
            errors.append(
                f"{frame_identifier} has unrecognized 'frame_status': {status!r}. "
                f"Allowed values are {sorted(list(ALLOWED_FRAME_STATUSES))}."
            )
            continue

        persons = f_entry.get("persons")
        if not isinstance(persons, list):
            errors.append(f"{frame_identifier} 'persons' must be a list, got {type(persons).__name__}.")
            continue

        # 9. Check empty vs persons consistency
        if status == "REVIEWED_NO_PERSONS":
            reviewed_empty_count += 1
            if len(persons) > 0:
                errors.append(
                    f"{frame_identifier} is marked REVIEWED_NO_PERSONS but contains {len(persons)} person observations."
                )
        elif status == "REVIEWED_PERSONS_PRESENT":
            reviewed_person_count += 1
            if len(persons) == 0:
                errors.append(
                    f"{frame_identifier} is marked REVIEWED_PERSONS_PRESENT but contains 0 person observations."
                )

            seen_labels_in_frame: Set[str] = set()

            for p_idx, person in enumerate(persons):
                if not isinstance(person, dict):
                    errors.append(f"{frame_identifier} person observation {p_idx} must be a dictionary.")
                    continue

                total_observations += 1
                track_label = person.get("annotator_track_label")
                if not isinstance(track_label, str) or not track_label.strip():
                    errors.append(f"{frame_identifier} person {p_idx} missing valid 'annotator_track_label'.")
                else:
                    if track_label in seen_labels_in_frame:
                        errors.append(
                            f"{frame_identifier} contains duplicate 'annotator_track_label' '{track_label}'."
                        )
                    seen_labels_in_frame.add(track_label)

                # Bounding box validation
                bbox = person.get("image_bbox")
                if bbox is not None:
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        errors.append(f"{frame_identifier} person '{track_label}' image_bbox must be 4 elements.")
                    else:
                        if not all(_is_finite_number(v) for v in bbox):
                            errors.append(f"{frame_identifier} person '{track_label}' image_bbox contains non-finite values.")
                        else:
                            x1, y1, x2, y2 = [float(v) for v in bbox]
                            if x1 < 0.0 or y1 < 0.0 or x2 <= x1 or y2 <= y1:
                                errors.append(
                                    f"{frame_identifier} person '{track_label}' has invalid image_bbox coordinates "
                                    f"[{x1}, {y1}, {x2}, {y2}]."
                                )

                # Ground contact validation (can be None for occluded/truncated people)
                contact = person.get("ground_contact_px")
                if contact is not None:
                    if not isinstance(contact, (list, tuple)) or len(contact) != 2:
                        errors.append(
                            f"{frame_identifier} person '{track_label}' ground_contact_px must be 2 elements [x, y]."
                        )
                    else:
                        if not all(_is_finite_number(v) for v in contact):
                            errors.append(f"{frame_identifier} person '{track_label}' ground_contact_px contains non-finite values.")
                        else:
                            cx, cy = [float(v) for v in contact]
                            if cx < 0.0 or cy < 0.0:
                                errors.append(
                                    f"{frame_identifier} person '{track_label}' ground_contact_px has negative coordinates [{cx}, {cy}]."
                                )

                # Uncertainty validation
                uncertainty = person.get("uncertainty", "certain")
                if not isinstance(uncertainty, str) or uncertainty not in ALLOWED_UNCERTAINTIES:
                    errors.append(
                        f"{frame_identifier} person '{track_label}' has unrecognized uncertainty {uncertainty!r}. "
                        f"Allowed: {sorted(list(ALLOWED_UNCERTAINTIES))}."
                    )

                # Zone validation
                zone = person.get("zone_id")
                if zone is not None:
                    if not isinstance(zone, str) or not zone.strip():
                        errors.append(f"{frame_identifier} person '{track_label}' has invalid zone_id: {zone!r}.")
                    elif allowed_zones is not None and zone not in allowed_zones:
                        errors.append(
                            f"{frame_identifier} person '{track_label}' references zone '{zone}' "
                            f"not in trusted profile allowed_zones: {sorted(list(allowed_zones))}."
                        )

    # 10. Coverage validation against trusted SamplingManifest
    if active_manifest is not None:
        expected_seg_order = [s.segment_id for s in active_manifest.segments]
        filtered_observed = [s for s in observed_segment_order if s in declared_segment_ids]
        filtered_expected = [s for s in expected_seg_order if s in filtered_observed]
        if filtered_observed != filtered_expected:
            coverage_errors.append(
                f"Segment ordering mismatch: observed {filtered_observed}, expected {filtered_expected}."
            )

        for seg_manifest in active_manifest.segments:
            expected_frames = seg_manifest.expected_frame_indices
            expected_set = set(expected_frames)
            actual_frames = frames_by_segment.get(seg_manifest.segment_id, [])
            actual_set = set(actual_frames)

            missing_frames = expected_set - actual_set
            if missing_frames:
                sample_missing = sorted(list(missing_frames))[:5]
                coverage_errors.append(
                    f"Omitted expected frames in segment '{seg_manifest.segment_id}': "
                    f"expected {len(expected_set)} frames, found {len(actual_set)}. "
                    f"Missing {len(missing_frames)} frame(s) (e.g. {sample_missing})."
                )

            unexpected_frames = actual_set - expected_set
            if unexpected_frames:
                sample_unexpected = sorted(list(unexpected_frames))[:5]
                coverage_errors.append(
                    f"Unexpected frames in segment '{seg_manifest.segment_id}': "
                    f"{len(unexpected_frames)} frame(s) outside expected range (e.g. {sample_unexpected})."
                )

    all_errors = errors + coverage_errors
    schema_valid = (len(errors) == 0 and len(pending_frames) == 0)

    if active_manifest is not None:
        if len(coverage_errors) == 0 and schema_valid:
            coverage_certified = True
            coverage_status = "certified"
        else:
            coverage_certified = False
            coverage_status = "failed"
    else:
        coverage_certified = False
        coverage_status = "failed" if require_trusted_manifest else "uncertified"

    is_valid = (len(all_errors) == 0 and len(pending_frames) == 0)

    return WorkbookValidationReport(
        is_valid=is_valid,
        total_frames=total_frames,
        reviewed_empty_frames=reviewed_empty_count,
        reviewed_person_frames=reviewed_person_count,
        pending_frames=pending_frames,
        total_person_observations=total_observations,
        errors=all_errors,
        warnings=warnings,
        schema_valid=schema_valid,
        coverage_certified=coverage_certified,
        coverage_status=coverage_status,
        trusted_manifest_used=active_manifest.dataset_name if active_manifest else None,
        trusted_zones_enforced=sorted(list(allowed_zones)) if allowed_zones is not None else None,
    )


def convert_workbook_to_ground_truth_sequences(
    workbook: Dict[str, Any],
    manifest: Optional[SamplingManifest] = None,
    allowed_zones: Optional[Sequence[str]] = None,
    expected_sampling_step: Optional[int] = None,
    require_trusted_manifest: bool = False,
    profile_path: Optional[str] = None,
) -> Dict[str, GroundTruthSequence]:
    """
    Validate and convert a completed human annotation workbook into canonical GroundTruthSequence instances.

    Each disjoint segment in the workbook is converted into its own distinct
    GroundTruthSequence to strictly prevent episodes or transitions from bridging
    across separate evaluation windows.

    Absence-Aware Event Continuity:
    - Confirmed absence (REVIEWED_NO_PERSONS) explicitly breaks event continuity without
      inventing synthetic departure timestamps or null-zone person observations.
    - An intervening empty frame prevents emitting direct transitions (A -> empty -> B yields
      NO direct transition).
    - An intervening empty frame prevents continuous stay inference (A -> empty -> A yields
      two separate entries into A).
    - Production zero-bridging semantics: an unassigned observation (zone_id = None)
      breaks an active stay and prevents direct transition to another zone.
    - Single-frame visits have duration_sec = 0.0s.
    - Segment boundaries: events are NEVER computed across segment gaps.

    Preserves exact reviewed frame identities:
    - Empty reviewed frames (REVIEWED_NO_PERSONS) are recorded in seq.metadata["empty_frame_indices"].
    - Evaluated frame indices are recorded in seq.metadata["evaluated_frame_indices"].
    - Detailed reviewed frame records are stored in seq.metadata["reviewed_frames"].
    - GroundTruthPoint instances retain annotator uncertainty flags ('certain', 'occluded', etc.).

    Args:
        workbook: Deserialized annotation workbook dictionary.
        manifest: Optional explicit SamplingManifest.
        allowed_zones: Optional list of approved retail zone identifier strings.
        expected_sampling_step: Optional integer check for expected step size.
        require_trusted_manifest: If True, requires trusted manifest for conversion.

    Returns:
        Dict[str, GroundTruthSequence]: Mapping of segment_id -> GroundTruthSequence.

    Raises:
        IncompleteAnnotationError: If workbook contains pending or unreviewed frames.
        AnnotationValidationError: If workbook contains schema or coverage inconsistencies.
    """
    report = validate_annotation_workbook(
        workbook,
        manifest=manifest,
        allowed_zones=allowed_zones,
        expected_sampling_step=expected_sampling_step,
        require_trusted_manifest=require_trusted_manifest,
        profile_path=profile_path,
    )
    if not report.is_valid:
        if report.pending_frames:
            raise IncompleteAnnotationError(
                f"Cannot convert workbook: contains {len(report.pending_frames)} pending frames. "
                f"First pending frame_idx: {report.pending_frames[0]}."
            )
        raise AnnotationValidationError(
            f"Cannot convert workbook: {len(report.errors)} validation errors detected:\n"
            + "\n".join(report.errors[:10])
        )

    dataset_name = str(workbook.get("dataset_name", "sequence"))
    source_fps = float(workbook.get("source_fps", 30.0))
    target_fps = float(workbook.get("target_fps", source_fps))
    frames_data = workbook["frames"]

    # Group frames by segment_id (default to "seg_01" if unsegmented)
    frames_by_segment: Dict[str, List[Dict[str, Any]]] = {}
    for f in frames_data:
        seg_id = str(f.get("segment_id", "seg_01"))
        if seg_id not in frames_by_segment:
            frames_by_segment[seg_id] = []
        frames_by_segment[seg_id].append(f)

    sequences: Dict[str, GroundTruthSequence] = {}

    for seg_id, seg_frames in frames_by_segment.items():
        points: List[GroundTruthPoint] = []
        track_observations: Dict[str, List[Tuple[int, float, Optional[str]]]] = {}

        empty_frame_indices: List[int] = []
        evaluated_frame_indices: List[int] = []
        reviewed_frames_record: List[Dict[str, Any]] = []

        for f in seg_frames:
            f_idx = int(f["frame_idx"])
            t_sec = float(f["timestamp_sec"])
            status = str(f["frame_status"])
            persons_in_frame = f.get("persons", [])

            evaluated_frame_indices.append(f_idx)
            reviewed_frames_record.append({
                "frame_idx": f_idx,
                "timestamp_sec": t_sec,
                "frame_status": status,
                "person_count": len(persons_in_frame),
            })

            if status == "REVIEWED_NO_PERSONS":
                empty_frame_indices.append(f_idx)

            elif status == "REVIEWED_PERSONS_PRESENT":
                for person in persons_in_frame:
                    track_label = str(person["annotator_track_label"])
                    bbox = person.get("image_bbox")
                    contact = person.get("ground_contact_px")
                    zone_id = person.get("zone_id")
                    uncertainty = person.get("uncertainty", "certain")

                    # Handle missing ground contact without fabricating coordinates
                    img_x = float(contact[0]) if contact is not None else None
                    img_y = float(contact[1]) if contact is not None else None

                    pt = GroundTruthPoint(
                        gt_id=track_label,
                        frame_idx=f_idx,
                        timestamp_sec=t_sec,
                        image_bbox=[float(v) for v in bbox] if bbox is not None else None,
                        image_x=img_x,
                        image_y=img_y,
                        floor_x=None,  # Not invented; floor coordinates require verified survey
                        floor_y=None,
                        zone_id=zone_id,
                        visible=True,
                        uncertainty=uncertainty,
                    )
                    points.append(pt)

                    if track_label not in track_observations:
                        track_observations[track_label] = []
                    track_observations[track_label].append((f_idx, t_sec, zone_id))

        empty_frame_set = set(empty_frame_indices)

        # Extract events per track strictly within this segment
        events: List[GroundTruthEvent] = []
        for track_label, obs_list in track_observations.items():
            sorted_obs = sorted(obs_list, key=lambda x: (x[0], x[1]))
            current_zone: Optional[str] = None
            prev_obs_frame: Optional[int] = None

            for obs_idx, (f_idx, t_sec, z_id) in enumerate(sorted_obs):
                # Break event continuity if confirmed absence occurred between observations
                if prev_obs_frame is not None:
                    intervening_empty = any(
                        prev_obs_frame < ef < f_idx for ef in empty_frame_set
                    )
                    if intervening_empty:
                        # Confirmed absence occurred between prev_obs_frame and f_idx.
                        # Break event continuity: resets current_zone to None without
                        # inventing synthetic departure timestamps or null-zone points.
                        current_zone = None

                prev_obs_frame = f_idx

                if z_id == current_zone:
                    continue  # Staying in same zone during continuous observation

                if current_zone is not None:
                    if z_id is not None:
                        # Direct transition between consecutive non-None zones
                        events.append(
                            GroundTruthEvent(
                                event_type="transition",
                                gt_id=track_label,
                                frame_idx=f_idx,
                                timestamp_sec=t_sec,
                                from_zone_id=current_zone,
                                to_zone_id=z_id,
                            )
                        )
                    else:
                        # Exiting to unassigned (None)
                        events.append(
                            GroundTruthEvent(
                                event_type="exit",
                                gt_id=track_label,
                                frame_idx=f_idx,
                                timestamp_sec=t_sec,
                                from_zone_id=current_zone,
                                to_zone_id=None,
                            )
                        )
                else:
                    # Entering a zone from unassigned or after broken continuity
                    if z_id is not None:
                        events.append(
                            GroundTruthEvent(
                                event_type="entry",
                                gt_id=track_label,
                                frame_idx=f_idx,
                                timestamp_sec=t_sec,
                                from_zone_id=None,
                                to_zone_id=z_id,
                            )
                        )

                current_zone = z_id

        seq = GroundTruthSequence(
            schema_version="1.0",
            sequence_id=f"{dataset_name}_{seg_id}",
            source="human_annotation_workbook",
            fps=target_fps,
            points=points,
            events=events,
            calibration_points=[],  # No invented calibration landmarks
            metadata={
                "dataset_name": dataset_name,
                "segment_id": seg_id,
                "source_fps": source_fps,
                "target_fps": target_fps,
                "total_frames_in_segment": len(seg_frames),
                "empty_frame_indices": sorted(empty_frame_indices),
                "evaluated_frame_indices": sorted(evaluated_frame_indices),
                "reviewed_frames": reviewed_frames_record,
                "coverage_certified": report.coverage_certified,
                "coverage_status": report.coverage_status,
                "trusted_manifest_used": report.trusted_manifest_used,
                "evaluation_category": workbook.get("evaluation_category", "diagnostic_evaluation"),
            },
        )
        sequences[seg_id] = seq

    return sequences
