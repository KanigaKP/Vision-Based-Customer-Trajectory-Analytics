"""
Ground-Truth Representation & Validation Schema Module
=====================================================
Defines standardized, machine-readable schemas and validation rules for
ground-truth annotations (points, events, calibration points, sequences).

Phase 8.1 Scope:
Provides data structures and JSON serialization for ground-truth reference data.
Does NOT compute homography accuracy, footfall comparison, dwell analytics,
heatmaps, or layout recommendations.
"""

from dataclasses import dataclass, field
import json
import math
import os
from typing import List, Dict, Any, Optional, Tuple


SUPPORTED_EVENT_TYPES = ("entry", "exit", "transition")
SUPPORTED_CALIBRATION_ROLES = ("training", "held_out")


def _is_finite_number(val: Any) -> bool:
    """Helper to check if a value is a finite int or float (excluding booleans)."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(val)


@dataclass
class GroundTruthPoint:
    """
    Represents one manually or externally annotated observation of an individual.

    Required fields:
        gt_id: Canonical ground-truth identity string within the sequence.
        frame_idx: Non-negative integer frame index.
        timestamp_sec: Non-negative finite timestamp in seconds.

    Optional fields:
        image_bbox: Four-element list [x1, y1, x2, y2] bounding box in image coordinates.
        image_x: Annotated image-space X coordinate (e.g., bottom-center).
        image_y: Annotated image-space Y coordinate (e.g., bottom-center).
        floor_x: Externally measured or annotated floor-plane X coordinate.
        floor_y: Externally measured or annotated floor-plane Y coordinate.
        zone_id: Optional retail zone identifier string, or None if outside all zones.
        visible: Boolean visibility flag.
    """
    gt_id: str
    frame_idx: int
    timestamp_sec: float
    image_bbox: Optional[List[float]] = None
    image_x: Optional[float] = None
    image_y: Optional[float] = None
    floor_x: Optional[float] = None
    floor_y: Optional[float] = None
    zone_id: Optional[str] = None
    visible: bool = True
    uncertainty: Optional[str] = None

    def __post_init__(self):
        # 1. gt_id validation
        if not isinstance(self.gt_id, str) or not self.gt_id.strip():
            raise ValueError(f"gt_id must be a non-empty string. Got: {self.gt_id!r}")

        # 2. frame_idx validation
        if isinstance(self.frame_idx, bool) or not isinstance(self.frame_idx, int) or self.frame_idx < 0:
            raise ValueError(f"frame_idx must be a non-negative integer. Got: {self.frame_idx!r}")

        # 3. timestamp_sec validation
        if not _is_finite_number(self.timestamp_sec) or self.timestamp_sec < 0.0:
            raise ValueError(f"timestamp_sec must be a non-negative finite number. Got: {self.timestamp_sec!r}")
        self.timestamp_sec = float(self.timestamp_sec)

        # 4. image_bbox validation
        if self.image_bbox is not None:
            if not isinstance(self.image_bbox, (list, tuple)) or len(self.image_bbox) != 4:
                raise ValueError(
                    f"image_bbox must contain exactly four values [x1, y1, x2, y2]. Got: {self.image_bbox!r}"
                )
            for i, val in enumerate(self.image_bbox):
                if not _is_finite_number(val):
                    raise ValueError(
                        f"image_bbox element at index {i} must be a finite number. Got: {val!r}"
                    )
            self.image_bbox = [float(v) for v in self.image_bbox]

        # 5. image_x / image_y completeness & finiteness validation
        if (self.image_x is None) != (self.image_y is None):
            raise ValueError(
                f"image_x and image_y must either both be present or both be absent. "
                f"Got image_x={self.image_x!r}, image_y={self.image_y!r}"
            )
        if self.image_x is not None and self.image_y is not None:
            if not _is_finite_number(self.image_x) or not _is_finite_number(self.image_y):
                raise ValueError(
                    f"image_x and image_y must be finite numbers. "
                    f"Got image_x={self.image_x!r}, image_y={self.image_y!r}"
                )
            self.image_x = float(self.image_x)
            self.image_y = float(self.image_y)

        # 6. floor_x / floor_y completeness & finiteness validation
        if (self.floor_x is None) != (self.floor_y is None):
            raise ValueError(
                f"floor_x and floor_y must either both be present or both be absent. "
                f"Got floor_x={self.floor_x!r}, floor_y={self.floor_y!r}"
            )
        if self.floor_x is not None and self.floor_y is not None:
            if not _is_finite_number(self.floor_x) or not _is_finite_number(self.floor_y):
                raise ValueError(
                    f"floor_x and floor_y must be finite numbers. "
                    f"Got floor_x={self.floor_x!r}, floor_y={self.floor_y!r}"
                )
            self.floor_x = float(self.floor_x)
            self.floor_y = float(self.floor_y)

        # 7. zone_id validation
        if self.zone_id is not None and not isinstance(self.zone_id, str):
            raise ValueError(f"zone_id must be a string or None. Got: {self.zone_id!r}")

        # 8. visible validation
        if not isinstance(self.visible, bool):
            raise ValueError(f"visible flag must be a boolean. Got: {self.visible!r}")

        # 9. uncertainty validation
        if self.uncertainty is not None:
            if not isinstance(self.uncertainty, str) or not self.uncertainty.strip():
                raise ValueError(f"uncertainty must be a non-empty string or None. Got: {self.uncertainty!r}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        data = {
            "gt_id": self.gt_id,
            "frame_idx": self.frame_idx,
            "timestamp_sec": self.timestamp_sec,
            "image_bbox": list(self.image_bbox) if self.image_bbox is not None else None,
            "image_x": self.image_x,
            "image_y": self.image_y,
            "floor_x": self.floor_x,
            "floor_y": self.floor_y,
            "zone_id": self.zone_id,
            "visible": self.visible,
        }
        if self.uncertainty is not None:
            data["uncertainty"] = self.uncertainty
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruthPoint":
        """Reconstruct GroundTruthPoint from dictionary."""
        return cls(
            gt_id=data["gt_id"],
            frame_idx=data["frame_idx"],
            timestamp_sec=data["timestamp_sec"],
            image_bbox=data.get("image_bbox"),
            image_x=data.get("image_x"),
            image_y=data.get("image_y"),
            floor_x=data.get("floor_x"),
            floor_y=data.get("floor_y"),
            zone_id=data.get("zone_id"),
            visible=data.get("visible", True),
            uncertainty=data.get("uncertainty"),
        )


@dataclass
class GroundTruthEvent:
    """
    Represents an externally or manually annotated temporal event (entry, exit, transition).

    Required fields:
        event_type: One of ('entry', 'exit', 'transition').
        gt_id: Canonical ground-truth identity string.
        frame_idx: Frame index where event took place.
        timestamp_sec: Video timestamp in seconds.

    Optional fields:
        from_zone_id: Origin zone string (or None).
        to_zone_id: Destination zone string (or None).
    """
    event_type: str
    gt_id: str
    frame_idx: int
    timestamp_sec: float
    from_zone_id: Optional[str] = None
    to_zone_id: Optional[str] = None

    def __post_init__(self):
        # 1. gt_id validation
        if not isinstance(self.gt_id, str) or not self.gt_id.strip():
            raise ValueError(f"gt_id must be a non-empty string. Got: {self.gt_id!r}")

        # 2. frame_idx validation
        if isinstance(self.frame_idx, bool) or not isinstance(self.frame_idx, int) or self.frame_idx < 0:
            raise ValueError(f"frame_idx must be a non-negative integer. Got: {self.frame_idx!r}")

        # 3. timestamp_sec validation
        if not _is_finite_number(self.timestamp_sec) or self.timestamp_sec < 0.0:
            raise ValueError(f"timestamp_sec must be a non-negative finite number. Got: {self.timestamp_sec!r}")
        self.timestamp_sec = float(self.timestamp_sec)

        # 4. event_type validation
        if self.event_type not in SUPPORTED_EVENT_TYPES:
            raise ValueError(
                f"event_type must be one of {SUPPORTED_EVENT_TYPES}. Got: {self.event_type!r}"
            )

        # 5. Coherent zone fields per event type
        if self.event_type == "entry":
            if not isinstance(self.to_zone_id, str) or not self.to_zone_id.strip():
                raise ValueError("An 'entry' event requires a non-empty 'to_zone_id'.")
            if self.from_zone_id is not None and self.from_zone_id == self.to_zone_id:
                raise ValueError("An 'entry' event cannot have identical from_zone_id and to_zone_id.")

        elif self.event_type == "exit":
            if not isinstance(self.from_zone_id, str) or not self.from_zone_id.strip():
                raise ValueError("An 'exit' event requires a non-empty 'from_zone_id'.")
            if self.to_zone_id is not None and self.from_zone_id == self.to_zone_id:
                raise ValueError("An 'exit' event cannot have identical from_zone_id and to_zone_id.")

        elif self.event_type == "transition":
            if not isinstance(self.from_zone_id, str) or not self.from_zone_id.strip():
                raise ValueError("A 'transition' event requires a non-empty 'from_zone_id'.")
            if not isinstance(self.to_zone_id, str) or not self.to_zone_id.strip():
                raise ValueError("A 'transition' event requires a non-empty 'to_zone_id'.")
            if self.from_zone_id == self.to_zone_id:
                raise ValueError(
                    f"A 'transition' event requires different from_zone_id and to_zone_id. "
                    f"Got {self.from_zone_id!r} == {self.to_zone_id!r}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "event_type": self.event_type,
            "gt_id": self.gt_id,
            "frame_idx": self.frame_idx,
            "timestamp_sec": self.timestamp_sec,
            "from_zone_id": self.from_zone_id,
            "to_zone_id": self.to_zone_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruthEvent":
        """Reconstruct GroundTruthEvent from dictionary."""
        return cls(
            event_type=data["event_type"],
            gt_id=data["gt_id"],
            frame_idx=data["frame_idx"],
            timestamp_sec=data["timestamp_sec"],
            from_zone_id=data.get("from_zone_id"),
            to_zone_id=data.get("to_zone_id"),
        )


@dataclass
class GroundTruthCalibrationPoint:
    """
    Represents an image-to-floor correspondence point used for homography.

    Required fields:
        label: Identifier string for the reference point (e.g. 'P1', 'CP1').
        image_x: Image-space X coordinate.
        image_y: Image-space Y coordinate.
        floor_x: Metric or grid floor-space X coordinate.
        floor_y: Metric or grid floor-space Y coordinate.
        role: Functional role, either 'training' (fit) or 'held_out' (independent check).
    """
    label: str
    image_x: float
    image_y: float
    floor_x: float
    floor_y: float
    role: str = "training"

    def __post_init__(self):
        # 1. label validation
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError(f"label must be a non-empty string. Got: {self.label!r}")

        # 2. coordinate validation
        for name, val in [
            ("image_x", self.image_x),
            ("image_y", self.image_y),
            ("floor_x", self.floor_x),
            ("floor_y", self.floor_y),
        ]:
            if not _is_finite_number(val):
                raise ValueError(f"{name} must be a finite number. Got: {val!r}")

        self.image_x = float(self.image_x)
        self.image_y = float(self.image_y)
        self.floor_x = float(self.floor_x)
        self.floor_y = float(self.floor_y)

        # 3. role validation
        if self.role not in SUPPORTED_CALIBRATION_ROLES:
            raise ValueError(
                f"role must be one of {SUPPORTED_CALIBRATION_ROLES}. Got: {self.role!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "label": self.label,
            "image_x": self.image_x,
            "image_y": self.image_y,
            "floor_x": self.floor_x,
            "floor_y": self.floor_y,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruthCalibrationPoint":
        """Reconstruct GroundTruthCalibrationPoint from dictionary."""
        return cls(
            label=data["label"],
            image_x=data["image_x"],
            image_y=data["image_y"],
            floor_x=data["floor_x"],
            floor_y=data["floor_y"],
            role=data.get("role", "training"),
        )


@dataclass
class GroundTruthSequence:
    """
    Represents a complete, machine-readable ground-truth annotated sequence.

    Required fields:
        schema_version: Semantic schema version string (e.g. '1.0').
        sequence_id: Unique sequence identifier string.
        source: Description of annotation source (e.g. 'controlled_recording', 'synthetic').
        fps: Sampling or recording frame rate in frames per second.

    Contained collections:
        points: Ordered list of GroundTruthPoint records.
        events: List of GroundTruthEvent records.
        calibration_points: List of GroundTruthCalibrationPoint records.
        metadata: Optional dictionary with supplementary sequence metadata.
    """
    schema_version: str
    sequence_id: str
    source: str
    fps: float
    points: List[GroundTruthPoint] = field(default_factory=list)
    events: List[GroundTruthEvent] = field(default_factory=list)
    calibration_points: List[GroundTruthCalibrationPoint] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # 1. String fields validation
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError(f"schema_version must be a non-empty string. Got: {self.schema_version!r}")
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError(f"sequence_id must be a non-empty string. Got: {self.sequence_id!r}")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError(f"source must be a non-empty string. Got: {self.source!r}")

        # 2. fps validation
        if not _is_finite_number(self.fps) or self.fps <= 0.0:
            raise ValueError(f"fps must be a positive finite number. Got: {self.fps!r}")
        self.fps = float(self.fps)

        # 3. Contained collection types validation
        if not isinstance(self.points, list):
            raise ValueError(f"points must be a list. Got: {type(self.points).__name__}")
        if not isinstance(self.events, list):
            raise ValueError(f"events must be a list. Got: {type(self.events).__name__}")
        if not isinstance(self.calibration_points, list):
            raise ValueError(f"calibration_points must be a list. Got: {type(self.calibration_points).__name__}")

        # 4. Calibration point uniqueness validation
        seen_calib_labels = set()
        for cp in self.calibration_points:
            if not isinstance(cp, GroundTruthCalibrationPoint):
                raise TypeError(f"Expected GroundTruthCalibrationPoint instance, got: {type(cp).__name__}")
            if cp.label in seen_calib_labels:
                raise ValueError(f"Duplicate calibration point label detected: '{cp.label}'")
            seen_calib_labels.add(cp.label)

        # 5. Point collection integrity & temporal consistency validation
        # Tracks last observed (frame_idx, timestamp_sec) per gt_id
        last_observation_by_gt: Dict[str, Tuple[int, float]] = {}
        seen_observations = set()

        for pt in self.points:
            if not isinstance(pt, GroundTruthPoint):
                raise TypeError(f"Expected GroundTruthPoint instance, got: {type(pt).__name__}")

            # Check for duplicate observation of same gt_id at same frame
            obs_key = (pt.gt_id, pt.frame_idx)
            if obs_key in seen_observations:
                raise ValueError(
                    f"Duplicate observation detected for gt_id '{pt.gt_id}' at frame_idx {pt.frame_idx}."
                )
            seen_observations.add(obs_key)

            # Check chronological consistency for this gt_id
            if pt.gt_id in last_observation_by_gt:
                prev_frame, prev_time = last_observation_by_gt[pt.gt_id]
                if pt.frame_idx < prev_frame:
                    raise ValueError(
                        f"Temporal frame ordering violation for gt_id '{pt.gt_id}': "
                        f"frame {pt.frame_idx} appears after frame {prev_frame}."
                    )
                if pt.timestamp_sec < prev_time:
                    raise ValueError(
                        f"Temporal timestamp ordering violation for gt_id '{pt.gt_id}': "
                        f"timestamp {pt.timestamp_sec}s appears after {prev_time}s."
                    )
            last_observation_by_gt[pt.gt_id] = (pt.frame_idx, pt.timestamp_sec)

        # 6. Event collection integrity
        for ev in self.events:
            if not isinstance(ev, GroundTruthEvent):
                raise TypeError(f"Expected GroundTruthEvent instance, got: {type(ev).__name__}")

        # 7. Metadata validation
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError(f"metadata must be a dictionary or None. Got: {type(self.metadata).__name__}")

    def get_gt_ids(self) -> List[str]:
        """Return sorted unique ground-truth IDs present in points."""
        return sorted(list({p.gt_id for p in self.points}))

    def get_points_for_gt_id(self, gt_id: str) -> List[GroundTruthPoint]:
        """Filter points for a specific ground-truth ID."""
        return [p for p in self.points if p.gt_id == gt_id]

    def get_events_for_gt_id(self, gt_id: str) -> List[GroundTruthEvent]:
        """Filter events for a specific ground-truth ID."""
        return [e for e in self.events if e.gt_id == gt_id]

    def get_training_calibration_points(self) -> List[GroundTruthCalibrationPoint]:
        """Return calibration points designated for model fitting."""
        return [cp for cp in self.calibration_points if cp.role == "training"]

    def get_held_out_calibration_points(self) -> List[GroundTruthCalibrationPoint]:
        """Return calibration points designated for independent evaluation."""
        return [cp for cp in self.calibration_points if cp.role == "held_out"]

    def get_empty_frame_indices(self) -> List[int]:
        """Return sorted list of frame indices that were explicitly reviewed as empty (0 persons)."""
        if self.metadata and "empty_frame_indices" in self.metadata:
            return sorted(list(self.metadata["empty_frame_indices"]))
        return []

    def get_evaluated_frame_indices(self) -> List[int]:
        """
        Return sorted list of all frame indices that were evaluated/reviewed.
        Includes both frames with person observations and frames reviewed as empty.
        """
        if self.metadata and "evaluated_frame_indices" in self.metadata:
            return sorted(list(self.metadata["evaluated_frame_indices"]))
        # Fallback: union of points frames and empty frames
        point_frames = {p.frame_idx for p in self.points}
        empty_frames = set(self.get_empty_frame_indices())
        return sorted(list(point_frames | empty_frames))

    def is_frame_evaluated(self, frame_idx: int) -> bool:
        """Check if a specific frame index was evaluated/reviewed."""
        return frame_idx in set(self.get_evaluated_frame_indices())

    def is_frame_empty(self, frame_idx: int) -> bool:
        """Check if a specific frame index was explicitly reviewed as empty (0 persons)."""
        return frame_idx in set(self.get_empty_frame_indices())

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire sequence to a deterministic JSON-serializable dictionary."""
        data = {
            "schema_version": self.schema_version,
            "sequence_id": self.sequence_id,
            "source": self.source,
            "fps": self.fps,
            "calibration_points": [cp.to_dict() for cp in self.calibration_points],
            "points": [p.to_dict() for p in self.points],
            "events": [e.to_dict() for e in self.events],
        }
        if self.metadata is not None:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruthSequence":
        """Reconstruct GroundTruthSequence from dictionary."""
        calibration_points = [
            GroundTruthCalibrationPoint.from_dict(cp_data)
            for cp_data in data.get("calibration_points", [])
        ]
        points = [
            GroundTruthPoint.from_dict(p_data)
            for p_data in data.get("points", [])
        ]
        events = [
            GroundTruthEvent.from_dict(e_data)
            for e_data in data.get("events", [])
        ]
        return cls(
            schema_version=data["schema_version"],
            sequence_id=data["sequence_id"],
            source=data["source"],
            fps=data["fps"],
            calibration_points=calibration_points,
            points=points,
            events=events,
            metadata=data.get("metadata"),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize sequence to JSON formatted string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "GroundTruthSequence":
        """Reconstruct sequence from JSON formatted string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save_to_json(self, file_path: str, indent: int = 2) -> None:
        """Save sequence to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=indent))

    @classmethod
    def load_from_json(cls, file_path: str) -> "GroundTruthSequence":
        """Load and reconstruct sequence from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
