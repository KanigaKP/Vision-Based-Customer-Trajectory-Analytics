"""
Vision-Based Customer Trajectory Analytics — Zone Visit Sequences & Direct Transitions.
========================================================================================
Phase 11.1: Extracts explainable, descriptive summaries of observed zone visit sequences
and direct zone-to-zone transitions from existing track trajectories.

METHODOLOGICAL & ARCHITECTURAL INVARIANTS:
1. Pure Analytics Layer:
   - Consumes pre-assigned TrajectoryPoint.zone_id observations (from Phase 6/7).
   - Never reruns detection, tracking, homography, Point-in-Polygon (PIP), or hysteresis filtering.
2. Clip-Local Track Identity:
   - Operates strictly on session-local track_id integers.
   - Never merges tracks, bridges identities, or infers person re-identification (Re-ID).
3. Descriptive Measurement of Observed Labels (No Behavioral Interpretation):
   - Measures strictly observed sequence labels and direct transition frequencies.
   - No assumptions or labels regarding customer intent, engagement, preference, interest,
     or behavioral classification (e.g., no "passing", "lingering", "wandering", or "browsing").
   - No duration thresholds for visit inclusion.
   - No recommendations, rankings, or causal assertions regarding store layout.
4. Observed Visit Run Semantics:
   - Consecutive observations with the same zone label represent one run in the observed sequence.
   - For a single-observation run, duration_sec is 0.0s.
   - Durations measure observed in-zone temporal spans (exit_time - entry_time), not continuous
     physically verified presence.
5. Strict Direct Transition Semantics:
   - A directed zone transition (A -> B) is counted ONLY when consecutive non-None zone runs
     directly change from one defined zone to another (A != B).
   - An unassigned or outside observation (zone_id = None) immediately breaks the transition chain.
   - Intervening outside observations are NEVER bridged: A -> None -> B yields ZERO direct transitions.
   - Transitions are never inferred before the initial observation or after the final observation
     of a track trajectory.
6. Rigorous Temporal & Schema Validation:
   - Rejects non-monotonic frame indices or timestamps with explicit ValueError exceptions.
   - Rejects non-finite (NaN, Inf), negative, or malformed values.
   - Leaves caller Trajectory objects completely unmodified.
"""

from dataclasses import dataclass, asdict, field
import math
from typing import List, Dict, Optional, Tuple, Any, Sequence, Union
from collections import defaultdict

from src.trajectory.trajectory import Trajectory, TrajectoryPoint


def _is_finite_number(val: Any) -> bool:
    """Check if value is a finite int or float (excluding booleans)."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(val)


def _validate_trajectory_point(pt: TrajectoryPoint) -> None:
    """Validate a single TrajectoryPoint for temporal and identifier integrity."""
    if not isinstance(pt.track_id, int) or isinstance(pt.track_id, bool):
        raise ValueError(
            f"track_id must be an integer, got {type(pt.track_id).__name__}: {pt.track_id!r}"
        )
    if pt.track_id < 0:
        raise ValueError(f"track_id must be non-negative, got: {pt.track_id}")

    if not isinstance(pt.frame_idx, int) or isinstance(pt.frame_idx, bool):
        raise ValueError(
            f"frame_idx must be an integer, got {type(pt.frame_idx).__name__}: {pt.frame_idx!r}"
        )
    if pt.frame_idx < 0:
        raise ValueError(f"frame_idx must be non-negative, got: {pt.frame_idx}")

    if not _is_finite_number(pt.timestamp_sec):
        raise ValueError(
            f"timestamp_sec must be a finite number, got {type(pt.timestamp_sec).__name__}: {pt.timestamp_sec!r}"
        )
    if pt.timestamp_sec < 0.0:
        raise ValueError(f"timestamp_sec must be non-negative, got: {pt.timestamp_sec}")

    if pt.zone_id is not None:
        if not isinstance(pt.zone_id, str):
            raise ValueError(
                f"zone_id must be a string or None, got {type(pt.zone_id).__name__}: {pt.zone_id!r}"
            )
        if not pt.zone_id.strip():
            raise ValueError("zone_id cannot be an empty or whitespace string.")


def _validate_trajectory(trajectory: Trajectory) -> None:
    """Validate full trajectory temporal ordering and point consistency."""
    if not isinstance(trajectory, Trajectory):
        raise TypeError(f"Expected Trajectory instance, got {type(trajectory).__name__}")

    if not isinstance(trajectory.track_id, int) or isinstance(trajectory.track_id, bool):
        raise ValueError(f"trajectory.track_id must be an integer, got {type(trajectory.track_id).__name__}")
    if trajectory.track_id < 0:
        raise ValueError(f"trajectory.track_id must be non-negative, got: {trajectory.track_id}")

    prev_frame: Optional[int] = None
    prev_time: Optional[float] = None

    for idx, pt in enumerate(trajectory.points):
        if not isinstance(pt, TrajectoryPoint):
            raise TypeError(f"Expected TrajectoryPoint at index {idx}, got {type(pt).__name__}")
        if pt.track_id != trajectory.track_id:
            raise ValueError(
                f"Point track_id ({pt.track_id}) does not match trajectory track_id ({trajectory.track_id}) at index {idx}"
            )
        _validate_trajectory_point(pt)

        if prev_frame is not None and pt.frame_idx < prev_frame:
            raise ValueError(
                f"Non-monotonic frame index at point {idx}: {pt.frame_idx} < previous frame {prev_frame}"
            )
        if prev_time is not None and pt.timestamp_sec < prev_time:
            raise ValueError(
                f"Non-monotonic timestamp at point {idx}: {pt.timestamp_sec} < previous timestamp {prev_time}"
            )

        prev_frame = pt.frame_idx
        prev_time = pt.timestamp_sec


@dataclass
class ZoneVisitRun:
    """
    Represents a maximal contiguous run of trajectory observations in a single non-None zone
    for a single clip-local track.

    Attributes:
        track_id: Clip-local tracker identifier integer (non-negative).
        zone_id: Spatial retail zone identifier string.
        entry_frame_idx: Frame index of the initial observation in this run.
        exit_frame_idx: Frame index of the final observation in this run.
        entry_timestamp_sec: Video timestamp in seconds at initial observation.
        exit_timestamp_sec: Video timestamp in seconds at final observation.
        observation_count: Total sampled trajectory observations during this run.
    """
    track_id: int
    zone_id: str
    entry_frame_idx: int
    exit_frame_idx: int
    entry_timestamp_sec: float
    exit_timestamp_sec: float
    observation_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, int) or isinstance(self.track_id, bool) or self.track_id < 0:
            raise ValueError(f"Invalid track_id: {self.track_id!r}")
        if not isinstance(self.zone_id, str) or not self.zone_id.strip():
            raise ValueError(f"Invalid zone_id: {self.zone_id!r}")
        if not isinstance(self.entry_frame_idx, int) or isinstance(self.entry_frame_idx, bool) or self.entry_frame_idx < 0:
            raise ValueError(f"Invalid entry_frame_idx: {self.entry_frame_idx!r}")
        if not isinstance(self.exit_frame_idx, int) or isinstance(self.exit_frame_idx, bool) or self.exit_frame_idx < self.entry_frame_idx:
            raise ValueError(
                f"exit_frame_idx ({self.exit_frame_idx}) cannot precede entry_frame_idx ({self.entry_frame_idx})"
            )
        if not _is_finite_number(self.entry_timestamp_sec) or self.entry_timestamp_sec < 0.0:
            raise ValueError(f"Invalid entry_timestamp_sec: {self.entry_timestamp_sec!r}")
        if not _is_finite_number(self.exit_timestamp_sec) or self.exit_timestamp_sec < self.entry_timestamp_sec:
            raise ValueError(
                f"exit_timestamp_sec ({self.exit_timestamp_sec}) cannot precede entry_timestamp_sec ({self.entry_timestamp_sec})"
            )
        if not isinstance(self.observation_count, int) or isinstance(self.observation_count, bool) or self.observation_count < 1:
            raise ValueError(f"observation_count must be at least 1, got: {self.observation_count!r}")

    @property
    def duration_sec(self) -> float:
        """
        Observed duration span in seconds between initial and final observation.
        Note: For a single observation, duration is 0.0s. This is an observed span,
        not physically verified continuous occupancy.
        """
        return round(max(0.0, self.exit_timestamp_sec - self.entry_timestamp_sec), 3)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration_sec"] = self.duration_sec
        return d


@dataclass
class ObservedTrackSequence:
    """
    Chronological sequence of observed zone visit runs and direct zone transitions
    for a single clip-local track.

    Attributes:
        track_id: Clip-local tracker identifier integer.
        runs: Chronological list of ZoneVisitRun instances for this track.
        zone_sequence: Ordered sequence of zone IDs for each observed run.
        direct_transitions: Chronological list of (from_zone, to_zone) pairs observed
            directly without any intervening unassigned (None) observations.
    """
    track_id: int
    runs: List[ZoneVisitRun]
    zone_sequence: List[str]
    direct_transitions: List[Tuple[str, str]]

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, int) or isinstance(self.track_id, bool) or self.track_id < 0:
            raise ValueError(f"Invalid track_id: {self.track_id!r}")
        for r in self.runs:
            if not isinstance(r, ZoneVisitRun):
                raise TypeError(f"Expected ZoneVisitRun in runs, got {type(r).__name__}")
            if r.track_id != self.track_id:
                raise ValueError(f"Run track_id ({r.track_id}) does not match sequence track_id ({self.track_id})")
        for trans in self.direct_transitions:
            if not isinstance(trans, tuple) or len(trans) != 2:
                raise ValueError(f"Invalid transition tuple: {trans!r}")
            if not isinstance(trans[0], str) or not isinstance(trans[1], str):
                raise ValueError(f"Transition zones must be strings, got: {trans!r}")
            if trans[0] == trans[1]:
                raise ValueError(f"Direct transition must be between distinct zones, got: {trans!r}")

    @property
    def total_runs(self) -> int:
        return len(self.runs)

    @property
    def total_transitions(self) -> int:
        return len(self.direct_transitions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "runs": [r.to_dict() for r in self.runs],
            "zone_sequence": list(self.zone_sequence),
            "direct_transitions": [{"from_zone": f, "to_zone": t} for f, t in self.direct_transitions],
            "total_runs": self.total_runs,
            "total_transitions": self.total_transitions,
        }


@dataclass
class ZoneActivitySummary:
    """
    Descriptive activity summary for a single spatial zone derived from observed runs
    and direct transitions.

    IMPORTANT METHODOLOGICAL BOUNDARY:
        distinct_track_count represents the count of distinct clip-local tracker IDs observed
        within this zone. Tracker IDs are temporary session-local integers assigned by ByteTrack;
        they do NOT represent unique physical customers, individuals, or persistent shopper identities.
        Due to occlusions, re-entries, and tracker fragmentation, one physical person may generate
        multiple track IDs, or a track ID may be terminated prematurely.

    Attributes:
        zone_id: Spatial retail zone identifier string.
        observed_run_count: Total count of contiguous observed visit runs in this zone (integer >= 0).
        distinct_track_count: Count of distinct clip-local tracker IDs with at least one observed
            run in this zone (integer >= 0).
        direct_incoming_count: Count of direct transitions entering this zone directly from any
            other defined zone without intervening unassigned (None) observations (integer >= 0).
        direct_outgoing_count: Count of direct transitions leaving this zone directly to any other
            defined zone without intervening unassigned (None) observations (integer >= 0).
    """
    zone_id: str
    observed_run_count: int
    distinct_track_count: int
    direct_incoming_count: int
    direct_outgoing_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.zone_id, str) or not self.zone_id.strip():
            raise ValueError(f"Invalid zone_id: {self.zone_id!r}")
        for field_name, val in [
            ("observed_run_count", self.observed_run_count),
            ("distinct_track_count", self.distinct_track_count),
            ("direct_incoming_count", self.direct_incoming_count),
            ("direct_outgoing_count", self.direct_outgoing_count),
        ]:
            if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                raise ValueError(f"{field_name} must be a non-negative integer, got: {val!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DirectTransitionSummary:
    """
    Aggregated descriptive summary of direct zone-to-zone transitions and per-zone activity.

    Attributes:
        total_tracks: Total number of tracks evaluated.
        total_transitions: Total count of direct transitions observed.
        known_zone_ids: Configured or discovered zone identifiers in deterministic order.
        transition_counts: Mapping from (from_zone, to_zone) tuple to observed count.
        transition_matrix: Nested mapping from from_zone -> {to_zone: count}.
        track_sequences: Per-track sequence records.
        zone_summaries: Per-zone descriptive activity summaries (observed_run_count,
            distinct_track_count, direct_incoming_count, direct_outgoing_count).
    """
    total_tracks: int
    total_transitions: int
    known_zone_ids: List[str]
    transition_counts: Dict[Tuple[str, str], int]
    transition_matrix: Dict[str, Dict[str, int]]
    track_sequences: List[ObservedTrackSequence]
    zone_summaries: Dict[str, ZoneActivitySummary] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tracks": self.total_tracks,
            "total_transitions": self.total_transitions,
            "known_zone_ids": list(self.known_zone_ids),
            "transition_counts": {
                f"{k[0]}->{k[1]}": v for k, v in sorted(self.transition_counts.items())
            },
            "transition_matrix": {
                z: dict(cols) for z, cols in sorted(self.transition_matrix.items())
            },
            "zone_summaries": {
                z: zs.to_dict() for z, zs in sorted(self.zone_summaries.items())
            },
            "track_sequences": [s.to_dict() for s in self.track_sequences],
        }



class TransitionAnalyzer:
    """
    Analyzes pre-evaluated trajectory observations to extract descriptive
    zone visit sequences and direct zone-to-zone transitions.
    """

    def __init__(self, known_zone_ids: Optional[Sequence[str]] = None) -> None:
        """
        Initialize TransitionAnalyzer.

        Args:
            known_zone_ids: Optional collection of expected zone identifiers.
                If provided, output matrices and summaries include these zones even
                if no transitions are observed for them.
        """
        if known_zone_ids is not None:
            if isinstance(known_zone_ids, (str, bytes)):
                raise TypeError("known_zone_ids must be a sequence of strings, not a single string")
            cleaned: List[str] = []
            for z in known_zone_ids:
                if not isinstance(z, str) or not z.strip():
                    raise ValueError(f"Invalid zone identifier in known_zone_ids: {z!r}")
                if z not in cleaned:
                    cleaned.append(z)
            self.known_zone_ids = cleaned
        else:
            self.known_zone_ids = []

    def analyze_trajectory(self, trajectory: Trajectory) -> ObservedTrackSequence:
        """
        Extract observed zone visit runs and direct zone-to-zone transitions
        from a single zone-evaluated trajectory.

        Args:
            trajectory: Trajectory with populated TrajectoryPoint.zone_id observations.

        Returns:
            ObservedTrackSequence for this track.
        """
        _validate_trajectory(trajectory)

        runs: List[ZoneVisitRun] = []
        direct_transitions: List[Tuple[str, str]] = []

        if not trajectory.points:
            return ObservedTrackSequence(
                track_id=trajectory.track_id,
                runs=[],
                zone_sequence=[],
                direct_transitions=[],
            )

        current_zone: Optional[str] = None
        current_run_pts: List[TrajectoryPoint] = []

        for pt in trajectory.points:
            pt_zone = pt.zone_id

            if pt_zone == current_zone:
                # Same zone continuation
                if current_zone is not None:
                    current_run_pts.append(pt)
            else:
                # Zone boundary change detected
                # 1. Close current run if exiting a defined zone
                if current_zone is not None and current_run_pts:
                    runs.append(
                        ZoneVisitRun(
                            track_id=trajectory.track_id,
                            zone_id=current_zone,
                            entry_frame_idx=current_run_pts[0].frame_idx,
                            exit_frame_idx=current_run_pts[-1].frame_idx,
                            entry_timestamp_sec=current_run_pts[0].timestamp_sec,
                            exit_timestamp_sec=current_run_pts[-1].timestamp_sec,
                            observation_count=len(current_run_pts),
                        )
                    )
                    # 2. Record direct zone-to-zone transition ONLY if transitioning directly
                    #    to another defined non-None zone without intervening outside observations.
                    if pt_zone is not None and pt_zone != current_zone:
                        direct_transitions.append((current_zone, pt_zone))

                    current_run_pts = []

                # 3. Start new run if entering a defined zone
                if pt_zone is not None:
                    current_run_pts = [pt]

                current_zone = pt_zone

        # Close final active run if trajectory ended inside a defined zone
        if current_zone is not None and current_run_pts:
            runs.append(
                ZoneVisitRun(
                    track_id=trajectory.track_id,
                    zone_id=current_zone,
                    entry_frame_idx=current_run_pts[0].frame_idx,
                    exit_frame_idx=current_run_pts[-1].frame_idx,
                    entry_timestamp_sec=current_run_pts[0].timestamp_sec,
                    exit_timestamp_sec=current_run_pts[-1].timestamp_sec,
                    observation_count=len(current_run_pts),
                )
            )

        zone_sequence = [r.zone_id for r in runs]

        return ObservedTrackSequence(
            track_id=trajectory.track_id,
            runs=runs,
            zone_sequence=zone_sequence,
            direct_transitions=direct_transitions,
        )

    def analyze(self, trajectories: Sequence[Trajectory]) -> DirectTransitionSummary:
        """
        Aggregate observed visit sequences and direct zone transitions across multiple tracks.

        Args:
            trajectories: Collection of zone-evaluated Trajectory instances.

        Returns:
            DirectTransitionSummary containing aggregated transition counts, matrix,
            and per-track sequence records.
        """
        if not isinstance(trajectories, (list, tuple)):
            raise TypeError(
                f"Expected list or tuple of Trajectory instances, got {type(trajectories).__name__}"
            )

        track_sequences: List[ObservedTrackSequence] = []
        transition_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        discovered_zones: List[str] = []

        seen_track_ids = set()
        for idx, traj in enumerate(trajectories):
            if not isinstance(traj, Trajectory):
                raise TypeError(f"Expected Trajectory instance at index {idx}, got {type(traj).__name__}")
            if traj.track_id in seen_track_ids:
                raise ValueError(f"Duplicate track_id {traj.track_id} found in trajectories list at index {idx}")
            seen_track_ids.add(traj.track_id)

            seq = self.analyze_trajectory(traj)
            track_sequences.append(seq)

            for trans in seq.direct_transitions:
                transition_counts[trans] += 1

            for r in seq.runs:
                if r.zone_id not in discovered_zones:
                    discovered_zones.append(r.zone_id)

        # Build full deterministic list of zone identifiers
        all_zones_set = set(self.known_zone_ids) | set(discovered_zones)
        all_zones = sorted(list(all_zones_set))

        # Build square transition matrix
        transition_matrix: Dict[str, Dict[str, int]] = {
            from_z: {to_z: 0 for to_z in all_zones} for from_z in all_zones
        }
        for (from_z, to_z), count in transition_counts.items():
            if from_z in transition_matrix and to_z in transition_matrix[from_z]:
                transition_matrix[from_z][to_z] = count

        # Collect runs per zone
        runs_by_zone: Dict[str, List[ZoneVisitRun]] = defaultdict(list)
        for seq in track_sequences:
            for r in seq.runs:
                runs_by_zone[r.zone_id].append(r)

        # Build per-zone descriptive activity summaries
        zone_summaries: Dict[str, ZoneActivitySummary] = {}
        for zone_id in all_zones:
            zone_runs = runs_by_zone.get(zone_id, [])
            observed_run_count = len(zone_runs)
            distinct_track_count = len(set(r.track_id for r in zone_runs))
            direct_incoming_count = sum(
                transition_matrix[from_z][zone_id]
                for from_z in all_zones
                if from_z != zone_id
            )
            direct_outgoing_count = sum(
                transition_matrix[zone_id][to_z]
                for to_z in all_zones
                if to_z != zone_id
            )
            zone_summaries[zone_id] = ZoneActivitySummary(
                zone_id=zone_id,
                observed_run_count=observed_run_count,
                distinct_track_count=distinct_track_count,
                direct_incoming_count=direct_incoming_count,
                direct_outgoing_count=direct_outgoing_count,
            )

        total_transitions = sum(transition_counts.values())

        return DirectTransitionSummary(
            total_tracks=len(track_sequences),
            total_transitions=total_transitions,
            known_zone_ids=all_zones,
            transition_counts=dict(transition_counts),
            transition_matrix=transition_matrix,
            track_sequences=track_sequences,
            zone_summaries=zone_summaries,
        )


def extract_zone_visit_runs(trajectory: Trajectory) -> List[ZoneVisitRun]:
    """
    Convenience function: Extract chronological ZoneVisitRun instances for a single trajectory.
    """
    analyzer = TransitionAnalyzer()
    return analyzer.analyze_trajectory(trajectory).runs


def extract_track_sequence(trajectory: Trajectory) -> ObservedTrackSequence:
    """
    Convenience function: Extract ObservedTrackSequence for a single trajectory.
    """
    analyzer = TransitionAnalyzer()
    return analyzer.analyze_trajectory(trajectory)


def compute_direct_transitions(
    trajectories: Union[Trajectory, Sequence[Trajectory]],
    known_zone_ids: Optional[Sequence[str]] = None,
) -> DirectTransitionSummary:
    """
    Convenience function: Compute direct zone transitions across one or more trajectories.
    """
    if isinstance(trajectories, Trajectory):
        traj_list = [trajectories]
    elif isinstance(trajectories, (list, tuple)):
        traj_list = list(trajectories)
    else:
        raise TypeError(
            f"Expected Trajectory or sequence of Trajectories, got {type(trajectories).__name__}"
        )

    analyzer = TransitionAnalyzer(known_zone_ids=known_zone_ids)
    return analyzer.analyze(traj_list)


def compute_zone_activity_summaries(
    trajectories_or_summary: Union[Trajectory, Sequence[Trajectory], DirectTransitionSummary],
    known_zone_ids: Optional[Sequence[str]] = None,
) -> Dict[str, ZoneActivitySummary]:
    """
    Convenience function: Compute or extract per-zone activity summaries
    (observed visit-run count, distinct clip-local track count, and direct incoming/outgoing counts).

    IMPORTANT METHODOLOGICAL BOUNDARY:
        distinct_track_count represents the count of distinct clip-local tracker IDs observed
        within a zone. Tracker IDs are temporary session-local integers assigned by ByteTrack;
        they do NOT represent unique physical customers, individuals, or persistent shopper identities.
        Due to occlusions, re-entries, and tracker fragmentation, one physical person may generate
        multiple track IDs, or a track ID may be terminated prematurely.

    Args:
        trajectories_or_summary: A single Trajectory, a sequence of Trajectories,
            or an already-computed DirectTransitionSummary.
        known_zone_ids: Optional sequence of known zone IDs.

    Returns:
        Mapping zone_id -> ZoneActivitySummary for all evaluated zones.
    """
    if isinstance(trajectories_or_summary, DirectTransitionSummary):
        if known_zone_ids is not None:
            if isinstance(known_zone_ids, (str, bytes)):
                raise TypeError("known_zone_ids must be a sequence of strings, not a single string")
            summaries = dict(trajectories_or_summary.zone_summaries)
            for z in known_zone_ids:
                if not isinstance(z, str) or not z.strip():
                    raise ValueError(f"Invalid zone identifier in known_zone_ids: {z!r}")
                if z not in summaries:
                    summaries[z] = ZoneActivitySummary(
                        zone_id=z,
                        observed_run_count=0,
                        distinct_track_count=0,
                        direct_incoming_count=0,
                        direct_outgoing_count=0,
                    )
            return summaries
        return dict(trajectories_or_summary.zone_summaries)

    summary = compute_direct_transitions(trajectories_or_summary, known_zone_ids=known_zone_ids)
    return dict(summary.zone_summaries)

