"""
Vision-Based Customer Trajectory Analytics — Footfall Analytics Module.
======================================================================
Interprets Phase 6 zone-evaluated spatial trajectories to compute:
1. Unique tracks per zone (distinct track IDs visiting a zone)
2. Zone entry events (transitions into a zone)
3. Zone visits (contiguous temporal intervals in a single zone)
4. Basic visit duration (duration between entry and exit timestamps)
5. Per-frame occupancy (number of active tracks in a zone at a specific frame)
6. Zone-to-zone transition matrix (flow transitions between defined zones)

IMPORTANT METHODOLOGICAL BOUNDARIES:
- Input trajectories MUST already have TrajectoryPoint.zone_id assigned.
  No PIP or hysteresis filtering is recomputed here.
- Track IDs are temporary session-local tracker IDs, not customer identities.
- Basic visit duration is an engineering metric between sampled frames,
  NOT physically validated dwell time. Do not claim sub-second accuracy.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import defaultdict

from src.trajectory.trajectory import Trajectory, TrajectoryPoint


@dataclass
class ZoneEntryEvent:
    """
    Represents a discrete transition event into a spatial retail zone.

    Attributes:
        track_id (int): Session-local identifier of the entering track.
        zone_id (str): Identifier of the zone entered.
        from_zone_id (Optional[str]): Prior zone, or None if entering from outside.
        frame_idx (int): Frame index where the entry was first detected.
        timestamp_sec (float): Video timestamp (seconds) at the entry point.
    """
    track_id: int
    zone_id: str
    from_zone_id: Optional[str]
    frame_idx: int
    timestamp_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ZoneVisit:
    """
    Represents a contiguous temporal stay of a single track within one zone.

    Attributes:
        track_id (int): Session-local identifier of the visiting track.
        zone_id (str): Identifier of the visited zone.
        entry_frame_idx (int): Frame index of the initial point in this visit.
        exit_frame_idx (int): Frame index of the final point in this visit.
        entry_timestamp_sec (float): Video timestamp at entry.
        exit_timestamp_sec (float): Video timestamp at exit.
        point_count (int): Total sampled trajectory points during this stay.
    """
    track_id: int
    zone_id: str
    entry_frame_idx: int
    exit_frame_idx: int
    entry_timestamp_sec: float
    exit_timestamp_sec: float
    point_count: int

    @property
    def duration_sec(self) -> float:
        """
        Duration of the visit in seconds.
        Note: If a visit contains a single sampled point, duration is 0.0s.
        """
        return round(max(0.0, self.exit_timestamp_sec - self.entry_timestamp_sec), 3)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration_sec"] = self.duration_sec
        return d


@dataclass
class FootfallSummary:
    """
    Aggregated footfall analytics summary across all evaluated trajectories.
    """
    unique_tracks_per_zone: Dict[str, int]
    entry_events_per_zone: Dict[str, int]
    visits_per_zone: Dict[str, int]
    total_visit_duration_sec: Dict[str, float]
    average_visit_duration_sec: Dict[str, float]
    transition_matrix: Dict[str, Dict[str, int]]
    entries: List[ZoneEntryEvent]
    visits: List[ZoneVisit]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unique_tracks_per_zone": self.unique_tracks_per_zone,
            "entry_events_per_zone": self.entry_events_per_zone,
            "visits_per_zone": self.visits_per_zone,
            "total_visit_duration_sec": self.total_visit_duration_sec,
            "average_visit_duration_sec": self.average_visit_duration_sec,
            "transition_matrix": self.transition_matrix,
            "total_entries": len(self.entries),
            "total_visits": len(self.visits),
            "entries": [e.to_dict() for e in self.entries],
            "visits": [v.to_dict() for v in self.visits],
        }


class FootfallAnalyzer:
    """
    Analyzes zone-assigned trajectories to extract discrete footfall metrics,
    contiguous visits, durations, occupancies, and zone-to-zone transitions.
    """

    def __init__(self, known_zone_ids: Optional[List[str]] = None):
        """
        Initialize FootfallAnalyzer.

        Args:
            known_zone_ids (Optional[List[str]]): List of expected zone identifiers.
                If provided, metrics will include zero-counts for zones with no activity.
        """
        self.known_zone_ids = list(known_zone_ids) if known_zone_ids else []

    def analyze_trajectory(
        self,
        trajectory: Trajectory
    ) -> Tuple[List[ZoneEntryEvent], List[ZoneVisit], List[Tuple[str, str]]]:
        """
        Extract entry events, contiguous visits, and direct zone transitions
        from a single zone-evaluated trajectory.

        Args:
            trajectory (Trajectory): Trajectory with populated TrajectoryPoint.zone_id.

        Returns:
            Tuple: (entry_events, visits, transitions)
                - entry_events: List[ZoneEntryEvent]
                - visits: List[ZoneVisit]
                - transitions: List[Tuple[from_zone, to_zone]] between non-None zones
        """
        entries: List[ZoneEntryEvent] = []
        visits: List[ZoneVisit] = []
        transitions: List[Tuple[str, str]] = []

        if not trajectory.points:
            return entries, visits, transitions

        current_zone: Optional[str] = None
        current_visit_pts: List[TrajectoryPoint] = []

        for pt in trajectory.points:
            pt_zone = pt.zone_id

            if pt_zone == current_zone:
                # Same zone continuation
                if current_zone is not None:
                    current_visit_pts.append(pt)
            else:
                # Transition detected
                # 1. Close active visit if exiting a valid zone
                if current_zone is not None and current_visit_pts:
                    visits.append(ZoneVisit(
                        track_id=trajectory.track_id,
                        zone_id=current_zone,
                        entry_frame_idx=current_visit_pts[0].frame_idx,
                        exit_frame_idx=current_visit_pts[-1].frame_idx,
                        entry_timestamp_sec=current_visit_pts[0].timestamp_sec,
                        exit_timestamp_sec=current_visit_pts[-1].timestamp_sec,
                        point_count=len(current_visit_pts)
                    ))
                    current_visit_pts = []

                # 2. Record zone-to-zone transition if moving between two valid zones
                if current_zone is not None and pt_zone is not None:
                    transitions.append((current_zone, pt_zone))

                # 3. If entering a valid zone, record entry event and start new visit
                if pt_zone is not None:
                    entries.append(ZoneEntryEvent(
                        track_id=trajectory.track_id,
                        zone_id=pt_zone,
                        from_zone_id=current_zone,
                        frame_idx=pt.frame_idx,
                        timestamp_sec=pt.timestamp_sec
                    ))
                    current_visit_pts = [pt]

                current_zone = pt_zone

        # Close final active visit if trajectory ended inside a zone
        if current_zone is not None and current_visit_pts:
            visits.append(ZoneVisit(
                track_id=trajectory.track_id,
                zone_id=current_zone,
                entry_frame_idx=current_visit_pts[0].frame_idx,
                exit_frame_idx=current_visit_pts[-1].frame_idx,
                entry_timestamp_sec=current_visit_pts[0].timestamp_sec,
                exit_timestamp_sec=current_visit_pts[-1].timestamp_sec,
                point_count=len(current_visit_pts)
            ))

        return entries, visits, transitions

    def analyze(self, trajectories: List[Trajectory]) -> FootfallSummary:
        """
        Aggregate footfall analytics across a collection of trajectories.

        Args:
            trajectories (List[Trajectory]): Evaluated trajectories.

        Returns:
            FootfallSummary: Aggregated footfall results.
        """
        all_entries: List[ZoneEntryEvent] = []
        all_visits: List[ZoneVisit] = []
        all_transitions: List[Tuple[str, str]] = []

        # Track IDs per zone for unique count
        zone_tracks: Dict[str, Set[int]] = defaultdict(set)

        for traj in trajectories:
            entries, visits, trans = self.analyze_trajectory(traj)
            all_entries.extend(entries)
            all_visits.extend(visits)
            all_transitions.extend(trans)

            for v in visits:
                zone_tracks[v.zone_id].add(traj.track_id)

        # Collect all active zones (known zones + any observed zones)
        active_zones = set(self.known_zone_ids)
        for e in all_entries:
            active_zones.add(e.zone_id)
        for v in all_visits:
            active_zones.add(v.zone_id)

        sorted_zones = sorted(list(active_zones))

        # Compute metric aggregates
        unique_tracks: Dict[str, int] = {z: len(zone_tracks[z]) for z in sorted_zones}
        
        entry_counts: Dict[str, int] = {z: 0 for z in sorted_zones}
        for e in all_entries:
            entry_counts[e.zone_id] += 1

        visit_counts: Dict[str, int] = {z: 0 for z in sorted_zones}
        duration_sums: Dict[str, float] = {z: 0.0 for z in sorted_zones}
        for v in all_visits:
            visit_counts[v.zone_id] += 1
            duration_sums[v.zone_id] += v.duration_sec

        duration_sums = {z: round(val, 3) for z, val in duration_sums.items()}

        avg_durations: Dict[str, float] = {}
        for z in sorted_zones:
            count = visit_counts[z]
            avg_durations[z] = round(duration_sums[z] / count, 3) if count > 0 else 0.0

        # Build transition matrix
        # Initialise transition matrix with zeros for all zone pairs
        transition_matrix: Dict[str, Dict[str, int]] = {
            z_from: {z_to: 0 for z_to in sorted_zones}
            for z_from in sorted_zones
        }
        for z_from, z_to in all_transitions:
            if z_from in transition_matrix and z_to in transition_matrix[z_from]:
                transition_matrix[z_from][z_to] += 1

        return FootfallSummary(
            unique_tracks_per_zone=unique_tracks,
            entry_events_per_zone=entry_counts,
            visits_per_zone=visit_counts,
            total_visit_duration_sec=duration_sums,
            average_visit_duration_sec=avg_durations,
            transition_matrix=transition_matrix,
            entries=all_entries,
            visits=all_visits,
        )

    def get_occupancy_snapshot(
        self,
        trajectories: List[Trajectory],
        frame_idx: int,
        known_zone_ids: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Compute zone occupancy at a specific frame index.

        Args:
            trajectories (List[Trajectory]): List of evaluated trajectories.
            frame_idx (int): The query frame index.
            known_zone_ids (Optional[List[str]]): Optional zone IDs override.
                Defaults to self.known_zone_ids if None.

        Returns:
            Dict[str, int]: Mapping from zone_id to count of present tracks at frame_idx.
        """
        zones = known_zone_ids if known_zone_ids is not None else self.known_zone_ids
        occupancy: Dict[str, int] = defaultdict(int)
        if zones:
            for z in zones:
                occupancy[z] = 0

        for traj in trajectories:
            for pt in traj.points:
                if pt.frame_idx == frame_idx:
                    if pt.zone_id is not None:
                        occupancy[pt.zone_id] += 1
                    break  # One point per frame per trajectory

        return dict(occupancy)

    def get_occupancy_timeline(
        self,
        trajectories: List[Trajectory],
        known_zone_ids: Optional[List[str]] = None
    ) -> Dict[int, Dict[str, int]]:
        """
        Compute zone occupancy across all frames observed in the trajectories.

        Args:
            trajectories (List[Trajectory]): List of evaluated trajectories.
            known_zone_ids (Optional[List[str]]): Optional zone IDs override.
                Defaults to self.known_zone_ids if None.

        Returns:
            Dict[int, Dict[str, int]]: Mapping frame_idx -> {zone_id -> track_count}.
        """
        zones = known_zone_ids if known_zone_ids is not None else self.known_zone_ids
        timeline: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        all_frames = set()
        for traj in trajectories:
            for pt in traj.points:
                all_frames.add(pt.frame_idx)
                if pt.zone_id is not None:
                    timeline[pt.frame_idx][pt.zone_id] += 1

        result: Dict[int, Dict[str, int]] = {}
        sorted_frames = sorted(list(all_frames))
        
        for f in sorted_frames:
            frame_occ = dict(timeline[f])
            if zones:
                for z in zones:
                    if z not in frame_occ:
                        frame_occ[z] = 0
            result[f] = frame_occ

        return result

