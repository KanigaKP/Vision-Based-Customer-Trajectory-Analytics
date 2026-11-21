"""
Vision-Based Customer Trajectory Analytics — Refined Dwell Analytics Foundation.
================================================================================
Phase 9.1: Derives per-track, per-zone dwell episodes and basic temporal
summaries from pre-assigned spatial trajectory observations.

KEY ARCHITECTURAL & METHODOLOGICAL PRINCIPLES:
1. Pure Analytics Layer:
   - Consumes pre-assigned TrajectoryPoint.zone_id observations (from Phase 6/7).
   - Never reruns detection, tracking, homography, PIP geometry, or hysteresis.
2. Clip-Local Track Identity:
   - Track IDs are temporary session-local tracker IDs; never merged or re-identified.
3. Observed Duration Span (No Continuous Physical Dwell Assumptions):
   - Durations measure observed in-zone temporal spans (exit_time - entry_time).
   - No extrapolation before initial or after final observation of an episode.
   - For single-frame visits, duration_sec is 0.0s.
4. Strict Episode Boundaries & Gap Semantics:
   - An observation assigned to None (outside defined zones) breaks an active episode.
   - Separate visits to the same zone are recorded as separate DwellEpisode records.
   - Consecutive observations in the same zone without intervening breaks are joined
     into one episode; the computed duration is an observed span and may overstate
     continuous occupancy if intermediate observations were missing.
5. Rigorous Temporal Validation:
   - Missing, invalid (NaN, Inf, negative, non-numeric), or non-monotonic timestamps
     are rejected with explicit ValueError exceptions; never bridged with guessed durations.

"""

from dataclasses import dataclass, asdict
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
class DwellEpisode:
    """
    Represents a contiguous temporal stay of a single clip-local track within one zone.

    Attributes:
        track_id: Clip-local tracker identifier integer.
        zone_id: Spatial retail zone identifier string.
        entry_frame_idx: Frame index of the initial observation in this episode.
        exit_frame_idx: Frame index of the final observation in this episode.
        entry_timestamp_sec: Video timestamp in seconds at initial observation.
        exit_timestamp_sec: Video timestamp in seconds at final observation.
        duration_sec: Observed in-zone duration span in seconds (exit_timestamp - entry_timestamp).
                      NOTE: This is an observed span between sampled detections, NOT a verified
                      measure of continuous physical dwell. For a single-frame observation,
                      duration_sec is 0.0s.
        point_count: Total sampled trajectory observations during this episode.
    """
    track_id: int
    zone_id: str
    entry_frame_idx: int
    exit_frame_idx: int
    entry_timestamp_sec: float
    exit_timestamp_sec: float
    duration_sec: float
    point_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, int) or isinstance(self.track_id, bool) or self.track_id < 0:
            raise ValueError(f"Invalid track_id: {self.track_id!r}")
        if not isinstance(self.zone_id, str) or not self.zone_id.strip():
            raise ValueError(f"Invalid zone_id: {self.zone_id!r}")
        if not isinstance(self.entry_frame_idx, int) or self.entry_frame_idx < 0:
            raise ValueError(f"Invalid entry_frame_idx: {self.entry_frame_idx!r}")
        if not isinstance(self.exit_frame_idx, int) or self.exit_frame_idx < self.entry_frame_idx:
            raise ValueError(
                f"exit_frame_idx ({self.exit_frame_idx}) cannot precede entry_frame_idx ({self.entry_frame_idx})"
            )
        if not _is_finite_number(self.entry_timestamp_sec) or self.entry_timestamp_sec < 0.0:
            raise ValueError(f"Invalid entry_timestamp_sec: {self.entry_timestamp_sec!r}")
        if not _is_finite_number(self.exit_timestamp_sec) or self.exit_timestamp_sec < self.entry_timestamp_sec:
            raise ValueError(
                f"exit_timestamp_sec ({self.exit_timestamp_sec}) cannot precede entry_timestamp_sec ({self.entry_timestamp_sec})"
            )
        if not _is_finite_number(self.duration_sec) or self.duration_sec < 0.0:
            raise ValueError(f"Invalid duration_sec: {self.duration_sec!r}")
        if not isinstance(self.point_count, int) or self.point_count < 1:
            raise ValueError(f"point_count must be at least 1, got: {self.point_count!r}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert DwellEpisode to dictionary representation."""
        return asdict(self)


@dataclass
class TrackDwellSummary:
    """
    Temporal dwell summary for a single clip-local track across all visited zones.

    Attributes:
        track_id: Clip-local tracker identifier.
        total_dwell_sec: Total observed dwell time in seconds across all zones.
        total_episodes: Total number of dwell episodes across all zones.
        zone_dwell_sec: Mapping zone_id -> total observed dwell time in seconds.
        zone_episode_count: Mapping zone_id -> number of distinct episodes.
        episodes: Chronological list of DwellEpisode instances for this track.
    """
    track_id: int
    total_dwell_sec: float
    total_episodes: int
    zone_dwell_sec: Dict[str, float]
    zone_episode_count: Dict[str, int]
    episodes: List[DwellEpisode]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "total_dwell_sec": self.total_dwell_sec,
            "total_episodes": self.total_episodes,
            "zone_dwell_sec": dict(self.zone_dwell_sec),
            "zone_episode_count": dict(self.zone_episode_count),
            "episodes": [e.to_dict() for e in self.episodes],
        }


def _calculate_median(values: Sequence[float]) -> float:
    """
    Calculate deterministic median of a sequence of float values.
    
    If length is odd (2m + 1), returns the middle element.
    If length is even (2m), returns the arithmetic mean of the two middle elements.
    If sequence is empty, returns 0.0.
    All outputs rounded to 3 decimal places.
    """
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        med = float(s[mid])
    else:
        med = float((s[mid - 1] + s[mid]) / 2.0)
    return round(med, 3)


@dataclass
class ZoneSpanDistribution:
    """
    Descriptive distribution of observed in-zone duration spans for a spatial zone.

    NOTE ON EMPTY-GROUP CONVENTION VS. ZERO-DURATION EPISODES:
        When episode_count == 0 (an unvisited zone), all span metrics (min, median,
        mean, max) evaluate to 0.0s by convention. The condition episode_count == 0
        unambiguously denotes the absence of observed episodes, NOT the presence of
        episodes with zero duration. Conversely, an observed single-point visit
        produces episode_count >= 1 with an observed span of 0.0s.

    Attributes:
        episode_count: Total count of observed episodes in this zone (0 for unvisited zones).
        min_span_sec: Minimum observed in-zone span in seconds (0.0 if empty).
        median_span_sec: Deterministic median observed in-zone span in seconds (0.0 if empty).
                         Provides a robust non-parametric summary of central tendency.
        mean_span_sec: Arithmetic mean observed in-zone span in seconds (0.0 if empty).
        max_span_sec: Maximum observed in-zone span in seconds (0.0 if empty).
    """
    episode_count: int
    min_span_sec: float
    median_span_sec: float
    mean_span_sec: float
    max_span_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ZoneDwellSummary:
    """
    Temporal dwell summary for a specific spatial zone across all evaluated tracks.

    NOTE ON EMPTY-GROUP INTERPRETATION:
        When total_episodes == 0 (episode_count == 0), the zone had no observed visits.
        Summary metrics (mean, median, min, max, total_dwell_sec) evaluate to 0.0s by convention.
        Callers must inspect total_episodes == 0 (or episode_count == 0) to distinguish
        an unvisited zone from a zone with observed zero-duration single-point visits.

    Attributes:
        zone_id: Spatial retail zone identifier string.
        total_dwell_sec: Total observed dwell span in seconds across all episodes.
        total_episodes: Total number of distinct episodes in this zone.
        unique_tracks: Number of distinct clip-local track IDs that entered this zone.
        mean_dwell_sec: Average episode duration span in seconds (0.0 if total_episodes == 0).
        median_dwell_sec: Median episode duration span in seconds (0.0 if total_episodes == 0).
                          Provides a robust non-parametric summary of central tendency.
        min_dwell_sec: Minimum episode duration span in seconds (0.0 if total_episodes == 0).
        max_dwell_sec: Maximum episode duration span in seconds (0.0 if total_episodes == 0).
        episodes: Chronological list of DwellEpisode instances in this zone.
    """
    zone_id: str
    total_dwell_sec: float
    total_episodes: int
    unique_tracks: int
    mean_dwell_sec: float
    median_dwell_sec: float
    min_dwell_sec: float
    max_dwell_sec: float
    episodes: List[DwellEpisode]

    @property
    def episode_count(self) -> int:
        """Alias for total_episodes to match distribution terminology."""
        return self.total_episodes

    @property
    def distribution(self) -> ZoneSpanDistribution:
        """Descriptive distribution of observed in-zone spans for this zone."""
        return ZoneSpanDistribution(
            episode_count=self.total_episodes,
            min_span_sec=self.min_dwell_sec,
            median_span_sec=self.median_dwell_sec,
            mean_span_sec=self.mean_dwell_sec,
            max_span_sec=self.max_dwell_sec,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "total_dwell_sec": self.total_dwell_sec,
            "total_episodes": self.total_episodes,
            "episode_count": self.total_episodes,
            "unique_tracks": self.unique_tracks,
            "mean_dwell_sec": self.mean_dwell_sec,
            "median_dwell_sec": self.median_dwell_sec,
            "min_dwell_sec": self.min_dwell_sec,
            "max_dwell_sec": self.max_dwell_sec,
            "distribution": self.distribution.to_dict(),
            "episodes": [e.to_dict() for e in self.episodes],
        }



@dataclass
class DwellSummary:
    """
    Aggregated temporal dwell analytics summary across all evaluated tracks and zones.

    Attributes:
        total_episodes: Total count of dwell episodes across all tracks and zones.
        total_dwell_sec: Total accumulated dwell duration in seconds.
        by_track: Mapping track_id -> TrackDwellSummary.
        by_zone: Mapping zone_id -> ZoneDwellSummary.
        episodes: Flat list of all extracted DwellEpisode instances.
    """
    total_episodes: int
    total_dwell_sec: float
    by_track: Dict[int, TrackDwellSummary]
    by_zone: Dict[str, ZoneDwellSummary]
    episodes: List[DwellEpisode]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_episodes": self.total_episodes,
            "total_dwell_sec": self.total_dwell_sec,
            "by_track": {tid: s.to_dict() for tid, s in self.by_track.items()},
            "by_zone": {zid: s.to_dict() for zid, s in self.by_zone.items()},
            "episodes": [e.to_dict() for e in self.episodes],
        }


def extract_dwell_episodes(
    trajectory_or_points: Union[Trajectory, Sequence[TrajectoryPoint]],
    track_id: Optional[int] = None,
) -> List[DwellEpisode]:
    """
    Derive contiguous dwell episodes from an evaluated trajectory or sequence of points.

    An episode is a contiguous sequence of observations within the same non-None zone.
    Any observation with zone_id=None or transitioning to another zone breaks the episode.
    Separate visits to the same zone are recorded as separate DwellEpisode instances.

    TREATMENT OF OBSERVATION GAPS:
    Observations belonging to the same track and zone with no intervening out-of-zone
    or different-zone observations are joined into the same episode. The resulting duration_sec
    is the observed temporal span between the first and last observation; it does not assume
    or verify continuous presence during unobserved intervals, and may overstate continuous
    occupancy if observations were missed or dropped.

    Args:
        trajectory_or_points: Either a Trajectory instance, or a Sequence of TrajectoryPoint instances.
        track_id: Optional explicit track_id override when passing raw point sequences.

    Returns:
        List[DwellEpisode]: Chronological list of dwell episodes.
    """
    if isinstance(trajectory_or_points, Trajectory):
        _validate_trajectory(trajectory_or_points)
        effective_track_id = trajectory_or_points.track_id
        points = trajectory_or_points.points
    elif isinstance(trajectory_or_points, (list, tuple)):
        points = list(trajectory_or_points)
        if not points:
            return []
        effective_track_id = track_id if track_id is not None else points[0].track_id
        # Build synthetic trajectory for validation
        dummy_traj = Trajectory(track_id=effective_track_id, points=points)
        _validate_trajectory(dummy_traj)
    else:
        raise TypeError(
            f"Expected Trajectory or Sequence[TrajectoryPoint], got {type(trajectory_or_points).__name__}"
        )

    if not points:
        return []

    episodes: List[DwellEpisode] = []
    current_zone: Optional[str] = None
    current_pts: List[TrajectoryPoint] = []

    for pt in points:
        pt_zone = pt.zone_id

        if pt_zone == current_zone:
            # Same zone continuation
            if current_zone is not None:
                current_pts.append(pt)
        else:
            # Zone transition or exit detected
            # 1. Close active episode if exiting a defined zone
            if current_zone is not None and current_pts:
                dur = round(max(0.0, current_pts[-1].timestamp_sec - current_pts[0].timestamp_sec), 3)
                episodes.append(DwellEpisode(
                    track_id=effective_track_id,
                    zone_id=current_zone,
                    entry_frame_idx=current_pts[0].frame_idx,
                    exit_frame_idx=current_pts[-1].frame_idx,
                    entry_timestamp_sec=current_pts[0].timestamp_sec,
                    exit_timestamp_sec=current_pts[-1].timestamp_sec,
                    duration_sec=dur,
                    point_count=len(current_pts),
                ))
                current_pts = []

            # 2. If entering a valid zone, start a new episode
            if pt_zone is not None:
                current_pts = [pt]

            current_zone = pt_zone

    # Close trailing active episode if trajectory ended inside a zone
    if current_zone is not None and current_pts:
        dur = round(max(0.0, current_pts[-1].timestamp_sec - current_pts[0].timestamp_sec), 3)
        episodes.append(DwellEpisode(
            track_id=effective_track_id,
            zone_id=current_zone,
            entry_frame_idx=current_pts[0].frame_idx,
            exit_frame_idx=current_pts[-1].frame_idx,
            entry_timestamp_sec=current_pts[0].timestamp_sec,
            exit_timestamp_sec=current_pts[-1].timestamp_sec,
            duration_sec=dur,
            point_count=len(current_pts),
        ))

    return episodes


class DwellAnalyzer:
    """
    Analyzes zone-assigned trajectories to produce explainable per-track
    and per-zone temporal dwell summaries and episode breakdowns.
    """

    def __init__(self, known_zone_ids: Optional[Sequence[str]] = None):
        """
        Initialize DwellAnalyzer.

        Args:
            known_zone_ids: Optional list of expected zone identifiers.
                If provided, zone summaries will include zero-activity entries
                for known zones with no observed visits.
        """
        if known_zone_ids is not None:
            if not isinstance(known_zone_ids, (list, tuple, set)):
                raise TypeError(
                    f"known_zone_ids must be a sequence of strings, got {type(known_zone_ids).__name__}"
                )
            for z in known_zone_ids:
                if not isinstance(z, str) or not z.strip():
                    raise ValueError(f"Each known_zone_id must be a non-empty string, got: {z!r}")
            self.known_zone_ids: List[str] = sorted(list(set(known_zone_ids)))
        else:
            self.known_zone_ids = []

    def extract_episodes(self, trajectory: Trajectory) -> List[DwellEpisode]:
        """Extract dwell episodes from a single evaluated trajectory."""
        return extract_dwell_episodes(trajectory)

    def analyze_track(self, trajectory: Trajectory) -> TrackDwellSummary:
        """
        Produce a TrackDwellSummary for a single evaluated trajectory.

        Args:
            trajectory: Trajectory with populated TrajectoryPoint.zone_id.

        Returns:
            TrackDwellSummary: Per-track dwell metrics and episode list.
        """
        episodes = self.extract_episodes(trajectory)
        zone_durations: Dict[str, float] = defaultdict(float)
        zone_counts: Dict[str, int] = defaultdict(int)

        total_sec = 0.0
        for ep in episodes:
            zone_durations[ep.zone_id] += ep.duration_sec
            zone_counts[ep.zone_id] += 1
            total_sec += ep.duration_sec

        # Round durations to 3 decimal places
        rounded_zone_durations = {z: round(val, 3) for z, val in zone_durations.items()}

        return TrackDwellSummary(
            track_id=trajectory.track_id,
            total_dwell_sec=round(total_sec, 3),
            total_episodes=len(episodes),
            zone_dwell_sec=dict(rounded_zone_durations),
            zone_episode_count=dict(zone_counts),
            episodes=episodes,
        )

    def analyze(self, trajectories: Sequence[Trajectory]) -> DwellSummary:
        """
        Aggregate dwell analytics across a collection of evaluated trajectories.

        Args:
            trajectories: Sequence of Trajectory objects with assigned zone_ids.

        Returns:
            DwellSummary: Full summary containing by_track, by_zone, and all episodes.
        """
        if not isinstance(trajectories, (list, tuple)):
            raise TypeError(f"trajectories must be a sequence, got {type(trajectories).__name__}")

        by_track: Dict[int, TrackDwellSummary] = {}
        all_episodes: List[DwellEpisode] = []
        zone_episodes: Dict[str, List[DwellEpisode]] = defaultdict(list)
        zone_unique_tracks: Dict[str, set] = defaultdict(set)

        for traj in trajectories:
            summary = self.analyze_track(traj)
            by_track[traj.track_id] = summary
            all_episodes.extend(summary.episodes)

            for ep in summary.episodes:
                zone_episodes[ep.zone_id].append(ep)
                zone_unique_tracks[ep.zone_id].add(ep.track_id)

        # Collect all active zones (known zones + observed zones)
        active_zones = set(self.known_zone_ids)
        for ep in all_episodes:
            active_zones.add(ep.zone_id)

        sorted_zones = sorted(list(active_zones))

        by_zone: Dict[str, ZoneDwellSummary] = {}
        for z in sorted_zones:
            eps = zone_episodes.get(z, [])
            count = len(eps)
            if count == 0:
                by_zone[z] = ZoneDwellSummary(
                    zone_id=z,
                    total_dwell_sec=0.0,
                    total_episodes=0,
                    unique_tracks=0,
                    mean_dwell_sec=0.0,
                    median_dwell_sec=0.0,
                    min_dwell_sec=0.0,
                    max_dwell_sec=0.0,
                    episodes=[],
                )
            else:
                spans = [e.duration_sec for e in eps]
                total_dur = round(sum(spans), 3)
                mean_dur = round(total_dur / count, 3)
                median_dur = _calculate_median(spans)
                min_dur = round(min(spans), 3)
                max_dur = round(max(spans), 3)
                unique_cnt = len(zone_unique_tracks.get(z, set()))
                by_zone[z] = ZoneDwellSummary(
                    zone_id=z,
                    total_dwell_sec=total_dur,
                    total_episodes=count,
                    unique_tracks=unique_cnt,
                    mean_dwell_sec=mean_dur,
                    median_dwell_sec=median_dur,
                    min_dwell_sec=min_dur,
                    max_dwell_sec=max_dur,
                    episodes=eps,
                )

        total_dwell_overall = round(sum(e.duration_sec for e in all_episodes), 3)

        return DwellSummary(
            total_episodes=len(all_episodes),
            total_dwell_sec=total_dwell_overall,
            by_track=by_track,
            by_zone=by_zone,
            episodes=all_episodes,
        )


def analyze_dwell(
    trajectories: Sequence[Trajectory],
    known_zone_ids: Optional[Sequence[str]] = None,
) -> DwellSummary:
    """
    Convenience function to analyze a sequence of trajectories and produce a DwellSummary.

    Args:
        trajectories: Sequence of Trajectory objects with populated zone_ids.
        known_zone_ids: Optional sequence of expected zone identifiers.

    Returns:
        DwellSummary: Aggregated dwell metrics across tracks and zones.
    """
    analyzer = DwellAnalyzer(known_zone_ids=known_zone_ids)
    return analyzer.analyze(trajectories)


def compute_zone_span_distribution(episodes: Sequence[DwellEpisode]) -> ZoneSpanDistribution:
    """
    Compute descriptive distribution statistics of observed in-zone duration spans
    directly from a sequence of DwellEpisode records.

    Args:
        episodes: Sequence of DwellEpisode instances for a zone.

    Returns:
        ZoneSpanDistribution: Episode count, min, median, arithmetic mean, and max observed spans.
    """
    if not episodes:
        return ZoneSpanDistribution(
            episode_count=0,
            min_span_sec=0.0,
            median_span_sec=0.0,
            mean_span_sec=0.0,
            max_span_sec=0.0,
        )
    spans = [e.duration_sec for e in episodes]
    total = sum(spans)
    count = len(spans)
    return ZoneSpanDistribution(
        episode_count=count,
        min_span_sec=round(min(spans), 3),
        median_span_sec=_calculate_median(spans),
        mean_span_sec=round(total / count, 3),
        max_span_sec=round(max(spans), 3),
    )

