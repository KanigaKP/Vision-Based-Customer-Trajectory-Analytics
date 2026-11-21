"""
Temporal Hysteresis Sensitivity Analysis Module
===============================================
Measures how spatial zone assignments, discrete entry events, zone-to-zone transitions,
and boundary oscillations vary across different temporal hysteresis window thresholds (N).

Architectural Rules & Methodological Boundaries:
1. Reuses Existing Phase 6 & Phase 7 Production Logic:
   - Reruns existing ZoneEvaluator.evaluate_trajectory(...) for each tested window N.
   - Reruns existing FootfallAnalyzer.analyze(...) to compute genuine FootfallSummary metrics.
   - Does NOT duplicate Point-in-Polygon (PIP) testing or footfall event aggregation.
2. Complete Caller State Preservation (Zero Mutation):
   - ZoneEvaluator mutates TrajectoryPoint.zone_id in-place.
   - To prevent contamination across window runs or alteration of caller-owned data,
     independent trajectory clones are evaluated for each window.
   - Caller-owned Trajectory and Zone objects remain unmodified.
3. Descriptive Measurement, Not Optimal Threshold Selection:
   - Reports raw event metrics, boundary transitions, and deltas relative to a baseline.
   - Does NOT declare any hysteresis setting 'optimal' or invent arbitrary accuracy thresholds.
   - Synthetic test fixtures demonstrate software sensitivity mechanics only and do not
     constitute physical real-world camera or layout validation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Sequence, Union
import copy
from collections.abc import Sequence as SequenceABC, Set as SetABC

from src.analytics.zone import Zone, ZoneConfig
from src.analytics.zone_evaluator import ZoneEvaluator
from src.analytics.footfall import FootfallAnalyzer, FootfallSummary
from src.trajectory.trajectory import Trajectory, TrajectoryPoint


@dataclass
class HysteresisWindowMetrics:
    """
    Evaluated metrics for a specific temporal hysteresis window N.

    Attributes:
        window: Integer hysteresis window threshold N.
        total_entry_events: Sum of all discrete zone entry events across evaluated zones.
        entry_events_per_zone: Mapping of zone_id -> entry count.
        total_transitions: Sum of all inter-zone transitions across defined zones.
        transition_matrix: 2D dictionary mapping from_zone -> to_zone -> count.
        total_zone_changes: Total consecutive point zone label changes across all trajectories.
        unique_tracks_per_zone: Mapping of zone_id -> distinct track count.
        delta_entry_events: Difference in total entry events relative to baseline (N - baseline).
        delta_transitions: Difference in total transitions relative to baseline.
        delta_zone_changes: Difference in total zone changes relative to baseline.
        delta_entries_per_zone: Per-zone entry differences relative to baseline.
    """
    window: int
    total_entry_events: int
    entry_events_per_zone: Dict[str, int]
    total_transitions: int
    transition_matrix: Dict[str, Dict[str, int]]
    total_zone_changes: int
    unique_tracks_per_zone: Dict[str, int]
    delta_entry_events: int = 0
    delta_transitions: int = 0
    delta_zone_changes: int = 0
    delta_entries_per_zone: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "window": self.window,
            "total_entry_events": self.total_entry_events,
            "entry_events_per_zone": dict(self.entry_events_per_zone),
            "total_transitions": self.total_transitions,
            "transition_matrix": {k: dict(v) for k, v in self.transition_matrix.items()},
            "total_zone_changes": self.total_zone_changes,
            "unique_tracks_per_zone": dict(self.unique_tracks_per_zone),
            "delta_entry_events": self.delta_entry_events,
            "delta_transitions": self.delta_transitions,
            "delta_zone_changes": self.delta_zone_changes,
            "delta_entries_per_zone": dict(self.delta_entries_per_zone),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HysteresisWindowMetrics":
        """Reconstruct HysteresisWindowMetrics from dictionary."""
        return cls(
            window=int(data["window"]),
            total_entry_events=int(data["total_entry_events"]),
            entry_events_per_zone={k: int(v) for k, v in data.get("entry_events_per_zone", {}).items()},
            total_transitions=int(data["total_transitions"]),
            transition_matrix={
                k: {k2: int(v2) for k2, v2 in v.items()}
                for k, v in data.get("transition_matrix", {}).items()
            },
            total_zone_changes=int(data["total_zone_changes"]),
            unique_tracks_per_zone={k: int(v) for k, v in data.get("unique_tracks_per_zone", {}).items()},
            delta_entry_events=int(data.get("delta_entry_events", 0)),
            delta_transitions=int(data.get("delta_transitions", 0)),
            delta_zone_changes=int(data.get("delta_zone_changes", 0)),
            delta_entries_per_zone={k: int(v) for k, v in data.get("delta_entries_per_zone", {}).items()},
        )


@dataclass
class HysteresisSensitivityResult:
    """
    Container for temporal hysteresis sensitivity evaluation results.

    Attributes:
        dataset_name: Identifier string.
        baseline_window: The reference window N used to compute deltas.
        evaluated_windows: Sorted list of evaluated hysteresis window values.
        window_metrics: List of HysteresisWindowMetrics in sorted window order.
        evidence_status: 'HYSTERESIS_SENSITIVITY_EVALUATED' or 'INSUFFICIENT_EVIDENCE'.
        evidence_note: Methodological scope explanation.
    """
    dataset_name: str
    baseline_window: int
    evaluated_windows: List[int]
    window_metrics: List[HysteresisWindowMetrics]
    evidence_status: str
    evidence_note: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "baseline_window": self.baseline_window,
            "evaluated_windows": list(self.evaluated_windows),
            "window_metrics": [m.to_dict() for m in self.window_metrics],
            "evidence_status": self.evidence_status,
            "evidence_note": self.evidence_note,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HysteresisSensitivityResult":
        """Reconstruct HysteresisSensitivityResult from dictionary."""
        return cls(
            dataset_name=data.get("dataset_name", "default"),
            baseline_window=int(data["baseline_window"]),
            evaluated_windows=[int(w) for w in data.get("evaluated_windows", [])],
            window_metrics=[HysteresisWindowMetrics.from_dict(m) for m in data.get("window_metrics", [])],
            evidence_status=data["evidence_status"],
            evidence_note=data["evidence_note"],
        )


class HysteresisSensitivityValidator:
    """
    Evaluates sensitivity of spatial zone evaluation and footfall metrics
    across varied temporal hysteresis windows.
    """

    @staticmethod
    def _clone_trajectories(trajectories: List[Trajectory]) -> List[Trajectory]:
        """
        Deep-copy trajectory objects to guarantee caller trajectories are never mutated.
        """
        clones: List[Trajectory] = []
        for traj in trajectories:
            cloned_points = [
                TrajectoryPoint(
                    track_id=pt.track_id,
                    frame_idx=pt.frame_idx,
                    timestamp_sec=pt.timestamp_sec,
                    image_x=pt.image_x,
                    image_y=pt.image_y,
                    floor_x=pt.floor_x,
                    floor_y=pt.floor_y,
                    zone_id=pt.zone_id,
                )
                for pt in traj.points
            ]
            clones.append(
                Trajectory(
                    track_id=traj.track_id,
                    points=cloned_points,
                    class_id=traj.class_id,
                    class_name=traj.class_name,
                )
            )
        return clones

    @classmethod
    def analyze_sensitivity(
        cls,
        zones: Union[List[Zone], Dict[str, Any]],
        trajectories: List[Trajectory],
        windows: Optional[Union[Sequence[int], Set[int]]] = None,
        baseline_window: Optional[int] = None,
        dataset_name: str = "default",
    ) -> HysteresisSensitivityResult:
        """
        Evaluate footfall sensitivity across multiple temporal hysteresis window settings.

        Args:
            zones: List of Zone objects or dictionary configuration.
            trajectories: Evaluated Trajectory objects with populated floor coordinates.
            windows: Sequence or set of positive integer hysteresis window values N (default: 1..10).
            baseline_window: Reference window N for calculating deltas. Defaults to 3 if evaluated,
                otherwise the first sorted evaluated window.
            dataset_name: Identifier string.

        Returns:
            HysteresisSensitivityResult with per-window metrics and baseline deltas.

        Raises:
            TypeError: If input types are invalid.
            ValueError: If windows list is empty, contains non-positive numbers, duplicates,
                or if baseline_window is not among evaluated windows.
        """
        # 1. Normalize and validate zones
        if isinstance(zones, list):
            zone_list = zones
        elif isinstance(zones, dict):
            zone_list, _ = ZoneConfig.load_from_config(zones)
        else:
            raise TypeError(f"zones must be a List[Zone] or configuration dict. Got: {type(zones).__name__}")

        if not zone_list:
            raise ValueError("zones list cannot be empty.")

        for idx, z in enumerate(zone_list):
            if not isinstance(z, Zone):
                raise TypeError(f"Item at index {idx} in zones is not a Zone instance. Got: {type(z).__name__}")

        # 2. Validate trajectories
        if not isinstance(trajectories, list):
            raise TypeError(f"trajectories must be a list. Got: {type(trajectories).__name__}")

        for idx, traj in enumerate(trajectories):
            if not isinstance(traj, Trajectory):
                raise TypeError(f"Item at index {idx} in trajectories is not a Trajectory instance. Got: {type(traj).__name__}")

        # 3. Validate windows
        if windows is None:
            sorted_windows = list(range(1, 11))
        else:
            if isinstance(windows, (str, bytes, bytearray)) or not isinstance(windows, (SequenceABC, SetABC, set)):
                raise TypeError(f"windows must be a non-string sequence of integers. Got: {type(windows).__name__}")
            if len(windows) == 0:
                raise ValueError("windows sequence cannot be empty.")

            seen_windows: Set[int] = set()
            win_list: List[int] = []
            for w in windows:
                if isinstance(w, bool) or not isinstance(w, int):
                    raise TypeError(f"Hysteresis window value must be an integer. Got: {w!r}")
                if w <= 0:
                    raise ValueError(f"Hysteresis window must be a positive integer (> 0). Got: {w!r}")
                if w in seen_windows:
                    raise ValueError(f"Duplicate hysteresis window value found: {w}")
                seen_windows.add(w)
                win_list.append(w)

            sorted_windows = sorted(win_list)

        # 4. Resolve baseline window
        if baseline_window is not None:
            if isinstance(baseline_window, bool) or not isinstance(baseline_window, int):
                raise TypeError(f"baseline_window must be an integer. Got: {baseline_window!r}")
            if baseline_window not in sorted_windows:
                raise ValueError(
                    f"Specified baseline_window {baseline_window} is not among evaluated windows {sorted_windows}."
                )
            resolved_baseline = baseline_window
        else:
            # Default to 3 if present, otherwise first sorted window
            resolved_baseline = 3 if 3 in sorted_windows else sorted_windows[0]

        # 5. Handle empty trajectory input explicitly
        if len(trajectories) == 0:
            return HysteresisSensitivityResult(
                dataset_name=dataset_name,
                baseline_window=resolved_baseline,
                evaluated_windows=sorted_windows,
                window_metrics=[],
                evidence_status="INSUFFICIENT_EVIDENCE",
                evidence_note="No trajectories provided for hysteresis sensitivity analysis.",
            )

        # 6. Evaluate each hysteresis window using cloned trajectories
        raw_metrics: Dict[int, HysteresisWindowMetrics] = {}
        known_zone_ids = [z.zone_id for z in zone_list]

        for w in sorted_windows:
            # Clone trajectories to prevent cross-window mutation or caller contamination
            cloned_trajs = cls._clone_trajectories(trajectories)

            # Evaluate PIP and hysteresis
            evaluator = ZoneEvaluator(zones=zone_list, hysteresis_window=w)
            for traj in cloned_trajs:
                evaluator.evaluate_trajectory(traj, hysteresis_window=w)

            # Analyze footfall
            analyzer = FootfallAnalyzer(known_zone_ids=known_zone_ids)
            summary = analyzer.analyze(cloned_trajs)

            # Compute consecutive zone transitions across trajectory points
            zone_changes = 0
            for traj in cloned_trajs:
                pts = traj.points
                for i in range(1, len(pts)):
                    if pts[i].zone_id != pts[i - 1].zone_id:
                        zone_changes += 1

            total_entries = sum(summary.entry_events_per_zone.values())
            total_trans = sum(
                sum(target_map.values())
                for target_map in summary.transition_matrix.values()
            )

            raw_metrics[w] = HysteresisWindowMetrics(
                window=w,
                total_entry_events=total_entries,
                entry_events_per_zone=dict(summary.entry_events_per_zone),
                total_transitions=total_trans,
                transition_matrix={k: dict(v) for k, v in summary.transition_matrix.items()},
                total_zone_changes=zone_changes,
                unique_tracks_per_zone=dict(summary.unique_tracks_per_zone),
            )

        # 7. Compute deltas relative to baseline
        base_m = raw_metrics[resolved_baseline]
        window_metrics_list: List[HysteresisWindowMetrics] = []

        for w in sorted_windows:
            m = raw_metrics[w]
            delta_entries = m.total_entry_events - base_m.total_entry_events
            delta_trans = m.total_transitions - base_m.total_transitions
            delta_changes = m.total_zone_changes - base_m.total_zone_changes

            delta_per_zone: Dict[str, int] = {}
            for z_id in known_zone_ids:
                delta_per_zone[z_id] = m.entry_events_per_zone.get(z_id, 0) - base_m.entry_events_per_zone.get(z_id, 0)

            m.delta_entry_events = delta_entries
            m.delta_transitions = delta_trans
            m.delta_zone_changes = delta_changes
            m.delta_entries_per_zone = delta_per_zone

            window_metrics_list.append(m)

        return HysteresisSensitivityResult(
            dataset_name=dataset_name,
            baseline_window=resolved_baseline,
            evaluated_windows=sorted_windows,
            window_metrics=window_metrics_list,
            evidence_status="HYSTERESIS_SENSITIVITY_EVALUATED",
            evidence_note=(
                f"Evaluated temporal hysteresis sensitivity across {len(sorted_windows)} windows "
                f"(baseline N={resolved_baseline}). Note: Raw sensitivity measurements of existing implementation "
                f"on supplied trajectories only; does not claim empirical optimality or real-world accuracy."
            ),
        )


def analyze_hysteresis_sensitivity(
    zones: Union[List[Zone], Dict[str, Any]],
    trajectories: List[Trajectory],
    windows: Optional[Union[Sequence[int], Set[int]]] = None,
    baseline_window: Optional[int] = None,
    dataset_name: str = "default",
) -> HysteresisSensitivityResult:
    """
    Convenience function to evaluate hysteresis window sensitivity.
    """
    return HysteresisSensitivityValidator.analyze_sensitivity(
        zones=zones,
        trajectories=trajectories,
        windows=windows,
        baseline_window=baseline_window,
        dataset_name=dataset_name,
    )
