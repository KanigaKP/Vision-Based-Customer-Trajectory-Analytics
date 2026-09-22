"""
Vision-Based Customer Trajectory Analytics — Zone Evaluator Module.

This module evaluates 2D floor-coordinate points against defined retail store zones
using Point-in-Polygon (PIP) testing and configurable temporal hysteresis filtering.
"""

from typing import List, Optional, Tuple, Dict, Any
from matplotlib.path import Path
import numpy as np

from src.analytics.zone import Zone
from src.trajectory.trajectory import Trajectory, TrajectoryPoint


class ZoneEvaluator:
    """
    Evaluates spatial points and trajectories against a set of floor-plane retail zones.

    Attributes:
        zones (List[Zone]): List of spatial Zone definitions.
        hysteresis_window (int): Default number of consecutive frames required
            for a zone transition to be confirmed. (Configurable default N=3).
    """

    def __init__(
        self,
        zones: List[Zone],
        hysteresis_window: int = 3
    ):
        """
        Initialize ZoneEvaluator with zone definitions and hysteresis buffer threshold.

        Args:
            zones (List[Zone]): Defined retail store zones.
            hysteresis_window (int): Frame buffer count N for boundary stability.
        """
        self.zones = zones
        self.hysteresis_window = max(1, int(hysteresis_window))
        
        # Build Matplotlib Path objects for fast PIP checking
        self._path_cache: Dict[str, Path] = {}
        for z in self.zones:
            # Matplotlib Path requires (x, y) float coordinates
            self._path_cache[z.zone_id] = Path(np.array(z.polygon, dtype=np.float64))

    def evaluate_point(self, x: Optional[float], y: Optional[float]) -> Optional[str]:
        """
        Perform raw Point-in-Polygon (PIP) test for a single 2D floor coordinate.

        Args:
            x (Optional[float]): Floor X coordinate.
            y (Optional[float]): Floor Y coordinate.

        Returns:
            Optional[str]: zone_id of the matching zone, or None if outside all defined zones.
        """
        if x is None or y is None:
            return None

        point = (float(x), float(y))
        for z in self.zones:
            path = self._path_cache[z.zone_id]
            if path.contains_point(point):
                return z.zone_id

        return None

    def evaluate_trajectory(
        self,
        trajectory: Trajectory,
        hysteresis_window: Optional[int] = None
    ) -> List[Optional[str]]:
        """
        Evaluate all points in a trajectory with temporal hysteresis boundary filtering.

        Args:
            trajectory (Trajectory): Trajectory object containing TrajectoryPoint instances.
            hysteresis_window (Optional[int]): Override default hysteresis window N.

        Returns:
            List[Optional[str]]: Ordered list of assigned zone_ids for each trajectory point.
        """
        n_win = self.hysteresis_window if hysteresis_window is None else max(1, int(hysteresis_window))
        assigned_zones: List[Optional[str]] = []

        if not trajectory.points:
            return assigned_zones

        current_zone: Optional[str] = None
        candidate_zone: Optional[str] = None
        candidate_count: int = 0

        for idx, pt in enumerate(trajectory.points):
            raw_zone = self.evaluate_point(pt.floor_x, pt.floor_y)

            if n_win <= 1:
                current_zone = raw_zone
            else:
                if idx == 0:
                    # Initial point assignment
                    current_zone = raw_zone
                    candidate_zone = raw_zone
                    candidate_count = 1
                elif raw_zone == current_zone:
                    # Point remains in confirmed zone
                    candidate_zone = raw_zone
                    candidate_count = 0
                else:
                    # Point is in a candidate zone different from confirmed zone
                    if raw_zone == candidate_zone:
                        candidate_count += 1
                    else:
                        candidate_zone = raw_zone
                        candidate_count = 1

                    if candidate_count >= n_win:
                        current_zone = candidate_zone

            pt.zone_id = current_zone
            assigned_zones.append(current_zone)

        return assigned_zones
