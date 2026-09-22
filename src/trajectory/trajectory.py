"""
Trajectory Representation & Extraction Module
=============================================
Defines spatial-temporal trajectory containers and extracts ordered trajectory points
from ByteTrack tracking histories.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np

from src.tracking.bytetrack import Track, TrackPoint


@dataclass
class TrajectoryPoint:
    """Individual spatial-temporal trajectory point."""
    track_id: int
    frame_idx: int
    timestamp_sec: float
    image_x: float                 # Bottom-center x = (x1 + x2) / 2
    image_y: float                 # Bottom-center y = y2
    floor_x: Optional[float] = None # Transformed floor coordinate x
    floor_y: Optional[float] = None # Transformed floor coordinate y
    zone_id: Optional[str] = None   # Spatial zone assignment (None if outside defined zones)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Trajectory:
    """Container for a single tracked individual's spatial trajectory over time."""

    def __init__(
        self,
        track_id: int,
        points: Optional[List[TrajectoryPoint]] = None,
        class_id: int = 0,
        class_name: str = "person"
    ):
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.points: List[TrajectoryPoint] = points if points is not None else []

    def add_point(self, pt: TrajectoryPoint):
        """Append a new trajectory point preserving temporal order."""
        if self.points and pt.frame_idx < self.points[-1].frame_idx:
            raise ValueError(f"Temporal ordering violation: frame {pt.frame_idx} < {self.points[-1].frame_idx}")
        self.points.append(pt)

    @property
    def start_frame(self) -> Optional[int]:
        return self.points[0].frame_idx if self.points else None

    @property
    def end_frame(self) -> Optional[int]:
        return self.points[-1].frame_idx if self.points else None

    @property
    def duration_sec(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return round(self.points[-1].timestamp_sec - self.points[0].timestamp_sec, 3)

    @property
    def point_count(self) -> int:
        return len(self.points)

    def calculate_image_distance(self) -> float:
        """Calculate total path length in image pixel space."""
        if len(self.points) < 2:
            return 0.0
        pts = np.array([[p.image_x, p.image_y] for p in self.points], dtype=np.float32)
        diffs = np.diff(pts, axis=0)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        return round(float(np.sum(dists)), 2)

    def calculate_floor_distance(self) -> float:
        """Calculate total path length in floor coordinate units (metres/grid)."""
        if len(self.points) < 2:
            return 0.0
        # Only compute if floor_x and floor_y are populated
        valid_pts = [[p.floor_x, p.floor_y] for p in self.points if p.floor_x is not None and p.floor_y is not None]
        if len(valid_pts) < 2:
            return 0.0
        pts = np.array(valid_pts, dtype=np.float32)
        diffs = np.diff(pts, axis=0)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        return round(float(np.sum(dists)), 2)

    def get_image_points(self) -> np.ndarray:
        """Return image-space points as (N, 2) float32 numpy array."""
        if not self.points:
            return np.empty((0, 2), dtype=np.float32)
        return np.array([[p.image_x, p.image_y] for p in self.points], dtype=np.float32)

    def get_floor_points(self) -> np.ndarray:
        """Return floor-space points as (N, 2) float32 numpy array (skips un-transformed)."""
        valid = [[p.floor_x, p.floor_y] for p in self.points if p.floor_x is not None and p.floor_y is not None]
        if not valid:
            return np.empty((0, 2), dtype=np.float32)
        return np.array(valid, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "point_count": self.point_count,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_sec": self.duration_sec,
            "path_length_image_px": self.calculate_image_distance(),
            "path_length_floor_units": self.calculate_floor_distance(),
            "points": [p.to_dict() for p in self.points]
        }


class TrajectoryExtractor:
    """Extracts Trajectory objects from Track tracking history."""

    @staticmethod
    def extract_from_track(track: Track) -> Trajectory:
        """Extract a single Trajectory from a Track object."""
        traj_points = []
        for pt in track.history:
            # Bottom-center calculation: (x1 + x2)/2, y2
            cx, cy = pt.bottom_center
            t_pt = TrajectoryPoint(
                track_id=track.track_id,
                frame_idx=pt.frame_id,
                timestamp_sec=pt.timestamp_sec,
                image_x=cx,
                image_y=cy
            )
            traj_points.append(t_pt)

        # Ensure temporal ordering
        traj_points.sort(key=lambda p: p.frame_idx)

        return Trajectory(
            track_id=track.track_id,
            points=traj_points,
            class_id=track.class_id,
            class_name=track.class_name
        )

    @classmethod
    def extract_from_tracks(cls, tracks: List[Track]) -> List[Trajectory]:
        """Extract a list of Trajectories from a list of Track objects."""
        return [cls.extract_from_track(t) for t in tracks if len(t.history) > 0]
