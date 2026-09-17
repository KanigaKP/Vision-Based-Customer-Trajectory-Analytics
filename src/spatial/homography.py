"""
Spatial Homography & Floor-Plane Mapping Module
===============================================
Computes perspective transformation matrices between image coordinates and
ground floor-plane coordinates. Transforms image-plane trajectories into floor coordinates
and computes geometric consistency diagnostics.

SCIENTIFIC DISCLAIMER:
A mathematical homography transformation does NOT prove absolute metric accuracy unless
physically measured on-site. Reprojection and check-point errors measure geometric consistency.
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

from src.trajectory.trajectory import Trajectory, TrajectoryPoint


@dataclass
class HomographyDiagnostics:
    """Container for homography calibration quality metrics."""
    dataset_name: str
    unit: str
    num_calibration_points: int
    training_reprojection_mean_error: float
    training_reprojection_max_error: float
    num_check_points: int
    check_point_mean_error: Optional[float]
    check_point_max_error: Optional[float]
    homography_matrix: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HomographyCalibrator:
    """
    Computes and manages image-to-floor-plane homography transformations.
    Camera/view specific — requires distinct calibration per camera setup.
    """

    def __init__(
        self,
        image_points: List[List[float]],
        floor_points: List[List[float]],
        check_points_image: Optional[List[List[float]]] = None,
        check_points_floor: Optional[List[List[float]]] = None,
        unit: str = "metres",
        dataset_name: str = "default"
    ):
        self.dataset_name = dataset_name
        self.unit = unit

        # Validate correspondence points
        self.src_pts = np.array(image_points, dtype=np.float32)
        self.dst_pts = np.array(floor_points, dtype=np.float32)

        self._validate_points(self.src_pts, self.dst_pts)

        # Optional independent check points
        self.check_src = np.array(check_points_image, dtype=np.float32) if check_points_image else None
        self.check_dst = np.array(check_points_floor, dtype=np.float32) if check_points_floor else None

        # Compute 3x3 Homography Matrix
        self.H, self.status = cv2.findHomography(self.src_pts, self.dst_pts)
        if self.H is None:
            raise ValueError(f"Failed to compute homography matrix for dataset {dataset_name}. Check correspondence points.")

        # Compute Diagnostics
        self.diagnostics = self._compute_diagnostics()

    @staticmethod
    def _validate_points(src: np.ndarray, dst: np.ndarray):
        """Validate input points: >= 4 points, matching dimensions, non-collinear."""
        if len(src) < 4 or len(dst) < 4:
            raise ValueError(f"At least 4 correspondence points are required. Got src={len(src)}, dst={len(dst)}")
        if len(src) != len(dst):
            raise ValueError(f"Source and destination point counts must match. Got src={len(src)}, dst={len(dst)}")

        # Check for collinearity by checking triangle areas of triplets
        # If all 3-point combinations have ~0 area, points are collinear
        x, y = src[:, 0], src[:, 1]
        area_sum = 0.0
        for i in range(len(src) - 2):
            for j in range(i + 1, len(src) - 1):
                for k in range(j + 1, len(src)):
                    area = 0.5 * abs(x[i]*(y[j] - y[k]) + x[j]*(y[k] - y[i]) + x[k]*(y[i] - y[j]))
                    area_sum += area
        if area_sum < 1e-3:
            raise ValueError("Correspondence points are collinear or degenerate. Cannot compute valid 2D homography.")

    def _compute_diagnostics(self) -> HomographyDiagnostics:
        """Calculate training reprojection errors and independent check point errors."""
        # 1. Training Reprojection Error
        transformed_src = self.transform_points(self.src_pts)
        train_errors = np.linalg.norm(transformed_src - self.dst_pts, axis=1)
        mean_train = float(np.mean(train_errors))
        max_train = float(np.max(train_errors))

        # 2. Independent Check-Point Error (if provided)
        mean_check = None
        max_check = None
        num_check = 0

        if self.check_src is not None and self.check_dst is not None and len(self.check_src) > 0:
            transformed_check = self.transform_points(self.check_src)
            check_errors = np.linalg.norm(transformed_check - self.check_dst, axis=1)
            mean_check = float(np.mean(check_errors))
            max_check = float(np.max(check_errors))
            num_check = len(self.check_src)

        return HomographyDiagnostics(
            dataset_name=self.dataset_name,
            unit=self.unit,
            num_calibration_points=len(self.src_pts),
            training_reprojection_mean_error=round(mean_train, 4),
            training_reprojection_max_error=round(max_train, 4),
            num_check_points=num_check,
            check_point_mean_error=round(mean_check, 4) if mean_check is not None else None,
            check_point_max_error=round(max_check, 4) if max_check is not None else None,
            homography_matrix=self.H.tolist()
        )

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Transform a single 2D image point (x, y) to floor coordinates (floor_x, floor_y)."""
        pts = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.H)
        fx, fy = transformed[0, 0]
        return round(float(fx), 2), round(float(fy), 2)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform an (N, 2) numpy array of image points to floor coordinates (N, 2).
        """
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts_3d = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts_3d, self.H)
        return transformed.reshape(-1, 2)

    def transform_trajectory(self, trajectory: Trajectory) -> Trajectory:
        """
        Transform all points in a Trajectory instance to floor coordinates.
        Preserves original image_x, image_y while setting floor_x, floor_y.
        """
        if not trajectory.points:
            return trajectory

        img_pts = np.array([[p.image_x, p.image_y] for p in trajectory.points], dtype=np.float32)
        floor_pts = self.transform_points(img_pts)

        transformed_points = []
        for orig_pt, f_pt in zip(trajectory.points, floor_pts):
            t_pt = TrajectoryPoint(
                track_id=orig_pt.track_id,
                frame_idx=orig_pt.frame_idx,
                timestamp_sec=orig_pt.timestamp_sec,
                image_x=orig_pt.image_x,
                image_y=orig_pt.image_y,
                floor_x=round(float(f_pt[0]), 2),
                floor_y=round(float(f_pt[1]), 2)
            )
            transformed_points.append(t_pt)

        return Trajectory(
            track_id=trajectory.track_id,
            points=transformed_points,
            class_id=trajectory.class_id,
            class_name=trajectory.class_name
        )

    def transform_trajectories(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """Transform a list of Trajectory instances."""
        return [self.transform_trajectory(t) for t in trajectories]

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any], dataset_name: str) -> "HomographyCalibrator":
        """Factory method instantiating calibrator from configuration dictionary."""
        cfg = config_dict.get("homography", {}).get(dataset_name)
        if not cfg:
            raise KeyError(f"No homography calibration found in config for dataset: {dataset_name}")

        unit = cfg.get("unit", "metres")
        img_pts = cfg.get("image_points")
        flr_pts = cfg.get("floor_points")
        check_cfg = cfg.get("check_points", {})
        check_img = check_cfg.get("image") if check_cfg else None
        check_flr = check_cfg.get("expected_floor") if check_cfg else None

        return cls(
            image_points=img_pts,
            floor_points=flr_pts,
            check_points_image=check_img,
            check_points_floor=check_flr,
            unit=unit,
            dataset_name=dataset_name
        )


# ============================================================
# Visualization Helpers
# ============================================================

def visualize_correspondences(
    img_bgr: np.ndarray,
    calibrator: HomographyCalibrator
) -> np.ndarray:
    """Draw calibration correspondence points and check points on camera image."""
    annotated = img_bgr.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    # Draw training points
    for i, pt in enumerate(calibrator.src_pts):
        x, y = int(pt[0]), int(pt[1])
        c = colors[i % len(colors)]
        cv2.circle(annotated, (x, y), 8, c, -1)
        label = f"P{i+1}"
        cv2.putText(annotated, label, (x + 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    # Draw check points
    if calibrator.check_src is not None:
        for j, pt in enumerate(calibrator.check_src):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(annotated, (x, y), 8, (255, 128, 0), -1)
            cv2.putText(annotated, f"CP{j+1}", (x + 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

    return annotated


def visualize_floor_trajectories(
    trajectories: List[Trajectory],
    grid_size: Tuple[int, int] = (600, 600),
    floor_bounds: Tuple[float, float, float, float] = (-2.0, 12.0, -2.0, 12.0),
    title: str = "Top-Down Floor Map Trajectories"
) -> np.ndarray:
    """
    Render top-down floor-plane trajectories on a 2D grid map.
    
    floor_bounds: (min_x, max_x, min_y, max_y) in floor units.
    """
    canvas_w, canvas_h = grid_size
    min_x, max_x, min_y, max_y = floor_bounds

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245

    # Draw grid lines
    scale_x = canvas_w / (max_x - min_x)
    scale_y = canvas_h / (max_y - min_y)

    # Draw grid
    for x_val in range(int(np.floor(min_x)), int(np.ceil(max_x)) + 1, 2):
        px = int((x_val - min_x) * scale_x)
        if 0 <= px < canvas_w:
            cv2.line(canvas, (px, 0), (px, canvas_h), (220, 220, 220), 1)
            cv2.putText(canvas, f"{x_val}m", (px + 2, canvas_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    for y_val in range(int(np.floor(min_y)), int(np.ceil(max_y)) + 1, 2):
        py = int((max_y - y_val) * scale_y)
        if 0 <= py < canvas_h:
            cv2.line(canvas, (0, py), (canvas_w, py), (220, 220, 220), 1)
            cv2.putText(canvas, f"{y_val}m", (5, py - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    cv2.putText(canvas, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    palette = [
        (255, 0, 0), (0, 165, 255), (0, 255, 0), (255, 0, 255),
        (0, 0, 255), (128, 0, 128), (0, 128, 128), (128, 128, 0)
    ]

    for idx, traj in enumerate(trajectories):
        flr_pts = traj.get_floor_points()
        if len(flr_pts) < 1:
            continue

        color = palette[traj.track_id % len(palette)]
        canvas_pts = []
        for fx, fy in flr_pts:
            px = int((fx - min_x) * scale_x)
            py = int((max_y - fy) * scale_y)
            canvas_pts.append((px, py))

        # Draw trajectory lines
        for i in range(1, len(canvas_pts)):
            cv2.line(canvas, canvas_pts[i-1], canvas_pts[i], color, 2)

        # Draw trajectory endpoints
        cv2.circle(canvas, canvas_pts[0], 5, (0, 255, 0), -1)  # Green start dot
        cv2.circle(canvas, canvas_pts[-1], 6, color, -1)        # Current end dot
        cv2.putText(canvas, f"ID:{traj.track_id}", (canvas_pts[-1][0] + 5, canvas_pts[-1][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return canvas
