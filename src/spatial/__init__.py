"""
Spatial Processing Package
==========================
Provides homography calibration, point and trajectory transformations, and floor map rendering.
"""

from src.spatial.homography import (
    HomographyCalibrator,
    HomographyDiagnostics,
    visualize_correspondences,
    visualize_floor_trajectories
)

__all__ = [
    "HomographyCalibrator",
    "HomographyDiagnostics",
    "visualize_correspondences",
    "visualize_floor_trajectories",
]
