"""
Multi-Object Tracking Package
=============================
Provides ByteTrack multi-object tracking modules and track containers.
"""

from src.tracking.bytetrack import (
    ByteTracker,
    Track,
    TrackPoint,
    TrackState,
    TrackingResult,
    compute_iou_batch
)
from src.tracking.kalman_filter import KalmanFilter

__all__ = [
    "ByteTracker",
    "Track",
    "TrackPoint",
    "TrackState",
    "TrackingResult",
    "compute_iou_batch",
    "KalmanFilter",
]
