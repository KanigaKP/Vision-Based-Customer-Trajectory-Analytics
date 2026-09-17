"""
Kalman Filter for Bounding Box Tracking
======================================
8-State Kalman Filter [x, y, a, h, vx, vy, va, vh] for 2D bounding box tracking.
x, y: bounding box center
a: aspect ratio (width / height)
h: bounding box height
vx, vy, va, vh: state velocities
"""

from typing import Tuple
import numpy as np


class KalmanFilter:
    """8-state Kalman Filter for bounding box trajectory prediction."""

    def __init__(self):
        # 8-state vector [x, y, a, h, vx, vy, va, vh]
        self._motion_mat = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0

        self._update_mat = np.eye(4, 8, dtype=np.float32)

        # Standard deviation weights
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track state mean and covariance from 4D measurement [x, y, a, h].
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict state mean and covariance for next time step dt.
        """
        motion_mat = np.eye(8, dtype=np.float32)
        for i in range(4):
            motion_mat[i, i + 4] = dt

        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)

        mean = np.dot(motion_mat, mean)
        covariance = np.linalg.multi_dot([motion_mat, covariance, motion_mat.T]) + motion_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update state mean and covariance with measurement [x, y, a, h].
        """
        std = [
            self._std_weight_position * measurement[3],
            self._std_weight_position * measurement[3],
            1e-1,
            self._std_weight_position * measurement[3],
        ]
        innovation_cov = np.diag(np.square(std)).astype(np.float32)

        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot([self._update_mat, covariance, self._update_mat.T]) + innovation_cov

        kalman_gain = np.linalg.multi_dot([
            covariance,
            self._update_mat.T,
            np.linalg.inv(projected_cov)
        ])

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot([kalman_gain, projected_cov, kalman_gain.T])
        return new_mean, new_covariance

    @staticmethod
    def box_to_z(bbox: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to measurement vector z = [x, y, a, h]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        a = w / (h + 1e-6)
        return np.array([cx, cy, a, h], dtype=np.float32)

    @staticmethod
    def z_to_box(mean: np.ndarray) -> np.ndarray:
        """Convert state mean [x, y, a, h, ...] to [x1, y1, x2, y2]."""
        cx, cy, a, h = mean[:4]
        w = a * h
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)
