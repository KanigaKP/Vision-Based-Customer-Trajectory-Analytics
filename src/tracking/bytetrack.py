"""
ByteTrack Multi-Object Tracker (ONNX Detection Interface)
=========================================================
Implements the ByteTrack algorithm (Zhang et al., ECCV 2022) using Kalman Filtering
and 2-stage Hungarian matching (scipy.optimize.linear_sum_assignment).

IMPORTANT METHODOLOGICAL BOUNDARY:
Track IDs are TEMPORARY SESSION-LOCAL IDENTIFIERS.
They represent continuous visual movement within a single camera session.
They are NOT biometric, facial, or cross-session customer identities.
NO re-identification (Re-ID) or facial recognition is performed.
"""

from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import scipy.optimize
import time
import cv2

from src.detection.postprocess import Detection, DetectionResult
from src.tracking.kalman_filter import KalmanFilter


class TrackState(Enum):
    """Lifecycle states for a tracked object."""
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class TrackPoint:
    """Individual trajectory point in image-space coordinates."""
    frame_id: int
    timestamp_sec: float
    bbox: List[float]             # [x1, y1, x2, y2]
    bottom_center: List[float]    # [x_center, y_bottom] -> [(x1+x2)/2, y2]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Track:
    """Represents a single session-local tracked individual over time."""

    _count = 0

    def __init__(
        self,
        bbox: List[float],
        confidence: float,
        frame_id: int,
        timestamp_sec: float,
        class_id: int = 0,
        class_name: str = "person"
    ):
        Track._count += 1
        self.track_id: int = Track._count
        self.bbox: List[float] = [float(v) for v in bbox]
        self.confidence: float = float(confidence)
        self.class_id: int = class_id
        self.class_name: str = class_name
        self.state: TrackState = TrackState.TRACKED
        
        self.start_frame: int = frame_id
        self.start_timestamp_sec: float = timestamp_sec
        self.last_seen_frame: int = frame_id
        self.last_seen_timestamp_sec: float = timestamp_sec
        
        self.hits: int = 1
        self.time_since_update: int = 0
        
        # Kalman Filter state
        self.kalman = KalmanFilter()
        z = self.kalman.box_to_z(np.array(self.bbox, dtype=np.float32))
        self.mean, self.covariance = self.kalman.initiate(z)

        # In-memory trajectory history
        self.history: List[TrackPoint] = []
        self._add_point(frame_id, timestamp_sec, self.bbox, self.confidence)

    @classmethod
    def reset_id_counter(cls, start_id: int = 0):
        """Reset internal track ID counter for fresh test runs."""
        cls._count = start_id

    @property
    def bottom_center(self) -> List[float]:
        """Derive bottom-center point [x_center, y_bottom] from bounding box."""
        x1, y1, x2, y2 = self.bbox
        x_center = round((x1 + x2) / 2.0, 2)
        y_bottom = round(y2, 2)
        return [x_center, y_bottom]

    def _add_point(self, frame_id: int, timestamp_sec: float, bbox: List[float], confidence: float):
        point = TrackPoint(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            bbox=[round(float(v), 2) for v in bbox],
            bottom_center=self.bottom_center,
            confidence=round(float(confidence), 4)
        )
        self.history.append(point)

    def predict(self, dt: float = 1.0):
        """Predict next bounding box position using Kalman filter."""
        self.mean, self.covariance = self.kalman.predict(self.mean, self.covariance, dt=dt)
        # Update current estimated bbox from mean
        predicted_box = self.kalman.z_to_box(self.mean)
        self.bbox = [round(float(v), 2) for v in predicted_box]

    def update(self, detection: Detection, frame_id: int, timestamp_sec: float):
        """Update track state with a new matched detection."""
        self.bbox = [float(v) for v in detection.bbox]
        self.confidence = float(detection.confidence)
        self.last_seen_frame = frame_id
        self.last_seen_timestamp_sec = timestamp_sec
        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.TRACKED

        z = self.kalman.box_to_z(np.array(self.bbox, dtype=np.float32))
        self.mean, self.covariance = self.kalman.update(self.mean, self.covariance, z)

        self._add_point(frame_id, timestamp_sec, self.bbox, self.confidence)

    def mark_missed(self, frame_id: int, timestamp_sec: float):
        """Mark track as missed in current frame."""
        self.time_since_update += 1
        self.state = TrackState.LOST

    def mark_removed(self):
        """Mark track as permanently removed."""
        self.state = TrackState.REMOVED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "bottom_center": self.bottom_center,
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "state": self.state.name,
            "start_frame": self.start_frame,
            "start_timestamp_sec": self.start_timestamp_sec,
            "last_seen_frame": self.last_seen_frame,
            "last_seen_timestamp_sec": self.last_seen_timestamp_sec,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "history_length": len(self.history)
        }


@dataclass
class TrackingResult:
    """Container for tracking outputs for a single frame."""
    frame_id: int
    timestamp_sec: float
    active_tracks: List[Track]
    total_tracks_created: int
    num_active_tracks: int
    num_lost_tracks: int
    detector_time_ms: float
    tracker_time_ms: float
    total_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_sec": self.timestamp_sec,
            "total_tracks_created": self.total_tracks_created,
            "num_active_tracks": self.num_active_tracks,
            "num_lost_tracks": self.num_lost_tracks,
            "detector_time_ms": self.detector_time_ms,
            "tracker_time_ms": self.tracker_time_ms,
            "total_time_ms": self.total_time_ms,
            "tracks": [t.to_dict() for t in self.active_tracks]
        }


def compute_iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Intersection-over-Union (IoU) between two sets of boxes [x1, y1, x2, y2].
    Returns matrix of shape (len(boxes1), len(boxes2)).
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    b1 = np.expand_dims(boxes1, axis=1)  # (N, 1, 4)
    b2 = np.expand_dims(boxes2, axis=0)  # (1, M, 4)

    inter_x1 = np.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = np.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = np.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = np.minimum(b1[..., 3], b2[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])

    union = area1 + area2 - inter_area
    iou = np.where(union > 0, inter_area / union, 0.0)
    return iou.astype(np.float32)


class ByteTracker:
    """
    ByteTrack Multi-Object Tracker.
    Associates detections across temporal frames using 2-stage IoU-Hungarian bipartite matching.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.50,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.50,
        match_thresh: float = 0.80,         # Max IoU distance (1 - IoU) for 1st match
        low_match_thresh: float = 0.50,     # Max IoU distance (1 - IoU) for 2nd match
        max_time_lost_sec: float = 2.0
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.low_match_thresh = low_match_thresh
        self.max_time_lost_sec = max_time_lost_sec

        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []

    def update(
        self,
        detections: List[Detection],
        frame_id: int,
        timestamp_sec: float,
        detector_time_ms: float = 0.0
    ) -> TrackingResult:
        """
        Process detections for current frame and update active tracks using ByteTrack algorithm.
        """
        t0 = time.time()

        # Step 1: Split detections into high-confidence and low-confidence sets
        det_high: List[Detection] = []
        det_low: List[Detection] = []

        for d in detections:
            if d.confidence >= self.track_high_thresh:
                det_high.append(d)
            elif d.confidence >= self.track_low_thresh:
                det_low.append(d)

        # Step 2: Predict new positions for all existing active/lost tracks using Kalman filter
        all_candidate_tracks = self.tracked_tracks + self.lost_tracks
        for track in all_candidate_tracks:
            dt = 1.0  # Time step index
            if len(track.history) > 1:
                last_t = track.history[-1].timestamp_sec
                prev_t = track.history[-2].timestamp_sec
                if last_t > prev_t:
                    dt = timestamp_sec - last_t
            track.predict(dt=dt)

        # Step 3: First Association — High-Confidence Detections with Active + Lost Tracks
        unmatched_tracks_1: List[Track] = []
        unmatched_det_high: List[Detection] = []

        if len(all_candidate_tracks) > 0 and len(det_high) > 0:
            track_boxes = np.array([t.bbox for t in all_candidate_tracks], dtype=np.float32)
            det_boxes = np.array([d.bbox for d in det_high], dtype=np.float32)

            iou_matrix = compute_iou_batch(track_boxes, det_boxes)
            cost_matrix = 1.0 - iou_matrix

            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

            matched_track_indices = set()
            matched_det_indices = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= self.match_thresh:
                    matched_track = all_candidate_tracks[r]
                    matched_track.update(det_high[c], frame_id, timestamp_sec)
                    matched_track_indices.add(r)
                    matched_det_indices.add(c)
                    
                    # If this was a lost track, recover it to tracked_tracks
                    if matched_track in self.lost_tracks:
                        self.lost_tracks.remove(matched_track)
                    if matched_track not in self.tracked_tracks:
                        self.tracked_tracks.append(matched_track)

            for i, track in enumerate(all_candidate_tracks):
                if i not in matched_track_indices:
                    unmatched_tracks_1.append(track)

            for j, d in enumerate(det_high):
                if j not in matched_det_indices:
                    unmatched_det_high.append(d)
        else:
            unmatched_tracks_1 = list(all_candidate_tracks)
            unmatched_det_high = list(det_high)

        # Step 4: Second Association — Low-Confidence Detections with Unmatched Tracks
        unmatched_tracks_2: List[Track] = []

        if len(unmatched_tracks_1) > 0 and len(det_low) > 0:
            track_boxes = np.array([t.bbox for t in unmatched_tracks_1], dtype=np.float32)
            det_boxes = np.array([d.bbox for d in det_low], dtype=np.float32)

            iou_matrix = compute_iou_batch(track_boxes, det_boxes)
            cost_matrix = 1.0 - iou_matrix

            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

            matched_track_indices = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= self.low_match_thresh:
                    matched_track = unmatched_tracks_1[r]
                    matched_track.update(det_low[c], frame_id, timestamp_sec)
                    matched_track_indices.add(r)
                    
                    if matched_track in self.lost_tracks:
                        self.lost_tracks.remove(matched_track)
                    if matched_track not in self.tracked_tracks:
                        self.tracked_tracks.append(matched_track)

            for i, track in enumerate(unmatched_tracks_1):
                if i not in matched_track_indices:
                    unmatched_tracks_2.append(track)
        else:
            unmatched_tracks_2 = list(unmatched_tracks_1)

        # Step 5: Handle unmatched tracks -> Mark as LOST
        for track in unmatched_tracks_2:
            track.mark_missed(frame_id, timestamp_sec)
            if track in self.tracked_tracks:
                self.tracked_tracks.remove(track)
            if track not in self.lost_tracks:
                self.lost_tracks.append(track)

        # Step 6: Initialize new tracks for unmatched high-confidence detections
        for d in unmatched_det_high:
            if d.confidence >= self.new_track_thresh:
                new_track = Track(
                    bbox=d.bbox,
                    confidence=d.confidence,
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    class_id=d.class_id,
                    class_name=d.class_name
                )
                self.tracked_tracks.append(new_track)

        # Step 7: Clean up lost tracks exceeding max_time_lost_sec
        active_lost: List[Track] = []
        for track in self.lost_tracks:
            time_lost = timestamp_sec - track.last_seen_timestamp_sec
            if time_lost > self.max_time_lost_sec:
                track.mark_removed()
                self.removed_tracks.append(track)
            else:
                active_lost.append(track)
        self.lost_tracks = active_lost

        tracker_time_ms = (time.time() - t0) * 1000.0
        total_time_ms = detector_time_ms + tracker_time_ms

        return TrackingResult(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            active_tracks=list(self.tracked_tracks),
            total_tracks_created=Track._count,
            num_active_tracks=len(self.tracked_tracks),
            num_lost_tracks=len(self.lost_tracks),
            detector_time_ms=round(detector_time_ms, 2),
            tracker_time_ms=round(tracker_time_ms, 2),
            total_time_ms=round(total_time_ms, 2)
        )

    @staticmethod
    def annotate_tracks(
        img_bgr: np.ndarray,
        tracks: List[Track],
        box_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Annotate image with Track IDs and bounding boxes for visual verification."""
        annotated = img_bgr.copy()
        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
            
            label = f"ID:{track.track_id} ({track.confidence:.2f})"
            (w_label, h_label), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            y_text = max(y1, h_label + 10)
            cv2.rectangle(
                annotated,
                (x1, y_text - h_label - 5),
                (x1 + w_label + 10, y_text + baseline),
                box_color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y_text - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2
            )
            
            # Draw bottom-center point marker (representative trajectory point)
            cx, cy = [int(v) for v in track.bottom_center]
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            
            # Draw brief trajectory line from recent history
            if len(track.history) > 1:
                pts = [tuple(int(v) for v in pt.bottom_center) for pt in track.history[-10:]]
                for i in range(1, len(pts)):
                    cv2.line(annotated, pts[i - 1], pts[i], (0, 255, 255), 2)
                    
        return annotated
