"""
Unit & Integration Tests for ByteTrack Tracker Module
=====================================================
Tests tracker initialization, zero detections, track creation, track persistence,
IoU matching, track history, bottom-center trajectory point derivation, and missing detection recovery.
"""

import pytest
import numpy as np
from src.detection.postprocess import Detection
from src.tracking.bytetrack import (
    ByteTracker,
    Track,
    TrackPoint,
    TrackState,
    TrackingResult,
    compute_iou_batch
)
from src.tracking.kalman_filter import KalmanFilter


@pytest.fixture(autouse=True)
def reset_track_counter():
    """Reset Track ID counter before each test to ensure deterministic IDs."""
    Track.reset_id_counter(0)


# ============================================================
# 1. Tracker Initialization Tests
# ============================================================

def test_tracker_initialization():
    tracker = ByteTracker(track_high_thresh=0.50, track_low_thresh=0.10)
    assert tracker.track_high_thresh == 0.50
    assert tracker.track_low_thresh == 0.10
    assert len(tracker.tracked_tracks) == 0
    assert len(tracker.lost_tracks) == 0


def test_iou_batch_math():
    boxes1 = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    boxes2 = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    
    iou_matrix = compute_iou_batch(boxes1, boxes2)
    assert iou_matrix.shape == (2, 2)
    assert iou_matrix[0, 0] == pytest.approx(1.0)
    assert iou_matrix[0, 1] == pytest.approx(0.0)
    assert iou_matrix[1, 0] == pytest.approx(0.0)


# ============================================================
# 2. Empty Detections / Zero Detections Test
# ============================================================

def test_empty_detections_input():
    tracker = ByteTracker()
    res = tracker.update(detections=[], frame_id=1, timestamp_sec=0.0)
    
    assert isinstance(res, TrackingResult)
    assert res.num_active_tracks == 0
    assert res.total_tracks_created == 0
    assert res.frame_id == 1


# ============================================================
# 3. Track Creation & Single Person Across Frames
# ============================================================

def test_single_person_track_creation_and_persistence():
    tracker = ByteTracker(track_high_thresh=0.50)
    
    det1 = [Detection(bbox=[100.0, 100.0, 150.0, 200.0], confidence=0.85)]
    res1 = tracker.update(det1, frame_id=1, timestamp_sec=0.0)
    
    assert res1.num_active_tracks == 1
    assert res1.total_tracks_created == 1
    track = res1.active_tracks[0]
    assigned_id = track.track_id
    assert track.track_id > 0
    assert track.bottom_center == [125.0, 200.0]
    
    # Frame 2: Person moves slightly
    det2 = [Detection(bbox=[104.0, 102.0, 154.0, 202.0], confidence=0.88)]
    res2 = tracker.update(det2, frame_id=2, timestamp_sec=0.2)
    
    assert res2.num_active_tracks == 1
    assert res2.active_tracks[0].track_id == assigned_id
    assert len(res2.active_tracks[0].history) == 2


# ============================================================
# 4. Multiple People Tracking Test
# ============================================================

def test_multiple_people_tracking():
    tracker = ByteTracker()
    
    dets_frame1 = [
        Detection(bbox=[10.0, 10.0, 50.0, 100.0], confidence=0.90),
        Detection(bbox=[300.0, 300.0, 350.0, 400.0], confidence=0.85),
    ]
    res1 = tracker.update(dets_frame1, frame_id=1, timestamp_sec=0.0)
    assert res1.num_active_tracks == 2
    ids_f1 = {t.track_id for t in res1.active_tracks}
    
    # Frame 2: Both people move slightly
    dets_frame2 = [
        Detection(bbox=[12.0, 11.0, 52.0, 101.0], confidence=0.91),
        Detection(bbox=[305.0, 302.0, 355.0, 402.0], confidence=0.86),
    ]
    res2 = tracker.update(dets_frame2, frame_id=2, timestamp_sec=0.2)
    assert res2.num_active_tracks == 2
    ids_f2 = {t.track_id for t in res2.active_tracks}
    
    # Track IDs should persist
    assert ids_f1 == ids_f2


# ============================================================
# 5. Temporary Missing Detection & Recovery Test
# ============================================================

def test_temporary_missing_detection():
    tracker = ByteTracker(max_time_lost_sec=1.0)
    
    # Frame 1: Person detected
    det1 = [Detection(bbox=[50.0, 50.0, 90.0, 150.0], confidence=0.80)]
    res1 = tracker.update(det1, frame_id=1, timestamp_sec=0.0)
    track_id = res1.active_tracks[0].track_id
    
    # Frame 2: Temporary missing detection (occlusion)
    res2 = tracker.update([], frame_id=2, timestamp_sec=0.2)
    assert res2.num_active_tracks == 0
    assert res2.num_lost_tracks == 1
    
    # Frame 3: Detection recovers nearby
    det3 = [Detection(bbox=[52.0, 51.0, 92.0, 151.0], confidence=0.82)]
    res3 = tracker.update(det3, frame_id=3, timestamp_sec=0.4)
    assert res3.num_active_tracks == 1
    assert res3.active_tracks[0].track_id == track_id  # Recovered ID!


def test_track_removal_after_max_time_lost():
    tracker = ByteTracker(max_time_lost_sec=0.5)
    
    # Frame 1: Detected
    det1 = [Detection(bbox=[50.0, 50.0, 90.0, 150.0], confidence=0.80)]
    tracker.update(det1, frame_id=1, timestamp_sec=0.0)
    
    # Frame 2: Missing at t=0.2s -> Lost
    res2 = tracker.update([], frame_id=2, timestamp_sec=0.2)
    assert res2.num_lost_tracks == 1
    
    # Frame 3: Missing at t=1.0s (> 0.5s max lost time) -> Removed
    res3 = tracker.update([], frame_id=3, timestamp_sec=1.0)
    assert res3.num_active_tracks == 0
    assert res3.num_lost_tracks == 0
    assert len(tracker.removed_tracks) == 1


# ============================================================
# 6. Track History & Trajectory Point Verification
# ============================================================

def test_track_history_and_bottom_center():
    track = Track(bbox=[100.0, 100.0, 200.0, 300.0], confidence=0.90, frame_id=1, timestamp_sec=0.0)
    
    assert track.bottom_center == [150.0, 300.0]
    assert len(track.history) == 1
    assert track.history[0].bottom_center == [150.0, 300.0]
    assert track.history[0].frame_id == 1
    assert track.history[0].timestamp_sec == 0.0

    # Update with new detection
    det = Detection(bbox=[110.0, 105.0, 210.0, 305.0], confidence=0.92)
    track.update(det, frame_id=2, timestamp_sec=0.2)
    
    assert len(track.history) == 2
    assert track.history[1].bottom_center == [160.0, 305.0]
    assert track.history[1].frame_id == 2
