# Phase 4 Report — ByteTrack Multi-Object Tracking Integration

**Date:** 2026-09-17  
**Status:** ✅ PASS

---

## 1. Objective

Integrate **ByteTrack Multi-Object Tracking** (`src/tracking/bytetrack.py`) with the existing YOLO11 ONNX person detector (`src/detection/detector.py`). The module associates person detections across consecutive temporal frames, assigns temporary session-local track IDs, maintains an in-memory trajectory history, and calculates representative image-space bottom-center reference points $[(x_1 + x_2)/2, y_2]$ for downstream trajectory processing.

---

## 2. Project Definition & Track ID Scope Boundary

> **CRITICAL ARCHITECTURAL BOUNDARY:**  
> Track IDs assigned by ByteTrack are **TEMPORARY SESSION-LOCAL IDENTIFIERS** valid only within a single continuous video session.  
> - Track IDs are **NOT** facial identities, biometric identifiers, cross-session customer IDs, or persistent database identities.  
> - **No re-identification (Re-ID) system**, facial recognition, or appearance-based cross-camera matching is implemented.  
> - If an individual exits the camera frame and re-enters after their track has been removed, the system naturally assigns a new session-local track ID.

---

## 3. ByteTrack Architecture

The tracking layer is organized under `src/tracking/`:

```
src/tracking/
├── __init__.py          # Package exports
├── bytetrack.py         # ByteTracker, Track, TrackPoint, TrackingResult
└── kalman_filter.py     # 8-state Kalman Filter (x, y, a, h, vx, vy, va, vh)
```

### Pipeline Flow

```
Input Frame (BGR Image)
        │
        ├──> YOLO11PersonDetector.detect()
        │       └── Outputs List[Detection] (bbox [x1, y1, x2, y2], confidence)
        │
        └──> ByteTracker.update(detections, frame_id, timestamp_sec)
                │
                ├── 1. Split Detections -> High-Conf (>= 0.50) & Low-Conf (0.10..0.50)
                ├── 2. Kalman Filter State Prediction for Active/Lost Tracks
                ├── 3. 1st Bipartite Match: High-Conf Detections x Active+Lost Tracks (IoU Distance)
                │       └── scipy.optimize.linear_sum_assignment (Hungarian Match)
                ├── 4. 2nd Bipartite Match: Low-Conf Detections x Unmatched Active Tracks (IoU Distance)
                ├── 5. Update Matched Tracks (Kalman measurement update & history append)
                ├── 6. Mark Unmatched Tracks as LOST (remove if time_lost > max_time_lost_sec)
                ├── 7. Initialize New Tracks for Unmatched High-Conf Detections
                │
                └── Outputs TrackingResult Container
                        ├── Active Tracks (track_id, bbox, confidence, bottom_center)
                        └── In-Memory Trajectory History (List[TrackPoint])
```

---

## 4. Detector-to-Tracker Interface

The tracker accepts standard `Detection` objects produced by `YOLO11PersonDetector`:

```python
# Input Interface: List[Detection]
detector_result = detector.detect(frame, frame_id=idx, timestamp_sec=ts)

# Output Interface: TrackingResult
tracking_result = tracker.update(
    detections=detector_result.detections,
    frame_id=idx,
    timestamp_sec=ts,
    detector_time_ms=detector_result.total_time_ms
)
```

Each active track in `tracking_result.active_tracks` exposes:
- `track_id`: Integer ID (1, 2, 3, …)
- `bbox`: `[x1, y1, x2, y2]` in original image pixels
- `confidence`: Detection score float
- `class_id`: `0` (`person`)
- `bottom_center`: `[(x1 + x2) / 2.0, y2]`
- `history`: List of `TrackPoint` entries

---

## 5. Track History & Representative Trajectory Point

For each active track, the system maintains an in-memory trajectory history (`track.history` containing `TrackPoint` instances).

Each `TrackPoint` records:
- `frame_id`: Frame index
- `timestamp_sec`: Temporal timestamp
- `bbox`: `[x1, y1, x2, y2]`
- `bottom_center`: `[x_center, y_bottom]` where $x_{\text{center}} = \frac{x_1 + x_2}{2}$ and $y_{\text{bottom}} = y_2$
- `confidence`: Confidence score

> **TRAJECTORY NOTE:**  
> The bottom-center point $[(x_1 + x_2)/2, y_2]$ represents the person's feet contact location on the ground plane in image-space coordinates. This point will be consumed by the homography/floor mapping phase in future work.

---

## 6. Frame Sampling Relationship

- The detector and tracker process sampled frames at a target rate of **5.0 FPS** ($\Delta t = 0.2\text{s}$).
- The `ByteTracker` receives frames in exact temporal sequence, updating Kalman Filter predictions using actual elapsed time ($\Delta t = t_{\text{curr}} - t_{\text{prev}}$).
- Lost track timeout `max_time_lost_sec` is set to `2.0` seconds in `config.yaml`, allowing tracks to survive up to ~10 consecutive sampled frame misses during temporary occlusion.

---

## 7. MOT17 Tracking Demonstration Results

- **Dataset:** `MOT17-04-FRCNN` consecutive sequence (10 frames: `000001.jpg` to `000010.jpg`).
- **Results:**
  - Track IDs assigned and persisted cleanly across frames (9 to 10 active tracks per frame).
  - Total tracks created across 10 frames: **11 tracks total** (demonstrating high identity persistence).
  - Saved visualization artifacts:
    - `outputs/reports/phase_04_mot17_tracking_frame_01.jpg`
    - `outputs/reports/phase_04_mot17_tracking_frame_05.jpg`
    - `outputs/reports/phase_04_mot17_tracking_frame_10.jpg`

---

## 8. MERL Qualitative Demonstration

- **Video:** `data/merl/10_1_crop.mp4` (20 sampled frames at 5.0 FPS).
- **Saved Visual Samples:**
  - `outputs/reports/phase_04_merl_tracking_sample_1_frame0.jpg`
  - `outputs/reports/phase_04_merl_tracking_sample_2_frame6.jpg`
  - `outputs/reports/phase_04_merl_tracking_sample_3_frame12.jpg`

> **QUALITATIVE DEMONSTRATION DISCLAIMER:**  
> MERL provides fine-grained action labels (*Reach/Retract/Inspect*), **NOT** continuous multi-object tracking ground truth.  
> **No tracking accuracy metrics (MOTA/IDF1) were calculated on MERL.** MERL is used exclusively for qualitative visualization.

---

## 9. Performance Measurements

Measurements performed on AMD Ryzen 5 7520U CPU (4C/8T):

| Metric | Measured Value |
|---|---|
| **Average Detector Latency** | **84.68 ms** |
| **Average ByteTrack Overhead** | **0.50 ms** |
| **Combined Detection + Tracking Latency** | **85.18 ms** |
| **Model Inference Speed** | **11.81 FPS** |
| **Combined Pipeline Speed** | **11.74 FPS** |

> **PERFORMANCE OBSERVATION:**  
> The ByteTrack tracking algorithm adds negligible computational overhead (**0.50 ms** per frame), maintaining combined detection + tracking speed at ~11.7 FPS on CPU.

---

## 10. Automated Tests Summary

Automated test suite implemented in `tests/test_tracking.py`:

| Test Name | Verified Behavior | Status |
|---|---|---|
| `test_tracker_initialization` | Configurable thresholds, empty initial lists | ✅ PASSED |
| `test_iou_batch_math` | Matrix IoU math $[1 - \text{IoU}]$ | ✅ PASSED |
| `test_empty_detections_input` | Zero detections handled gracefully without errors | ✅ PASSED |
| `test_single_person_track_creation_and_persistence` | Track ID assignment and persistence across frames | ✅ PASSED |
| `test_multiple_people_tracking` | Multiple simultaneous tracks maintained without ID swaps | ✅ PASSED |
| `test_temporary_missing_detection` | Track recovers after temporary 1-frame occlusion | ✅ PASSED |
| `test_track_removal_after_max_time_lost` | Track removed after exceeding `max_time_lost_sec` | ✅ PASSED |
| `test_track_history_and_bottom_center` | History points and $[(x_1+x_2)/2, y_2]$ math | ✅ PASSED |

**Full Test Suite Execution (`pytest -v`):**  
**27 passed in 2.88s** (10 video preprocessing + 9 detection + 8 tracking tests).

---

## 11. Files & Modules Created / Modified

| File Path | Description |
|---|---|
| `src/tracking/__init__.py` | Package exports for `ByteTracker`, `Track`, `TrackPoint`, `TrackingResult` |
| `src/tracking/bytetrack.py` | ByteTrack tracker implementation with 2-stage IoU-Hungarian matching |
| `src/tracking/kalman_filter.py` | 8-state Kalman Filter for bounding box motion prediction |
| `scripts/verify_phase_04.py` | Benchmark & visualization script for MOT17 and MERL tracking |
| `tests/test_tracking.py` | Suite of 8 automated unit/integration tests for ByteTrack |
| `config.yaml` | Updated with `tracking` section parameters |
| `requirements.txt` | Updated with `scipy` dependency |
| `docs/PHASE_04_REPORT.md` | This completion report |
| `docs/DECISIONS.md` | Updated with DEC-016 |

---

## 12. Decisions Made

- **DEC-016:** Implemented ByteTrack multi-object tracking (`src/tracking/bytetrack.py`) using 2-stage IoU-Hungarian bipartite matching (`scipy.optimize.linear_sum_assignment`) and 8-state Kalman filtering, defining track IDs strictly as temporary session-local identifiers. See [`docs/DECISIONS.md`](DECISIONS.md) for full entry.

---

## 13. Limitations

- **Session-Local Scope:** Track IDs reset between sessions. No cross-session re-identification is performed.
- **Occlusion Limits:** If an object remains hidden longer than `max_time_lost_sec` (2.0s), its track is removed and a new track ID will be assigned upon re-emergence.

---

## 14. Phase 4 Acceptance Criteria Checklist

| # | Criterion | Status |
|---|---|---|
| 1 | ByteTrack integrated into codebase | ✅ PASS |
| 2 | Existing YOLO11 detection output feeds tracker | ✅ PASS |
| 3 | Temporary session-local Track IDs assigned | ✅ PASS |
| 4 | Track IDs persist across consecutive sampled frames | ✅ PASS |
| 5 | Multiple people tracked simultaneously | ✅ PASS |
| 6 | Track history recorded in memory | ✅ PASS |
| 7 | Bottom-center image coordinates available | ✅ PASS |
| 8 | MERL tracking treated strictly as qualitative | ✅ PASS |
| 9 | MOT17 prepared for later tracking evaluation | ✅ PASS |
| 10| Empty/missing detections handled gracefully | ✅ PASS |
| 11| Existing tests (Phases 0–3) remain passing | ✅ PASS |
| 12| New tracking tests pass (`27/27 pytest passed`) | ✅ PASS |
| 13| Actual tracking latency measured (0.50 ms overhead) | ✅ PASS |
| 14| No unsupported tracking accuracy claims made | ✅ PASS |

---

## 15. Recommendation for Phase 5

**Phase 4 is COMPLETE with status PASS.**

The project is ready to proceed to **Phase 5: Trajectory Extraction & Homography Floor-Plane Mapping**, transforming accumulated image-space bottom-center trajectory points into real-world floor-plane spatial coordinates.
