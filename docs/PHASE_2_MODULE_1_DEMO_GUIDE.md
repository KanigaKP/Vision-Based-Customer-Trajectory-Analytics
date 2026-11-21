# Phase-2 Review: Module-1 Detection & Tracking Demo Evidence Guide

**Target Document:** `docs/PHASE_2_MODULE_1_DEMO_GUIDE.md`  
**Purpose:** Presentation and review evidence guide for the Module-1 (YOLO11 Person Detection + ByteTrack Multi-Object Tracking) demonstration.  
**Scope:** Engineering review and viva/presentation evidence organization only. No new functionality, algorithm redesign, or feature expansion.

---

## 1. Demo Purpose & Architectural Boundary

### 1.1 Purpose
The Module-1 demonstration showcases the operational integration of:
1. **YOLO11 ONNX Person Detection** (`src/detection/`): Extracting person bounding boxes, confidence scores, and class labels from video frames using ONNX Runtime on CPU.
2. **ByteTrack Multi-Object Tracking** (`src/tracking/`): Associating temporal detections using a two-stage Hungarian matching algorithm with Kalman filter motion prediction to maintain clip-local track continuity and calculate image-space bottom-center contact points.

The demo runs on a **10-frame consecutive image sequence** from `MOT17-04-FRCNN` and evaluates a **20-frame sampled segment** of the `MERL` retail dataset.

### 1.2 Architectural Boundaries & Constraints
> [!IMPORTANT]
> **Strict Identification Scope Boundary:**
> - **Clip-Local IDs Only:** Track IDs (e.g., `ID:1`, `ID:2`) assigned by ByteTrack are **temporary session-local integers** valid only during continuous visibility within a single video sequence.
> - **No Biometrics or Re-ID:** The system does **NOT** perform facial recognition, appearance feature embedding, biometric identification, or cross-camera re-identification (Re-ID).
> - **No Cross-Session Identity:** If an individual leaves the field of view and exceeds the `max_time_lost_sec` timeout (2.0s), the track is deleted. If they re-enter, a new clip-local ID is assigned.
> - **Dataset Distinction:** The MOT17-04 sample consists of **outdoor pedestrian-street footage**, NOT retail store footage. It serves as an engineering pipeline test, NOT a retail-domain validation.

---

## 2. Source Code & Test Walkthrough

Below are the verified repository paths for the Module-1 components:

| Component | Repository Path | Key Responsibilities |
|---|---|---|
| **Detector Class** | [`src/detection/detector.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/detection/detector.py) | ONNX Runtime session, 640×640 letterbox scaling, person filtering (`target_class_id: 0`). |
| **Detection Postprocessing** | [`src/detection/postprocess.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/detection/postprocess.py) | NMS IoU suppression (0.45), bounding box coordinate un-letterboxing, `Detection` dataclasses. |
| **Detection Package Export** | [`src/detection/__init__.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/detection/__init__.py) | Exports `YOLO11PersonDetector`, `Detection`, `DetectionResult`, `letterbox`. |
| **ByteTrack Algorithm** | [`src/tracking/bytetrack.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/tracking/bytetrack.py) | Two-stage bipartite matching (`scipy.optimize.linear_sum_assignment`), `Track` state machine, bottom-center computation. |
| **Kalman Filter** | [`src/tracking/kalman_filter.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/tracking/kalman_filter.py) | 8-state bounding box filter $[x, y, a, h, v_x, v_y, v_a, v_h]$ for temporal motion forecasting. |
| **Tracking Package Export** | [`src/tracking/__init__.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/tracking/__init__.py) | Exports `ByteTracker`, `Track`, `TrackPoint`, `TrackingResult`, `compute_iou_batch`. |
| **Verification Runner** | [`scripts/verify_phase_04.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/scripts/verify_phase_04.py) | End-to-end benchmark script executing MOT17-04 sequence and MERL video demo. |
| **Detector Tests** | [`tests/test_detection.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/tests/test_detection.py) | 9 unit/integration tests (model loading, letterbox math, NMS, confidence thresholds). |
| **Tracker Tests** | [`tests/test_tracking.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/tests/test_tracking.py) | 8 unit/integration tests (initialization, IoU matrix, persistence, occlusion recovery, history). |
| **Historical Phase Report** | [`docs/PHASE_04_REPORT.md`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/docs/PHASE_04_REPORT.md) | Historical Phase 4 report (dated 2026-09-17). |
| **Configuration** | [`config.yaml`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/config.yaml) | `detection` (conf: 0.40, iou: 0.45) and `tracking` (high: 0.50, low: 0.10, lost: 2.0s) parameters. |
| **Model Asset** | `data/models/yolo11n.onnx` | YOLO11 nano model exported to ONNX format (10,720,346 bytes). |

---

## 3. Examiner Run / Show Sequence

When presenting the Module-1 demo to an examiner, follow this structured review sequence:

### Step 1: Pre-Execution Context & Dataset Clarification
- Direct the examiner to the input dataset: `data/mot17/train/MOT17-04-FRCNN/img1/` (frames `000001.jpg` to `000010.jpg`).
- **Clarify upfront:** This is an outdoor street scene with walking pedestrians, used strictly to verify multi-object bounding box association across consecutive frames. It is NOT retail store footage.

### Step 2: Running or Reviewing the Verification Script
- **Live Execution Command (PowerShell):**
  ```powershell
  .\venv\Scripts\python.exe scripts/verify_phase_04.py
  ```
- *Alternative Non-Destructive Review:* Point directly to the already generated outputs in `outputs/reports/` (detailed in Section 5) to avoid re-running during presentation.

### Step 3: Show Visual Tracking Progression
Open and present the three key MOT17 annotated frames in chronological sequence:
1. **Frame 01:** [`outputs/reports/phase_04_mot17_tracking_frame_01.jpg`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/outputs/reports/phase_04_mot17_tracking_frame_01.jpg)  
   *What to highlight:* 12 raw detections yield 9 initial active tracks (`ID:1` through `ID:9`). Each person has a green bounding box, confidence label, and a red bottom-center foot contact point.
2. **Frame 05:** [`outputs/reports/phase_04_mot17_tracking_frame_05.jpg`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/outputs/reports/phase_04_mot17_tracking_frame_05.jpg)  
   *What to highlight:* 10 active tracks. Note that `ID:1` through `ID:9` maintain their exact identities despite pedestrian movement.
3. **Frame 10:** [`outputs/reports/phase_04_mot17_tracking_frame_10.jpg`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/outputs/reports/phase_04_mot17_tracking_frame_10.jpg)  
   *What to highlight:* 10 active tracks (`ID:1`–`ID:9` and `ID:11`). Total tracks created over 10 frames is 11, demonstrating high identity stability with zero arbitrary ID churning. Small yellow directional tails visualize Kalman velocity vectors.

### Step 4: Show Structured JSON Output
Open [`outputs/reports/phase_04_tracking_summary.json`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/outputs/reports/phase_04_tracking_summary.json):
```json
{
  "tracker_algorithm": "ByteTrack (scipy.optimize.linear_sum_assignment + KalmanFilter)",
  "mot17_sequence_test": {
    "frames_processed": 10,
    "total_tracks_created": 11,
    "final_active_tracks": 10,
    "avg_detector_ms": 105.37,
    "avg_tracker_ms": 2.21
  },
  "merl_video_test": {
    "sampled_frames_processed": 20,
    "total_tracks_created": 0,
    "avg_detector_ms": 98.36,
    "avg_tracker_ms": 0.01
  }
}
```

---

## 4. Spoken Demo Outline (60–90 Seconds)

| Time Window | Topic | Spoken Script Outline |
|---|---|---|
| **0:00 – 0:20** | **Input & Detection** | *"In Module 1, our vision pipeline begins by passing sampled video frames into a YOLO11-nano model running via ONNX Runtime on CPU. The model detects person instances at 640×640 resolution, filtering out non-person classes and applying Non-Maximum Suppression."* |
| **0:20 – 0:45** | **Tracking & ID Association** | *"Raw detections feed into our ByteTrack implementation. Instead of discarding lower-confidence detections, ByteTrack uses a two-stage Hungarian matching process coupled with an 8-state Kalman Filter. As we advance from Frame 1 to Frame 10, you can see IDs 1 through 9 remain perfectly stable across consecutive frames without swapping."* |
| **0:45 – 1:05** | **Trajectory Contact Point** | *"Notice the red centroid marker at the bottom-center of each bounding box $[(x_1+x_2)/2, y_2]$. This represents the individual's foot contact position on the ground plane in image space. This point is accumulated in track history and directly feeds our Phase 5 homography floor transformation."* |
| **1:05 – 1:30** | **Summary & Scope Boundary** | *"Across this 10-frame test sequence, exactly 11 tracks were created with 10 active at sequence end. Crucially, these track IDs are strictly temporary and clip-local. The system does not use facial recognition, biometric tags, or persistent customer re-identification."* |

---

## 5. Evidence Provenance

To maintain complete audit integrity, project evidence is divided into three distinct categories:

### Category A: Freshly Verified Evidence (Antigravity Report, 2026-09-27)
- **Full Automated Regression:** **137/137 tests passing** in 4.73s (`pytest -q -p no:cacheprovider`).
- **Detection Unit Tests:** **9/9 tests passing** in `tests/test_detection.py` (2.15s).
- **Tracking Unit Tests:** **8/8 tests passing** in `tests/test_tracking.py` (0.95s).
- **Pipeline Runner Execution:** `scripts/verify_phase_04.py` completed cleanly with exit code 0.
- **MOT17-04 Pipeline Metrics:** 10 frames processed, 11 total tracks created, 10 active tracks at frame 10.
- **MERL Feasibility Metrics:** 20 sampled frames processed, 0 detections, 0 tracks created.

### Category B: Existing Generated Artifacts (`outputs/reports/`)
The following generated output files exist on disk:

| File Path | File Size | Timestamp | Nature of File |
|---|---|---|---|
| `outputs/reports/phase_04_mot17_tracking_frame_01.jpg` | 394,667 bytes | 2026-09-27 13:36:47 | Generated annotated frame |
| `outputs/reports/phase_04_mot17_tracking_frame_05.jpg` | 392,761 bytes | 2026-09-27 13:36:47 | Generated annotated frame |
| `outputs/reports/phase_04_mot17_tracking_frame_10.jpg` | 391,905 bytes | 2026-09-27 13:36:48 | Generated annotated frame |
| `outputs/reports/phase_04_merl_tracking_sample_1_frame0.jpg` | 168,332 bytes | 2026-09-27 13:36:48 | Generated annotated frame |
| `outputs/reports/phase_04_merl_tracking_sample_2_frame6.jpg` | 192,134 bytes | 2026-09-27 13:36:48 | Generated annotated frame |
| `outputs/reports/phase_04_merl_tracking_sample_3_frame12.jpg` | 193,140 bytes | 2026-09-27 13:36:48 | Generated annotated frame |
| `outputs/reports/phase_04_tracking_summary.json` | 619 bytes | 2026-09-27 13:36:50 | Generated summary JSON |

> [!NOTE]
> **Artifact vs. Raw Log Distinction:**  
> The files above are **generated presentation outputs and structured JSON summaries**, NOT raw execution logs. Raw console stdout/stderr logs from terminal or subprocess pytest executions are ephemeral and are not stored in `outputs/reports/`.
>
> **Git-Ignore Status:**  
> In accordance with project configuration (`.gitignore` lines 48–55), `outputs/reports/*` is explicitly excluded from version control. A "clean" Git worktree status reflects these Git-ignore rules and does not mean generated outputs were absent or un-refreshed.

### Category C: Historical Documentation & Results (`docs/PHASE_04_REPORT.md`)
- **Document Date:** 2026-09-17.
- **Historical Scope:** Documented initial Phase 4 spike completion when the codebase had 27 total automated tests (Phases 0–4).
- **Separation Notice:** Do **NOT** merge historical report metrics with fresh regression figures. The regression suite has since expanded from 27 to 137 tests across subsequent phases (Phases 5–8).

---

## 6. Rigorous Evidence Status Classification

| Dimension | Classification | Factual Basis & Justification |
|---|---|---|
| **Pipeline Implementation** | **IMPLEMENTED: YES** | Complete YOLO11 detector, ByteTracker, Kalman filter, and visualization routines are fully implemented in production source modules. |
| **Operational Execution** | **OPERATIONALLY VERIFIED: YES** | Verified by fresh Antigravity report: runner completes end-to-end without errors; 9/9 detection and 8/8 tracking tests pass; bounding boxes and persistent track IDs are visually confirmed on consecutive frames. |
| **Tracking Accuracy Benchmark** | **QUANTITATIVELY VALIDATED: NO** | **No matched ground-truth tracking benchmark was performed.** The 10-frame MOT17 clip is an engineering check, not a full benchmark sequence. Standard MOT metrics (**MOTA, MOTP, IDF1, HOTA, ID switches**) were **NOT** computed against ground-truth labels. |
| **Retail Performance** | **QUANTITATIVELY VALIDATED: NO** | **Retail performance is NOT validated.** Testing on the MERL retail clip (`10_1_crop.mp4`) resulted in **0 detections and 0 tracks**. |

### Detailed Clarification on MERL Retail Clip Result
The MERL dataset clip was tested strictly as a domain-transfer feasibility inspection. Because the camera is mounted directly on the ceiling pointing straight down into an overhead retail aisle with high lighting glare and shelf occlusion, the stock COCO-trained YOLO11n model (trained predominantly on eye-level and angled perspectives) failed to detect persons (`total_tracks_created: 0`). **This zero-detection result conclusively proves that retail performance is NOT validated by this demo.**

---

## 7. Omission of Performance & Latency Claims

> [!WARNING]
> **Latency and FPS Claims Are Excluded From Presentation:**
> Specific detector inference times, tracking overhead numbers, and combined frames-per-second (FPS) figures are **intentionally omitted** from presentation claims.
>
> **Reason for Omission:**  
> In the historical report (2026-09-17), timing was reported as 84.68 ms detection latency and 11.74 FPS on an AMD Ryzen 5 CPU. In the fresh run (2026-09-27), timing was recorded as 100.70 ms detection latency and 9.86 FPS on the current execution environment. Because CPU inference timing fluctuates substantially depending on host machine hardware, background process contention, thermal throttling, and ONNX Runtime CPU thread scheduling, these figures do not constitute a stable, presentation-ready benchmark.

---

## 8. Presentation Checklist & Claims to Avoid

### 8.1 Demonstration Checklist (What to Open & Show)
- [ ] **Frame 01:** Show `outputs/reports/phase_04_mot17_tracking_frame_01.jpg` (initial detections, bounding boxes, `ID:1`–`ID:9`, foot contact points).
- [ ] **Frame 05:** Show `outputs/reports/phase_04_mot17_tracking_frame_05.jpg` (identity persistence across movement).
- [ ] **Frame 10:** Show `outputs/reports/phase_04_mot17_tracking_frame_10.jpg` (continuity of tracks, appearance of `ID:11`, motion tails).
- [ ] **Summary JSON:** Open `outputs/reports/phase_04_tracking_summary.json` (point out 11 created tracks, 10 active tracks on MOT17, and transparently acknowledge 0 tracks on MERL).
- [ ] **Code Architecture:** Show `src/detection/detector.py` and `src/tracking/bytetrack.py` to demonstrate modular implementation.
- [ ] **Automated Testing:** Reference `tests/test_detection.py` (9 tests) and `tests/test_tracking.py` (8 tests) within the 137-test full regression suite.

### 8.2 Short List of Claims to AVOID
1. ❌ **Do NOT claim tracking accuracy percentages:** Do not present track count (11 tracks across 10 frames) or active count (10 tracks) as an "accuracy" or "recall" rate.
2. ❌ **Do NOT claim benchmark validation:** Do not claim MOTA, MOTP, IDF1, or HOTA scores.
3. ❌ **Do NOT claim retail readiness:** Do not claim the model is validated for retail stores (MERL produced 0 detections).
4. ❌ **Do NOT claim biometric or facial tracking:** Do not claim the system recognizes customers, stores identities in databases, or tracks individuals across sessions.
5. ❌ **Do NOT claim real-time guarantees:** Do not promise real-time performance based on fluctuating CPU inference speeds.
