# Phase 3 Report — YOLO11 Person Detection Module Implementation

**Date:** 2026-09-17  
**Status:** ✅ PASS

---

## 1. Objective

Build, evaluate, and benchmark a clean, reusable **YOLO11 Person Detection Module** (`src/detection/`) powered by ONNX Runtime CPU inference (`ONNX_RUNTIME_CPU_PATH`). The pipeline consumes video frames lazily from the existing `FrameSampler` (Phase 2), letterboxes and normalizes input tensors, executes ONNX CPU inference, filters exclusively for COCO class ID `0` (`person`), applies vectorised Non-Maximum Suppression (NMS), and outputs structured person detection bounding boxes and confidence scores.

---

## 2. Detection Architecture

The detection layer is organized into a modular package under `src/detection/`:

```
src/detection/
├── __init__.py         # Package exports
├── detector.py        # YOLO11PersonDetector using ONNX Runtime InferenceSession
└── postprocess.py     # Letterbox preprocessing, coordinate scaling, NMS filtering
```

### Pipeline Sequence

```
Input Video Frame (BGR Image)
        │
        ├──> FrameSampler (Phase 2)
        │       └── Yields (frame_idx, timestamp_sec, frame_bgr)
        │
        ├──> YOLO11PersonDetector.preprocess()
        │       ├── Aspect-ratio letterboxing to (640, 640, 3) with pad (114, 114, 114)
        │       ├── BGR to RGB color conversion
        │       ├── Float32 normalization to [0.0, 1.0]
        │       └── Layout transposition to NCHW tensor: (1, 3, 640, 640)
        │
        ├──> ONNX Runtime CPU Inference Engine
        │       └── session.run(["output0"], {"images": input_tensor})
        │               └── Output tensor shape: (1, 84, 8400)
        │
        ├──> postprocess_yolo11_output()
        │       ├── Squeeze batch -> (84, 8400) matrix
        │       ├── Extract boxes [cx, cy, w, h] and COCO Person class scores (row 4)
        │       ├── Confidence threshold filtering (conf >= 0.40)
        │       ├── Bounding box coordinate un-letterboxing to original image space
        │       └── Vectorised NMS filtering (cv2.dnn.NMSBoxes, IoU = 0.45)
        │
        └──> DetectionResult Container
                ├── Filtered person detections (bbox [x1, y1, x2, y2], confidence)
                └── Timing breakdown (preprocess, inference, postprocess, total ms)
```

---

## 3. ONNX Model Details

| Property | Value |
|---|---|
| **Model Asset** | `data/models/yolo11n.onnx` |
| **Model Architecture** | YOLO11 Nano (`yolo11n`) |
| **File Size** | 10.22 MB |
| **Inference Engine** | ONNX Runtime 1.30.0 |
| **Execution Provider**| `CPUExecutionProvider` |
| **PyTorch Runtime Dependency** | **None** (Zero PyTorch DLL execution) |

---

## 4. Input & Output Tensor Specifications

### Input Tensor (`images`)
- **Shape:** `[1, 3, 640, 640]`
- **Data Type:** `float32`
- **Layout:** `NCHW` (Batch = 1, Channels = 3, Height = 640, Width = 640)
- **Range:** Normalized $[0.0, 1.0]$ float32 RGB

### Output Tensor (`output0`)
- **Shape:** `[1, 84, 8400]`
- **Data Type:** `float32`
- **Layout:**
  - Row `0`: `cx` (center X in 640×640 pixel space)
  - Row `1`: `cy` (center Y in 640×640 pixel space)
  - Row `2`: `w` (box width in 640×640 pixel space)
  - Row `3`: `h` (box height in 640×640 pixel space)
  - Rows `4..83`: Class confidence probabilities across COCO 80 categories (Row `4` = Class `0` `person`)
  - Columns `0..8399`: 8,400 candidate anchor predictions

---

## 5. Preprocessing & Letterboxing

To preserve aspect ratios without distortion, images undergo **letterboxing**:
1. Compute scale ratio $r = \min(640 / W_{orig}, 640 / H_{orig})$.
2. Resize unpadded image to $(W_{new}, H_{new}) = (\text{round}(W_{orig} \times r), \text{round}(H_{orig} \times r))$.
3. Compute padding offsets $(dw, dh) = ((640 - W_{new}) / 2, (640 - H_{new}) / 2)$.
4. Pad borders with neutral gray $(114, 114, 114)$ to achieve exact $640 \times 640 \times 3$ size.
5. Convert BGR to RGB, scale to $[0.0, 1.0]$, transpose to NCHW format.

---

## 6. Post-Processing & NMS

1. Candidate anchors are extracted from row index `4` of the output matrix for COCO class ID `0` (`person`).
2. Candidate boxes with `person_score < conf_threshold` (default `0.40`) are discarded.
3. Filtered boxes $[cx, cy, w, h]$ are transformed to $[x_1, y_1, x_2, y_2]$ in letterbox space, then scaled back to original image space:
   $$x_1^{orig} = \text{clip}\left(\frac{x_1 - dw}{r}, 0, W_{orig}\right), \quad y_1^{orig} = \text{clip}\left(\frac{y_1 - dh}{r}, 0, H_{orig}\right)$$
4. Vectorised Non-Maximum Suppression (`cv2.dnn.NMSBoxes`) is applied with `iou_threshold = 0.45` to eliminate duplicate/overlapping boxes.

---

## 7. Person Filtering & Class Scoping

- **Class Target:** COCO Class ID `0` (`person`).
- **All non-person COCO classes** (vehicles, animals, objects) are ignored.
- Each returned detection contains `bbox: [x1, y1, x2, y2]`, `confidence: float`, `class_id: 0`, `class_name: "person"`.

---

## 8. Confidence & NMS Thresholds

Quality thresholds configured in `config.yaml`:
- `conf_threshold: 0.40`
- `iou_threshold: 0.45`

> **IMPORTANT CONFIGURATION NOTE:**  
> The confidence threshold `0.40` and IoU threshold `0.45` are **configurable engineering starting values**, NOT scientifically validated optimal numbers. They can be tuned per scenario in `config.yaml`.

---

## 9. Single-Frame Test Results (`MOT17-04-FRCNN/img1/000001.jpg`)

| Metric | Measured Value |
|---|---|
| **Test Image** | `MOT17-04-FRCNN/img1/000001.jpg` (1920 × 1080) |
| **Raw Candidate Anchors** | 8,400 |
| **Filtered Person Detections** | **12 persons** |
| **Confidence Distribution** | min = 0.43, mean = 0.60, max = 0.79 |
| **Preprocessing Latency** | 9.16 ms |
| **ONNX Inference Latency** | **82.96 ms** |
| **Post-processing Latency** | 0.96 ms |
| **Total Single-Frame Time** | **93.08 ms** |
| **Model Inference Speed** | **12.05 Model FPS** |
| **Detector Speed** | **10.74 Detector FPS** |
| **Saved Visualization** | `outputs/reports/phase_03_mot17_frame1_detections.jpg` |

---

## 10. Sampled Video Test Results (`MERL 10_1_crop.mp4`)

- **Video Input:** `data/merl/10_1_crop.mp4` (920 × 680 @ 30 FPS).
- **Sampling Strategy:** 20 sampled frames at 5.0 FPS target using `FrameSampler`.
- **Saved Sample Frames:**
  - `outputs/reports/phase_03_merl_sample_1_frame0.jpg`
  - `outputs/reports/phase_03_merl_sample_2_frame6.jpg`
  - `outputs/reports/phase_03_merl_sample_3_frame12.jpg`
- **JSON Validation Log:** `outputs/reports/phase_03_detection_summary.json`.

---

## 11. Actual Performance Measurements

Measurements performed on AMD Ryzen 5 7520U (4C/8T CPU):

| Stage | Mean Latency (ms) | Speed (FPS) |
|---|---|---|
| **Model Warmup / Initialization** | 98.86 ms | N/A (One-time) |
| **Preprocessing (Letterbox + Transpose)** | 9.16 ms | 109.17 FPS |
| **ONNX Runtime CPU Inference** | **80.62 ms** | **12.40 Model FPS** |
| **Post-processing (Unpad + NMS)** | 0.96 ms | 1041.67 FPS |
| **Single-Frame End-to-End Detector** | **86.27 ms** | **11.59 Detector FPS** |
| **End-to-End Video Pipeline** | **102.46 ms** | **9.76 Pipeline FPS** |

> **PERFORMANCE NOTE:**  
> Model Inference FPS (~12.4 FPS) measures purely the ONNX runtime forward pass.  
> End-to-End Video Pipeline FPS (~9.76 FPS) measures disk reading, frame decoding, preprocessing, inference, postprocessing, and annotation. The system is designed for **offline processing**, not real-time CCTV streaming.

---

## 12. Automated Tests Summary

Automated tests implemented in `tests/test_detection.py`:

| Test Name | Verified Behavior | Status |
|---|---|---|
| `test_missing_model_raises_error` | Detector raises `FileNotFoundError` when model is missing | ✅ PASSED |
| `test_detector_initialization` | Session initializes on CPU, metadata exposed | ✅ PASSED |
| `test_letterbox_preprocessing` | Aspect-ratio letterbox scaling and padding math | ✅ PASSED |
| `test_detector_preprocess_tensor_shape` | Tensor dimensions `(1, 3, 640, 640)`, float32, range [0, 1] | ✅ PASSED |
| `test_detect_empty_image_raises_error` | Rejects empty numpy array gracefully | ✅ PASSED |
| `test_detection_output_structure` | `DetectionResult` fields and timing metrics | ✅ PASSED |
| `test_confidence_threshold_behavior` | Conf threshold 0.95 vs 0.20 filtering behavior | ✅ PASSED |
| `test_detection_on_known_test_frame` | MOT17-04 frame 1 person bounding boxes within image bounds | ✅ PASSED |
| `test_annotate_image` | Bounding box visualization drawing | ✅ PASSED |

**Full Test Suite Result (`pytest -v`):**  
**19 passed in 2.38s** (10 video preprocessing tests + 9 detection tests).

---

## 13. Files & Modules Created / Modified

| File Path | Description |
|---|---|
| `src/detection/__init__.py` | Package exports for `YOLO11PersonDetector`, `Detection`, `DetectionResult` |
| `src/detection/detector.py` | Main detector module powered by ONNX Runtime CPU |
| `src/detection/postprocess.py` | Letterbox preprocessing, coordinate scaling, NMS filtering |
| `scripts/verify_phase_03.py` | Single-frame and video pipeline benchmark script |
| `tests/test_detection.py` | Suite of 9 automated unit/integration tests for detection |
| `config.yaml` | Updated with `detection` model path and threshold parameters |
| `docs/PHASE_03_REPORT.md` | This completion report |
| `docs/DECISIONS.md` | Updated with DEC-015 |

---

## 14. Problems Encountered & Resolutions

1. **OpenCV NMS Return Format Variation:**
   - *Problem:* `cv2.dnn.NMSBoxes` returns either a flat array or a 2D array depending on OpenCV version.
   - *Resolution:* Added robust index flattening logic handling both 1D and 2D return formats seamlessly.

---

## 15. Decisions Made

- **DEC-015:** Implemented letterbox aspect-ratio preprocessing and vectorised NMS post-processing for `YOLO11PersonDetector` using ONNX Runtime CPU. See [`docs/DECISIONS.md`](DECISIONS.md) for full entry.

---

## 16. Limitations

- **CPU Throughput:** Detection speed is ~12 Model FPS on CPU. High-resolution continuous video is sampled offline rather than processed at 30 FPS real-time.
- **Occlusion Sensitivity:** Small or heavily occluded pedestrians in deep perspective backgrounds may fall below the 0.40 confidence threshold.

---

## 17. Scientific Rules & Accuracy Disclaimer

> **SCIENTIFIC ACCURACY DISCLAIMER:**  
> Detection visualizations and filtered candidate counts demonstrate that the detector pipeline functions correctly.  
> **They do NOT constitute a formal detection accuracy evaluation (e.g. mAP@50 or precision/recall).**  
> No claims of "95% accuracy", "high accuracy", or "real-time capability" are made. Formal ground-truth benchmark evaluation will be conducted when evaluation protocols are implemented.

---

## 18. Phase 3 Acceptance Criteria Checklist

| # | Criterion | Status |
|---|---|---|
| 1 | YOLO11n ONNX model loads successfully through ONNX Runtime | ✅ PASS |
| 2 | PyTorch is NOT required for runtime inference | ✅ PASS |
| 3 | Image preprocessing verified against actual ONNX model metadata | ✅ PASS |
| 4 | YOLO11n inference succeeds on test frame | ✅ PASS |
| 5 | Person-only filtering works (COCO class ID 0) | ✅ PASS |
| 6 | Bounding boxes and confidence values produced | ✅ PASS |
| 7 | NMS / post-processing works correctly | ✅ PASS |
| 8 | Annotated detections visualized in `outputs/reports/` | ✅ PASS |
| 9 | Detector works with sampled video frames (`FrameSampler`) | ✅ PASS |
| 10| Actual inference / performance measurements recorded | ✅ PASS |
| 11| Automated tests pass (`19/19 pytest passed`) | ✅ PASS |
| 12| No unsupported accuracy claims made | ✅ PASS |

---

## 19. Recommendation for Phase 4

**Phase 3 is COMPLETE with status PASS.**

The project is ready to proceed to **Phase 4: ByteTrack Multi-Object Tracking Integration**, connecting the person detection outputs produced in Phase 3 to persistent multi-object tracking IDs across consecutive frames.
