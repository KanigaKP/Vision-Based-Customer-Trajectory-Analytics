# Phase 2 Report — Video Quality Verification & Preprocessing Layer

**Date:** 2026-09-17  
**Status:** ✅ PASS

---

## 1. Objective

Build and validate a modular, robust **Video Quality Verification & Preprocessing Layer** under `src/video/`. The module provides reliable technical metadata extraction, graceful error handling for invalid/corrupted files, configurable quality indicators (resolution, FPS, brightness, sharpness/blur), and memory-efficient frame sampling prior to downstream object detection.

---

## 2. Existing Environment & Context

| Component | Status / Version |
|---|---|
| **OS** | Windows 11 Home Single Language (Build 26200, x64) |
| **Python** | 3.14.6 (`venv/`) |
| **Inference Engine** | ONNX Runtime (`onnxruntime` 1.30.0, `CPUExecutionProvider`) |
| **YOLO11 Model Asset**| `data/models/yolo11n.onnx` (10.22 MB) |
| **Primary Test Data**| `data/merl/10_1_crop.mp4` (920×680 @ 30 FPS, 3796 frames, 126.53s) |
| **Testing Framework**| `pytest` 9.1.1 |

---

## 3. Architecture of Video Preprocessing Layer

The preprocessing layer is organized into a modular package under `src/video/`:

```
src/video/
├── __init__.py       # Package exports
├── inspector.py      # Video validation, metadata extraction, graceful error handling
├── quality.py        # Quality evaluation (resolution, FPS, brightness, sharpness)
└── sampler.py        # Memory-efficient lazy frame sampler (generator pattern)
```

### Module Responsibilities

```
Input Video File Path
        │
        ├──> VideoInspector (inspector.py)
        │       ├── File existence & extension validation
        │       ├── OpenCV stream opening & readability test
        │       └── Metadata extraction (Width, Height, FPS, FourCC, Frames, Duration)
        │
        ├──> QualityEvaluator (quality.py)
        │       ├── Resolution & FPS checks against config warning thresholds
        │       ├── Mean brightness calculation (grayscale pixel mean)
        │       ├── Sharpness evaluation (Laplacian variance operator)
        │       └── Overall status determination (PASS / WARNING / ERROR)
        │
        └──> FrameSampler (sampler.py)
                ├── Target sampling rate calculation (e.g. 5.0 FPS)
                ├── Memory-efficient frame skipping (no full video memory loading)
                └── Generator yielding (frame_index, timestamp_sec, frame_bgr)
```

---

## 4. Files & Modules Created / Modified

| File Path | Description |
|---|---|
| `src/video/__init__.py` | Package marker exporting `VideoInspector`, `QualityEvaluator`, `FrameSampler` |
| `src/video/inspector.py` | Video metadata extractor and graceful error handler |
| `src/video/quality.py` | Quality evaluator computing brightness, Laplacian variance sharpness, and thresholds |
| `src/video/sampler.py` | Generator-based frame sampler for low-memory frame extraction |
| `scripts/verify_phase_02.py` | Verification spike script processing MERL dataset video |
| `tests/test_video_preprocessing.py` | Suite of 10 automated unit and integration tests |
| `config.yaml` | Updated with video preprocessing, sampling, and configurable quality thresholds |
| `requirements.txt` | Updated with `onnxruntime`, `huggingface_hub`, `pytest` |
| `docs/PHASE_02_REPORT.md` | This completion report |
| `docs/DECISIONS.md` | Updated with DEC-014 |

---

## 5. Metadata Extracted

The `VideoInspector` class extracts structured metadata encapsulated in the `VideoMetadata` dataclass:

```json
{
  "file_path": "C:\\...\\data\\merl\\10_1_crop.mp4",
  "file_name": "10_1_crop.mp4",
  "file_size_bytes": 15298150,
  "exists": true,
  "valid_extension": true,
  "is_readable": true,
  "fourcc": "875967080.0",
  "codec_name": "h264",
  "width": 920,
  "height": 680,
  "fps": 30.0,
  "total_frames": 3796,
  "duration_seconds": 126.53,
  "sample_frame_accessible": true,
  "status": "VALID",
  "error_message": null
}
```

---

## 6. Quality Indicators & Configurable Thresholds

Quality indicators are evaluated against configurable engineering warning thresholds defined in `config.yaml`:

```yaml
video:
  allowed_extensions: [".mp4", ".avi", ".mov", ".mkv"]
  sampling:
    target_fps: 5.0
    max_sampled_frames: 300
  quality_thresholds:
    min_width: 640
    min_height: 480
    min_fps: 10.0
    brightness:
      min_mean: 40.0
      max_mean: 230.0
    sharpness:
      min_laplacian_var: 100.0
```

> **IMPORTANT METHODOLOGICAL DISCLAIMER:**  
> Quality thresholds (e.g. `min_laplacian_var: 100.0`, `min_mean: 40.0`) are **configurable engineering warning thresholds**, NOT scientifically validated universal physical constants. They serve to alert developers/users to potential recording anomalies (e.g., underexposure, blur, low resolution) without discarding frames or claiming an absolute quality score.

### Evaluated Metrics

1. **Resolution:** Verified width/height against minimum bounds.
2. **FPS:** Verified native stream frame rate.
3. **Brightness:** Mean pixel intensity ($\mu_{gray} \in [0, 255]$). Flags underexposition ($< 40$) or overexposition ($> 230$).
4. **Sharpness:** Variance of the Laplacian operator ($\text{Var}(\Delta I_{gray})$). Flags motion or focus blur ($< 100$).
5. **Frame Readability:** Verified decodability across sampled positions.

---

## 7. Frame Sampling Strategy

- **Design:** Generator-based lazy reader (`sample_frames()`).
- **Memory Efficiency:** Reads frames sequentially or jumps to specified index, keeping only one frame buffer in memory at any time.
- **Target Sampling Rate:** Configurable (default `5.0 FPS`). Native frames are sampled at step interval $S = \max(1, \lfloor \text{FPS}_{native} / \text{FPS}_{target} \rceil)$.
- **Output:** Tuple of `(frame_index, timestamp_seconds, frame_bgr)`.

---

## 8. Test Cases & Test Suite Summary

Automated tests implemented in `tests/test_video_preprocessing.py`:

| # | Test Name | Purpose | Result |
|---|---|---|---|
| 1 | `test_inspector_missing_video_path` | Verify graceful handling when file path does not exist | ✅ PASSED |
| 2 | `test_inspector_empty_path` | Verify graceful handling for empty/None string | ✅ PASSED |
| 3 | `test_inspector_invalid_extension` | Verify rejection of non-video extensions (`.txt`) | ✅ PASSED |
| 4 | `test_inspector_corrupted_video` | Verify handling of corrupt/zero-byte `.mp4` file without crashing | ✅ PASSED |
| 5 | `test_inspector_valid_video` | Verify metadata extraction on `data/merl/10_1_crop.mp4` | ✅ PASSED |
| 6 | `test_quality_evaluator_missing_file` | Verify quality evaluator returns `ERROR` status for missing file | ✅ PASSED |
| 7 | `test_quality_evaluator_valid_video` | Verify brightness & sharpness computation on MERL video | ✅ PASSED |
| 8 | `test_brightness_and_sharpness_math` | Verify synthetic black, white, and checkerboard frame math | ✅ PASSED |
| 9 | `test_frame_sampler_valid_video` | Verify lazy frame generator order, timestamps, and shape | ✅ PASSED |
| 10| `test_frame_sampler_invalid_video` | Verify `ValueError` raised gracefully on unreadable input | ✅ PASSED |

**Execution Command:** `pytest -v`  
**Result:** **10 passed in 1.19s**

---

## 9. Actual Test Results on Project Data (`data/merl/10_1_crop.mp4`)

Executing `scripts/verify_phase_02.py` produced the following verification results:

```
======================================================================
PHASE 2 — VIDEO PREPROCESSING & QUALITY VERIFICATION SPIKE
======================================================================

[1/4] Inspecting Video Input: data\merl\10_1_crop.mp4
  Status                  : VALID
  File Name               : 10_1_crop.mp4
  File Size               : 14.59 MB
  Resolution              : 920 x 680
  FPS                     : 30.00
  Total Frames            : 3796
  Duration                : 126.53 seconds
  Codec / FourCC          : h264 (875967080.0)
  Sample Frame Accessible : True

[2/4] Evaluating Quality Indicators against Configured Warning Thresholds...
  Overall Quality Status  : PASS
  Resolution Status       : PASS (920x680)
  FPS Status              : PASS (30.00 FPS)
  Brightness Status       : PASS (Mean: 131.5 / 255)
  Sharpness Status        : PASS (Laplacian Var: 371.9)
  Readability Status      : PASS

[3/4] Sampling Frames at Target Rate (5.0 FPS)...
  Saved representative frame 1: outputs\reports\merl_sampled_frame_1_idx0.jpg
  Saved representative frame 2: outputs\reports\merl_sampled_frame_2_idx6.jpg
  Saved representative frame 3: outputs\reports\merl_sampled_frame_3_idx12.jpg
  Sampled 30 frames efficiently (Target FPS: 5.0).

[4/4] Exported Validation Summary JSON: outputs\reports\phase_02_video_validation_summary.json

======================================================================
PHASE 2 VERIFICATION SPIKE SUCCESSFUL
======================================================================
```

### Generated Artifacts (`outputs/reports/`)

- `outputs/reports/merl_sampled_frame_1_idx0.jpg` — Representative sampled frame at $t=0.00\text{s}$
- `outputs/reports/merl_sampled_frame_2_idx6.jpg` — Representative sampled frame at $t=0.20\text{s}$
- `outputs/reports/merl_sampled_frame_3_idx12.jpg` — Representative sampled frame at $t=0.40\text{s}$
- `outputs/reports/phase_02_video_validation_summary.json` — Complete JSON metadata & quality log

---

## 10. Problems Encountered & Resolutions

1. **OpenCV Exception Risks on Corrupt Streams:**
   - *Problem:* Calling OpenCV methods on corrupt or 0-byte video files can throw unhandled exceptions or crash.
   - *Resolution:* Wrapped OpenCV calls inside `VideoInspector` and `QualityEvaluator` with `isOpened()`, frame reading checks, and `try/except/finally` blocks ensuring `cap.release()` is always called.

---

## 11. Decisions Made

- **DEC-014:** Adopted modular video preprocessing architecture (`src/video/`) with generator-based frame sampling and configurable warning thresholds in `config.yaml`. See [`docs/DECISIONS.md`](DECISIONS.md) for full entry.

---

## 12. Limitations

- **Heuristic Indicators:** Laplacian variance sharpness and mean grayscale brightness are global frame heuristics. They indicate gross blur or lighting issues but do not detect localized object occlusions.
- **Offline Processing:** Sampling is optimized for offline file-based processing, not live RTSP streaming.

---

## 13. Phase 2 Acceptance Criteria Checklist

| # | Criterion | Status |
|---|---|---|
| 1 | Valid video can be opened reliably | ✅ PASS |
| 2 | Video metadata can be extracted | ✅ PASS |
| 3 | Invalid input is handled gracefully (no crashes) | ✅ PASS |
| 4 | Brightness indicator works | ✅ PASS |
| 5 | Sharpness indicator works (Laplacian variance) | ✅ PASS |
| 6 | Frame sampling works without loading full video into memory | ✅ PASS |
| 7 | Representative sample frames generated in `outputs/reports/` | ✅ PASS |
| 8 | At least one existing project video (`MERL 10_1_crop.mp4`) processed | ✅ PASS |
| 9 | Basic automated tests pass (`10/10 pytest passed`) | ✅ PASS |
| 10| Results documented and no unsupported quality claims made | ✅ PASS |

---

## 14. Recommendation for Phase 3

**Phase 2 is COMPLETE with status PASS.**

The project is ready to proceed to **Phase 3: YOLO11 Person Detection Module Implementation** using the established `ONNX_RUNTIME_CPU_PATH` engine (`data/models/yolo11n.onnx`) and the `FrameSampler` generator built in Phase 2.
