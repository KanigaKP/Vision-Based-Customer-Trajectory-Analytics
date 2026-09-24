# Decision Log

This document records key architectural and technical decisions made throughout the project.

---

## DEC-001 — Python Version

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 0 |
| **Decision** | Use Python 3.14.6 (system-installed version) |
| **Rationale** | This is the version available on the development machine. No compatibility issues have been identified with the Phase 0 dependency stack. Compatibility with Ultralytics/YOLO11 and other future dependencies will be verified in Phase 1 before proceeding. |
| **Alternatives** | Install Python 3.11 or 3.12 for broader library compatibility. Will revisit if Ultralytics or ByteTrack requires it. |

---

## DEC-002 — Virtual Environment Tool

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 0 |
| **Decision** | Use Python's built-in `venv` module |
| **Rationale** | Lightweight, no extra tooling required, sufficient for a single-project academic prototype. |
| **Alternatives** | conda, poetry, pipenv — unnecessary complexity for this project. |

---

## DEC-003 — Phase 0 Dependency Scope

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 0 |
| **Decision** | Install only foundational scientific/utility packages: numpy, opencv-python, pandas, matplotlib, pyyaml |
| **Rationale** | Phase 0 is environment validation only. Heavy dependencies (ultralytics, streamlit, etc.) are deferred to their respective implementation phases to avoid premature breakage and keep the environment minimal. |
| **Alternatives** | Install the full stack upfront — rejected to reduce Phase 0 risk and keep the environment clean. |

---

## DEC-004 — No Version Pinning in Phase 0

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 0 |
| **Decision** | Install latest compatible versions without hard pins in Phase 0 |
| **Rationale** | Python 3.14 is recent; arbitrary pins from older projects are likely to cause install failures. Actual installed versions are recorded in PHASE_00_REPORT.md and can be frozen via `pip freeze` once validated. |
| **Alternatives** | Pin exact versions — will be done after Phase 0 validation succeeds (via `pip freeze > requirements-lock.txt`). |

---

## DEC-005 — Project Structure

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 0 |
| **Decision** | Adopt the structure proposed in the project specification with no modifications |
| **Rationale** | The proposed structure cleanly separates concerns (data, source, dashboard, tests, outputs, docs) and matches the pipeline stages. No technical reason to deviate. |
| **Alternatives** | Add `notebooks/` directory — deferred; will add if needed. |

---

## DEC-006 — Database Choice

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 0 |
| **Decision** | Use SQLite (stdlib `sqlite3`) — no external database server |
| **Rationale** | Zero-install, file-based, sufficient for a single-user research prototype. Keeps deployment trivial for academic evaluation. |
| **Alternatives** | PostgreSQL, MongoDB — unnecessary for this project scale. |

---

## DEC-007 — No Docker/Cloud Infrastructure

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 0 |
| **Decision** | Do not use Docker, cloud services, Redis, Kafka, or any enterprise infrastructure |
| **Rationale** | This is an academic research prototype. The environment is a single Windows machine. Adding infrastructure would add complexity without benefit. |
| **Alternatives** | N/A |

---

## DEC-008 — CPU-Only Inference

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 1 |
| **Decision** | Run YOLO11 and PyTorch in CPU-only mode |
| **Rationale** | No NVIDIA GPU is present. Only an AMD Radeon integrated GPU (512 MB VRAM) exists. CUDA is not available. CPU-only inference is acceptable for processing recorded video in an academic prototype. |
| **Alternatives** | Use DirectML or ONNX Runtime with AMD GPU — may be explored if CPU inference is too slow. |

---

## DEC-009 — Smart App Control Blocker (CRITICAL)

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 1 |
| **Decision** | User must disable Windows Smart App Control to allow PyTorch DLLs to load |
| **Rationale** | Windows Smart App Control (WDAC) blocks unsigned DLLs. PyTorch's `torch.dll` is unsigned and fails to load with `[WinError 4551]`. This affects ALL machines with SAC enabled, regardless of path. There is no per-app bypass. |
| **Alternatives** | None that avoid disabling SAC. Once disabled, SAC cannot be re-enabled without a Windows reinstall. |
| **Impact** | BLOCKS PyTorch/YOLO11 until resolved |

---

## DEC-010 — MOT17-04-FRCNN as Primary Evaluation Sequence

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 1 |
| **Decision** | Use MOT17-04-FRCNN as the primary tracking evaluation sequence |
| **Rationale** | Static camera, 1920×1080, 30 FPS, 1050 frames, 141 unique tracks, high pedestrian density. Ideal for MOTA/IDF1 evaluation. |
| **Alternatives** | MOT17-09 (indoor) — could be added as a secondary sequence. |

---

## DEC-011 — MERL Dataset Role: Demonstration Only

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 1 |
| **Decision** | Use MERL Shopping Dataset for retail-context demonstration only. Do NOT compute MOTA/IDF1 from MERL. |
| **Rationale** | MERL provides action annotations (reach, retract, inspect) — NOT continuous identity tracking ground truth. Computing tracking metrics from action labels would be methodologically invalid. |
| **Alternatives** | N/A — this is a constraint of the dataset, not a choice. |

---

## DEC-012 — Homography Feasibility Confirmed

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 1 |
| **Decision** | Image-to-floor-plane homography is feasible for zone analytics |
| **Rationale** | Tested on MOT17-04 frame. Near-field accuracy is excellent; far-field degrades with deep perspective. The controlled college footage (more overhead, shorter depth) should give better results. |
| **Alternatives** | Skip homography and work in pixel coordinates — rejected; floor-plane mapping is essential for meaningful spatial analytics. |

---

## DEC-013 — ONNX Runtime CPU Inference Path (`ONNX_RUNTIME_CPU_PATH`)

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 1.5 |
| **Decision** | Adopt `onnxruntime` CPU execution path (`yolo11n.onnx`) for object detection |
| **Rationale** | PyTorch's native `torch.dll` triggers Windows Smart App Control `[WinError 4551]`. `onnxruntime` 1.30.0 runs cleanly without security errors, achieving 86.5 ms (~11.6 FPS) single-frame CPU detection latency. Completely avoids disabling OS security settings or changing Python versions. |
| **Alternatives** | Disable Smart App Control (rejected per security directive); Python version migration (unnecessary since ONNX Runtime works in Python 3.14). |

---

## DEC-014 — Modular Video Preprocessing & Configurable Quality Warning Thresholds

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 2 |
| **Decision** | Build `src/video/` package (`VideoInspector`, `QualityEvaluator`, `FrameSampler`) with configurable engineering warning thresholds in `config.yaml` |
| **Rationale** | Video quality parameters (resolution, FPS, mean brightness, Laplacian variance sharpness) provide early detection of input video anomalies without throwing unhandled OpenCV exceptions or claiming universal physical limits. Frame sampling uses a generator pattern to keep memory consumption low. |
| **Alternatives** | Hardcoded universal quality thresholds — rejected; quality requirements vary per camera setup, so thresholds must remain configurable in `config.yaml`. |

---

## DEC-015 — YOLO11 ONNX Person Detector Architecture & Letterbox NMS Pipeline

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 3 |
| **Decision** | Implement `YOLO11PersonDetector` in `src/detection/` using ONNX Runtime CPU with letterbox preprocessing and vectorised NMS post-processing, filtering exclusively for COCO class `0` (`person`) |
| **Rationale** | Letterbox preprocessing preserves image aspect ratios without spatial distortion. Filtering output tensor row 4 extracts person category scores, and OpenCV `cv2.dnn.NMSBoxes` eliminates duplicate overlaps. Operates with zero PyTorch runtime dependencies. |
| **Alternatives** | Direct stretch resize (rejected due to spatial distortion); PyTorch runtime wrapper (rejected due to Windows Smart App Control DLL blocking). |

---

## DEC-016 — ByteTrack Multi-Object Tracking Architecture & Temporary Session-Local Track ID Definition

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 4 |
| **Decision** | Implement `ByteTracker` in `src/tracking/` using 2-stage IoU-Hungarian bipartite matching (`scipy.optimize.linear_sum_assignment`) and 8-state Kalman filtering, defining track IDs strictly as temporary session-local identifiers |
| **Rationale** | ByteTrack preserves both high and low-confidence detections to maintain track continuity across temporal occlusions. Track IDs are session-local visual tracking labels, NOT biometric, facial, or cross-session customer identities. Operates with minimal CPU overhead (0.50 ms per frame). |
| **Alternatives** | Re-identification / appearance embeddings (rejected per project scope boundary); DeepSORT / BoT-SORT (rejected per specific ByteTrack requirement). |

---

## DEC-017 — Spatial Homography Floor-Plane Transformation & Bottom-Center Trajectory Extraction

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 5 |
| **Decision** | Implement `HomographyCalibrator` in `src/spatial/` and `TrajectoryExtractor` in `src/trajectory/`, transforming bottom-center image points $[(x_1+x_2)/2, y_2]$ to dataset-specific 2D floor-plane coordinates while preserving original image coordinates |
| **Rationale** | The bottom-center of the bounding box represents the contact point of a person's feet with the floor plane, where 2D homography is geometrically valid. Preserving original image coordinates alongside floor coordinates enables debugging and overlays. Separate calibrations per camera view enforce dataset independence. |
| **Alternatives** | Bounding box center coordinates (rejected due to height/posture variation); single global homography matrix (rejected because homography is camera-dependent). |

---

## DEC-018 — Manual Store Zone Definition & Temporal Hysteresis Boundary Handling

| Field | Value |
|---|---|
| **Date** | 2026-09-22 |
| **Phase** | 6 |
| **Decision** | Implement `Zone` dataclass, `ZoneConfig` helper, `ZoneEvaluator` in `src/analytics/`, and `ZoneVisualizer` in `src/spatial/zone_visualizer.py`. Define zones in 2D floor-plane coordinates and apply a configurable temporal hysteresis buffer ($N=3$ default) for boundary transition stability |
| **Rationale** | Defining zones in floor-plane coordinate space ensures scale and perspective invariance across different camera angles. Temporal hysteresis buffer ($N=3$ frames) eliminates boundary jitter when trajectory points pass near polygon borders. `homography.py` remains focused on image-to-floor coordinate transformation, maintaining clear architectural separation of concerns. Points outside defined zones explicitly use `zone_id = None`. |
| **Alternatives** | Pixel-space zone definitions (rejected due to distortion across camera angles); zero hysteresis window (rejected due to boundary jitter); modifying `homography.py` to include zone visualization (rejected to preserve Phase 5 separation of concerns). |

---

## DEC-019 — Footfall Analytics Metric Definitions & Event Extraction Architecture

| Field | Value |
|---|---|
| **Date** | 2026-09-24 |
| **Phase** | 7 |
| **Decision** | Implement Footfall Analytics in `src/analytics/footfall.py` interpreting pre-assigned `TrajectoryPoint.zone_id` sequences into distinct metrics: unique tracks per zone, zone entry events, contiguous zone visits, basic visit durations, per-frame occupancy, and lightweight zone-to-zone transition matrix |
| **Rationale** | Decouples spatial evaluation (Phase 6) from discrete temporal event interpretation (Phase 7). Enforces strict mathematical distinctions between unique tracks, entry events, contiguous visits, and occupancy. Acknowledges track-fragmentation bias and sampling constraints while explicitly documenting that tracker IDs are session-local rather than verified customer identities. Defers SQLite, heatmaps, and layout optimization to subsequent phases. |
| **Alternatives** | Conflating unique tracks and entries (rejected due to bias on re-entrant visits); re-evaluating PIP within footfall (rejected to prevent redundant computation and architectural coupling); introducing SQLite prematurely (rejected to keep scope minimal and memory-efficient). |


