# Phase 15 — Final Integration and Evaluation Report

**Project**: Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
**Roadmap Phase**: Phase 15 — Final Integration and Evaluation  
**Status**: COMPLETE / OPERATIONALLY VERIFIED (Software Prototype Complete; Validated/Actionable Retail Layout Insights Not Established)  
**Initial Phase 15 Baseline**: 227/227 tests passing; **latest follow-up regression**: 256/256 passing  
**Active Local Viewer**: `http://localhost:8501` (Local Streamlit Server)  

---

## 1. Executive Summary & Prototype Status

Phase 15 concludes the software prototype implementation for the "Vision-Based Customer Trajectory Analytics for Retail Layout Optimization" project. All planned modules from Phase 0 through Phase 14 have been implemented, tested, and operationally verified within the local environment.

> [!IMPORTANT]
> **Methodological Invariant & Core Empirical Finding**:
> Successful software integration, clean test suites, and an interactive local dashboard **do NOT establish quantitative tracking accuracy, physical metric ground-truth calibration, customer dwell accuracy, or retail layout performance.**
>
> The software prototype is complete, but validated or actionable retail layout insights are not established:
> - **MERL Footage**: Operational engineering check on 100 sampled video frames of actual retail footage, but its reported result is sparse and boundary-truncated (71 of 74 points out-of-bounds due to single-camera field-of-view cutoff) and its floor calibration is unverified.
> - **MOT17 Sequence**: 10-frame image-sequence engineering check on outdoor pedestrians; a non-retail sequence and not a video container test.
> - **Phase 12 Spatial Observations**: Implements an evidence-gated spatial observation prototype. While MERL is retail footage, its unverified calibration and boundary truncation withhold observations; MOT17 is withheld due to its non-retail context and unverified calibration. Thus, Phase 12 produces no eligible retail observations from current inputs (`is_eligible: false`, `status: "insufficient_evidence"`). Software prototype is complete; validated or actionable retail layout insights are not.
> - **Phase 14 Viewer**: Operational local presentation layer displaying descriptive metrics and evidence gating boundaries without inferring image/report pairings from filenames.

Phase 15 represents **Final Integration and Evaluation** of the software prototype. In accordance with approved project decisions ([DEC-007](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/docs/DECISIONS.md)), Phase 15 does NOT involve cloud deployment, Docker, or enterprise cloud infrastructure.

---

## 2. Summary of Implemented Phases

| Phase | Description | Key Modules / Outputs | Status |
| :--- | :--- | :--- | :---: |
| **Phase 0–2** | Foundation, Video Ingestion & Frame Sampling | `src/video/`, OpenCV sampling, FPS alignment | Operationally Verified |
| **Phase 3–4** | YOLO11-n ONNX Person Detection & ByteTrack Tracking | `src/detection/`, `src/tracking/` | Operationally Verified |
| **Phase 5–7** | Planar Homography, Zones & Footfall Events | `src/spatial/`, `src/trajectory/` | Operationally Verified |
| **Phase 8** | Comprehensive Validation Framework | `src/validation/`, `scripts/verify_phase_08.py` (70 tests) | Operationally Verified |
| **Phase 9** | Dwell Foundations & Hysteresis Filtering | `src/analytics/dwell.py`, temporal span metrics | Operationally Verified |
| **Phase 10** | Trajectory Density Grid & Heatmap Rendering | `src/analytics/density.py`, `scripts/verify_phase_10.py` | Operationally Verified |
| **Phase 11** | Descriptive Zone Transitions & Activity Summaries | `src/analytics/transitions.py`, run counts | Operationally Verified |
| **Phase 12** | Evidence-Gated Spatial Observation Prototype | `src/analytics/spatial_observations.py` | Operationally Verified (No eligible retail insights from current data) |
| **Phase 13** | Local Evidence-Aware Report Export (`v1.0.0`) | `src/reporting/`, `outputs/reports/*.json` | Operationally Verified |
| **Phase 14** | Local Streamlit Report Viewer | `dashboard/report_loader.py`, `dashboard/app.py` | Operationally Verified |
| **Phase 15** | Final Integration & Evaluation | Full regression suite, runner audits, documentation | **COMPLETE** (Software Prototype Complete) |

---

## 3. Data Actually Used in Evaluation

No additional datasets were gathered or downloaded for Phase 15. The evaluation relies strictly on the actual data present in the repository:

1. **MERL Shopping Dataset (`data/merl/10_1_crop.mp4`)**:
   - Evaluated as a local recorded-video file (100 sampled frames at 5.0 target FPS) of actual retail store footage.
   - Floor calibration status is unverified configured units (`calibration_status: "unverified_configured_units"`), lacking independent physical metric survey.
   - Produced 1 continuous track trajectory.
   - Resulted in 74 evaluated points: 3 binned within floor bounds, 71 out-of-bounds due to customer positioning at the single-camera field-of-view cutoff (sparse, boundary-truncated).
   - Confirmed as an operational integration check; does not support retail layout recommendations or quantitative validation.
2. **MOT17-04 Image Sequence (`data/mot17/train/MOT17-04-FRCNN/img1`)**:
   - Evaluated as 10 consecutive image frames (frames 000001–000010).
   - Extracted 11 track trajectories representing outdoor pedestrians in a non-retail environment (`environment_type: "pedestrian_outdoor"`).
   - Resulted in 94 evaluated points: 37 binned, 57 out-of-bounds.
   - Confirmed as an image-sequence engineering check; does not represent a retail store environment or a video container test.
3. **Synthetic Reference Fixtures**:
   - Programmatically generated 2D coordinates used in unit and integration test suites.
   - Verify software logic, conservation invariants, hysteresis state machines, and schema validation.

---

## 4. Verification Commands & Execution Results

### 1. Full Regression Test Suite (227 tests)
```powershell
.\venv\Scripts\python.exe -m pytest -v -p no:cacheprovider
```
- **Result**: **227 passed in 10.66s** across 20 test files.
- **Coverage**: Includes YOLO11-n ONNX detection, ByteTrack tracking, trajectory extraction, homography, zones, footfall, hysteresis, dwell, density, transitions, spatial observations, report export, report loader, and Streamlit `AppTest` runtime smoke tests.

### 2. Phase 8 Comprehensive Validation Runner
```powershell
.\venv\Scripts\python.exe scripts/verify_phase_08.py
```
- **Result**: **SUCCESS** (70/70 component validation tests passed in 1.80s).
- **Scope**: Ground-truth schema, independent held-out homography, zone agreement, footfall/transition validation, track fragmentation, and hysteresis sensitivity.

### 3. Phase 10 Trajectory Density Integration Runner
```powershell
.\venv\Scripts\python.exe scripts/verify_phase_10.py
```
- **Result**: **SUCCESS**.
  - Frame-sequence pipeline on MOT17-04 (10 frames, 11 trajectories, 94 points).
  - Recorded-video-file pipeline on MERL (100 frames, 1 trajectory, 74 points).
  - Regenerated `phase_10_density_heatmap_mot17.png` and `phase_10_density_heatmap_merl.png` alongside their respective summary JSON files.

### 4. Phase 13 Schema & Evidence Gate Verification
- Confirmed that `phase_13_analysis_report_merl_sample.json` and `phase_13_analysis_report_mot17_sample.json` conform to `SCHEMA_VERSION = "1.0.0"`.
- Both reports correctly gate spatial observations with `is_eligible: false` and `status: "insufficient_evidence"`.
- Unsupplied sections (`dwell_analytics`, `transition_analytics`) are explicitly marked `"not_available"` with caller reasons, while evaluated zero metrics are preserved as numeric `0`.
- The `source_type` whitelist strictly accepts `"recorded_video"`, `"image_sequence"`, `"synthetic_fixture"`, and `"controlled_trial"`, while rejecting `"live_stream"`, `"rtsp_stream"`, and `"live_cctv"`.

---

## 5. UI Runtime & Visual Inspection Status

### Local Streamlit Server
- **Launch Command**: `streamlit run dashboard/app.py --server.headless true --server.port 8501`
- **Active URL**: `http://localhost:8501` (running via daemon task `task-2725`).
- **HTTP Check**: Verified responding with `200 OK` and serving the Streamlit single-page application.
- **AppTest Programmatic Smoke Tests**:
  - `test_streamlit_app_runtime_smoke_test_merl_view`: Verified default MERL view, provenance details, mandatory sparse/boundary-truncated context warnings, Phase 12 gating banner, density metrics, and omitted heatmap notice.
  - `test_streamlit_app_runtime_smoke_test_mot17_view`: Verified switching to MOT17 view, non-retail pedestrian context warning, and gating alert.
  - `test_streamlit_app_runtime_handles_unsupported_report`: Verified graceful error handling when selecting non-Phase-13 or malformed JSON reports.

### Visual Browser Inspection Disclosure
- **Tool Availability**: Interactive browser-based visual inspection tools (e.g. Playwright, Selenium, or browser screenshot capture) are **not available** in the execution environment.
- **Explicit Boundary**: Per project standards, **we do not claim verified visual responsiveness across desktop and narrow/mobile viewports based solely on AppTest execution or source layout declarations.**
- The project owner can perform direct visual inspection by opening `http://localhost:8501` in a standard desktop or mobile web browser.

---

## 6. Comprehensive Evidence Classification Matrix

| Component / Subsystem | Implemented | Operationally Verified | Quantitatively Validated | Retail-Performance Validated | Empirical Limitation & Boundary |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Video Ingestion & Frame Sampler** (`src/video/`) | **YES** | **YES** | **NO** | **NO** | Validated on local files; optical compression artifacts and variable frame rates require field verification. |
| **YOLO11-n ONNX Detection & ByteTrack** (`src/detection/`, `src/tracking/`) | **YES** | **YES** | **NO** | **NO** | Tested on sample frames using YOLO11-n ONNX; customer occlusions, retail carts, and tracking fragmentation are not benchmarked against physical ground truth. |
| **Planar Homography & Zones** (`src/spatial/`, `src/trajectory/`) | **YES** | **YES** | **NO** | **NO** | Evaluated on configured floor units and synthetic correspondences; physical camera height and lens distortion require surveyed ground truth. |
| **Validation Framework** (`src/validation/`, Phase 8) | **YES** | **YES** | **NO** | **NO** | Verifies software agreement mechanics; does not substitute for real-world retail store annotations. |
| **Dwell & Hysteresis Analytics** (`src/analytics/dwell.py`, Phase 9) | **YES** | **YES** | **NO** | **NO** | Measures observed temporal spans ($t_{\text{exit}} - t_{\text{entry}}$) between discrete detections; does not prove continuous physical dwell. |
| **Trajectory Density Grid** (`src/analytics/density.py`, Phase 10) | **YES** | **YES** | **NO** | **NO** | Accumulates discrete sampled detection counts; does not measure physical crowd occupancy or unique visitors. |
| **Zone Transitions & Activity** (`src/analytics/transitions.py`, Phase 11) | **YES** | **YES** | **NO** | **NO** | Aggregates clip-local tracking histories; does not represent unique human individuals or shopping journeys. |
| **Spatial Observation Prototype** (`src/analytics/spatial_observations.py`, Phase 12) | **YES** | **YES** | **NO** | **NO** | Evidence gate operational; produces no eligible retail observations from current inputs (`is_eligible: false`, `status: "insufficient_evidence"`). MERL is retail footage, but its calibration is unverified and trajectory distribution is boundary-truncated; MOT17 is a non-retail outdoor pedestrian sequence. Software prototype complete; validated or actionable retail layout insights are NOT. |
| **Report Export Engine** (`src/reporting/`, Phase 13) | **YES** | **YES** | **NO** | **NO** | Formats local JSON reports (`v1.0.0`); does not upgrade empirical evidence tier or validate accuracy. |
| **Local Streamlit Viewer** (`dashboard/`, Phase 14) | **YES** | **YES** (AppTest & Live Server) | **NO** | **NO** | Presentation utility running on `http://localhost:8501`; does not alter underlying data or perform live video analysis. |

---

## 7. In-Scope Methodological Limitations & Empirical Boundaries

The prototype is designed strictly as an in-scope, single-camera, recorded-video spatial trajectory analytics tool. The following methodological limitations and empirical boundaries define the software prototype and its current evidence base:

1. **Independent Metric Floor Calibration Prerequisite**:
   - Physical ground-truth survey of store floor coordinates (e.g. laser distance measurement of tile grid or verified physical landmarks) is required to establish metric scale ($x, y$ in meters). Unverified configured units remain non-metric engineering coordinates and cannot validate physical distance or speed metrics.
2. **Single-Camera Field-of-View & Boundary Truncation**:
   - The approved project scope is strictly single-camera. In single-camera footage such as the MERL sample, customer trajectories near frame margins undergo severe boundary truncation where tracked coordinates fall outside the floor boundary polygon (71 of 74 points out-of-bounds). Single-camera placement, mounting angle, and floor boundary coverage inherently limit spatial completeness; multi-camera coverage is outside approved project scope.
3. **Clip-Local Tracking & Tracker Fragmentation**:
   - Track IDs are strictly clip-local and anonymous. Occlusions, visual cross-overs, and detection drops cause tracker fragmentation and ID switching, which limit the interpretation of track counts and dwell duration metrics. Customer Re-ID and cross-temporal identity association remain strictly out of scope.
4. **Evidence Gating & Absence of Retail Layout Insights**:
   - Phase 12 implements an evidence-gated spatial observation prototype that programmatically evaluates input eligibility. Crucially, MERL is actual recorded retail footage, but its reported result is sparse/boundary-truncated (71 of 74 points out-of-bounds) and its calibration status is unverified; MOT17 is a non-retail outdoor pedestrian image sequence. Consequently, both inputs fail Phase 12 eligibility criteria, and Phase 12 produces no eligible retail observations from current inputs (`is_eligible: false`, `status: "insufficient_evidence"`). While the software prototype is fully implemented and operationally verified, validated or actionable retail layout insights are not established.
5. **Strictly Non-Causal Spatial Scope**:
   - The analytics pipeline provides descriptive spatial metrics (binned coordinate counts, temporal bounding intervals, zone sequence counts) from recorded video. Causal inference, shopper intent modeling, and layout redesign recommendations are outside the approved project scope and cannot be derived from spatial trajectory counts alone. Excluded integrations (such as sales transaction data or live streaming) are not supported.

---

*Phase 15 Final Integration and Evaluation complete. Software prototype implementation concluded; validated or actionable retail layout insights are not established.*


## 11. Full-Clip Demo Accuracy Follow-Up

After the initial Phase 15 report, the MERL visual demo was rerun over the complete local recorded clip using the configured 5.0 FPS sampler and existing detection, tracking, homography, zone, dwell, transition, and density components. The previous 100-frame generated artifacts were preserved in `outputs/reports/demo_real_100frame_archive_20261030/`. The reproducible command is:

```powershell
.\venv\Scripts\python.exe scripts/run_vscode_demo.py --max-frames 700
```

The clip yielded **633 sampled frames** (sampled time span 126.4 seconds), **512 person detections**, **3 clip-local tracks**, and **508 trajectory points**. Zone evaluation assigned 88 points to configured zones (53 to `shelf_a`, 35 to `main_aisle`) and left 420 unassigned. Dwell summaries contained 5 episodes and 17.2 seconds of observed temporal span; direct transition count was 2. Density conservation was verified: **508 = 169 binned + 339 out of bounds + 0 untransformed**; maximum cell count was 29.

The run exposed and corrected misleading demo reporting: per-zone assignment counts are now computed from actual evaluator outputs, and captions no longer label nonzero activity as evaluated zeros or reuse 100-frame counts. The dashboard reads current summary values, displays the sampled duration, and keeps its recommendation status **WITHHELD**. The full-clip artifacts are generated in `outputs/reports/` and displayed in the Demo Outputs view.

**Evidence boundary:** This longer run improves the operational example but does not validate retail measurements or recommendations. 339 of 508 points remain outside the configured density bounds; floor coordinates remain unsurveyed; no independent MERL track annotations or retail layout outcome reference are present. Retail layout recommendations remain withheld.

**Follow-up verification:** focused visual-demo/dashboard suites: 32 passed; full regression suite: 252 passed; Phase 8 verification runner: 70 passed; Phase 10 verification runner: success (its standard 100-frame MERL and MOT17 engineering paths).


## Full-Clip Pipeline Metrics Expansion (2026-10-31)

The offline demo runner and local Demo Outputs view were extended to expose additional metrics from existing project modules. The full locally available MERL clip was processed at the configured 5 FPS; no external footage, data, or packages were downloaded.

- Input: 920 × 680 H.264 video, 30 FPS, 3,796 native frames, 126.53 seconds. 633 frames were sampled across 126.4 seconds.
- Video-quality checks: PASS against configurable engineering warning thresholds (brightness 131.47, Laplacian sharpness variance 371.92; no warnings). This is not empirical accuracy validation.
- Detection: 512 person detections. Mean confidence 0.7891; latest full-run mean and P95 inference times 82.356 ms and 105.428 ms on ONNX Runtime CPU.
- Training status: NOT PERFORMED. The project uses pre-trained YOLO11-n ONNX weights; there is no project training run or training dataset.
- Tracking: 3 clip-local tracks with 508 trajectory points; 23 unexpected sampled-frame gaps are reported as descriptive diagnostics. Tracking accuracy, MOTA/IDF1, and fragmentation rate remain unavailable because no annotated MERL person-ID reference exists.
- Configured homography: 4 correspondences and 1 configured check point; training reprojection mean error 0.0 and configured check-point error 0.2727 normalized units. These are configuration consistency diagnostics, not independent floor survey validation.
- Zones and footfall: 88 points were assigned to configured zones (53 shelf_a, 35 main_aisle); 420 were unassigned. Five entry events and five contiguous visits were recorded. These are clip-local tracker events.
- Dwell: 5 observed episodes spanning 17.2 seconds. This is sampled temporal span, not continuous physical dwell.
- Occupancy timeline: at most 1 clip-local track was sampled in shelf_a or main_aisle; 0 in shelf_b and checkout. This is not physical occupancy or unique visitors.
- Direct transitions: 2 observed transitions.
- Density: 508 = 169 binned + 339 out of bounds + 0 untransformed; maximum cell count 29.
- Evidence status: Phase 12 remains insufficient evidence; layout recommendations remain WITHHELD.

Additional outputs: outputs/reports/demo_real_tracking_video.webm, outputs/reports/demo_real_zone_occupancy.png, and expanded outputs/reports/demo_real_summary.json. The Streamlit view displays input/video quality, pretrained model and training status, homography diagnostics, footfall, occupancy, tracking diagnostics, dwell, density, transitions, and recommendation eligibility.

Verification after the change: 33 focused demo/dashboard tests passed; 253 tests passed in the full repository suite; the Phase 8 runner passed 70 tests; the Phase 10 integration runner completed successfully. The Phase 10 runner remains a separate 100-frame integration check, distinct from this 633-frame full local clip demo.

This expands operational reporting only. No quantitative retail validation or actionable layout optimization was established.

## Local Upload and Exploratory Shelf Candidate Follow-Up (2026-11-01)

The local Streamlit demo now provides an indoor recorded-video upload workflow. Analysis begins only after the user selects a video and presses **Analyze uploaded video**. It runs existing `FrameSampler`, YOLO11-n ONNX detection, ByteTracker, and trajectory extraction locally; the annotated WebM and JSON summary can be downloaded. No online stream, external inference service, or project model training is involved. The pretrained detector was not trained by this project, and indoor training annotations are absent.

For arbitrary camera views, the app limits output to detection/tracking. Homography, retail zones, density, dwell, transitions, and shelf candidate output are marked unavailable unless the user explicitly selects the MERL camera profile and confirms the footage uses the same camera view and crop. That selection is user-provided and does not validate calibration.

For the current full MERL demo, the app displays `shelf_a` as an **exploratory product-placement candidate**, based on 10.2 seconds of observed dwell span, 53 sampled in-zone points, and 3 observed episodes; `shelf_b` has zero observed values for those metrics. The deterministic comparison uses dwell span, then sampled points, episodes, and stable zone ID. It uses no learned weights or invented thresholds. This is an example of a recommendation-style software output only: validated/actionable layout recommendations remain **WITHHELD** because the floor scale is unverified and the run is boundary-truncated.

The staff-facing UI now shows the MERL indoor recording, the explicitly labeled synthetic fixture, and the local upload workflow. The outdoor MOT17 sample remains available to the engineering verification runner and tests, but is excluded from the staff-facing report selector and demo tabs.

Verification after this follow-up: focused runner/dashboard tests **36 passed**; full repository regression **256 passed**; Phase 8 runner **70 passed**; Phase 10 runner completed successfully (MOT17 image-sequence engineering check and MERL 100-frame video-file check). A separate 100-frame local MERL smoke run exercised the upload tracking-only branch and produced 74 detections, 1 clip-local track, and a decodable annotated WebM. The live Streamlit page was inspected: the MERL tab displays its video controls, download buttons, measured metrics, candidate comparison, and evidence withholding; the upload tab displays the local upload control and camera-calibration limitation.
