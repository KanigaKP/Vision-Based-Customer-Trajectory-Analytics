# Phase 10.1 — Floor-Plane Trajectory Density Heatmap Foundation

**Project**: Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
**Roadmap Phase**: Phase 10 — Trajectory Density / Heatmaps (Slice 10.1: Density Foundation & Heatmap Rendering)  
**Status**: COMPLETE / OPERATIONALLY VERIFIED  
**Regression Baseline**: 162/162 tests passing (55 baseline + 82 Phase 8 validation + 16 Phase 9 dwell + 9 Phase 10 density tests)  
**Verification Runner**: [`scripts/verify_phase_08.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/scripts/verify_phase_08.py) (All 70 component tests pass cleanly)  

---

## 1. Executive Summary & Objective

Phase 10.1 introduces the foundational spatial binning engine and rendering architecture for **floor-plane trajectory density heatmaps**. Operating strictly downstream of Phase 5's homography coordinate transformation, this module converts continuous 2D floor-plane trajectory points into a deterministic discrete spatial grid and renders technically faithful top-down density heatmaps.

### Core Architectural Invariants:
1. **Pure Spatial Binning Layer**: Consumes pre-transformed floor coordinates (`TrajectoryPoint.floor_x`, `floor_y`). Never reruns detection, tracking, homography, Point-in-Polygon zone assignment, or dwell analysis.
2. **Explicit Bounds & Grid Resolution**: Grid spatial bounds (`min_x`, `max_x`, `min_y`, `max_y`) and cell resolution must be explicitly provided via configuration or caller parameters. The system **never** dynamically infers bounds from observed point extrema, preventing spatial distortion, grid-drift, and frame-dependent scaling.
3. **Discrete Technical Rendering**: Heatmaps are rendered using nearest-neighbor interpolation (`interpolation='nearest'`), preserving the exact discrete cell counts without applying artificial Gaussian blurring, smoothing filters, Kernel Density Estimation (KDE), or synthetic hotspot thresholds.
4. **Clip-Local Identity & Camera Independence**: Operates on session-local tracking IDs within a single camera calibration view. No cross-camera stitching, visual Re-ID, or identity merging is performed.

---

## 2. Spatial Grid Semantics & Boundary Handling

### 2.1 Coordinate System, Scale & Orientation
- **Coordinate Space**: 2D Cartesian floor-plane space where $+X$ extends to the right (width) and $+Y$ extends upwards (height), measured in **configured floor-coordinate units**.
- **Physical Scale Qualification**:
  - In `config.yaml`, the MERL homography calibration is explicitly declared with `unit: "normalized_units"` (`coordinate_type: "aisle_ground_grid"`). A numeric extent such as $[0.0, 10.0] \times [0.0, 6.0]$ is an arbitrary normalized coordinate grid, **not a physically surveyed 10-metre by 6-metre floor**.
  - Physical metric scale has **not been independently verified or surveyed on-site** for these video sequences.
  - Consequently, axis titles and documentation use the neutral designation **“configured floor units”** rather than asserting physical metres.
- **Matrix Mapping**:
  - Grid matrix `counts` has shape `(num_bins_y, num_bins_x)`.
  - Row index $r \in [0, N_y - 1]$ indexes Y-axis bins from `min_y` to `max_y`.
  - Column index $c \in [0, N_x - 1]$ indexes X-axis bins from `min_x` to `max_x`.
  - Rendering with `origin='lower'` and `extent=[min_x, max_x, min_y, max_y]` aligns row 0 with `min_y` (bottom), maintaining 1:1 physical alignment with floor coordinates.

### 2.2 Binning Formulation & Boundary Rules
For a grid with bounds $[x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$, bin widths are $\Delta x = \frac{x_{\max} - x_{\min}}{N_x}$ and $\Delta y = \frac{y_{\max} - y_{\min}}{N_y}$.

For each valid observation $(x, y)$:
1. **Half-Open Cell Intervals**: An interior cell $(r, c)$ encompasses $[x_{\min} + c\Delta x, x_{\min} + (c+1)\Delta x)$ along X, and $[y_{\min} + r\Delta y, y_{\min} + (r+1)\Delta y)$ along Y.
2. **Upper Boundary Clamping**: Observations lying exactly on the upper boundary ($x = x_{\max}$ or $y = y_{\max}$) are clamped into the outermost bins ($c = N_x - 1$ or $r = N_y - 1$).
3. **Out-of-Bounds Points**: Any observation with $x < x_{\min}$, $x > x_{\max}$, $y < y_{\min}$, or $y > y_{\max}$ is excluded from the grid and recorded in the diagnostic counter `out_of_bounds_count`.
4. **Untransformed / Missing Coordinates**: Points where `floor_x` or `floor_y` is `None` or non-finite (`NaN`, `Inf`) are excluded from the grid and recorded in `untransformed_count`.
5. **Conservation Invariant**:
   $$\text{total\_evaluated\_points} = \text{total\_binned\_points} + \text{out\_of\_bounds\_count} + \text{untransformed\_count}$$
6. **Empty Input**: If the trajectory list is empty or contains zero valid points, the module returns a zero-filled array of shape $(N_y, N_x)$ with `total_binned_points = 0` and `max_cell_count = 0`.

---

## 3. What the Counts Represent and Do NOT Represent

> [!IMPORTANT]
> **Strict Semantic Boundary on Density Values**:
> - **What Counts Represent**:
>   Each cell value represents the **discrete count of sampled trajectory observations** whose floor coordinates fell within that spatial bin during video sampling at the target frame rate (e.g., 5 FPS). Each valid detection contributes strictly $+1$ to its containing cell.
> - **What Counts DO NOT Represent**:
>   - **Unique Visitors**: Raw point counts do *not* measure unique individuals. A single customer standing still contributes 5 counts per second at 5 FPS (150 counts in 30 seconds).
>   - **Continuous Physical Dwell**: Points are discrete frame samples; they do not measure unobserved continuous dwell.
>   - **Physical Occupancy**: Raw counts do not represent simultaneous instantaneous crowd density or capacity utilization.
>   - **Customer Conversion / Retail Performance**: Spatial observation frequency cannot be interpreted as purchasing interest, product engagement, sales conversion, or layout efficiency without empirical point-of-sale (POS) and customer behavior data.

---

## 4. API & Data Structures

### 4.1 `GridBounds` (Frozen Dataclass)
Defines explicit spatial extents:
```python
@dataclass(frozen=True)
class GridBounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
```
- Validates that coordinates are finite numbers and that `max_x > min_x` and `max_y > min_y`.
- Helper `from_tuple((min_x, max_x, min_y, max_y))` enables seamless integration with existing configuration bounds.

### 4.2 `TrajectoryDensityGrid` (Dataclass)
Container for spatial density outputs:
- `bounds: GridBounds`
- `counts: np.ndarray` (2D integer array, shape `(num_bins_y, num_bins_x)`)
- `num_bins_x: int`, `num_bins_y: int`
- `cell_size_x: float`, `cell_size_y: float`
- `total_evaluated_points: int`
- `total_binned_points: int`
- `out_of_bounds_count: int`
- `untransformed_count: int`
- `max_cell_count: int` (property)
- `to_dict() -> Dict[str, Any]`

### 4.3 Core Functions
- [`compute_density_grid(trajectories, bounds, cell_size=0.5, grid_shape=None)`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/density.py): Evaluates trajectories and returns `TrajectoryDensityGrid`.
- [`render_density_heatmap(grid, output_path=None, title=..., cmap='viridis', zones=None)`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/density.py): Renders the grid using `matplotlib.pyplot.imshow(..., interpolation='nearest')` and saves to PNG with optional zone polygon outlines and colorbar.

---

## 5. Verification & Test Inventory

The focused test suite [`tests/test_density.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/tests/test_density.py) provides 11 comprehensive unit and integration tests:

| Test Name | Verifies |
|---|---|
| `test_grid_bounds_validation` | Dimension ordering, finite number enforcement, containment logic. |
| `test_empty_trajectories_input` | Empty trajectory list and 0-point trajectories return zero-filled grid without exceptions. |
| `test_known_cell_counts_and_orientation` | Known coordinate mapping to specific $(row, col)$ cells, Cartesian orientation (row 0 bottom, col 0 left). |
| `test_boundary_coordinates_semantics` | Half-open interval inclusion and upper-boundary clamping ($x=x_{\max}, y=y_{\max}$). |
| `test_out_of_bounds_and_untransformed_points` | Handling of $X/Y$ coordinates beyond bounds, `None`/`NaN` points, and conservation invariant. |
| `test_multi_track_accumulation` | Multi-track additive accumulation into shared grid cells without ID merging. |
| `test_explicit_grid_shape_parameter` | Setting exact grid shape $(N_x, N_y)$ and bin bounds calculation. |
| `test_to_dict_serialization` | Complete dictionary / JSON serializability of grid and metadata. |
| `test_rendering_produces_nonblank_heatmap` | `render_density_heatmap` produces readable, non-empty PNG file with optional zone overlays and verifies coordinate labels. |
| `test_density_pipeline_integration_boundary` | End-to-end integration boundary from Trajectory objects through density grid, zone overlay, and JSON export without external dependencies. |
| `test_rendering_does_not_mutate_matplotlib_backend` | Verifies isolated non-interactive rendering does not mutate the process-wide Matplotlib backend. |
| `test_sparse_and_boundary_truncated_trajectories_handling` | Validates density grid and rendering with sparse, boundary-truncated trajectories (matching MERL video pattern). |

---

## 6. Verification Heatmap Artifact & Labeling Audit (Phase 10.1)

An operational verification heatmap was rendered and inspected at `outputs/reports/phase_10_density_heatmap_sample.png`:

- **Synthetic Fixture Purpose**: Evaluates operational rendering mechanics (Cartesian orientation, nearest-neighbor discrete cell rendering, axis boundary scaling, colorbar legend, and zone polygon overlay).
- **Explicit Labeling Invariant**: The generated image is titled:
  > **Synthetic Verification Fixture — Trajectory Density Heatmap**  
  > *(Configured MERL Grid Bounds; Synthetic Trajectories — Not Real Shopper Data)*
- **Data Provenance**: Uses synthetic programmatic trajectories over configured MERL grid bounds ($[0.0, 10.0] \times [0.0, 6.0]$ configured units, cell size $0.5 \times 0.5$ configured units). It does **not** contain or represent real human shopper trajectories from the MERL shopping dataset participants.
- **Coordinate Units**: Labeled as `"configured floor units"` on both axes, explicitly avoiding unwarranted claims of surveyed metric ground truth.

---

## 7. Frame-Sequence Pipeline Integration Check (Phase 10.2)

### 7.1 Pipeline Architecture
Phase 10.2 connects the discrete spatial density engine to sequential image frames (`img1/*.jpg`) decoded via `cv2.imread()`:
```
Sequential Image Frames (cv2.imread)
         │
         ▼
[ YOLO11PersonDetector ]  (data/models/yolo11n.onnx)
         │  Detections (bounding boxes + confidence)
         ▼
[ ByteTracker ]           (IoU-based association & track life-cycle)
         │  Active / Lost / Removed Tracks
         ▼
[ TrajectoryExtractor ]   (Clip-local trajectory collection)
         │  Trajectories with image coordinates (bottom-center)
         ▼
[ HomographyCalibrator ]  (dataset-specific planar homography: mot17_04)
         │  Trajectories with (floor_x, floor_y)
         ▼
[ compute_density_grid ]  (Phase 10 spatial binning with explicit GridBounds)
         │  TrajectoryDensityGrid (counts, diagnostics, metadata)
         ▼
[ render_density_heatmap ] & [ to_dict() JSON Export ]
```

### 7.2 MOT17-04 Engineering Integration Check
- **Data Source**: Local 10-frame image sequence [`data/mot17/train/MOT17-04-FRCNN/img1`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/data/mot17/train/MOT17-04-FRCNN/img1) (frames 000001–000010 read via `cv2.imread`).
- **Input Type & Boundary**: Ingests ten consecutive JPEG image files from disk. This is a **frame-sequence / frame-level pipeline integration check**, not a video-container decoding run.
- **Detector & Tracker**: YOLO11n ONNX detector (`conf=0.40`, `iou=0.45`) and ByteTracker (`high_thresh=0.50`, `max_lost=2.0s`).
- **Calibration**: `HomographyCalibrator.from_config(config, dataset_name="mot17_04")` mapping image coordinates to configured floor plane.
- **Bounds & Resolution**: `GridBounds(min_x=0.0, max_x=10.0, min_y=0.0, max_y=8.0)` with `cell_size=0.5` (producing a $16 \times 20$ grid of 320 cells).
- **Quantitative Results**:
  - Total evaluated points: 94
  - Successfully binned points: 37 (mapped into top cells $y \in [7.5, 8.0]$ in the South Corridor Zone)
  - Out-of-bounds points: 57 ($x < 0$ or $x > 10$, strictly tracked in diagnostic counter)
  - Conservation invariant verified: $94 = 37 + 57 + 0$.
  - Peak cell count: 10 observations.
- **Outputs**:
  - Image: `outputs/reports/phase_10_density_heatmap_mot17.png`
  - Summary JSON: `outputs/reports/phase_10_density_summary_mot17.json`
- **Labeling & Evidence Classification**:
  - Title: *"MOT17-04 Pedestrian Trajectory Density Heatmap (Engineering Integration Check — Not Retail Validation)"*
  - Axis labels: *"Floor X Coordinates (configured floor units)"* and *"Floor Y Coordinates (configured floor units)"*
  - **Explicit Limitation**: Labeled strictly as a *limited pedestrian-tracking engineering integration check*. It does **not** constitute retail customer validation, an MOT benchmark, or physical metric ground-truth calibration.

---

## 8. Recorded-Video-File Pipeline Integration Check (Phase 10.3)

### 8.1 Video Decoding & Pipeline Ingestion
Phase 10.3 evaluates the full recorded-video-file decoding and trajectory density path:
```
Recorded Video File (.mp4 container)
         │
         ▼
[ FrameSampler (cv2.VideoCapture) ]  (Lazy 5.0 FPS sampling, 100 frames)
         │  Decoded BGR frames + timestamps
         ▼
[ YOLO11PersonDetector ]  (data/models/yolo11n.onnx)
         │  Detections (bounding boxes + confidence)
         ▼
[ ByteTracker ]           (IoU-based association & track life-cycle)
         │  Active / Lost / Removed Tracks
         ▼
[ TrajectoryExtractor ]   (Clip-local trajectory collection)
         │  Trajectories with image coordinates (bottom-center)
         ▼
[ HomographyCalibrator ]  (dataset-specific planar homography: merl)
         │  Trajectories with (floor_x, floor_y)
         ▼
[ compute_density_grid ]  (Phase 10 spatial binning with explicit GridBounds)
         │  TrajectoryDensityGrid (counts, diagnostics, metadata)
         ▼
[ render_density_heatmap ] & [ to_dict() JSON Export ]
```

### 8.2 MERL Video Integration Details & Diagnostics
- **Video Source Clip**: Local video file [`data/merl/10_1_crop.mp4`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/data/merl/10_1_crop.mp4) (920x680 resolution, 30.0 FPS native, 3796 total frames, duration 126.53s).
- **Video Reader**: Existing [`FrameSampler`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/video/sampler.py) class (decoding via `cv2.VideoCapture`).
- **Sampling Parameters**: Configured target sampling rate of `5.0 FPS` (`step = 6` native frames); 100 sampled frames decoded across $t \in [0.0\text{s}, 19.8\text{s}]$ (processing time: 12.63s).
- **Detection & Tracking**: YOLO11n ONNX detector (`data/models/yolo11n.onnx`) and ByteTracker tracking 1 customer trajectory containing 74 trajectory observations.
- **Spatial Mapping**: `HomographyCalibrator.from_config(config, dataset_name="merl")` projects bottom-center pixel locations to configured floor coordinates.
- **Grid Configuration**: `GridBounds(min_x=0.0, max_x=10.0, min_y=0.0, max_y=6.0)` configured floor units, `cell_size = 0.5` ($12 \text{ rows} \times 20 \text{ cols} = 240\text{ cells}$).
- **Quantitative Diagnostics & Conservation**:
  - Total evaluated points: 74
  - Successfully binned points: 3 (in top-right corner $x \in [9.5, 10.0], y \in [5.0, 6.0]$)
  - Out-of-bounds points: 71 (pedestrian walked near bottom camera margin, mapping to $y > 6.0$ or $x > 10.0$)
  - Untransformed points: 0
  - **Conservation Invariant Verified**: $\text{total\_evaluated} (74) = \text{binned} (3) + \text{out\_of\_bounds} (71) + \text{untransformed} (0)$.
  - Max cell count: 2 observations.
- **Generated Artifacts**:
  - Image: [`outputs/reports/phase_10_density_heatmap_merl.png`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/outputs/reports/phase_10_density_heatmap_merl.png)
  - Summary JSON: [`outputs/reports/phase_10_density_summary_merl.json`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/outputs/reports/phase_10_density_summary_merl.json)
- **Labeling Standard & Explicit Limitations**:
  - Heatmap title: *"MERL Retail Video Trajectory Density Heatmap\n(Sparse / Boundary-Truncated Sample — Not Quantitative Validation)"*
  - Axis labels: *"Floor X Coordinates (configured floor units)"* and *"Floor Y Coordinates (configured floor units)"*
  - **Boundary Classification**: Confirms operational execution of the recorded-video-file decoding path. Observations are **sparse and boundary-truncated** (71 of 74 points outside configured retail zone bounds). This output is **not** presented as useful retail customer analytics, shopper behavior insights, or quantitative validation.

---

## 9. Matplotlib Backend Isolation

- [`src/analytics/density.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/density.py) uses pure object-oriented Matplotlib rendering via `matplotlib.figure.Figure` and `matplotlib.backends.backend_agg.FigureCanvasAgg`.
- Does **not** invoke `matplotlib.use("Agg")` at module import, eliminating global side effects on callers' plotting environments.
- Preserves the process-wide backend before and after rendering (verified in `test_rendering_does_not_mutate_matplotlib_backend`).

---

## 10. Verification Script & Reproducibility

The complete Phase 10 verification suite is executed via:
```powershell
python scripts/verify_phase_10.py
```
This script executes both:
1. `run_mot17_density_pipeline`: Frame-sequence integration check on MOT17-04 local image files.
2. `run_merl_video_density_pipeline`: Recorded-video-file integration check on MERL `10_1_crop.mp4` via `FrameSampler`.

---

## 11. Evidence Classification Matrix

| Category | Status | Boundary & Justification |
| :--- | :---: | :--- |
| **IMPLEMENTED** | **YES** | Discrete spatial binning ([`compute_density_grid`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/density.py)), container ([`TrajectoryDensityGrid`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/density.py)), bounds ([`GridBounds`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/density.py)), isolated faithful rendering ([`render_density_heatmap`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/density.py)), and dual-mode pipeline integration runner ([`scripts/verify_phase_10.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/scripts/verify_phase_10.py)) are implemented. |
| **OPERATIONALLY VERIFIED** | **YES** | 12 unit and integration tests pass in [`tests/test_density.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/tests/test_density.py); all 70 Phase 8 verification tests pass in [`scripts/verify_phase_08.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/scripts/verify_phase_08.py); full 165-test regression suite passes; frame-level integration verified on MOT17-04 image frames; video-file decoding and trajectory binning verified on MERL video; backend preservation verified. |
| **QUANTITATIVELY VALIDATED** | **NO** | Evaluated on MOT17-04 engineering sample, MERL boundary-truncated video sample, and synthetic fixtures. Spatial mapping accuracy depends on camera planar homography consistency, not sub-centimeter physical surveyor verification. Coordinates are labeled as configured floor units. |
| **RETAIL-PERFORMANCE VALIDATED** | **NO** | Raw sampled observation counts are spatial tracking artifacts; no claims regarding shopper interest, dwell-to-purchase correlation, layout optimization, or sales uplift are asserted. The MERL result is sparse and boundary-truncated; a usable retail analytics heatmap was not established. |

---

> [!NOTE]
> Phase 10.3 is complete. Execution has halted per instruction without proceeding to later phases.
