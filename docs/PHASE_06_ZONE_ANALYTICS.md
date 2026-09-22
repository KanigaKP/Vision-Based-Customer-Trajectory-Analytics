# Phase 6 Completion Report — Manual Zone Definition & Spatial Zone Analytics

**Project**: Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
**Date**: September 22, 2026  
**Status**: COMPLETE / PASS  

---

## 1. Executive Summary & Objective

Phase 6 implements the **Manual Zone Definition & Spatial Zone Analytics** layer for the retail analytics pipeline. This phase consumes 2D floor-space trajectory points $(x_{\text{floor}}, y_{\text{floor}})$ generated in Phase 5, evaluates which defined retail store zone each point falls within using Point-in-Polygon (PIP) testing, applies a configurable temporal hysteresis buffer ($N=3$ consecutive frames default) to eliminate boundary jitter during transitions, and visualizes zone overlays on top-down floor maps.

> [!IMPORTANT]
> **Scope Disclaimer**:  
> "Phase 6 establishes spatial zone assignment only. Footfall, dwell, heatmap generation, zone classification, and layout optimization insights are implemented in later phases."

---

## 2. Architecture Integration

The Phase 6 spatial zone analytics layer builds directly upon the output of Phase 5 while preserving architectural decoupling:

```
Input Video Frame
      ↓
[Phase 2] FrameSampler (5 FPS)
      ↓
[Phase 3] YOLO11PersonDetector (ONNX Runtime CPU)
      ↓
[Phase 4] ByteTracker (Session-Local Track IDs + BBoxes)
      ↓
[Phase 5] TrajectoryExtractor + HomographyCalibrator (Image → 2D Floor Coordinates)
      ↓
[Phase 6] ZoneEvaluator + Temporal Hysteresis (Floor Coordinates → Zone Assignment)
      ↓
[Phase 6] ZoneVisualizer (Top-Down Floor Map Overlay Renderings)
```

### Architectural Separation of Concerns:
- **Phase 5 (`src/spatial/homography.py`)**: Responsible exclusively for perspective transformation from 2D pixel space to 2D floor-coordinate space.
- **Phase 6 (`src/analytics/zone.py`, `src/analytics/zone_evaluator.py`)**: Responsible for floor-coordinate point-in-polygon evaluation, YAML persistence, and hysteresis filtering.
- **Phase 6 (`src/spatial/zone_visualizer.py`)**: Dedicated module responsible for top-down floor map rendering of zone overlays, labels, and trajectory paths.

---

## 3. Data Model & Zone Representation

Zones are defined as 2D polygons in metric or ground-grid floor coordinates $(x_{\text{floor}}, y_{\text{floor}})$, matching the output space of `HomographyCalibrator`.

### Zone Dataclass (`src/analytics/zone.py`)

| Attribute | Type | Description | Example |
|---|---|---|---|
| `zone_id` | `str` | Unique identifier string | `"shelf_a"` |
| `name` | `str` | Human-readable zone label | `"Shelf A (Snacks & Beverages)"` |
| `polygon` | `List[Tuple[float, float]]` | Vertices defining polygon boundary | `[(0.5, 1.0), (4.0, 1.0), (4.0, 5.0), (0.5, 5.0)]` |
| `color` | `str` | HEX or color string for rendering | `"#3498db"` |
| `description` | `Optional[str]` | Detailed retail section description | `"Left product shelf display section"` |

### Data Model Rules:
- Points outside all defined store zones explicitly receive `zone_id = None`.
- Polygons must contain at least 3 non-degenerate vertices.

---

## 4. Point-in-Polygon (PIP) & Temporal Hysteresis Algorithm

### A. Point-in-Polygon Evaluation
Each 2D floor coordinate point $(x_{\text{floor}}, y_{\text{floor}})$ is evaluated against defined zone polygons using `matplotlib.path.Path.contains_point()`.

### B. Configurable Temporal Hysteresis Filtering
When a person walks near or along zone boundaries, spatial measurement noise can cause rapid toggling between adjacent zones. To prevent boundary jitter, `ZoneEvaluator` implements a temporal hysteresis buffer:

- **Candidate State Buffer**: Tracks candidate zone transitions across $N$ consecutive sampled frames.
- **Configurable Window ($N=3$ default)**: A zone transition is confirmed only after $N$ consecutive frames fall into the new zone. Single-frame boundary noise or brief boundary passes do not trigger a zone change.
- **Note**: $N=3$ is a configurable engineering threshold, NOT a scientifically optimal limit.

---

## 5. Configuration & Persistence (`config.yaml`)

Zone definitions and hysteresis parameters are loaded from `config.yaml`:

```yaml
zones:
  hysteresis_window: 3

  merl:
    - zone_id: "shelf_a"
      name: "Shelf A (Snacks & Beverages)"
      polygon:
        - [0.5, 1.0]
        - [4.0, 1.0]
        - [4.0, 5.0]
        - [0.5, 5.0]
      color: "#3498db"

    - zone_id: "shelf_b"
      name: "Shelf B (Produce & Dairy)"
      polygon:
        - [6.0, 1.0]
        - [9.5, 1.0]
        - [9.5, 5.0]
        - [6.0, 5.0]
      color: "#2ecc71"

    - zone_id: "main_aisle"
      name: "Central Aisle"
      polygon:
        - [4.0, 1.0]
        - [6.0, 1.0]
        - [6.0, 5.0]
        - [4.0, 5.0]
      color: "#f39c12"

    - zone_id: "checkout"
      name: "Checkout Counter Area"
      polygon:
        - [0.0, 0.0]
        - [10.0, 0.0]
        - [10.0, 1.0]
        - [0.0, 1.0]
      color: "#e74c3c"
```

---

## 6. Verification & Test Results

### A. Automated Unit Tests (`pytest -v`)
All **44 unit tests** (37 existing baseline + 7 new Phase 6 zone tests) passed:

1. `test_zone_creation_and_validation`: Validates dataclass vertex checking and parameter validation (`PASSED`).
2. `test_point_inside_zone`: Validates PIP test returns correct `zone_id` for interior points (`PASSED`).
3. `test_point_outside_zone`: Validates PIP test explicitly returns `None` for exterior points (`PASSED`).
4. `test_boundary_hysteresis`: Validates single-frame noise does not trigger zone changes ($N=3$) (`PASSED`).
5. `test_zone_transition_after_hysteresis`: Validates transition confirmation after $N=3$ consecutive frames (`PASSED`).
6. `test_multi_zone_evaluation`: Validates evaluation across multiple retail zones (`PASSED`).
7. `test_zone_config_save_load`: Validates YAML configuration serialization and deserialization (`PASSED`).

```
============================= 44 passed in 5.68s ==============================
```

### B. End-to-End Verification Script (`scripts/verify_phase_06.py`)
Execution output:
```
===========================================================================
PHASE 6 — MANUAL ZONE DEFINITION & SPATIAL ZONE ANALYTICS VERIFICATION
===========================================================================

[1/2] Verifying MERL Supermarket Retail Zone Pipeline...
  Loaded 4 retail zones for MERL (Hysteresis Window N = 3).
    - [shelf_a] 'Shelf A (Snacks & Beverages)': 4 vertices, Color: #3498db
    - [shelf_b] 'Shelf B (Produce & Dairy)': 4 vertices, Color: #2ecc71
    - [main_aisle] 'Central Aisle': 4 vertices, Color: #f39c12
    - [checkout] 'Checkout Counter Area': 4 vertices, Color: #e74c3c
  Evaluated 20 trajectory points across 2 tracks.
    - Zone 'shelf_a': 9 point visits (45.0%)
    - Zone 'shelf_b': 10 point visits (50.0%)
    - Zone 'main_aisle': 1 point visits (5.0%)
    - Zone 'checkout': 0 point visits (0.0%)
    - Zone 'Unassigned / Outside Zones': 0 point visits (0.0%)
  Saved MERL zone visualization map: outputs\reports\phase_06_zone_analytics_merl.png

[2/2] Verifying MOT17-04 Benchmark Corridor Zone Pipeline...
  Loaded 2 benchmark zones for MOT17_04 (Hysteresis Window N = 3).
    - [zone_north] 'North Corridor Zone': 4 vertices, Color: #9b59b6
    - [zone_south] 'South Corridor Zone': 4 vertices, Color: #1abc9c
  Saved MOT17-04 zone visualization map: outputs\reports\phase_06_zone_analytics_mot17.png

===========================================================================
VERIFICATION SUCCESSFUL: Verification script completes successfully
and all expected outputs are generated without errors.
===========================================================================
```

### C. Visual Inspection of Rendered Outputs
- `outputs/reports/phase_06_zone_analytics_merl.png`: Shows MERL store zones (Shelf A, Shelf B, Central Aisle, Checkout) with semi-transparent fills, dashed outlines, centered text labels, color-coded trajectory points, and marked start/end trajectory locations.
- `outputs/reports/phase_06_zone_analytics_mot17.png`: Shows MOT17 corridor north/south zones with complete trajectory point assignments.

---

## 7. Limitations & Scope Boundaries

1. **Manual Definition Only**: Zones are defined via YAML coordinates. Automated zone discovery or auto-segmentation is outside scope.
2. **2D Plane Assumption**: Point-in-polygon evaluation assumes flat floor planes without vertical customer height dimensions.
3. **No Footfall / Dwell Aggregation**: Temporal duration calculations, visit counts, dwell heatmaps, and layout optimization recommendations belong to Phase 7+.

---

## 8. Git Commit Record

- **Commit Message**: `Phase 6 Complete: Manual Zone Definition & Spatial Zone Analytics Layer`
