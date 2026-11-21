# UI/UX Foundation & Stitch Design Prototype Specification
**Vision-Based Customer Trajectory Analytics for Retail Layout Optimization**

- **Document Version:** 1.1.0 (Rigorous Analytics & Repository Ground Truth Correction)
- **Status:** Architectural Foundation & Design Prototype Corrected
- **Date:** 2026-11-03
- **Author:** Senior Product Designer & Frontend Architect
- **Stitch Project:** [`Vision Trajectory Analytics - Retail Spatial System`](https://stitch.withgoogle.com/projects/14372996755834353810)
- **Stitch Project Resource ID:** `projects/14372996755834353810`
- **Local Interactive Clickable Prototype:** [`docs/design_prototype/index.html`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/docs/design_prototype/index.html)

---

## 1. Executive Summary & Correction Objectives

This specification defines the product UX architecture, design system, and clickable prototype for the **Vision-Based Customer Trajectory Analytics** prototype.

Following a rigorous repository audit against production configuration ([`config.yaml`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/config.yaml)), scene profile specifications ([`configs/merl_scene_profile.json`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/configs/merl_scene_profile.json)), analytics modules ([`src/analytics/`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/)), and report manifests ([`outputs/reports/`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/outputs/reports/)), this revision applies mandatory ground-truth corrections:
1. **Purged Metric Coordinate Inventions:** Removed all claims of physical meters, measured speeds ($m/s$), and real-world dimensions for the single-camera MERL retail dataset. Coordinates are strictly represented in **configured floor units** within the configured bounds $X \in [0.0, 10.0]$ and $Y \in [0.0, 6.0]$.
2. **Purged Invented Evidence Rules:** Eliminated the fabricated $Y > 7.5\text{m}$ perspective cutoff and the invented $t > 5.0\text{s}$ dwell threshold. Evidence gating reflects only the criteria actually implemented in [`src/analytics/spatial_observations.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/spatial_observations.py).
3. **Corrected Density Semantics:** Replaced all references to "Gaussian density" or "crowd occupancy" with **discrete sampled trajectory-point binning** ($interpolation='nearest'$).
4. **Strict Separation of Runs:** Clearly distinguished the 100-frame MERL engineering check, the full-clip MERL demonstration (633 frames), synthetic design fixtures, and the MOT17 non-retail comparator using exact manifest values.
5. **Prominent Illustrative Disclosures:** Positioned explicit notices adjacent to interactive controls, clarifying that simulated scrubbers and visual point correspondences are prototype interactions, not verified synchronized video/track playback.
6. **Fully Offline Local Prototype:** Eliminated all external CDN dependencies (fonts, styles, icons), ensuring [`docs/design_prototype/index.html`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/docs/design_prototype/index.html) is 100% self-contained and responsive across 1440px desktop and 390px mobile viewports.

---

## 2. Information Architecture & Product Structure

The user interface consolidates the previously fragmented Streamlit views into a single, cohesive workflow comprising six structured views:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SIX CORE PRODUCT VIEWS                             │
├───────────────────────────────┬─────────────────────────────────────────────┤
│ Product Section               │ Purpose & User Workflow                     │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ 1. Overview                   │ High-level telemetry summary, active run    │
│                               │ status, and manifest run inventory.         │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ 2. Guided New Analysis        │ Step-by-step setup: video input, planar     │
│                               │ homography correspondence, and parameters.  │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ 3. Unified Results Workspace  │ Single selected run view: orthogonal floor  │
│                               │ diagram, layer toggles, and metrics.        │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ 4. Placement Candidate Detail │ Exploratory shelf hotspot evaluation under  │
│                               │ explicit evidence-gating rules.             │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ 5. Evidence & Run Details     │ Model lineage, coordinate reference audit,  │
│                               │ and capability classification matrix.       │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ 6. Sample Library             │ Segregated inspection of real MERL,         │
│                               │ synthetic benchmarks, and MOT17 comparator. │
└───────────────────────────────┴─────────────────────────────────────────────┘
```

---

## 3. Ground Truth Analytics & Run Separation

Values must never be conflated across runs. Each run manifest in [`outputs/reports/`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/outputs/reports/) reflects distinct evaluation boundaries:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PERSISTED RUN METRICS INVENTORY                                       │
├─────────────────────────┬─────────────────────────┬─────────────────────────┬───────────────────────────┤
│ Metric / Field          │ Capped 100-Frame MERL   │ Saved Full-Clip MERL    │ Synthetic Benchmark       │
│                         │ (phase_13_merl_sample)  │ (demo_real_summary)     │ (demo_synthetic_summary)  │
├─────────────────────────┼─────────────────────────┼─────────────────────────┼───────────────────────────┤
│ Dataset Nature          │ Real Retail (Capped)    │ Real Retail (Full Clip) │ Programmatic Fixture      │
│ Evaluated Frames        │ 100 frames              │ 633 frames (126.4s)     │ Synthetic (Programmatic)  │
│ Sampling Rate           │ 5.0 FPS                 │ 5.0 FPS                 │ 5.0 FPS                   │
│ Evaluated Tracks        │ 1 track                 │ 3 tracks (T1, T2, T3)   │ 2 simulated tracks        │
│ Total Evaluated Points  │ 74 points               │ 508 points              │ 66 points                 │
│ In-Zone Points          │ 0 points (0.0%)         │ 88 points (17.3%)       │ 66 points (100.0%)        │
│ In-Zone Breakdown       │ 0 in all defined zones  │ shelf_a: 53, main: 35,  │ checkout: 24, main: 24,   │
│                         │                         │ shelf_b: 0, checkout: 0 │ shelf_a: 9, shelf_b: 9    │
│ Out-of-Bounds Points    │ 71 points               │ 339 points              │ 0 points                  │
│ Binned Density Points   │ 3 points                │ 169 points              │ 66 points                 │
│ Max Cell Count          │ 2 observations          │ 29 observations         │ 28 observations           │
│ Evaluated Dwell Span    │ 0.0s (0 episodes)       │ shelf_a: 10.2s (3 visits│ 11.2s total (10 episodes) │
│                         │                         │ main: 7.0s (2 visits)   │ checkout: 4.8s, main: 4.8s│
│ Direct Transitions      │ 0 transitions           │ 2 direct transitions    │ 8 direct transitions      │
│                         │                         │ (shelf_a <-> main_aisle)│ (bidirectional in zones)  │
│ Exploratory Candidate   │ None                    │ shelf_a hotspot cell    │ Synthetic demonstration   │
│                         │                         │ [2.5..3.0, 4.5..5.0]    │ active (Non-real)         │
│ Actionable Rec. Status  │ WITHHELD                │ WITHHELD                │ WITHHELD                  │
└─────────────────────────┴─────────────────────────┴─────────────────────────┴───────────────────────────┘
```

> [!IMPORTANT]
> **MOT17-04 Non-Retail Outdoor Comparator:**
> Evaluated for detector smoke testing only (10 frames, 11 tracks, 94 points, 37 binned). Dwell analytics and zone transitions are classified as **`not_available`** (not evaluated). Recommendations are strictly **WITHHELD** because outdoor pedestrian walkways cannot support retail merchandising observations.

---

## 4. Coordinate Units & Spatial Geometry

### 4.1 Floor Bounds & Coordinate Frame
- **Coordinate Reference:** Configured floor units (`unit: "normalized_units"` / `coordinate_type: "aisle_ground_grid"`).
- **Physical Scale:** Unverified. No independent surveying has established physical metric dimensions ($m$) or ground truth velocity ($m/s$).
- **Configured Bounds:**
  - $X \in [0.0, 10.0]$ configured floor units (aisle width span: $10.0$ units)
  - $Y \in [0.0, 6.0]$ configured floor units (aisle depth span: $6.0$ units)
  - Grid cell size: $0.5 \times 0.5$ units ($20 \times 12$ matrix = 240 cells total)

### 4.2 Configured Retail Zone Polygons
From [`config.yaml`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/config.yaml) and [`configs/merl_scene_profile.json`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/configs/merl_scene_profile.json):
- **`shelf_a` (Left Snack & Beverage Gondola):**
  - Vertices: $[(0.5, 1.0), (4.0, 1.0), (4.0, 5.0), (0.5, 5.0)]$
  - Color: `#3498DB` (Soft Blue)
- **`shelf_b` (Right Produce & Dairy Gondola):**
  - Vertices: $[(6.0, 1.0), (9.5, 1.0), (9.5, 5.0), (6.0, 5.0)]$
  - Color: `#2ECC71` (Soft Green)
- **`main_aisle` (Central Pedestrian Walkway):**
  - Vertices: $[(4.0, 1.0), (6.0, 1.0), (6.0, 5.0), (4.0, 5.0)]$
  - Color: `#F39C12` (Amber/Orange)
- **`checkout` (Front Counter Area):**
  - Vertices: $[(0.0, 0.0), (10.0, 0.0), (10.0, 1.0), (0.0, 1.0)]$
  - Color: `#E74C3C` (Red/Coral)

---

## 5. Evidence-Gating Criteria (Phase 12 Invariants)

The platform implements deterministic evidence gating based strictly on caller-provided provenance metadata ([`EvidenceContext`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/spatial_observations.py#L73-L80)):
1. **`coverage_status = "boundary_truncated"`:**
   - Single-camera surveillance suffers from severe field-of-view boundary truncation. Trajectories are truncated as shoppers enter or leave the cropped frame.
   - *Result:* Spatial trajectories cannot establish representative spatial coverage across the store layout.
2. **`calibration_status = "unverified_configured_units"`:**
   - Camera calibration is based on configured pixel-to-plane correspondences without independent metric field surveying.
   - *Result:* Unverified coordinates cannot support prescriptive architectural or merchandising decisions.
3. **`environment_type != "retail_store"`:**
   - Outdoor pedestrian image sequences (MOT17) lack retail fixtures, shelves, and shopping context.
   - *Result:* Retail spatial observations are strictly ineligible.
4. **Zero Dwell Invariant:**
   - A zero dwell value is an evaluated numerical observation, **NOT** a gating eligibility rule. Eligibility is governed strictly by coverage, calibration, and environment provenance.

---

## 6. Capability Mapping Matrix

| Architectural Function | Implementation Location | Capability Classification | Notes |
| :--- | :--- | :--- | :--- |
| **Monocular Detection** | `src/detection/` | **Implemented** | YOLO11-n ONNX running on CPUExecutionProvider (smoke inference). |
| **Multi-Object Tracking** | `src/tracking/` | **Implemented** | ByteTrack (clip-local tracker IDs; no cross-session Re-ID). |
| **Planar Homography** | `src/spatial/` | **Implemented** | SVD-based planar homography projection ($3 \times 3$ tensor $H$). |
| **Scene Profile Validation** | [`src/spatial/scene_profile.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/spatial/scene_profile.py) | **Implemented** | Structural & geometric sanity checks; **not** proof of independent physical calibration. |
| **Temporal Debouncing** | [`src/analytics/zone_evaluator.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/zone_evaluator.py) | **Implemented** | Configured hysteresis window ($N=3$ frames) belongs to zone assignment. |
| **Dwell Episode Summaries** | [`src/analytics/dwell.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/dwell.py) | **Implemented** | Observed temporal spans between sampled detections (exit minus entry). |
| **Discrete Density Grid** | [`src/analytics/density.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/density.py) | **Implemented** | 2D deterministic binning ($interpolation='nearest'$); no Gaussian smoothing. |
| **Direct Transition Counts** | [`src/analytics/transitions.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/transitions.py) | **Implemented** | Immediate consecutive zone movements; non-Markovian counts. |
| **Evidence Gating Protocol** | [`src/analytics/spatial_observations.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/spatial_observations.py) | **Implemented** | Deterministic withholding of actionable merchandising recommendations. |
| **Interactive Point Selection** | `docs/design_prototype/` | **New Capability Required** | Proposed visual drag-and-drop tool on video frame. |
| **Polygon Boundary Editor** | `docs/design_prototype/` | **New Capability Required** | Proposed interactive canvas tool for defining retail zones. |
| **Synchronized Video Scrubber** | `docs/design_prototype/` | **New Capability Required** | Real-time synchronized video frame and trajectory vector playback. |
| **Dynamic Run Cancellation** | `src/` | **New Capability Required** | Ability to gracefully abort long-running video pipeline executions. |

---

## 7. Stitch Project Update Status

Google Stitch project `14372996755834353810` was updated directly via StitchMCP:

- **Stitch Project Title:** `Vision Trajectory Analytics - Retail Spatial System`
- **Stitch Project URL:** [https://stitch.withgoogle.com/projects/14372996755834353810](https://stitch.withgoogle.com/projects/14372996755834353810)
- **Screen 2 Updated:**
  - Resource ID: `projects/14372996755834353810/screens/25a9a1964e774179a26879981a329c0e`
  - Title: `New Analysis & Calibration Setup (Strict Repository Calibration)`
  - Updates: Replaced physical meters with configured floor units $[0..10.0] \times [0..6.0]$; injected config correspondences (P1, P2, P3, P4, CP1); removed invented $Y > 7.5\text{m}$ notice; added illustrative interaction disclosure.
- **Screen 3 Updated:**
  - Resource ID: `projects/14372996755834353810/screens/8b89ca33e7a6416ba5ed2a7f8851ced4`
  - Title: `Results Workspace & Spatial Optimizer (Strict Analytics Correction)`
  - Updates: Replaced floor geometry with configured zone layout; loaded exact MERL demonstration values (508 points, 88 in-zone, 10.2s dwell in `shelf_a`); added discrete density cell counts; added prominent scrubber disclosure banner.

---

## 8. Local Prototype Verification & Responsive Layout

The standalone interactive prototype is codified at [`docs/design_prototype/index.html`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/docs/design_prototype/index.html).

### 8.1 Offline Architecture
- **Zero Remote Assets:** No Google Fonts, no external CDN stylesheets, and no third-party JavaScript libraries.
- **System Font Stack:** Uses `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif` for interface typography and `SF Mono, Monaco, Cascadia Code, Roboto Mono, monospace` for tabular coordinates.
- **Vector Graphics:** All floor plans, camera perspectives, and icons are rendered as inline SVGs.

### 8.2 Responsive Viewport Verification
- **1440px Desktop Viewport:**
  - Multi-pane two-column workspace: $58\%$ left spatial floor plan canvas ($500 \times 300$ metric aspect ratio) alongside $42\%$ right analytical metrics and evidence drawer.
  - Sticky header with persistent 6-view tab navigation.
- **390px Mobile Viewport:**
  - Verified via native CSS media queries (`@media (max-width: 480px)`).
  - Single-column stacked layout: navigation tabs collapse into a touch-scrollable horizontal rail; 2x2 metric cards stack into clean individual tiles; floor plan scales responsively with touch-friendly scrubber controls.
- **Empty & Partial State Handling:**
  - When MOT17 comparator is selected, dwell and transition cards explicitly display `Not Evaluated` pills rather than misleading zeroes.
  - When the capped 100-frame MERL check is selected, evaluated zero dwell and transitions are rendered alongside the exact $3/74$ binned points and $71$ out-of-bounds count.

---

## 9. Remaining UI Implementation Tasks

With the UI/UX foundation and interactive prototype corrected, production UI enhancement in Streamlit can proceed incrementally:

1. **Streamlit App Navigation Refactor ([`dashboard/app.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/dashboard/app.py)):**
   - Replace 7 fragmented sidebar radio buttons with the consolidated 6-section product tabs (`Overview`, `New Analysis`, `Results Workspace`, `Placement Candidate`, `Evidence & Runs`, `Sample Library`).
2. **SVG Floor-Plane Component Integration:**
   - Embed the orthogonal 2D floor diagram into Streamlit using `st.components.v1.html` to display configured zones (`shelf_a`, `shelf_b`, `main_aisle`, `checkout`) and discrete density cells.
3. **Standardized Evidence Withholding Component:**
   - Implement a shared UI helper in `dashboard/` rendering the calibrated amber status card (`STATUS: RECOMMENDATION WITHHELD`) whenever `is_eligible == False`.
4. **Scene Profile Validator Integration:**
   - Wire the upload wizard in `dashboard/app.py` to [`src/spatial/scene_profile.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/spatial/scene_profile.py) to validate uploaded scene JSON files prior to pipeline execution.
