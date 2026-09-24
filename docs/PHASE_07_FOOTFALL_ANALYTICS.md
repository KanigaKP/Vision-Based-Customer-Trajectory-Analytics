# Phase 7 Completion Report — Footfall Analytics

**Project**: Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
**Date**: September 24, 2026  
**Status**: COMPLETE / PASS  
**Git HEAD**: `ed3768b` (Phase 7 changes uncommitted per instructions)  
**Test Suite**: 55/55 PASSED (44 baseline + 11 new)

---

## 1. Objective

Phase 7 implements the **Footfall Analytics** module for the computer-vision retail movement pipeline. This module operates downstream of spatial zone assignment, consuming trajectories whose spatial coordinates have already been mapped to the 2D floor plane via homography (Phase 5) and whose spatial zone labels (`TrajectoryPoint.zone_id`) have been assigned via Point-in-Polygon (PIP) testing and temporal hysteresis filtering (Phase 6).

Phase 7 interprets the discrete sequence of zone assignments for each tracked trajectory to compute:
1. Unique tracks per zone
2. Zone entry events
3. Contiguous zone visits
4. Basic visit durations
5. Per-frame occupancy snapshots and timelines
6. Zone-to-zone transition matrices

---

## 2. Architecture Integration & Responsibilities

The pipeline responsibilities are strictly decoupled across phases:

```
[Phase 2] Video Frame Sampling (5 FPS generator)
   ↓
[Phase 3] YOLO11 Person Detection (ONNX Runtime CPU)
   ↓
[Phase 4] ByteTrack Multi-Object Tracking (Session-local Track IDs)
   ↓
[Phase 5] Trajectory Extraction + Homography (Bottom-center image → 2D Floor Coordinates)
   ↓
[Phase 6] Manual Zone Definition + Point-in-Polygon + Hysteresis (Floor Coordinates → zone_id)
   ↓
[Phase 7] FOOTFALL ANALYTICS (Interprets zone_id sequences into discrete events & metrics)
```

### Architectural Separation:
- **Phase 6 owns**: Polygon spatial boundaries, geometric Point-in-Polygon testing, and boundary stabilization ($N=3$ temporal hysteresis). It populates `TrajectoryPoint.zone_id`.
- **Phase 7 owns**: Interpreting the temporal sequence of `zone_id` values per track.
- **Rule**: Phase 7 **never** reruns homography, PIP geometry, tracking association, or hysteresis. It operates purely on pre-evaluated `Trajectory` objects.

---

## 3. Metric Definitions & Critical Distinctions

In retail analytics, "footfall" is often used ambiguously. This project enforces precise, distinct mathematical definitions:

| Metric | Definition | Unit / Type | Interpretation |
|---|---|---|---|
| **Unique Tracks Per Zone** | Count of distinct session-local `track_id` values observed in zone $Z$ at least once. | Integer count ($\ge 0$) | Reach / distinct individual presence in a zone. |
| **Zone Entry Events** | Count of discrete transition occurrences into zone $Z$ from outside (`None`) or another zone. | Discrete event count | Traffic volume and zone access frequency. |
| **Zone Visits** | Contiguous temporal intervals during which a track remains continuously in zone $Z$. | Contiguous stay intervals | Distinct browsing/stay sessions. |
| **Basic Visit Duration** | Elapsed duration between the initial entry timestamp and final exit timestamp of a visit: $\Delta t = t_{\text{exit}} - t_{\text{entry}}$. | Seconds (float, 3 decimals) | Stay length per visit (not exact physical dwell time). |
| **Per-Frame Occupancy** | Count of active tracks present in zone $Z$ at a specific frame index $f$. | Tracks present at frame $f$ | Point-in-time zone density / crowding snapshot. |
| **Zone-to-Zone Transitions** | Aggregated count of direct movements between two valid zones ($Z_A \to Z_B$, where $Z_A \ne Z_B$). | Transition pair frequency | Inter-zone shopper flow paths. |

### Critical Distinction Example:
If Track 42 follows the path:
$$\text{None} \to \text{shelf\_a} \to \text{shelf\_b} \to \text{shelf\_a} \to \text{None}$$

Then:
- **Unique tracks in `shelf_a`**: **1** (single distinct track ID 42)
- **Zone entry events into `shelf_a`**: **2** (initial entry from outside + re-entry from `shelf_b`)
- **Zone visits in `shelf_a`**: **2** (two separate contiguous stay intervals)
- **Occupancy of `shelf_a`**: **1** while Track 42 is in `shelf_a`, **0** otherwise.

---

## 4. Algorithms & Implementation Details

All footfall logic is implemented in `src/analytics/footfall.py` with zero external dependencies beyond Python standard libraries and existing dataclasses.

### 4.1 Entry Event & Visit Consolidation Algorithm
For each trajectory, a deterministic finite-state transition loop processes chronologically ordered points:

```python
current_zone = None
current_visit_pts = []

for pt in trajectory.points:
    pt_zone = pt.zone_id

    if pt_zone == current_zone:
        if current_zone is not None:
            current_visit_pts.append(pt)
    else:
        # Transition detected
        if current_zone is not None and current_visit_pts:
            # Finalize previous contiguous visit
            visits.append(ZoneVisit(
                track_id=trajectory.track_id,
                zone_id=current_zone,
                entry_frame_idx=current_visit_pts[0].frame_idx,
                exit_frame_idx=current_visit_pts[-1].frame_idx,
                entry_timestamp_sec=current_visit_pts[0].timestamp_sec,
                exit_timestamp_sec=current_visit_pts[-1].timestamp_sec,
                point_count=len(current_visit_pts)
            ))
            current_visit_pts = []

        if current_zone is not None and pt_zone is not None:
            # Record direct inter-zone transition
            transitions.append((current_zone, pt_zone))

        if pt_zone is not None:
            # Record entry event and begin new visit
            entries.append(ZoneEntryEvent(
                track_id=trajectory.track_id,
                zone_id=pt_zone,
                from_zone_id=current_zone,
                frame_idx=pt.frame_idx,
                timestamp_sec=pt.timestamp_sec
            ))
            current_visit_pts = [pt]

        current_zone = pt_zone

# Finalize any trailing visit if trajectory ends inside a zone
if current_zone is not None and current_visit_pts:
    visits.append(ZoneVisit(...))
```

### 4.2 Basic Visit Duration
- For any `ZoneVisit`, `duration_sec = round(max(0.0, exit_timestamp_sec - entry_timestamp_sec), 3)`.
- If a visit consists of a single sampled frame ($N=1$), duration is $0.0\text{ s}$.
- Total visit duration per zone: sum of durations for all visits in that zone.
- Average visit duration per zone: total duration divided by visit count (or $0.0\text{ s}$ if zero visits).

### 4.3 Per-Frame Occupancy
- `get_occupancy_snapshot(trajectories, frame_idx, known_zone_ids=None)`: Evaluates all trajectories at `frame_idx`, counting how many tracks have `pt.zone_id == zone`.
- `get_occupancy_timeline(trajectories, known_zone_ids=None)`: Scans all sampled frames present in trajectories and constructs a frame-by-frame occupancy dictionary.

### 4.4 Zone-to-Zone Transition Matrix
- Initialized as a 2D square matrix `{from_zone: {to_zone: count}}` for all known zones.
- Populated only when a track transitions directly between two distinct defined zones ($Z_A \to Z_B$ where $Z_A \ne \text{None}$ and $Z_B \ne \text{None}$).
- Movements to/from `None` (unassigned store areas) do not contaminate the inter-zone matrix.

### 4.5 Computational Complexity
- **Time Complexity**: $\mathcal{O}(P)$ where $P = \sum_{t} |t.\text{points}|$ is the total number of sampled trajectory points across all tracks. This is optimal linear time.
- **Space Complexity**: $\mathcal{O}(E + V)$ where $E$ is the number of entry events and $V$ is the number of visits ($V \le P$).

---

## 5. Methodological Limitations & Viva Defenses

To maintain academic rigor and defend this project during an MCA viva examination, the following limitations are explicitly documented:

1. **Tracker IDs vs. Customer Identities**:
   > *Track IDs are session-local ByteTrack assignments rather than verified customer identities.*
   No facial recognition, biometric identification, or visual re-identification (Re-ID) is performed.
2. **Track Fragmentation Bias**:
   > *Zone entry events represent tracker-level entries. Track fragmentation may cause the count to exceed the true number of physical customer entries.*
   If a customer is temporarily occluded for longer than `max_time_lost_sec` (2.0s) and is assigned a new track ID upon reappearance, an additional entry event and unique track will be recorded.
3. **Temporal Resolution (5 FPS Sampling)**:
   Frame sampling is configured at 5 FPS ($\Delta t = 200\text{ ms}$). Incursions occurring entirely within an inter-frame interval cannot be observed.
4. **Phase 6 Hysteresis Window ($N=3$) Effect**:
   The $N=3$ frame hysteresis filter stabilizes boundaries and suppresses high-frequency boundary jitter, but introduces a 2-frame detection delay ($\sim 400\text{ ms}$) and suppresses very brief transient crossings.
5. **No Ground-Truth Claims from Synthetic Data**:
   Synthetic trajectory sequences and MERL action annotations validate algorithmic correctness and software invariants. Absolute physical footfall validation will be performed on controlled student-recorded footage with manual ground-truth counts.

---

## 6. Testing Results

The full automated test suite was executed via pytest:
```
platform win32 -- Python 3.14.6, pytest-9.1.1
collected 55 items

tests/test_detection.py ......................... [ 16%] (9 tests)
tests/test_footfall.py .......................... [ 36%] (11 tests)
tests/test_homography.py ........................ [ 47%] (6 tests)
tests/test_tracking.py .......................... [ 61%] (8 tests)
tests/test_trajectory.py ........................ [ 69%] (4 tests)
tests/test_video_preprocessing.py ............... [ 87%] (10 tests)
tests/test_zones.py ............................. [100%] (7 tests)

============================= 55 passed in 58.52s =============================
```

### Breakdown of Phase 7 Unit Tests (`tests/test_footfall.py`):
1. `test_zone_entry_event_detection`: Transitions from `None` into a zone produce an entry event with accurate timestamps.
2. `test_zone_exit_produces_no_entry`: Moving to `None` finalizes visits without emitting an entry event.
3. `test_re_entry_counted_separately`: $A \to B \to A$ path generates 2 entries and 2 visits in $A$, and 1 unique track.
4. `test_zone_visit_contiguous_interval`: Contiguous 5-frame stay generates 1 visit with point_count=5.
5. `test_unique_track_count_per_zone`: Multiple distinct tracks in the same zone accurately produce unique track counts.
6. `test_occupancy_snapshot_at_frame`: Snapshot queries at target frame indices return correct per-zone track counts.
7. `test_trajectory_with_no_zone_assignments`: Trajectories entirely in `None` produce 0 entries, 0 visits, and 0 unique tracks.
8. `test_single_point_trajectory`: Single-point visits inside a zone yield duration $0.0\text{ s}$ and point_count=1; single-point `None` yields 0 events.
9. `test_basic_visit_duration`: Validates duration sums and averages across multiple visits.
10. `test_zone_to_zone_transition_matrix`: Validates inter-zone transition counts ($A \to B$).
11. `test_occupancy_timeline`: Validates frame-by-frame occupancy dictionary construction.

---

## 7. Verification Script Results (`scripts/verify_phase_07.py`)

The end-to-end verification script was executed against MERL supermarket retail zone coordinates:

```text
===========================================================================
PHASE 7 — FOOTFALL ANALYTICS VERIFICATION
===========================================================================
  Loaded 4 zones for MERL: ['shelf_a', 'shelf_b', 'main_aisle', 'checkout']
  Hysteresis Window N = 3

[1/3] Running Phase 6 ZoneEvaluator to assign TrajectoryPoint.zone_id...
  Evaluated 4 trajectories with Phase 6 ZoneEvaluator.
    - Track 101 (9 pts): None -> shelf_a (4 pts) -> main_aisle (3 pts) -> None
    - Track 102 (10 pts): shelf_b (4 pts) -> main_aisle (3 pts) -> checkout (3 pts)
    - Track 103 (8 pts): shelf_a (3 pts) -> main_aisle (2 pts) -> shelf_a (2 pts) -> None
    - Track 104 (4 pts): None -> None -> None -> None

[2/3] Executing FootfallAnalyzer on evaluated trajectories...

---------------------------------------------------------------------------
ZONE ID         | UNIQUE TRACKS  | ENTRIES   | VISITS   | AVG DURATION
---------------------------------------------------------------------------
shelf_a         | 2              | 3         | 3        | 0.40s
shelf_b         | 1              | 1         | 1        | 0.60s
main_aisle      | 3              | 3         | 3        | 0.33s
checkout        | 1              | 1         | 1        | 0.40s
---------------------------------------------------------------------------

  Zone-to-Zone Transition Matrix:
  FROM / TO      | shelf_a    | shelf_b    | main_aisle | checkout  
  ------------------------------------------------------------------
  shelf_a        | 0          | 0          | 2          | 0         
  shelf_b        | 0          | 0          | 1          | 0         
  main_aisle     | 1          | 0          | 0          | 1         
  checkout       | 0          | 0          | 0          | 0         

  Occupancy Snapshots at Selected Frames:
    - Frame  2: {'shelf_a': 2, 'shelf_b': 1, 'main_aisle': 0, 'checkout': 0}
    - Frame  5: {'shelf_a': 0, 'shelf_b': 0, 'main_aisle': 3, 'checkout': 0}
    - Frame  8: {'shelf_a': 1, 'shelf_b': 0, 'main_aisle': 0, 'checkout': 1}

[3/3] Verifying metric invariants...
  All metric invariants verified successfully.

  Saved structured report to: outputs\reports\phase_07_footfall_report.json

===========================================================================
VERIFICATION SUCCESSFUL: Phase 7 Footfall Analytics passed all checks.
===========================================================================
```

Verification output report generated at:
[`outputs/reports/phase_07_footfall_report.json`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/outputs/reports/phase_07_footfall_report.json)

---

## 8. Strict Scope Boundaries

To prevent premature architectural bloat, Phase 7 strictly avoided the following:
- **No Heatmaps** (spatial density maps belong to Phase 8)
- **No Zone Classification** (e.g. high/low traffic tagging belongs to layout analytics)
- **No Recommendations / Optimization Engine** (belongs to Phase 9)
- **No Streamlit Dashboard** (UI layer belongs to dashboard phase)
- **No SQLite / Database Persistence** (deferred until batch multi-video storage is required)
- **No Re-ID or Facial Biometrics** (permanently excluded from project scope)
- **No Sales / POS / SKU Data Integration** (computer-vision trajectory analytics only)
