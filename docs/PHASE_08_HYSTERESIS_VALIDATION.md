# Phase 8.6 — Temporal Hysteresis Sensitivity Analysis Specification

**Project**: Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
**Phase**: 8.6 (Temporal Hysteresis Sensitivity Validation Layer)  
**Status**: COMPLETE / VERIFIED  
**Regression Status**: 125/125 tests passing (55 Phase 0–7 baseline + 12 Phase 8.1 schema tests + 9 Phase 8.2 homography validation tests + 11 Phase 8.3 zone validation tests + 11 Phase 8.4 footfall validation tests + 14 Phase 8.5 fragmentation tests + 13 Phase 8.6 hysteresis sensitivity tests)  

---

## 1. Phase 8.6 Purpose & Scope

In spatial retail analytics, Point-in-Polygon (PIP) testing near zone boundaries often exhibits high-frequency jitter due to camera perspective distortion, tracker bounding-box fluctuation, or natural customer posture shifts. Phase 6 introduced a temporal hysteresis window threshold ($N \ge 1$), requiring $N$ consecutive frames inside a candidate zone before confirming a transition.

The objective of Phase 8.6 is to implement a non-intrusive validation layer that systematically evaluates the sensitivity of spatial zone assignments, discrete entry events, zone-to-zone transitions, and boundary oscillation across varying values of the hysteresis window threshold $N \in [1, 10]$.

### Architectural Invariants:
1. **Reuse Existing Production Logic**:
   - Executes the existing [`ZoneEvaluator.evaluate_trajectory(...)`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/zone_evaluator.py#L69-L123) across each evaluated window $N$.
   - Executes the existing [`FootfallAnalyzer.analyze(...)`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/footfall.py#L210-L280) to obtain genuine `FootfallSummary` metrics.
   - Does **not** duplicate PIP geometry or footfall event aggregation logic.
2. **Zero-Mutation Guarantee (Caller Data Isolation)**:
   - `ZoneEvaluator.evaluate_trajectory` mutates `TrajectoryPoint.zone_id` in-place.
   - To guarantee complete data isolation, `HysteresisSensitivityValidator` constructs independent deep clones of all trajectories for each evaluated window.
   - Caller-owned `Trajectory` and `Zone` objects are never mutated.
3. **Descriptive Sensitivity Measurement, Not "Optimal" Selection**:
   - The module records raw event frequencies, transitions, and deltas relative to a baseline window.
   - It does **not** declare any hysteresis setting "optimal", nor does it invent arbitrary accuracy scores.
   - Deciding an operational window requires store-specific tolerance trade-offs between rapid transition detection ($N=1$) and false boundary noise rejection ($N \ge 3$).

---

## 2. Accepted Inputs & Pre-Conditions

Implemented in [`src/validation/hysteresis_validation.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/validation/hysteresis_validation.py):

1. **Spatial Zones (`zones`)**:
   - Accepts `List[Zone]` or a dictionary configuration compatible with `ZoneConfig.load_from_config(...)`.
   - Rejects empty zone lists.
2. **Trajectories (`trajectories`)**:
   - Accepts `List[Trajectory]` with populated 2D floor coordinates (`floor_x`, `floor_y`).
   - If empty, returns `INSUFFICIENT_EVIDENCE` cleanly without error.
3. **Hysteresis Windows (`windows`)**:
   - Accepts any non-string `Sequence[int]` (such as `list`, `tuple`, or `range` objects like `range(1, 4)`) as well as `Set[int]`. Defaults to `range(1, 11)`.
   - **Validation Rules**:
     - Cannot be empty (`len(windows) == 0` raises `ValueError`).
     - Must be strictly positive integers ($N > 0$).
     - Rejects `str`, `bytes`, `bytearray`, and non-sequence scalars or mappings (`TypeError`).
     - Rejects `bool` values (e.g. `True`), floating-point values, and strings within collections (`TypeError`).
     - Rejects duplicate window values (`ValueError`).
     - Output metrics are returned in deterministic ascending sorted order.
4. **Baseline Window Selection (`baseline_window`)**:
   - If specified by caller: must be an element of the evaluated windows.
   - If not specified: defaults to $N=3$ if $3$ is evaluated; otherwise defaults to the first sorted evaluated window.

---

## 3. Evaluated Metrics & Delta Semantics

For each evaluated window $N$, the validator measures:
- **`total_entry_events`**: Sum of discrete entries across all zones.
- **`entry_events_per_zone`**: Map of `zone_id` $\to$ entry count.
- **`total_transitions`**: Sum of direct zone-to-zone transitions ($Z_A \to Z_B$ where $Z_A \ne Z_B$).
- **`transition_matrix`**: 2D mapping of `from_zone` $\to$ `to_zone` $\to$ transition count.
- **`total_zone_changes`**: Count of consecutive trajectory points where `zone_id` changed (measures boundary stability).
- **`unique_tracks_per_zone`**: Map of `zone_id` $\to$ distinct track count.

### Delta Semantics:
Relative to baseline window $N_{\text{base}}$:
- $\Delta \text{Entries}(N) = \text{total\_entries}(N) - \text{total\_entries}(N_{\text{base}})$
- $\Delta \text{Transitions}(N) = \text{total\_transitions}(N) - \text{total\_transitions}(N_{\text{base}})$
- $\Delta \text{ZoneChanges}(N) = \text{total\_zone\_changes}(N) - \text{total\_zone\_changes}(N_{\text{base}})$
- $\Delta \text{EntriesPerZone}(N)[Z] = \text{entries}(N)[Z] - \text{entries}(N_{\text{base}})[Z]$

For $N = N_{\text{base}}$, all deltas are identically zero.

---

## 4. Data Structures & Serialization

### 4.1 `HysteresisWindowMetrics`
- `window`: Evaluated window $N$.
- `total_entry_events`: Total entries.
- `entry_events_per_zone`: Per-zone entry counts.
- `total_transitions`: Total transitions.
- `transition_matrix`: 2D transition matrix.
- `total_zone_changes`: Total zone label transitions.
- `unique_tracks_per_zone`: Distinct tracks per zone.
- `delta_entry_events`: Delta vs baseline.
- `delta_transitions`: Delta vs baseline.
- `delta_zone_changes`: Delta vs baseline.
- `delta_entries_per_zone`: Per-zone deltas vs baseline.

### 4.2 `HysteresisSensitivityResult`
- `dataset_name`: Sequence identifier.
- `baseline_window`: Baseline reference window $N_{\text{base}}$.
- `evaluated_windows`: Sorted list of evaluated window values.
- `window_metrics`: List of `HysteresisWindowMetrics` in sorted window order.
- `evidence_status`: `"HYSTERESIS_SENSITIVITY_EVALUATED"` or `"INSUFFICIENT_EVIDENCE"`.
- `evidence_note`: Explicit methodological caveats and scope description.

---

## 5. Evidence Status Classifications & Methodological Scope

| Status | Trigger Condition | Interpretation |
|---|---|---|
| **`INSUFFICIENT_EVIDENCE`** | Empty trajectory list (`trajectories == []`). | No trajectory data provided to evaluate hysteresis. Sensitivity cannot be computed. Prevents artificial metrics. |
| **`HYSTERESIS_SENSITIVITY_EVALUATED`** | Valid zones and one or more trajectories evaluated across valid windows. | Descriptive sensitivity metrics and deltas computed across supplied data. |

### Dataset Caveats & Future Application:
- **Synthetic Fixtures**:
  - Used in `tests/test_hysteresis_validation.py` to confirm boundary noise filtering, delta calculations, caller immutability, and serialization.
  - Software verification only; **not** physical real-world camera or store validation.
- **MERL Shopping Dataset**:
  - Captures retail aisles with stationary shoppers and glancing passes.
  - Useful for observing boundary jitter on real visual trajectories, but ground-truth dwell times must be collected to evaluate trade-offs empirically.
- **Controlled Footage / CAVIAR**:
  - Controlled recordings with deliberate boundary crossings allow quantifying the exact delay introduced by $N \ge 3$ (e.g. at 5 FPS, $N=3$ adds a $0.4\text{ s}$ confirmation lag).
