# Phase 8.3 — Zone Assignment Agreement Validation Specification

**Project**: Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
**Phase**: 8.3 (Zone Assignment Agreement Validation Layer)  
**Status**: COMPLETE / VERIFIED  
**Regression Status**: 87/87 tests passing (55 Phase 0–7 baseline + 12 Phase 8.1 schema tests + 9 Phase 8.2 homography validation tests + 11 Phase 8.3 zone validation tests)  

---

## 1. Phase 8.3 Objective & Architectural Boundary

The objective of Phase 8.3 is to implement an independent validation layer that compares existing Phase 6 spatial zone assignments against independently supplied ground-truth zone annotations on explicitly aligned observation points.

### Architectural Invariants:
1. **Decoupled from Zone Evaluation**:
   - Phase 8.3 compares pre-existing zone labels.
   - It does **not** alter, tune, or rerun point-in-polygon evaluation.
   - It does **not** rerun or modify temporal hysteresis.
   - The existing Phase 6 evaluator uses the repository’s current implementation in [`src/analytics/zone_evaluator.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/zone_evaluator.py) (which utilizes `matplotlib.path.Path` for Point-in-Polygon evaluation). No claim is made about `cv2.pointPolygonTest`.
   - Phase 6 spatial logic remains untouched.
2. **Descriptive Statistic, Not Universal Accuracy**:
   - Phase 8.3 remains descriptive label agreement on explicitly aligned observations.
   - The metric computed is **label agreement** (\(\text{matched} / \text{total}\)) across explicitly aligned test observation points.
   - It is a descriptive statistic on the evaluated sample, not an unverified claim of global physical zone classification accuracy.
3. **Decoupled from Downstream Analytics**:
   - Phase 8.3 does not evaluate footfall counts, dwell durations, transition matrices, fragmentation, heatmaps, or store recommendations. Those remain downstream analytics to be validated separately.

---

## 2. Explicit Correspondence Policy

To prevent methodological circularity and optimistic error reporting:
- **Mandatory Explicit ID Mapping**:
  - Requires an explicit dictionary `id_mapping: Dict[str, int]` mapping ground-truth identities (`gt_id: str`) to tracker IDs (`track_id: int`).
  - Automatic identity matching, appearance Re-ID, bounding-box IoU matching, or spatial distance heuristics are **strictly prohibited**.
- **No Silent Alignment Inference**:
  - When `require_complete_alignment=True` (default), every ground-truth point must have an exact matching predicted tracker observation at that frame index. If any point is missing, a `ValueError` is raised immediately.
- **Duplicate Key Rejection**:
  - Duplicate observations sharing `(gt_id, frame_idx)` or `(track_id, frame_idx)` are strictly rejected with explicit errors to ensure one-to-one temporal correspondence.

---

## 3. Categorical Representation of Outside Areas (`None`)

In retail floor analytics, unassigned floor space, aisles not designated as named retail zones, and exterior walking paths are valid physical states:
- **Representation**: Outside areas are represented strictly by `None` (not sentinel strings like `"outside"` or `"unknown"`).
- **Categorical Treatment**: `None` is treated as a valid category in label matching, observation counts, distributions, and confusion matrices.
- **Bi-directional Disagreement Tracking**:
  - **Missed Detection (Zone \(\to\) None)**: Ground truth specifies an active retail zone, but the tracker assigned `None` (e.g. boundary cutoff or delayed entry).
  - **Spurious Detection (None \(\to\) Zone)**: Ground truth specifies outside/unassigned floor, but the tracker assigned a named zone (e.g. tracking jitter or projection drift).
  - **Cross-Zone Confusion (Zone A \(\to\) Zone B)**: Track assigned to an adjacent or incorrect retail zone.
- **JSON Serialization Integrity**:
  - Python dictionaries permit `None` keys; standard JSON converts dictionary keys to strings (`"null"`).
  - Deserialization via `from_dict()` leverages `_normalize_zone_key()` to restore `None` keys seamlessly, preserving dictionary structure and matrix alignment.

---

## 4. Data Structures & Output Representation

Implemented in [`src/validation/zone_validation.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/validation/zone_validation.py):

### 4.1 Input Observation
- **`AlignedZoneObservation`**:
  - `gt_id`: Ground-truth individual identity string (non-empty).
  - `track_id`: Integer tracking session identifier.
  - `frame_idx`: Non-negative integer video frame index.
  - `timestamp_sec`: Non-negative finite numeric float video timestamp in seconds (strictly rejects booleans, non-numeric types, NaNs, and infinities).
  - `expected_zone`: Ground-truth zone label (or `None`).
  - `predicted_zone`: Tracker-assigned zone label (or `None`).
  - `is_match`: Boolean property returning `expected_zone == predicted_zone`.

### 4.2 Disagreement Record
- **`ZoneDisagreement`**:
  - Captures exact context for each observation where `is_match` is `False`.
  - Fields: `gt_id`, `track_id`, `frame_idx`, `timestamp_sec`, `expected_zone`, `predicted_zone`.

### 4.3 Validation Result Container
- **`ZoneValidationResult`**:
  - `total_aligned_observations`: Number of evaluated observation pairs.
  - `matched_zone_count`: Count of observations where labels matched identically.
  - `mismatched_zone_count`: Count of observations where labels differed.
  - `agreement_rate`: Descriptive proportion \(\text{matched} / \text{total}\) (or `None` if empty).
  - `expected_zone_counts`: Frequency distribution of ground-truth zones.
  - `predicted_zone_counts`: Frequency distribution of predicted zones.
  - `confusion_matrix`: 2D mapping `expected_zone -> predicted_zone -> count` covering all observed zones plus `None`.
  - `disagreements`: List of `ZoneDisagreement` instances.
  - `evidence_status`: `"ZONE_AGREEMENT_EVALUATED"` or `"INSUFFICIENT_EVIDENCE"`.
  - `evidence_note`: Explicit methodological scope note.
  - `dataset_name`: Identifier string for evaluation run.

---

## 5. Validation Procedure

```
GroundTruthPoint collection + TrajectoryPoint collection
                           │
                           ▼
             Explicit id_mapping: gt_id -> track_id
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
   Key Validation Checks        Temporal Alignment
   - No duplicates              (track_id, frame_idx) == (mapped_id, frame_idx)
   - No missing observations    
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
               AlignedZoneObservation pairs
                           │
                           ▼
                   ZoneValidator
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
 If observations empty:             If observations present:
 - INSUFFICIENT_EVIDENCE            - Compute agreement rate
 - agreement_rate = None            - Populate 2D confusion matrix (with None)
                                    - Record ZoneDisagreement details
                                    - ZONE_AGREEMENT_EVALUATED
```

---

## 6. Confusion Matrix Structure & Determinism

The confusion matrix generated by `ZoneValidator` is deterministic:
1. All unique zone names across expected observations, predicted observations, and optional `known_zones` are sorted alphabetically.
2. `None` (unassigned/outside) is appended as the final row and column index if present.
3. Every cell \((i, j)\) contains the integer count of observations where `expected_zone == i` and `predicted_zone == j`.
4. Row totals identically match `expected_zone_counts`.
5. Column totals identically match `predicted_zone_counts`.

---

## 7. Evidence Status Classifications & Methodological Scope

| Status | Trigger Condition | Interpretation |
|---|---|---|
| **`INSUFFICIENT_EVIDENCE`** | Empty observation list or zero valid alignments. | No aligned zone pairs provided. Zone label agreement cannot be evaluated. Prevents artificial 0% or 100% metrics. |
| **`ZONE_AGREEMENT_EVALUATED`** | One or more valid aligned observations. | Descriptive agreement rate and confusion matrix computed across supplied test data. |

> [!NOTE]
> Synthetic test fixtures in `tests/test_zone_validation.py` verify that alignment parsing, error detection, `None` handling, and confusion matrix math work as designed. They do not represent empirical real-world customer behavior or physical camera calibration accuracy.
