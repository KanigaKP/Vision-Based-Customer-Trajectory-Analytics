# Phase 8.4 — Footfall Event Validation Specification

**Project**: Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
**Phase**: 8.4 (Footfall Event Validation Layer)  
**Status**: COMPLETE / VERIFIED  
**Regression Status**: 98/98 tests passing (55 Phase 0–7 baseline + 12 Phase 8.1 schema tests + 9 Phase 8.2 homography validation tests + 11 Phase 8.3 zone validation tests + 11 Phase 8.4 footfall validation tests)  

---

## 1. Phase 8.4 Objective & Architectural Boundary

The objective of Phase 8.4 is to implement an independent validation layer that compares aggregate Phase 7 footfall metrics against independently supplied, manually annotated discrete footfall events (`GroundTruthEvent`).

### Architectural Invariants:
1. **Decoupled from Footfall Analysis**:
   - Phase 8.4 validates pre-computed `FootfallSummary` outputs directly.
   - It does **not** rerun `FootfallAnalyzer`, Point-in-Polygon (PIP), or temporal hysteresis filtering.
   - It does **not** recompute entry events, visits, or transitions from raw trajectories.
2. **Decoupled from Identity & Re-ID**:
   - Tracker IDs are session-local identifiers and are **not** required to equal ground-truth individual identities (`gt_id`).
   - Does **not** perform appearance Re-ID, track rematching, or heuristic event pairing.
   - Compares aggregate event frequencies per zone and zone-pair.
3. **Decoupled from Refined Dwell / Temporal Analytics**:
   - Does **not** compute refined dwell distributions, gap-tolerant visits, temporal smoothing, or occupancy interpolation.
   - Temporal dwell refinement and visit consolidation belong to Phase 9.
4. **Descriptive Consistency Statistic, Not Universal Accuracy**:
   - Reports **metric consistency** (\(\text{matched} / \text{total}\)) across evaluated zone entry counts and transition counts.
   - Uses terms such as "event-count agreement" and "metric consistency", not unverified claims of universal physical footfall accuracy.

---

## 2. Validation Inputs & Consumption Rules

Phase 8.4 consumes:
1. **Phase 7 Outputs**:
   - [`FootfallSummary`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/footfall.py#L86-L111):
     - `entry_events_per_zone: Dict[str, int]`: Total discrete entry occurrences per zone.
     - `transition_matrix: Dict[str, Dict[str, int]]`: Direct flow counts between zone pairs.
2. **Phase 8.1 Ground-Truth Representations**:
   - [`GroundTruthEvent`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/validation/ground_truth.py#L159-L240):
     - `event_type="entry"`: Grouped by `to_zone_id`.
     - `event_type="transition"`: Grouped by `(from_zone_id, to_zone_id)`.
   - [`GroundTruthSequence`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/validation/ground_truth.py#L318-L345): Sequence container supplying `events: List[GroundTruthEvent]`.

---

## 3. Aggregate Comparison Methodology

```
GroundTruthEvent collection (or GroundTruthSequence) + FootfallSummary
                           │
                           ▼
          Filter supported event types (entry, transition)
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
   If events empty:            If events present:
   - INSUFFICIENT_EVIDENCE     - Group GT entries by to_zone_id
   - metric_consistency = None - Group GT transitions by (from, to)
                               - Read predicted counts from summary
                                         │
                                         ▼
                               Evaluate Metric Consistency
                               - Entry counts per zone
                               - Transition counts per zone pair
                               - Record FootfallMetricDisagreement
                               - FOOTFALL_COMPARISON_EVALUATED
```

### 3.1 Zone Entry Counts
- **Expected Count**: Number of `GroundTruthEvent` records where `event_type == "entry"` and `to_zone_id == zone`.
  - Supports entries from exterior / unassigned areas (`from_zone_id is None`) and inter-zone entries (`from_zone_id == prior_zone`).
- **Predicted Count**: Read directly from `summary.entry_events_per_zone.get(zone, 0)`.
- **Comparison**: Evaluated for all relevant zones. Disagreements preserve signed difference and absolute difference.

### 3.2 Zone-to-Zone Transitions
- **Expected Count**: Number of `GroundTruthEvent` records where `event_type == "transition"`, `from_zone_id == z_from`, and `to_zone_id == z_to` (\(z_{\text{from}} \ne z_{\text{to}}\)).
- **Predicted Count**: Read directly from `summary.transition_matrix.get(z_from, {}).get(z_to, 0)`.
- **Comparison**: Evaluated for all valid zone pairs \((z_{\text{from}}, z_{\text{to}})\) with \(z_{\text{from}} \ne z_{\text{to}}\). Disagreements preserve signed difference and absolute difference.

### 3.3 Unique Presence Qualification
- While `FootfallSummary.unique_tracks_per_zone` reports distinct session-local `track_id` counts, ground-truth presence reflects true human individuals (`gt_id`).
- Track fragmentation (ID switches, temporary occlusions) can inflate tracker counts relative to ground-truth individuals.
- Therefore, tracker-level unique tracks and ground-truth individual counts are **not equivalent**, and no identity accuracy claim is made.

---

## 4. Data Structures & Output Representation

Implemented in [`src/validation/footfall_validation.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/validation/footfall_validation.py):

### 4.1 Comparison Records
- **`EntryCountComparison`**:
  - `zone_id`: Zone identifier.
  - `expected_count`: Ground-truth entry count.
  - `predicted_count`: Tracker-derived entry count from `FootfallSummary`.
  - `difference`: Signed difference (\(\text{predicted} - \text{expected}\)).
  - `absolute_difference`: Magnitude of count discrepancy.
  - `is_match`: Boolean flag (\(\text{expected} == \text{predicted}\)).
- **`TransitionCountComparison`**:
  - `from_zone_id`: Origin zone identifier.
  - `to_zone_id`: Destination zone identifier.
  - `expected_count`: Ground-truth transition count.
  - `predicted_count`: Tracker-derived transition count from `FootfallSummary`.
  - `difference`: Signed difference (\(\text{predicted} - \text{expected}\)).
  - `absolute_difference`: Magnitude of count discrepancy.
  - `is_match`: Boolean flag.
  - `transition_key`: String key formatted as `"from->to"`.

### 4.2 Disagreement Record
- **`FootfallMetricDisagreement`**:
  - `metric_type`: `"entry_count"` or `"transition_count"`.
  - `key`: Zone name or transition key (`"from->to"`).
  - `expected_count`: Ground-truth count.
  - `predicted_count`: Predicted count.
  - `difference`: Signed difference.
  - `absolute_difference`: Absolute difference.
  - `evidence_note`: Methodological explanation explicitly clarifying that an aggregate difference does not identify the specific low-level tracking, detection, or trajectory cause.

### 4.3 Validation Result Container
- **`FootfallValidationResult`**:
  - `dataset_name`: Identifier string.
  - `expected_entry_counts`: Map of zone \(\to\) ground-truth entry count.
  - `predicted_entry_counts`: Map of zone \(\to\) predicted entry count.
  - `entry_comparisons`: List of `EntryCountComparison`.
  - `expected_transition_counts`: Map of `"from->to"` \(\to\) ground-truth transition count.
  - `predicted_transition_counts`: Map of `"from->to"` \(\to\) predicted transition count.
  - `transition_comparisons`: List of `TransitionCountComparison`.
  - `total_compared_metrics`: Count of evaluated comparisons.
  - `matched_metric_count`: Count of comparisons where expected equals predicted.
  - `mismatched_metric_count`: Count of comparisons where expected differs from predicted.
  - `metric_consistency_rate`: Descriptive proportion \(\text{matched} / \text{total}\) (or `None` if empty).
  - `evidence_status`: `"FOOTFALL_COMPARISON_EVALUATED"` or `"INSUFFICIENT_EVIDENCE"`.
  - `evidence_note`: Explicit scope caveat.
  - `disagreements`: List of `FootfallMetricDisagreement`.

---

## 5. Evidence Status Classifications & Methodological Scope

| Status | Trigger Condition | Interpretation |
|---|---|---|
| **`INSUFFICIENT_EVIDENCE`** | Empty ground-truth events or zero usable entry/transition records. | No usable ground-truth events provided. Footfall metric consistency cannot be evaluated. Prevents artificial 0% or 100% metrics. |
| **`FOOTFALL_COMPARISON_EVALUATED`** | One or more valid ground-truth entry or transition events. | Aggregate event-count consistency computed across supplied test data. |

### Dataset Evidence Context:
- **Synthetic Fixtures**:
  - Exercise pipeline mechanics (event counting, discrepancy detection, serialization).
  - Software verification only; **not** physical real-world validation.
- **MERL Shopping Dataset**:
  - Provides development and retail context overhead video.
  - Does **not** contain formal continuous ground-truth trajectory or discrete footfall annotations without manual labeling.
- **MOT17**:
  - Multi-object tracking benchmark, primarily evaluating bounding-box association, not retail layout zones.
- **CAVIAR / Controlled Footage**:
  - May support stronger empirical footfall validation only when explicit ground-truth event annotations are supplied.
