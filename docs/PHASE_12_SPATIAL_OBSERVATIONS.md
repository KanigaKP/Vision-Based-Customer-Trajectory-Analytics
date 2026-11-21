# Phase 12.1 — Evidence-Gated, Explainable Spatial Observation Prototype

**Project**: Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
**Roadmap Phase**: Phase 12 — Spatial Analytics & Layout Insights (Slice 12.1: Evidence-Gated Observation Prototype)  
**Status**: COMPLETE / OPERATIONALLY VERIFIED  
**Regression Baseline**: 200/200 tests passing  
**Verification Runners**:
- [`scripts/verify_phase_08.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/scripts/verify_phase_08.py): PASSED (70/70 component validation tests)
- [`scripts/verify_phase_10.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/scripts/verify_phase_10.py): PASSED (MOT17 image sequence & MERL video container integration runs)

---

## 1. Executive Summary & Objective

Phase 12.1 implements a minimal, deterministic analytics layer that consumes pre-computed spatial summaries from Phase 9 (Dwell), Phase 10 (Density), and Phase 11 (Transitions), and emits:
1. **Plain, traceable descriptive observations** where directly supported by caller-provided evidence context; or
2. An explicit **"insufficient evidence"** result with detailed explanations when input evidence is unrepresentative, uncalibrated, non-retail, or missing.

### Core Architectural Invariants:
1. **Explicit Caller Provenance Gating**:
   - The engine never infers evidence tiers, suitability, or representativeness from numeric values, thresholds, or filenames.
   - Requires caller-supplied [`EvidenceContext`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/spatial_observations.py) specifying `environment_type`, `calibration_status`, and `coverage_status`.
   - If metadata indicates boundary-truncated footage (e.g., local MERL video run) or non-retail environments (e.g., MOT17 pedestrian street), retail spatial observations are explicitly withheld.
2. **Pure Analytics Layer**:
   - Consumes existing [`DwellSummary`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/dwell.py), [`TrajectoryDensityGrid`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/density.py), and [`DirectTransitionSummary`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/transitions.py).
   - Does not modify detector, tracker, homography, zone assignment, footfall, dwell, density, or transition algorithms.
3. **Strictly Non-Causal Descriptive Observations**:
   - Every emitted observation references its exact source metrics and numeric values.
   - Describes observed sample counts; never claims customer intent, preference, browsing interest, or sales conversion.
   - Tracker IDs are clip-local tracker IDs, not unique physical customers.
   - Observed dwell durations are discrete frame spans, not verified continuous physical dwell.
   - Floor coordinates and extents are in configured floor units; physical metric scale is unverified unless confirmed by caller provenance.
4. **Strictly Non-Prescriptive**:
   - No zone rankings, scores, or "top/popular/underperforming" labels.
   - No store layout redesign recommendations or commercial claims.
   - No invented numeric thresholds (e.g., no arbitrary dwell cutoffs).
5. **No Phase 13 Scope Spillover**:
   - Pure in-memory deterministic dataclass models.
   - No JSON report writing, file export, SQLite database persistence, or Streamlit dashboards.

---

## 2. Evidence-Gating Eligibility Rules

An input is evaluated by [`evaluate_evidence_eligibility()`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/spatial_observations.py) against the following strict criteria:

| Dimension | Acceptable Values | Rejection Condition | Withholding Explanation |
|---|---|---|---|
| **Evidence Context Presence** | `EvidenceContext` instance | `None` | *"Evidence context is missing; provenance, environment type, and spatial coverage cannot be verified without explicit caller metadata."* |
| **Environment Type** | `retail_store`, `retail_floor`, `store_environment`, `supermarket`, `grocery_store` | Non-retail (e.g., `pedestrian_outdoor`, `laboratory`, `synthetic`) | *"Environment type '{env}' is not a retail store environment; retail spatial observations cannot be generated from non-retail settings (e.g. outdoor pedestrian footage)."* |
| **Coverage Status** | `representative`, `full_coverage`, `adequate_in_zone` | `boundary_truncated` (e.g., MERL case) | *"Coverage status is 'boundary_truncated': spatial trajectories suffer from severe camera field-of-view boundary truncation and cannot support representative spatial observations across store zones."* |
| **Coverage Status** | Same as above | `sparse_unrepresentative`, `unknown` | *"Coverage status '{cov}' indicates trajectory observations are unrepresentative or unverified."* |
| **Calibration Status** | `surveyed_metric`, `independently_checked`, `field_verified`, `measured_floor_grid` | `unverified_configured_units`, `configured_only`, `unverified`, `uncalibrated` | *"Calibration status is '{cal}': spatial calibration has not been independently surveyed or field-verified; unverified or configured-only coordinates cannot support spatial layout observations."* |
| **Input Data Population** | Populated Phase 9–11 summaries | Zero episodes, zero points, or all summaries `None` | *"No populated trajectory observations or analytics summaries were provided to evaluate."* |

---

## 3. Evaluation of Current Repository Evidence

### 3.1 Local MERL Recorded Video Run
When the local MERL run diagnostics (`data/merl/10_1_crop.mp4`, 100 frames, 74 points, 3 binned, 71 out-of-bounds) are evaluated under accurate caller provenance:
```python
context = EvidenceContext(
    environment_type="retail_store",
    calibration_status="unverified_configured_units",
    coverage_status="boundary_truncated",
    notes="MERL recorded video run with 71/74 out-of-bounds points",
)
```
**Outcome**:
- `is_eligible`: `False`
- `status`: `"insufficient_evidence"`
- `observations`: `()` (empty)
- `reasons_withheld`:
  1. *"Coverage status is 'boundary_truncated': spatial trajectories suffer from severe camera field-of-view boundary truncation and cannot support representative spatial observations across store zones."*
  2. *"Calibration status is 'unverified_configured_units': spatial calibration has not been independently surveyed or field-verified; unverified or configured-only coordinates cannot support spatial layout observations."*

### 3.2 Local MOT17 Image Sequence Run
When the local MOT17 run diagnostics (`data/mot17/.../img1/`, 10 frames, 11 tracks, 94 points) are evaluated:
```python
context = EvidenceContext(
    environment_type="pedestrian_outdoor",
    calibration_status="unverified_configured_units",
    coverage_status="representative",
    notes="MOT17-04 outdoor pedestrian tracking sequence (unverified calibration)",
)
```
**Outcome**:
- `is_eligible`: `False`
- `status`: `"insufficient_evidence"`
- `observations`: `()` (empty)
- `reasons_withheld`:
  1. *"Environment type 'pedestrian_outdoor' is not a retail store environment; retail spatial observations cannot be generated from non-retail settings (e.g. outdoor pedestrian footage)."*
  2. *"Calibration status is 'unverified_configured_units': spatial calibration has not been independently surveyed or field-verified; unverified or configured-only coordinates cannot support spatial layout observations."*

---

## 4. API & Data Structures

### 4.1 `EvidenceContext` (Frozen Dataclass)
```python
@dataclass(frozen=True)
class EvidenceContext:
    environment_type: str
    calibration_status: str
    coverage_status: str
    notes: Optional[str] = None
```
- Validates non-empty string types at instantiation.
- Immutable; exposes `to_dict()`.

### 4.2 `DescriptiveSpatialObservation` (Frozen Dataclass)
```python
@dataclass(frozen=True)
class DescriptiveSpatialObservation:
    observation_id: str
    observation_type: str
    zone_id: Optional[str]
    summary_text: str
    source_metrics: Dict[str, Any]
    caveats: Tuple[str, ...]
```
- Holds exact numeric source metrics and non-causal descriptive text.
- Immutable; exposes `to_dict()`.

### 4.3 `SpatialObservationResult` (Frozen Dataclass)
```python
@dataclass(frozen=True)
class SpatialObservationResult:
    is_eligible: bool
    status: str
    observations: Tuple[DescriptiveSpatialObservation, ...]
    reasons_withheld: Tuple[str, ...]
    evidence_context: Optional[EvidenceContext]
    input_diagnostics: Dict[str, Any]
```

### 4.4 Engine & Convenience Function
- [`SpatialObservationEngine.analyze(evidence_context, dwell_summary, transition_summary, density_grid)`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/spatial_observations.py): Pure in-memory analyzer.
- [`generate_spatial_observations(...)`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/spatial_observations.py): Convenience entry point.

---

## 5. Verification & Test Inventory

The focused test suite [`tests/test_spatial_observations.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/tests/test_spatial_observations.py) contains 12 comprehensive tests:

| Test Name | Verifies |
|---|---|
| `test_missing_evidence_context_withholds_observations` | `None` context withholds observations with explicit missing-context reason. |
| `test_mot17_like_non_retail_environment_withholds_observations` | Outdoor pedestrian environment is rejected with non-retail explanation. |
| `test_merl_like_boundary_truncated_coverage_withholds_observations` | Boundary-truncated coverage is rejected with truncation explanation. |
| `test_uncalibrated_status_withholds_observations` | Uncalibrated setup is rejected with calibration explanation. |
| `test_unverified_calibration_withholds_observations` | Unverified or configured-only calibration status is rejected with unverified calibration explanation. |
| `test_multiple_ineligibility_reasons_accumulate` | Multiple disqualifying metadata entries accumulate cleanly. |
| `test_eligible_evidence_context_produces_traceable_observations` | Eligible context generates traceable observations for activity, dwell, transitions, and density. |
| `test_language_is_strictly_non_causal_and_includes_caveats` | Prohibits evaluative terms ('popular', 'intent', 'recommend', 'unique people') and verifies caveats. |
| `test_empty_summaries_withholds_observations` | Eligible context with zero data yields insufficient evidence. |
| `test_evidence_context_validation_rejections` | Empty strings in `EvidenceContext` raise `ValueError`. |
| `test_serialization_round_trip` | Complete dictionary / JSON serializability via `to_dict()`. |
| `test_frozen_dataclass_immutability` | Frozen dataclasses prevent accidental in-place mutation. |

---

## 6. Evidence Classification Matrix

| Category | Status | Boundary & Justification |
| :--- | :---: | :--- |
| **IMPLEMENTED** | **YES** | Data models (`EvidenceContext`, `DescriptiveSpatialObservation`, `SpatialObservationResult`), eligibility evaluator, observation generator, and test suite are implemented. |
| **OPERATIONALLY VERIFIED** | **YES** | 12 focused tests pass in [`tests/test_spatial_observations.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/tests/test_spatial_observations.py); full 200-test regression suite passes; Phase 8 runner (70 tests) passes; Phase 10 integration runner passes. |
| **QUANTITATIVELY VALIDATED** | **NO** | Software mechanics are verified using synthetic fixtures. No independent ground truth has been benchmarked for spatial observation accuracy. |
| **RETAIL-PERFORMANCE VALIDATED** | **NO** | Emitted observations are strictly non-causal descriptions of discrete observations. No claims regarding shopper intent, dwell-to-purchase correlation, layout optimization, or sales uplift are asserted. The MERL and MOT17 runs are properly withheld. |

---

> [!NOTE]
> Phase 12.1 is complete. Execution halts strictly here without proceeding to Phase 12.2 or Phase 13.
