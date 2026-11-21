# Phase 13.1 — Local Evidence-Aware Analysis Report Export

**Project**: Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
**Roadmap Phase**: Phase 13 — Export & Storage (Slice 13.1: Local Analysis Report Export)  
**Status**: COMPLETE / OPERATIONALLY VERIFIED  
**Regression Baseline**: 213/213 tests passing (200 baseline + 13 Phase 13 report tests)  
**Verification Runners**:
- [`scripts/verify_phase_08.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/scripts/verify_phase_08.py): PASSED (70/70 component validation tests)
- [`scripts/verify_phase_10.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/scripts/verify_phase_10.py): PASSED (MOT17 sequential frame and MERL video container pipeline checks)

---

## 1. Executive Summary & Objective

Phase 13.1 implements a lightweight, reproducible local reporting layer that serializes explicitly provided analysis-run metadata and available Phase 9–12 analytics summaries into a versioned, deterministic JSON report.

### Core Architectural Invariants:
1. **Versioned & Deterministic Schema**:
   - Schema version string: `SCHEMA_VERSION = "1.0.0"`.
   - Keys and sections are generated in fixed, deterministic dictionary order.
2. **Explicit Caller Run & Provenance Metadata**:
   - Requires [`AnalysisRunMetadata`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/reporting/analysis_report.py) with explicit `run_id`, `dataset_name`, and `source_type`.
   - Allowed `source_type` values are strictly restricted to recorded/offline sources: `"recorded_video"`, `"image_sequence"`, `"synthetic_fixture"`, and `"controlled_trial"` (where `controlled_trial` represents recorded project trial footage).
   - Live video streams (`"live_stream"`, RTSP streams, live CCTV, webcam capture) are strictly prohibited and rejected by validation.
   - The engine never infers evidence tiers, dataset origins, or calibration status from numeric values, filenames, or directory paths.
3. **Truthful Representation of Missing vs. Zero Data**:
   - Includes only summaries actually supplied by the caller.
   - Missing Phase 9 (Dwell), Phase 10 (Density), Phase 11 (Transitions), or Phase 12 (Observations) inputs are marked explicitly as `{"status": "not_available", "reason": "..."}`, never silently zero-filled.
   - Evaluated zero-count metrics (e.g. an evaluated retail zone with 0 visit runs) are preserved as numeric `0` within an `"available"` section, distinguishable from omitted data.
4. **Privacy & Clip-Local Tracker Identity Boundaries**:
   - Does not serialize raw person identities or expose per-track frame/coordinate histories.
   - All track counts are explicitly designated as clip-local tracker IDs assigned by ByteTrack, **not unique human customers**, noting upward count bias from tracking fragmentation.
5. **Physical Dwell & Density Semantics**:
   - Dwell metrics explicitly designate observed temporal spans ($t_{\text{exit}} - t_{\text{entry}}$) between discrete sampled detections, not verified continuous physical dwell.
   - Density grids explicitly designate discrete sampled detection counts at frame sampling rate, not simultaneous physical occupancy or unique visitors.
6. **Local Output Location**:
   - Reports are written strictly to the repository's established ignored reports directory (`outputs/reports/`).
   - Does not modify `.gitignore`.
   - No external databases, no cloud uploads, no PDF rendering, no Streamlit UI.

---

## 2. Report Schema Structure (`v1.0.0`)

An exported JSON report follows this deterministic structure:

```json
{
  "schema_version": "1.0.0",
  "report_id": "phase_13_analysis_report_merl_sample",
  "generated_at_utc": "2026-10-30T03:16:55.347417+00:00",
  "metadata": {
    "run_id": "phase_10_merl_sample",
    "dataset_name": "merl_shopping_dataset",
    "source_type": "recorded_video",
    "target_fps": 5.0,
    "total_frames_evaluated": 100,
    "source_path": "data/merl/10_1_crop.mp4",
    "notes": "Local MERL video container run evaluated through Phase 10 & 12"
  },
  "evidence_context": {
    "status": "available",
    "environment_type": "retail_store",
    "calibration_status": "unverified_configured_units",
    "coverage_status": "boundary_truncated",
    "notes": "Operational MERL 100-frame video container integration check"
  },
  "dwell_analytics": {
    "status": "not_available",
    "reason": "Dwell analytics summary was not supplied for this run."
  },
  "density_analytics": {
    "status": "available",
    "bounds": {
      "min_x": 0.0,
      "max_x": 10.0,
      "min_y": 0.0,
      "max_y": 6.0,
      "unit": "configured_floor_units"
    },
    "grid_shape": [12, 20],
    "cell_size": [0.5, 0.5],
    "total_evaluated_points": 74,
    "total_binned_points": 3,
    "out_of_bounds_count": 71,
    "untransformed_count": 0,
    "max_cell_density": 2,
    "conservation_invariant_verified": true,
    "caveats": [
      "Point density accumulates discrete sampled detections per grid cell at the frame sampling rate; it does NOT measure unique visitors, physical occupancy, or dwell duration.",
      "Coordinates and bounds are expressed in configured floor units; physical metric scale has not been independently surveyed."
    ]
  },
  "transition_analytics": {
    "status": "not_available",
    "reason": "Zone transitions summary was not supplied for this run."
  },
  "spatial_observations": {
    "status": "insufficient_evidence",
    "is_eligible": false,
    "observations_count": 0,
    "observations": [],
    "reasons_withheld": [
      "Coverage status is 'boundary_truncated': spatial trajectories suffer from severe camera field-of-view boundary truncation and cannot support representative spatial observations across store zones.",
      "Calibration status is 'unverified_configured_units': spatial calibration has not been independently surveyed or field-verified; unverified or configured-only coordinates cannot support spatial layout observations."
    ],
    "caveats": [
      "Spatial observations are plain descriptive statements of discrete sample counts where evidence is eligible.",
      "Where evidence is insufficient, observations are withheld with explicit causal reasons.",
      "No retail layout optimization recommendations, zone rankings, or customer intent claims are asserted."
    ]
  },
  "evidence_summary": {
    "environment_type": "retail_store",
    "calibration_status": "unverified_configured_units",
    "coverage_status": "boundary_truncated",
    "spatial_observation_eligibility": "insufficient_evidence",
    "classification": {
      "implemented": true,
      "operationally_verified": true,
      "quantitatively_validated": false,
      "retail_performance_validated": false
    },
    "disclaimer": "This report is a software artifact recording descriptive metrics and evidence gating status. It does NOT establish empirical accuracy, physical ground truth, or retail layout performance."
  }
}
```

---

## 3. Treatment of Missing vs. Zero Data

| Scenario | Input Parameter | Serialized Section Status | Representation in JSON |
|---|---|:---:|---|
| **Summary Not Supplied** | `dwell_summary=None` | `"not_available"` | `{"status": "not_available", "reason": "Dwell analytics summary was not supplied for this run."}` |
| **Zone Evaluated with Zero Activity** | `ZoneActivitySummary(observed_run_count=0)` | `"available"` | `{"status": "available", "zone_activity_summaries": {"aisle_1": {"observed_run_count": 0, ...}}}` |
| **Observations Withheld** | `SpatialObservationResult(is_eligible=False)` | `"insufficient_evidence"` | `{"status": "insufficient_evidence", "is_eligible": false, "observations_count": 0, "reasons_withheld": [...]}` |

---

## 4. API & Data Structures

### 4.1 `AnalysisRunMetadata` (Frozen Dataclass)
```python
@dataclass(frozen=True)
class AnalysisRunMetadata:
    run_id: str
    dataset_name: str
    source_type: str
    target_fps: Optional[float] = None
    total_frames_evaluated: Optional[int] = None
    source_path: Optional[str] = None
    notes: Optional[str] = None
```
- Validates non-empty string types for required fields.
- Validates `source_type` against `ALLOWED_SOURCE_TYPES = ("recorded_video", "image_sequence", "synthetic_fixture", "controlled_trial")`.
  - Offline/recorded sources only (`controlled_trial` represents recorded project trial footage).
  - Live video streams (`"live_stream"`, RTSP, CCTV, webcam) are strictly prohibited and rejected by validation.
- Validates numeric ranges for optional sampling rate and frame counts.

### 4.2 `AnalysisReport` (Frozen Dataclass)
```python
@dataclass(frozen=True)
class AnalysisReport:
    schema_version: str
    report_id: str
    generated_at_utc: str
    metadata: AnalysisRunMetadata
    evidence_context_section: Dict[str, Any]
    dwell_section: Dict[str, Any]
    density_section: Dict[str, Any]
    transition_section: Dict[str, Any]
    observation_section: Dict[str, Any]
    evidence_summary: Dict[str, Any]
```
- Exposes `to_dict()` and `to_json(indent=2)`.

### 4.3 Builder & Exporter Functions
- [`build_analysis_report(...)`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/reporting/analysis_report.py): Pure in-memory constructor.
- [`export_analysis_report(report, output_path=None, reports_dir='outputs/reports')`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/reporting/analysis_report.py): Exports report to a local JSON file.
- [`generate_and_export_analysis_report(...)`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/reporting/analysis_report.py): Convenience wrapper.

---

## 5. Verification & Test Inventory

The focused test suite [`tests/test_analysis_report.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/tests/test_analysis_report.py) contains 13 comprehensive tests:

| Test Name | Verifies |
|---|---|
| `test_metadata_validation_rejections` | Rejects empty strings, negative FPS, negative frame counts, and disallowed types. |
| `test_source_type_prohibits_live_streams_and_accepts_recorded_sources` | Prohibits `live_stream`, RTSP, and CCTV while accepting permitted offline/recorded sources. |
| `test_build_report_requires_valid_metadata` | Enforces explicit `AnalysisRunMetadata` instance requirement. |
| `test_complete_inputs_report` | Correctly serializes report with all Phase 9–12 inputs populated. |
| `test_partial_inputs_report_marks_missing_as_not_available` | Omitted inputs are marked `not_available`, not zero-filled. |
| `test_missing_versus_zero_distinction` | Evaluated zero counts remain distinguishable from unsupplied analytics. |
| `test_merl_insufficient_evidence_serialization` | Accurately records MERL boundary truncation and unverified calibration. |
| `test_mot17_insufficient_evidence_serialization` | Accurately records MOT17 outdoor pedestrian setting and unverified calibration. |
| `test_privacy_boundaries_and_clip_local_track_labels` | Prohibits raw person/trajectory coordinate histories; labels track counts as clip-local tracker IDs. |
| `test_caveats_populated_in_all_available_sections` | Attaches dwell span, point density, and configured unit caveats. |
| `test_deterministic_serialization` | Guarantees byte-for-byte identical JSON serialization across calls. |
| `test_export_analysis_report_to_disk` | Verifies export to disk, JSON round-trip, and schema adherence. |
| `test_generate_and_export_convenience_function` | Verifies one-step convenience function. |

---

## 6. Evidence Classification Matrix

| Component | Implemented | Operationally Verified | Quantitatively Validated | Retail-Performance Validated | Empirical Boundary / Limitation |
|---|:---:|:---:|:---:|:---:|---|
| **Run Metadata Schema** ([`AnalysisRunMetadata`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/reporting/analysis_report.py#L60-L98)) | **YES** | **YES** | **NO** | **NO** | Metadata representation only; relies on caller-supplied provenance. |
| **Analysis Report Model** ([`AnalysisReport`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/reporting/analysis_report.py#L270-L300)) | **YES** | **YES** | **NO** | **NO** | Structured schema representation; faithfully records available vs. missing sections and evidence withholding notices. |
| **Deterministic Local JSON Exporter** ([`export_analysis_report`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/reporting/analysis_report.py#L340-L375)) | **YES** | **YES** | **NO** | **NO** | Software export mechanism writing to `outputs/reports/`; does not modify external databases or validate empirical accuracy. |

> [!IMPORTANT]
> **Methodological Boundary**: A generated analysis report is a software artifact recording descriptive metrics and evidence gating status. Generating or serializing a report **does NOT establish empirical accuracy, physical ground truth, or retail layout performance.**

---

*Phase 13.1 complete. Execution halts strictly here without proceeding to Phase 13.2 or Phases 14–15.*
