# Phase 8.1 — Ground-Truth Representation & Validation Schema Specification

**Project**: Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
**Phase**: 8.1 (Schema & Test Infrastructure Only)  
**Status**: COMPLETE / VERIFIED  
**Regression Status**: 67/67 tests passing (55 Phase 0–7 baseline + 12 Phase 8.1 schema tests)  

---

## 1. Phase 8.1 Objective

The objective of Phase 8.1 is to establish a standardized, machine-readable ground-truth data schema and synthetic test infrastructure to support upcoming validation tasks in Phase 8.

Phase 8.1 is strictly limited to data representation, validation constraints, and JSON serialization. It does **not** implement:
- Homography calibration accuracy evaluation
- Spatial zone assignment validation
- Footfall counting validation
- Dwell time validation
- Tracker fragmentation analysis
- Hysteresis sensitivity analysis
- Heatmaps or spatial density maps
- Recommendations or layout optimization engines

> [!IMPORTANT]
> **Methodological Disclaimer**:  
> No real-world physical accuracy or ground-truth validity is established by Phase 8.1. All test fixtures are synthetic data structures created solely to verify schema integrity, validation logic, and serialization determinism. Phase 8.1 does not implement validation metrics or downstream analytics.

---

## 2. Schema Classes & Architecture

All ground-truth data structures are implemented using standard-library dataclasses in [`src/validation/ground_truth.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/validation/ground_truth.py) and exported via [`src/validation/__init__.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/validation/__init__.py).

```
src/validation/
├── __init__.py          # Public package exports
└── ground_truth.py      # Core data models and validation constraints
```

### 2.1 `GroundTruthPoint`
Represents an individual ground-truth observation of a person at a specific frame and timestamp.

| Field | Type | Required | Description |
|---|---|---|---|
| `gt_id` | `str` | Yes | Canonical identity string (e.g., `"P1"`, `"person_A"`) within a sequence. Non-empty. |
| `frame_idx` | `int` | Yes | Non-negative video frame index. |
| `timestamp_sec` | `float` | Yes | Non-negative finite video timestamp in seconds. |
| `image_bbox` | `Optional[List[float]]` | No | Bounding box `[x1, y1, x2, y2]` in image pixels. When present, must contain exactly 4 finite values. |
| `image_x` | `Optional[float]` | No | Image X coordinate (e.g. bottom-center). Must be paired with `image_y`. |
| `image_y` | `Optional[float]` | No | Image Y coordinate (e.g. bottom-center). Must be paired with `image_x`. |
| `floor_x` | `Optional[float]` | No | Ground-truth floor plane X coordinate. Must be paired with `floor_y`. |
| `floor_y` | `Optional[float]` | No | Ground-truth floor plane Y coordinate. Must be paired with `floor_x`. |
| `zone_id` | `Optional[str]` | No | Retail zone identifier string, or `None` if in an unassigned area. |
| `visible` | `bool` | No | Visibility flag indicating whether the person is visually unobstructed (default: `True`). |

#### Validation Constraints:
- `gt_id` must be a non-empty string.
- `frame_idx` must be a non-negative integer.
- `timestamp_sec` must be finite and $\ge 0.0$.
- `image_bbox`, if provided, must contain exactly four finite values.
- `(image_x is None) == (image_y is None)`: coordinate pairs must either both be present or both absent.
- `(floor_x is None) == (floor_y is None)`: coordinate pairs must either both be present or both absent.
- Missing coordinates are **not** automatically inferred.

---

### 2.2 `GroundTruthEvent`
Represents a manually or externally annotated discrete temporal event (entry, exit, or inter-zone transition).

| Field | Type | Required | Description |
|---|---|---|---|
| `event_type` | `str` | Yes | Event classification: must be one of `("entry", "exit", "transition")`. |
| `gt_id` | `str` | Yes | Canonical identity string of the individual associated with the event. |
| `frame_idx` | `int` | Yes | Non-negative frame index where the event occurred. |
| `timestamp_sec` | `float` | Yes | Non-negative finite video timestamp in seconds. |
| `from_zone_id` | `Optional[str]` | Conditional | Origin zone string, or `None` if entering from outside. |
| `to_zone_id` | `Optional[str]` | Conditional | Destination zone string, or `None` if exiting to outside. |

#### Event Coherence Rules:
- **`entry`**: `to_zone_id` is mandatory; `from_zone_id` may be `None` (entered from outside) or another zone.
- **`exit`**: `from_zone_id` is mandatory; `to_zone_id` may be `None` (exited to outside) or another zone.
- **`transition`**: Both `from_zone_id` and `to_zone_id` are mandatory and must be different (`from_zone_id != to_zone_id`).
- Events are purely representational; they are **never** calculated automatically in this schema.

---

### 2.3 `GroundTruthCalibrationPoint`
Represents an image-to-floor correspondence point used for homography calibration and verification.

| Field | Type | Required | Description |
|---|---|---|---|
| `label` | `str` | Yes | Non-empty reference label (e.g. `"P1"`, `"CP1"`). |
| `image_x` | `float` | Yes | Finite pixel X coordinate in camera space. |
| `image_y` | `float` | Yes | Finite pixel Y coordinate in camera space. |
| `floor_x` | `float` | Yes | Finite floor X coordinate in metric or grid units. |
| `floor_y` | `float` | Yes | Finite floor Y coordinate in metric or grid units. |
| `role` | `str` | Yes | Functional calibration role: `"training"` or `"held_out"` (default: `"training"`). |

#### Critical Distinction: Training vs. Held-Out Points
To avoid methodological circularity in camera calibration assessment:
- **`role="training"`**: Reference points utilized to compute the $3 \times 3$ homography transformation matrix $\mathbf{H}$. Residual errors on these points represent *training reprojection error*.
- **`role="held_out"`**: Independent check points that were **not** used to estimate $\mathbf{H}$. Errors on these points measure *out-of-sample geometric consistency*.
- The schema strictly preserves this distinction. Training points must **never** be treated as independent validation points.

---

### 2.4 `GroundTruthSequence`
Container representing an entire ground-truth annotated video sequence.

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | `str` | Yes | Version identifier (e.g., `"1.0"`). |
| `sequence_id` | `str` | Yes | Unique sequence identifier (e.g., `"SYNTHETIC_S01"`, `"controlled_20260920_S01"`). |
| `source` | `str` | Yes | Origin description (e.g., `"controlled_recording"`, `"synthetic_fixture"`). |
| `fps` | `float` | Yes | Finite positive frame rate in frames per second ($> 0$). |
| `points` | `List[GroundTruthPoint]` | No | List of person observation points. |
| `events` | `List[GroundTruthEvent]` | No | List of ground-truth events. |
| `calibration_points` | `List[GroundTruthCalibrationPoint]` | No | List of calibration reference points. |
| `metadata` | `Optional[Dict[str, Any]]` | No | Optional dictionary for supplementary sequence metadata. |

#### Sequence Integrity Rules:
- `fps` must be finite and $> 0.0$.
- Duplicate observations: For any given `gt_id`, at most one observation point is allowed per `frame_idx`.
- Temporal ordering: For any given `gt_id`, observations must be non-decreasing in both `frame_idx` and `timestamp_sec`.
- Calibration uniqueness: Calibration point labels must be unique within the sequence.

---

## 3. Serialization Specification

All classes provide standard-library JSON serialization methods:
- `to_dict()`: Converts instance into a deterministic, primitive dictionary.
- `from_dict(data)`: Reconstructs instance from dictionary with full validation.
- `to_json(indent=2)`: Serializes directly to a formatted JSON string.
- `from_json(json_str)`: Reconstructs from JSON string.
- `save_to_json(file_path)`: Writes to a JSON file on disk.
- `load_from_json(file_path)`: Reads and parses a JSON file from disk.

No external serialization dependencies (e.g., Pydantic, Marshmallow, Pandas, NumPy pickle) are used.

---

## 4. Synthetic Test Fixtures

The test suite in [`tests/test_ground_truth.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/tests/test_ground_truth.py) defines a deterministic synthetic fixture sequence (`synthetic_sequence`) containing:
- 2 distinct ground-truth identities (`person_A`, `person_B`)
- Observations across frames 0–3 with bounding boxes and floor coordinates
- Point locations spanning both assigned retail zones (`zone_entrance`, `zone_shelf`, `zone_reading`) and an unassigned exterior state (`None`)
- 3 coherent temporal events (1 `entry`, 1 `transition`, 1 `exit`)
- 4 training calibration points (`P1`–`P4`) and 2 held-out check points (`CP1`–`CP2`)

> **Notice**: Synthetic fixtures serve solely to verify data structures, boundary validations, and serialization fidelity in CI/unit tests. They provide zero quantitative accuracy or physical evidence.

---

## 5. Scope Boundaries

Phase 8.1 strictly enforces the following boundaries:
- **No homography accuracy evaluation** (calculating reprojection or check point errors from GT points is deferred).
- **No zone validation metrics** (calculating confusion matrices or IoU between tracker and GT zones is deferred).
- **No footfall or dwell validation** (comparing tracker footfall/visits to GT events is deferred).
- **No fragmentation metrics** (matching ByteTrack session IDs to GT IDs is deferred).
- **No database, dashboard, or report generation**.
