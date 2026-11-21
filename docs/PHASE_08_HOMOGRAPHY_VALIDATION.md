# Phase 8.2 — Independent Homography Validation Specification

**Project**: Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
**Phase**: 8.2 (Homography Validation Layer)  
**Status**: COMPLETE / VERIFIED  
**Regression Status**: 76/76 tests passing (55 Phase 0–7 baseline + 12 Phase 8.1 schema tests + 9 Phase 8.2 homography validation tests)  

---

## 1. Phase 8.2 Objective

The objective of Phase 8.2 is to implement a mathematically defensible validation procedure for camera-to-floor homography transformations using independent held-out reference points.

Homography estimation ($3 \times 3$ matrix $\mathbf{H}$) requires a minimum of 4 non-collinear correspondence pairs. Historically in computer vision projects, calibration quality is often reported using **training reprojection error** (the residual distance when reprojecting the same points used to estimate $\mathbf{H}$). In academic and technical reviews, citing training reprojection error as proof of real-world accuracy is considered methodological circularity (in-sample overfitting).

Phase 8.2 establishes:
1. Strict separation between **training calibration points** (used exclusively to fit $\mathbf{H}$) and **held-out check points** (used exclusively for out-of-sample error evaluation).
2. Verification that held-out points are **never** supplied to the estimator (`cv2.findHomography`).
3. An explicit **`INSUFFICIENT_EVIDENCE`** reporting status when no independent held-out points are provided, preventing manufactured validation claims.
4. **Full-Precision Numeric Evaluation**: Independent validation errors are computed using the unrounded vectorized `transform_points()` method. Presentation serialization may format or round values only after error measurement is complete, ensuring validation precision is never degraded by display-oriented rounding.

---

## 2. Distinction Between Training Reprojection and Held-Out Validation

| Concept | Fitting Role | What It Measures | Academic & Viva Validity |
|---|---|---|---|
| **Training Reprojection Error** | `role="training"` (used in `cv2.findHomography`) | Numerical residual of the mathematical fit on points supplied to the solver. | **Fitting diagnostic only**. Does *not* validate out-of-sample metric accuracy. |
| **Independent Held-Out Error** | `role="held_out"` (strictly excluded from fitting) | Euclidean distance between unrounded predicted floor coordinates and physically measured reference coordinates. | **Valid out-of-sample geometric consistency**. Reflects true perspective mapping error on un-fitted locations. |

> [!IMPORTANT]
> Modifying held-out reference coordinates changes the validation error but does **not** change the estimated homography matrix $\mathbf{H}$. This architectural invariant is verified by unit test `test_held_out_independence_does_not_alter_fitted_matrix`.

---

## 3. Data Representation & Numeric Precision

Phase 8.2 consumes the Phase 8.1 calibration representation and produces structured, serializable validation results implemented in [`src/validation/homography_validation.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/validation/homography_validation.py):

### 3.1 Input Representation
- **`GroundTruthCalibrationPoint`**:
  - `label`: Point identifier (e.g. `'P1'`, `'CP1'`).
  - `image_x`, `image_y`: Image pixel coordinates.
  - `floor_x`, `floor_y`: Metric or grid floor coordinates.
  - `role`: `"training"` or `"held_out"`.
- Can be provided directly as a `List[GroundTruthCalibrationPoint]` or extracted from a `GroundTruthSequence`.

### 3.2 Output Representation
- **`HeldOutPointValidation`**:
  - `label`: Check point identifier.
  - `image_point`: `(image_x, image_y)`.
  - `reference_floor_point`: `(floor_x, floor_y)` ground-truth reference.
  - `predicted_floor_point`: Full-precision `(pred_x, pred_y)` produced by $\mathbf{H}$ via `transform_points()`.
  - `error`: Unrounded Euclidean distance $\sqrt{(\text{pred}_x - \text{ref}_x)^2 + (\text{pred}_y - \text{ref}_y)^2}$.
  - `unit`: Metric unit string (e.g. `'metres'`).
- **`HomographyValidationResult`**:
  - `training_point_count`: Number of training correspondence pairs ($N_{\text{train}} \ge 4$).
  - `held_out_point_count`: Number of held-out check points ($N_{\text{held}} \ge 0$).
  - `training_reprojection_mean_error`: Mean in-sample reprojection error.
  - `training_reprojection_max_error`: Maximum in-sample reprojection error.
  - `held_out_mean_error`: Out-of-sample mean Euclidean error across held-out points (or `None`).
  - `held_out_max_error`: Out-of-sample maximum Euclidean error across held-out points (or `None`).
  - `held_out_results`: List of `HeldOutPointValidation` records.
  - `evidence_status`: `"HELD_OUT_EVALUATED"` or `"INSUFFICIENT_EVIDENCE"`.
  - `evidence_note`: Methodological explanation of validation validity.
  - `homography_matrix`: Serialized $3 \times 3$ transformation matrix.

> **Precision Invariant**:  
> In earlier prototypes, helper methods like `transform_point()` rounded outputs to 2 decimal places ($1\text{ cm}$ discretization). In Phase 8.2, all held-out validation points are evaluated through the unrounded `transform_points()` vectorized pathway to prevent artificial discretization errors from corrupting sub-centimeter accuracy benchmarks. Presentation formatting in `to_dict()` serializes standard Python floats without measurement rounding.

---

## 4. Validation Procedure

```
GroundTruthCalibrationPoint collection (or GroundTruthSequence)
   │
   ├─► Filter role == "training"  (N >= 4)
   │      │
   │      ▼
   │   HomographyCalibrator (computes H, training reprojection diagnostics)
   │
   └─► Filter role == "held_out"
          │
          ├─► If N == 0:
          │      Return INSUFFICIENT_EVIDENCE (held_out_mean_error = None)
          │
          └─► If N >= 1:
                 Transform held-out image points via calibrator.transform_points() (unrounded)
                 Compute full-precision Euclidean error vs reference floor point
                 Return HELD_OUT_EVALUATED with per-point and aggregate metrics
```

---

## 5. Evidence Status Interpretation & Real-World Dataset Requirements

Phase 8.2 classifies homography evidence levels explicitly:

1. **`INSUFFICIENT_EVIDENCE`**:
   - Triggered when only training points are provided ($N_{\text{held}} = 0$).
   - The estimated matrix $\mathbf{H}$ is returned with training reprojection errors, but no independent validation claim is made.
2. **`HELD_OUT_EVALUATED` (Synthetic / Small Sample)**:
   - Evaluates geometric consistency against supplied check points.
   - **Important Limitation**: One or a small number of held-out points (e.g., 1–3 check points) is **insufficient for a broad physical accuracy claim**. Point errors can vary across different floor regions due to lens distortion and perspective foreshortening.
3. **Dataset Provenance Requirements for True Validation**:
   - **MERL Shopping Dataset**: Functions as **development / retail-context demonstration data**. The camera view has non-zero lens distortion and lacks officially measured ground metric surveying. Check points in `config.yaml` demonstrate pipeline feasibility, but cannot be claimed as surveyed metric ground truth without on-site calibration records.
   - **CAVIAR Benchmark Dataset**: An intended external benchmark source featuring calibrated corridor surveillance geometry where ground-plane measurements are applicable.
   - **Controlled Student-Recorded Footage**: An intended empirical validation source where physical floor markers measured on-site with metric tape (as specified in `docs/CONTROLLED_VIDEO_RECORDING_SPEC.md`) provide genuine ground-truth reference points for both training and held-out validation.
   - **Current Repository Status**: The repository **does not currently contain quantitative CAVIAR or controlled-footage validation evidence**. No claim is made that CAVIAR has been executed or that physical accuracy is established for this project prior to ingesting authentic calibration datasets.

---

## 6. Synthetic Unit Test Verification

The module is verified by 9 deterministic unit tests in [`tests/test_homography_validation.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/tests/test_homography_validation.py):

| Test Name | Verified Invariant |
|---|---|
| `test_valid_training_and_held_out_evaluation` | Verifies estimation on 4 training points and accurate error computation across 3 held-out points. |
| `test_held_out_independence_does_not_alter_fitted_matrix` | Proves that altering held-out reference coordinates changes validation error but leaves $\mathbf{H}$ strictly identical. |
| `test_insufficient_evidence_when_no_held_out_points` | Verifies that passing 0 held-out points produces `INSUFFICIENT_EVIDENCE` status with `None` errors. |
| `test_fewer_than_four_training_points_raises_error` | Verifies that supplying $< 4$ training points raises a clear `ValueError`. |
| `test_collinear_training_points_raises_error` | Verifies that degenerate collinear training points raise a geometry `ValueError`. |
| `test_preservation_of_labels_and_metadata` | Verifies labels, metric units, and dataset names are preserved in per-point results. |
| `test_serialization_round_trip` | Verifies full dictionary conversion (`to_dict` / `from_dict`) without information loss. |
| `test_validate_sequence_adapter` | Verifies validation seamlessly consumes `GroundTruthSequence` instances. |
| `test_unrounded_transformation_path_precision` | Explicitly verifies that predicted coordinates and Euclidean errors maintain sub-centimeter float precision (mapping `(123.4, 456.7) -> (1.234, 4.567)`), failing if 2-decimal rounded helpers are used. |

> **Disclaimer**: These tests demonstrate software implementation integrity, precision invariance, and geometric correctness under known projective mappings. They do not constitute empirical physical validation.

---

## 7. Scope Boundaries

Phase 8.2 strictly preserves all existing pipeline components and excludes:
- Spatial zone assignment validation (Phase 8.3+)
- Footfall counting and dwell validation
- Tracker fragmentation evaluation
- Hysteresis sensitivity analysis
- Heatmaps or density maps
- Recommendations or layout optimization
- SQLite persistence or Streamlit dashboards
