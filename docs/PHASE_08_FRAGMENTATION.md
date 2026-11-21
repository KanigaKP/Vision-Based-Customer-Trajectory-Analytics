# Phase 8.5 — Track Fragmentation Analysis Specification

**Project**: Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
**Phase**: 8.5 (Track Fragmentation Validation Layer)  
**Status**: COMPLETE / VERIFIED  
**Regression Status**: 112/112 tests passing (55 Phase 0–7 baseline + 12 Phase 8.1 schema tests + 9 Phase 8.2 homography validation tests + 11 Phase 8.3 zone validation tests + 11 Phase 8.4 footfall validation tests + 14 Phase 8.5 fragmentation tests)  

---

## 1. Phase 8.5 Purpose & Scope

In multi-object tracking (MOT), temporary occlusions, detector dropouts, and rapid directional shifts frequently cause continuous visual targets to be split into multiple session-local track identifiers (ID switches or track fragmentation).

The objective of Phase 8.5 is to implement an independent, non-intrusive validation layer that measures **observed track fragmentation under supplied explicit correspondence**.

### Key Architectural Invariants:
1. **Measurement Layer Only**:
   - Phase 8.5 is purely an observational evaluation tool.
   - It does **not** modify ByteTrack tracking algorithms, Kalman filters, association costs, or trajectory extraction.
   - It does **not** attempt to "stitch", merge, or repair fragmented tracks in production.
2. **Explicit Correspondence Requirement**:
   - The analysis requires an explicit mapping provided by the caller:
     $$\text{gt\_id} \to \text{List}[\text{track\_id}]$$
   - It strictly prohibits automated track rematching, spatial proximity heuristics, bounding-box IoU heuristics, or visual Re-ID.
   - Missing or ambiguous correspondence is rejected immediately.
3. **Descriptive Measurement, Not MOT Benchmark Claim**:
   - Reports observed fragmentation and temporal frame coverage on the supplied evaluated sample.
   - Does **not** claim MOTA, MOTP, IDF1, or universal tracking accuracy without full benchmark annotations and protocol compliance.

---

## 2. Accepted Inputs & Pre-Conditions

Implemented in [`src/validation/fragmentation.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/validation/fragmentation.py):

1. **Ground-Truth Observations**:
   - `gt_observations`: `List[GroundTruthPoint]` or [`GroundTruthSequence`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/validation/ground_truth.py#L318-L345).
   - Provides canonical ground-truth identities (`gt_id: str`) and observed frame indices (`frame_idx: int`).
2. **Tracker Observations**:
   - `tracker_observations`: Flexible adapter accepting `List[Trajectory]`, `List[TrajectoryPoint]`, `List[Track]`, or pre-indexed `Dict[int, Set[int]]`.
   - Extracts observed active frame sets per integer `track_id`.
3. **Explicit Correspondence Mapping**:
   - `correspondence`: `Dict[str, List[int]]` mapping each ground-truth identity to one or more tracker IDs.
   - **Validation Rules**:
     - Cannot be empty or `None`.
     - Each `gt_id` must be present in the ground-truth observations.
     - Each `track_id` must be present in the tracker observations.
     - Duplicate tracker IDs in a single person's mapping are rejected with `ValueError`.
     - Ambiguous reuse of a tracker ID across multiple ground-truth persons is rejected with `ValueError`.
     - Every ground-truth identity present in the observations must have an explicit mapping; missing identities raise `ValueError`.

---

## 3. Mathematical Definitions & Metrics

### 3.1 Fragmentation Definition
For each ground-truth person $p$ with explicitly mapped tracker tracks $T(p) = \{t_1, t_2, \dots, t_k\}$:
- **Fragment Count**:
  $$k(p) = |T(p)|$$
- **Fragmented Status**:
  $$\text{is\_fragmented}(p) = \begin{cases} \text{True} & \text{if } k(p) > 1 \\ \text{False} & \text{if } k(p) = 1 \end{cases}$$
- **Fragmentation Rate**:
  $$\text{Fragmentation Rate} = \frac{\sum_{p} [\text{is\_fragmented}(p)]}{N_{\text{gt\_identities}}}$$
- **Mean Fragments Per Identity**:
  $$\mu_{\text{fragments}} = \frac{\sum_{p} k(p)}{N_{\text{gt\_identities}}}$$

### 3.2 Overlap-Safe Frame Coverage
Let $F_{\text{gt}}(p)$ be the set of video frames where person $p$ is present in ground truth.  
Let $F_{\text{tracker}}(t)$ be the set of video frames where tracker track $t$ is active.

The mapped tracker frame coverage is the **union** of frames across all mapped tracks:
$$F_{\text{mapped}}(p) = \bigcup_{t \in T(p)} F_{\text{tracker}}(t)$$

> [!IMPORTANT]
> Using the set union guarantees that overlapping frames between concurrently active or transitioning tracks are **never double-counted**.

- **Covered Frames**:
  $$F_{\text{covered}}(p) = F_{\text{gt}}(p) \cap F_{\text{mapped}}(p)$$
- **Person Coverage Ratio**:
  $$\text{Coverage Ratio}(p) = \frac{|F_{\text{covered}}(p)|}{|F_{\text{gt}}(p)|}$$
- **Overall Coverage Ratio**:
  $$\text{Overall Coverage} = \frac{\sum_{p} |F_{\text{covered}}(p)|}{\sum_{p} |F_{\text{gt}}(p)|}$$

---

## 4. Result Structure & Serialization

### 4.1 `PersonFragmentationRecord`
- `gt_id`: Ground-truth individual identifier.
- `associated_track_ids`: Sorted list of mapped tracker track IDs.
- `fragment_count`: Total mapped tracks.
- `is_fragmented`: Boolean flag.
- `gt_frame_count`: $|F_{\text{gt}}(p)|$.
- `mapped_tracker_frame_count`: $|F_{\text{mapped}}(p)|$.
- `covered_frame_count`: $|F_{\text{covered}}(p)|$.
- `coverage_ratio`: Fraction of GT frames observed by the tracker.

### 4.2 `TrackFragmentationResult`
- `dataset_name`: Identifier string.
- `person_records`: List of `PersonFragmentationRecord` sorted deterministically by `gt_id`.
- `total_gt_identities`: Total evaluated individuals.
- `total_fragmented_identities`: Individuals with fragment count $> 1$.
- `total_unfragmented_identities`: Individuals with fragment count $== 1$.
- `total_observed_fragments`: Sum of all mapped tracks.
- `fragmentation_rate`: Proportion of identities that experienced fragmentation.
- `mean_fragments_per_identity`: Average tracks per person.
- `overall_coverage_ratio`: Aggregate frame coverage across all persons.
- `evidence_status`: `"FRAGMENTATION_EVALUATED"` or `"INSUFFICIENT_EVIDENCE"`.
- `evidence_note`: Explicit methodological caveats and scope description.

---

## 5. Evidence Status Classifications & Methodological Scope

| Status | Trigger Condition | Interpretation |
|---|---|---|
| **`INSUFFICIENT_EVIDENCE`** | Empty ground-truth observations list. | No ground-truth observations supplied. Fragmentation cannot be evaluated. Prevents artificial 0% or 100% metrics. |
| **`FRAGMENTATION_EVALUATED`** | One or more valid ground-truth identities with complete correspondence. | Descriptive fragmentation and frame coverage computed across supplied test data. |

---

## 6. Dataset Evaluation Caveats & Future Application

- **Synthetic Fixtures**:
  - Used in `tests/test_fragmentation.py` to verify counting logic, duplicate rejection, overlap handling, and JSON serialization.
  - Software verification only; **not** physical real-world tracking validation.
- **MOT17**:
  - Benchmark specifically designed for multi-object tracking evaluation.
  - Can be used with this validation layer only when MOT17 ground-truth bounding box IDs are converted to `GroundTruthPoint` representations and paired with explicit tracker correspondence.
- **CAVIAR & MERL Datasets**:
  - CAVIAR contains multi-corridor overhead tracking clips that exhibit fragmentation during hallway occlusions.
  - MERL shopping sequences capture retail aisles but require manual continuous identity annotation before fragmentation can be rigorously quantified.
- **Identity Caveat**:
  - Tracker-level fragmentation illustrates why raw unique track counts overestimate human footfall volume in Phase 7 without Re-ID or external identity association.
