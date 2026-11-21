# Phase 11.1 & 11.2 — Zone Visit Sequences, Direct Transitions & Descriptive Per-Zone Activity Summaries

**Project**: Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
**Roadmap Phase**: Phase 11 — Zone Transition & Sequence Analytics  
- Slice 11.1: Descriptive Visit Sequences & Direct Transitions  
- Slice 11.2: Descriptive Per-Zone Activity Summaries  
**Status**: COMPLETE / OPERATIONALLY VERIFIED  
**Regression Baseline**: 188/188 tests passing (55 Phase 0–7 baseline + 82 Phase 8 validation + 16 Phase 9 dwell + 12 Phase 10 density + 23 Phase 11 transitions)  
**Verification Runner**: [`scripts/verify_phase_08.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/scripts/verify_phase_08.py) (All 70 component tests pass cleanly)

---

## 1. Executive Summary & Objective

Phase 11 introduces explainable, descriptive summaries of **observed zone visit sequences**, **direct zone-to-zone transitions**, and **per-zone activity summaries** from existing track trajectories.

Operating strictly as a downstream analytics module, Phase 11 consumes pre-evaluated `TrajectoryPoint.zone_id` sequences without modifying upstream detection, tracking, homography, or zone-assignment stages. It maintains strict methodological discipline:
1. **Descriptive Measurement of Observed Labels**: Quantifies observed zone labels, direct transition frequencies, and activity counts. Makes no assumptions and assigns no labels regarding customer intent, engagement, preference, interest, or behavioral classification (e.g., no "passing", "lingering", "browsing", or "wandering" labels).
2. **Pure Analytics Layer**: Consumes ordered observations from `Trajectory` objects with pre-evaluated `zone_id`. Does not rerun Point-in-Polygon (PIP) geometry or modify temporal hysteresis buffers.
3. **Clip-Local Track Identity**: Operates strictly on session-local `track_id` integers. Never merges identities, joins disjoint tracks, or infers person re-identification (Re-ID). Transitions and distinct track counts are evaluated strictly within each clip-local track, never across tracks.
4. **Observed Visit Run Semantics**: Consecutive observations with the same zone label represent a single run in the observed sequence. For single-frame visits, the observed span is 0.0s. No duration threshold is applied to discard short visits.
5. **Strict Direct Transition Semantics**: A directed transition ($A \to B$) is counted **only** when consecutive non-`None` zone runs change directly from one defined zone to another ($A \neq B$). An unassigned or outside observation (`zone_id = None`) immediately breaks the transition chain. Intervening outside observations are **never** bridged ($A \to \text{None} \to B$ yields **zero** direct transitions).
6. **No Boundary Extrapolation**: Transitions are never inferred before the initial observation or after the final observation of a track.
7. **Descriptive Integer Counts**: All activity metrics are non-negative integers. No percentages, normalized transition probabilities, rankings, "popular zone" labels, or causal claims are computed.
8. **Distinguishable Unvisited vs. Missing Zones**: Unvisited configured zones evaluate to explicit 0 counts across all metrics, distinguishing them from unconfigured or absent zones.
9. **No Layout Optimization / Causal Claims**: Does not rank store aisles, score store layouts, or claim sales uplift.

---

## 2. Architecture Integration & Responsibilities

The pipeline responsibilities remain strictly decoupled:

```
[Phase 2] Frame Sampling (5 FPS generator)
   ↓
[Phase 3] YOLO11 Person Detection (ONNX Runtime CPU)
   ↓
[Phase 4] ByteTrack Multi-Object Tracking (Session-local Track IDs)
   ↓
[Phase 5] Trajectory Extraction + Homography (Image space → 2D Floor Coordinates)
   ↓
[Phase 6] Manual Zone Definition + Point-in-Polygon + Hysteresis (Floor Coordinates → zone_id)
   ↓
[Phase 7] Footfall Analytics (Zone entry events, aggregate visit counts)
   ↓
[Phase 9] Refined Dwell Analytics Foundation (Per-track, per-zone dwell episodes & span distributions)
   ↓
[Phase 10] Floor-Plane Trajectory Density (Configured grid accumulation & heatmap visualization)
   ↓
[Phase 11.1] DESCRIPTIVE ZONE VISIT SEQUENCES & DIRECT TRANSITIONS
   ↓
[Phase 11.2] DESCRIPTIVE PER-ZONE ACTIVITY SUMMARIES (Runs, distinct clip-local tracks, incoming/outgoing)
```

### Decoupling Invariants:
- Upstream detection, tracking, homography, and zone evaluation modules remain unchanged.
- `src/analytics/footfall.py` (Phase 7), `src/analytics/dwell.py` (Phase 9), and `src/analytics/density.py` (Phase 10) are unaffected.
- Phase 11 is implemented in `src/analytics/transitions.py` and exported via `src/analytics/__init__.py`.

---

## 3. Data Model & API Specification

### 3.1 `ZoneVisitRun` (Dataclass)
Represents a maximal contiguous sequence of trajectory observations within a single non-`None` zone for a single track:

| Field | Type | Description |
|---|---|---|
| `track_id` | `int` | Clip-local tracker identifier ($\ge 0$). |
| `zone_id` | `str` | Visited zone identifier string. |
| `entry_frame_idx` | `int` | Frame index of initial observation in this run. |
| `exit_frame_idx` | `int` | Frame index of final observation in this run ($\ge \text{entry\_frame\_idx}$). |
| `entry_timestamp_sec` | `float` | Video timestamp in seconds at initial observation ($\ge 0.0$). |
| `exit_timestamp_sec` | `float` | Video timestamp in seconds at final observation ($\ge \text{entry\_timestamp\_sec}$). |
| `observation_count` | `int` | Total sampled observations during this run ($\ge 1$). |
| `duration_sec` *(property)* | `float` | Observed duration span in seconds ($\max(0.0, t_{\text{exit}} - t_{\text{entry}})$). |

Methods:
- `to_dict() -> Dict[str, Any]`: Serializes run to dictionary including computed `duration_sec`.

### 3.2 `ObservedTrackSequence` (Dataclass)
Represents the chronological sequence of observed zone visit runs and direct zone-to-zone transitions for a single track:

| Field | Type | Description |
|---|---|---|
| `track_id` | `int` | Clip-local tracker identifier ($\ge 0$). |
| `runs` | `List[ZoneVisitRun]` | Chronological sequence of visit runs. |
| `zone_sequence` | `List[str]` | Ordered list of zone IDs for each observed run. |
| `direct_transitions` | `List[Tuple[str, str]]` | Chronological list of $(A, B)$ direct transition pairs observed without intervening `None`. |
| `total_runs` *(property)* | `int` | Total number of observed runs. |
| `total_transitions` *(property)* | `int` | Total count of direct transitions. |

Methods:
- `to_dict() -> Dict[str, Any]`: Serializes track sequence and nested runs.

### 3.3 `ZoneActivitySummary` (Dataclass — Phase 11.2)
Descriptive per-zone activity summary derived from observed runs and direct transitions:

| Field | Type | Description |
|---|---|---|
| `zone_id` | `str` | Spatial retail zone identifier string. |
| `observed_run_count` | `int` | Total count of contiguous observed visit runs in this zone ($\ge 0$). |
| `distinct_track_count` | `int` | Count of distinct clip-local tracker IDs with at least one observed run in this zone ($\ge 0$). |
| `direct_incoming_count` | `int` | Count of direct transitions entering this zone directly from any other defined zone without intervening `None` ($\ge 0$). |
| `direct_outgoing_count` | `int` | Count of direct transitions leaving this zone directly to any other defined zone without intervening `None` ($\ge 0$). |

> [!IMPORTANT]
> **Methodological Boundary on Track Identity & Fragmentation**:
> `distinct_track_count` measures the count of distinct **clip-local tracker IDs** assigned by ByteTrack. It does **NOT** represent unique human customers or shopper identities. Tracking fragmentation (e.g. caused by occlusion, illumination changes, or false negatives) causes a single physical customer to be assigned multiple track IDs, biasing distinct track counts upward.

Methods:
- `to_dict() -> Dict[str, Any]`: Serializes summary to dictionary.

### 3.4 `DirectTransitionSummary` (Dataclass)
Aggregated descriptive summary of direct zone transitions and per-zone activity across all evaluated tracks:

| Field | Type | Description |
|---|---|---|
| `total_tracks` | `int` | Total number of evaluated tracks. |
| `total_transitions` | `int` | Total number of direct transitions observed. |
| `known_zone_ids` | `List[str]` | Sorted list of configured or discovered zone identifiers. |
| `transition_counts` | `Dict[Tuple[str, str], int]` | Mapping $(A, B) \mapsto \text{count}$. |
| `transition_matrix` | `Dict[str, Dict[str, int]]` | Square $M \times M$ nested mapping $A \mapsto \{B: \text{count}\}$. |
| `track_sequences` | `List[ObservedTrackSequence]` | Per-track sequence records. |
| `zone_summaries` | `Dict[str, ZoneActivitySummary]` | Per-zone activity summaries for all known and discovered zones. |

Methods:
- `to_dict() -> Dict[str, Any]`: Serializes summary into JSON-compatible dictionary format including `zone_summaries`.

### 3.5 Core Classes & Convenience Functions
```python
class TransitionAnalyzer:
    def __init__(self, known_zone_ids: Optional[Sequence[str]] = None): ...
    def analyze_trajectory(self, trajectory: Trajectory) -> ObservedTrackSequence: ...
    def analyze(self, trajectories: Sequence[Trajectory]) -> DirectTransitionSummary: ...

def extract_zone_visit_runs(trajectory: Trajectory) -> List[ZoneVisitRun]: ...
def extract_track_sequence(trajectory: Trajectory) -> ObservedTrackSequence: ...
def compute_direct_transitions(
    trajectories: Union[Trajectory, Sequence[Trajectory]],
    known_zone_ids: Optional[Sequence[str]] = None,
) -> DirectTransitionSummary: ...
def compute_zone_activity_summaries(
    trajectories_or_summary: Union[Trajectory, Sequence[Trajectory], DirectTransitionSummary],
    known_zone_ids: Optional[Sequence[str]] = None,
) -> Dict[str, ZoneActivitySummary]: ...
```

---

## 4. Count Definitions & Transition Invariants

### 4.1 Count Definitions for Zone $Z$
1. **`observed_run_count`**: Total number of contiguous visit runs observed in zone $Z$ across all tracks. If a single track enters $Z$, leaves to `None`, and re-enters $Z$, it contributes $2$ runs.
2. **`distinct_track_count`**: Cardinality of the set of distinct clip-local `track_id` integers observed in zone $Z$. In the example above, the track ID is the same, so it contributes $1$ distinct track.
3. **`direct_incoming_count`**: Sum of direct transitions from any other defined zone $X$ into $Z$ without intervening `None`: $\sum_{X \neq Z} \text{transition\_matrix}[X][Z]$.
4. **`direct_outgoing_count`**: Sum of direct transitions from $Z$ to any other defined zone $Y$ without intervening `None`: $\sum_{Y \neq Z} \text{transition\_matrix}[Z][Y]$.

### 4.2 Distinguishing Unvisited vs. Missing Zones
- **Configured Unvisited Zone**: If zone $Z$ is present in `known_zone_ids` but no observations fall within it, `Z in summary.zone_summaries` is `True`, and all counts (`observed_run_count`, `distinct_track_count`, `direct_incoming_count`, `direct_outgoing_count`) are explicitly `0`.
- **Unconfigured / Missing Zone**: If zone $W$ was never specified in `known_zone_ids` and was never observed in any trajectory, `W in summary.zone_summaries` is `False`. Callers can therefore distinguish zero activity from absence of data.

### 4.3 Direct Transition Invariant Rules
1. **$A \to A \to A$**: Evaluated as **1 run**, **0 transitions**.
2. **$A \to B$**: Consecutive non-`None` observations in different zones produce **2 runs**, **1 direct transition** $(A, B)$.
3. **$A \to \text{None} \to B$**: Intervening `None` breaks the chain. Produces **2 runs**, **0 direct transitions**. For Zone $A$, `direct_outgoing_count == 0`. For Zone $B$, `direct_incoming_count == 0`.
4. **$\text{None} \to A$**: Trajectory begins in unassigned area and enters zone. Produces **1 run**, **0 transitions**. No entry before first observation is inferred.
5. **$A \to \text{Trajectory End}$**: Trajectory terminates inside zone. Produces **1 run**, **0 transitions**. No exit after last observation is inferred.
6. **$A \to B \to A$**: Direct cycle produces **3 runs**, **2 direct transitions** ($(A, B)$ and $(B, A)$).
7. **Multiple Tracks**: Track 1 visiting $A$ and Track 2 visiting $B$ never produces a transition. Transitions are evaluated strictly within each clip-local track.

---

## 5. Evidence Classification Matrix

| Component | Implemented | Operationally Verified | Quantitatively Validated | Retail-Performance Validated | Empirical Boundary / Limitation |
|---|---|---|---|---|---|
| **Observed Zone Run Extractor** | YES | YES | NO | NO | Measures observed discrete sample sequence; missing intermediate detections are joined into one observed span. |
| **Direct Transition Classifier** | YES | YES | NO | NO | Strictly direct transitions; intervening unassigned observations break the transition chain and are not bridged. |
| **Aggregated Transition Matrix** | YES | YES | NO | NO | Descriptive transition counting only; no shopper intent, choice model, or layout recommendations implied. |
| **Per-Zone Activity Summaries** | YES | YES | NO | NO | Pure integer counts; distinct tracks are clip-local tracker IDs biased by tracker fragmentation; does NOT count unique people. |
| **Multi-Track Isolation** | YES | YES | NO | NO | Strictly clip-local `track_id`; no cross-camera Re-ID or track merging. |
| **Strict Schema Validation** | YES | YES | NO | NO | Rejects non-monotonic timestamps/frames and non-finite values; leaves caller inputs unmutated. |

---

## 6. Verification Suite Inventory

The Phase 11 test suite in [`tests/test_transitions.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/tests/test_transitions.py) contains 23 focused tests:

### Phase 11.1 Tests:
1. `test_empty_trajectories_and_empty_points`: Handles empty collections, trajectories with no points, and trajectories with only `None` points.
2. `test_single_zone_repeated_observations`: Consecutive points in the same zone form 1 run and 0 transitions.
3. `test_direct_transition_between_distinct_zones`: Direct $A \to B$ transition creates 2 runs and 1 transition $(A, B)$.
4. `test_direct_cycle_and_oscillation`: Direct $A \to B \to A$ oscillation produces 3 runs and 2 direct transitions.
5. `test_unassigned_observation_breaks_transition_chain`: $A \to \text{None} \to B$ produces 2 runs and 0 transitions.
6. `test_unassigned_points_at_boundaries_and_reentrant_same_zone`: None at boundaries and $A \to \text{None} \to A$ produces separate runs without transitions.
7. `test_mixed_direct_and_broken_transitions`: Mixed sequence $A \to B \to \text{None} \to C \to D$ correctly records only $(A, B)$ and $(C, D)$.
8. `test_multi_track_isolation_and_aggregation`: Transitions are strictly within each track; multiple tracks aggregate cleanly into total summary.
9. `test_known_zone_ids_config_preserves_zero_count_zones`: Configured known zones retain 0 counts in transition matrix.
10. `test_caller_trajectories_and_points_are_not_mutated`: Caller `Trajectory` objects remain immutable.
11. `test_single_point_run_duration_is_zero`: Single-observation run has `duration_sec == 0.0`.
12. `test_dataclass_and_serialization_round_trip`: `to_dict()` outputs valid, JSON-serializable dictionaries.
13. `test_temporal_and_schema_validation_rejections`: Rejects non-monotonic timestamps/frames, non-finite values, empty zone strings, duplicate track IDs, and malformed inputs.
14. `test_zone_visit_run_direct_instantiation_validation`: Validates constraints in `ZoneVisitRun.__post_init__`.
15. `test_observed_track_sequence_direct_instantiation_validation`: Validates constraints in `ObservedTrackSequence.__post_init__`.

### Phase 11.2 Tests:
16. `test_zone_activity_summary_unvisited_known_zone_distinguishable_from_missing`: Verifies that unvisited known zones evaluate to 0 counts and are distinguishable from missing/unconfigured zones.
17. `test_zone_activity_summary_repeated_runs_same_track`: Re-entering the same zone yields `observed_run_count > 1` but `distinct_track_count == 1`.
18. `test_zone_activity_summary_multiple_clip_local_tracks`: Multiple tracks visiting a zone increment `distinct_track_count`.
19. `test_zone_activity_summary_incoming_and_outgoing_direct_transitions`: Direct cycle $A \to B \to C \to A$ yields exact incoming and outgoing counts per zone.
20. `test_zone_activity_summary_none_breaks_transition_chain`: Intervening `None` observation yields zero incoming and outgoing direct transitions.
21. `test_zone_activity_summary_convenience_function_signatures`: Verifies `compute_zone_activity_summaries` works with single `Trajectory`, `Sequence[Trajectory]`, and `DirectTransitionSummary`.
22. `test_zone_activity_summary_serialization_round_trip`: Verifies JSON-serializability of `ZoneActivitySummary` and nested `DirectTransitionSummary.to_dict()`.
23. `test_zone_activity_summary_validation_rejections`: Rejects negative counts, boolean counts, non-integer types, and invalid zone IDs.
