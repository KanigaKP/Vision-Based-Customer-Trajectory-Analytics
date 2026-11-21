# Phase 9.1 & 9.2 — Refined Dwell Analytics Foundation & Descriptive Span Distributions

**Project**: Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
**Roadmap Phase**: Phase 9 — Refined Dwell / Temporal Analytics (Slice 9.1: Foundation; Slice 9.2: Descriptive Distributions)  
**Status**: COMPLETE / OPERATIONALLY VERIFIED  
**Regression Baseline**: 153/153 tests passing (55 Phase 0–7 baseline + 82 Phase 8 validation + 16 Phase 9 dwell tests)  
**Verification Runner**: [`scripts/verify_phase_08.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/scripts/verify_phase_08.py) (All 70 component tests pass cleanly)  

---

## 1. Executive Summary & Objective

Phase 9.1 introduces the foundational data models, validators, and extraction algorithms for **per-track, per-zone dwell episodes and basic temporal summaries**. Building directly on the spatial trajectory representation and pre-evaluated zone assignments established in Phases 5 through 7, this module derives explainable dwell episodes while maintaining strict methodological invariants:

1. **Pure Analytics Layer**: Consumes existing `Trajectory` and `TrajectoryPoint.zone_id` assignments without rerunning detection, tracking, homography, Point-in-Polygon (PIP), or hysteresis filtering.
2. **Clip-Local Track Identity**: Operates strictly on session-local `track_id` integers. Never merges identities, tracks across camera views, or infers biometric/re-identification correspondence.
3. **Observed In-Zone Duration Span**: Durations are calculated strictly from the first and last observed timestamps within an episode ($\Delta t_{\text{span}} = t_{\text{exit}} - t_{\text{entry}}$). This represents an **observed in-zone span** between discrete detections, not an exact or verified measure of continuous physical dwell.
4. **Strict Episode Boundaries & Gap Semantics**: An observation outside defined zones (`zone_id = None`) breaks an active episode immediately and is never bridged. Separate visits to the same zone by the same track are recorded as distinct `DwellEpisode` records. Valid observations with the same track and zone are joined into an episode without arbitrary gap thresholds; the resulting duration is an observed span that may overstate continuous occupancy when intermediate observations are missing.
5. **Rigorous Temporal Validation**: Rejects missing (`None`), non-finite (`NaN`, `Inf`), negative, boolean, or non-monotonic timestamps and frame indices with explicit `ValueError` exceptions; never guesses or fills gaps.

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
[Phase 7] Footfall Analytics (Zone entry events, transitions, aggregate visit counts)
   ↓
[Phase 9.1] REFINED DWELL ANALYTICS FOUNDATION (Per-track, per-zone episodes & temporal summaries)
```

### Decoupling Rules:
- Upstream detection, tracking, homography, and zone evaluation modules are **not** altered.
- `src/analytics/footfall.py` is **not** altered; Phase 7 continues providing aggregate footfall metrics independently.
- Phase 9.1 operates downstream in `src/analytics/dwell.py`, providing fine-grained per-track and per-zone dwell episode breakdowns.

---

## 3. Data Structures & API Specification

### 3.1 `DwellEpisode` (Dataclass)
Represents a contiguous sequence of observations of a single clip-local track within one zone:

| Field | Type | Description |
|---|---|---|
| `track_id` | `int` | Clip-local tracker identifier ($\ge 0$). |
| `zone_id` | `str` | Non-empty retail zone identifier. |
| `entry_frame_idx` | `int` | Frame index of initial observation in episode. |
| `exit_frame_idx` | `int` | Frame index of final observation in episode ($\ge \text{entry\_frame\_idx}$). |
| `entry_timestamp_sec` | `float` | Video timestamp in seconds at initial observation ($\ge 0.0$). |
| `exit_timestamp_sec` | `float` | Video timestamp in seconds at final observation ($\ge \text{entry\_timestamp\_sec}$). |
| `duration_sec` | `float` | Observed in-zone duration span in seconds: $\text{round}(\max(0.0, t_{\text{exit}} - t_{\text{entry}}), 3)$. |
| `point_count` | `int` | Number of sampled observations in episode ($\ge 1$). |

### 3.2 `TrackDwellSummary` (Dataclass)
Per-track summary for a single clip-local track:
- `track_id`: `int`
- `total_dwell_sec`: `float` (sum of episode duration spans across all visited zones)
- `total_episodes`: `int` (total episode count across all visited zones)
- `zone_dwell_sec`: `Dict[str, float]` (total dwell span seconds per zone)
- `zone_episode_count`: `Dict[str, int]` (episode count per zone)
- `episodes`: `List[DwellEpisode]` (chronological list of episodes)

### 3.3 `ZoneSpanDistribution` (Dataclass — Phase 9.2)
Descriptive distribution of observed in-zone duration spans for a spatial zone:

| Field | Type | Description |
|---|---|---|
| `episode_count` | `int` | Total count of observed episodes in this zone ($\ge 0$). |
| `min_span_sec` | `float` | Minimum observed in-zone duration span in seconds ($0.0$ if empty). |
| `median_span_sec` | `float` | Deterministic median observed in-zone duration span in seconds ($0.0$ if empty). |
| `mean_span_sec` | `float` | Arithmetic mean observed in-zone duration span in seconds ($0.0$ if empty). |
| `max_span_sec` | `float` | Maximum observed in-zone duration span in seconds ($0.0$ if empty). |

### 3.4 `ZoneDwellSummary` (Dataclass — Updated Phase 9.2)
Per-zone summary across all tracks:
- `zone_id`: `str`
- `total_dwell_sec`: `float` (sum of all episode duration spans in this zone)
- `total_episodes`: `int` (number of episodes in this zone; alias property: `episode_count`)
- `unique_tracks`: `int` (count of distinct `track_id`s entering this zone)
- `mean_dwell_sec`: `float` (arithmetic mean observed span, $\text{total\_dwell\_sec} / \text{total\_episodes}$, or $0.0$)
- `median_dwell_sec`: `float` (deterministic median observed span, or $0.0$)
- `min_dwell_sec`: `float` (minimum episode duration span in seconds, or $0.0$)
- `max_dwell_sec`: `float` (maximum episode duration span in seconds, or $0.0$)
- `distribution`: `ZoneSpanDistribution` (property exposing structured distribution dataclass)
- `episodes`: `List[DwellEpisode]`

### 3.5 `DwellSummary` (Dataclass)
Global aggregation:
- `total_episodes`: `int`
- `total_dwell_sec`: `float`
- `by_track`: `Dict[int, TrackDwellSummary]`
- `by_zone`: `Dict[str, ZoneDwellSummary]`
- `episodes`: `List[DwellEpisode]`

---

## 4. Dwell Episode & Duration Semantics

### 4.1 Observed In-Zone Duration Span vs. Continuous Physical Dwell
For an episode consisting of sampled points $P = [p_0, p_1, \dots, p_{k-1}]$ where all points have `zone_id` $Z$:
$$\Delta t_{\text{span}} = t_{k-1} - t_0$$
- If $k = 1$ (single-observation episode): $\Delta t_{\text{span}} = t_0 - t_0 = 0.0\text{ s}$.
- If $k > 1$: $\Delta t_{\text{span}} = \text{round}(t_{k-1} - t_0, 3)$.

> [!IMPORTANT]
> **Observed Span vs. Continuous Physical Dwell**:
> The calculated duration $\Delta t_{\text{span}} = t_{\text{exit}} - t_{\text{entry}}$ is an **observed in-zone duration span**. It represents the elapsed time between the first and last recorded detections of that track within the zone polygon.
> It must **not** be presented as an exact measure of continuous physical dwell without supporting continuity assumptions. Without continuous ground truth or sub-frame sensor verification, whether an individual remained continuously present or stationary throughout that interval is unobserved between discrete frames.

### 4.2 Handling of Exterior Space (`None`) and Transitions
- When an observation has `zone_id = None`, any active episode is closed at the preceding observation.
- Exterior observations are never assigned a dwell episode.
- If a track moves $Z_A \to \text{None} \to Z_A$, two distinct episodes are recorded for $Z_A$. The unassigned temporal gap is completely excluded from dwell time.
- When transitioning directly $Z_A \to Z_B$, the episode for $Z_A$ closes at $t_{\text{prev}}$ and a new episode for $Z_B$ starts at $t_{\text{curr}}$.

### 4.3 Descriptive Distribution Statistics of Observed Spans (Phase 9.2)
For each zone $Z$, given its set of extracted episodes $E_Z = [e_0, e_1, \dots, e_{K-1}]$ with observed duration spans $S = [s_0, s_1, \dots, s_{K-1}]$ where $s_i = e_i.\text{duration\_sec}$, the following descriptive statistics are derived deterministically:

1. **Episode Count ($K$)**: Total number of observed episodes in zone $Z$.
2. **Minimum Observed Span ($\min$)**:
   $$\text{min\_span} = \begin{cases} 0.0\text{ s} & \text{if } K = 0 \\ \text{round}(\min(S), 3) & \text{if } K > 0 \end{cases}$$
3. **Deterministic Median Observed Span ($\text{med}$)**:
   Let $S_{\text{sorted}} = \text{sort}(S)$.
   - If $K = 0$: $\text{median\_span} = 0.0\text{ s}$.
   - If $K$ is odd ($K = 2m + 1$): $\text{median\_span} = \text{round}(S_{\text{sorted}}[m], 3)$.
   - If $K$ is even ($K = 2m$): $\text{median\_span} = \text{round}\left(\frac{S_{\text{sorted}}[m-1] + S_{\text{sorted}}[m]}{2.0}, 3\right)$.
   *Note*: The median provides a robust non-parametric summary of central tendency alongside the arithmetic mean. No assumption or claim is made regarding the specific distribution shape, skewness, or parametric characteristics of empirical customer trajectory data.
4. **Arithmetic Mean Observed Span ($\mu$)**:
   $$\text{mean\_span} = \begin{cases} 0.0\text{ s} & \text{if } K = 0 \\ \text{round}\left(\frac{1}{K}\sum_{i=0}^{K-1} s_i, 3\right) & \text{if } K > 0 \end{cases}$$
5. **Maximum Observed Span ($\max$)**:
   $$\text{max\_span} = \begin{cases} 0.0\text{ s} & \text{if } K = 0 \\ \text{round}(\max(S), 3) & \text{if } K > 0 \end{cases}$$

**Empty-Group Conventions vs. Observed Zero-Duration Episodes**:
- **Unvisited Zones ($K=0$)**: Evaluates to `episode_count = 0` (and `total_episodes = 0`), with all span summary statistics (`min_span_sec`, `median_span_sec`, `mean_span_sec`, `max_span_sec`, `total_dwell_sec`) set to $0.0\text{ s}$ by convention. The condition `episode_count = 0` unambiguously signals the complete absence of observed episodes in that zone, and must not be misinterpreted as an observed stay of zero duration.
- **Observed Single-Point Visits**: When a track is observed inside a zone for exactly one frame, an episode is recorded with $t_{\text{entry}} = t_{\text{exit}}$, resulting in an observed span of $0.0\text{ s}$. In this case, `episode_count \ge 1`, and the $0.0\text{ s}$ span is an actual observed data point included in count, min, median, and mean calculations.


---

## 5. Sampling Limitations, Gap Uncertainty & Hysteresis Semantics

### 5.1 Sampling Quantization Mathematics
Consider an individual whose true physical visit begins at continuous time $t_{\text{phys\_entry}}$ and ends at continuous time $t_{\text{phys\_exit}}$, with true physical duration:
$$D_{\text{phys}} = t_{\text{phys\_exit}} - t_{\text{phys\_entry}}$$

**Idealized Sampling Bounds (Explicit Assumptions)**:
Under the strict assumptions that:
1. Video is sampled uniformly at period $\Delta t_s = 1 / \text{FPS}$ without dropped or missing frames,
2. The individual remains continuously inside the zone throughout the stay,
3. The stay is bordered immediately by an observed out-of-zone detection at $t_{-1} = t_0 - \Delta t_s$ and an observed out-of-zone detection at $t_k = t_{k-1} + \Delta t_s$:

Then the physical entry and exit times satisfy:
$$t_0 - \Delta t_s < t_{\text{phys\_entry}} \le t_0$$
$$t_{k-1} \le t_{\text{phys\_exit}} < t_{k-1} + \Delta t_s$$

Subtracting these inequalities yields:
$$t_{k-1} - t_0 \le D_{\text{phys}} < (t_{k-1} - t_0) + 2\Delta t_s$$
$$\Delta t_{\text{span}} \le D_{\text{phys}} < \Delta t_{\text{span}} + \frac{2}{\text{FPS}}$$

Under these idealized assumptions:
- The observed span $\Delta t_{\text{span}} = t_{k-1} - t_0$ is a **lower bound** on continuous physical dwell, underestimating it by up to $2\Delta t_s = 2/\text{FPS}$.
- For a single observation ($k=1$), $\Delta t_{\text{span}} = 0.0\text{ s}$, while the physical stay lies in $(0, 2\Delta t_s)$.

**Real-World Limitation (Unbounded Without Assumptions)**:
In actual computer vision pipelines, detections may be missed, tracks may coast unobserved across multiple frames, camera framerates may fluctuate, and video boundaries truncate tracks. Therefore, without empirical continuous ground truth or verified continuity assumptions, **the actual continuous visit duration cannot be bounded from the current discrete evidence alone**.

### 5.2 Widely Separated Observations & Missing Evidence
- **Implementation Treatment**: Observations belonging to the same track and zone with no intervening out-of-zone (`None`) or different-zone observations are grouped into a single episode.
- **No Arbitrary Gap Threshold**: The module intentionally does **not** introduce an arbitrary time-gap threshold (e.g., splitting episodes if $\Delta t > 2\text{s}$) because introducing uncalibrated heuristic thresholds creates arbitrary bias.
- **Overstatement Risk**: The computed `duration_sec` represents the observed span between the first and last observation ($t_{\text{last}} - t_{\text{first}}$). If intermediate observations were missed (due to detector dropouts, long occlusions, or intermittent track coasting), this span **may overstate continuous occupancy**.
- **Multiple Sources of Gap Uncertainty**: A track-ID change is **not** the only source of fragmentation or gap uncertainty. Detector misses, partial track gaps within the same track ID, camera framerate drops, and variable sampling intervals also produce substantial observation gaps within a single continuous track.

### 5.3 Phase 6 Hysteresis Semantics ($N=3$)
In Phase 6 (`ZoneEvaluator` in [`src/analytics/zone_evaluator.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/zone_evaluator.py)), temporal hysteresis operates with the following boundary-specific rules:
1. **Initial In-Zone Assignment (No Entry Delay)**: An initial in-zone assignment has **no entry confirmation delay**. When a track is first observed (`idx == 0`), `current_zone` immediately adopts the raw Point-in-Polygon result on that initial frame (`current_zone = raw_zone`, `candidate_count = 1`).
2. **Later Confirmed Transitions (Entry Latency)**: The $N-1$ frame entry confirmation latency applies **only** when zone entry is produced by a confirmed transition after the track's initial point (`idx > 0`). In this case, transitioning from an established zone state to a new candidate zone requires $N$ consecutive candidate frames, producing $N-1$ frames of confirmation latency after the first candidate frame. Not every visit lasting at least $N$ frames has delayed entry.
3. **Exit Confirmation Latency**: Exit latency depends on the corresponding confirmed transition out. Exiting a zone (whether transitioning into exterior space `None` or into an adjacent zone) similarly requires $N$ consecutive candidate frames in the new candidate state, producing an $N-1$ frame confirmation latency after the first out-of-zone frame.
4. **Conditions for Temporal Shift**: The concept of a visit being "temporally shifted" applies **only when both relevant boundaries are confirmed transitions under the stated conditions** (i.e. entry occurred via a later confirmed transition into the zone, and exit occurred via a confirmed transition out). If a visit begins at track creation (`idx == 0`), entry is immediate while exit is delayed, extending the observed span; if tracking terminates while inside the zone without an observed exit transition, no exit confirmation latency occurs.
5. **Sub-$N$ Stay Suppression**: Any transient stay lasting fewer than $N$ consecutive frames ($< N$ frames) after track initiation is never confirmed and is completely filtered out.
6. **Sub-$N$ Chatter Suppression**: Any transient excursion outside the zone lasting fewer than $N$ frames is suppressed, preserving the active zone assignment across brief boundary oscillations.

---

## 6. Phase 8 Test Count Reconciliation

When verifying the repository, two different test counts are reported for Phase 8:

| Scope | Test Count | Files Included | Purpose |
|---|---|---|---|
| **Phase 8 Verification Runner** ([`scripts/verify_phase_08.py`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/scripts/verify_phase_08.py)) | **70 tests** | `test_ground_truth.py` (12)<br>`test_homography_validation.py` (9)<br>`test_zone_validation.py` (11)<br>`test_footfall_validation.py` (11)<br>`test_fragmentation.py` (14)<br>`test_hysteresis_validation.py` (13) | Directly executes the 6 Phase 8 component test suites (Phases 8.1 through 8.6). Intentionally omits its own meta-test file to avoid circular execution. |
| **Full Regression Suite** (`pytest`) | **82 tests** | All 70 component tests above<br>+ `test_phase_08_verification.py` (12) | Standard `pytest` discovery collects all test files under `tests/`, including the 12 unit tests that verify the runner itself (manifest ordering, CLI options, returncode mapping, dry-run mode). |

Both counts are exact and correct; they simply represent different scopes:
$$\text{Full Phase 8 Suite (82)} = \text{Runner Component Tests (70)} + \text{Runner Self-Tests (12)}$$

Adding the 55 baseline tests (Phases 0–7) and 16 Phase 9 dwell tests yields the full regression baseline:
$$\text{Total Regression Baseline} = 55 \text{ (Phases 0–7)} + 82 \text{ (Phase 8)} + 16 \text{ (Phase 9)} = \mathbf{153\text{ tests passed}}$$

---

## 7. Evidence Classification Matrix

| Category | Status | Details |
|---|---|---|
| **IMPLEMENTED** | **YES** | Data models ([`DwellEpisode`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/dwell.py#L112), [`TrackDwellSummary`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/dwell.py#L149), [`ZoneDwellSummary`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/dwell.py#L179), [`ZoneSpanDistribution`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/dwell.py#L218), [`DwellSummary`](file:///c:/Users/sarca/OneDrive/Documents/projects/Vision-Based-Customer-Trajectory-Analytics/src/analytics/dwell.py#L254)), validators, distribution calculators, and test suites are fully implemented. |
| **OPERATIONALLY VERIFIED** | **YES** | 16 focused unit tests pass; all 82 Phase 8 tests pass; full 153-test regression suite passes. |
| **QUANTITATIVELY VALIDATED** | **NO** | Tested exclusively against synthetic trajectory fixtures. No surveyor-calibrated stopwatch dwell ground truth has been benchmarked. |
| **RETAIL-PERFORMANCE VALIDATED** | **NO** | Evaluated on synthetic tests only; no empirical validation against real retail store customer visits. |

> [!IMPORTANT]
> **Operational Verification vs. Empirical Validation**:
> Unit tests prove that algorithms, state machines, temporal validation rules, and aggregation formulas execute predictably and reject corrupted inputs. Passing unit tests does **not** constitute empirical validation of retail dwell time accuracy.

---

## 8. Negative Scope & Future Roadmap Alignment

In compliance with the project roadmap:
- **Phase 9.1**: Implemented foundation (per-track/per-zone episodes, summaries, strict validation).
- **Phase 9.2**: Implemented descriptive distribution statistics of observed in-zone duration spans (count, min, median, mean, max).
- **Phase 9.3+ (Future)**: Threshold-based classification (e.g., passing vs. lingering) remains deferred.
- **Phase 10 (Future)**: Trajectory Density / Heatmaps *(Unimplemented)*.
- **Phase 11 (Future)**: Zone Behavior & Interaction Classification *(Unimplemented)*.
- **Phase 12 (Future)**: Retail Layout Optimization Recommendations *(Unimplemented)*.
- **Out of Scope**: No Re-ID, no multi-camera stitching, no live CCTV, no POS integration, no sales conversion claims.
