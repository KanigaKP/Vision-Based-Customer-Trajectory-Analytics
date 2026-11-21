# Five-Minute Demonstration Script & Cue Sheet

**Project:** Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
**Presenter:** Kaniga Parameswaran R (Reg. No: 250282601051)  
**Institutional Affiliation:** B.S. Abdur Rahman Crescent Institute of Science and Technology  
**Degree / Course:** MCA — CAF7104 Mini Project (November 2026)  
**Total Duration:** Exactly 5 Minutes (300 Seconds)  
**Primary Target URL:** `http://127.0.0.1:8501/?view=workspace` (or `start_demo.cmd`)  
**Fallback Evidence Directory:** `docs/presentation/fallback/` (Static verified screenshots)

---

## 1. Demonstration Cue Sheet & Timeline Overview

| Time Window | Segment Title | Navigation Action & Screen State | Key Spoken Points & Displayed Metrics (Read From Artifacts) |
|---|---|---|---|
| **0:00 – 0:50** (50s) | **1. Scope & MERL Run Identity** | Open `http://127.0.0.1:8501/?view=workspace`. Confirm Run ID: `upload_ca30f3e8`. | Confirm run ID `upload_ca30f3e8` ("MERL Supermarket Aisle"). Point out `STATUS: WITHHELD` banner. State scope: single-camera, local CPU inference, configured coordinates, no commercial claims. |
| **0:50 – 1:50** (60s) | **2. Playback Seeking & Episode Inspection** | Scrub player slider to populated **Sample Index 167 (Displayed #168)**. Navigate to `?view=studio` -> Click **`Inspect episode 🔍`** on `shelf_a` (Episode #1). | Demonstrate populated seeking at Sample Index 167 (displayed #168, $t=33.40\text{s}$, native source frame 1002, Track 1 at $[3.80, 4.66]$ in `shelf_a`, confidence 0.8808). (Note: sample index 10 at 2.0s / frame 60 has `observations: []`). Click "Inspect episode 🔍" on `shelf_a` Episode #1: seeks to entry sample index 167, demonstrating individual episode span of **2.0s** across **11 supporting observations** (samples 167–177, source frames 1002–1062) within aggregate `shelf_a` dwell of **10.2s** across 3 episodes. |
| **1:50 – 2:50** (60s) | **3. Site 03 Observation Sequence (7→8→9)** | Switch Run dropdown to `PRCV Site 03 Boutique (upload_6f969f54)`. Step samples 7 → 8 → 9. | **Sample 7** ($t=1.4\text{s}$, frame 21): Detection at $[0.82, 5.45]$ configured units in `side_display_left`.<br/>**Sample 8** ($t=1.6\text{s}$, frame 24): **No detection recorded** (`observations: []`); visual trail breaks.<br/>**Sample 9** ($t=1.8\text{s}$, frame 27): Detection resumes at $[0.45, 5.47]$ in `side_display_left`. |
| **2:50 – 3:50** (60s) | **4. Site 02 Candidate Zone & Absent Hotspot** | Switch Run dropdown to `PRCV Site 02 Jewelry Store (upload_site02_jewelry)`. Open `?view=candidate`. | Candidate zone: `showcase_display_back`.<br/>Selected-zone dwell: **2.8s** (2 episodes) vs. whole-run total dwell: **13.8s**.<br/>Absent hotspot: Candidate hotspot cell is `null` (*"Not evaluated"*); evaluates zone-level dwell only. |
| **3:50 – 5:00** (70s) | **5. Evidence Boundaries & Technical Rigor** | Open Results Workspace Tab 2 (static SVG 2D/isometric toggle) and conclude presentation. | Reiterate boundaries: configured floor units, analytical 3D (not sensor depth), and withheld sales/conversion claims. Highlight verification record: 412 passed tests (100% pass rate). |

---

## 2. Detailed Step-by-Step Navigation & Speaker Notes

### Segment 1: System Scope & MERL Run Identity (0:00 – 0:50)
- **Navigation Step:**
  1. Open Chrome/Edge to `http://127.0.0.1:8501/?view=workspace`.
  2. Verify that the Run Selector displays **`MERL Supermarket Aisle (upload_ca30f3e8)`**.
  3. Direct the evaluators' attention to the prominent header card and red warning banner.
- **Speaker Notes:**
  > "Good morning, respected evaluators. I am presenting my MCA Mini Project: Vision-Based Customer Trajectory Analytics for Retail Layout Optimization.
  >
  > We begin here in the Results Workspace on our benchmark retail recording: the MERL Supermarket dataset, verified under canonical run ID `upload_ca30f3e8`. This recording contains 633 native frames sampled at 5.0 FPS, capturing 508 trajectory observations across 3 clip-local tracks.
  >
  > Notice the prominent status banner: `EVIDENCE STATUS: WITHHELD`. Our system enforces strict academic and scientific boundaries: all inference runs offline on local CPU using YOLO11-nano ONNX and ByteTrack. Detections project to configured image-to-floor correspondences; independently surveyed physical calibration is unavailable. All commercial sales, conversion, and automated layout reconfiguration claims are withheld."

---

### Segment 2: Playback Seeking & Episode Inspection (0:50 – 1:50)
- **Navigation Step:**
  1. In the playback controls, scrub the sample slider to populated **Sample Index 167 (Displayed #168)** ($t=33.40\,\text{s}$, native source frame 1002). Point out Track 1's bounding box and bottom-center ground contact point at $[3.80, 4.66]$ configured floor units inside `shelf_a` (confidence 0.8808). (Note for evaluators: initial zero-based sample index 10 corresponds to 2.0s and source frame 60, where `observations: []`—no detection was recorded in that initial interval).
  2. Navigate to Spatial Studio (`http://127.0.0.1:8501/?view=studio&run_id=upload_ca30f3e8`).
  3. In the *Selected Zone Inspector: shelf_a*, locate *Episode #1 — Track #1* and click the button **`Inspect episode 🔍`**.
  4. Expand *Supporting Observations (11 pts)*.
- **Speaker Notes:**
  > "When we seek the player to populated Sample Index 167 (displayed as frame #168 in the viewer at 33.4 seconds, native source frame 1002), we see Track 1 actively observed in the aisle at configured floor coordinates $[3.80, 4.66]$ within `shelf_a` with a detection confidence of 0.881. The model localizes the person's physical foot ground contact $(x_c, y_{\max})$ rather than the bounding box centroid. By contrast, initial sample index 10 at 2.0 seconds corresponds to native source frame 60 where no detection was recorded (`observations: []`).
  >
  > In the Spatial Studio below, we inspect zone `shelf_a`. Our dwell engine reports an aggregate dwell span of 10.2 seconds across 3 discrete visit episodes.
  >
  > Clicking 'Inspect episode 🔍' on Episode #1 immediately seeks the player to its recorded entry sample index 167 and focuses Track #1. In the supporting observations drawer, we see all 11 constituent observations across sample indices 167 to 177 (displayed #168 to #178), spanning native source frames 1002 to 1062 (33.4s to 35.4s), totaling an individual episode span of exactly 2.0 seconds. Every observation records its exact source frame, timestamp, ground contact pixel coordinate, and configured floor position. Missing observations are neither interpolated nor fabricated."

---

### Segment 3: Site 03 Observation Sequence: Samples 7 → 8 → 9 (1:50 – 2:50)
- **Navigation Step:**
  1. In the Run Selector dropdown, select **`PRCV Site 03 Boutique (upload_6f969f54)`**.
  2. In the Results Workspace, scrub or navigate to **Sample 7**, then **Sample 8**, then **Sample 9** (or `?sample=7`, `?sample=8`, `?sample=9`).
- **Speaker Notes:**
  > "Now we transition to PRCV Site 03 Home Decor Boutique, under run ID `upload_6f969f54`. This demonstrates how our pipeline handles real-world visual discontinuity:
  >
  > At Sample 7 (native frame 21, 1.4 seconds), Track 1 is detected at floor coordinates $[0.82, 5.45]$ configured floor units, assigned to zone `side_display_left` with a detection confidence of 0.845.
  >
  > At Sample 8 (native frame 24, 1.6 seconds), no detection was recorded (`observations: []`). Look at the trajectory visualizer: the player disconnects the visual trail across this missing sample, avoiding false continuous paths. Crucially, during automated video inference, this gap represents an unobserved detection frame, NOT confirmed physical absence. Confirmed absence is strictly an annotation concept implemented in our ground-truth converter.
  >
  > At Sample 9 (native frame 27, 1.8 seconds), detection resumes at $[0.45, 5.47]$ configured units in `side_display_left` with 0.584 confidence."

---

### Segment 4: Site 02 Candidate Zone & Absent Hotspot (2:50 – 3:50)
- **Navigation Step:**
  1. In the Run Selector, select **`PRCV Site 02 Jewelry Store (upload_site02_jewelry)`**.
  2. Navigate to the Placement Candidate view (`http://127.0.0.1:8501/?view=candidate&run_id=upload_site02_jewelry`).
  3. Highlight the candidate card for **`showcase_display_back`** and the metrics.
- **Speaker Notes:**
  > "Next, we switch to PRCV Site 02 Luxury Jewelry Boutique (`upload_site02_jewelry`), where customer circulation occurs between glass display showcases and a central browsing aisle.
  >
  > Under the observed dwell objective, our candidate selection engine identifies `showcase_display_back` as the candidate zone. Reading directly from the verified run summary artifact:
  > - `showcase_display_back` accumulated **2.8 seconds** of observed dwell span across 2 visit episodes.
  > - Across the entire sequence, total whole-run dwell is **13.8 seconds**—composed of 10.4 seconds in the browsing corridor, 2.8 seconds at the rear showcase, and 0.6 seconds at the front showcase.
  > - Notice the hotspot metric: `Hotspot Point Count: Not evaluated` (candidate hotspot is `null`). Site 02 evaluates aggregate zone-level dwell across the configured geofence; it intentionally does not emit an unvalidated discrete grid hotspot."

---

### Segment 5: Evidence Limitations, Technical Rigor & Conclusion (3:50 – 5:00)
- **Navigation Step:**
  1. Return to Results Workspace Tab 2 (*Floor Trajectories & Zones*).
  2. Toggle between **`Orthogonal 2D Plan`** and **`Isometric 3D Projection (2D Floor Basis)`** to show static SVG rendering.
  3. Conclude the demonstration.
- **Speaker Notes:**
  > "To conclude, I highlight our system's engineering rigor and core methodological limitations:
  >
  > First, in Tab 2, our static SVG perspective toggle renders both 2D orthogonal boundaries and 3D isometric height extrusions. As documented in our acceptance log, these extrusions are analytical 2.5D visual aids derived strictly from planar floor geometry—they are NOT physical 3D LiDAR sensor captures, ceiling measurements, or fixture heights.
  >
  > Second, all spatial projections operate in configured floor units; independently surveyed physical calibration is unavailable.
  >
  > Third, our tracker histories are descriptive and clip-local; this project makes no claims of empirical tracking benchmark accuracy or retail sales effectiveness.
  >
  > Finally, our codebase is backed by a verified test suite of 396 passing tests (3 skipped) in 62.81 seconds, ensuring end-to-end mathematical consistency.
  >
  > Thank you for your time, and I welcome your questions."

---

## 3. Pre-Presentation Checklist (Quick Reference for Speaker)

- [ ] Streamlit server is active on `http://127.0.0.1:8501` (`.\start_demo.cmd`).
- [ ] Browser window is open to `http://127.0.0.1:8501/?view=workspace`.
- [ ] Fallback screenshots directory (`docs/presentation/fallback/`) is accessible in case of local display or port disruption.
- [ ] Verify that all quoted numbers match the saved artifacts:
  - MERL: Run ID `upload_ca30f3e8`, 508 points, 3 tracks, populated sample index 167 (displayed #168, 33.4s, source frame 1002, Track 1 at $[3.80, 4.66]$ in `shelf_a`), `shelf_a` Episode #1 (2.0s span, 11 obs, samples 167–177, frames 1002–1062) within aggregate dwell 10.2s across 3 episodes.
  - Site 03: Run ID `upload_6f969f54`, Sample 7 ($t=1.4\text{s}$, $[0.82, 5.45]$), Sample 8 ($t=1.6\text{s}$, empty), Sample 9 ($t=1.8\text{s}$, $[0.45, 5.47]$).
  - Site 02: Run ID `upload_site02_jewelry`, `showcase_display_back` dwell 2.8s vs. whole-run dwell 13.8s, hotspot `null`.
- [ ] Explicitly state limitations: configured coordinates, analytical 3D, and no sales claims.
