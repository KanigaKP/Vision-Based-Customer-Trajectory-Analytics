# Staff Demonstration Guide: Vision-Based Customer Trajectory Analytics

**Project**: Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
**Architecture**: Direction A Streamlit Web Application (Local, Offline, Evidence-Gated)  
**Status**: Software Prototype Complete & Operationally Verified (396 passed, 3 skipped across 29 test suites in 62.81s)  
**Target Duration**: 5 Minutes  
**Target Audience**: Technical Leadership, Engineering Staff, and Retail Product Stakeholders  

---

## 1. Quickstart: Local Launch

The application runs entirely offline. All model inferences (YOLO11-n ONNX via ONNX Runtime CPU), ByteTrack tracking, homography projections, and spatial analytics execute locally without external API or cloud calls.

### Clean Installation & Setup
To install dependencies from a clean state:
```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe scripts/check_prerequisites.py
.\start_demo.cmd
```
> [!NOTE]
> `start_demo.cmd` requires a release-local virtual environment at `.\venv\Scripts\python.exe`. Clean-machine installation from an empty OS was not verified; verification commands were executed using the local test environment.

### Launch Command
Execute from the repository root:
```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.headless true --server.port 8501
```
- **Local URL**: `http://localhost:8501`
- **Port Verification** (if already running in background):
  ```powershell
  Test-NetConnection -ComputerName localhost -Port 8501
  ```

---

## 2. Opening the Three Curated Retail Examples

The workspace includes three authoritative retail recordings. They can be opened either via the UI or directly via deep links:

| Retail Dataset | UI Navigation | Direct URL |
| :--- | :--- | :--- |
| **MERL Supermarket** | Overview → Available Runs → Click **Open Run** on MERL card | `http://localhost:8501/?view=workspace&run_id=uploaded_runs/upload_ca30f3e8/run_summary.json` |
| **PRCV Site 03 Boutique** | Results Workspace → **Switch Run** dropdown → Select Site 03 | `http://localhost:8501/?view=workspace&run_id=uploaded_runs/upload_38d7afe1/run_summary.json` |
| **PRCV Site 02 Luxury Jewelry** | Results Workspace → **Switch Run** dropdown → Select Site 02 | `http://localhost:8501/?view=workspace&run_id=uploaded_runs/upload_site02_jewelry/run_summary.json` |

*Note: The **Switch Run** dropdown at the top of Results Workspace, Placement Candidate, and Evidence Details allows switching between these runs at any point.*

---

## 3. Five-Minute Demonstration Walkthrough Script

### Minute 1: Overview & Dynamic Workspace Scope
1. Navigate to **Overview** (`http://localhost:8501/?view=overview`).
2. Point out the six clean primary destinations in the left sidebar; highlight that engineering/legacy tools are tucked inside `🛠️ Advanced & Legacy Views`.
3. Review the **Local Workspace Summary**:
   - Dynamic aggregation across discovered runs: 13 analysis runs, 7 unique source video fingerprints, 10 retail runs.
   - Distinct frame counts: 2,584 sampled frames vs. 24,570 source footage frames.
4. Review the **Sample Library Curated Statistics**:
   - 3 curated retail baselines totaling 840 sampled frames (MERL 633, Site 03 134, Site 02 73) and 12 clip-local tracks.
5. Emphasize the **Evidence Status: WITHHELD** banner: all commercial placement recommendations are gated.

### Minute 2: MERL Supermarket — Tracking, Trajectories & Hotspot
1. Click **Open Run** on the MERL card or navigate to `?view=workspace&run_id=uploaded_runs/upload_ca30f3e8/run_summary.json`.
2. **Tab 1 (Video & Tracking Frame)**:
   - Click **Play** or use **Next Sample** to step through frames.
   - Show how the perspective video bounding box and the orthogonal floor map position move in lockstep.
   - Point out the **Sampled Observation Inspector**: shows clip-local track ID, assigned zone (`shelf_a` or `main_aisle`), and unverified floor coordinates.
3. **Tab 4 (Observed Dwell Spans)**:
   - Show **Total Dwell Span**: **17.2 s** across the whole recording (5 visit episodes).
   - Show per-zone breakdown: `shelf_a` = **10.20 s** (3 episodes), `main_aisle` = **7.00 s** (2 episodes).
4. Navigate to **Placement Candidate** (`?view=candidate`):
   - Selected candidate: **Shelf A (Snacks & Beverages)** (`shelf_a`) based strictly on longest observed dwell span (10.2 s vs. 0.0 s in `shelf_b`).
   - Lineage: 53 in-zone points, 3 episodes, clip-local tracks #1 and #3.
   - Hotspot Cell: Point out the highlighted **0.5-unit grid hotspot cell**: bounds **X [2.5, 3.0]**, **Y [4.5, 5.0]** with **10 sampled points**.
   - Correspondence: Golden overlay on camera frame matches gold dashed box `★ [SELECTED CANDIDATE]` on the floor map.

### Minute 3: Site 02 Luxury Jewelry — Two-Candidate Comparison & Sensitivity
1. Switch run to **PRCV Site 02 Luxury Jewelry** via the top dropdown (`?view=workspace&run_id=uploaded_runs/upload_site02_jewelry/run_summary.json`).
2. **Tab 4 (Observed Dwell Spans)** & **Tab 5 (Direct Transitions)**:
   - Total observed dwell: **13.8 s** across 4 episodes (`browsing_aisle` 10.4s, `showcase_display_back` 2.8s, `showcase_display_front` 0.6s).
   - Direct Transitions: **3 transitions** across 3 active directed edges (`showcase_display_back -> browsing_aisle`: 1, `browsing_aisle -> showcase_display_front`: 1, `showcase_display_front -> showcase_display_back`: 1).
3. Navigate to **Placement Candidate** (`?view=candidate`):
   - Candidate zone: **Rear Jewelry Showcase** (`showcase_display_back`) with **2.80 s** dwell span.
   - Runner-Up margin: **+2.20 s (+366.7%)** lead over Front Jewelry Showcase (`showcase_display_front`, 0.60 s).
   - Hotspot neutrality: Hotspot density displays **"Not evaluated"** with caption stating candidate was evaluated at the zone aggregate level (no fictitious hotspot cell is rendered).
4. **Comparison Table & Multi-Objective Sensitivity**:
   - Scroll to **Comparison with Observed Alternatives**: both candidate shelf zones were observed in this clip.
   - Toggle the **Objective Sensitivity Comparison** radio buttons (`Observed Dwell Span`, `Sampled Points`, `Visit Episodes`).
   - Highlight **Descriptive Alignment**: all 3 supported objectives independently select `showcase_display_back` (2.80s dwell, 16 points, 2 episodes).

### Minute 4: Upload & Camera Profile Association Workflow
1. Navigate to **New Analysis** (`http://localhost:8501/?view=new`).
2. **Select Recorded Footage**: Choose or drag-and-drop a local retail video (e.g., `data/merl/10_1_crop.mp4`).
   - Video preview immediately confirms duration, FPS, and frame count.
3. **Associate Camera Profile**:
   - In Step 2, select **MERL Supermarket Profile** from the profile selector.
   - The interactive visual editor renders camera perspective calibration points, floor coordinate bounds, and candidate shelf zones.
4. **Execution & Lineage Guarantee**:
   - Explain that clicking **Analyze Video with Profile** runs the end-to-end production pipeline locally and writes authoritative `run_summary.json`, timeline sidecars, and annotated WebM video.
   - Clarify that without selecting a profile, the pipeline runs detection/tracking only, safely withholding uncalibrated spatial analytics.

### Minute 5: Evidence Details, Lineage & Verification
1. Navigate to **Evidence and Run Details** (`?view=evidence`).
2. Show **Point Conservation Invariant**:
   - Spatial density conservation verified: $\text{Total} = \text{Binned} + \text{OutOfBounds} + \text{Untransformed}$.
   - Zone conservation verified: $\text{Total} = \text{InZone} + \text{Unassigned}$.
3. Show **Model Inference & Tracking Lineage**:
   - Pre-trained `YOLO11-n ONNX` on `ONNX Runtime CPU` with `ByteTrack`.
   - Clip-local tracking IDs explicitly noted; no facial recognition or customer re-identification.
4. Review **Evidence Limitations**:
   - Explain why all recommendations are `WITHHELD`: single-camera boundary truncation, uncalibrated floor scale, and lack of commercial sales correlation.

---

## 4. Key Metrics Reference Table (Authoritative Artifact Values)

| Metric / Parameter | MERL Supermarket | PRCV Site 03 Boutique | PRCV Site 02 Luxury Jewelry |
| :--- | :--- | :--- | :--- |
| **Run Summary Path** | `uploaded_runs/upload_ca30f3e8` | `uploaded_runs/upload_38d7afe1` | `uploaded_runs/upload_site02_jewelry` |
| **Source Video Frames** | 3,796 frames @ 30.0 FPS | 400 frames @ 15.0 FPS | 218 frames @ 15.0 FPS |
| **Sampled Frames** | 633 frames @ 5.0 FPS | 134 frames @ 5.0 FPS | 73 frames @ 5.0 FPS |
| **Clip Tracks** | 3 clip-local tracks | 5 clip-local tracks | 4 clip-local tracks |
| **Total Whole-Run Dwell** | **17.20 s** (5 episodes) | **1.80 s** (3 episodes) | **13.80 s** (4 episodes) |
| **Candidate Zone** | `shelf_a` (Shelf A) | `side_display_left` (Left Wall Display) | `showcase_display_back` (Rear Jewelry Showcase) |
| **Candidate Zone Dwell** | **10.20 s** (3 episodes) | **1.60 s** (1 episode) | **2.80 s** (2 episodes) |
| **Runner-Up Zone Dwell** | `shelf_b`: 0.00 s (+10.20s lead) | `display_table_center`: 0.00 s (+1.60s lead) | `showcase_display_front`: 0.60 s (+2.20s lead) |
| **Hotspot Bounds** | **X [2.5, 3.0], Y [4.5, 5.0]** | **X [0.5, 1.0], Y [3.5, 4.0]** | *Not evaluated* (Zone aggregate evaluation) |
| **Hotspot Sampled Points** | **10 points** | **3 points** | *None* (null) |
| **Direct Transitions** | 2 (`shelf_a <-> main_aisle`) | 0 (no multi-zone transitions) | 3 (`back -> aisle -> front -> back`) |
| **Recommendation Status** | **WITHHELD** | **WITHHELD** | **WITHHELD** |

---

## 5. What the Prototype Demonstrates vs. What Remains Unvalidated

### What the Prototype Demonstrates
- **End-to-End Local Execution**: Ingests raw video, performs ONNX inference, tracks bounding boxes, maps planar homography, and computes spatial dwell/density/transitions completely offline.
- **Strict Conservation & Accounting**: Mathematical verification that all sampled trajectory points are accounted for across binned, out-of-bounds, and unassigned states.
- **Transparent, Non-Learned Candidate Selection**: Candidate zones are selected using unambiguous observed traffic ranking without arbitrary weights, synthetic scores, or black-box thresholds.
- **Synchronized Visual Verification**: Direct frame-accurate alignment between perspective video footage and floor map coordinates.
- **Privacy by Design**: Operates strictly on temporary clip-local bounding box tracking IDs; zero facial recognition, zero biometrics, and zero cross-camera tracking.

### What Remains Unvalidated (Methodological Guardrails)
- **Floor Metric Scale**: Configured image-to-floor correspondences; independently surveyed physical calibration is unavailable. Homography mappings are configured in arbitrary floor units.
- **Truncated Single-Camera Field of View**: Observations represent only what was visible in a single camera frame; shoppers entering or exiting the frame are not tracked across the store.
- **Commercial Intent & Sales Lift**: Observed footfall dwell does not measure customer purchasing intent, countertop touchpoints, product engagement, or retail revenue impact.
- **Physical Installation Clearance**: Highlighted candidate regions indicate observed customer foot traffic, not architectural installation feasibility, ADA walking aisle compliance, or electrical clearance.
