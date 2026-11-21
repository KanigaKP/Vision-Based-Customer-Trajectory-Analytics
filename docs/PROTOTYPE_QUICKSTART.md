# Ready-to-run local prototype

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

Double-click `start_demo.cmd` in the repository. Open http://127.0.0.1:8501/?view=all for existing visual results, or http://127.0.0.1:8501/?view=upload to process a recorded video. If the application is already running, use the existing URL rather than starting another instance.

## Demonstrate a real run
Upload `data/merl/10_1_crop.mp4`. Choose the existing MERL camera profile, or upload `configs/merl_scene_profile.json` in the camera setup expander (not both). Select the candidate objective and start analysis. Wait for processing to complete before inspecting the tracking video, trajectories, zone tables, heatmap, dwell spans and exploratory candidate. Keep the terminal running during the demonstration.

## A different recorded camera
Provide a camera-specific JSON profile: video width/height, matching image/floor correspondences, bounds, cell size, convex non-overlapping zone polygons, candidate zone IDs and hysteresis window. The downloadable UI example illustrates the format only; its coordinates must be replaced. The MERL profile is specific to its existing crop and must not be reused for another camera. Without a profile, spatial results remain unavailable rather than invented. Configured image-to-floor correspondences; independently surveyed physical calibration is unavailable.

## Output meaning
The pretrained YOLO11-n ONNX model performs inference; this project does not train it. IDs are clip-local. Density counts sampled trajectory points; dwell measures observed spans. Exploratory candidates rank the specified observed metrics and do not establish optimal product placement or sales effects. Unverified calibration and coverage keep actionable recommendations withheld. Synthetic examples remain separate from real recordings.

## Verification
Run from the repository in PowerShell:

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -q -p no:cacheprovider
.\venv\Scripts\python.exe scripts/verify_phase_08.py
```

396 regression tests passed (3 skipped, out of 399 collected) in 62.81s across 29 modular suites; 70 component checks passed in 1.86s via verify_phase_08.py. The local server health endpoint responded successfully. Camera-specific profile validation covers dimensions, correspondence geometry, bounds, zone overlap, candidate IDs and grid size. See run outputs under `outputs/reports/`; generated files are ignored by Git.
