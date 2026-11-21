"""
Linked Analysis Workspace Regression Tests.
============================================
Verifies:
1. Exact canonical run ID matching and rejection of substring matches.
2. Planar inverse homography (H^-1) camera polygon projection.
3. Dynamic invariant conservation derivation (Pass, Fail, Not Evaluated).
4. Explicit candidate eligibility derivation from candidate_zone_ids / comparison_by_shelf.
5. Authoritative metrics across MERL, Site 03, Site 02, and Out-of-bounds runs.
6. AppTest rendering of the Linked Analysis Workspace across desktop and mobile layouts.
"""

import os
import json
from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.run_adapter import (
    StandardizedRunData,
    discover_standardized_runs,
    load_standardized_run,
    load_run_observation_timeline,
    _compute_camera_polygons,
)
from dashboard.theme import (
    generate_floor_plan_svg,
    generate_isometric_floor_plan_svg,
)


def _resolve_canonical_run_id(query: str, runs: list) -> str:
    run_ids = [r.run_id for r in runs]
    return next((rid for rid in run_ids if query == rid or (rid.startswith("uploaded_runs/") and query == rid.split("/")[1])), None)


def test_canonical_run_id_strict_matching():
    """Verify exact canonical run ID matching without substring inference."""
    runs = discover_standardized_runs("outputs/reports")
    
    # Exact canonical directory IDs must resolve
    assert _resolve_canonical_run_id("upload_ca30f3e8", runs) is not None
    assert _resolve_canonical_run_id("upload_38d7afe1", runs) is not None
    assert _resolve_canonical_run_id("upload_site02_jewelry", runs) is not None
    assert _resolve_canonical_run_id("0a24bbf3de", runs) is not None
    assert _resolve_canonical_run_id("demo_real_summary", runs) is not None

    # Loose substrings must NOT resolve
    assert _resolve_canonical_run_id("ca30f", runs) is None
    assert _resolve_canonical_run_id("site02", runs) is None
    assert _resolve_canonical_run_id("38d7", runs) is None
    assert _resolve_canonical_run_id("jewelry", runs) is None
    assert _resolve_canonical_run_id("non_existent_run_xyz", runs) is None


def test_camera_polygon_inverse_homography():
    """Verify planar H^-1 transforms floor polygons back to pixel coordinates."""
    merl_profile_path = Path("configs/merl_scene_profile.json")
    if not merl_profile_path.exists():
        pytest.skip("MERL scene profile not found")

    with open(merl_profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    # In scene profile, zones is a list of zone dicts
    zones = {z["zone_id"]: [tuple(float(c) for c in pt) for pt in z["polygon"]] for z in profile.get("zones", [])}
    assert "shelf_a" in zones

    cam_polys = _compute_camera_polygons(zones, profile, None)
    assert "shelf_a" in cam_polys
    poly = cam_polys["shelf_a"]
    assert len(poly) == 4
    # Bounding box of projected polygon in pixel coordinates
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    # MERL image is 920x680
    assert all(0 <= x <= 920 for x in xs)
    assert all(0 <= y <= 680 for y in ys)
    # Check expected coordinate range for shelf_a
    assert 150 <= min(xs) <= 250
    assert 350 <= max(xs) <= 450


def test_dynamic_invariant_conservation():
    """Verify dynamic conservation status and exact constituent counts are derived from run fields."""
    runs = discover_standardized_runs("outputs/reports")
    run_map = {r.run_id: r for r in runs}

    # 1. MERL: shelf_a; 2 transitions; 508 = 169 + 339 + 0
    merl_id = _resolve_canonical_run_id("upload_ca30f3e8", runs)
    assert merl_id is not None
    merl = run_map[merl_id]
    assert merl.density_conservation_verified is True
    assert merl.zone_conservation_verified is True
    assert merl.total_points == 508
    assert merl.binned_points == 169
    assert merl.out_of_bounds_points == 339
    assert (merl.untransformed_points or 0) == 0
    assert merl.total_points == (merl.binned_points + merl.out_of_bounds_points + (merl.untransformed_points or 0))
    assert merl.total_transitions == 2

    # 2. Site 03: side_display_left; 0 transitions; 15 = 14 + 1 + 0
    site03_id = _resolve_canonical_run_id("upload_38d7afe1", runs)
    assert site03_id is not None
    site03 = run_map[site03_id]
    assert site03.density_conservation_verified is True
    assert site03.zone_conservation_verified is True
    assert site03.total_points == 15
    assert site03.binned_points == 14
    assert site03.out_of_bounds_points == 1
    assert (site03.untransformed_points or 0) == 0
    assert site03.total_points == (site03.binned_points + site03.out_of_bounds_points + (site03.untransformed_points or 0))
    assert site03.total_transitions == 0

    # 3. Site 02: showcase_display_back; 3 transitions; 70 = 66 + 4 + 0
    site02_id = _resolve_canonical_run_id("upload_site02_jewelry", runs)
    assert site02_id is not None
    site02 = run_map[site02_id]
    assert site02.density_conservation_verified is True
    assert site02.zone_conservation_verified is True
    assert site02.total_points == 70
    assert site02.binned_points == 66
    assert site02.out_of_bounds_points == 4
    assert (site02.untransformed_points or 0) == 0
    assert site02.total_points == (site02.binned_points + site02.out_of_bounds_points + (site02.untransformed_points or 0))
    assert site02.total_transitions == 3


def test_candidate_eligibility_no_token_parsing():
    """Verify candidate eligibility derives strictly from candidate_zone_ids / comparison_by_shelf."""
    runs = discover_standardized_runs("outputs/reports")
    run_map = {r.run_id: r for r in runs}

    merl_id = _resolve_canonical_run_id("upload_ca30f3e8", runs)
    merl = run_map[merl_id]
    assert merl.candidate_zone_ids == ["shelf_a", "shelf_b"]
    assert "shelf_a" in merl.candidate_zone_ids
    assert "shelf_b" in merl.candidate_zone_ids

    site02_id = _resolve_canonical_run_id("upload_site02_jewelry", runs)
    site02 = run_map[site02_id]
    assert site02.candidate_zone_ids == ["showcase_display_front", "showcase_display_back"]

    site03_id = _resolve_canonical_run_id("upload_38d7afe1", runs)
    site03 = run_map[site03_id]
    assert site03.candidate_zone_ids == ["side_display_left", "display_table_center"]


def test_authoritative_metrics_all_curated_runs():
    """Verify authoritative metrics for MERL, Site 03, Site 02, and Out-of-bounds runs."""
    runs = discover_standardized_runs("outputs/reports")
    run_map = {r.run_id: r for r in runs}

    # 1. MERL
    merl_id = _resolve_canonical_run_id("upload_ca30f3e8", runs)
    merl = run_map[merl_id]
    assert merl.candidate_zone == "shelf_a"
    assert merl.candidate_dwell_sec == 10.2
    assert merl.total_dwell_sec == 17.2
    assert merl.candidate_hotspot_cell["x_min"] == 2.5
    assert merl.candidate_hotspot_cell["x_max"] == 3.0
    assert merl.candidate_hotspot_cell["y_min"] == 4.5
    assert merl.candidate_hotspot_cell["y_max"] == 5.0
    assert merl.candidate_hotspot_cell["sampled_points"] == 10

    # 2. Site 03
    site03_id = _resolve_canonical_run_id("upload_38d7afe1", runs)
    site03 = run_map[site03_id]
    assert site03.candidate_zone == "side_display_left"
    assert site03.candidate_dwell_sec == 1.6
    assert site03.total_dwell_sec == 1.8
    assert site03.candidate_hotspot_cell["x_min"] == 0.5
    assert site03.candidate_hotspot_cell["x_max"] == 1.0
    assert site03.candidate_hotspot_cell["y_min"] == 3.5
    assert site03.candidate_hotspot_cell["y_max"] == 4.0
    assert site03.candidate_hotspot_cell["sampled_points"] == 3

    # 3. Site 02
    site02_id = _resolve_canonical_run_id("upload_site02_jewelry", runs)
    site02 = run_map[site02_id]
    assert site02.candidate_zone == "showcase_display_back"
    assert site02.candidate_dwell_sec == 2.8
    assert site02.total_dwell_sec == 13.8
    assert site02.candidate_hotspot_cell is None
    assert site02.total_transitions == 3

    # 4. Out-of-bounds run
    oob_id = _resolve_canonical_run_id("0a24bbf3de", runs)
    oob = run_map[oob_id]
    assert oob.candidate_zone is None
    assert oob.candidate_dwell_sec is None
    assert oob.total_dwell_sec == 0.0
    assert oob.in_zone_points == 0
    assert oob.unassigned_points == 479
    assert oob.out_of_bounds_points == 479


def test_apptest_linked_workspace_rendering():
    """Verify AppTest successfully renders Linked Analysis Workspace."""
    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.query_params["view"] = "spatial_studio"
    at.query_params["run_id"] = "upload_ca30f3e8"
    at.query_params["stage"] = "6"
    at.run()
    assert not at.exception, f"AppTest raised exception: {at.exception}"


def test_analytical_3d_geometry_and_provenance():
    """Verify analytical 3D geometry fields, recorded upload provenance, and dwell distinctions."""
    runs = discover_standardized_runs("outputs/reports")
    run_map = {r.run_id: r for r in runs}

    # 1. Out-of-bounds run recorded upload provenance (not synthetic)
    oob_id = _resolve_canonical_run_id("0a24bbf3de", runs)
    assert oob_id is not None
    oob = run_map[oob_id]
    assert oob.source_type == "recorded_video"
    assert oob.is_synthetic is False
    assert oob.dataset_name == "user_uploaded_indoor_video"
    assert oob.candidate_hotspot_cell is None

    # 2. MERL Dwell distinctions
    merl_id = _resolve_canonical_run_id("upload_ca30f3e8", runs)
    merl = run_map[merl_id]
    assert merl.zone_dwell is not None
    assert "shelf_a" in merl.zone_dwell
    assert (merl.zone_dwell["shelf_a"].get("observed_dwell_sec") or merl.zone_dwell["shelf_a"].get("observed_dwell_span_sec")) == 10.2
    assert (merl.zone_dwell["checkout"].get("observed_dwell_sec", merl.zone_dwell["checkout"].get("observed_dwell_span_sec"))) == 0.0
    assert merl.candidate_hotspot_cell is not None
    assert merl.candidate_hotspot_cell["sampled_points"] == 10

    # 3. Site 02 Zone-level placement with absent hotspot
    site02_id = _resolve_canonical_run_id("upload_site02_jewelry", runs)
    site02 = run_map[site02_id]
    assert site02.candidate_hotspot_cell is None
    assert site02.candidate_zone == "showcase_display_back"
    assert (site02.zone_dwell["showcase_display_back"].get("observed_dwell_sec") or site02.zone_dwell["showcase_display_back"].get("observed_dwell_span_sec")) == 2.8


def test_analytical_3d_component_and_presentation_requirements():
    """Verify player component adheres to exact prompt presentation requirements."""
    player_path = Path("dashboard/components/synchronized_player/index.html")
    assert player_path.exists()
    content = player_path.read_text(encoding="utf-8")

    # Requirement 1: Projected using configured homography
    assert "Projected using configured homography" in content
    assert "Inverse Camera Projection Calibrated" not in content

    # Requirement 2: 3D Analytical canvas and containers
    assert "3D Analytical" in content
    assert "floor-3d-canvas" in content
    assert "floor-3d-container" in content
    assert "toolbar-3d-strip" in content
    assert "floor-3d-legend-strip" in content

    # Requirement 3: Whole-run observed dwell Labelled Seconds Scale & Disclaimer
    assert "Analytical 3D Visualization" in content
    assert "Whole-run Observed Dwell" in content or "Whole-run observed dwell" in content
    assert "Whole-run observed dwell scale:" in content
    assert "Whole-run dwell (s)" in content

    # Requirement 4: Narrow screen default and accessible controls
    assert "checkNarrowScreenDefault" in content
    assert "orbit3D" in content
    assert "pan3D" in content
    assert "zoom3D" in content
    assert "update3DLegend" in content

    # Placement Candidate integer point count card
    candidate_view_path = Path("dashboard/views/placement_candidate.py")
    cand_content = candidate_view_path.read_text(encoding="utf-8")
    assert "Hotspot Point Count" in cand_content
    assert "pts" in cand_content


def _execute_node_engine(script_body: str) -> dict:
    """Helper to execute JavaScript snippets against the production spatial_player_engine.js module."""
    import subprocess
    wrapper = f"""
    const engine = require('./dashboard/components/synchronized_player/spatial_player_engine.js');
    {script_body}
    """
    res = subprocess.run(
        ["node", "-e", wrapper],
        check=True,
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )
    return json.loads(res.stdout)


def test_analytical_3d_production_js_dynamic_scale():
    """Verify production JS dynamic scale calculation and handling of dwell values above 20 seconds."""
    js_code = """
    const cStd = engine.computeMaxScaleDwell({ z1: { observed_dwell_sec: 10.2 }, z2: { observed_dwell_sec: 0.0 } });
    const c45 = engine.computeMaxScaleDwell({ z1: { observed_dwell_sec: 45.0 }, z2: { observed_dwell_sec: 12.0 } });
    const c85 = engine.computeMaxScaleDwell({ z1: { observed_dwell_sec: 85.0 }, z2: { observed_dwell_sec: 15.0 } });
    const c135 = engine.computeMaxScaleDwell({ z1: { observed_dwell_sec: 135.0 } });
    const ticks45 = engine.computeRulerTicks(c45);
    const ticks85 = engine.computeRulerTicks(c85);
    const ticks135 = engine.computeRulerTicks(c135);
    console.log(JSON.stringify({ cStd, c45, c85, c135, ticks45, ticks85, ticks135 }));
    """
    data = _execute_node_engine(js_code)

    assert data["cStd"] == 20.0
    assert data["c45"] == 50.0
    assert data["c85"] == 100.0
    assert data["c135"] == 150.0

    # Verify ticks contain labelled scale
    assert len(data["ticks45"]) == 5
    assert data["ticks45"][-1]["label"] == "50s"
    assert data["ticks85"][-1]["label"] == "100s"
    assert data["ticks135"][-1]["label"] == "150s"


def test_analytical_3d_production_js_missing_zero_dwell_distinction():
    """Verify production JS strict distinction among positive dwell, evaluated zero dwell, and missing dwell."""
    js_code = """
    const pos = engine.classifyDwellState(10.2, 'recorded');
    const zeroVal = engine.classifyDwellState(0.0, 'evaluated_zero');
    const missingVal = engine.classifyDwellState(null, 'missing');
    const undefinedVal = engine.classifyDwellState(undefined, undefined);
    console.log(JSON.stringify({ pos, zeroVal, missingVal, undefinedVal }));
    """
    data = _execute_node_engine(js_code)

    # Positive dwell
    assert data["pos"]["isRecorded"] is True
    assert data["pos"]["isZero"] is False
    assert data["pos"]["isMissing"] is False
    assert data["pos"]["state"] == "positive"
    assert "10.2s" in data["pos"]["label"]

    # Evaluated zero
    assert data["zeroVal"]["isRecorded"] is False
    assert data["zeroVal"]["isZero"] is True
    assert data["zeroVal"]["isMissing"] is False
    assert data["zeroVal"]["state"] == "evaluated_zero"
    assert "0.0s" in data["zeroVal"]["label"]

    # Missing dwell
    assert data["missingVal"]["isRecorded"] is False
    assert data["missingVal"]["isZero"] is False
    assert data["missingVal"]["isMissing"] is True
    assert data["missingVal"]["state"] == "missing"

    assert data["undefinedVal"]["isMissing"] is True


def test_analytical_3d_production_js_trajectory_gap_breaking():
    """Verify production JS trajectory gap breaking across unobserved sample index gaps."""
    js_code = """
    const pointsWithGap = [
      { sampleIdx: 0, pt: [1.0, 2.0] },
      { sampleIdx: 1, pt: [1.2, 2.1] },
      { sampleIdx: 2, pt: [1.4, 2.3] },
      // Sample 3 is unobserved / dropped!
      { sampleIdx: 4, pt: [2.0, 3.0] },
      { sampleIdx: 5, pt: [2.2, 3.2] }
    ];
    const segs = engine.segmentTrajectoryTrails(pointsWithGap);
    console.log(JSON.stringify({ segs }));
    """
    data = _execute_node_engine(js_code)
    segs = data["segs"]

    assert len(segs) == 2
    assert len(segs[0]) == 3
    assert len(segs[1]) == 2
    assert segs[0][-1]["sampleIdx"] == 2
    assert segs[1][0]["sampleIdx"] == 4
    # Confirms no continuous segment spans across sample 2 to sample 4!


def test_episode_recovery_and_evidence_reconciliation():
    """Verify evidence-linked episode recovery across MERL, Site 03, Site 02, and missing timeline runs."""
    from dashboard.views.spatial_studio import recover_zone_dwell_episodes

    runs = discover_standardized_runs("outputs/reports")

    # 1. MERL retail baseline: upload_ca30f3e8
    merl_run = next((r for r in runs if "ca30f3e8" in r.run_id), None)
    assert merl_run is not None
    merl_tl = load_run_observation_timeline(merl_run)
    assert merl_tl is not None

    merl_episodes, merl_err = recover_zone_dwell_episodes(merl_run, merl_tl, "shelf_a")
    assert merl_err is None
    assert merl_episodes is not None
    assert len(merl_episodes) == 3

    # Validate sum of durations equals aggregate dwell 10.2s
    total_ep_dur = sum(e["duration_sec"] for e in merl_episodes)
    assert round(total_ep_dur, 1) == 10.2
    assert merl_episodes[0]["track_id"] == 1
    assert merl_episodes[0]["start_sample_idx"] == 167
    assert merl_episodes[0]["end_sample_idx"] == 177
    assert round(merl_episodes[0]["duration_sec"], 1) == 2.0
    assert merl_episodes[0]["has_gaps"] is False

    # Episode 2 has observation gap at sample 382
    assert merl_episodes[1]["track_id"] == 3
    assert merl_episodes[1]["has_gaps"] is True
    assert merl_episodes[1]["gaps"][0]["from_sample"] == 381
    assert merl_episodes[1]["gaps"][0]["to_sample"] == 383
    assert round(merl_episodes[1]["duration_sec"], 1) == 6.8

    # Unvisited zone in MERL: checkout
    checkout_eps, _ = recover_zone_dwell_episodes(merl_run, merl_tl, "checkout")
    assert checkout_eps == []

    # 2. Site 03 baseline: upload_38d7afe1
    site03_run = next((r for r in runs if "38d7afe1" in r.run_id), None)
    assert site03_run is not None
    site03_tl = load_run_observation_timeline(site03_run)
    assert site03_tl is not None

    site03_eps, _ = recover_zone_dwell_episodes(site03_run, site03_tl, "side_display_left")
    assert site03_eps is not None
    assert len(site03_eps) == 1
    assert site03_eps[0]["track_id"] == 1
    assert round(site03_eps[0]["duration_sec"], 1) == 1.6
    assert site03_eps[0]["has_gaps"] is True  # gap at sample 8

    # 3. Site 02 baseline: upload_site02_jewelry
    site02_run = next((r for r in runs if "site02_jewelry" in r.run_id), None)
    assert site02_run is not None
    site02_tl = load_run_observation_timeline(site02_run)
    assert site02_tl is not None

    site02_eps, _ = recover_zone_dwell_episodes(site02_run, site02_tl, "showcase_display_back")
    assert site02_eps is not None
    assert len(site02_eps) == 2
    assert round(sum(e["duration_sec"] for e in site02_eps), 1) == 2.8

    # 4. Missing timeline run: demo_real_summary
    demo_run = next((r for r in runs if r.run_id == "demo_real_summary"), None)
    assert demo_run is not None
    demo_tl = load_run_observation_timeline(demo_run)
    assert demo_tl is None

    missing_eps, missing_reason = recover_zone_dwell_episodes(demo_run, demo_tl, "shelf_a")
    assert missing_eps is None
    assert "run_observation_timeline.json" in missing_reason
    assert "observed_dwell_span_sec" in missing_reason


def test_episode_recovery_non_fabricating_defaults():
    """Verify episode recovery avoids fabricating defaults for missing timestamps, frames, coords, and null zones."""
    from dashboard.views.spatial_studio import recover_zone_dwell_episodes

    runs = discover_standardized_runs("outputs/reports")
    merl_run = next(r for r in runs if "ca30f3e8" in r.run_id)

    # 1. Non-monotonic timestamp fails gracefully without inventing synthetic points
    non_monotonic_timeline = {
        "playback_mapping": [
            {
                "sample_idx": 0,
                "source_frame_idx": 0,
                "source_timestamp_sec": 5.0,
                "observations": [{"track_id": 1, "floor_point": [1, 1], "assigned_zone": "shelf_a"}]
            },
            {
                "sample_idx": 1,
                "source_frame_idx": 1,
                "source_timestamp_sec": 3.0,  # Non-monotonic!
                "observations": [{"track_id": 1, "floor_point": [1, 1], "assigned_zone": "shelf_a"}]
            }
        ]
    }
    eps, err = recover_zone_dwell_episodes(merl_run, non_monotonic_timeline, "shelf_a")
    assert eps is None
    assert "Non-monotonic source timestamp" in err

    # 2. Non-monotonic frame index fails gracefully
    non_monotonic_frame_timeline = {
        "playback_mapping": [
            {
                "sample_idx": 0,
                "source_frame_idx": 10,
                "source_timestamp_sec": 1.0,
                "observations": [{"track_id": 1, "floor_point": [1, 1], "assigned_zone": "shelf_a"}]
            },
            {
                "sample_idx": 1,
                "source_frame_idx": 5,  # Non-monotonic!
                "source_timestamp_sec": 2.0,
                "observations": [{"track_id": 1, "floor_point": [1, 1], "assigned_zone": "shelf_a"}]
            }
        ]
    }
    eps, err = recover_zone_dwell_episodes(merl_run, non_monotonic_frame_timeline, "shelf_a")
    assert eps is None
    assert "Non-monotonic source frame index" in err

    # 3. Missing coordinates remain None, but temporal inspection is preserved
    missing_coords_timeline = {
        "playback_mapping": [
            {
                "sample_idx": 0,
                "source_frame_idx": 0,
                "source_timestamp_sec": 0.0,
                "observations": [{"track_id": 1, "floor_point": None, "image_ground_point": None, "assigned_zone": "shelf_a"}]
            },
            {
                "sample_idx": 1,
                "source_frame_idx": 5,
                "source_timestamp_sec": 1.0,
                "observations": [{"track_id": 1, "floor_point": None, "image_ground_point": None, "assigned_zone": "shelf_a"}]
            }
        ]
    }
    eps, err = recover_zone_dwell_episodes(merl_run, missing_coords_timeline, "shelf_a")
    assert err is None
    assert eps is not None
    assert len(eps) == 1
    assert eps[0]["duration_sec"] == 1.0
    assert eps[0]["supporting_observations"][0]["floor_point"] is None
    assert eps[0]["supporting_observations"][0]["image_ground_point"] is None

    # 4. Explicit null zone assignment is preserved (not replaced through truthiness)
    null_zone_timeline = {
        "playback_mapping": [
            {
                "sample_idx": 0,
                "source_frame_idx": 0,
                "source_timestamp_sec": 0.0,
                "observations": [{"track_id": 1, "assigned_zone": None, "zone_id": "shelf_a"}]
            }
        ]
    }
    eps, err = recover_zone_dwell_episodes(merl_run, null_zone_timeline, "shelf_a")
    assert err is None
    assert eps == []


def test_analytical_3d_production_js_projection_and_camera_transforms():
    """Verify production JS 3D projection mathematical transformations under orbit, pan, and zoom."""
    js_code = """
    const baseCam = { yaw: -0.45, pitch: 0.85, panX: 0, panY: 0, scy: 0 };
    const pt0 = engine.project3D(5.0, 3.0, 1.0, 5.0, 3.0, 50, 800, 500, baseCam);

    // Orbit: change yaw and pitch
    const orbitedCam = { yaw: -0.25, pitch: 0.95, panX: 0, panY: 0, scy: 0 };
    const ptOrbit = engine.project3D(6.0, 4.0, 1.0, 5.0, 3.0, 50, 800, 500, orbitedCam);

    // Pan: shift panX and panY
    const pannedCam = { yaw: -0.45, pitch: 0.85, panX: 40, panY: -20, scy: 0 };
    const ptPan = engine.project3D(5.0, 3.0, 1.0, 5.0, 3.0, 50, 800, 500, pannedCam);

    // Zoom: scaled view scale S
    const ptZoom = engine.project3D(6.0, 4.0, 1.0, 5.0, 3.0, 75, 800, 500, baseCam);

    console.log(JSON.stringify({ pt0, ptOrbit, ptPan, ptZoom }));
    """
    data = _execute_node_engine(js_code)

    # Origin center projects to canvas center + pan
    assert round(data["pt0"]["x"], 1) == 400.0
    # Pan shifts canvas coordinates directly
    assert data["ptPan"]["x"] == data["pt0"]["x"] + 40
    assert data["ptPan"]["y"] == data["pt0"]["y"] - 20
    # Zoom increases deviation from center
    assert abs(data["ptZoom"]["x"] - 400) > 0


def test_apptest_linked_workspace_interactions():
    """Verify state preservation across rerun, inspect episode seek, and run switching via AppTest."""
    runs = discover_standardized_runs("outputs/reports")
    merl_run = next(r for r in runs if "ca30f3e8" in r.run_id)
    site02_run = next(r for r in runs if "site02_jewelry" in r.run_id)

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")

    # 0. Verify guest access restriction: guest is blocked by protected access barrier
    at_guest = AppTest.from_file(app_path, default_timeout=30)
    at_guest.query_params["view"] = "spatial_studio"
    at_guest.run()
    barrier_text = " ".join([m.value for m in at_guest.markdown] + [w.value for w in at_guest.warning])
    assert "Protected Feature Access" in barrier_text or "Authentication Required" in barrier_text

    at = AppTest.from_file(app_path, default_timeout=30)
    at.session_state["authenticated_user"] = "reviewer"
    at.query_params["view"] = "spatial_studio"
    at.query_params["run_id"] = "upload_ca30f3e8"
    at.query_params["stage"] = "6"
    at.run()
    assert not at.exception

    # 1. Zone selection and rerun preserves sample index
    at.session_state[f"studio_sample_idx_{merl_run.run_id}"] = 170
    at.session_state["studio_selected_zone"] = "shelf_b"
    at.run()
    assert not at.exception
    assert at.session_state[f"studio_sample_idx_{merl_run.run_id}"] == 170

    # 2. Inspect Episode button click seeks sample and selects track
    # For MERL shelf_a episode 1: start_sample_idx is 167, track is 1
    at.session_state["studio_selected_zone"] = "shelf_a"
    at.run()
    btn_key = f"btn_inspect_ep_{merl_run.run_id}_shelf_a_1"
    inspect_btn = next((b for b in at.button if b.key == btn_key), None)
    assert inspect_btn is not None
    inspect_btn.click().run()
    assert not at.exception
    assert at.session_state[f"studio_sample_idx_{merl_run.run_id}"] == 167
    assert at.session_state["studio_active_track"] == "1"
    assert at.session_state["studio_selected_zone"] == "shelf_a"

    # 3. Site 02 front-showcase episode inspection seeks to sample 60 and track 1
    at.query_params["run_id"] = "upload_site02_jewelry"
    at.query_params["zone_id"] = "showcase_display_front"
    at.run()
    assert not at.exception
    btn_s02_key = f"btn_inspect_ep_{site02_run.run_id}_showcase_display_front_1"
    inspect_s02_btn = next((b for b in at.button if b.key == btn_s02_key), None)
    assert inspect_s02_btn is not None
    inspect_s02_btn.click().run()
    assert not at.exception
    assert at.session_state[f"studio_sample_idx_{site02_run.run_id}"] == 60
    assert at.session_state["studio_active_track"] == "1"
    assert at.session_state["studio_selected_zone"] == "showcase_display_front"

    # 4. Run switching clears stale state
    at.query_params["run_id"] = "upload_38d7afe1"
    at.run()
    assert not at.exception
    assert "upload_38d7afe1" in at.session_state["authoritative_run_id"]
    assert at.session_state.get("studio_selected_zone") != "showcase_display_front"


def test_playback_observation_gap_behavior_at_sample_8():
    """Verify production playback behavior at sample 8 observation gap for upload_6f969f54.
    
    Verifies:
    1. Actual profile bounds are X [0, 10], Y [0, 9] (not [0, 7] x [0, 5]).
    2. Timeline sample 7 contains Track 1 observation (time 1.40s, frame 21).
    3. Timeline sample 8 contains exactly 0 observations (time 1.60s, frame 24, observations == []).
    4. Timeline sample 9 contains Track 1 observation (time 1.80s, frame 27).
    5. Track 1 trajectory trail breaks across sample 8 (spatial_player_engine.segmentTrajectoryTrails).
    6. Episode recovery for side_display_left identifies the gap at sample 8 and sets has_gaps = True.
    """
    from dashboard.views.spatial_studio import recover_zone_dwell_episodes

    runs = discover_standardized_runs("outputs/reports")
    matched_run = next((r for r in runs if "upload_6f969f54" in r.run_id or "38d7afe1" in r.run_id), None)
    assert matched_run is not None

    # 1. Profile bounds check
    assert matched_run.bounds == (0.0, 10.0, 0.0, 9.0)

    # 2. Timeline checks
    tl = load_run_observation_timeline(matched_run)
    assert tl is not None
    mapping = tl.get("playback_mapping", [])
    assert len(mapping) >= 11

    s7 = mapping[7]
    s8 = mapping[8]
    s9 = mapping[9]

    # Sample 7: Track 1 observed
    assert s7["sample_index"] == 7
    assert s7["source_timestamp_sec"] == 1.4
    assert s7["source_frame_idx"] == 21
    assert len(s7["observations"]) == 1
    assert s7["observations"][0]["track_id"] == 1
    assert s7["observations"][0]["assigned_zone"] == "side_display_left"

    # Sample 8: Observation gap (observations == [])
    assert s8["sample_index"] == 8
    assert s8["source_timestamp_sec"] == 1.6
    assert s8["source_frame_idx"] == 24
    assert len(s8["observations"]) == 0
    assert s8["observations"] == []

    # Sample 9: Track 1 resumed
    assert s9["sample_index"] == 9
    assert s9["source_timestamp_sec"] == 1.8
    assert s9["source_frame_idx"] == 27
    assert len(s9["observations"]) == 1
    assert s9["observations"][0]["track_id"] == 1
    assert s9["observations"][0]["assigned_zone"] == "side_display_left"

    # 3. Production JS trail gap breaking: segmentTrajectoryTrails
    # Collect all floor points for Track 1 up to sample 10 (sample 10 completes post-gap segment)
    pts_track1_all = []
    for i in range(11):
        for obs in mapping[i].get("observations", []):
            if obs.get("track_id") == 1 and obs.get("floor_point"):
                pts_track1_all.append({"sampleIdx": i, "pt": obs["floor_point"]})

    # Track 1 has points at samples 2, 3, 4, 5, 6, 7 (6 pts) and 9, 10 (2 pts), with 8 missing
    sample_indices = [p["sampleIdx"] for p in pts_track1_all]
    assert 8 not in sample_indices
    assert 7 in sample_indices
    assert 9 in sample_indices
    assert 10 in sample_indices

    # Check at Sample 8: trail only has segment 1 (samples 2..7), no point carried to 8
    pts_up_to_8 = [p for p in pts_track1_all if p["sampleIdx"] <= 8]
    js_code_8 = f"""
    const pts = {json.dumps(pts_up_to_8)};
    const segs = engine.segmentTrajectoryTrails(pts);
    console.log(JSON.stringify({{ segs }}));
    """
    data_8 = _execute_node_engine(js_code_8)
    segs_8 = data_8["segs"]
    assert len(segs_8) == 1
    assert segs_8[0][-1]["sampleIdx"] == 7  # Trail terminates at sample 7; sample 8 is not in trail

    # Check across gap up to Sample 10: broken into 2 separate segments across sample 8 gap
    js_code_10 = f"""
    const pts = {json.dumps(pts_track1_all)};
    const segs = engine.segmentTrajectoryTrails(pts);
    console.log(JSON.stringify({{ segs }}));
    """
    data_10 = _execute_node_engine(js_code_10)
    segs_10 = data_10["segs"]
    assert len(segs_10) == 2
    # Segment 1 ends at sample 7
    assert segs_10[0][-1]["sampleIdx"] == 7
    # Segment 2 starts at sample 9 (no line across gap 7->9)
    assert segs_10[1][0]["sampleIdx"] == 9

    # 4. Episode recovery recognizes gap at sample 8
    eps, err = recover_zone_dwell_episodes(matched_run, tl, "side_display_left")
    assert err is None
    assert eps is not None
    assert len(eps) == 1
    ep = eps[0]
    assert ep["track_id"] == 1
    assert ep["has_gaps"] is True
    assert any(g.get("from_sample") == 7 and g.get("to_sample") == 9 for g in ep.get("gaps", []))





