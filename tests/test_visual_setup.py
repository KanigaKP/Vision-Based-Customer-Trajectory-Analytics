"""
Unit and integration tests for Direction A visual single-camera calibration setup.
==================================================================================
Covers:
1. Coordinate conversion under display scaling, letterboxing, and aspect ratios.
2. Representative video frame extraction and metadata verification.
3. Profile creation, SVD homography diagnostics, and structural validation.
4. Actionable rejection of non-convex and invalid zone geometry.
5. JSON import/export round-trip and dimension mismatch rejection.
6. End-to-end integration with offline analysis pipeline using custom scene profile.
"""

import os
import json
import pytest
import numpy as np

from dashboard.visual_setup import (
    screen_to_original_coords,
    original_to_screen_coords,
    extract_representative_frame,
    create_default_profile,
    validate_and_diagnose_profile,
    export_profile_json,
    import_profile_json,
)
from src.spatial.scene_profile import validate_scene_profile


# ---------------------------------------------------------------------------
# 1. Coordinate Conversion Tests
# ---------------------------------------------------------------------------

def test_screen_to_original_coords_1to1():
    """Exact 1:1 scale when container matches original dimensions."""
    orig_x, orig_y = screen_to_original_coords(100.0, 200.0, 920.0, 680.0, 920.0, 680.0)
    assert orig_x == 100.0
    assert orig_y == 200.0


def test_screen_to_original_coords_pillarbox():
    """Pillarbox padding when container is wider than original aspect ratio."""
    # Orig: 920x680 (~1.353), Container: 1840x680 (scale = min(2.0, 1.0) = 1.0)
    # Rendered w = 920, offset_x = (1840 - 920)/2 = 460
    orig_x, orig_y = screen_to_original_coords(460.0 + 100.0, 200.0, 1840.0, 680.0, 920.0, 680.0)
    assert orig_x == 100.0
    assert orig_y == 200.0


def test_screen_to_original_coords_letterbox():
    """Letterbox padding when container is taller than original aspect ratio."""
    # Orig: 920x680, Container: 920x1360 (scale = min(1.0, 2.0) = 1.0)
    # Rendered h = 680, offset_y = (1360 - 680)/2 = 340
    orig_x, orig_y = screen_to_original_coords(100.0, 340.0 + 200.0, 920.0, 1360.0, 920.0, 680.0)
    assert orig_x == 100.0
    assert orig_y == 200.0


def test_coordinate_roundtrip():
    """Converting orig -> screen -> orig should be an identity operation."""
    c_w, c_h = 800.0, 600.0
    o_w, o_h = 1920.0, 1080.0

    test_points = [(100.0, 150.0), (960.0, 540.0), (1800.0, 1000.0)]
    for ox, oy in test_points:
        sx, sy = original_to_screen_coords(ox, oy, c_w, c_h, o_w, o_h)
        rx, ry = screen_to_original_coords(sx, sy, c_w, c_h, o_w, o_h)
        assert abs(rx - ox) <= 0.05
        assert abs(ry - oy) <= 0.05


def test_invalid_dimensions_raise():
    """Non-positive dimensions should raise ValueError."""
    with pytest.raises(ValueError):
        screen_to_original_coords(10, 10, -100, 680, 920, 680)
    with pytest.raises(ValueError):
        original_to_screen_coords(10, 10, 800, 0, 920, 680)


# ---------------------------------------------------------------------------
# 2. Representative Frame Extraction Tests
# ---------------------------------------------------------------------------

def test_extract_representative_frame_merl_video():
    """Extract representative frame from existing MERL video clip."""
    merl_video = "data/merl/10_1_crop.mp4"
    if not os.path.exists(merl_video):
        pytest.skip("MERL video not present in local data directory")

    info = extract_representative_frame(merl_video, frame_idx=0)
    assert info["width"] == 920
    assert info["height"] == 680
    assert info["total_frames"] > 0
    assert info["fps"] > 0
    assert info["has_frame"] is True
    assert isinstance(info["jpeg_base64"], str)
    assert len(info["jpeg_base64"]) > 500  # Non-trivial base64 image payload


def test_extract_representative_frame_from_bytes():
    """Extract representative frame from raw bytes."""
    merl_video = "data/merl/10_1_crop.mp4"
    if not os.path.exists(merl_video):
        pytest.skip("MERL video not present in local data directory")

    with open(merl_video, "rb") as f:
        v_bytes = f.read()

    info = extract_representative_frame(v_bytes, frame_idx=10)
    assert info["frame_idx"] == 10
    assert info["width"] == 920
    assert info["height"] == 680


# ---------------------------------------------------------------------------
# 3. Profile Construction & Diagnostics Tests
# ---------------------------------------------------------------------------

def test_create_default_profile_structure():
    """Default profile must pass full validation and compute homography condition."""
    profile = create_default_profile(920, 680, name="Test Retail View")
    assert profile["image_width"] == 920
    assert profile["image_height"] == 680
    assert len(profile["image_points"]) == 4
    assert len(profile["floor_points"]) == 4
    assert len(profile["zones"]) == 3
    assert "shelf_a" in profile["candidate_zone_ids"]

    is_valid, err, diag = validate_and_diagnose_profile(profile, video_width=920, video_height=680)
    assert is_valid is True
    assert err is None
    assert diag is not None
    assert diag["is_valid"] is True
    assert diag["condition_number"] > 0.0
    assert "mean_reprojection_error" in diag


def test_rejection_of_dimension_mismatch():
    """Profile must be rejected if its image dimensions mismatch the video."""
    profile = create_default_profile(920, 680)
    # Target video is 1920x1080
    is_valid, err, diag = validate_and_diagnose_profile(profile, video_width=1920, video_height=1080)
    assert is_valid is False
    assert "image_width" in err or "dimension" in err or "1920" in err


# ---------------------------------------------------------------------------
# 4. Geometry Rejection & Non-Convex Polygons
# ---------------------------------------------------------------------------

def test_rejection_of_concave_zone():
    """Zones with concave polygons must be rejected with actionable message."""
    profile = create_default_profile(920, 680)
    # Create concave arrowhead polygon
    profile["zones"][0]["polygon"] = [
        [1.0, 1.0],
        [4.0, 1.0],
        [2.5, 2.5],  # Indented vertex making it concave
        [4.0, 4.0],
        [1.0, 4.0],
    ]
    is_valid, err, diag = validate_and_diagnose_profile(profile, video_width=920, video_height=680)
    assert is_valid is False
    assert "convex" in err.lower()


def test_rejection_of_collinear_correspondence():
    """Collinear correspondence points must be rejected as degenerate homography."""
    profile = create_default_profile(920, 680)
    # Set all image points in a horizontal line
    profile["image_points"] = [[100, 300], [200, 300], [300, 300], [400, 300]]
    is_valid, err, diag = validate_and_diagnose_profile(profile, video_width=920, video_height=680)
    assert is_valid is False
    assert err is not None


# ---------------------------------------------------------------------------
# 5. JSON Import & Export Round-Trip
# ---------------------------------------------------------------------------

def test_json_import_export_roundtrip():
    """Exporting a profile to JSON and importing it back must preserve all data."""
    profile = create_default_profile(1280, 720, name="Supermarket Aisle 4")
    json_str = export_profile_json(profile)
    assert isinstance(json_str, str)

    imported, err = import_profile_json(json_str, expected_width=1280, expected_height=720)
    assert err is None
    assert imported is not None
    assert imported["name"] == "Supermarket Aisle 4"
    assert imported["image_width"] == 1280
    assert imported["image_height"] == 720
    assert len(imported["zones"]) == 3
    assert imported["candidate_zone_ids"] == ["shelf_a", "shelf_b"]


def test_import_malformed_json():
    """Malformed JSON string must return error without unhandled exception."""
    imported, err = import_profile_json("{not-valid-json", expected_width=920, expected_height=680)
    assert imported is None
    assert "Malformed JSON" in err


# ---------------------------------------------------------------------------
# 6. Pipeline Execution Integration Test
# ---------------------------------------------------------------------------

def test_run_pipeline_with_custom_visual_profile(tmp_path):
    """Verify that a custom visual profile correctly executes through run_real_data_demo."""
    merl_video = "data/merl/10_1_crop.mp4"
    if not os.path.exists(merl_video):
        pytest.skip("MERL video not present in local data directory")

    import yaml
    from scripts.run_vscode_demo import run_real_data_demo

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    custom_profile = create_default_profile(920, 680, name="Custom Integration Test")
    out_dir = str(tmp_path / "custom_test_run")

    summary = run_real_data_demo(
        config,
        reports_dir=out_dir,
        max_frames=10,
        video_path=merl_video,
        artifact_prefix="test_custom",
        use_merl_profile=False,
        scene_profile=custom_profile,
    )

    assert summary is not None
    assert summary["dataset_name"] in {"merl_shopping_dataset", "user_uploaded_indoor_video"}
    assert summary["total_frames_evaluated"] == 10
    assert "generated_artifacts" in summary
    assert "dwell_analytics" in summary
    assert "scene_profile" in summary
    assert summary["scene_profile"]["name"] == "Custom Integration Test"

