"""
Visual Single-Camera Setup Module for Direction A New Analysis.
==============================================================
Provides:
1. Video frame extraction and representative sampling using OpenCV.
2. Mathematically exact coordinate conversion under resizing, letterboxing, and aspect ratios.
3. Profile construction, structural validation, and homography diagnostics.
4. Bidirectional state synchronization between Streamlit and interactive SVG/HTML editor.
5. JSON profile import and export with strict dimension and geometry verification.
"""

import os
import json
import math
import base64
import tempfile
from typing import Dict, Any, List, Tuple, Optional, Union

import cv2
import numpy as np

from src.spatial.scene_profile import validate_scene_profile, scene_config
from src.spatial import HomographyCalibrator


def screen_to_original_coords(
    screen_x: float,
    screen_y: float,
    container_w: float,
    container_h: float,
    orig_w: float,
    orig_h: float,
) -> Tuple[float, float]:
    """
    Convert display/screen coordinates to original decoded pixel coordinates,
    accounting for aspect ratio preservation, letterboxing, and scale.
    """
    if container_w <= 0 or container_h <= 0 or orig_w <= 0 or orig_h <= 0:
        raise ValueError("Dimensions must be positive finite numbers.")

    scale = min(container_w / orig_w, container_h / orig_h)
    rendered_w = orig_w * scale
    rendered_h = orig_h * scale

    # Letterbox / pillarbox padding offsets
    offset_x = (container_w - rendered_w) / 2.0
    offset_y = (container_h - rendered_h) / 2.0

    rel_x = screen_x - offset_x
    rel_y = screen_y - offset_y

    orig_x = rel_x / scale
    orig_y = rel_y / scale

    # Clamp to original pixel boundaries
    orig_x = max(0.0, min(float(orig_w - 1), orig_x))
    orig_y = max(0.0, min(float(orig_h - 1), orig_y))

    return round(orig_x, 2), round(orig_y, 2)


def original_to_screen_coords(
    orig_x: float,
    orig_y: float,
    container_w: float,
    container_h: float,
    orig_w: float,
    orig_h: float,
) -> Tuple[float, float]:
    """
    Convert original decoded pixel coordinates to display container coordinates.
    """
    if container_w <= 0 or container_h <= 0 or orig_w <= 0 or orig_h <= 0:
        raise ValueError("Dimensions must be positive finite numbers.")

    scale = min(container_w / orig_w, container_h / orig_h)
    rendered_w = orig_w * scale
    rendered_h = orig_h * scale

    offset_x = (container_w - rendered_w) / 2.0
    offset_y = (container_h - rendered_h) / 2.0

    screen_x = offset_x + orig_x * scale
    screen_y = offset_y + orig_y * scale

    return round(screen_x, 2), round(screen_y, 2)


def extract_representative_frame(
    video_source: Union[str, bytes],
    frame_idx: int = 0,
    max_dimension: int = 1280,
) -> Dict[str, Any]:
    """
    Extract a representative frame from a video file path or raw bytes using OpenCV.
    Returns metadata and base64-encoded JPEG image for visual canvas display.
    """
    temp_path = None
    if isinstance(video_source, bytes):
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        with os.fdopen(fd, "wb") as f:
            f.write(video_source)
        actual_path = temp_path
    else:
        actual_path = video_source

    if not os.path.isfile(actual_path):
        raise FileNotFoundError(f"Video file not found: {actual_path}")

    cap = cv2.VideoCapture(actual_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream: {actual_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

        target_idx = max(0, min(total_frames - 1, frame_idx)) if total_frames > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            # Fallback to reading first available frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            target_idx = 0

        if not ret or frame is None:
            raise ValueError(f"Failed to decode video frame at index {frame_idx}")

        timestamp_sec = round(target_idx / fps, 3) if fps > 0 else 0.0

        # Encode representative frame to base64 JPEG
        ret_enc, enc_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret_enc:
            raise ValueError("Failed to encode representative frame to JPEG")

        b64_str = base64.b64encode(enc_buf).decode("ascii")

        return {
            "width": orig_w,
            "height": orig_h,
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "frame_idx": target_idx,
            "timestamp_sec": timestamp_sec,
            "jpeg_base64": b64_str,
            "has_frame": True,
        }
    finally:
        cap.release()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def create_default_profile(
    width: int,
    height: int,
    name: str = "Custom Retail Camera View",
) -> Dict[str, Any]:
    """
    Create a safe, structurally valid initial scene profile scaled to the video dimensions.
    """
    w, h = float(width), float(height)

    # 4 perspective ground plane corners proportional to image size
    image_points = [
        [round(0.16 * w, 1), round(0.88 * h, 1)],  # P1: Bottom-Left
        [round(0.82 * w, 1), round(0.88 * h, 1)],  # P2: Bottom-Right
        [round(0.76 * w, 1), round(0.51 * h, 1)],  # P3: Top-Right
        [round(0.22 * w, 1), round(0.51 * h, 1)],  # P4: Top-Left
    ]

    # Matching floor points in configured units
    floor_points = [
        [0.0, 6.0],   # P1 -> Bottom-Left of floor
        [10.0, 6.0],  # P2 -> Bottom-Right of floor
        [10.0, 0.0],  # P3 -> Top-Right of floor
        [0.0, 0.0],   # P4 -> Top-Left of floor
    ]

    bounds = [0.0, 10.0, 0.0, 6.0]

    zones = [
        {
            "zone_id": "shelf_a",
            "name": "Shelf A (Snacks)",
            "polygon": [[0.5, 1.0], [4.0, 1.0], [4.0, 5.0], [0.5, 5.0]],
            "color": "#3498DB",
        },
        {
            "zone_id": "shelf_b",
            "name": "Shelf B (Produce)",
            "polygon": [[6.0, 1.0], [9.5, 1.0], [9.5, 5.0], [6.0, 5.0]],
            "color": "#2ECC71",
        },
        {
            "zone_id": "main_aisle",
            "name": "Main Aisle",
            "polygon": [[4.0, 1.0], [6.0, 1.0], [6.0, 5.0], [4.0, 5.0]],
            "color": "#F39C12",
        },
    ]

    return {
        "name": name,
        "image_width": int(width),
        "image_height": int(height),
        "image_points": image_points,
        "floor_points": floor_points,
        "bounds": bounds,
        "cell_size": 0.5,
        "hysteresis_window": 3,
        "zones": zones,
        "candidate_zone_ids": ["shelf_a", "shelf_b"],
    }


def validate_and_diagnose_profile(
    profile: Dict[str, Any],
    video_width: Optional[int] = None,
    video_height: Optional[int] = None,
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate profile against strict schema and compute homography condition diagnostics.
    """
    try:
        validated = validate_scene_profile(profile, width=video_width, height=video_height)
        
        # Compute Homography diagnostics via SVD
        src = validated["image_points"]
        dst = validated["floor_points"]
        calibrator = HomographyCalibrator(
            src,
            dst,
            unit="configured_units",
            dataset_name=validated.get("name", "custom_scene"),
        )
        diag = calibrator.diagnostics.to_dict()
        diag["is_valid"] = True

        # SVD condition number of 3x3 homography matrix
        H = np.array(calibrator.H, dtype=np.float64)
        _, s, _ = np.linalg.svd(H)
        cond_num = float(s[0] / s[-1]) if s[-1] > 1e-12 else float("inf")
        diag["condition_number"] = round(cond_num, 2)
        diag["mean_reprojection_error"] = diag.get("training_reprojection_mean_error", 0.0)

        return True, None, diag
    except Exception as exc:
        return False, str(exc), None


def export_profile_json(profile: Dict[str, Any]) -> str:
    """Export profile as a formatted JSON string."""
    return json.dumps(profile, indent=2)


def import_profile_json(
    json_str: str,
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse and validate imported JSON profile.
    Rejects incompatible dimensions or broken polygon geometry.
    """
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return None, "Imported file must be a JSON object"
        
        validated = validate_scene_profile(data, width=expected_width, height=expected_height)
        return validated, None
    except json.JSONDecodeError as exc:
        return None, f"Malformed JSON syntax: {exc}"
    except Exception as exc:
        return None, f"Profile validation error: {exc}"


# Declare custom component
try:
    import streamlit.components.v1 as components
    _COMPONENT_DIR = os.path.join(os.path.dirname(__file__), "components", "visual_editor")
    if os.path.exists(_COMPONENT_DIR):
        _visual_editor_component = components.declare_component("visual_editor", path=_COMPONENT_DIR)
    else:
        _visual_editor_component = None
except Exception:
    _visual_editor_component = None


def render_visual_editor(
    profile: Dict[str, Any],
    frame_info: Optional[Dict[str, Any]] = None,
    mode: str = "camera",
    selected_zone_id: Optional[str] = None,
    key: str = "visual_editor_canvas",
) -> Optional[Dict[str, Any]]:
    """
    Render the visual calibration editor component in Streamlit.
    Returns any user updates dispatched via streamlit:setComponentValue.
    """
    if _visual_editor_component is None:
        return None

    jpeg_b64 = frame_info.get("jpeg_base64", "") if frame_info else ""
    img_w = frame_info.get("width", profile.get("image_width", 920)) if frame_info else profile.get("image_width", 920)
    img_h = frame_info.get("height", profile.get("image_height", 680)) if frame_info else profile.get("image_height", 680)

    return _visual_editor_component(
        profile=profile,
        jpeg_base64=jpeg_b64,
        image_width=int(img_w),
        image_height=int(img_h),
        mode=mode,
        selected_zone_id=selected_zone_id,
        key=key,
        default=None,
    )

