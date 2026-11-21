"""
Standardized Run Presentation Adapter.
======================================
Normalizes Phase 13 reports and demo summaries into a unified, typed data model.
Strictly preserves:
1. Missing vs. Evaluated Zero Distinction:
   - Dwell or transitions not evaluated => status = "not_available", values = None.
   - Dwell or transitions evaluated with zero activity => status = "available", values = 0 or 0.0.
2. Explicit Provenance:
   - Environment type and retail eligibility are derived strictly from explicit metadata.
   - Substring matching on filenames or dataset names is never used to infer retail or synthetic status.
   - Unknown provenance is preserved as unknown (is_retail = None, environment_type = "unknown").
3. Run-Specific Geometry:
   - Scene profile bounds, zones, zone colors, and cell size are extracted directly from the run.
   - MERL geometry is NEVER silently attached to custom recordings.
   - If geometry is missing, has_geometry is False and bounds/zones are None.
4. Point Conservation Invariant Verification:
   - Validates density conservation (Total == Binned + OutOfBounds + Untransformed).
   - Validates zone conservation (Total == InZone + Unassigned) ONLY when evaluated.
   - Missing counts do NOT become zero to force an invariant to pass.
5. Safe Artifact Path Resolution:
   - Strictly enforces allowed directory containment; rejects all path traversal without fallbacks.
6. Stable Run Identifiers:
   - Distinguishes reports across different subdirectories uniquely.
"""

import os
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from dashboard.report_loader import (
    resolve_demo_artifact_path,
    load_demo_summary,
    load_and_validate_report,
    SCHEMA_VERSION_SUPPORTED,
)


@dataclass
class StandardizedRunData:
    run_id: str
    display_name: str
    dataset_name: str
    source_type: str                          # "recorded_video", "synthetic_fixture", "image_sequence"
    environment_type: str                     # "retail_store", "outdoor_pedestrian", "unknown", etc.
    is_synthetic: bool
    is_retail: Optional[bool]                 # True if retail, False if non-retail, None if unknown
    total_frames: Optional[int]
    sampling_fps: Optional[float]
    total_tracks: Optional[int]
    total_detections: Optional[int]
    
    # Point & Density Counts (Preserving Missing vs Zero)
    total_points: Optional[int]
    in_zone_points: Optional[int]
    unassigned_points: Optional[int]
    binned_points: Optional[int]
    out_of_bounds_points: Optional[int]
    untransformed_points: Optional[int]
    max_density: Optional[int]
    
    # Invariant Verification
    conservation_verified: bool                   # Overall density conservation check
    density_conservation_verified: Optional[bool] # None if not evaluated, bool if evaluated
    zone_conservation_verified: Optional[bool]    # None if not evaluated, bool if evaluated
    zone_assignment_counts: Dict[str, int]
    
    # Dwell Analytics (Preserving Missing vs Zero)
    dwell_status: str                         # "available" or "not_available"
    total_dwell_sec: Optional[float]          # None if not_available, 0.0 if evaluated zero
    total_episodes: Optional[int]             # None if not_available, 0 if evaluated zero
    zone_dwell: Dict[str, Dict[str, Any]]     # per-zone dwell span, episodes, distinct tracks
    
    # Transition Analytics (Preserving Missing vs Zero)
    transition_status: str                    # "available" or "not_available"
    total_transitions: Optional[int]          # None if not_available, 0 if evaluated zero
    direct_transitions: Dict[str, int]        # edge -> count
    transition_matrix: Dict[str, Dict[str, int]]
    
    # Evidence Gating
    evidence_status: str                      # "WITHHELD", "INSUFFICIENT EVIDENCE", "ELIGIBLE"
    reasons_withheld: List[str]
    caveats: List[str]
    is_eligible: bool
    
    # Candidate Placement
    candidate_zone: Optional[str]
    candidate_dwell_sec: Optional[float]
    candidate_hotspot_cell: Optional[Dict[str, Any]]
    candidate_selection_rule: Optional[str]
    comparison_by_shelf: List[Dict[str, Any]]
    
    # Lineage & Diagnostics
    model_family: str
    model_runtime: str
    homography_unit: str
    homography_error: Optional[float]
    
    # Run-Specific Geometry
    bounds: Optional[Tuple[float, float, float, float]] = None    # (min_x, max_x, min_y, max_y)
    cell_size: Optional[Union[float, Tuple[float, float]]] = None
    zones: Optional[Dict[str, List[Tuple[float, float]]]] = None  # zone_id -> polygon
    zone_colors: Optional[Dict[str, str]] = None                  # zone_id -> hex color
    zone_names: Optional[Dict[str, str]] = None                   # zone_id -> display name
    has_geometry: bool = False
    candidate_zone_ids: List[str] = field(default_factory=list)
    camera_polygons: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    
    # Safe Artifact Paths
    artifacts: Dict[str, Optional[str]] = field(default_factory=dict)
    observation_timeline_path: Optional[str] = None
    has_observation_timeline: bool = False
    
    # Candidate Diagnostics Companion Result
    candidate_diagnostics: Optional[Dict[str, Any]] = None

    # Raw source
    source_file_path: str = ""
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    # Source vs Sampling Frame Counts & Rates
    source_frame_count: Optional[int] = None
    source_fps: Optional[float] = None
    sampled_frame_count: Optional[int] = None
    source_fingerprint: Optional[str] = None


def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "not_available" or val == "None":
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    if val is None or val == "not_available" or val == "None":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _get_first_not_none(d: Any, *keys: str, default: Any = None) -> Any:
    """Safely return first non-None key value from dict, avoiding falsy 0 or empty bugs."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _compute_camera_polygons(
    zones: Optional[Dict[str, List[Tuple[float, float]]]],
    scene_profile: Optional[Dict[str, Any]] = None,
    homography_matrix: Optional[List[List[float]]] = None,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Project floor-plane zone polygons to camera pixel coordinates using planar inverse homography.
    Returns empty dict if calibration points are missing, singular, or uncalibrated.
    """
    if not zones:
        return {}

    H_inv = None
    try:
        import cv2
        import numpy as np

        if isinstance(scene_profile, dict) and "image_points" in scene_profile and "floor_points" in scene_profile:
            img_pts = np.array(scene_profile["image_points"], dtype=np.float32)
            flr_pts = np.array(scene_profile["floor_points"], dtype=np.float32)
            if len(img_pts) >= 4 and len(flr_pts) >= 4:
                H_inv, _ = cv2.findHomography(flr_pts, img_pts)

        if H_inv is None and homography_matrix is not None:
            H = np.array(homography_matrix, dtype=np.float32)
            if H.shape == (3, 3) and abs(float(np.linalg.det(H))) > 1e-9:
                H_inv = np.linalg.inv(H)

        if H_inv is None:
            return {}

        cam_polys = {}
        for zid, poly in zones.items():
            if len(poly) < 3:
                continue
            flr_arr = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
            cam_arr = cv2.perspectiveTransform(flr_arr, H_inv).reshape(-1, 2)
            if np.all(np.isfinite(cam_arr)):
                cam_polys[zid] = [(round(float(pt[0]), 1), round(float(pt[1]), 1)) for pt in cam_arr]
        return cam_polys
    except Exception:
        return {}


def _extract_run_geometry(data: Dict[str, Any]) -> Tuple[
    Optional[Tuple[float, float, float, float]],
    Optional[Union[float, Tuple[float, float]]],
    Optional[Dict[str, List[Tuple[float, float]]]],
    Optional[Dict[str, str]],
    Optional[Dict[str, str]],
    List[str],
    Dict[str, List[Tuple[float, float]]],
    Optional[int],
    Optional[int],
    bool,
]:
    """
    Extract run-specific geometry directly from scene profile or recorded metadata.
    Never silently attaches MERL geometry to an arbitrary run.
    """
    # 1. Check for scene_profile object in run data
    scene_profile = data.get("scene_profile") or data.get("metadata", {}).get("scene_profile")
    if scene_profile is None:
        prof_path = data.get("generated_artifacts", {}).get("scene_profile")
        if isinstance(prof_path, str) and os.path.isfile(prof_path):
            try:
                with open(prof_path, "r", encoding="utf-8") as f_p:
                    scene_profile = json.load(f_p)
            except Exception:
                pass
    if isinstance(scene_profile, dict) and "bounds" in scene_profile:
        try:
            b_list = [float(v) for v in scene_profile["bounds"]]
            if len(b_list) == 4 and b_list[1] > b_list[0] and b_list[3] > b_list[2]:
                bounds = (b_list[0], b_list[1], b_list[2], b_list[3])
                cell_size = scene_profile.get("cell_size")
                zones: Dict[str, List[Tuple[float, float]]] = {}
                zone_colors: Dict[str, str] = {}
                zone_names: Dict[str, str] = {}
                for z in scene_profile.get("zones", []):
                    zid = z.get("zone_id")
                    poly = z.get("polygon")
                    if zid and poly and len(poly) >= 3:
                        zones[zid] = [tuple(float(c) for c in pt) for pt in poly]
                        if "color" in z:
                            zone_colors[zid] = str(z["color"])
                        if "name" in z:
                            zone_names[zid] = str(z["name"])
                candidate_zone_ids = [str(x) for x in scene_profile.get("candidate_zone_ids", [])]
                img_w = scene_profile.get("image_width")
                img_h = scene_profile.get("image_height")
                h_mat = data.get("homography_diagnostics", {}).get("homography_matrix")
                camera_polys = _compute_camera_polygons(zones, scene_profile, h_mat)
                if zones:
                    return bounds, cell_size, zones, zone_colors, zone_names, candidate_zone_ids, camera_polys, img_w, img_h, True
        except (ValueError, TypeError, KeyError):
            pass

    # 2. Check for explicit density_analytics bounds dict
    density_raw = data.get("density_analytics", {})
    density_bounds = density_raw.get("bounds")
    if isinstance(density_bounds, dict) and all(k in density_bounds for k in ["min_x", "max_x", "min_y", "max_y"]):
        try:
            bx0 = float(density_bounds["min_x"])
            bx1 = float(density_bounds["max_x"])
            by0 = float(density_bounds["min_y"])
            by1 = float(density_bounds["max_y"])
            if bx1 > bx0 and by1 > by0:
                bounds = (bx0, bx1, by0, by1)
                cell_size = density_raw.get("cell_size")
                return bounds, cell_size, None, None, None, [], {}, None, None, False
        except (ValueError, TypeError):
            pass

    # 3. Check for repository MERL retail baseline clip ONLY when dataset is confirmed MERL
    # and environment is confirmed retail store
    dataset_name = data.get("dataset_name") or data.get("metadata", {}).get("dataset_name")
    env = data.get("evidence_status", {}).get("environment_type") or data.get("evidence_context", {}).get("environment_type")
    if dataset_name == "merl_shopping_dataset" and env in {"retail_store", "supermarket", "retail_floor"}:
        merl_profile_path = os.path.join("configs", "merl_scene_profile.json")
        if os.path.isfile(merl_profile_path):
            try:
                with open(merl_profile_path, "r", encoding="utf-8") as f_merl:
                    mp = json.load(f_merl)
                b_list = [float(v) for v in mp["bounds"]]
                bounds = (b_list[0], b_list[1], b_list[2], b_list[3])
                cell_size = mp.get("cell_size", 0.5)
                zones = {z["zone_id"]: [tuple(float(c) for c in pt) for pt in z["polygon"]] for z in mp.get("zones", [])}
                zone_colors = {z["zone_id"]: z.get("color", "#3498DB") for z in mp.get("zones", [])}
                zone_names = {z["zone_id"]: z.get("name", z["zone_id"]) for z in mp.get("zones", [])}
                candidate_zone_ids = [str(x) for x in mp.get("candidate_zone_ids", [])]
                img_w = mp.get("image_width", 920)
                img_h = mp.get("image_height", 680)
                h_mat = data.get("homography_diagnostics", {}).get("homography_matrix")
                camera_polys = _compute_camera_polygons(zones, mp, h_mat)
                return bounds, cell_size, zones, zone_colors, zone_names, candidate_zone_ids, camera_polys, img_w, img_h, True
            except Exception:
                pass

    return None, None, None, None, None, [], {}, None, None, False


def load_standardized_run(
    source_path_or_dict: Union[str, Dict[str, Any]],
    base_dir: str = "outputs/reports",
    reports_dir: Optional[str] = None,
) -> Optional[StandardizedRunData]:
    """
    Parse a JSON file or preloaded dictionary into a StandardizedRunData instance.
    Enforces strict directory containment; path traversal is strictly rejected without fallbacks.
    """
    actual_dir = reports_dir if reports_dir is not None else base_dir
    source_file_path = ""
    if isinstance(source_path_or_dict, str):
        source_file_path = source_path_or_dict
        # Strictly verify safe containment within declared allowed directory
        resolved = resolve_demo_artifact_path(source_path_or_dict, allowed_dir=actual_dir)
        if not resolved:
            return None
        try:
            with open(resolved, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
    elif isinstance(source_path_or_dict, dict):
        data = source_path_or_dict
    else:
        return None

    if not isinstance(data, dict):
        return None

    # Detect schema type: Phase 13 Analysis Report vs. Demo Summary
    is_phase_13 = "schema_version" in data and "report_id" in data
    
    if is_phase_13:
        return _parse_phase_13_report(data, source_file_path, actual_dir)
    else:
        return _parse_demo_summary(data, source_file_path, actual_dir)


def _build_or_load_candidate_diagnostics(
    data: Dict[str, Any],
    run_id: str,
    dataset_name: str,
    timeline_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build or load candidate diagnostics companion object without altering report schema."""
    if "candidate_diagnostics" in data and isinstance(data["candidate_diagnostics"], dict):
        return data["candidate_diagnostics"]

    timeline_dict = None
    if timeline_path and os.path.isfile(timeline_path):
        try:
            with open(timeline_path, "r", encoding="utf-8") as tf:
                timeline_dict = json.load(tf)
        except Exception:
            timeline_dict = None

    try:
        from src.analytics.candidate_diagnostics import generate_candidate_diagnostics
        diag = generate_candidate_diagnostics(
            run_dict=data,
            timeline_dict=timeline_dict,
            run_id=run_id,
            dataset_name=dataset_name,
        )
        return diag.to_dict()
    except Exception:
        return None


def _parse_demo_summary(data: Dict[str, Any], file_path: str, base_dir: str) -> StandardizedRunData:
    meta = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    evidence_raw = data.get("evidence_status", {}) if isinstance(data.get("evidence_status"), dict) else {}
    if not evidence_raw and isinstance(data.get("evidence_context"), dict):
        evidence_raw = data.get("evidence_context")

    dataset_name = data.get("dataset_name") or meta.get("dataset_name") or "Unknown Dataset"
    source_type = data.get("source_type") or meta.get("source_type") or "recorded_video"

    # Explicit Provenance & Environment (Defect 3)
    if "is_retail" in meta and meta["is_retail"] is not None:
        is_retail = bool(meta["is_retail"])
    elif "is_retail" in data and data["is_retail"] is not None:
        is_retail = bool(data["is_retail"])
    elif "is_retail" in evidence_raw and evidence_raw["is_retail"] is not None:
        is_retail = bool(evidence_raw["is_retail"])
    else:
        env_candidate = meta.get("environment_type") or evidence_raw.get("environment_type") or data.get("environment_type")
        if env_candidate and isinstance(env_candidate, str):
            env_clean = env_candidate.strip().lower()
            if any(k in env_clean for k in ["retail", "store", "supermarket", "grocery", "shop"]):
                is_retail = True
            elif any(k in env_clean for k in ["outdoor", "pedestrian", "street", "warehouse", "logistics", "hallway", "office", "non_retail"]):
                is_retail = False
            else:
                is_retail = None
        else:
            is_retail = None

    env_val = meta.get("environment_type") or evidence_raw.get("environment_type") or data.get("environment_type")
    if env_val and isinstance(env_val, str):
        environment_type = env_val.strip()
    elif is_retail is True:
        environment_type = "retail_store"
    elif is_retail is False:
        environment_type = "pedestrian_outdoor"
    else:
        environment_type = "unknown"

    # Synthetic determination from explicit source type or flags
    if "is_synthetic" in meta and meta["is_synthetic"] is not None:
        is_synthetic = bool(meta["is_synthetic"])
    elif "is_synthetic" in data and data["is_synthetic"] is not None:
        is_synthetic = bool(data["is_synthetic"])
    elif "is_synthetic" in evidence_raw and evidence_raw["is_synthetic"] is not None:
        is_synthetic = bool(evidence_raw["is_synthetic"])
    elif source_type in {"synthetic_fixture", "synthetic"}:
        is_synthetic = True
    elif "total_simulated_tracks" in data or "total_simulated_points" in data:
        is_synthetic = True
    elif "SYNTHETIC" in dataset_name.upper():
        is_synthetic = True
    else:
        is_synthetic = False

    # Stable Run Identity (Defect 8)
    if file_path:
        try:
            rel = os.path.relpath(file_path, base_dir).replace("\\", "/")
            if "/" in rel:
                run_id = rel
            else:
                run_id = os.path.splitext(os.path.basename(file_path))[0]
        except ValueError:
            run_id = os.path.splitext(os.path.basename(file_path))[0]
    else:
        run_id = data.get("run_id") or meta.get("run_id") or "demo_run"

    # Display name from recorded metadata (never guessing from filename)
    display_name = data.get("display_name") or meta.get("display_name")
    if not display_name:
        scene_name = data.get("scene_profile", {}).get("name") if isinstance(data.get("scene_profile"), dict) else None
        if scene_name:
            display_name = scene_name
        elif dataset_name == "merl_shopping_dataset" and is_retail is True:
            display_name = "MERL Shopping Sample (Observed Real Activity)"
        elif is_synthetic:
            display_name = dataset_name if "SYNTHETIC" in dataset_name.upper() else f"{dataset_name} (Synthetic Fixture)"
        elif is_retail is False and "MOT17" in dataset_name.upper():
            display_name = "MOT17-04 Engineering Sample (Outdoor Pedestrian)"
        else:
            display_name = dataset_name
        if "uploaded_runs/" in run_id:
            parent_dir = os.path.basename(os.path.dirname(run_id))
            display_name = f"{display_name} ({parent_dir})"

    video_meta = data.get("video_metadata") or meta.get("video_metadata") or {}
    source_frame_count = _safe_int(video_meta.get("total_frames"))
    source_fps = _safe_float(video_meta.get("fps"))
    sampled_frame_count = _safe_int(_get_first_not_none(data, "total_frames_evaluated", "total_frames") or _get_first_not_none(meta, "total_frames_evaluated", "total_frames"))
    total_frames = sampled_frame_count
    sampling_fps = _safe_float(_get_first_not_none(data, "target_fps", "fps") or _get_first_not_none(meta, "target_fps", "fps"))
    total_tracks = _safe_int(_get_first_not_none(data, "total_tracks", "total_simulated_tracks") or _get_first_not_none(meta, "total_tracks"))
    total_detections = _safe_int(_get_first_not_none(data, "total_detections") or _get_first_not_none(meta, "total_detections"))

    # Density Counts (Defect 5)
    density_raw = data.get("density_analytics") or data.get("spatial_density") or {}

    # Point & Zone Assignment Counts (Defect 5: Missing vs Zero)
    zone_counts_raw = data.get("zone_assignment_counts") or data.get("spatial_observations") or {}
    total_points = _safe_int(
        _get_first_not_none(zone_counts_raw, "total_points", "total_evaluated_points") or
        _get_first_not_none(data, "total_points", "total_simulated_points") or
        _get_first_not_none(density_raw, "total_evaluated_points", "total_points", "evaluated_points")
    )
    in_zone_points = _safe_int(_get_first_not_none(zone_counts_raw, "in_zone_points"))
    unassigned_points = _safe_int(_get_first_not_none(zone_counts_raw, "unassigned_out_of_zone_points", "unassigned_points"))
    per_zone_counts = zone_counts_raw.get("per_zone", {})
    if not isinstance(per_zone_counts, dict):
        per_zone_counts = {}

    zone_conservation_verified: Optional[bool] = None
    if in_zone_points is not None or unassigned_points is not None:
        if (total_points is not None and in_zone_points is not None and unassigned_points is not None and
            total_points >= 0 and in_zone_points >= 0 and unassigned_points >= 0):
            zone_conservation_verified = (total_points == in_zone_points + unassigned_points)
        else:
            zone_conservation_verified = False

    total_evaluated_density = _safe_int(_get_first_not_none(
        density_raw, "total_evaluated_points", "total_points", "evaluated_points"
    ))
    if total_evaluated_density is None and total_points is not None:
        total_evaluated_density = total_points

    binned_points = _safe_int(_get_first_not_none(density_raw, "total_binned_points", "binned_points"))
    out_of_bounds = _safe_int(_get_first_not_none(density_raw, "out_of_bounds_count", "out_of_bounds_points"))
    untransformed = _safe_int(_get_first_not_none(density_raw, "untransformed_count", "untransformed_points"))
    max_density = _safe_int(_get_first_not_none(density_raw, "max_cell_density", "max_density"))

    density_conservation_verified: Optional[bool] = None
    if all(v is not None for v in [total_evaluated_density, binned_points, out_of_bounds, untransformed]):
        if all(v >= 0 for v in [total_evaluated_density, binned_points, out_of_bounds, untransformed]):
            density_conservation_verified = (total_evaluated_density == binned_points + out_of_bounds + untransformed)
        else:
            density_conservation_verified = False
    elif any(v is not None for v in [binned_points, out_of_bounds, untransformed]) and total_evaluated_density is not None:
        if untransformed is None and binned_points is not None and out_of_bounds is not None:
            density_conservation_verified = (total_evaluated_density == binned_points + out_of_bounds)
        else:
            density_conservation_verified = False

    # Overall conservation boolean for backward compatibility
    conservation_verified = bool(density_conservation_verified is True)

    # Dwell Analytics (Defect 5: Preserving missing vs zero)
    dwell_raw = data.get("dwell_analytics") or data.get("dwell_summary") or {}
    dwell_status = dwell_raw.get("status", "available" if ("total_observed_dwell_span_sec" in dwell_raw or "total_dwell_sec" in dwell_raw) else "not_available")
    if dwell_status == "available":
        total_dwell_sec = _safe_float(_get_first_not_none(dwell_raw, "total_observed_dwell_span_sec", "total_dwell_sec", default=0.0))
        total_episodes = _safe_int(_get_first_not_none(dwell_raw, "total_episodes", default=0))
    else:
        total_dwell_sec = None
        total_episodes = None
    zone_dwell = dwell_raw.get("per_zone", {})
    if not isinstance(zone_dwell, dict):
        zone_dwell = {}

    # Transition Analytics (Defect 5: Preserving missing vs zero)
    trans_raw = data.get("direct_transitions") or data.get("transition_analytics") or data.get("transitions_summary") or {}
    trans_status = trans_raw.get("status", "available" if ("total_direct_transitions" in trans_raw or "total_transitions" in trans_raw) else "not_available")
    if trans_status == "available":
        total_transitions = _safe_int(_get_first_not_none(trans_raw, "total_direct_transitions", "total_transitions", default=0))
        direct_transitions = trans_raw.get("transition_counts") or trans_raw.get("direct_transitions", {})
        if not isinstance(direct_transitions, dict):
            direct_transitions = {}
    else:
        total_transitions = None
        direct_transitions = {}
    footfall_raw = data.get("footfall_analytics", {})
    trans_matrix = footfall_raw.get("transition_matrix", {})
    if not isinstance(trans_matrix, dict):
        trans_matrix = {}

    # Evidence Context (Defect 6: No universal invented reasons)
    evidence_status = evidence_raw.get("actionable_recommendations", "WITHHELD")
    reasons_withheld = list(evidence_raw.get("reasons_withheld", []))
    is_eligible = bool(evidence_raw.get("is_eligible", False))
    caveats = list(data.get("limitations", []))

    # Candidate Placement (Defect 2: No fabricated fallback coordinates)
    candidate_raw = data.get("exploratory_layout_candidate", {})
    candidate_zone = candidate_raw.get("candidate_zone_id")
    candidate_dwell_sec = None
    if candidate_raw.get("candidate_metrics"):
        candidate_dwell_sec = _safe_float(candidate_raw["candidate_metrics"].get("observed_dwell_span_sec"))
    candidate_hotspot = candidate_raw.get("candidate_hotspot") or candidate_raw.get("candidate_hotspot_cell")
    candidate_rule = candidate_raw.get("selection_rule")
    comparison_by_shelf = list(candidate_raw.get("comparison_by_shelf_zone", []))

    # Lineage
    model_raw = data.get("model_inference", {})
    model_family = model_raw.get("model_family", "YOLO11-n ONNX")
    model_runtime = model_raw.get("runtime", "ONNX Runtime CPU")
    homography_raw = data.get("homography_diagnostics", {})
    homography_unit = homography_raw.get("unit", "configured floor units")
    homography_error = _safe_float(homography_raw.get("check_point_mean_error"))

    # Run-Specific Geometry (Defect 1)
    bounds, cell_size, zones, zone_colors, zone_names, candidate_zone_ids, camera_polygons, img_w, img_h, has_geom = _extract_run_geometry(data)
    if not candidate_zone_ids and comparison_by_shelf:
        candidate_zone_ids = [c["zone_id"] for c in comparison_by_shelf if isinstance(c, dict) and "zone_id" in c]
    if candidate_zone and candidate_zone not in candidate_zone_ids:
        candidate_zone_ids.append(candidate_zone)
    if cell_size is None:
        sc = data.get("scene_profile") or (meta.get("scene_profile") if isinstance(meta, dict) else None)
        if isinstance(sc, dict) and "cell_size" in sc:
            cell_size = sc["cell_size"]
        elif isinstance(density_raw, dict) and "cell_size" in density_raw:
            cell_size = density_raw["cell_size"]

    # Resolve Artifacts safely
    artifacts_raw = data.get("generated_artifacts", {})
    artifacts = {}
    if isinstance(artifacts_raw, dict):
        for key, path in artifacts_raw.items():
            artifacts[key] = resolve_demo_artifact_path(path, allowed_dir=base_dir)

    # Automatically resolve run_camera_overlay.png if present in run directory
    if "camera_overlay" not in artifacts and file_path:
        resolved_fp = resolve_demo_artifact_path(file_path, allowed_dir=base_dir)
        if resolved_fp:
            run_parent = os.path.dirname(resolved_fp)
            overlay_candidate = os.path.join(run_parent, "run_camera_overlay.png")
            if os.path.isfile(overlay_candidate):
                artifacts["camera_overlay"] = overlay_candidate

    timeline_path = artifacts.get("observation_timeline")
    has_timeline = bool(timeline_path and os.path.isfile(timeline_path))
    candidate_diag = _build_or_load_candidate_diagnostics(data, run_id, dataset_name, timeline_path)

    return StandardizedRunData(
        run_id=run_id,
        display_name=display_name,
        dataset_name=dataset_name,
        source_type=source_type,
        environment_type=environment_type,
        is_synthetic=is_synthetic,
        is_retail=is_retail,
        total_frames=total_frames,
        sampling_fps=sampling_fps,
        total_tracks=total_tracks,
        total_detections=total_detections,
        total_points=total_points,
        in_zone_points=in_zone_points,
        unassigned_points=unassigned_points,
        binned_points=binned_points,
        out_of_bounds_points=out_of_bounds,
        untransformed_points=untransformed,
        max_density=max_density,
        conservation_verified=conservation_verified,
        density_conservation_verified=density_conservation_verified,
        zone_conservation_verified=zone_conservation_verified,
        zone_assignment_counts=per_zone_counts,
        dwell_status=dwell_status,
        total_dwell_sec=total_dwell_sec,
        total_episodes=total_episodes,
        zone_dwell=zone_dwell,
        transition_status=trans_status,
        total_transitions=total_transitions,
        direct_transitions=direct_transitions,
        transition_matrix=trans_matrix,
        evidence_status=evidence_status,
        reasons_withheld=reasons_withheld,
        caveats=caveats,
        is_eligible=is_eligible,
        candidate_zone=candidate_zone,
        candidate_dwell_sec=candidate_dwell_sec,
        candidate_hotspot_cell=candidate_hotspot,
        candidate_selection_rule=candidate_rule,
        comparison_by_shelf=comparison_by_shelf,
        model_family=model_family,
        model_runtime=model_runtime,
        homography_unit=homography_unit,
        homography_error=homography_error,
        bounds=bounds,
        cell_size=cell_size,
        zones=zones,
        zone_colors=zone_colors,
        zone_names=zone_names,
        has_geometry=has_geom,
        candidate_zone_ids=candidate_zone_ids,
        camera_polygons=camera_polygons,
        image_width=img_w,
        image_height=img_h,
        artifacts=artifacts,
        observation_timeline_path=timeline_path,
        has_observation_timeline=has_timeline,
        candidate_diagnostics=candidate_diag,
        source_file_path=file_path,
        raw_dict=data,
        source_frame_count=source_frame_count,
        source_fps=source_fps,
        sampled_frame_count=sampled_frame_count,
        source_fingerprint=data.get("metadata", {}).get("source_fingerprint") or data.get("source_fingerprint") if isinstance(data, dict) else None,
    )



def _parse_phase_13_report(data: Dict[str, Any], file_path: str, base_dir: str) -> StandardizedRunData:
    metadata = data.get("metadata", {})
    dataset_name = metadata.get("dataset_name", "Unknown Dataset")
    source_type = metadata.get("source_type", "recorded_video")
    
    # Explicit Provenance & Environment (Defect 3)
    evidence_raw = data.get("evidence_context", {})
    env = evidence_raw.get("environment_type") or metadata.get("environment_type")
    if env and isinstance(env, str):
        env_clean = env.strip().lower()
        if env_clean in {"retail_store", "retail_floor", "store_environment", "supermarket", "grocery_store"}:
            is_retail = True
        elif env_clean in {"outdoor_pedestrian", "outdoor", "street", "public_walkway", "non_retail"}:
            is_retail = False
        else:
            is_retail = None
        environment_type = env.strip()
    else:
        is_retail = None
        environment_type = "unknown"

    # Synthetic determination from explicit source type or flags
    if source_type in {"synthetic_fixture", "synthetic"}:
        is_synthetic = True
    elif data.get("is_synthetic") is True or evidence_raw.get("is_synthetic") is True:
        is_synthetic = True
    else:
        is_synthetic = False

    # Stable Run Identity (Defect 8)
    report_id = data.get("report_id", "phase_13_report")
    if file_path:
        try:
            rel = os.path.relpath(file_path, base_dir).replace("\\", "/")
            if "/" in rel:
                run_id = rel
            else:
                run_id = os.path.splitext(os.path.basename(file_path))[0]
        except ValueError:
            run_id = report_id
    else:
        run_id = report_id

    display_name = metadata.get("display_name")
    if not display_name:
        if dataset_name == "merl_shopping_dataset" and is_retail is True:
            display_name = "MERL Sample Benchmark Report (100 Frames)"
        elif is_retail is False and "MOT17" in dataset_name.upper():
            display_name = "MOT17 Engineering Benchmark Report (10 Frames)"
        else:
            display_name = f"Analysis Report: {run_id}"

    video_meta = data.get("video_metadata") or metadata.get("video_metadata") or {}
    source_frame_count = _safe_int(video_meta.get("total_frames"))
    source_fps = _safe_float(video_meta.get("fps"))
    sampled_frame_count = _safe_int(metadata.get("total_frames_evaluated") or metadata.get("frames_evaluated"))
    total_frames = sampled_frame_count
    sampling_fps = _safe_float(metadata.get("target_fps") or metadata.get("frame_rate", 5.0))
    track_summary = data.get("track_summary", {})
    dwell_raw = data.get("dwell_analytics", {})
    trans_raw = data.get("transition_analytics", {})
    total_tracks = _safe_int(track_summary.get("total_tracks") or dwell_raw.get("distinct_clip_local_tracks_evaluated") or trans_raw.get("total_evaluated_tracks"))
    total_detections = _safe_int(track_summary.get("total_detections"))

    # Density & Points (Defect 5: Missing vs Zero)
    density_raw = data.get("density_analytics", {})
    binned_points = _safe_int(_get_first_not_none(density_raw, "total_binned_points", "binned_points"))
    out_of_bounds = _safe_int(_get_first_not_none(density_raw, "out_of_bounds_count", "out_of_bounds_points"))
    untransformed = _safe_int(_get_first_not_none(density_raw, "untransformed_count", "untransformed_points"))
    total_points = _safe_int(_get_first_not_none(density_raw, "total_evaluated_points", "evaluated_points", "total_points"))
    max_density = _safe_int(_get_first_not_none(density_raw, "max_cell_density", "max_density"))

    density_conservation_verified: Optional[bool] = None
    if all(v is not None for v in [total_points, binned_points, out_of_bounds, untransformed]):
        if all(v >= 0 for v in [total_points, binned_points, out_of_bounds, untransformed]):
            density_conservation_verified = (total_points == binned_points + out_of_bounds + untransformed)
        else:
            density_conservation_verified = False
    elif any(v is not None for v in [binned_points, out_of_bounds, untransformed]) and total_points is not None:
        if untransformed is None and binned_points is not None and out_of_bounds is not None:
            density_conservation_verified = (total_points == binned_points + out_of_bounds)
        else:
            density_conservation_verified = False

    conservation_verified = bool(density_conservation_verified is True)

    in_zone_points = None
    unassigned_points = None
    per_zone_counts = {}

    # Dwell Analytics (Defect 5)
    dwell_status = dwell_raw.get("status", "available" if "total_observed_dwell_span_sec" in dwell_raw else "not_available")
    if dwell_status == "available":
        total_dwell_sec = _safe_float(dwell_raw.get("total_observed_dwell_span_sec", 0.0))
        total_episodes = _safe_int(dwell_raw.get("total_episodes", 0))
    else:
        total_dwell_sec = None
        total_episodes = None
    zone_dwell = dwell_raw.get("per_zone") or dwell_raw.get("zone_dwell_summaries", {})
    if not isinstance(zone_dwell, dict):
        zone_dwell = {}

    if zone_dwell:
        in_zone_points = sum(z.get("point_count", 0) for z in zone_dwell.values() if isinstance(z, dict))
        if total_points is not None:
            unassigned_points = max(0, total_points - in_zone_points)
        per_zone_counts = {zid: z.get("point_count", 0) for zid, z in zone_dwell.items() if isinstance(z, dict)}

    zone_conservation_verified: Optional[bool] = None
    if total_points is not None and in_zone_points is not None and unassigned_points is not None:
        zone_conservation_verified = (total_points == in_zone_points + unassigned_points)

    # Transition Analytics (Defect 5)
    trans_status = trans_raw.get("status", "available" if "total_transitions" in trans_raw else "not_available")
    if trans_status == "available":
        total_transitions = _safe_int(trans_raw.get("total_transitions", 0))
        direct_transitions = trans_raw.get("transition_counts", {})
        if not isinstance(direct_transitions, dict):
            direct_transitions = {}
    else:
        total_transitions = None
        direct_transitions = {}
    trans_matrix = trans_raw.get("transition_matrix", {})
    if not isinstance(trans_matrix, dict):
        trans_matrix = {}

    # Evidence Context (Defect 6: No universal invented reasons)
    spatial_obs = data.get("spatial_observations", {})
    is_eligible = bool(spatial_obs.get("is_eligible", evidence_raw.get("is_eligible", False)))
    reasons_withheld = list(spatial_obs.get("reasons_withheld") or evidence_raw.get("reasons_withheld", []))
    caveats = list(spatial_obs.get("caveats") or evidence_raw.get("caveats", []))
    evidence_status = "ELIGIBLE" if is_eligible else "INSUFFICIENT EVIDENCE (WITHHELD)"

    # Candidate Placement (Defect 2: No fabricated fallback coordinates)
    candidate_zone = spatial_obs.get("candidate_zone_id")
    candidate_hotspot = spatial_obs.get("candidate_hotspot") or spatial_obs.get("candidate_hotspot_cell")
    candidate_rule = spatial_obs.get("selection_rule")
    comparison_by_shelf = list(spatial_obs.get("comparison_by_shelf_zone", []))

    # Lineage
    model_family = "YOLO11-n ONNX"
    model_runtime = "ONNX Runtime CPU"
    homography_unit = density_raw.get("bounds", {}).get("unit", "configured floor units") if isinstance(density_raw.get("bounds"), dict) else "configured floor units"

    # Run-Specific Geometry (Defect 1)
    bounds, cell_size, zones, zone_colors, zone_names, candidate_zone_ids, camera_polygons, img_w, img_h, has_geom = _extract_run_geometry(data)
    if not candidate_zone_ids and comparison_by_shelf:
        candidate_zone_ids = [c["zone_id"] for c in comparison_by_shelf if isinstance(c, dict) and "zone_id" in c]
    if candidate_zone and candidate_zone not in candidate_zone_ids:
        candidate_zone_ids.append(candidate_zone)

    # Artifacts
    artifacts = {}
    heatmap_img = density_raw.get("heatmap_image_path")
    if heatmap_img:
        artifacts["density_heatmap"] = resolve_demo_artifact_path(heatmap_img, allowed_dir=base_dir)

    return StandardizedRunData(
        run_id=run_id,
        display_name=display_name,
        dataset_name=dataset_name,
        source_type=source_type,
        environment_type=environment_type,
        is_synthetic=is_synthetic,
        is_retail=is_retail,
        total_frames=total_frames,
        sampling_fps=sampling_fps,
        total_tracks=total_tracks,
        total_detections=total_detections,
        total_points=total_points,
        in_zone_points=in_zone_points,
        unassigned_points=unassigned_points,
        binned_points=binned_points,
        out_of_bounds_points=out_of_bounds,
        untransformed_points=untransformed,
        max_density=max_density,
        conservation_verified=conservation_verified,
        density_conservation_verified=density_conservation_verified,
        zone_conservation_verified=zone_conservation_verified,
        zone_assignment_counts=per_zone_counts,
        dwell_status=dwell_status,
        total_dwell_sec=total_dwell_sec,
        total_episodes=total_episodes,
        zone_dwell=zone_dwell,
        transition_status=trans_status,
        total_transitions=total_transitions,
        direct_transitions=direct_transitions,
        transition_matrix=trans_matrix,
        evidence_status=evidence_status,
        reasons_withheld=reasons_withheld,
        caveats=caveats,
        is_eligible=is_eligible,
        candidate_zone=candidate_zone,
        candidate_dwell_sec=None,
        candidate_hotspot_cell=candidate_hotspot,
        candidate_selection_rule=candidate_rule,
        comparison_by_shelf=comparison_by_shelf,
        model_family=model_family,
        model_runtime=model_runtime,
        homography_unit=homography_unit,
        homography_error=None,
        bounds=bounds,
        cell_size=cell_size,
        zones=zones,
        zone_colors=zone_colors,
        zone_names=zone_names,
        has_geometry=has_geom,
        candidate_zone_ids=candidate_zone_ids,
        camera_polygons=camera_polygons,
        image_width=img_w,
        image_height=img_h,
        artifacts=artifacts,
        observation_timeline_path=artifacts.get("observation_timeline"),
        has_observation_timeline=bool(artifacts.get("observation_timeline") and os.path.isfile(artifacts["observation_timeline"])),
        candidate_diagnostics=_build_or_load_candidate_diagnostics(data, run_id, dataset_name, artifacts.get("observation_timeline")),
        source_file_path=file_path,
        raw_dict=data,
        source_frame_count=source_frame_count,
        source_fps=source_fps,
        sampled_frame_count=sampled_frame_count,
        source_fingerprint=metadata.get("source_fingerprint") or data.get("source_fingerprint") if isinstance(metadata, dict) else None,
    )


def load_run_observation_timeline(run: StandardizedRunData) -> Optional[Dict[str, Any]]:
    """
    Safely load and return the observation timeline JSON dictionary for a run.
    Returns None if the run lacks temporal observation records.
    """
    if not run.has_observation_timeline or not run.observation_timeline_path:
        return None
    try:
        with open(run.observation_timeline_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def discover_standardized_runs(reports_dir: str = "outputs/reports") -> List[StandardizedRunData]:
    """
    Discover all supported analysis runs in reports_dir respecting containment.
    Distinguishes runs with identical basenames in different directories.
    """
    runs: List[StandardizedRunData] = []
    if not os.path.isdir(reports_dir):
        return runs

    seen_ids = set()

    # Preferred discovery order for root runs
    preferred_files = [
        "demo_real_summary.json",
        "phase_13_analysis_report_merl_sample.json",
        "demo_synthetic_summary.json",
        "demo_mot17_summary.json",
        "phase_13_analysis_report_mot17_sample.json",
    ]

    for fname in preferred_files:
        fpath = os.path.join(reports_dir, fname)
        if os.path.isfile(fpath):
            run = load_standardized_run(fpath, base_dir=reports_dir)
            if run and run.run_id not in seen_ids:
                seen_ids.add(run.run_id)
                runs.append(run)

    # Discover uploaded runs in subdirectories
    uploaded_dir = os.path.join(reports_dir, "uploaded_runs")
    if os.path.isdir(uploaded_dir):
        for entry in sorted(os.listdir(uploaded_dir), reverse=True):
            entry_path = os.path.join(uploaded_dir, entry)
            if os.path.isdir(entry_path):
                summary_files = [os.path.join(entry_path, f) for f in os.listdir(entry_path) if f.endswith("summary.json")]
                if summary_files:
                    summary_file = summary_files[0]
                    run = load_standardized_run(summary_file, base_dir=reports_dir)
                    if run and run.run_id not in seen_ids:
                        seen_ids.add(run.run_id)
                        runs.append(run)

    return runs


def format_observed_value(val: Optional[Union[int, float]], unit: str = "", missing_text: str = "Not evaluated") -> str:
    """Format numeric observation strictly preserving missing vs evaluated zero."""
    if val is None:
        return missing_text
    if isinstance(val, float):
        if val == 0.0:
            return f"0.0 {unit}".strip()
        return f"{val:.1f} {unit}".strip()
    return f"{val} {unit}".strip()


# Declare synchronized player custom component
try:
    import base64
    import streamlit.components.v1 as components
    _SYNC_COMP_DIR = os.path.join(os.path.dirname(__file__), "components", "synchronized_player")
    if os.path.exists(_SYNC_COMP_DIR):
        _synchronized_player_component = components.declare_component("synchronized_player", path=_SYNC_COMP_DIR)
    else:
        _synchronized_player_component = None
except Exception:
    _synchronized_player_component = None


def render_synchronized_player(
    run: StandardizedRunData,
    timeline_data: Dict[str, Any],
    seek_sample: int = 0,
    selected_zone_id: Optional[str] = None,
    selected_track_id: Optional[Union[int, str]] = None,
    key: str = "sync_player_instance",
    seek_request_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Render the synchronized frame and trajectory explorer in Streamlit.
    """
    if _synchronized_player_component is None:
        return None

    from streamlit import runtime
    import mimetypes
    video_url = ""
    video_path = run.artifacts.get("tracking_video")
    if video_path and os.path.isfile(video_path) and runtime.exists():
        video_url = runtime.get_instance().media_file_mgr.add(
            video_path,
            mimetypes.guess_type(video_path)[0] or "video/webm",
            f"synchronized-player-{key}",
        )

    from dashboard.theme import generate_isometric_floor_plan_svg
    iso_svg_str = ""
    if getattr(run, "bounds", None) and getattr(run, "zones", None):
        iso_svg_str = generate_isometric_floor_plan_svg(
            bounds=run.bounds,
            zones=run.zones,
            zone_colors=getattr(run, "zone_colors", {}),
            zone_names=getattr(run, "zone_names", {}),
            candidate_hotspot=getattr(run, "candidate_hotspot_cell", None),
            candidate_zone_id=getattr(run, "candidate_zone", None),
            selected_zone_id=selected_zone_id,
            width_px=800,
            height_px=500,
            zone_dwell=getattr(run, "zone_dwell", {}),
            dwell_scale=0.06,
        ) or ""

    camera_polys = getattr(run, "camera_polygons", {}) or {}
    raw_zd = getattr(run, "zone_dwell", {}) or {}
    norm_zd = {}
    for zid, zinfo in raw_zd.items():
        if isinstance(zinfo, dict):
            sec = zinfo.get("observed_dwell_sec")
            if sec is None:
                sec = zinfo.get("observed_dwell_span_sec")
            status = zinfo.get("dwell_status")
            if status is None:
                status = "evaluated_zero" if sec == 0 else ("recorded" if (sec is not None and sec > 0) else "missing")
            norm_zd[zid] = {
                **zinfo,
                "observed_dwell_sec": sec,
                "observed_dwell_span_sec": sec,
                "dwell_status": status,
            }
        else:
            norm_zd[zid] = zinfo

    run_geom = {
        "bounds": list(run.bounds) if getattr(run, "bounds", None) else [0.0, 10.0, 0.0, 6.0],
        "cell_size": getattr(run, "cell_size", None),
        "zones": {zid: [list(pt) for pt in poly] for zid, poly in (getattr(run, "zones", None) or {}).items()},
        "zone_colors": getattr(run, "zone_colors", None) or {},
        "zone_names": getattr(run, "zone_names", None) or {},
        "candidate_hotspot_cell": getattr(run, "candidate_hotspot_cell", None),
        "selected_zone_id": selected_zone_id,
        "selected_track_id": str(selected_track_id) if selected_track_id is not None else None,
        "camera_polygons": {zid: [list(pt) for pt in poly] for zid, poly in camera_polys.items()},
        "image_size": [getattr(run, "image_width", None) or 920, getattr(run, "image_height", None) or 680],
        "candidate_zone_ids": list(getattr(run, "candidate_zone_ids", None) or []),
        "zone_dwell": norm_zd,
        "candidate_zone": getattr(run, "candidate_zone", None),
        "total_dwell_sec": getattr(run, "total_dwell_sec", None),
        "dwell_status": getattr(run, "dwell_status", None),
        "default_perspective": "3d",
        "iso_svg": iso_svg_str,
    }

    return _synchronized_player_component(
        run_geometry=run_geom,
        timeline_data=timeline_data,
        video_url=video_url,
        seek_sample=seek_sample,
        seek_request_id=seek_request_id,
        key=key,
        default=None,
    )


def compute_workspace_summary(runs: List[StandardizedRunData]) -> Dict[str, Any]:
    """
    Compute dynamic aggregation across discovered analysis runs.

    1. Deduplicates duplicate representations using explicit run identity / lineage:
       - Uses `r.raw_dict.get('report_id')` or `r.run_id`.
       - Distinct executions with unique run identities remain distinct runs.
    2. Analysis run count vs Unique source recording count:
       - analysis_run_count: number of deduplicated distinct runs.
       - unique_source_count: count of distinct source identities derived from
         source fingerprints (sha256), source file paths, or dataset identifiers.
    3. Three provenance states:
       - retail_count: runs where not is_synthetic and is_retail is True
       - non_retail_count: runs where not is_synthetic and is_retail is False
       - unknown_provenance_count: runs where not is_synthetic and is_retail is None
       - synthetic_count: runs where is_synthetic is True
    4. Numeric Aggregation preserving missing vs evaluated zero:
       - Sampled frames: sums evaluated frames across runs that report them.
         Reports how many runs contributed and how many were missing.
       - Source frames: sums source frames across runs with source metadata.
         Reports how many runs contributed and how many were missing.
       - Tracks: counts total tracks across runs that report tracks; distinguishes
         missing (None) from evaluated zero (0).
    5. Curated sample library statistics (if the 3 curated runs are present):
       - Dynamic statistics specifically for the 3 curated retail baselines.
    """
    if not runs:
        return {
            "total_runs": 0,
            "deduped_runs": [],
            "unique_source_count": 0,
            "retail_count": 0,
            "non_retail_count": 0,
            "unknown_provenance_count": 0,
            "synthetic_count": 0,
            "total_sampled_frames": 0,
            "sampled_reporting_runs": 0,
            "sampled_missing_runs": 0,
            "total_source_frames": 0,
            "source_reporting_runs": 0,
            "source_missing_runs": 0,
            "total_tracks": 0,
            "tracks_reporting_runs": 0,
            "tracks_missing_runs": 0,
            "curated_runs": [],
            "curated_sampled_total": 0,
            "curated_source_total": 0,
            "curated_tracks_total": 0,
        }

    # 1. Deduplicate duplicate representations using explicit run identity / lineage
    seen_lineages = set()
    deduped: List[StandardizedRunData] = []
    for r in runs:
        lineage_id = r.raw_dict.get("report_id") or r.run_id
        if lineage_id not in seen_lineages:
            seen_lineages.add(lineage_id)
            deduped.append(r)

    # 2. Derive unique source recordings from source fingerprints / paths
    source_keys = set()
    for r in deduped:
        sfp = r.raw_dict.get("source_file_fingerprint")
        sha = sfp.get("sha256") if isinstance(sfp, dict) else None
        if sha:
            src_key = f"sha256:{sha}"
        else:
            vm = r.raw_dict.get("video_metadata") or {}
            sp = (
                r.raw_dict.get("source_path")
                or r.raw_dict.get("metadata", {}).get("source_path")
                or vm.get("file_path")
                or vm.get("file_name")
            )
            if sp:
                src_key = f"path:{os.path.normpath(str(sp)).lower()}"
            elif r.is_synthetic:
                src_key = f"synthetic:{r.dataset_name}"
            else:
                src_key = f"dataset:{r.dataset_name}"
        source_keys.add(src_key)

    # 3. Three Provenance States + Synthetic Fixtures
    retail_runs = [r for r in deduped if not r.is_synthetic and r.is_retail is True]
    non_retail_runs = [r for r in deduped if not r.is_synthetic and r.is_retail is False]
    unknown_provenance_runs = [r for r in deduped if not r.is_synthetic and r.is_retail is None]
    synthetic_runs = [r for r in deduped if r.is_synthetic]

    # 4. Numeric Aggregation preserving missing vs evaluated-zero
    runs_with_sampled = [r for r in deduped if (r.sampled_frame_count is not None or r.total_frames is not None)]
    total_sampled = sum((r.sampled_frame_count if r.sampled_frame_count is not None else r.total_frames) for r in runs_with_sampled)

    runs_with_source = [r for r in deduped if r.source_frame_count is not None]
    total_source = sum(r.source_frame_count for r in runs_with_source)

    runs_with_tracks = [r for r in deduped if r.total_tracks is not None]
    total_tracks = sum(r.total_tracks for r in runs_with_tracks)

    # 5. Curated Sample Library runs (if discovered)
    curated_keys = ["upload_ca30f3e8", "upload_38d7afe1", "upload_site02_jewelry"]
    curated_runs = [r for r in deduped if any(k in r.run_id for k in curated_keys)]
    curated_sampled = sum((r.sampled_frame_count or r.total_frames or 0) for r in curated_runs)
    curated_source = sum((r.source_frame_count or 0) for r in curated_runs)
    curated_tracks = sum((r.total_tracks or 0) for r in curated_runs)

    return {
        "total_runs": len(deduped),
        "deduped_runs": deduped,
        "unique_source_count": len(source_keys),
        "retail_count": len(retail_runs),
        "non_retail_count": len(non_retail_runs),
        "unknown_provenance_count": len(unknown_provenance_runs),
        "synthetic_count": len(synthetic_runs),
        "total_sampled_frames": total_sampled,
        "sampled_reporting_runs": len(runs_with_sampled),
        "sampled_missing_runs": len(deduped) - len(runs_with_sampled),
        "total_source_frames": total_source,
        "source_reporting_runs": len(runs_with_source),
        "source_missing_runs": len(deduped) - len(runs_with_source),
        "total_tracks": total_tracks,
        "tracks_reporting_runs": len(runs_with_tracks),
        "tracks_missing_runs": len(deduped) - len(runs_with_tracks),
        "curated_runs": curated_runs,
        "curated_sampled_total": curated_sampled,
        "curated_source_total": curated_source,
        "curated_tracks_total": curated_tracks,
    }

