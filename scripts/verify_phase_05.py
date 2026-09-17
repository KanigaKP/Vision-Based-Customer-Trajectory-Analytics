"""
Phase 5 Verification & Benchmark Script
========================================
Executes full trajectory extraction and homography floor-plane mapping pipeline:
YOLO11 Person Detector -> ByteTrack -> Trajectory Extraction -> Homography Transformation -> Top-Down Floor Map.
Generates visualization artifacts in outputs/reports/.
"""

import os
import sys
import time
import json
import numpy as np
import cv2
import yaml

sys.path.insert(0, os.path.abspath("."))

from src.video import FrameSampler
from src.detection import YOLO11PersonDetector
from src.tracking import ByteTracker, Track
from src.trajectory import TrajectoryExtractor, Trajectory
from src.spatial import (
    HomographyCalibrator,
    visualize_correspondences,
    visualize_floor_trajectories
)


def main():
    print("=" * 70)
    print("PHASE 5 — TRAJECTORY EXTRACTION & HOMOGRAPHY FLOOR MAPPING SPIKE")
    print("=" * 70)

    # 1. Load Configuration
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    det_cfg = config.get("detection", {})
    track_cfg = config.get("tracking", {})

    reports_dir = os.path.join("outputs", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # ============================================================
    # 2. MOT17-04 Sequence Test & Calibration
    # ============================================================
    print("\n[1/4] Calibrating & Testing MOT17-04 Homography...")

    mot17_calibrator = HomographyCalibrator.from_config(config, dataset_name="mot17_04")
    diag_mot17 = mot17_calibrator.diagnostics
    
    print(f"  Dataset Name              : {diag_mot17.dataset_name}")
    print(f"  Coordinate System Unit    : {diag_mot17.unit}")
    print(f"  Training Calibration Pts  : {diag_mot17.num_calibration_points}")
    print(f"  Training Reprojection Error: Mean={diag_mot17.training_reprojection_mean_error:.4f}m, Max={diag_mot17.training_reprojection_max_error:.4f}m")
    if diag_mot17.num_check_points > 0:
        print(f"  Check-Point Error (Check): Mean={diag_mot17.check_point_mean_error:.4f}m, Max={diag_mot17.check_point_max_error:.4f}m")

    # Run Detection + Tracking + Trajectory Extraction on 10 MOT17 frames
    mot17_dir = os.path.join("data", "mot17", "train", "MOT17-04-FRCNN", "img1")
    mot17_frames = sorted([f for f in os.listdir(mot17_dir) if f.endswith(".jpg")])[:10]

    Track.reset_id_counter(0)
    detector = YOLO11PersonDetector(
        model_path=det_cfg.get("model_path", "data/models/yolo11n.onnx"),
        conf_threshold=det_cfg.get("conf_threshold", 0.40)
    )
    tracker = ByteTracker(
        track_high_thresh=track_cfg.get("track_high_thresh", 0.50),
        max_time_lost_sec=track_cfg.get("max_time_lost_sec", 2.0)
    )

    t0_pipeline = time.time()
    for idx, fname in enumerate(mot17_frames, start=1):
        fpath = os.path.join(mot17_dir, fname)
        img = cv2.imread(fpath)
        timestamp_sec = round((idx - 1) / 30.0, 3)

        det_res = detector.detect(img, frame_id=idx, timestamp_sec=timestamp_sec)
        tracker.update(det_res.detections, frame_id=idx, timestamp_sec=timestamp_sec)

    # Extract all trajectories (active + lost + removed)
    all_mot17_tracks = tracker.tracked_tracks + tracker.lost_tracks + tracker.removed_tracks
    mot17_trajectories = TrajectoryExtractor.extract_from_tracks(all_mot17_tracks)

    # Transform trajectories using Homography
    t0_hom = time.time()
    transformed_mot17_trajectories = mot17_calibrator.transform_trajectories(mot17_trajectories)
    t_hom_ms = (time.time() - t0_hom) * 1000.0

    print(f"  Extracted & Transformed   : {len(transformed_mot17_trajectories)} Trajectories")
    print(f"  Homography Latency        : {t_hom_ms:.3f} ms for {len(transformed_mot17_trajectories)} trajectories ({t_hom_ms/max(1, len(transformed_mot17_trajectories)):.4f} ms/traj)")

    # Save Correspondence Visualization
    img_first = cv2.imread(os.path.join(mot17_dir, mot17_frames[0]))
    corresp_img = visualize_correspondences(img_first, mot17_calibrator)
    out_corresp_path = os.path.join(reports_dir, "phase_05_mot17_calibration_correspondences.jpg")
    cv2.imwrite(out_corresp_path, corresp_img)
    print(f"  Saved Calibration Image   : {out_corresp_path}")

    # Save Top-Down Floor Map Visualization
    floormap_img = visualize_floor_trajectories(
        transformed_mot17_trajectories,
        grid_size=(600, 600),
        floor_bounds=(-2.0, 12.0, -2.0, 12.0),
        title="MOT17-04 Top-Down Floor Map Trajectories (Metres)"
    )
    out_floormap_path = os.path.join(reports_dir, "phase_05_mot17_floormap_trajectories.jpg")
    cv2.imwrite(out_floormap_path, floormap_img)
    print(f"  Saved Top-Down Floor Map  : {out_floormap_path}")

    # ============================================================
    # 3. MERL Retail Qualitative Demonstration
    # ============================================================
    print("\n[2/4] Calibrating & Testing MERL Retail Video Homography...")

    merl_calibrator = HomographyCalibrator.from_config(config, dataset_name="merl")
    diag_merl = merl_calibrator.diagnostics

    print(f"  Dataset Name              : {diag_merl.dataset_name}")
    print(f"  Coordinate System Unit    : {diag_merl.unit}")
    print(f"  Training Calibration Pts  : {diag_merl.num_calibration_points}")
    print(f"  Training Reprojection Error: Mean={diag_merl.training_reprojection_mean_error:.4f}, Max={diag_merl.training_reprojection_max_error:.4f}")

    merl_video_path = os.path.join("data", "merl", "10_1_crop.mp4")
    sampler = FrameSampler(target_fps=5.0)

    Track.reset_id_counter(0)
    merl_tracker = ByteTracker(track_high_thresh=track_cfg.get("track_high_thresh", 0.50))

    for frame_idx, timestamp_sec, frame in sampler.sample_frames(merl_video_path, max_frames=20):
        det_res = detector.detect(frame, frame_id=frame_idx, timestamp_sec=timestamp_sec)
        merl_tracker.update(det_res.detections, frame_id=frame_idx, timestamp_sec=timestamp_sec)

    all_merl_tracks = merl_tracker.tracked_tracks + merl_tracker.lost_tracks + merl_tracker.removed_tracks
    merl_trajectories = TrajectoryExtractor.extract_from_tracks(all_merl_tracks)
    transformed_merl_trajectories = merl_calibrator.transform_trajectories(merl_trajectories)

    merl_floormap = visualize_floor_trajectories(
        transformed_merl_trajectories,
        grid_size=(600, 600),
        floor_bounds=(-1.0, 11.0, -1.0, 7.0),
        title="MERL Retail Qualitative Top-Down Floor Map"
    )
    out_merl_floormap = os.path.join(reports_dir, "phase_05_merl_floormap_trajectories.jpg")
    cv2.imwrite(out_merl_floormap, merl_floormap)
    print(f"  Saved MERL Top-Down Map   : {out_merl_floormap}")

    # ============================================================
    # 4. Controlled College Recording Configuration Readiness
    # ============================================================
    print("\n[3/4] Verifying Controlled College Footage Readiness...")
    college_calibrator = HomographyCalibrator.from_config(config, dataset_name="college_controlled")
    diag_college = college_calibrator.diagnostics
    print(f"  College Calibrator Ready  : YES ({diag_college.num_calibration_points} calibration points configured)")
    print(f"  Spec Reference            : docs/CONTROLLED_VIDEO_RECORDING_SPEC.md")

    # ============================================================
    # 5. Export JSON Validation Artifact
    # ============================================================
    summary_data = {
        "mot17_04_calibration": diag_mot17.to_dict(),
        "merl_calibration": diag_merl.to_dict(),
        "college_controlled_calibration": diag_college.to_dict(),
        "homography_performance": {
            "num_trajectories_transformed": len(transformed_mot17_trajectories),
            "total_transformation_time_ms": round(t_hom_ms, 3),
            "per_trajectory_time_ms": round(t_hom_ms / max(1, len(transformed_mot17_trajectories)), 4)
        },
        "sample_trajectories": [t.to_dict() for t in transformed_mot17_trajectories[:3]]
    }

    json_summary_path = os.path.join(reports_dir, "phase_05_trajectory_summary.json")
    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[4/4] Exported Trajectory Summary JSON: {json_summary_path}")

    print("\n" + "=" * 70)
    print("PHASE 5 TRAJECTORY & HOMOGRAPHY SPIKE SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    main()
