"""
Phase 10.3 Verification Script — Trajectory Density Pipeline Integration
========================================================================
Runs two separate deterministic integration checks:
1. Frame-Sequence Integration Check (MOT17-04):
   - Ingests 10 sequential image files via cv2.imread.
   - Evaluates frame-level detection, tracking, homography, and density binning.
   - Saves: outputs/reports/phase_10_density_heatmap_mot17.png (.json).

2. Recorded-Video-File Integration Check (MERL):
   - Ingests mp4 video container (data/merl/10_1_crop.mp4) using FrameSampler (cv2.VideoCapture).
   - Lazily decodes 100 sampled frames at 5.0 FPS target.
   - Evaluates video decoding, detection, tracking, homography, and density binning.
   - Saves: outputs/reports/phase_10_density_heatmap_merl.png (.json).
   - Explicitly notes boundary-truncated, sparse observations (71/74 out-of-bounds).

EVIDENCE CLASSIFICATION & BOUNDARIES:
- These runs provide operational engineering verification of the pipeline.
- Neither run constitutes retail customer validation, an MOT benchmark, or
  physical metric calibration.
- Coordinates are expressed in configured floor units, not surveyed metres.
"""

import os
import sys
import json
import time
import cv2
import yaml

sys.path.insert(0, os.path.abspath("."))

from src.video import FrameSampler
from src.detection import YOLO11PersonDetector
from src.tracking import ByteTracker, Track
from src.trajectory import TrajectoryExtractor
from src.spatial import HomographyCalibrator
from src.analytics.zone import ZoneConfig
from src.analytics.density import (
    GridBounds,
    compute_density_grid,
    render_density_heatmap,
)


def run_mot17_density_pipeline(config: dict, reports_dir: str) -> dict:
    """Run detection, tracking, homography, and density heatmap on local MOT17-04 image frames."""
    print("\n[1/2] Executing Frame-Sequence Pipeline on MOT17-04 Local Image Sample...")

    det_cfg = config.get("detection", {})
    track_cfg = config.get("tracking", {})

    model_path = det_cfg.get("model_path", "data/models/yolo11n.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Detector model weights not found at {model_path}")

    mot17_dir = os.path.join("data", "mot17", "train", "MOT17-04-FRCNN", "img1")
    if not os.path.exists(mot17_dir):
        raise FileNotFoundError(f"MOT17-04 frame directory not found at {mot17_dir}")

    mot17_frames = sorted([f for f in os.listdir(mot17_dir) if f.endswith(".jpg")])[:10]
    print(f"  Loaded {len(mot17_frames)} consecutive frames from {mot17_dir}")

    # Initialize components
    Track.reset_id_counter(0)
    detector = YOLO11PersonDetector(
        model_path=model_path,
        conf_threshold=det_cfg.get("conf_threshold", 0.40),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
    )
    tracker = ByteTracker(
        track_high_thresh=track_cfg.get("track_high_thresh", 0.50),
        max_time_lost_sec=track_cfg.get("max_time_lost_sec", 2.0),
    )

    # Process frames
    t0 = time.time()
    for idx, fname in enumerate(mot17_frames, start=1):
        fpath = os.path.join(mot17_dir, fname)
        img = cv2.imread(fpath)
        timestamp_sec = round((idx - 1) / 30.0, 3)

        det_res = detector.detect(img, frame_id=idx, timestamp_sec=timestamp_sec)
        tracker.update(det_res.detections, frame_id=idx, timestamp_sec=timestamp_sec)

    tracking_time_sec = time.time() - t0

    # Extract & transform trajectories
    all_tracks = tracker.tracked_tracks + tracker.lost_tracks + tracker.removed_tracks
    trajectories = TrajectoryExtractor.extract_from_tracks(all_tracks)
    calibrator = HomographyCalibrator.from_config(config, dataset_name="mot17_04")
    transformed_trajectories = calibrator.transform_trajectories(trajectories)

    print(f"  Extracted {len(transformed_trajectories)} trajectories from {len(mot17_frames)} frames ({tracking_time_sec:.2f}s)")

    # Load bounds and zones
    mot17_bounds = GridBounds(0.0, 10.0, 0.0, 8.0)
    mot17_zones, _ = ZoneConfig.load_from_config(config, dataset_key="mot17_04")

    # Compute density grid
    grid = compute_density_grid(
        transformed_trajectories,
        bounds=mot17_bounds,
        cell_size=0.5,
    )

    print(f"  Density Grid Evaluated:")
    print(f"    - Grid Shape            : {grid.num_bins_y} rows x {grid.num_bins_x} cols")
    print(f"    - Cell Size             : {grid.cell_size_x:.2f} x {grid.cell_size_y:.2f} configured floor units")
    print(f"    - Total Evaluated Points: {grid.total_evaluated_points}")
    print(f"    - Successfully Binned   : {grid.total_binned_points}")
    print(f"    - Out of Bounds Points  : {grid.out_of_bounds_count}")
    print(f"    - Max Cell Density      : {grid.max_cell_count} observations")

    # Render heatmap
    out_png = os.path.join(reports_dir, "phase_10_density_heatmap_mot17.png")
    render_density_heatmap(
        grid=grid,
        output_path=out_png,
        title="MOT17-04 Pedestrian Trajectory Density Heatmap\n(Engineering Integration Check — Not Retail Validation)",
        zones=mot17_zones,
        unit_label="configured floor units",
    )
    print(f"  Saved Heatmap Visualization: {out_png}")

    # Export summary JSON
    summary_data = {
        "pipeline": "frame_sequence_workflow",
        "input_type": "consecutive_image_files",
        "dataset": "mot17_04",
        "frames_evaluated": len(mot17_frames),
        "trajectories_count": len(transformed_trajectories),
        "grid_summary": grid.to_dict(),
        "evidence_notice": (
            "Limited frame-level pedestrian-tracking engineering integration check using sequential image files. "
            "Does NOT constitute a verified video-file ingestion run, retail customer validation, shopper analytics, "
            "or formal MOT evaluation."
        ),
    }

    out_json = os.path.join(reports_dir, "phase_10_density_summary_mot17.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Saved Summary JSON          : {out_json}")

    return summary_data


def run_merl_video_density_pipeline(config: dict, reports_dir: str) -> dict:
    """Execute recorded-video-file pipeline on local MERL video sample via FrameSampler."""
    print("\n[2/2] Executing Recorded-Video-File Pipeline on MERL Local Video...")

    video_path = os.path.join("data", "merl", "10_1_crop.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"MERL video sample not found at {video_path}")

    model_path = config.get("detection", {}).get("model_path", "data/models/yolo11n.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Detector model weights not found at {model_path}")

    target_fps = float(config.get("video", {}).get("sampling", {}).get("target_fps", 5.0))
    sampler = FrameSampler(target_fps=target_fps)

    det_cfg = config.get("detection", {})
    track_cfg = config.get("tracking", {})

    detector = YOLO11PersonDetector(
        model_path=model_path,
        conf_threshold=det_cfg.get("conf_threshold", 0.40),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
    )
    Track.reset_id_counter(0)
    tracker = ByteTracker(
        track_high_thresh=track_cfg.get("track_high_thresh", 0.50),
        max_time_lost_sec=track_cfg.get("max_time_lost_sec", 2.0),
    )

    # Lazily sample frames from video container using existing FrameSampler
    max_frames_to_sample = 100
    sampled_frames_count = 0
    t0 = time.time()

    for frame_idx, timestamp_sec, frame in sampler.sample_frames(video_path, max_frames=max_frames_to_sample):
        det_res = detector.detect(frame, frame_id=frame_idx, timestamp_sec=timestamp_sec)
        tracker.update(det_res.detections, frame_id=frame_idx, timestamp_sec=timestamp_sec)
        sampled_frames_count += 1

    processing_time_sec = time.time() - t0

    # Extract & transform trajectories
    all_tracks = tracker.tracked_tracks + tracker.lost_tracks + tracker.removed_tracks
    trajectories = TrajectoryExtractor.extract_from_tracks(all_tracks)
    calibrator = HomographyCalibrator.from_config(config, dataset_name="merl")
    transformed_trajectories = calibrator.transform_trajectories(trajectories)

    print(f"  Decoded & processed {sampled_frames_count} sampled frames ({processing_time_sec:.2f}s)")
    print(f"  Extracted {len(transformed_trajectories)} trajectories from video stream")

    # Load bounds and zones
    merl_bounds = GridBounds(0.0, 10.0, 0.0, 6.0)
    merl_zones, _ = ZoneConfig.load_from_config(config, dataset_key="merl")

    # Compute density grid
    grid = compute_density_grid(
        transformed_trajectories,
        bounds=merl_bounds,
        cell_size=0.5,
    )

    print(f"  Density Grid Evaluated:")
    print(f"    - Grid Shape            : {grid.num_bins_y} rows x {grid.num_bins_x} cols")
    print(f"    - Cell Size             : {grid.cell_size_x:.2f} x {grid.cell_size_y:.2f} configured floor units")
    print(f"    - Total Evaluated Points: {grid.total_evaluated_points}")
    print(f"    - Successfully Binned   : {grid.total_binned_points}")
    print(f"    - Out of Bounds Points  : {grid.out_of_bounds_count}")
    print(f"    - Untransformed Points  : {grid.untransformed_count}")
    print(f"    - Max Cell Density      : {grid.max_cell_count} observations")

    # Verify conservation invariant
    assert grid.total_evaluated_points == (
        grid.total_binned_points + grid.out_of_bounds_count + grid.untransformed_count
    ), "Conservation invariant violated in video density grid computation!"

    # Render heatmap with explicit sparse / boundary-truncated label
    out_png = os.path.join(reports_dir, "phase_10_density_heatmap_merl.png")
    render_density_heatmap(
        grid=grid,
        output_path=out_png,
        title="MERL Retail Video Trajectory Density Heatmap\n(Sparse / Boundary-Truncated Sample — Not Quantitative Validation)",
        zones=merl_zones,
        unit_label="configured floor units",
    )
    print(f"  Saved Heatmap Visualization: {out_png}")

    # Export summary JSON
    summary_data = {
        "pipeline": "recorded_video_file_workflow",
        "input_type": "mp4_video_container",
        "video_reader": "FrameSampler (cv2.VideoCapture)",
        "source_clip": "data/merl/10_1_crop.mp4",
        "sampling_target_fps": target_fps,
        "sampled_frames_count": sampled_frames_count,
        "trajectories_count": len(transformed_trajectories),
        "grid_summary": grid.to_dict(),
        "evidence_notice": (
            "Recorded-video-file engineering integration check. Demonstrates successful decoding "
            "and trajectory projection through the pipeline. Due to customer positioning near camera boundary, "
            "observations are sparse and mostly out-of-bounds (71 of 74 points). "
            "Does NOT constitute validated retail customer analytics, shopper behavior insights, "
            "or physical metric ground-truth calibration."
        ),
    }

    out_json = os.path.join(reports_dir, "phase_10_density_summary_merl.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Saved Summary JSON          : {out_json}")

    return summary_data


def main():
    print("=" * 75)
    print("PHASE 10.3 — TRAJECTORY DENSITY INTEGRATION CHECK")
    print("=" * 75)

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    reports_dir = os.path.join("outputs", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    mot17_summary = run_mot17_density_pipeline(config, reports_dir)
    merl_summary = run_merl_video_density_pipeline(config, reports_dir)

    print("\n" + "=" * 75)
    print("PHASE 10.3 INTEGRATION VERIFICATION COMPLETE (SUCCESS)")
    print("=" * 75)


if __name__ == "__main__":
    main()

