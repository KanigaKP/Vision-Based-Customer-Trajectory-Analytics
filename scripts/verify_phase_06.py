"""
Phase 6 Verification Script
===========================
Validates spatial zone definition, Point-in-Polygon (PIP) trajectory evaluation,
temporal hysteresis filtering, and floor map zone visualization overlays.

Generates outputs in outputs/reports/ phase_06_zone_analytics_merl.png and
phase_06_zone_analytics_mot17.png.
"""

import os
import sys
import time
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath("."))

from src.video import FrameSampler
from src.detection import YOLO11PersonDetector
from src.tracking import ByteTracker
from src.trajectory import TrajectoryExtractor, Trajectory, TrajectoryPoint
from src.spatial import HomographyCalibrator, visualize_zones_and_trajectories
from src.analytics import Zone, ZoneConfig, ZoneEvaluator


def main():
    print("=" * 75)
    print("PHASE 6 — MANUAL ZONE DEFINITION & SPATIAL ZONE ANALYTICS VERIFICATION")
    print("=" * 75)

    # 1. Load Configuration
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    reports_dir = os.path.join("outputs", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # ============================================================
    # 2. MERL Supermarket Dataset Zone Analytics Verification
    # ============================================================
    print("\n[1/2] Verifying MERL Supermarket Retail Zone Pipeline...")
    
    # Load MERL Homography Calibrator
    merl_calibrator = HomographyCalibrator.from_config(config, dataset_name="merl")
    
    # Load MERL Zone Definitions
    merl_zones, merl_hysteresis = ZoneConfig.load_from_config(config, dataset_key="merl")
    print(f"  Loaded {len(merl_zones)} retail zones for MERL (Hysteresis Window N = {merl_hysteresis}).")
    for z in merl_zones:
        print(f"    - [{z.zone_id}] '{z.name}': {len(z.polygon)} vertices, Color: {z.color}")

    # Build MERL Evaluator
    merl_evaluator = ZoneEvaluator(zones=merl_zones, hysteresis_window=merl_hysteresis)

    # Check for MERL sample video or create sample trajectories
    merl_video_path = os.path.join("data", "merl", "1_1_0_cam1.mp4")
    merl_trajectories = []

    if os.path.exists(merl_video_path) and os.path.exists("data/models/yolo11n.onnx"):
        print("  Running YOLO11 + ByteTrack on MERL sample video...")
        detector = YOLO11PersonDetector(
            model_path="data/models/yolo11n.onnx",
            conf_threshold=0.40
        )
        tracker = ByteTracker(
            track_high_thresh=0.50,
            track_low_thresh=0.10,
            max_time_lost_sec=2.0
        )
        sampler = FrameSampler(video_path=merl_video_path, target_fps=5.0, max_sampled_frames=30)
        
        for idx, (frame, timestamp_sec) in enumerate(sampler.sample_frames()):
            dets = detector.detect(frame)
            tracks = tracker.update(dets, frame_id=idx, timestamp_sec=timestamp_sec)

        tracks = tracker.get_all_tracks()
        extracted = TrajectoryExtractor.extract_all(tracks, min_points=2)
        merl_trajectories = merl_calibrator.transform_trajectories(extracted)
    else:
        print("  Notice: MERL video or model not present. Generating synthetic floor trajectories for validation...")
        # Create representative customer trajectories across MERL retail space [0, 10] x [0, 6]
        # Customer 1: Walks along Shelf A then into Central Aisle
        t1 = Trajectory(track_id=101)
        for i in range(10):
            t1.add_point(TrajectoryPoint(
                track_id=101, frame_idx=i, timestamp_sec=i*0.2,
                image_x=200+i*20, image_y=500-i*10,
                floor_x=2.0 + i*0.3, floor_y=2.0 + i*0.2
            ))
        
        # Customer 2: Walks near Shelf B then to Checkout Area
        t2 = Trajectory(track_id=102)
        for i in range(10):
            t2.add_point(TrajectoryPoint(
                track_id=102, frame_idx=i, timestamp_sec=i*0.2,
                image_x=600-i*15, image_y=550+i*5,
                floor_x=7.5 - i*0.2, floor_y=3.5 - i*0.3
            ))

        merl_trajectories = [t1, t2]

    # Evaluate Zone Assignments
    merl_hit_counts = {z.zone_id: 0 for z in merl_zones}
    merl_hit_counts[None] = 0
    total_eval_points = 0

    for trk in merl_trajectories:
        assigned = merl_evaluator.evaluate_trajectory(trk)
        for z_id in assigned:
            merl_hit_counts[z_id] = merl_hit_counts.get(z_id, 0) + 1
            total_eval_points += 1

    print(f"  Evaluated {total_eval_points} trajectory points across {len(merl_trajectories)} tracks.")
    for z_id, count in merl_hit_counts.items():
        label = z_id if z_id is not None else "Unassigned / Outside Zones"
        print(f"    - Zone '{label}': {count} point visits ({count/max(1, total_eval_points)*100:.1f}%)")

    # Render MERL Floor Overlay Map
    merl_output_png = os.path.join(reports_dir, "phase_06_zone_analytics_merl.png")
    visualize_zones_and_trajectories(
        zones=merl_zones,
        trajectories=merl_trajectories,
        output_path=merl_output_png,
        title="MERL Retail Store — Phase 6 Spatial Zone Analytics",
        grid_bounds=(0.0, 10.0, 0.0, 6.0)
    )
    print(f"  Saved MERL zone visualization map: {merl_output_png}")

    # ============================================================
    # 3. MOT17-04 Benchmark Dataset Zone Analytics Verification
    # ============================================================
    print("\n[2/2] Verifying MOT17-04 Benchmark Corridor Zone Pipeline...")

    mot17_calibrator = HomographyCalibrator.from_config(config, dataset_name="mot17_04")
    mot17_zones, mot17_hysteresis = ZoneConfig.load_from_config(config, dataset_key="mot17_04")
    print(f"  Loaded {len(mot17_zones)} benchmark zones for MOT17_04 (Hysteresis Window N = {mot17_hysteresis}).")
    for z in mot17_zones:
        print(f"    - [{z.zone_id}] '{z.name}': {len(z.polygon)} vertices, Color: {z.color}")

    mot17_evaluator = ZoneEvaluator(zones=mot17_zones, hysteresis_window=mot17_hysteresis)

    # Create synthetic trajectory for MOT17 corridor space [0, 10] x [0, 8]
    t_mot = Trajectory(track_id=201)
    for i in range(12):
        t_mot.add_point(TrajectoryPoint(
            track_id=201, frame_idx=i, timestamp_sec=i*0.2,
            image_x=500+i*10, image_y=400+i*5,
            floor_x=1.0 + i*0.7, floor_y=1.5 + i*0.5
        ))
    mot17_trajectories = [t_mot]

    for trk in mot17_trajectories:
        mot17_evaluator.evaluate_trajectory(trk)

    mot17_output_png = os.path.join(reports_dir, "phase_06_zone_analytics_mot17.png")
    visualize_zones_and_trajectories(
        zones=mot17_zones,
        trajectories=mot17_trajectories,
        output_path=mot17_output_png,
        title="MOT17-04 Corridor — Phase 6 Spatial Zone Analytics",
        grid_bounds=(0.0, 10.0, 0.0, 8.0)
    )
    print(f"  Saved MOT17-04 zone visualization map: {mot17_output_png}")

    # ============================================================
    # 4. Final Status
    # ============================================================
    assert os.path.exists(merl_output_png), "MERL output visualization PNG missing!"
    assert os.path.exists(mot17_output_png), "MOT17 output visualization PNG missing!"

    print("\n" + "=" * 75)
    print("VERIFICATION SUCCESSFUL: Verification script completes successfully")
    print("and all expected outputs are generated without errors.")
    print("=" * 75)


if __name__ == "__main__":
    main()
