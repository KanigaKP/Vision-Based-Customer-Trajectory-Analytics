"""
Phase 4 Verification & Tracking Benchmark Script
================================================
Runs YOLO11 Person Detector + ByteTrack Multi-Object Tracker pipeline
on MOT17-04 sequence and MERL dataset video.
Measures detector vs tracker latency overhead and exports visual artifacts.
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


def main():
    print("=" * 70)
    print("PHASE 4 — BYTETRACK MULTI-OBJECT TRACKING INTEGRATION SPIKE")
    print("=" * 70)

    # Load configuration
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    det_cfg = config.get("detection", {})
    track_cfg = config.get("tracking", {})

    # 1. Initialize Detector & Tracker
    print("\n[1/4] Initializing Detector & ByteTracker...")
    Track.reset_id_counter(0)
    detector = YOLO11PersonDetector(
        model_path=det_cfg.get("model_path", "data/models/yolo11n.onnx"),
        conf_threshold=det_cfg.get("conf_threshold", 0.40),
        iou_threshold=det_cfg.get("iou_threshold", 0.45)
    )

    tracker = ByteTracker(
        track_high_thresh=track_cfg.get("track_high_thresh", 0.50),
        track_low_thresh=track_cfg.get("track_low_thresh", 0.10),
        new_track_thresh=track_cfg.get("new_track_thresh", 0.50),
        match_thresh=track_cfg.get("match_thresh", 0.80),
        low_match_thresh=track_cfg.get("low_match_thresh", 0.50),
        max_time_lost_sec=track_cfg.get("max_time_lost_sec", 2.0)
    )

    reports_dir = os.path.join("outputs", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # 2. Task 4: MOT17-04 Consecutive Sequence Tracking Test
    mot17_dir = os.path.join("data", "mot17", "train", "MOT17-04-FRCNN", "img1")
    print(f"\n[2/4] Consecutive Sequence Tracking Test on MOT17-04 ({mot17_dir})...")

    mot17_frames = sorted([f for f in os.listdir(mot17_dir) if f.endswith(".jpg")])[:10]
    print(f"  Found {len(mot17_frames)} consecutive frames to process.")

    mot17_results = []
    det_times_mot17 = []
    track_times_mot17 = []

    for idx, fname in enumerate(mot17_frames, start=1):
        fpath = os.path.join(mot17_dir, fname)
        img = cv2.imread(fpath)
        timestamp_sec = round((idx - 1) / 30.0, 3)

        # Detect
        det_result = detector.detect(img, frame_id=idx, timestamp_sec=timestamp_sec)
        det_times_mot17.append(det_result.total_time_ms)

        # Track
        track_result = tracker.update(
            detections=det_result.detections,
            frame_id=idx,
            timestamp_sec=timestamp_sec,
            detector_time_ms=det_result.total_time_ms
        )
        track_times_mot17.append(track_result.tracker_time_ms)

        print(f"  Frame {idx:02d} ({fname}): Detections={det_result.person_count}, "
              f"Active Tracks={track_result.num_active_tracks}, Total Tracks Created={track_result.total_tracks_created}, "
              f"Det Latency={det_result.total_time_ms:.1f}ms, Track Latency={track_result.tracker_time_ms:.2f}ms")

        mot17_results.append(track_result)

        # Save annotated visualization for frame 1, 5, 10
        if idx in (1, 5, 10):
            annotated = ByteTracker.annotate_tracks(img, track_result.active_tracks)
            out_img_name = f"phase_04_mot17_tracking_frame_{idx:02d}.jpg"
            out_img_path = os.path.join(reports_dir, out_img_name)
            cv2.imwrite(out_img_path, annotated)
            print(f"    -> Saved MOT17 Tracking Output: {out_img_path}")

    # 3. Task 5: MERL Qualitative Video Tracking Demonstration
    merl_video_path = os.path.join("data", "merl", "10_1_crop.mp4")
    print(f"\n[3/4] Qualitative Video Tracking Demonstration on MERL Data ({merl_video_path})...")

    # Reset tracker for new video session
    Track.reset_id_counter(0)
    merl_tracker = ByteTracker(
        track_high_thresh=track_cfg.get("track_high_thresh", 0.50),
        track_low_thresh=track_cfg.get("track_low_thresh", 0.10),
        new_track_thresh=track_cfg.get("new_track_thresh", 0.50),
        match_thresh=track_cfg.get("match_thresh", 0.80),
        low_match_thresh=track_cfg.get("low_match_thresh", 0.50),
        max_time_lost_sec=track_cfg.get("max_time_lost_sec", 2.0)
    )

    sampler = FrameSampler(target_fps=5.0)
    merl_results = []
    det_times_merl = []
    track_times_merl = []

    saved_merl_count = 0
    max_merl_save = 3

    for frame_idx, timestamp_sec, frame in sampler.sample_frames(merl_video_path, max_frames=20):
        det_res = detector.detect(frame, frame_id=frame_idx, timestamp_sec=timestamp_sec)
        track_res = merl_tracker.update(
            detections=det_res.detections,
            frame_id=frame_idx,
            timestamp_sec=timestamp_sec,
            detector_time_ms=det_res.total_time_ms
        )

        det_times_merl.append(det_res.total_time_ms)
        track_times_merl.append(track_res.tracker_time_ms)
        merl_results.append(track_res)

        if saved_merl_count < max_merl_save:
            annotated_merl = ByteTracker.annotate_tracks(frame, track_res.active_tracks)
            out_merl_name = f"phase_04_merl_tracking_sample_{saved_merl_count + 1}_frame{frame_idx}.jpg"
            out_merl_path = os.path.join(reports_dir, out_merl_name)
            cv2.imwrite(out_merl_path, annotated_merl)
            print(f"  Saved MERL Tracking Sample {saved_merl_count + 1}: {out_merl_path} "
                  f"(Active Tracks: {track_res.num_active_tracks})")
            saved_merl_count += 1

    # 4. Task 9: Latency Breakdown & Performance Measurements
    print("\n[4/4] Latency Breakdown & Performance Summary:")
    
    avg_det_ms = float(np.mean(det_times_mot17 + det_times_merl))
    avg_track_ms = float(np.mean(track_times_mot17 + track_times_merl))
    avg_comb_ms = avg_det_ms + avg_track_ms

    print(f"  Average Detector Latency     : {avg_det_ms:.2f} ms")
    print(f"  Average ByteTrack Overhead   : {avg_track_ms:.2f} ms")
    print(f"  Combined Frame Latency       : {avg_comb_ms:.2f} ms")
    print(f"  Model Inference Speed        : {1000.0 / avg_det_ms:.2f} FPS")
    print(f"  Combined Detection+Track FPS : {1000.0 / avg_comb_ms:.2f} FPS")

    # Export JSON Summary
    summary_data = {
        "tracker_algorithm": "ByteTrack (scipy.optimize.linear_sum_assignment + KalmanFilter)",
        "mot17_sequence_test": {
            "frames_processed": len(mot17_results),
            "total_tracks_created": mot17_results[-1].total_tracks_created if mot17_results else 0,
            "final_active_tracks": mot17_results[-1].num_active_tracks if mot17_results else 0,
            "avg_detector_ms": round(float(np.mean(det_times_mot17)), 2),
            "avg_tracker_ms": round(float(np.mean(track_times_mot17)), 2)
        },
        "merl_video_test": {
            "sampled_frames_processed": len(merl_results),
            "total_tracks_created": merl_results[-1].total_tracks_created if merl_results else 0,
            "avg_detector_ms": round(float(np.mean(det_times_merl)), 2),
            "avg_tracker_ms": round(float(np.mean(track_times_merl)), 2)
        },
        "performance_metrics": {
            "avg_detector_latency_ms": round(avg_det_ms, 2),
            "avg_tracker_overhead_ms": round(avg_track_ms, 2),
            "combined_frame_latency_ms": round(avg_comb_ms, 2),
            "combined_fps": round(1000.0 / avg_comb_ms, 2)
        }
    }

    json_summary_path = os.path.join(reports_dir, "phase_04_tracking_summary.json")
    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nExported Tracking Summary JSON: {json_summary_path}")

    print("\n" + "=" * 70)
    print("PHASE 4 TRACKING INTEGRATION SPIKE SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    main()
