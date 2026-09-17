"""
Phase 2 Validation Script
=========================
Executes video input inspection, quality evaluation, and frame sampling
on existing project video data (data/merl/10_1_crop.mp4).
Generates validation report artifacts in outputs/reports/.
"""

import os
import sys
import json
import cv2
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.video import VideoInspector, QualityEvaluator, FrameSampler


def main():
    print("=" * 70)
    print("PHASE 2 — VIDEO PREPROCESSING & QUALITY VERIFICATION SPIKE")
    print("=" * 70)

    # Load configuration
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    video_cfg = config.get("video", {})
    allowed_exts = video_cfg.get("allowed_extensions", [".mp4", ".avi", ".mov", ".mkv"])
    quality_thresholds = video_cfg.get("quality_thresholds", {})
    sampling_cfg = video_cfg.get("sampling", {})
    target_fps = sampling_cfg.get("target_fps", 5.0)

    # 1. Initialize Modules
    inspector = VideoInspector(allowed_extensions=allowed_exts)
    evaluator = QualityEvaluator(thresholds_config=quality_thresholds)
    sampler = FrameSampler(target_fps=target_fps)

    # 2. Identify Test Input: MERL Shopping Video
    test_video_path = os.path.join("data", "merl", "10_1_crop.mp4")
    print(f"\n[1/4] Inspecting Video Input: {test_video_path}")
    
    meta = inspector.inspect(test_video_path)
    print(f"  Status                  : {meta.status}")
    print(f"  File Name               : {meta.file_name}")
    print(f"  File Size               : {meta.file_size_bytes / (1024*1024):.2f} MB")
    print(f"  Resolution              : {meta.width} x {meta.height}")
    print(f"  FPS                     : {meta.fps:.2f}")
    print(f"  Total Frames            : {meta.total_frames}")
    print(f"  Duration                : {meta.duration_seconds:.2f} seconds")
    print(f"  Codec / FourCC          : {meta.codec_name} ({meta.fourcc})")
    print(f"  Sample Frame Accessible : {meta.sample_frame_accessible}")

    if not meta.is_readable:
        print(f"ERROR: Test video unreadable ({meta.error_message})")
        sys.exit(1)

    # 3. Quality Evaluation
    print(f"\n[2/4] Evaluating Quality Indicators against Configured Warning Thresholds...")
    assessment = evaluator.evaluate(test_video_path, max_sample_frames=15)

    print(f"  Overall Quality Status  : {assessment.overall_status}")
    print(f"  Resolution Status       : {assessment.resolution_status} ({assessment.width}x{assessment.height})")
    print(f"  FPS Status              : {assessment.fps_status} ({assessment.fps:.2f} FPS)")
    print(f"  Brightness Status       : {assessment.brightness_status} (Mean: {assessment.brightness_mean:.1f} / 255)")
    print(f"  Sharpness Status        : {assessment.sharpness_status} (Laplacian Var: {assessment.sharpness_laplacian_var:.1f})")
    print(f"  Readability Status      : {assessment.readability_status}")

    if assessment.warnings:
        print("  Configured Warnings:")
        for w in assessment.warnings:
            print(f"    - [WARNING] {w}")
    if assessment.errors:
        print("  Errors:")
        for e in assessment.errors:
            print(f"    - [ERROR] {e}")

    # 4. Frame Sampling & Representative Frame Export
    print(f"\n[3/4] Sampling Frames at Target Rate ({target_fps} FPS)...")
    reports_dir = os.path.join("outputs", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    sampled_frames_info = []
    saved_samples_count = 0
    max_save_samples = 3

    for frame_idx, timestamp_sec, frame in sampler.sample_frames(test_video_path, override_target_fps=target_fps, max_frames=30):
        sampled_frames_info.append({
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp_sec,
            "shape": list(frame.shape)
        })

        # Save first 3 sampled frames as representative validation artifacts
        if saved_samples_count < max_save_samples:
            sample_filename = f"merl_sampled_frame_{saved_samples_count + 1}_idx{frame_idx}.jpg"
            out_img_path = os.path.join(reports_dir, sample_filename)
            
            # Annotate image with sampling info
            annotated = frame.copy()
            label = f"Frame {frame_idx} | t={timestamp_sec:.2f}s | MERL 10_1_crop"
            cv2.putText(annotated, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(out_img_path, annotated)
            print(f"  Saved representative frame {saved_samples_count + 1}: {out_img_path}")
            saved_samples_count += 1

    print(f"  Sampled {len(sampled_frames_info)} frames efficiently (Target FPS: {target_fps}).")

    # 5. Export JSON Validation Summary
    summary_data = {
        "metadata": meta.to_dict(),
        "quality_assessment": assessment.to_dict(),
        "sampling_summary": {
            "target_fps": target_fps,
            "sampled_count": len(sampled_frames_info),
            "sample_frames": sampled_frames_info[:5]  # first 5 entries
        }
    }
    json_summary_path = os.path.join(reports_dir, "phase_02_video_validation_summary.json")
    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[4/4] Exported Validation Summary JSON: {json_summary_path}")

    print("\n" + "=" * 70)
    print("PHASE 2 VERIFICATION SPIKE SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    main()
