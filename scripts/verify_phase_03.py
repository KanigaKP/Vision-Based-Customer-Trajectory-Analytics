"""
Phase 3 Verification & Benchmark Script
======================================
Executes single-frame detection test on MOT17-04 frame 1 and sampled video test on MERL data.
Measures latency breakdown and exports visualization artifacts.
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


def main():
    print("=" * 70)
    print("PHASE 3 — YOLO11 PERSON DETECTION MODULE SPIKE (ONNX RUNTIME CPU)")
    print("=" * 70)

    # 1. Load configuration
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    det_cfg = config.get("detection", {})
    model_path = det_cfg.get("model_path", "data/models/yolo11n.onnx")
    conf_thresh = det_cfg.get("conf_threshold", 0.40)
    iou_thresh = det_cfg.get("iou_threshold", 0.45)

    print(f"\n[1/4] Instantiating Detector from: {model_path}")
    detector = YOLO11PersonDetector(
        model_path=model_path,
        conf_threshold=conf_thresh,
        iou_threshold=iou_thresh
    )

    meta = detector.get_metadata()
    print(f"  Model Load Time           : {meta['load_time_ms']:.2f} ms")
    print(f"  Execution Providers      : {meta['providers']}")
    print(f"  Input Tensor Name/Shape  : {meta['input_name']} {meta['input_shape']}")
    print(f"  Output Tensor Name/Shape : {meta['output_name']} {meta['output_shape']}")
    print(f"  Configured Conf Threshold: {meta['conf_threshold']}")
    print(f"  Configured IoU Threshold : {meta['iou_threshold']}")

    # 2. Task 6: Single-Frame Test (MOT17-04 frame 1)
    mot17_frame_path = os.path.join("data", "mot17", "train", "MOT17-04-FRCNN", "img1", "000001.jpg")
    print(f"\n[2/4] Single-Frame Test on: {mot17_frame_path}")

    img_mot17 = cv2.imread(mot17_frame_path)
    if img_mot17 is None:
        print(f"ERROR: Could not load test frame {mot17_frame_path}")
        sys.exit(1)

    result_mot17 = detector.detect(img_mot17, frame_id=1, timestamp_sec=0.0)

    print(f"  Image Dimensions          : {result_mot17.image_width} x {result_mot17.image_height}")
    print(f"  Raw Candidate Anchors     : {result_mot17.raw_candidate_count}")
    print(f"  Filtered Person Detections: {result_mot17.person_count}")
    print(f"  Preprocessing Latency     : {result_mot17.preprocess_time_ms:.2f} ms")
    print(f"  ONNX Inference Latency    : {result_mot17.inference_time_ms:.2f} ms")
    print(f"  Post-processing Latency   : {result_mot17.postprocess_time_ms:.2f} ms")
    print(f"  Total Single-Frame Latency: {result_mot17.total_time_ms:.2f} ms")
    print(f"  Model Inference FPS       : {1000.0 / result_mot17.inference_time_ms:.2f} FPS")
    print(f"  End-to-End Detector FPS   : {1000.0 / result_mot17.total_time_ms:.2f} FPS")

    # Confidence distribution summary
    if result_mot17.detections:
        confs = [d.confidence for d in result_mot17.detections]
        print(f"  Confidence Distribution   : min={min(confs):.2f}, mean={np.mean(confs):.2f}, max={max(confs):.2f}")

    # Annotate and save single-frame output
    annotated_mot17 = detector.annotate_image(img_mot17, result_mot17.detections)
    reports_dir = os.path.join("outputs", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    mot17_out_path = os.path.join(reports_dir, "phase_03_mot17_frame1_detections.jpg")
    cv2.imwrite(mot17_out_path, annotated_mot17)
    print(f"  Saved Single-Frame Visualization: {mot17_out_path}")

    # 3. Task 7: Sampled Video Test (MERL Shopping Video)
    merl_video_path = os.path.join("data", "merl", "10_1_crop.mp4")
    print(f"\n[3/4] Sampled Video Test on: {merl_video_path}")

    sampler = FrameSampler(target_fps=5.0)
    video_results = []
    saved_video_samples = 0
    max_video_samples_save = 3

    t_video_start = time.time()
    sampled_frame_count = 0

    for frame_idx, timestamp_sec, frame in sampler.sample_frames(merl_video_path, max_frames=20):
        t_f_start = time.time()
        res = detector.detect(frame, frame_id=frame_idx, timestamp_sec=timestamp_sec)
        t_f_total = (time.time() - t_f_start) * 1000.0

        sampled_frame_count += 1
        video_results.append({
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp_sec,
            "person_count": res.person_count,
            "inference_time_ms": res.inference_time_ms,
            "total_time_ms": res.total_time_ms
        })

        if saved_video_samples < max_video_samples_save:
            annotated_frame = detector.annotate_image(frame, res.detections)
            out_frame_name = f"phase_03_merl_sample_{saved_video_samples + 1}_frame{frame_idx}.jpg"
            out_frame_path = os.path.join(reports_dir, out_frame_name)
            cv2.imwrite(out_frame_path, annotated_frame)
            print(f"  Saved Video Sample {saved_video_samples + 1}: {out_frame_path} ({res.person_count} persons detected)")
            saved_video_samples += 1

    total_pipeline_time_sec = time.time() - t_video_start
    avg_infer_ms = np.mean([r["inference_time_ms"] for r in video_results])
    avg_total_ms = np.mean([r["total_time_ms"] for r in video_results])

    print(f"\n[4/4] Performance Summary Across {sampled_frame_count} Sampled Video Frames:")
    print(f"  Average Model Inference Latency : {avg_infer_ms:.2f} ms ({1000.0/avg_infer_ms:.2f} Model FPS)")
    print(f"  Average Single-Frame End-to-End : {avg_total_ms:.2f} ms ({1000.0/avg_total_ms:.2f} Detector FPS)")
    print(f"  End-to-End Video Pipeline FPS   : {sampled_frame_count / total_pipeline_time_sec:.2f} Pipeline FPS")

    # Export JSON Summary
    summary_data = {
        "model_metadata": meta,
        "single_frame_mot17": result_mot17.to_dict(),
        "video_pipeline_merl": {
            "video_path": merl_video_path,
            "sampled_frames_processed": sampled_frame_count,
            "avg_inference_time_ms": round(float(avg_infer_ms), 2),
            "avg_detector_time_ms": round(float(avg_total_ms), 2),
            "model_fps": round(1000.0 / avg_infer_ms, 2),
            "pipeline_fps": round(sampled_frame_count / total_pipeline_time_sec, 2),
            "frame_results": video_results
        }
    }
    json_path = os.path.join(reports_dir, "phase_03_detection_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nExported Detection Validation JSON: {json_path}")

    print("\n" + "=" * 70)
    print("PHASE 3 DETECTION SPIKE SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    main()
