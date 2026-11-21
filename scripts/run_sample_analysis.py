"""
Reproducible Offline Sample-Analysis Pipeline & Report Generator.
==================================================================
Integrates all existing pipeline components (Phases 2-13) to evaluate local
sample data deterministically and export versioned analysis reports.

PIPELINE STACK & SEMANTICS:
1. Video Ingestion: FrameSampler (decodes frames at target FPS; recorded video only).
2. Person Detection: YOLO11PersonDetector (the existing pre-trained YOLO11-n ONNX model
   is used for inference and was not trained by this project. No detector accuracy or recall
   validation is claimed; detections serve as engineering smoke evidence).
3. Multi-Object Tracking: ByteTracker (assigns temporary, clip-local tracker IDs;
   no biometric re-identification, cross-camera tracking, or personal identities).
4. Trajectory Extraction: TrajectoryExtractor (extracts spatial coordinate series).
5. Planar Homography: HomographyCalibrator (projects coordinates to floor plane
   in configured floor units; physical metric scale is unverified).
6. Zone Evaluation & Hysteresis: ZoneEvaluator (Point-in-Polygon testing with
   temporal hysteresis filtering loaded from ZoneConfig across configured retail zones).
7. Dwell Analytics: analyze_dwell (measures observed temporal spans exit - entry;
   not continuous physical dwell. Records evaluated zeros for unvisited zones).
8. Trajectory Density: compute_density_grid (sampled trajectory-point counts with
   point-conservation verification; not physical occupancy or unique visitors).
9. Zone Transitions: compute_direct_transitions (chronological zone visit runs and
   direct transition counts; not Markovian probabilities. Records evaluated zeros for unvisited zones).
10. Spatial Observations & Gating: generate_spatial_observations (Phase 12 evidence
    gate; withholds observations when calibration or coverage is insufficient).
11. Report Export: build_analysis_report & export_analysis_report (Phase 13 schema v1.0.0).

EVIDENCE BOUNDARIES & CAVEATS:
- MERL Shopping Dataset: Actual recorded retail video footage, but its sample run
  is sparse and boundary-truncated (71 of 74 points out-of-bounds due to single-camera
  field-of-view cutoff) with unverified configured floor calibration units.
- MOT17-04: Non-retail outdoor pedestrian image sequence (engineering integration only).
- Evaluated Zero Metrics: Zones evaluated with 0 observed visits are preserved as
  numeric 0, clearly distinguishable from unsupplied/missing analytics sections.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Optional, Sequence
import cv2
import yaml

# Ensure project root is on Python path
sys.path.insert(0, os.path.abspath("."))

from src.video import FrameSampler
from src.detection import YOLO11PersonDetector
from src.tracking import ByteTracker, Track
from src.trajectory import TrajectoryExtractor
from src.spatial import HomographyCalibrator
from src.analytics.zone import ZoneConfig
from src.analytics.zone_evaluator import ZoneEvaluator
from src.analytics.dwell import analyze_dwell, DwellSummary
from src.analytics.density import GridBounds, compute_density_grid, TrajectoryDensityGrid
from src.analytics.transitions import compute_direct_transitions, DirectTransitionSummary
from src.analytics.spatial_observations import (
    EvidenceContext,
    SpatialObservationResult,
    generate_spatial_observations,
)
from src.reporting import (
    AnalysisRunMetadata,
    AnalysisReport,
    build_analysis_report,
    export_analysis_report,
)


def run_detector_inference_smoke_check(
    model_path: str = "data/models/yolo11n.onnx",
    test_image_path: str = "data/mot17/train/MOT17-04-FRCNN/img1/000001.jpg",
) -> int:
    """
    Run an inference smoke check on a single local image frame using ONNX Runtime CPU.

    NOTE: The existing YOLO11-n ONNX model is used for inference and was not trained by this project.
    This check provides engineering smoke verification of model loading and inference execution,
    NOT quantitative detector accuracy or recall validation.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Detector ONNX model weights not found at: {model_path}")
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Test image not found at: {test_image_path}")

    img = cv2.imread(test_image_path)
    if img is None:
        raise ValueError(f"Failed to read image at: {test_image_path}")

    detector = YOLO11PersonDetector(model_path=model_path)
    det_res = detector.detect(img, frame_id=1, timestamp_sec=0.0)
    detection_count = len(det_res.detections)
    print(f"[Inference Smoke Check] YOLO11-n ONNX (pre-trained inference model, not trained by this project) returned {detection_count} person detections on frame '{os.path.basename(test_image_path)}'")
    return detection_count


def setup_merl_zone_evaluator(config: Dict[str, Any]) -> Tuple[ZoneEvaluator, List[str], int]:
    """
    Configure and instantiate ZoneEvaluator for the MERL dataset using ZoneConfig.

    Obtains zones and configured hysteresis window from ZoneConfig.load_from_config(config, dataset_key="merl").
    Passes the returned hysteresis window to ZoneEvaluator instead of hardcoding any value.

    Args:
        config (Dict[str, Any]): Project configuration dictionary.

    Returns:
        Tuple[ZoneEvaluator, List[str], int]:
            - Instantiated ZoneEvaluator with configured zones and hysteresis_window.
            - List of known zone IDs.
            - Configured hysteresis window N.
    """
    merl_zones, hysteresis_window = ZoneConfig.load_from_config(config, dataset_key="merl")
    zone_evaluator = ZoneEvaluator(zones=merl_zones, hysteresis_window=hysteresis_window)
    known_zone_ids = [z.zone_id for z in merl_zones]
    return zone_evaluator, known_zone_ids, hysteresis_window


def run_merl_sample_analysis(
    config: Dict[str, Any],
    reports_dir: str = "outputs/reports",
    max_frames: int = 100,
) -> str:
    """
    Execute the integrated analysis pipeline on the local MERL retail video sample.

    Evaluates:
    - FrameSampler (100 sampled frames at 5.0 FPS)
    - YOLO11PersonDetector (ONNX Runtime CPU)
    - ByteTracker (session-local tracking)
    - TrajectoryExtractor & HomographyCalibrator
    - ZoneEvaluator (evaluates 74 points against configured MERL zones; all return None)
    - Dwell Analytics (0 episodes; evaluated zeros recorded for all defined zones)
    - Direct Transitions (0 transitions; evaluated zeros recorded for all defined zones)
    - Trajectory Density (74 total points: 3 binned, 71 out-of-bounds, conservation verified)
    - Phase 12 Spatial Observation Gating (withheld due to boundary truncation & unverified calibration)
    - Phase 13 Report Export (outputs/reports/phase_13_analysis_report_merl_sample.json)

    Returns:
        Path to exported report JSON file.
    """
    print("\n" + "=" * 75)
    print("RUNNING MERL SAMPLE INTEGRATED ANALYSIS PIPELINE")
    print("=" * 75)

    video_path = os.path.join("data", "merl", "10_1_crop.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"MERL video sample not found at: {video_path}")

    model_path = config.get("detection", {}).get("model_path", "data/models/yolo11n.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Detector weights not found at: {model_path}")

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
        track_low_thresh=track_cfg.get("track_low_thresh", 0.10),
        new_track_thresh=track_cfg.get("new_track_thresh", 0.50),
        match_thresh=track_cfg.get("match_thresh", 0.80),
        max_time_lost_sec=track_cfg.get("max_time_lost_sec", 2.0),
    )

    # 1. Video Sampling & Detection
    t0 = time.time()
    sampled_frames_count = 0
    total_detections_count = 0

    print(f"  Sampling video: {video_path} (target {target_fps} FPS, max {max_frames} frames)...")
    for frame_idx, timestamp_sec, frame in sampler.sample_frames(video_path, max_frames=max_frames):
        det_res = detector.detect(frame, frame_id=frame_idx, timestamp_sec=timestamp_sec)
        tracker.update(det_res.detections, frame_id=frame_idx, timestamp_sec=timestamp_sec)
        sampled_frames_count += 1
        total_detections_count += len(det_res.detections)

    tracking_time_sec = time.time() - t0
    print(f"  Processed {sampled_frames_count} frames ({total_detections_count} detections in {tracking_time_sec:.2f}s)")

    # 2. Trajectory Extraction & Homography
    all_tracks = tracker.tracked_tracks + tracker.lost_tracks + tracker.removed_tracks
    trajectories = TrajectoryExtractor.extract_from_tracks(all_tracks)
    calibrator = HomographyCalibrator.from_config(config, dataset_name="merl")
    transformed_trajectories = calibrator.transform_trajectories(trajectories)
    print(f"  Extracted {len(transformed_trajectories)} track trajectory(ies) with {sum(len(t.points) for t in transformed_trajectories)} points")

    # 3. Zone Evaluation with Hysteresis
    zone_evaluator, known_zone_ids, hysteresis_window = setup_merl_zone_evaluator(config)

    assigned_zone_list = []
    for traj in transformed_trajectories:
        assigned = zone_evaluator.evaluate_trajectory(traj)
        assigned_zone_list.extend(assigned)

    non_none_count = sum(1 for z in assigned_zone_list if z is not None)
    print(f"  Zone Evaluation: {len(assigned_zone_list)} points evaluated across {len(known_zone_ids)} zones (hysteresis_window={hysteresis_window})")
    print(f"    - In-Zone Points  : {non_none_count}")
    print(f"    - Unassigned (None): {len(assigned_zone_list) - non_none_count} (Customer at boundary margin outside zone polygons)")

    # 4. Dwell Analytics (Explicitly Evaluated with Zero Metrics for unvisited zones)
    dwell_summary = analyze_dwell(transformed_trajectories, known_zone_ids=known_zone_ids)
    print(f"  Dwell Analytics Evaluated:")
    print(f"    - Total Episodes   : {dwell_summary.total_episodes}")
    print(f"    - Total Dwell Span : {dwell_summary.total_dwell_sec:.2f}s")
    print(f"    - Evaluated Zones  : {len(dwell_summary.by_zone)} (All preserved as evaluated zeros)")

    # 5. Zone Transitions (Explicitly Evaluated with Zero Metrics)
    transition_summary = compute_direct_transitions(transformed_trajectories, known_zone_ids=known_zone_ids)
    print(f"  Transition Analytics Evaluated:")
    print(f"    - Total Transitions: {transition_summary.total_transitions}")
    print(f"    - Evaluated Zones  : {len(transition_summary.zone_summaries)} (All preserved as evaluated zeros)")

    # 6. Trajectory Density Grid
    merl_bounds = GridBounds(0.0, 10.0, 0.0, 6.0)
    density_grid = compute_density_grid(transformed_trajectories, bounds=merl_bounds, cell_size=0.5)
    print(f"  Density Grid Evaluated:")
    print(f"    - Total Evaluated  : {density_grid.total_evaluated_points}")
    print(f"    - Binned Points    : {density_grid.total_binned_points}")
    print(f"    - Out of Bounds    : {density_grid.out_of_bounds_count}")
    print(f"    - Conservation     : {density_grid.total_evaluated_points == (density_grid.total_binned_points + density_grid.out_of_bounds_count)}")

    # 7. Phase 12 Spatial Observation Gating
    evidence_context = EvidenceContext(
        environment_type="retail_store",
        calibration_status="unverified_configured_units",
        coverage_status="boundary_truncated",
        notes="Operational MERL 100-frame video container integration check",
    )
    obs_result = generate_spatial_observations(
        evidence_context=evidence_context,
        dwell_summary=dwell_summary,
        transition_summary=transition_summary,
        density_grid=density_grid,
    )
    print(f"  Phase 12 Gating Evaluated:")
    print(f"    - Observation Status: {obs_result.status} (is_eligible={obs_result.is_eligible})")
    print(f"    - Reasons Withheld  : {len(obs_result.reasons_withheld)}")

    # 8. Report Export (outputs/reports/phase_13_analysis_report_merl_sample.json)
    metadata = AnalysisRunMetadata(
        run_id="phase_10_merl_sample",
        dataset_name="merl_shopping_dataset",
        source_type="recorded_video",
        target_fps=target_fps,
        total_frames_evaluated=sampled_frames_count,
        source_path=video_path,
        notes="Local MERL video container run evaluated through integrated pipeline (Phases 2-12). Note: Existing YOLO11-n ONNX model used for inference was not trained by this project.",
    )

    report = build_analysis_report(
        metadata=metadata,
        evidence_context=evidence_context,
        dwell_summary=dwell_summary,
        density_grid=density_grid,
        transition_summary=transition_summary,
        observation_result=obs_result,
        report_id="phase_13_analysis_report_merl_sample",
    )

    os.makedirs(reports_dir, exist_ok=True)
    out_file = os.path.join(reports_dir, "phase_13_analysis_report_merl_sample.json")
    export_analysis_report(report, output_path=out_file)
    print(f"  Exported Analysis Report: {out_file} (Schema v{report.schema_version})")

    return out_file


def run_mot17_sample_analysis(
    config: Dict[str, Any],
    reports_dir: str = "outputs/reports",
    max_frames: int = 10,
) -> str:
    """
    Execute sample pipeline on MOT17-04 outdoor pedestrian image sequence.
    Exports report with unsupplied dwell/transitions to verify unavailable-vs-zero
    distinction in the dashboard viewer.
    """
    print("\n" + "=" * 75)
    print("RUNNING MOT17-04 SAMPLE ANALYSIS PIPELINE")
    print("=" * 75)

    mot17_dir = os.path.join("data", "mot17", "train", "MOT17-04-FRCNN", "img1")
    if not os.path.exists(mot17_dir):
        raise FileNotFoundError(f"MOT17-04 frame directory not found at: {mot17_dir}")

    mot17_frames = sorted([f for f in os.listdir(mot17_dir) if f.endswith(".jpg")])[:max_frames]
    model_path = config.get("detection", {}).get("model_path", "data/models/yolo11n.onnx")

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

    for idx, fname in enumerate(mot17_frames, start=1):
        fpath = os.path.join(mot17_dir, fname)
        img = cv2.imread(fpath)
        timestamp_sec = round((idx - 1) / 30.0, 3)
        det_res = detector.detect(img, frame_id=idx, timestamp_sec=timestamp_sec)
        tracker.update(det_res.detections, frame_id=idx, timestamp_sec=timestamp_sec)

    all_tracks = tracker.tracked_tracks + tracker.lost_tracks + tracker.removed_tracks
    trajectories = TrajectoryExtractor.extract_from_tracks(all_tracks)
    calibrator = HomographyCalibrator.from_config(config, dataset_name="mot17_04")
    transformed_trajectories = calibrator.transform_trajectories(trajectories)

    mot17_bounds = GridBounds(0.0, 10.0, 0.0, 8.0)
    density_grid = compute_density_grid(transformed_trajectories, bounds=mot17_bounds, cell_size=0.5)

    evidence_context = EvidenceContext(
        environment_type="pedestrian_outdoor",
        calibration_status="unverified_configured_units",
        coverage_status="representative",
        notes="MOT17-04 frame-sequence pedestrian tracking sample",
    )

    obs_result = generate_spatial_observations(
        evidence_context=evidence_context,
        density_grid=density_grid,
    )

    metadata = AnalysisRunMetadata(
        run_id="phase_10_mot17_sample",
        dataset_name="mot17_04_frcnn",
        source_type="image_sequence",
        target_fps=None,
        total_frames_evaluated=len(mot17_frames),
        source_path=mot17_dir,
        notes="Local MOT17 image sequence evaluated through Phase 10 & 12",
    )

    report = build_analysis_report(
        metadata=metadata,
        evidence_context=evidence_context,
        dwell_summary=None,  # Unsupplied to demonstrate NOT_AVAILABLE in viewer
        density_grid=density_grid,
        transition_summary=None,  # Unsupplied to demonstrate NOT_AVAILABLE in viewer
        observation_result=obs_result,
        report_id="phase_13_analysis_report_mot17_sample",
    )

    os.makedirs(reports_dir, exist_ok=True)
    out_file = os.path.join(reports_dir, "phase_13_analysis_report_mot17_sample.json")
    export_analysis_report(report, output_path=out_file)
    print(f"  Exported Analysis Report: {out_file} (Schema v{report.schema_version})")

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Reproducible Offline Sample-Analysis Pipeline")
    parser.add_argument("--dataset", choices=["merl", "mot17", "all"], default="all", help="Dataset sample to analyze")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--reports-dir", default="outputs/reports", help="Reports output directory")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Model inference smoke check
    run_detector_inference_smoke_check()

    # 2. Run selected pipelines
    if args.dataset in ("merl", "all"):
        run_merl_sample_analysis(config, reports_dir=args.reports_dir)
    if args.dataset in ("mot17", "all"):
        run_mot17_sample_analysis(config, reports_dir=args.reports_dir)

    print("\n" + "=" * 75)
    print("ALL SAMPLE ANALYSIS PIPELINES COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()
