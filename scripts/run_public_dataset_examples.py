"""Build reproducible, offline examples from locally fetched public datasets.

The examples deliberately keep incompatible ground-truth types separate. The
EgoCart example ranks annotated aisle sections and a 0.5 m sampled-location
hotspot. The CAVIAR example runs this repository's single-camera detector,
tracker, sampler and homography on a shopping-centre corridor clip. The Mall
example visualizes supplied head-point labels only. No model is trained here.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection import YOLO11PersonDetector
from src.spatial.homography import HomographyCalibrator
from src.tracking import ByteTracker, Track
from src.trajectory import TrajectoryExtractor
from src.video import FrameSampler


EXTERNAL = ROOT / "data" / "external"
REPORTS = ROOT / "outputs" / "reports"
CONFIG_PATH = ROOT / "config.yaml"


def _ensure_output_dir() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)


def _save_egocart_example() -> Dict[str, Any]:
    """Rank annotated EgoCart aisle sections and find an observed point cell."""
    base = EXTERNAL / "egocart"
    frames_payload = json.loads((base / "frames.json").read_text(encoding="utf-8"))["frames"]
    samples_payload = json.loads((base / "samples.json").read_text(encoding="utf-8"))["samples"]
    sample = next(s for s in samples_payload if str(s.get("sequence_id")) == "0")
    sample_id = sample["_id"]["$oid"]
    sequence = [r for r in frames_payload if r.get("_sample_id", {}).get("$oid") == sample_id]
    sequence.sort(key=lambda r: int(r["frame_number"]))
    fps = float(sample["metadata"]["frame_rate"])
    target_fps = 5.0
    step = max(1, round(fps / target_fps))
    sampled = [r for r in sequence if (int(r["frame_number"]) - 1) % step == 0]

    # EgoCart documents zones 1–9 as sequential aisle sections. Zones 10–16
    # are structural/perimeter regions and are not included in aisle ranking.
    aisle_counts = Counter(int(r["zone_id"]) for r in sampled if 1 <= int(r["zone_id"]) <= 9)
    ranked = sorted(
        ({"zone_id": zid, "sampled_frames": int(aisle_counts.get(zid, 0)),
          "observed_frame_span_sec": round(aisle_counts.get(zid, 0) / target_fps, 3)}
         for zid in range(1, 10)),
        key=lambda row: (-row["sampled_frames"], row["zone_id"]),
    )
    active = [r for r in ranked if r["sampled_frames"] > 0]
    candidate = active[0] if active else None

    # Fixed published map extents and 0.5 m cells; never fit bounds to points.
    bounds = (-20.2, 19.8, -9.5, 8.0)
    cell = 0.5
    selected_points = [(float(r["location_x"]), float(r["location_y"]))
                       for r in sampled if candidate and int(r["zone_id"]) == candidate["zone_id"]]
    nx = math.ceil((bounds[1] - bounds[0]) / cell)
    ny = math.ceil((bounds[3] - bounds[2]) / cell)
    grid = np.zeros((ny, nx), dtype=np.int32)
    for x, y in selected_points:
        if bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3]:
            col = min(nx - 1, int((x - bounds[0]) / cell))
            row = min(ny - 1, int((y - bounds[2]) / cell))
            grid[row, col] += 1
    hotspot = None
    if grid.any():
        max_count = int(grid.max())
        candidates = np.argwhere(grid == max_count)
        row, col = min((int(r), int(c)) for r, c in candidates)
        x0, y0 = bounds[0] + col * cell, bounds[2] + row * cell
        hotspot = {"x_min_m": round(x0, 3), "x_max_m": round(x0 + cell, 3),
                   "y_min_m": round(y0, 3), "y_max_m": round(y0 + cell, 3),
                   "center_x_m": round(x0 + cell / 2, 3),
                   "center_y_m": round(y0 + cell / 2, 3),
                   "sampled_pose_points": max_count}

    # Floor trajectory plot: colors reflect dataset aisle-zone IDs.
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for zid in range(1, 10):
        pts = [(float(r["location_x"]), float(r["location_y"])) for r in sampled
               if int(r["zone_id"]) == zid]
        if pts:
            arr = np.asarray(pts)
            ax.scatter(arr[:, 0], arr[:, 1], s=15, alpha=.65,
                       label=f"Aisle section {zid} ({len(pts)} sampled poses)")
    if hotspot:
        ax.scatter([hotspot["center_x_m"]], [hotspot["center_y_m"]], marker="*",
                   s=300, c="#ff1744", edgecolors="white", linewidths=1.2,
                   label="Highest-count cell within selected aisle")
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3])
    ax.set_xlabel("Dataset ground-truth X (m)"); ax.set_ylabel("Dataset ground-truth Y (m)")
    ax.set_title("EgoCart sequence 0 — sampled cart-camera poses by labeled aisle section\n"
                 "One supermarket sequence; not pedestrian tracks or product demand")
    ax.grid(alpha=.2); ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(REPORTS / "public_egocart_floor_trajectory.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ids = [str(r["zone_id"]) for r in ranked]
    vals = [r["sampled_frames"] for r in ranked]
    colors = ["#ef4444" if candidate and r["zone_id"] == candidate["zone_id"] else "#3b82f6" for r in ranked]
    ax.bar(ids, vals, color=colors)
    ax.set_xlabel("Dataset-labeled aisle section (zone 1–9)")
    ax.set_ylabel("Sampled ground-truth pose frames")
    ax.set_title("Aisle-section sample counts — one EgoCart sequence")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(REPORTS / "public_egocart_aisle_ranking.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.imshow(grid, origin="lower", extent=bounds, aspect="auto", interpolation="nearest", cmap="magma")
    if hotspot:
        ax.scatter([hotspot["center_x_m"]], [hotspot["center_y_m"]], marker="*", s=320,
                   c="#00e5ff", edgecolors="black", linewidths=1.0)
    ax.set_xlabel("Dataset ground-truth X (m)"); ax.set_ylabel("Dataset ground-truth Y (m)")
    ax.set_title((f"Highest-count 0.5 m cell inside aisle section {candidate['zone_id']}\n"
                  "Sampled cart-camera poses; not product interest or shopper dwell") if candidate else
                 "No sampled poses in labeled aisle sections")
    fig.colorbar(ax.images[0], ax=ax, label="Sampled pose points per cell")
    fig.tight_layout(); fig.savefig(REPORTS / "public_egocart_hotspot.png", dpi=150); plt.close(fig)

    cap = cv2.VideoCapture(str(base / "data" / "egocart_seq_0.mp4"))
    cap.release()
    return {
        "dataset_id": "egocart_seq_0", "venue": "supermarket",
        "source_type": "recorded_cart_camera_video_with_pose_and_zone_ground_truth",
        "source_video": "data/external/egocart/data/egocart_seq_0.mp4",
        "sequence_id": "0", "sequence_split": sample.get("tags", []),
        "encoded_fps": fps, "analysis_target_fps": target_fps,
        "source_frame_count": len(sequence), "sampled_pose_count": len(sampled),
        "sampled_frame_step": step, "sampled_duration_sec_at_encoded_fps": round(len(sampled) / target_fps, 3),
        "coordinate_system": "dataset ground-truth store coordinates in metres",
        "grid_bounds_m": {"min_x": bounds[0], "max_x": bounds[1], "min_y": bounds[2], "max_y": bounds[3]},
        "grid_cell_size_m": cell,
        "aisle_section_ranking": ranked,
        "candidate": {
            "status": "exploratory_observed_path_candidate" if candidate else "no_candidate",
            "selected_aisle_section": candidate["zone_id"] if candidate else None,
            "selection_rule": "highest sampled ground-truth cart-pose frame count among labeled aisle sections 1–9; ties by zone ID",
            "within_aisle_hotspot": hotspot,
            "validated_actionable_recommendation": "WITHHELD",
            "limitation": "One sequence from one supermarket; camera is mounted on a shopping cart. Pose presence is not a shopper identity, product interaction, dwell, demand, or sales outcome."
        },
        "artifacts": {"floor_trajectory": "outputs/reports/public_egocart_floor_trajectory.png",
                      "aisle_ranking": "outputs/reports/public_egocart_aisle_ranking.png",
                      "hotspot": "outputs/reports/public_egocart_hotspot.png"},
    }


def _save_mall_example() -> Dict[str, Any]:
    """Render the Mall dataset's provided frame-wise head-point ground truth."""
    mall_dir = EXTERNAL / "mall_crowd" / "extracted" / "mall_dataset"
    mat = loadmat(mall_dir / "mall_gt.mat", squeeze_me=True, struct_as_record=False)
    gt_frames = np.asarray(mat["frame"], dtype=object).reshape(-1)
    counts = np.asarray(mat["count"], dtype=np.int64).reshape(-1)
    points = [np.asarray(f.loc, dtype=np.float64).reshape(-1, 2) for f in gt_frames]
    total_gt_points = int(sum(len(p) for p in points))
    declared_count_total = int(counts.sum())
    if total_gt_points != declared_count_total:
        raise ValueError(f"Mall ground-truth count mismatch: points={total_gt_points}, counts={declared_count_total}")
    all_points = np.concatenate(points, axis=0)
    histogram, xedges, yedges = np.histogram2d(all_points[:, 0], all_points[:, 1], bins=(32, 24),
                                                range=((0, 640), (0, 480)))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(histogram.T, origin="upper", extent=(0, 640, 480, 0), aspect="auto",
              interpolation="nearest", cmap="magma")
    ax.set_xlim(0, 640); ax.set_ylim(480, 0)
    ax.set_xlabel("Image X (pixels)"); ax.set_ylabel("Image Y (pixels)")
    ax.set_title("Mall Dataset — aggregate annotated head-point density\n"
                 "2000 frames; not person trajectories, dwell, or retail-zone visits")
    fig.colorbar(ax.images[0], ax=ax, label="Annotated head points across all frames")
    fig.tight_layout(); fig.savefig(REPORTS / "public_mall_head_density.png", dpi=150); plt.close(fig)

    first_frame = cv2.imread(str(mall_dir / "frames" / "seq_000001.jpg"))
    if first_frame is None:
        raise FileNotFoundError("Mall first frame missing/unreadable")
    for x, y in points[0]:
        cv2.circle(first_frame, (int(round(x)), int(round(y))), 5, (0, 0, 255), -1)
    cv2.imwrite(str(REPORTS / "public_mall_annotated_frame.jpg"), first_frame)
    return {
        "dataset_id": "mall_crowd", "venue": "shopping_mall_webcam",
        "source_type": "image_sequence_with_ground_truth_head_points",
        "source_frames_dir": "data/external/mall_crowd/extracted/mall_dataset/frames",
        "frame_count": int(len(gt_frames)), "frame_size_px": [640, 480],
        "annotated_head_points": total_gt_points, "ground_truth_count_sum": declared_count_total,
        "count_conservation": total_gt_points == declared_count_total,
        "per_frame_count_range": {"min": int(counts.min()), "max": int(counts.max()),
                                   "mean": round(float(counts.mean()), 3)},
        "perspective_map_available": (mall_dir / "perspective_roi.mat").exists(),
        "physical_metric_map_available": False,
        "supports": "Crowd counting and image-coordinate head-point density only.",
        "does_not_support": "No persistent person IDs, dwell episodes, calibrated floor locations, aisle rows, or product placement recommendations.",
        "artifacts": {"head_density": "outputs/reports/public_mall_head_density.png",
                      "annotated_frame": "outputs/reports/public_mall_annotated_frame.jpg"},
    }


def _run_caviar_pipeline(config: Dict[str, Any], max_sampled_frames: int = 472) -> Dict[str, Any]:
    """Run this project's single-camera CV chain on one public mall clip."""
    clip = EXTERNAL / "caviar_shopping_centre" / "WalkByShop1cor.mpg"
    xml_path = EXTERNAL / "caviar_shopping_centre" / "cwbs1gt.xml"
    if not clip.is_file() or not xml_path.is_file():
        raise FileNotFoundError("CAVIAR sample video/annotation was not downloaded")
    # VideoInspector rejects .mpg by extension. Transcode the downloaded MPEG2
    # container once to a local MP4, then pass that MP4 through the repository's
    # FrameSampler. Count decoded source frames rather than trusting the MPG
    # container's incorrect CAP_PROP_FRAME_COUNT metadata.
    processing_clip = REPORTS / "public_caviar_source.mp4"
    source_cap = cv2.VideoCapture(str(clip))
    if not source_cap.isOpened():
        raise ValueError(f"Cannot open downloaded CAVIAR source clip: {clip}")
    source_fps = float(source_cap.get(cv2.CAP_PROP_FPS) or 25.0)
    source_frame_count = 0
    convert_writer = None
    try:
        while True:
            ok, source_frame = source_cap.read()
            if not ok or source_frame is None:
                break
            if convert_writer is None:
                height, width = source_frame.shape[:2]
                convert_writer = cv2.VideoWriter(str(processing_clip),
                                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                                 source_fps, (width, height))
                if not convert_writer.isOpened():
                    raise RuntimeError("OpenCV could not create the local CAVIAR MP4 conversion")
            convert_writer.write(source_frame)
            source_frame_count += 1
    finally:
        source_cap.release()
        if convert_writer is not None:
            convert_writer.release()
    if source_frame_count != 2360:
        raise ValueError(f"CAVIAR source decode produced {source_frame_count} frames; XML has 2360 annotated frames")
    # The source states the clip's annotated pixel-to-ground correspondences.
    # Use the first four as calibration and the next four as a holdout check.
    image_points = [[91, 163], [241, 163], [98, 266], [322, 265]]
    floor_points = [[0, 9.75], [2.90, 9.75], [0, -1.10], [2.90, -1.10]]
    check_image = [[60, 153], [359, 153], [50, 201], [367, 200]]
    check_floor = [[0, 0], [0, 9.75], [3.82, 0.98], [3.82, 8.78]]
    calibrator = HomographyCalibrator(image_points, floor_points,
                                      check_points_image=check_image,
                                      check_points_floor=check_floor,
                                      unit="metres", dataset_name="caviar_walkbyshop1_corridor")
    det_cfg = config.get("detection", {})
    trk_cfg = config.get("tracking", {})
    target_fps = float(config.get("video", {}).get("sampling", {}).get("target_fps", 5.0))
    detector = YOLO11PersonDetector(
        model_path=config.get("detection", {}).get("model_path", "data/models/yolo11n.onnx"),
        conf_threshold=det_cfg.get("conf_threshold", .40), iou_threshold=det_cfg.get("iou_threshold", .45))
    Track.reset_id_counter(0)
    tracker = ByteTracker(track_high_thresh=trk_cfg.get("track_high_thresh", .50),
                          track_low_thresh=trk_cfg.get("track_low_thresh", .10),
                          new_track_thresh=trk_cfg.get("new_track_thresh", .50),
                          match_thresh=trk_cfg.get("match_thresh", .80),
                          max_time_lost_sec=trk_cfg.get("max_time_lost_sec", 2.0))
    writer_path = REPORTS / "public_caviar_tracking_preview.mp4"
    writer = None
    sampled = detections = 0
    start_time = time.time()
    for frame_idx, timestamp, frame in FrameSampler(target_fps=target_fps,
                                                     max_sampled_frames=max_sampled_frames).sample_frames(str(processing_clip)):
        result = detector.detect(frame, frame_id=frame_idx, timestamp_sec=timestamp)
        tracker.update(result.detections, frame_id=frame_idx, timestamp_sec=timestamp)
        detections += len(result.detections)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (w, h))
        annotated = ByteTracker.annotate_tracks(frame, tracker.tracked_tracks)
        if writer.isOpened():
            writer.write(annotated)
        sampled += 1
    if writer is not None:
        writer.release()
    tracks = tracker.tracked_tracks + tracker.lost_tracks + tracker.removed_tracks
    trajectories = TrajectoryExtractor.extract_from_tracks(tracks)
    # The published check correspondences produce a very large residual in
    # this implementation. Keep image-space tracking as the demo and withhold
    # any floor projection, density grid, or spatial interpretation.
    xml_root = ET.parse(xml_path).getroot()
    annotated_frames = xml_root.findall("frame")
    gt_ids = set()
    for fr in annotated_frames:
        for obj in fr.findall("./objectlist/object"):
            oid = obj.attrib.get("id")
            if oid is not None:
                gt_ids.add(oid)
    return {
        "dataset_id": "caviar_walkbyshop1_corridor", "venue": "shopping_centre_corridor",
        "source_type": "recorded_video_with_frame_level_person_annotations",
        "source_video": "data/external/caviar_shopping_centre/WalkByShop1cor.mpg",
        "converted_reader_video": "outputs/reports/public_caviar_source.mp4",
        "source_container": "MPEG2 .mpg; decoded and re-encoded locally to MP4 because repository VideoInspector rejects .mpg extensions",
        "reader": "repository FrameSampler using OpenCV VideoCapture on the local MP4 conversion",
        "source_decoded_frames": source_frame_count,
        "decoded_sampled_frames": sampled, "sampling_target_fps": target_fps,
        "annotation_frames": len(annotated_frames), "annotated_person_ids": len(gt_ids),
        "project_model_inference": "YOLO11-n ONNX; pretrained inference weights; not trained by this project",
        "detections": detections, "clip_local_tracks": len(trajectories),
        "trajectory_points": sum(len(t.points) for t in trajectories),
        "tracking_accuracy_status": "not_quantified; dataset annotation schema not evaluated against tracker output in this run",
        "homography_diagnostics": calibrator.diagnostics.to_dict(),
        "spatial_projection_status": "withheld_failed_check_point_residual",
        "spatial_projection_reason": "Configured check-point residual is too large for spatial interpretation; no CAVIAR floor density is presented.",
        "zone_dwell_recommendation_status": "not_available; no validated store aisle polygons are supplied for this corridor clip",
        "processing_sec": round(time.time() - start_time, 3),
        "artifacts": {"tracking_preview": "outputs/reports/public_caviar_tracking_preview.mp4"},
        "evidence_limitations": ["This is a shopping-centre corridor, not a merchandise aisle.",
                                  "The published correspondences are not an independent physical survey.",
                                  "No product placement recommendation is supported."],
    }


def run() -> Dict[str, Any]:
    _ensure_output_dir()
    import yaml
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    egocart = _save_egocart_example()
    mall = _save_mall_example()
    caviar = _run_caviar_pipeline(config)

    prcv_dir = EXTERNAL / "prcv_site03" / "testcases" / "006_site03_tracks"
    prcv = json.loads((prcv_dir / "ground-truth.json").read_text(encoding="utf-8"))
    prcv_tracks = prcv["track_results"]["tracks"]
    prcv_summary = {
        "dataset_id": "prcv_site03_006", "venue": "retail_store",
        "source_type": "recorded_stereo_video_with_supplied_floor_tracks",
        "source_video": "data/external/prcv_site03/testcases/006_site03_tracks/C0001020_v1.mp4",
        "video_frame_count": 400, "native_fps": 15.0,
        "video_frame_size_px": [1792, 672], "camera_encoding": "side-by-side stereo pair; select one 896px sensor half for single-view analysis",
        "provided_track_records": len(prcv_tracks),
        "provided_track_ids": [t.get("id") for t in prcv_tracks],
        "provided_track_observations": sum(len(t.get("path", [])) for t in prcv_tracks),
        "provided_ground_truth": "Source tracker annotations; not a score of this repository's detector/tracker.",
        "scope_note": "One scenario from an anonymized real retail site; no shelf-row polygons included in this downloaded subset.",
    }
    report = {
        "report_type": "public_dataset_examples_and_scope",
        "generated_by": "scripts/run_public_dataset_examples.py",
        "sources": {
            "MERL": {"venue": "supermarket", "availability": "existing project clip; previously run through full pipeline",
                     "limitation": "Sparse/boundary-truncated; do not treat the existing sample as representative."},
            "EgoCart": egocart,
            "PRCV": prcv_summary,
            "CAVIAR": caviar,
            "Mall": mall,
        },
        "recommendation_interpretation": "Only the EgoCart annotated aisle-zone proxy produces an exploratory aisle/position candidate. It is a single cart-camera sequence and does not establish shopper demand or product placement efficacy.",
        "validated_actionable_recommendations": "WITHHELD",
    }
    out = REPORTS / "public_dataset_examples_summary.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(out),
                      "egocart_candidate": egocart["candidate"],
                      "mall": {"frames": mall["frame_count"], "head_points": mall["annotated_head_points"]},
                      "caviar": {"sampled_frames": caviar["decoded_sampled_frames"],
                                 "detections": caviar["detections"],
                                 "tracks": caviar["clip_local_tracks"],
                                 "points": caviar["trajectory_points"],
                                 "spatial_projection_status": caviar["spatial_projection_status"]}}, indent=2))
    return report


if __name__ == "__main__":
    run()
