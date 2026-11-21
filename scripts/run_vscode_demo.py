"""
Reproducible offline end-to-end demo for the locally available MERL video.
Runs the existing detector, tracker, trajectory, configured homography, zone,
footfall, dwell, density, transition, and evidence-gating components. The
YOLO11-n ONNX weights are pretrained; this project does not train them.
Recommendations are always withheld for the current unvalidated evidence.
"""
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle as MplRectangle
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath("."))

from src.analytics.density import GridBounds, compute_density_grid, render_density_heatmap
from src.analytics.dwell import DwellSummary, analyze_dwell
from src.analytics.footfall import FootfallAnalyzer
from src.analytics.spatial_observations import EvidenceContext, generate_spatial_observations
from src.analytics.transitions import compute_direct_transitions
from src.analytics.zone import Zone, ZoneConfig
from src.analytics.zone_evaluator import ZoneEvaluator
from src.detection import YOLO11PersonDetector
from src.spatial import HomographyCalibrator
from src.tracking import ByteTracker, Track
from src.trajectory import Trajectory, TrajectoryExtractor, TrajectoryPoint
from src.video import FrameSampler
from src.video.inspector import VideoInspector
from src.video.quality import QualityEvaluator


def render_floor_trajectory_diagram(
    zones: Sequence[Zone], trajectories: Sequence[Trajectory], output_path: str,
    title: str = "Floor-Plane Trajectory Diagram",
    grid_bounds: Optional[Tuple[float, float, float, float]] = (0.0, 10.0, 0.0, 6.0),
    unit_label: str = "configured floor units (unverified metric scale)",
    subtitle: str = "", is_synthetic: bool = False,
    highlight_zone_id: Optional[str] = None,
    highlight_hotspot: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    for zone in zones:
        polygon = np.asarray(zone.polygon)
        highlighted = zone.zone_id == highlight_zone_id
        ax.add_patch(MplPolygon(
            polygon, closed=True, facecolor=zone.color, alpha=.36 if highlighted else .18,
            edgecolor="#ff2d55" if highlighted else zone.color,
            linewidth=4 if highlighted else 2,
            linestyle="-" if highlighted else "--",
        ))
        center = polygon.mean(axis=0)
        ax.text(center[0], center[1], f"★ CANDIDATE: {zone.name}" if highlighted else zone.name,
                ha="center", va="center", fontsize=8,
                bbox={"facecolor":"white","alpha":.8,"edgecolor":zone.color})
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(trajectories))))
    for i, trajectory in enumerate(trajectories):
        points = [p for p in trajectory.points if p.floor_x is not None and p.floor_y is not None]
        if not points:
            continue
        xs = [p.floor_x for p in points]
        ys = [p.floor_y for p in points]
        ax.plot(xs, ys, marker=".", linewidth=1.5, color=colors[i % len(colors)],
                label=f"Track {trajectory.track_id} ({len(points)} sampled points)")
        ax.scatter(xs[0], ys[0], marker="o", color="green", zorder=5)
        ax.scatter(xs[-1], ys[-1], marker="s", color="red", zorder=5)
    if highlight_hotspot:
        hx,hy=float(highlight_hotspot["center_x"]),float(highlight_hotspot["center_y"])
        ax.scatter([hx],[hy],marker="*",s=260,color="#ff1744",edgecolor="white",linewidth=1.2,zorder=10,label="Observed hotspot cell")
        ax.annotate(f"Hotspot: {highlight_hotspot['sampled_points']} sampled points",(hx,hy),
                    xytext=(8,8),textcoords="offset points",fontsize=8,
                    bbox={"facecolor":"white","alpha":.88,"edgecolor":"#ff1744"})
    if grid_bounds:
        min_x, max_x, min_y, max_y = grid_bounds
        all_x = [p.floor_x for tr in trajectories for p in tr.points if p.floor_x is not None]
        all_y = [p.floor_y for tr in trajectories for p in tr.points if p.floor_y is not None]
        plot_min_x = min([min_x] + all_x) - .5
        plot_max_x = max([max_x] + all_x) + .5
        plot_min_y = min([min_y] + all_y) - .5
        plot_max_y = max([max_y] + all_y) + .5
        ax.set_xlim(plot_min_x, plot_max_x)
        ax.set_ylim(plot_min_y, plot_max_y)
        ax.add_patch(MplRectangle(
            (min_x, min_y), max_x - min_x, max_y - min_y,
            fill=False, edgecolor="#222222", linewidth=1.2, linestyle=":",
            label="Configured floor bounds", zorder=6,
        ))
    ax.set_xlabel(f"Floor X ({unit_label})")
    ax.set_ylabel(f"Floor Y ({unit_label})")
    display_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(display_title, pad=14)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=.5)
    if trajectories or grid_bounds:
        ax.legend(loc="upper right", fontsize=8)
    if is_synthetic:
        fig.text(.5, .01, "SYNTHETIC DEMONSTRATION — NOT REAL SHOPPER DATA", ha="center", color="red")
    else:
        fig.text(.5, .01, "Configured coordinates; physical scale is not independently surveyed",
                 ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_zone_hotspot(points,bounds=(0.0,10.0,0.0,6.0),cell_size=0.5):
    # Highest-count cell for already zone-assigned points; coordinates remain configured units.
    min_x,max_x,min_y,max_y=(float(v) for v in bounds)
    if cell_size<=0 or max_x<=min_x or max_y<=min_y:raise ValueError("Invalid hotspot bounds or cell size.")
    cols,rows=int(np.ceil((max_x-min_x)/cell_size)),int(np.ceil((max_y-min_y)/cell_size))
    counts={}
    for x,y in points:
        x,y=float(x),float(y)
        if min_x<=x<=max_x and min_y<=y<=max_y:
            col=min(int((x-min_x)/cell_size),cols-1);row=min(int((y-min_y)/cell_size),rows-1)
            counts[(row,col)]=counts.get((row,col),0)+1
    if not counts:return None
    (row,col),n=min(counts.items(),key=lambda item:(-item[1],item[0][0],item[0][1]))
    x0,y0=min_x+col*cell_size,min_y+row*cell_size
    return {"grid_row":row,"grid_column":col,"x_min":round(x0,4),"x_max":round(min(x0+cell_size,max_x),4),
            "y_min":round(y0,4),"y_max":round(min(y0+cell_size,max_y),4),
            "center_x":round((x0+min(x0+cell_size,max_x))/2,4),"center_y":round((y0+min(y0+cell_size,max_y))/2,4),
            "sampled_points":n,"unit":"configured floor units",
            "interpretation":"Highest sampled trajectory-point cell in this zone; not a validated demand location."}


def select_exploratory_product_zone_candidate(
    zone_ids: Sequence[str],
    observed_point_counts: Dict[str, int],
    dwell_by_zone: Dict[str, Any],
    transition_by_zone: Optional[Dict[str, Any]] = None,
    hotspot_by_zone: Optional[Dict[str, Dict[str, Any]]] = None,
    objective: str = "observed_dwell",
    candidate_zone_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Choose a transparent, unvalidated shelf candidate using observed counts only.

    Ordering is observed dwell span, then in-zone sampled points, then visit
    episodes, then stable zone ID. No thresholds, learned weights, or causal
    claims are applied. This is an exploratory placement prompt, not an
    actionable or validated recommendation.
    """
    if objective not in {"observed_dwell","sampled_points","visit_episodes"}:
        raise ValueError("objective must be observed_dwell, sampled_points, or visit_episodes")
    shelf_ids=sorted(candidate_zone_ids if candidate_zone_ids is not None else (zone_id for zone_id in zone_ids if zone_id.startswith("shelf_")))
    rows = []
    for zone_id in shelf_ids:
        dwell = dwell_by_zone.get(zone_id)
        if isinstance(dwell, dict):
            dwell_span = float(dwell.get("observed_dwell_span_sec", dwell.get("total_dwell_sec", 0.0)))
            episodes = int(dwell.get("episodes", dwell.get("total_episodes", 0)))
        else:
            dwell_span = float(getattr(dwell, "total_dwell_sec", 0.0))
            episodes = int(getattr(dwell, "total_episodes", 0))
        trans = (transition_by_zone or {}).get(zone_id, {})
        runs = int(trans.get("observed_run_count", 0)) if isinstance(trans, dict) else int(getattr(trans, "observed_run_count", 0))
        rows.append({
            "zone_id": zone_id,
            "observed_dwell_span_sec": round(dwell_span, 3),
            "sampled_in_zone_points": int(observed_point_counts.get(zone_id, 0)),
            "observed_visit_episodes": episodes,
            "observed_runs": runs,
        })
    order={"observed_dwell":("observed_dwell_span_sec","sampled_in_zone_points","observed_visit_episodes"),
           "sampled_points":("sampled_in_zone_points","observed_dwell_span_sec","observed_visit_episodes"),
           "visit_episodes":("observed_visit_episodes","observed_dwell_span_sec","sampled_in_zone_points")}[objective]
    rows.sort(key=lambda row:tuple(-row[k] for k in order)+(row["zone_id"],))
    active_rows=[r for r in rows if any(r[k]>0 for k in ("observed_dwell_span_sec","sampled_in_zone_points","observed_visit_episodes"))]
    selected = rows[0] if rows and any(
        row["observed_dwell_span_sec"] or row["sampled_in_zone_points"] or row["observed_visit_episodes"]
        for row in rows
    ) else None
    hotspot=(hotspot_by_zone or {}).get(selected["zone_id"]) if selected and len(active_rows)==1 else None
    if len(active_rows)==1:
        mode="within_zone_hotspot";rule="Only one shelf zone had observations; its hotspot is the configured cell with most sampled points (ties: row, then column)."
    elif active_rows:
        mode="multi_zone_comparison"
        labels={"observed_dwell":"observed dwell span","sampled_points":"sampled in-zone points","visit_episodes":"observed visit episodes"}
        rule=f"Among active shelf zones, prioritize {labels[objective]}; ties use other observed metrics, then zone ID. No thresholds or learned weights."
    else:mode,rule="no_observed_activity","No shelf zone had observed activity; no candidate is emitted."
    return {"status":"exploratory_unvalidated_candidate" if selected else "no_observed_shelf_activity",
            "recommendation_mode":mode,"objective":objective,"candidate_zone_id":selected["zone_id"] if selected else None,
            "candidate_metrics":selected,"candidate_hotspot":hotspot,"comparison_by_shelf_zone":rows,
            "selection_rule":rule,"interpretation":"Observed traffic candidate only; hotspot is sampled trajectory density, not product demand or a proven outcome.",
            "actionable_recommendation":"WITHHELD"}


def render_zone_dwell_barchart(
    dwell_summary: DwellSummary, output_path: str,
    title: str = "Zone Dwell & Visit Summary", is_synthetic: bool = False,
    source_label: str = "",
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    zones = list(dwell_summary.by_zone)
    spans = [dwell_summary.by_zone[z].total_dwell_sec for z in zones]
    episodes = [dwell_summary.by_zone[z].total_episodes for z in zones]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=150)
    ys = np.arange(len(zones))
    ax1.barh(ys, spans, color="#3498db")
    ax1.set_yticks(ys, zones)
    ax1.set_xlabel("Observed dwell span (seconds)")
    ax1.set_title("Observed sampled span (not continuous dwell)")
    ax1.grid(True, axis="x", linestyle=":")
    ax2.barh(ys, episodes, color="#e67e22")
    ax2.set_yticks(ys, zones)
    ax2.set_xlabel("Observed episodes (count)")
    ax2.set_title("Zone visit episodes")
    ax2.grid(True, axis="x", linestyle=":")
    fig.suptitle(title)
    note = "SYNTHETIC DEMONSTRATION — NOT REAL SHOPPER DATA" if is_synthetic else \
           f"{source_label}: clip-local tracker observations; no accuracy validation"
    fig.text(.5, .01, note, ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_zone_occupancy_timeline(
    timeline: Dict[int, Dict[str, int]], output_path: str, native_fps: float,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    frames = sorted(timeline)
    zones = sorted({z for values in timeline.values() for z in values})
    for zone_id in zones:
        ax.step([frame / native_fps for frame in frames],
                [timeline[frame].get(zone_id, 0) for frame in frames],
                where="post", label=zone_id)
    ax.set(xlabel="Video time (seconds)", ylabel="Clip-local tracked objects in zone",
           title="Sampled Zone Occupancy Timeline (Not Physical Occupancy)")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=":")
    if zones:
        ax.legend(title="Configured zone")
    fig.text(.5, .01, "Sampled tracker counts; not unique people or validated occupancy",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_3d_analytics_overview(density,dwell_summary,occupancy,native_fps,output_path,zones=(),trajectories=()):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    os.makedirs(os.path.dirname(os.path.abspath(output_path)),exist_ok=True)
    fig=plt.figure(figsize=(15,10),dpi=140)
    ax=fig.add_subplot(221,projection="3d")
    for zone in zones:
        poly=np.asarray(zone.polygon,dtype=float)
        ax.add_collection3d(Poly3DCollection([[(float(x),float(y),0.0) for x,y in poly]],
            facecolors=zone.color,edgecolors=zone.color,alpha=.22))
        closed=np.vstack((poly,poly[0]))
        ax.plot(closed[:,0],closed[:,1],np.zeros(len(closed)),color=zone.color,label=zone.name)
    for trajectory in trajectories:
        pts=[p for p in trajectory.points if p.floor_x is not None and p.floor_y is not None]
        if pts:ax.plot([p.floor_x for p in pts],[p.floor_y for p in pts],np.full(len(pts),.015),linewidth=1,label=f"Track {trajectory.track_id}")
    ax.set(title="Homography-mapped floor plane and tracks",xlabel="Floor X (configured units)",ylabel="Floor Y (configured units)",zlabel="Plane offset")
    ax.set_zlim(0,.12)

    ax=fig.add_subplot(222,projection="3d");b=density.bounds
    xs=np.linspace(b.min_x+density.cell_size_x/2,b.max_x-density.cell_size_x/2,density.num_bins_x)
    ys=np.linspace(b.min_y+density.cell_size_y/2,b.max_y-density.cell_size_y/2,density.num_bins_y)
    xx,yy=np.meshgrid(xs,ys);ax.plot_surface(xx,yy,density.counts.astype(float),cmap="viridis",linewidth=0)
    ax.set(title="Sampled density surface",xlabel="Floor X (configured units)",ylabel="Floor Y (configured units)",zlabel="Points / cell")

    ax=fig.add_subplot(223,projection="3d");names=list(dwell_summary.by_zone);pos=np.arange(len(names))
    spans=[float(dwell_summary.by_zone[z].total_dwell_sec) for z in names]
    ax.bar3d(pos,np.zeros(len(pos)),np.zeros(len(pos)),.65,.55,spans,color="#4c9be8")
    ax.set_xticks(pos,names,rotation=18,ha="right");ax.set(title="Observed dwell spans",ylabel="Zone",zlabel="Observed seconds")

    ax=fig.add_subplot(224,projection="3d");frames=sorted(occupancy);zones_t=sorted({z for f in frames for z in occupancy[f]})
    if frames and zones_t:
        groups=np.array_split(np.asarray(frames),min(24,len(frames)));bx=[];by=[];bh=[]
        for group in groups:
            if not len(group):continue
            for zi,name in enumerate(zones_t):
                peak=max(occupancy[int(f)].get(name,0) for f in group)
                if peak:bx.append(float(np.mean(group))/max(float(native_fps),1e-9));by.append(zi);bh.append(float(peak))
        if bh:ax.bar3d(bx,by,np.zeros(len(bh)),.45,[.55]*len(bh),bh,color="#e78a48")
        ax.set_yticks(np.arange(len(zones_t)),zones_t,rotation=18,ha="right")
    ax.set(title="Sampled zone counts over time",xlabel="Video time (seconds)",ylabel="Configured zone",zlabel="Peak sampled tracks")
    fig.suptitle("3D views — height encodes counts or observed spans, not physical elevation",fontsize=14)
    fig.text(.5,.01,"Homography maps image points to a 2D floor plane. Floor scale is unverified; track IDs are clip-local.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(output_path,bbox_inches="tight");plt.close(fig);return output_path

def _per_track_diagnostics(trajectories, expected_sample_step: int) -> List[Dict[str, Any]]:
    rows = []
    for tr in trajectories:
        pts = tr.points
        frames = [p.frame_idx for p in pts]
        times = [p.timestamp_sec for p in pts]
        gaps = sum(1 for a, b in zip(frames, frames[1:]) if b - a > expected_sample_step)
        span = max(times) - min(times) if times else 0.0
        rows.append({
            "track_id": tr.track_id, "sampled_point_count": len(pts),
            "first_frame_idx": min(frames) if frames else None,
            "last_frame_idx": max(frames) if frames else None,
            "observed_span_sec": round(span, 3),
            "image_path_length_px": tr.calculate_image_distance(),
            "configured_floor_path_length": tr.calculate_floor_distance(),
            "configured_floor_mean_speed_per_sec": round(tr.calculate_floor_distance() / span, 3) if span > 0 else None,
            "unexpected_sample_gaps": gaps,
        })
    return rows


def run_real_data_demo(config: Dict[str, Any], reports_dir: str = "outputs/reports",
                       max_frames: int = 100, video_path: Optional[str] = None,
                       progress_callback: Optional[Any] = None,
                       artifact_prefix: str = "demo_real",
                       use_merl_profile: bool = True,
                       candidate_objective: str = "observed_dwell",
                       scene_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    video_path = video_path or os.path.join("data", "merl", "10_1_crop.mp4")
    model_path = config.get("detection", {}).get("model_path", "data/models/yolo11n.onnx")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"MERL video not found: {video_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Configured ONNX model not found: {model_path}")
    os.makedirs(reports_dir, exist_ok=True)
    metadata = VideoInspector().inspect(video_path)

    # Compute source file fingerprint
    h_sha = hashlib.sha256()
    with open(video_path, "rb") as f_sha:
        while chunk := f_sha.read(65536):
            h_sha.update(chunk)
    source_sha256 = h_sha.hexdigest()
    source_file_size = os.path.getsize(video_path)
    source_file_fingerprint = {
        "sha256": source_sha256,
        "file_size_bytes": source_file_size,
        "file_name": os.path.basename(video_path),
    }

    profile_key = "merl"
    scene_label = "MERL"
    scene_profile_provenance = "not_configured"
    if scene_profile is not None:
        from src.spatial.scene_profile import validate_scene_profile, scene_config
        scene_profile = validate_scene_profile(scene_profile, metadata.width, metadata.height)
        config = scene_config(config, scene_profile)
        profile_key, scene_label = "uploaded_scene", scene_profile["name"]
        use_merl_profile = True
        scene_profile_provenance = "caller_supplied_profile"
    elif use_merl_profile:
        # Load canonical MERL scene profile
        merl_profile_file = os.path.join("configs", "merl_scene_profile.json")
        if os.path.isfile(merl_profile_file):
            with open(merl_profile_file, "r", encoding="utf-8") as f_mp:
                scene_profile = json.load(f_mp)
        else:
            scene_profile = {
                "name": "MERL configured camera",
                "image_width": metadata.width,
                "image_height": metadata.height,
                "bounds": [0.0, 10.0, 0.0, 6.0],
                "cell_size": 0.5,
                "image_points": config.get("homography", {}).get("merl", {}).get("image_points", []),
                "floor_points": config.get("homography", {}).get("merl", {}).get("floor_points", []),
                "zones": config.get("zones", {}).get("merl", []),
                "candidate_zone_ids": ["shelf_a", "shelf_b"],
                "hysteresis_window": config.get("zones", {}).get("hysteresis_window", 3),
            }
        scene_profile_provenance = "merl_preset"

    # Save effective scene profile if available
    scene_profile_path = None
    if scene_profile is not None:
        scene_profile_path = os.path.join(reports_dir, f"{artifact_prefix}_scene_profile.json")
        with open(scene_profile_path, "w", encoding="utf-8") as f_prof:
            json.dump(scene_profile, f_prof, indent=2)

    quality = QualityEvaluator(config.get("video", {}).get("quality_thresholds", {})).evaluate(
        video_path, max_sample_frames=15)
    video_cfg = config.get("video", {})
    target_fps = float(video_cfg.get("sampling", {}).get("target_fps", 5.0))
    det_cfg = config.get("detection", {})
    tracking_cfg = config.get("tracking", {})
    detector = YOLO11PersonDetector(
        model_path=model_path,
        conf_threshold=det_cfg.get("conf_threshold", .40),
        iou_threshold=det_cfg.get("iou_threshold", .45),
    )
    Track.reset_id_counter(0)
    tracker = ByteTracker(
        track_high_thresh=tracking_cfg.get("track_high_thresh", .50),
        track_low_thresh=tracking_cfg.get("track_low_thresh", .10),
        new_track_thresh=tracking_cfg.get("new_track_thresh", .50),
        match_thresh=tracking_cfg.get("match_thresh", .80),
        max_time_lost_sec=tracking_cfg.get("max_time_lost_sec", 2.0),
    )
    annotated_frame_path = os.path.join(reports_dir, f"{artifact_prefix}_tracking_frame.png")
    annotated_video_path = os.path.join(reports_dir, f"{artifact_prefix}_tracking_video.webm")
    writer = None
    sampled_times, inference_ms, confidences = [], [], []
    sampled_frame_records: List[Tuple[int, float]] = []
    detections_total = 0
    first_annotated = None
    started = time.time()
    for sample_count, (frame_idx, timestamp, frame) in enumerate(
            FrameSampler(target_fps=target_fps).sample_frames(video_path, max_frames=max_frames), start=1):
        result = detector.detect(frame, frame_id=frame_idx, timestamp_sec=timestamp)
        tracker.update(result.detections, frame_id=frame_idx, timestamp_sec=timestamp)
        sampled_times.append(float(timestamp))
        sampled_frame_records.append((int(frame_idx), float(timestamp)))
        inference_ms.append(float(result.inference_time_ms))
        confidences.extend(float(d.confidence) for d in result.detections)
        detections_total += len(result.detections)
        if progress_callback is not None and sample_count % 5 == 0:
            progress_callback(min(sample_count / max(1, max_frames), 0.99), sample_count)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(annotated_video_path, cv2.VideoWriter_fourcc(*"VP80"),
                                     target_fps, (w, h))
        annotated = ByteTracker.annotate_tracks(frame, tracker.tracked_tracks)
        if writer.isOpened():
            writer.write(annotated)
        if first_annotated is None and tracker.tracked_tracks:
            first_annotated = annotated.copy()
    processing_sec = time.time() - started
    video_written = bool(writer is not None and writer.isOpened())
    if writer is not None:
        writer.release()
    if first_annotated is not None:
        cv2.imwrite(annotated_frame_path, first_annotated)

    all_tracks = tracker.tracked_tracks + tracker.lost_tracks + tracker.removed_tracks
    trajectories = TrajectoryExtractor.extract_from_tracks(all_tracks)
    if progress_callback is not None:
        try:
            progress_callback(0.92, sample_count, "Analyzing trajectories, dwell times, and zones...")
        except TypeError:
            progress_callback(0.92, sample_count)
    if not use_merl_profile:
        summary = {
            "dataset_name": "user_uploaded_indoor_video",
            "scene_profile": None,
            "scene_profile_provenance": scene_profile_provenance,
            "source_file_fingerprint": source_file_fingerprint,
            "processing_settings": {
                "target_fps": target_fps,
                "max_frames_requested": max_frames,
                "candidate_objective": candidate_objective,
                "artifact_prefix": artifact_prefix,
                "use_merl_profile": use_merl_profile,
            },
            "source_type": "recorded_video",
            "source_path": os.path.basename(video_path),
            "total_frames_evaluated": len(sampled_times),
            "target_fps": target_fps,
            "total_detections": detections_total,
            "total_tracks": len(trajectories),
            "total_trajectory_points": sum(len(trajectory.points) for trajectory in trajectories),
            "processing_seconds": round(processing_sec, 3),
            "video_metadata": metadata.to_dict(),
            "video_quality_assessment": quality.to_dict(),
            "model_inference": {
                "model_family": "YOLO11-n ONNX", "runtime": "ONNX Runtime CPU",
                "training_status": "not_performed",
                "training_note": "Pre-trained inference weights; not trained or fine-tuned by this project.",
                "confidence_threshold": det_cfg.get("conf_threshold", .40),
                "iou_threshold": det_cfg.get("iou_threshold", .45),
                "mean_detection_confidence": round(float(np.mean(confidences)), 4) if confidences else None,
                "mean_inference_time_ms": round(float(np.mean(inference_ms)), 3) if inference_ms else None,
                "p95_inference_time_ms": round(float(np.percentile(inference_ms, 95)), 3) if inference_ms else None,
            },
            "track_diagnostics": {
                "track_id_scope": "clip_local",
                "per_track": [{"track_id": trajectory.track_id, "sampled_point_count": len(trajectory.points)}
                              for trajectory in trajectories],
            },
            "spatial_analysis_status": "not_available",
            "spatial_analysis_reason": "No camera-specific homography or retail zone calibration was selected for this uploaded clip.",
            "dwell_analytics": {"status": "not_available", "reason": "Camera-specific retail zones are not configured for this clip."},
            "direct_transitions": {"status": "not_available", "reason": "Camera-specific retail zones are not configured for this clip."},
            "exploratory_layout_candidate": {"status": "not_available", "candidate_zone_id": None},
            "evidence_status": {
                "spatial_observation_eligibility": "insufficient_evidence",
                "is_eligible": False,
                "actionable_recommendations": "WITHHELD",
                "recommendation_withholding_rationale": "No scene-specific floor calibration or zones were selected.",
                "reasons_withheld": ["No camera-specific homography or retail zones were selected; spatial metrics and candidates are unavailable."],
            },
            "generated_artifacts": {
                "tracking_frame": annotated_frame_path if first_annotated is not None else None,
                "tracking_video": annotated_video_path if video_written else None,
            },
        }
        from src.reporting.observation_timeline import build_observation_timeline_sidecar
        timeline_sidecar = build_observation_timeline_sidecar(
            dataset_name="user_uploaded_indoor_video",
            target_sampling_fps=target_fps,
            playback_fps=target_fps,
            sampled_frame_records=sampled_frame_records,
            all_tracks=all_tracks,
            trajectories=trajectories,
            bounds=None,
            video_metadata=metadata.to_dict(),
        )
        timeline_path = os.path.join(reports_dir, f"{artifact_prefix}_observation_timeline.json")
        timeline_sidecar.save_json(timeline_path)
        summary["generated_artifacts"]["observation_timeline"] = timeline_path

        if progress_callback is not None:
            progress_callback(1.0, len(sampled_times))
        with open(os.path.join(reports_dir, f"{artifact_prefix}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary
    calibrator = HomographyCalibrator.from_config(config, dataset_name=profile_key)
    trajectories = calibrator.transform_trajectories(trajectories)
    expected_step = max(1, int(round(metadata.fps / target_fps))) if target_fps < metadata.fps else 1
    track_rows = _per_track_diagnostics(trajectories, expected_step)

    configured_zones, hysteresis_window = ZoneConfig.load_from_config(config, dataset_key=profile_key)
    zone_evaluator = ZoneEvaluator(zones=configured_zones, hysteresis_window=hysteresis_window)
    known_zone_ids = [zone.zone_id for zone in configured_zones]
    assignments=[]
    zone_point_coordinates={zone_id:[] for zone_id in known_zone_ids}
    for trajectory in trajectories:
        track_assignments=zone_evaluator.evaluate_trajectory(trajectory);assignments.extend(track_assignments)
        for point,zone_id in zip(trajectory.points,track_assignments):
            if zone_id is not None and point.floor_x is not None and point.floor_y is not None:
                zone_point_coordinates[zone_id].append((float(point.floor_x),float(point.floor_y)))
    in_zone = sum(zone is not None for zone in assignments)
    per_zone_points = {zid: sum(zone == zid for zone in assignments) for zid in known_zone_ids}

    footfall_engine = FootfallAnalyzer(known_zone_ids=known_zone_ids)
    footfall = footfall_engine.analyze(trajectories)
    occupancy = footfall_engine.get_occupancy_timeline(trajectories, known_zone_ids=known_zone_ids)
    peak_occupancy = {}
    for zid in known_zone_ids:
        frame = max(occupancy, key=lambda f: occupancy[f].get(zid, 0), default=None)
        peak_occupancy[zid] = {
            "peak_clip_local_tracks": occupancy[frame].get(zid, 0) if frame is not None else 0,
            "frame_idx": frame,
            "timestamp_sec": round(frame / metadata.fps, 3) if frame is not None and metadata.fps else None,
        }

    if progress_callback is not None:
        try:
            progress_callback(0.96, sample_count, "Generating heatmaps, floor plans, and saving results...")
        except TypeError:
            progress_callback(0.96, sample_count)
    floor_path = os.path.join(reports_dir, f"{artifact_prefix}_floor_trajectory.png")
    bounds = GridBounds(*(scene_profile["bounds"] if scene_profile else (0.0, 10.0, 0.0, 6.0)))
    density = compute_density_grid(trajectories, bounds=bounds, cell_size=scene_profile["cell_size"] if scene_profile else .5)
    heatmap_path = os.path.join(reports_dir, f"{artifact_prefix}_density_heatmap.png")
    render_density_heatmap(density, heatmap_path,
        title=f"{scene_label}: Sampled Trajectory Counts (Not Physical Occupancy)",
        zones=zone_evaluator.zones,
        unit_label="configured floor units (unverified metric scale)",
        cbar_label="Sampled Trajectory Points (Count)")
    dwell = analyze_dwell(trajectories, known_zone_ids=known_zone_ids)
    dwell_path = os.path.join(reports_dir, f"{artifact_prefix}_zone_dwell.png")
    render_zone_dwell_barchart(dwell, dwell_path, title=f"{scene_label}: Observed Dwell Spans and Episodes",
                               source_label=video_path)
    occupancy_path = os.path.join(reports_dir, f"{artifact_prefix}_zone_occupancy.png")
    render_zone_occupancy_timeline(occupancy, occupancy_path, metadata.fps)
    transitions=compute_direct_transitions(trajectories,known_zone_ids=known_zone_ids)
    hotspots={z:compute_zone_hotspot(pts,(bounds.min_x,bounds.max_x,bounds.min_y,bounds.max_y),density.cell_size_x)
              for z,pts in zone_point_coordinates.items()}
    candidate=select_exploratory_product_zone_candidate(known_zone_ids,per_zone_points,dwell.by_zone,
        transitions.zone_summaries,hotspot_by_zone={z:h for z,h in hotspots.items() if h is not None},
        objective=candidate_objective,
        candidate_zone_ids=scene_profile["candidate_zone_ids"] if scene_profile else None)
    analytics_3d_path=os.path.join(reports_dir,f"{artifact_prefix}_3d_analytics.png")
    render_3d_analytics_overview(density,dwell,occupancy,metadata.fps,analytics_3d_path,zone_evaluator.zones,trajectories)
    render_floor_trajectory_diagram(
        zone_evaluator.zones, trajectories, floor_path,
        title=f"{os.path.basename(video_path)}: Floor-Plane Tracks and Configured Zones",
        subtitle=f"{sampled_times[0] if sampled_times else 0:.1f}–{sampled_times[-1] if sampled_times else 0:.1f}s @ {target_fps:g} sampled FPS",
        grid_bounds=(bounds.min_x,bounds.max_x,bounds.min_y,bounds.max_y),
        highlight_zone_id=candidate.get("candidate_zone_id"),
        highlight_hotspot=candidate.get("candidate_hotspot"),
    )
    is_repository_merl_clip = scene_profile is None and os.path.normcase(os.path.abspath(video_path)) == os.path.normcase(
        os.path.abspath(os.path.join("data", "merl", "10_1_crop.mp4"))
    )
    context = EvidenceContext(
        "retail_store", "unverified_configured_units",
        "boundary_truncated" if is_repository_merl_clip else "unknown",
        notes="Single-camera MERL run" if is_repository_merl_clip else
              "Caller-defined camera profile; calibration and scene match are not independently verified",
    )
    observations = generate_spatial_observations(
        context, dwell_summary=dwell, transition_summary=transitions, density_grid=density)

    summary = {
        "dataset_name": "merl_shopping_dataset" if is_repository_merl_clip else "user_uploaded_indoor_video",
        "scene_profile": scene_profile,
        "scene_profile_provenance": scene_profile_provenance,
        "source_file_fingerprint": source_file_fingerprint,
        "processing_settings": {
            "target_fps": target_fps,
            "max_frames_requested": max_frames,
            "candidate_objective": candidate_objective,
            "artifact_prefix": artifact_prefix,
            "use_merl_profile": use_merl_profile,
        },
        "source_type": "recorded_video",
        "source_path": video_path if is_repository_merl_clip else os.path.basename(video_path),
        "total_frames_evaluated": len(sampled_times),
        "target_fps": target_fps, "sampled_time_start_sec": sampled_times[0] if sampled_times else None,
        "sampled_time_end_sec": sampled_times[-1] if sampled_times else None,
        "sampled_duration_sec": round(sampled_times[-1] - sampled_times[0], 3) if sampled_times else 0.0,
        "total_detections": detections_total, "total_tracks": len(trajectories),
        "processing_seconds": round(processing_sec, 3),
        "sampled_frame_processing_rate_fps": round(len(sampled_times) / processing_sec, 3) if processing_sec else None,
        "video_metadata": metadata.to_dict(), "video_quality_assessment": quality.to_dict(),
        "tracking_video_format": "WebM with VP8 video codec for browser playback",
        "model_inference": {
            "model_family": "YOLO11-n ONNX", "runtime": "ONNX Runtime CPU",
            "model_path": model_path, "training_status": "not_performed",
            "training_note": "Pre-trained inference weights; not trained or fine-tuned by this project.",
            "confidence_threshold": det_cfg.get("conf_threshold", .40),
            "iou_threshold": det_cfg.get("iou_threshold", .45),
            "mean_detection_confidence": round(float(np.mean(confidences)), 4) if confidences else None,
            "mean_inference_time_ms": round(float(np.mean(inference_ms)), 3) if inference_ms else None,
            "p95_inference_time_ms": round(float(np.percentile(inference_ms, 95)), 3) if inference_ms else None,
        },
        "track_diagnostics": {
            "track_id_scope": "clip_local", "ground_truth_tracking_metrics": "not_available",
            "per_track": track_rows,
            "total_unexpected_sample_gaps": sum(t["unexpected_sample_gaps"] for t in track_rows),
            "note": "Descriptive tracker histories only; no annotated retail identity reference exists to score tracking accuracy or fragmentation.",
        },
        "homography_diagnostics": {
            **calibrator.diagnostics.to_dict(),
            "reference_status": "configured_correspondences_not_independently_surveyed",
        },
        "zone_assignment_counts": {
            "in_zone_points": in_zone, "unassigned_out_of_zone_points": len(assignments) - in_zone,
            "total_points": len(assignments), "hysteresis_window": hysteresis_window,
            "per_zone": per_zone_points,
        },
        "footfall_analytics": {"status": "available", **footfall.to_dict(),
            "note": "Entries/visits are clip-local track events, not verified distinct people."},
        "occupancy_analytics": {
            "status": "available", "sampled_frame_count": len(occupancy),
            "peak_clip_local_tracks_by_zone": peak_occupancy,
            "timeline_by_frame": {str(frame): row for frame, row in occupancy.items()},
            "note": "Sampled per-frame track counts; not physical occupancy or store capacity.",
        },
        "dwell_analytics": {
            "status": "available", "total_episodes": dwell.total_episodes,
            "total_observed_dwell_span_sec": dwell.total_dwell_sec,
            "per_zone": {z: {"observed_dwell_span_sec": d.total_dwell_sec,
                             "episodes": d.total_episodes,
                             "distinct_clip_local_tracks": d.unique_tracks,
                             "mean_dwell_sec": d.mean_dwell_sec,
                             "median_dwell_sec": d.median_dwell_sec,
                             "min_dwell_sec": d.min_dwell_sec,
                             "max_dwell_sec": d.max_dwell_sec}
                         for z, d in dwell.by_zone.items()},
            "note": "Observed temporal spans between sampled detections, not continuous physical dwell.",
        },
        "direct_transitions": {
            "status": "available", "total_direct_transitions": transitions.total_transitions,
            "transition_counts": {f"{a}->{b}": n for (a, b), n in transitions.transition_counts.items()},
            "per_zone": {z: {"observed_run_count": d.observed_run_count,
                             "incoming_direct_transitions": d.direct_incoming_count,
                             "outgoing_direct_transitions": d.direct_outgoing_count}
                         for z, d in transitions.zone_summaries.items()},
            "note": "Direct observed transition counts; not transition probabilities.",
        },
        "exploratory_layout_candidate": candidate,
        "density_analytics": {
            "status": "available", "total_evaluated_points": density.total_evaluated_points,
            "total_binned_points": density.total_binned_points,
            "out_of_bounds_count": density.out_of_bounds_count,
            "untransformed_count": density.untransformed_count,
            "max_cell_density": density.max_cell_count,
            "conservation_invariant_verified": density.total_evaluated_points == (
                density.total_binned_points + density.out_of_bounds_count + density.untransformed_count),
            "note": "Discrete sampled point counts; not physical occupancy or unique visitors.",
        },
        "evidence_status": {
            "environment_type": context.environment_type,
            "calibration_status": context.calibration_status,
            "coverage_status": context.coverage_status,
            "spatial_observation_eligibility": observations.status,
            "is_eligible": observations.is_eligible,
            "reasons_withheld": list(observations.reasons_withheld),
            "actionable_recommendations": "WITHHELD",
            "recommendation_withholding_rationale": "Calibration and representative spatial coverage are not independently validated; recommendations remain withheld.",
        },
        "limitations": [
            "The YOLO11-n ONNX weights are pretrained; this project did not train or validate detector accuracy.",
            "Single-camera recording; coverage and calibration require independent checks.",
            "Floor coordinates are configured units, not independently surveyed physical metres.",
            "Dwell spans, tracker IDs, transitions, and density are sampled observations and not validated customer behavior.",
            "Retail layout recommendations are withheld.",
        ],
        "generated_artifacts": {
            "tracking_frame": annotated_frame_path if first_annotated is not None else None,
            "tracking_video": annotated_video_path if video_written else None,
            "floor_trajectory": floor_path, "density_heatmap": heatmap_path,
            "zone_dwell": dwell_path, "zone_occupancy": occupancy_path,
            "analytics_3d_overview": analytics_3d_path,
            "scene_profile": scene_profile_path,
        },
    }
    from src.reporting.observation_timeline import build_observation_timeline_sidecar
    timeline_sidecar = build_observation_timeline_sidecar(
        dataset_name=profile_key if use_merl_profile else "user_uploaded_indoor_video",
        target_sampling_fps=target_fps,
        playback_fps=target_fps,
        sampled_frame_records=sampled_frame_records,
        all_tracks=all_tracks,
        trajectories=trajectories,
        bounds=tuple(scene_profile.get("bounds", (0.0, 10.0, 0.0, 6.0))) if scene_profile else (0.0, 10.0, 0.0, 6.0),
        video_metadata=metadata.to_dict(),
    )
    timeline_path = os.path.join(reports_dir, f"{artifact_prefix}_observation_timeline.json")
    timeline_sidecar.save_json(timeline_path)
    summary["generated_artifacts"]["observation_timeline"] = timeline_path

    if progress_callback is not None:
        try:
            progress_callback(1.0, len(sampled_times), "Saving run summary and artifacts...")
        except TypeError:
            progress_callback(1.0, len(sampled_times))
    summary_path = os.path.join(reports_dir, f"{artifact_prefix}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_synthetic_demonstration(config: Dict[str, Any],
                                  reports_dir: str = "outputs/reports") -> Dict[str, Any]:
    """Produce clearly labeled synthetic visuals that exercise nonzero analytics."""
    os.makedirs(reports_dir, exist_ok=True)
    zones, _ = ZoneConfig.load_from_config(config, dataset_key="merl")
    known = [zone.zone_id for zone in zones]
    locations = {
        "checkout": (5.0, .5), "main_aisle": (5.0, 2.8),
        "shelf_a": (2.2, 3.0), "shelf_b": (7.8, 3.0),
    }
    trajectories = []
    sequences = [
        [("checkout", 7), ("main_aisle", 7), ("shelf_a", 5), ("main_aisle", 7), ("checkout", 7)],
        [("checkout", 7), ("main_aisle", 7), ("shelf_b", 5), ("main_aisle", 7), ("checkout", 7)],
    ]
    for track_id, sequence in enumerate(sequences, start=101):
        trajectory = Trajectory(track_id=track_id)
        index = 0
        for zone_id, count in sequence:
            x, y = locations[zone_id]
            for _ in range(count):
                trajectory.add_point(TrajectoryPoint(track_id, index * 6, index * .2,
                    x * 10, y * 10, floor_x=x, floor_y=y, zone_id=zone_id))
                index += 1
        trajectories.append(trajectory)
    density = compute_density_grid(trajectories, bounds=GridBounds(0, 10, 0, 6), cell_size=.5)
    dwell = analyze_dwell(trajectories, known_zone_ids=known)
    transitions = compute_direct_transitions(trajectories, known_zone_ids=known)
    floor_path = os.path.join(reports_dir, "demo_synthetic_floor_trajectory.png")
    heatmap_path = os.path.join(reports_dir, "demo_synthetic_density_heatmap.png")
    dwell_path = os.path.join(reports_dir, "demo_synthetic_zone_dwell.png")
    render_floor_trajectory_diagram(zones, trajectories, floor_path,
        title="Synthetic Multi-Zone Tracks", is_synthetic=True)
    render_density_heatmap(density, heatmap_path,
        title="SYNTHETIC DEMONSTRATION — NOT REAL SHOPPER DATA",
        zones=zones, unit_label="synthetic configured units", cbar_label="Synthetic sampled point count")
    render_zone_dwell_barchart(dwell, dwell_path,
        title="Synthetic Dwell Summary", is_synthetic=True)
    summary = {
        "dataset_name": "SYNTHETIC DEMONSTRATION — NOT REAL SHOPPER DATA",
        "purpose": "Exercise pipeline metrics with programmatically generated tracks.",
        "disclaimer": "Synthetic coordinates; not actual customer footage or evidence.",
        "total_simulated_tracks": len(trajectories),
        "total_simulated_points": sum(len(t.points) for t in trajectories),
        "dwell_analytics": {
            "status": "available", "total_episodes": dwell.total_episodes,
            "total_observed_dwell_span_sec": dwell.total_dwell_sec,
            "per_zone": {z: {"observed_dwell_span_sec": d.total_dwell_sec,
                             "episodes": d.total_episodes,
                             "distinct_clip_local_tracks": d.unique_tracks,
                             "mean_dwell_sec": d.mean_dwell_sec,
                             "median_dwell_sec": d.median_dwell_sec,
                             "min_dwell_sec": d.min_dwell_sec,
                             "max_dwell_sec": d.max_dwell_sec}
                         for z, d in dwell.by_zone.items()},
        },
        "direct_transitions": {
            "status": "available", "total_direct_transitions": transitions.total_transitions,
            "transition_counts": {f"{a}->{b}": n for (a, b), n in transitions.transition_counts.items()},
            "note": "Direct transition counts; not Markovian probabilities.",
        },
        "density_analytics": {
            "status": "available", "total_evaluated_points": density.total_evaluated_points,
            "total_binned_points": density.total_binned_points,
            "out_of_bounds_count": density.out_of_bounds_count,
            "untransformed_count": density.untransformed_count,
            "max_cell_density": density.max_cell_count,
            "conservation_invariant_verified": density.total_evaluated_points == (
                density.total_binned_points + density.out_of_bounds_count + density.untransformed_count),
            "note": "Synthetic discrete point counts; not physical crowd occupancy.",
        },
        "actionable_recommendations": "WITHHELD (Synthetic demonstration data cannot support retail recommendations)",
        "generated_artifacts": {
            "floor_trajectory": floor_path, "density_heatmap": heatmap_path, "zone_dwell": dwell_path,
        },
    }
    with open(os.path.join(reports_dir, "demo_synthetic_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Offline full-pipeline visual demo")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--reports-dir", default="outputs/reports")
    parser.add_argument("--max-frames", type=int, default=700,
                        help="Maximum frames sampled from the local MERL clip; 700 covers the available clip")
    parser.add_argument("--no-synthetic", action="store_true")
    args = parser.parse_args()
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    real = run_real_data_demo(config, args.reports_dir, args.max_frames)
    synth = None if args.no_synthetic else build_synthetic_demonstration(config, args.reports_dir)
    print("Offline demo complete")
    print(f"Frames: {real['total_frames_evaluated']}; detections: {real['total_detections']}; tracks: {real['total_tracks']}")
    print(f"Zone points: {real['zone_assignment_counts']['in_zone_points']} in zones; "
          f"{real['zone_assignment_counts']['unassigned_out_of_zone_points']} unassigned")
    print(f"Footfall entries/visits: {real['footfall_analytics']['total_entries']}/{real['footfall_analytics']['total_visits']}")
    print(f"Dwell: {real['dwell_analytics']['total_observed_dwell_span_sec']}s across {real['dwell_analytics']['total_episodes']} episodes")
    print(f"Direct transitions: {real['direct_transitions']['total_direct_transitions']}")
    d = real['density_analytics']
    print(f"Density conservation: {d['total_evaluated_points']} = {d['total_binned_points']} + {d['out_of_bounds_count']} + {d['untransformed_count']}")
    print(f"Recommendation status: {real['evidence_status']['actionable_recommendations']}")
    print(f"Real summary: {os.path.join(args.reports_dir, 'demo_real_summary.json')}")
    if synth:
        print(f"Synthetic fixture summary: {os.path.join(args.reports_dir, 'demo_synthetic_summary.json')}")


if __name__ == "__main__":
    main()
