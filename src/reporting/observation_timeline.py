"""
Phase 13.1 — Observation Timeline Sidecar Module.
=================================================
Generates and normalizes temporal observation records aligned with video playback.
Links source-video frame indices and timestamps with annotated playback frames
and spatial-temporal trajectory observations.

METHODOLOGICAL INVARIANTS:
1. Pure Observation Lineage:
   - Records observed sampled detections and trajectory points without inferring missing states.
   - Absent coordinates or zones are represented as None/null, never fabricated zeroes.
2. Temporal Alignment:
   - Explicitly records both source timestamps/frames and playback timestamps/frames,
     accounting for sampling stride, frame caps, and encoding FPS.
3. Privacy & Non-Biometric Identifier Scope:
   - Track IDs are strictly temporary clip-local integers.
   - No facial features, identities, or biometric information are retained.
4. Scale Disclaimer:
   - Floor units represent configured spatial scale, not independently surveyed metres.
"""

import os
import json
from dataclasses import dataclass, field, asdict
import math
from typing import Dict, List, Optional, Any, Tuple, Union


@dataclass
class ObservationRecord:
    """Individual object observation at a specific sampled frame."""
    track_id: int
    image_ground_point: List[float]               # [x, y] bottom-center in image pixels
    bbox: Optional[List[float]] = None            # [x1, y1, x2, y2]
    confidence: Optional[float] = None
    floor_point: Optional[List[float]] = None     # [x, y] in configured floor units (8-decimal precision)
    is_out_of_bounds: bool = False
    assigned_zone: Optional[str] = None           # Zone ID or None if unassigned
    grid_cell: Optional[Tuple[int, int]] = None   # Explicit (row, col) bin index or None if out of bounds

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.grid_cell is not None:
            d["grid_cell"] = list(self.grid_cell)
        return d


@dataclass
class SampleTimelineEntry:
    """Synchronized record linking playback time, source time, and observations."""
    sample_index: int                             # 0-indexed sample count
    playback_frame_idx: int                       # Frame index in output annotated video
    playback_time_sec: float                      # Playback timestamp in seconds
    source_frame_idx: int                         # Native frame index in source video
    source_timestamp_sec: float                   # Native timestamp in source video (seconds)
    observations: List[ObservationRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "playback_frame_idx": self.playback_frame_idx,
            "playback_time_sec": round(self.playback_time_sec, 3),
            "source_frame_idx": self.source_frame_idx,
            "source_timestamp_sec": round(self.source_timestamp_sec, 3),
            "observations": [obs.to_dict() for obs in self.observations],
        }


@dataclass
class ObservationTimelineSidecar:
    """
    Versioned observation sidecar capturing the complete temporal-spatial
    lineage of a video analysis run.
    """
    dataset_name: str
    target_sampling_fps: float
    playback_fps: float
    total_samples: int
    video_metadata: Dict[str, Any]
    playback_mapping: List[SampleTimelineEntry]
    tracks_summary: Dict[str, Any]
    schema_version: str = "1.0"
    coordinate_units: str = "configured_floor_units"
    coordinate_units_disclaimer: str = (
        "Configured floor coordinates; not independently surveyed physical metres."
    )
    timestamp_basis: str = "source_video_seconds"
    source_fps: Optional[float] = None
    total_source_frames: Optional[int] = None
    source_duration_sec: Optional[float] = None
    sampled_source_time_span_sec: Optional[float] = None
    playback_duration_sec: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "coordinate_units": self.coordinate_units,
            "coordinate_units_disclaimer": self.coordinate_units_disclaimer,
            "timestamp_basis": self.timestamp_basis,
            "source_fps": self.source_fps,
            "target_sampling_fps": self.target_sampling_fps,
            "playback_fps": self.playback_fps,
            "total_samples": self.total_samples,
            "total_source_frames": self.total_source_frames,
            "source_duration_sec": self.source_duration_sec,
            "sampled_source_time_span_sec": self.sampled_source_time_span_sec,
            "playback_duration_sec": self.playback_duration_sec,
            "video_metadata": self.video_metadata,
            "playback_mapping": [entry.to_dict() for entry in self.playback_mapping],
            "tracks_summary": self.tracks_summary,
        }

    def save_json(self, output_path: str) -> str:
        """Serialize sidecar to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return output_path

    @classmethod
    def load_json(cls, input_path: str) -> "ObservationTimelineSidecar":
        """Load and parse sidecar from JSON file."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        entries = []
        for raw_entry in data.get("playback_mapping", []):
            obs_list = []
            for raw_obs in raw_entry.get("observations", []):
                raw_gc = raw_obs.get("grid_cell")
                grid_cell = (int(raw_gc[0]), int(raw_gc[1])) if raw_gc is not None else None
                obs_list.append(ObservationRecord(
                    track_id=raw_obs["track_id"],
                    image_ground_point=raw_obs["image_ground_point"],
                    bbox=raw_obs.get("bbox"),
                    confidence=raw_obs.get("confidence"),
                    floor_point=raw_obs.get("floor_point"),
                    is_out_of_bounds=raw_obs.get("is_out_of_bounds", False),
                    assigned_zone=raw_obs.get("assigned_zone"),
                    grid_cell=grid_cell,
                ))
            entries.append(SampleTimelineEntry(
                sample_index=raw_entry["sample_index"],
                playback_frame_idx=raw_entry["playback_frame_idx"],
                playback_time_sec=raw_entry["playback_time_sec"],
                source_frame_idx=raw_entry["source_frame_idx"],
                source_timestamp_sec=raw_entry["source_timestamp_sec"],
                observations=obs_list,
            ))

        return cls(
            schema_version=data.get("schema_version", "1.0"),
            dataset_name=data.get("dataset_name", "unknown"),
            coordinate_units=data.get("coordinate_units", "configured_floor_units"),
            coordinate_units_disclaimer=data.get("coordinate_units_disclaimer", ""),
            timestamp_basis=data.get("timestamp_basis", "source_video_seconds"),
            source_fps=float(data["source_fps"]) if data.get("source_fps") is not None else None,
            target_sampling_fps=float(data.get("target_sampling_fps", 5.0)),
            playback_fps=float(data.get("playback_fps", 5.0)),
            total_samples=int(data.get("total_samples", len(entries))),
            total_source_frames=int(data["total_source_frames"]) if data.get("total_source_frames") is not None else None,
            source_duration_sec=float(data["source_duration_sec"]) if data.get("source_duration_sec") is not None else None,
            sampled_source_time_span_sec=float(data["sampled_source_time_span_sec"]) if data.get("sampled_source_time_span_sec") is not None else None,
            playback_duration_sec=float(data["playback_duration_sec"]) if data.get("playback_duration_sec") is not None else None,
            video_metadata=data.get("video_metadata", {}),
            playback_mapping=entries,
            tracks_summary=data.get("tracks_summary", {}),
        )


def build_observation_timeline_sidecar(
    dataset_name: str,
    target_sampling_fps: float,
    playback_fps: float,
    sampled_frame_records: List[Tuple[int, float]], # List of (source_frame_idx, source_timestamp_sec)
    all_tracks: List[Any],                          # List of Track objects from ByteTracker
    trajectories: List[Any],                        # List of Trajectory objects with floor_x, floor_y, zone_id
    bounds: Optional[Tuple[float, float, float, float]] = None,
    video_metadata: Optional[Dict[str, Any]] = None,
    cell_size: Optional[Union[float, Tuple[float, float]]] = 0.5,
) -> ObservationTimelineSidecar:
    """
    Construct an ObservationTimelineSidecar from actual tracker outputs and evaluated trajectories.
    """
    # Build lookup for track bboxes and confidences by (track_id, frame_id)
    track_point_lookup: Dict[Tuple[int, int], Any] = {}
    for track in all_tracks:
        tid = getattr(track, "track_id", None)
        if tid is None:
            continue
        for tp in getattr(track, "history", []):
            fid = getattr(tp, "frame_id", None)
            if fid is not None:
                track_point_lookup[(tid, fid)] = tp

    # Build lookup for trajectory points by (track_id, frame_idx)
    traj_point_lookup: Dict[Tuple[int, int], Any] = {}
    tracks_visited_zones: Dict[int, set] = {}
    track_time_extents: Dict[int, Tuple[float, float, int]] = {}

    for traj in trajectories:
        tid = getattr(traj, "track_id", None)
        if tid is None:
            continue
        tracks_visited_zones.setdefault(tid, set())
        for pt in getattr(traj, "points", []):
            fid = getattr(pt, "frame_idx", None)
            if fid is not None:
                traj_point_lookup[(tid, fid)] = pt
            
            # Track extents
            ts = getattr(pt, "timestamp_sec", 0.0)
            if tid not in track_time_extents:
                track_time_extents[tid] = (ts, ts, 1)
            else:
                s_min, s_max, count = track_time_extents[tid]
                track_time_extents[tid] = (min(s_min, ts), max(s_max, ts), count + 1)

            zid = getattr(pt, "zone_id", None)
            if zid:
                tracks_visited_zones[tid].add(zid)

    # Frame bounds check helper
    min_x, max_x, min_y, max_y = bounds if bounds and len(bounds) == 4 else (None, None, None, None)

    # Construct timeline entries for every sampled frame
    playback_entries: List[SampleTimelineEntry] = []

    for sample_idx, (src_frame_idx, src_time_sec) in enumerate(sampled_frame_records):
        playback_time = sample_idx / playback_fps if playback_fps > 0 else 0.0
        
        # Find all track observations at this source_frame_idx
        obs_at_frame: List[ObservationRecord] = []

        # Find all tracks that have an observation at src_frame_idx
        for (tid, fid), tp in track_point_lookup.items():
            if fid == src_frame_idx:
                bbox = getattr(tp, "bbox", None)
                conf = getattr(tp, "confidence", None)
                bottom_center = getattr(tp, "bottom_center", None)
                
                # Check matching trajectory point
                traj_pt = traj_point_lookup.get((tid, fid))
                floor_pt = None
                is_oob = False
                zone_id = None
                grid_cell = None

                if traj_pt is not None:
                    fx = getattr(traj_pt, "floor_x", None)
                    fy = getattr(traj_pt, "floor_y", None)
                    if fx is not None and fy is not None:
                        floor_pt = [round(float(fx), 8), round(float(fy), 8)]
                        if min_x is not None and max_x is not None and min_y is not None and max_y is not None:
                            is_oob = not (min_x <= fx <= max_x and min_y <= fy <= max_y)
                            if not is_oob:
                                cs = cell_size if cell_size is not None else 0.5
                                cs_x = cs if isinstance(cs, (int, float)) else cs[0]
                                cs_y = cs if isinstance(cs, (int, float)) else cs[1]
                                num_bins_x = max(1, int(round((max_x - min_x) / cs_x)))
                                num_bins_y = max(1, int(round((max_y - min_y) / cs_y)))
                                cell_size_x = (max_x - min_x) / num_bins_x
                                cell_size_y = (max_y - min_y) / num_bins_y
                                if fx >= max_x:
                                    col = num_bins_x - 1
                                else:
                                    col = int(math.floor((fx - min_x) / cell_size_x))
                                    col = min(num_bins_x - 1, max(0, col))
                                if fy >= max_y:
                                    row = num_bins_y - 1
                                else:
                                    row = int(math.floor((fy - min_y) / cell_size_y))
                                    row = min(num_bins_y - 1, max(0, row))
                                grid_cell = (row, col)
                    zone_id = getattr(traj_pt, "zone_id", None)
                elif bottom_center:
                    floor_pt = None

                img_pt = [round(float(bottom_center[0]), 2), round(float(bottom_center[1]), 2)] if bottom_center else [0.0, 0.0]

                obs_at_frame.append(ObservationRecord(
                    track_id=int(tid),
                    image_ground_point=img_pt,
                    bbox=[round(float(v), 2) for v in bbox] if bbox else None,
                    confidence=round(float(conf), 4) if conf is not None else None,
                    floor_point=floor_pt,
                    is_out_of_bounds=is_oob,
                    assigned_zone=zone_id,
                    grid_cell=grid_cell,
                ))


        # Sort observations by track_id for deterministic order
        obs_at_frame.sort(key=lambda o: o.track_id)

        playback_entries.append(SampleTimelineEntry(
            sample_index=sample_idx,
            playback_frame_idx=sample_idx,
            playback_time_sec=playback_time,
            source_frame_idx=src_frame_idx,
            source_timestamp_sec=src_time_sec,
            observations=obs_at_frame,
        ))

    # Construct tracks summary with explicit gap analysis
    tracks_summary: Dict[str, Any] = {}
    for tid, (t_start, t_end, pt_count) in track_time_extents.items():
        obs_sample_indices = [
            entry.sample_index
            for entry in playback_entries
            if any(o.track_id == tid for o in entry.observations)
        ]
        if obs_sample_indices:
            first_idx = min(obs_sample_indices)
            last_idx = max(obs_sample_indices)
            span_indices = set(range(first_idx, last_idx + 1))
            missing_indices = sorted(list(span_indices - set(obs_sample_indices)))
        else:
            first_idx = None
            last_idx = None
            span_indices = set()
            missing_indices = []

        tracks_summary[str(tid)] = {
            "track_id": tid,
            "first_sample_index": first_idx,
            "last_sample_index": last_idx,
            "active_sample_span": len(span_indices),
            "observed_sample_count": len(obs_sample_indices),
            "missing_sample_indices": missing_indices,
            "sample_gaps": missing_indices,
            "start_source_time_sec": round(t_start, 3),
            "end_source_time_sec": round(t_end, 3),
            "duration_sec": round(t_end - t_start, 3),
            "zones_visited": sorted(list(tracks_visited_zones.get(tid, set()))),
        }

    # Video timing metrics
    vmeta = video_metadata or {}
    src_fps = float(vmeta.get("fps") or vmeta.get("native_fps") or 30.0) if vmeta else None
    tot_src_frames = int(vmeta.get("total_frames", 0)) if (vmeta and "total_frames" in vmeta) else None
    src_dur = float(vmeta.get("duration_seconds", 0.0)) if (vmeta and "duration_seconds" in vmeta) else None
    sampled_span = round(sampled_frame_records[-1][1] - sampled_frame_records[0][1], 3) if len(sampled_frame_records) > 1 else 0.0
    play_dur = round(len(playback_entries) / playback_fps, 3) if playback_fps > 0 else 0.0

    return ObservationTimelineSidecar(
        dataset_name=dataset_name,
        target_sampling_fps=target_sampling_fps,
        playback_fps=playback_fps,
        total_samples=len(playback_entries),
        video_metadata=vmeta,
        playback_mapping=playback_entries,
        tracks_summary=tracks_summary,
        source_fps=src_fps,
        total_source_frames=tot_src_frames,
        source_duration_sec=src_dur,
        sampled_source_time_span_sec=sampled_span,
        playback_duration_sec=play_dur,
    )
