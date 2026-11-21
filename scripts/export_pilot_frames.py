"""
Export Raw Source Frames for Diagnostic Annotation Pilots
=========================================================
Extracts exact sampled video frames according to the trusted SamplingManifests
for MERL Shopping and PRCV Site 03 clips into data/evaluation_pilot/.
Exports completely clean, raw frames with ZERO model overlays or bounding boxes.
Generates cryptographic checksums and machine-readable frame indexes.
"""

import hashlib
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from typing import Dict, List, Any

from src.validation.annotation_validator import (
    MERL_PILOT_MANIFEST,
    PRCV_SITE03_PILOT_MANIFEST,
    SamplingManifest,
)


def _compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def export_manifest_frames(
    video_path: str,
    manifest: SamplingManifest,
    output_dir: str,
    frame_prefix: str,
    package_root: str,
) -> Dict[str, Any]:
    """
    Extract raw unadulterated frames for all segments in a SamplingManifest.
    Preserves existing exported raw frames without modification.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    video_fingerprint = _compute_sha256(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    exported_records: List[Dict[str, Any]] = []
    missing_frames: List[int] = []

    # Map each frame index to its segment ID
    frame_to_segment: Dict[int, str] = {}
    for seg in manifest.segments:
        for f_idx in seg.expected_frame_indices:
            frame_to_segment[f_idx] = seg.segment_id

    all_expected = sorted(list(frame_to_segment.keys()))

    for f_idx in all_expected:
        filename = f"{frame_prefix}_frame_{f_idx:06d}.png"
        out_path = os.path.join(output_dir, filename)

        if not os.path.exists(out_path):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                missing_frames.append(f_idx)
                continue
            cv2.imwrite(out_path, frame)

        file_size_bytes = os.path.getsize(out_path)
        img_fingerprint = _compute_sha256(out_path)
        timestamp_sec = round(f_idx / source_fps, 3)
        segment_id = frame_to_segment[f_idx]

        rel_path = os.path.relpath(out_path, package_root).replace("\\", "/")
        abs_path = os.path.abspath(out_path).replace("\\", "/")

        exported_records.append({
            "frame_idx": f_idx,
            "timestamp_sec": timestamp_sec,
            "segment_id": segment_id,
            "filename": filename,
            "relative_path": rel_path,
            "absolute_path": abs_path,
            "image_dimensions": [source_width, source_height],
            "file_size_bytes": file_size_bytes,
            "sha256": img_fingerprint,
        })

    cap.release()

    is_verified = (len(missing_frames) == 0 and len(exported_records) == manifest.total_expected_frames)

    summary = {
        "dataset_name": manifest.dataset_name,
        "package_root": "data/evaluation_pilot",
        "package_root_absolute": os.path.abspath(package_root).replace("\\", "/"),
        "source_video_path": os.path.abspath(video_path).replace("\\", "/"),
        "source_video_sha256": video_fingerprint,
        "source_resolution": [source_width, source_height],
        "source_fps": source_fps,
        "total_video_frames": total_frames_in_video,
        "expected_frames_count": manifest.total_expected_frames,
        "exported_frames_count": len(exported_records),
        "missing_frames_count": len(missing_frames),
        "missing_frames": missing_frames,
        "is_verified": is_verified,
        "frames": exported_records,
    }

    return summary


def main():
    base_pilot_dir = os.path.abspath("data/evaluation_pilot")
    os.makedirs(base_pilot_dir, exist_ok=True)

    print("=== Updating Pilot Frame Indexes with Genuine Relative Paths ===")

    # 1. Export MERL Shopping Aisle
    merl_video = "data/merl/10_1_crop.mp4"
    merl_out_dir = os.path.join(base_pilot_dir, "merl_shopping_aisle", "raw_frames")
    print(f"Indexing MERL Shopping frames ({MERL_PILOT_MANIFEST.total_expected_frames} frames)...")
    merl_summary = export_manifest_frames(
        video_path=merl_video,
        manifest=MERL_PILOT_MANIFEST,
        output_dir=merl_out_dir,
        frame_prefix="merl",
        package_root=base_pilot_dir,
    )
    merl_index_path = os.path.join(base_pilot_dir, "merl_shopping_aisle", "frame_index.json")
    with open(merl_index_path, "w", encoding="utf-8") as f:
        json.dump(merl_summary, f, indent=2)
    print(f"  MERL: {merl_summary['exported_frames_count']} frames indexed, verified={merl_summary['is_verified']}")

    # 2. Export PRCV Site 03 Boutique
    prcv_video = "data/prcv_site03_boutique.mp4"
    prcv_out_dir = os.path.join(base_pilot_dir, "prcv_site03_boutique", "raw_frames")
    print(f"Indexing PRCV Site 03 frames ({PRCV_SITE03_PILOT_MANIFEST.total_expected_frames} frames)...")
    prcv_summary = export_manifest_frames(
        video_path=prcv_video,
        manifest=PRCV_SITE03_PILOT_MANIFEST,
        output_dir=prcv_out_dir,
        frame_prefix="prcv_site03",
        package_root=base_pilot_dir,
    )
    prcv_index_path = os.path.join(base_pilot_dir, "prcv_site03_boutique", "frame_index.json")
    with open(prcv_index_path, "w", encoding="utf-8") as f:
        json.dump(prcv_summary, f, indent=2)
    print(f"  PRCV Site 03: {prcv_summary['exported_frames_count']} frames indexed, verified={prcv_summary['is_verified']}")

    # 3. Consolidated Master Index
    master_index = {
        "pilot_identifier": "EVAL_PILOT_PHASE_01",
        "description": "Raw unadulterated video frames for human annotation pilot trials (Predictions Withheld)",
        "package_root": "data/evaluation_pilot",
        "package_root_absolute": os.path.abspath(base_pilot_dir).replace("\\", "/"),
        "datasets": {
            "merl_shopping_aisle": {
                "source_video": merl_summary["source_video_path"],
                "source_video_sha256": merl_summary["source_video_sha256"],
                "frame_count": merl_summary["exported_frames_count"],
                "resolution": merl_summary["source_resolution"],
                "fps": merl_summary["source_fps"],
                "frame_index_file_relative": "merl_shopping_aisle/frame_index.json",
                "frame_index_file_absolute": os.path.abspath(merl_index_path).replace("\\", "/"),
                "frames_directory_relative": "merl_shopping_aisle/raw_frames",
                "frames_directory_absolute": os.path.abspath(merl_out_dir).replace("\\", "/"),
                "is_verified": merl_summary["is_verified"],
            },
            "prcv_site03_boutique": {
                "source_video": prcv_summary["source_video_path"],
                "source_video_sha256": prcv_summary["source_video_sha256"],
                "frame_count": prcv_summary["exported_frames_count"],
                "resolution": prcv_summary["source_resolution"],
                "fps": prcv_summary["source_fps"],
                "frame_index_file_relative": "prcv_site03_boutique/frame_index.json",
                "frame_index_file_absolute": os.path.abspath(prcv_index_path).replace("\\", "/"),
                "frames_directory_relative": "prcv_site03_boutique/raw_frames",
                "frames_directory_absolute": os.path.abspath(prcv_out_dir).replace("\\", "/"),
                "is_verified": prcv_summary["is_verified"],
            },
        },
    }
    master_index_path = os.path.join(base_pilot_dir, "frame_index.json")
    with open(master_index_path, "w", encoding="utf-8") as f:
        json.dump(master_index, f, indent=2)
    print(f"Master index written to: {master_index_path}")
    print("=== Update Complete ===")


if __name__ == "__main__":
    main()
