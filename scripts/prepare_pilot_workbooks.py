"""
Generate Blank Independent Human Annotation Workbooks
=====================================================
Generates separate, identical blank annotation workbooks for two independent
human annotators (Annotator 1 and Annotator 2) across both diagnostic pilot datasets:
1. MERL Shopping Aisle (100 sampled frames across 2 segments)
2. PRCV Site 03 Boutique (100 sampled frames in 1 segment)

All frames are explicitly initialized to 'PENDING_REVIEW' with empty person lists.
ZERO annotations are fabricated.
"""

import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.validation.annotation_validator import (
    MERL_PILOT_MANIFEST,
    PRCV_SITE03_PILOT_MANIFEST,
)


def generate_blank_workbook(
    dataset_name: str,
    clip_path: str,
    source_fps: float,
    sampling_step: int,
    target_fps: float,
    source_resolution: list,
    segments: list,
    target_retail_zones: list,
    frame_index_path: str,
    annotator_id: str,
    output_path: str,
):
    with open(frame_index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    frames = []
    for f in index_data["frames"]:
        frames.append({
            "frame_idx": f["frame_idx"],
            "timestamp_sec": f["timestamp_sec"],
            "segment_id": f["segment_id"],
            "frame_status": "PENDING_REVIEW",
            "image_filename": f["filename"],
            "annotator_notes": "",
            "persons": [],
        })

    workbook = {
        "workbook_version": "1.0",
        "pilot_identifier": "EVAL_PILOT_PHASE_01",
        "annotator_id": annotator_id,
        "dataset_name": dataset_name,
        "clip_path": clip_path,
        "source_resolution": source_resolution,
        "source_fps": source_fps,
        "sampling_step": sampling_step,
        "target_fps": target_fps,
        "evaluation_category": "development_diagnostic_evaluation",
        "overall_review_status": "PENDING_PILOT_ANNOTATION",
        "provenance_reference": "docs/evaluation/PROVENANCE_LEDGER.md",
        "segments": segments,
        "target_retail_zones": target_retail_zones,
        "instructions": (
            "For each frame, review the corresponding raw PNG frame from the pilot directory. "
            "If 0 persons are in the frame, set frame_status to 'REVIEWED_NO_PERSONS' with persons: []. "
            "If >= 1 person is present, set frame_status to 'REVIEWED_PERSONS_PRESENT' and add an entry "
            "for each person with annotator_track_label, image_bbox [x1, y1, x2, y2], "
            "ground_contact_px [x, y] (or null if feet occluded), zone_id (or null if unassigned), "
            "and uncertainty ('certain', 'occluded', 'truncated', 'ambiguous_identity'). "
            "Do NOT guess or invent coordinates for occluded feet. "
            "Frames left as 'PENDING_REVIEW' will be rejected as incomplete."
        ),
        "frames": frames,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(workbook, f, indent=2)

    return len(frames)


def main():
    pilot_dir = os.path.abspath("data/evaluation_pilot")
    workbooks_dir = os.path.join(pilot_dir, "workbooks")
    os.makedirs(workbooks_dir, exist_ok=True)

    # Segments configuration matching trusted manifests
    merl_segments = [
        {
            "segment_id": "merl_seg1_baseline",
            "name": "Segment 1: Negative Baseline & Initial Aisle Traversal",
            "frame_range": [0, 294],
            "sampling_step": 6,
        },
        {
            "segment_id": "merl_seg2_shelf_a_candidate",
            "name": "Segment 2: Shelf A Area Approach, Presence & Depart",
            "frame_range": [960, 1254],
            "sampling_step": 6,
        },
    ]

    prcv_segments = [
        {
            "segment_id": "prcv_site03_seg1_boutique",
            "name": "Segment 1: Multi-Person Boutique Browsing & Crossing",
            "frame_range": [0, 297],
            "sampling_step": 3,
        }
    ]

    # Authoritative zone IDs from configs/merl_scene_profile.json
    merl_zones = ["shelf_a", "shelf_b", "main_aisle", "checkout"]

    # Authoritative zone IDs from configs/prcv_site03_scene_profile.json
    prcv_zones = [
        "side_display_left",
        "display_table_center",
        "center_aisle",
        "checkout_counter",
    ]

    merl_index = os.path.join(pilot_dir, "merl_shopping_aisle", "frame_index.json")
    prcv_index = os.path.join(pilot_dir, "prcv_site03_boutique", "frame_index.json")

    # 1. MERL Annotator 1 & 2
    for ann_id in ("annotator_1", "annotator_2"):
        out_path = os.path.join(workbooks_dir, f"merl_shopping_{ann_id}_blank.json")
        count = generate_blank_workbook(
            dataset_name="merl_shopping_aisle",
            clip_path="data/merl/10_1_crop.mp4",
            source_fps=30.0,
            sampling_step=6,
            target_fps=5.0,
            source_resolution=[920, 680],
            segments=merl_segments,
            target_retail_zones=merl_zones,
            frame_index_path=merl_index,
            annotator_id=ann_id,
            output_path=out_path,
        )
        print(f"Generated {ann_id} MERL workbook: {count} frames -> {out_path}")

    # 2. PRCV Site 03 Annotator 1 & 2
    for ann_id in ("annotator_1", "annotator_2"):
        out_path = os.path.join(workbooks_dir, f"prcv_site03_{ann_id}_blank.json")
        count = generate_blank_workbook(
            dataset_name="prcv_site03_boutique",
            clip_path="data/prcv_site03_boutique.mp4",
            source_fps=15.0,
            sampling_step=3,
            target_fps=5.0,
            source_resolution=[896, 672],
            segments=prcv_segments,
            target_retail_zones=prcv_zones,
            frame_index_path=prcv_index,
            annotator_id=ann_id,
            output_path=out_path,
        )
        print(f"Generated {ann_id} PRCV Site 03 workbook: {count} frames -> {out_path}")

    # 3. Synchronize blank templates to docs/evaluation/templates/
    templates_dir = os.path.abspath("docs/evaluation/templates")
    os.makedirs(templates_dir, exist_ok=True)
    generate_blank_workbook(
        dataset_name="merl_shopping_aisle",
        clip_path="data/merl/10_1_crop.mp4",
        source_fps=30.0,
        sampling_step=6,
        target_fps=5.0,
        source_resolution=[920, 680],
        segments=merl_segments,
        target_retail_zones=merl_zones,
        frame_index_path=merl_index,
        annotator_id="template_annotator",
        output_path=os.path.join(templates_dir, "merl_pilot_workbook_blank.json"),
    )
    generate_blank_workbook(
        dataset_name="prcv_site03_boutique",
        clip_path="data/prcv_site03_boutique.mp4",
        source_fps=15.0,
        sampling_step=3,
        target_fps=5.0,
        source_resolution=[896, 672],
        segments=prcv_segments,
        target_retail_zones=prcv_zones,
        frame_index_path=prcv_index,
        annotator_id="template_annotator",
        output_path=os.path.join(templates_dir, "prcv_site03_pilot_workbook_blank.json"),
    )
    print("Synchronized docs/evaluation/templates/ blank workbooks.")

    print("=== Blank Workbooks Ready ===")


if __name__ == "__main__":
    main()
