"""
Generate Configured Zone Reference Images (Zero Predictions)
============================================================
Projects configured retail zone polygons from authoritative scene profiles
onto reference camera frames via inverse homography.
Produces clean visual spatial references for annotators with:
- Zero model detections
- Zero tracking trails
- Zero candidate highlighting
- Zero dwell values or predictions
"""

import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np


def project_floor_polygon_to_image(floor_polygon: list, H_img2floor: np.ndarray) -> np.ndarray:
    """Project a floor-plane polygon back to image coordinates using inverse homography."""
    H_floor2img = np.linalg.inv(H_img2floor)
    pts = np.array(floor_polygon, dtype=np.float32).reshape(-1, 1, 2)
    img_pts = cv2.perspectiveTransform(pts, H_floor2img)
    return img_pts.reshape(-1, 2)


def render_zone_reference_image(
    image_path: str,
    profile_path: str,
    output_path: str,
    title: str,
) -> str:
    """
    Render a clean reference frame with authoritative zone polygons and neutral labels.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Reference image not found: {image_path}")
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    h, w = frame.shape[:2]

    # Compute homography from image_points to floor_points
    src_pts = np.array(profile["image_points"], dtype=np.float32)
    dst_pts = np.array(profile["floor_points"], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)

    overlay = frame.copy()

    # Distinct, accessible zone palette (BGR)
    palette = [
        (220, 120, 40),    # blue-ish
        (40, 180, 60),     # green-ish
        (30, 140, 230),    # orange-ish
        (180, 50, 160),    # purple-ish
        (50, 200, 220),    # yellow-ish
    ]

    zones = profile.get("zones", [])

    # Pass 1: Semi-transparent polygon fills
    for idx, zone in enumerate(zones):
        color = palette[idx % len(palette)]
        poly_floor = zone["polygon"]
        poly_img = project_floor_polygon_to_image(poly_floor, H)
        int_pts = np.int32(poly_img)
        cv2.fillPoly(overlay, [int_pts], color)

    alpha = 0.28
    frame = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

    # Pass 2: Polygon boundaries and centroid labels
    for idx, zone in enumerate(zones):
        color = palette[idx % len(palette)]
        poly_floor = zone["polygon"]
        poly_img = project_floor_polygon_to_image(poly_floor, H)
        int_pts = np.int32(poly_img)

        # Crisp border
        cv2.polylines(frame, [int_pts], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)

        # Centroid for label placement
        cx = int(np.mean(poly_img[:, 0]))
        cy = int(np.mean(poly_img[:, 1]))
        cx = max(60, min(w - 180, cx))
        cy = max(40, min(h - 40, cy))

        zid = zone["zone_id"]
        zname = zone.get("name", zid)
        label_primary = f"Zone: {zid}"
        label_secondary = f"({zname})"

        # Label background pill
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_p = 0.55
        scale_s = 0.45
        thick = 1

        (tw1, th1), _ = cv2.getTextSize(label_primary, font, scale_p, 2)
        (tw2, th2), _ = cv2.getTextSize(label_secondary, font, scale_s, thick)
        tw = max(tw1, tw2)
        th = th1 + th2 + 10

        pill_x1 = cx - tw // 2 - 8
        pill_y1 = cy - th // 2 - 4
        pill_x2 = cx + tw // 2 + 8
        pill_y2 = cy + th // 2 + 6

        # Draw filled dark pill
        cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), (20, 20, 20), -1)
        cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), color, 2)

        # Draw text
        cv2.putText(frame, label_primary, (cx - tw1 // 2, pill_y1 + th1 + 4), font, scale_p, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, label_secondary, (cx - tw2 // 2, pill_y1 + th1 + th2 + 8), font, scale_s, (210, 210, 210), thick, cv2.LINE_AA)

    # Header banner
    cv2.rectangle(frame, (0, 0), (w, 48), (25, 25, 25), -1)
    cv2.line(frame, (0, 48), (w, 48), (80, 80, 80), 1)
    cv2.putText(
        frame,
        title,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "PREDICTIONS WITHHELD -- ANNOTATE OBSERVATIONS ONLY",
        (w - 480, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 180, 255),
        1,
        cv2.LINE_AA,
    )

    # Footer banner
    cv2.rectangle(frame, (0, h - 36), (w, h), (25, 25, 25), -1)
    cv2.line(frame, (0, h - 36), (w, h - 36), (80, 80, 80), 1)
    footer_text = "Unassigned floor area outside colored polygons: assign zone_id = null. Feet occluded: ground_contact_px = null."
    cv2.putText(frame, footer_text, (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, frame)
    return os.path.abspath(output_path)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pilot_dir = os.path.join(repo_root, "data", "evaluation_pilot")

    # 1. MERL Shopping Aisle Reference Image
    merl_img = os.path.join(pilot_dir, "merl_shopping_aisle", "raw_frames", "merl_frame_000000.png")
    merl_profile = os.path.join(repo_root, "configs", "merl_scene_profile.json")
    merl_out = os.path.join(pilot_dir, "merl_shopping_aisle", "configured_zones_reference.png")

    print(f"Generating MERL zone reference image...")
    p1 = render_zone_reference_image(
        image_path=merl_img,
        profile_path=merl_profile,
        output_path=merl_out,
        title="MERL Shopping Aisle -- Configured Retail Zones Reference",
    )
    print(f"  Saved: {p1}")

    # 2. PRCV Site 03 Boutique Reference Image
    prcv_img = os.path.join(pilot_dir, "prcv_site03_boutique", "raw_frames", "prcv_site03_frame_000000.png")
    prcv_profile = os.path.join(repo_root, "configs", "prcv_site03_scene_profile.json")
    prcv_out = os.path.join(pilot_dir, "prcv_site03_boutique", "configured_zones_reference.png")

    print(f"Generating PRCV Site 03 zone reference image...")
    p2 = render_zone_reference_image(
        image_path=prcv_img,
        profile_path=prcv_profile,
        output_path=prcv_out,
        title="PRCV Site 03 Boutique -- Configured Retail Zones Reference",
    )
    print(f"  Saved: {p2}")

    # Copy to docs/evaluation/ and artifact dir
    docs_eval_dir = os.path.join(repo_root, "docs", "evaluation")
    artifact_eval_dir = r"C:\Users\sarca\.gemini\antigravity\brain\9f864dae-64d5-473b-8662-0d7602771db9\evaluation"
    os.makedirs(artifact_eval_dir, exist_ok=True)

    for src_name, dst_name in [
        (merl_out, "merl_configured_zones_reference.png"),
        (prcv_out, "prcv_site03_configured_zones_reference.png"),
    ]:
        dest_doc = os.path.join(docs_eval_dir, dst_name)
        dest_art = os.path.join(artifact_eval_dir, dst_name)
        img = cv2.imread(src_name)
        cv2.imwrite(dest_doc, img)
        cv2.imwrite(dest_art, img)
        print(f"  Mirrored {dst_name} to docs and artifacts.")

    print("=== Configured Zone Reference Images Generated Successfully ===")


if __name__ == "__main__":
    main()
