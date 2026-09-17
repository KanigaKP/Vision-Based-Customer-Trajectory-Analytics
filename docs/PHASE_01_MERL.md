# Phase 1 — MERL Shopping Dataset Accessibility & Feasibility Report

**Date:** 2026-09-17  
**Status:** ✅ PASS (Demonstration Only)

---

## 1. Overview

The MERL Shopping Dataset provides overhead video recordings of shoppers interacting with grocery shelves. This report documents the accessibility and inspection results for a sample video sequence from the dataset.

> **CRITICAL METHODOLOGICAL BOUNDARY:**  
> MERL annotations consist of fine-grained action segmentations (*Reach To Shelf*, *Retract From Shelf*, *Hand In Shelf*, *Inspect Product*, *Inspect Shelf*).  
> **MERL does NOT provide continuous multi-object tracking ground truth (tracking IDs across frames).**  
> Therefore, tracking performance metrics such as **MOTA** or **IDF1** must **NOT** be calculated on MERL data. MERL is strictly used as a retail-context demonstration environment.

---

## 2. Sample Video Metadata (`10_1_crop.mp4`)

| Property | Measured Value |
|---|---|
| **Video File** | `data/merl/10_1_crop.mp4` |
| **Source Archive** | `Videos_MERL_Shopping_Dataset.zip` (Official MERL Repository) |
| **Resolution** | 920 × 680 pixels |
| **Frame Rate** | 30.00 FPS |
| **Total Frames** | 3,796 frames |
| **Duration** | 126.53 seconds (~2.1 minutes) |
| **OpenCV Verification** | Passed (Successfully opened and decoded frames via `cv2.VideoCapture`) |

---

## 3. Visual & Spatial Inspection

Visual inspection was performed on extracted frame `outputs/reports/merl_sample_frame.jpg`:

| Criteria | Assessment | Details |
|---|---|---|
| **Camera Viewpoint** | Fixed Overhead / Steep Oblique | Mounted statically looking directly down at a store shelf aisle. |
| **Camera Stability** | Fixed / Stationary | Completely stationary camera with zero pan/tilt/zoom movement. |
| **Floor Visibility** | Visible | The tiled/patterned floor space in front of the shelving unit is clearly visible. |
| **People Visibility** | Visible | Shoppers entering and exiting the aisle are fully observable. |
| **Movement Observability**| High | Customer entry, movement along the shelf, stopping/dwelling, and exit trajectories are clearly observable. |
| **Homography Feasibility**| Feasible (Localized) | A floor-plane homography mapping is feasible for the visible floor region in front of the shelves. |

---

## 4. Dataset Role & Limitations

1. **Role in Pipeline:** Demonstrates detection, tracking, and spatial zone analytics in an actual retail shelf setting.
2. **Key Limitation:** No continuous identity tracking ground truth.
3. **Usage Rule:** Evaluation of tracking algorithms (MOTA/IDF1) will be conducted on MOT17; validation of analytics logic will be conducted on controlled college footage.
