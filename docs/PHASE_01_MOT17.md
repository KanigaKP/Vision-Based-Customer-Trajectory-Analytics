# Phase 1 — MOT17 Accessibility Report

**Date:** 2026-09-17

## Source

| Field | Value |
|---|---|
| **Dataset** | MOT17 (MOTChallenge) |
| **Official URL** | https://motchallenge.net/data/MOT17/ |
| **Download source** | HuggingFace (`Morrison1025/MOT17`) — selective file download |
| **Full dataset size** | ~5.6 GB (MOT17.zip) |
| **Downloaded subset** | MOT17-04-FRCNN training sequence (10 frames + GT + seqinfo) |

## Selected Sequence: MOT17-04-FRCNN

| Field | Value |
|---|---|
| **Sequence name** | MOT17-04-FRCNN |
| **Image directory** | `img1/` |
| **Frame rate** | 30 FPS |
| **Total frames** | 1050 |
| **Resolution** | 1920 × 1080 |
| **Image format** | JPEG |
| **Ground truth file** | `gt/gt.txt` (3.5 MB, 108,005 lines) |
| **Unique track IDs** | 141 |
| **Persons in frame 1** | 42 (with `conf=1`) |

## Sequence Contents Verified

| Item | Present | Path |
|---|---|---|
| `seqinfo.ini` | ✅ | `data/mot17/train/MOT17-04-FRCNN/seqinfo.ini` |
| `gt/gt.txt` | ✅ | `data/mot17/train/MOT17-04-FRCNN/gt/gt.txt` |
| `img1/` (10 frames) | ✅ | `data/mot17/train/MOT17-04-FRCNN/img1/000001.jpg` … `000010.jpg` |
| `det/det.txt` | ✅ | `data/mot17/train/MOT17-04-SDP/det/det.txt` |

## GT Format

```
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<visibility>
```

Example: `1,1,1363,569,103,241,1,1,0.86014`

- `conf=1` → active annotation; `conf=0` → ignored region
- `class=1` → pedestrian
- `visibility` → occlusion ratio (0–1)

## Scene Characteristics

| Characteristic | Assessment |
|---|---|
| **Scene type** | Outdoor pedestrian street (Japan) |
| **Camera type** | Static, elevated (~3rd floor or balcony) |
| **Camera angle** | ~30–40° downward oblique |
| **Lighting** | Evening/night with street lamps |
| **Ground plane** | Visible, paved tiles, approximately flat |
| **Pedestrian density** | High (30–50 people per frame) |
| **Occlusions** | Frequent (lamp posts, cars, inter-person) |
| **Suitable for tracking evaluation** | ✅ YES — dense, with GT |

## Homography Suitability

| Factor | Assessment |
|---|---|
| **Ground plane visible** | Yes — paved street surface |
| **Camera fixed** | Yes |
| **Perspective depth** | Very deep (strong foreshortening) |
| **Floor-plane homography quality** | Moderate — accurate in near field, degrades at depth |
| **Suitable for floor-plane homography** | ⚠️ PARTIALLY — deep perspective limits accuracy |

> **Important:** MOT17 is an outdoor pedestrian street scene, NOT a retail/supermarket environment.
> It is used exclusively as an external tracking benchmark for MOTA/IDF1 evaluation.
> It is NOT presented as retail surveillance data.

## Frames Downloaded

Only 10 frames (of 1050) were downloaded for feasibility inspection. The full sequence will be needed for tracking evaluation in later phases.
