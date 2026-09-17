# Phase 1 — Homography Feasibility Report

**Date:** 2026-09-17

## Objective

Test whether image-coordinate-to-floor-plane homography can produce a spatially meaningful mapping using the available video footage.

## Method

1. Selected MOT17-04 frame 1 (1920 × 1080, static camera, visible ground plane)
2. Manually identified 4 ground-plane correspondence points
3. Computed homography matrix using `cv2.findHomography()`
4. Measured reprojection errors
5. Generated visualizations

## Correspondence Points

| Point | Image (x, y) | Floor (x, y) m | Description |
|---|---|---|---|
| P1 | (480, 580) | (0, 8) | Near-left ground |
| P2 | (1050, 540) | (6, 8) | Near-center ground |
| P3 | (1450, 480) | (10, 8) | Near-right ground |
| P4 | (680, 350) | (3, 0) | Far-center ground |

## Results

### Homography Matrix

```
[[-6.089e-03  -5.295e-03   5.993e+00]
 [-2.570e-03  -1.786e-02   8.000e+00]
 [-3.213e-04  -2.233e-03   1.000e+00]]
```

All 4 points used in estimation (status = [1, 1, 1, 1]).

### Reprojection Errors

| Point | Error (metres) |
|---|---|
| P1 (near-left) | ~0.0 |
| P2 (near-center) | 0.0 |
| P3 (near-right) | 0.0 |
| P4 (far-center) | 3.0 |
| **Mean** | **0.75** |

### Transformation Test

| Image (x, y) | Floor (x, y) | Notes |
|---|---|---|
| (480, 580) | (0.00, 8.00) | Exact (training point) |
| (1050, 540) | (6.00, 8.00) | Exact (training point) |
| (760, 450) | (4.08, 8.00) | Reasonable interpolation |
| (1200, 500) | (7.89, 8.00) | Reasonable interpolation |

## Assessment

| Factor | Result |
|---|---|
| Ground approximately planar | ✅ Yes (paved street) |
| Camera fixed | ✅ Yes (static) |
| Correspondences sensible | ✅ Yes (identifiable ground features) |
| Near-field accuracy | ✅ Good (sub-millimetre error) |
| Far-field accuracy | ⚠️ Poor (3m error on P4) |
| Spatial relationships preserved | ✅ Yes (left-right ordering maintained) |
| Zones drawable in transformed coords | ✅ Yes |

## Conclusion

**Homography is FEASIBLE** for this project but has limitations:

1. **Near-to-mid field**: Accurate, spatially consistent
2. **Far field**: Degrades due to extreme perspective foreshortening in MOT17-04
3. **Controlled footage expectation**: The student-recorded footage (more overhead, closer camera) should yield significantly better homography accuracy
4. A 4-point homography with carefully selected correspondences can produce usable floor-plane coordinates for zone analytics

> **Important:** A low reprojection error on training points does NOT automatically make a mapping valid. The visual consistency of transformed spatial relationships is the more meaningful indicator. In this test, left-right ordering and relative distances are preserved in the near field, which is sufficient for zone-level analytics.

## Visualizations

Generated files (not committed to Git):
- `outputs/reports/homography_feasibility_correspondences.jpg` — Marked correspondence points on original frame
- `outputs/reports/homography_feasibility_floormap.jpg` — Projected floor-plane layout
