# Controlled Video Recording Specification

**Project:** Vision-Based Customer Trajectory Analytics  
**Phase:** 1  
**Date:** 2026-09-17  
**Purpose:** Define requirements for recording controlled validation footage

---

## 1. Purpose of Controlled Footage

This footage provides **ground-truth validation** for the analytics layer of the pipeline:
- Footfall counting
- Dwell-time measurement
- Zone transitions
- Person crossings
- Temporary occlusion handling
- Boundary-case behaviour

> **Important:** This is controlled validation footage recorded in an academic setting.  
> It is NOT supermarket CCTV. It is NOT presented as retail surveillance data.

---

## 2. Location Selection

### Recommended Locations (in order of preference)

1. **College library** — natural movement, defined zones (shelves, reading areas, computers)
2. **Academic block common area** — corridors, seating, notice boards
3. **Indoor lab or seminar hall** — controllable, minimal external foot traffic

### Location Requirements

| Requirement | Details |
|---|---|
| Indoor | Required (consistent lighting, no weather interference) |
| Sufficient floor area | At least ~6m x 8m visible floor space |
| Floor visibility | Floor must be clearly visible from camera angle |
| Defined functional areas | At least 4 distinct zones possible |
| Controllable access | Ability to stage controlled scenarios |
| Permission | Obtain verbal/written permission from appropriate authority |
| Privacy | Avoid capturing bystanders who have not consented |

---

## 3. Camera Setup

| Parameter | Specification |
|---|---|
| **Camera type** | Smartphone camera (rear), action camera, or webcam on tripod |
| **Mounting** | Fixed on tripod or stable elevated surface |
| **Height** | Elevated: 2–3 metres above floor (use shelf top, mezzanine, or tripod) |
| **Angle** | ~30°–45° downward tilt from horizontal (bird's-eye oblique) |
| **Orientation** | Landscape |
| **Resolution** | Minimum 1080p (1920×1080) |
| **Frame rate** | 25–30 FPS |
| **Stabilisation** | Camera must remain completely stationary throughout recording |
| **Field of view** | Wide enough to capture all defined zones simultaneously |
| **Zoom** | Do NOT zoom in — keep wide view |
| **Focus** | Fixed focus (disable autofocus if possible) |
| **Face emphasis** | Do NOT point camera at face level; elevated angle naturally reduces facial detail |

### Camera Placement Diagram (Top-Down)

```
        [Camera - elevated, tilted down]
              |
              v
    +----------------------------+
    |  Z1       |    Z2          |
    | Entrance  | Shelf/Resource |
    |           |                |
    +-----------+----------------+
    |  Z5       |    Z3          |
    |  Exit     | Reading/       |
    |           | Activity Area  |
    +-----------+----------------+
    |           Z4               |
    |    Computer/Work Area      |
    +----------------------------+
```

---

## 4. Zone Definitions

Define **4–6 zones** based on the actual location chosen. Example zones for a **college library**:

| Zone ID | Zone Name | Description |
|---|---|---|
| Z1 | Entrance | Doorway / entry corridor (2–3m from door) |
| Z2 | Shelf / Resource Area | Bookshelves, resource racks |
| Z3 | Reading / Activity Area | Tables, chairs, individual seating |
| Z4 | Computer / Work Area | Desktop terminals or laptop desks |
| Z5 | Exit | Doorway / exit corridor (may overlap with Z1) |

> **Note:** Adjust zone names and boundaries to match the actual location.  
> Zone boundaries should be physically identifiable (floor markings, furniture edges, columns).

### Zone Boundary Marking

For ground-truth accuracy, consider:
- Placing small coloured tape markers at zone boundary corners on the floor
- These serve as reference points for homography calibration
- Use at least 4 markers per zone boundary (corners)

---

## 5. Controlled Scenarios

Record the following scenarios. Each scenario should be clearly separated (brief pause between scenarios).

| Scenario ID | Name | Description | Min Duration |
|---|---|---|---|
| **S1** | Single-person walk | One person walks through all zones in sequence (Z1→Z2→Z3→Z4→Z5) | 60s |
| **S2** | Multiple people | 3–4 people moving independently through different zones simultaneously | 90s |
| **S3** | Short dwell | One person enters a zone, pauses ~5s, moves to next zone | 60s |
| **S4** | Long dwell | One person enters a zone and stays stationary for ~30–60s | 90s |
| **S5** | Repeated zone visit | One person visits Z2, goes to Z3, returns to Z2, then exits | 90s |
| **S6** | Two-person crossing | Two people walk toward each other and cross paths in the same zone | 45s |
| **S7** | Zone boundary | One person walks slowly along a zone boundary (tests edge assignment) | 45s |
| **S8** | Temporary occlusion | One person walks behind a shelf/pillar and re-emerges (partial occlusion) | 45s |

**Total estimated recording time: ~9 minutes** (within the 5–10 minute target)

### Scenario Execution Tips

- Have participants walk at **natural speed** (not exaggerated slow/fast)
- Participants should wear **distinct clothing** (different colours) for multi-person scenarios
- Announce scenario number verbally at the start of each (helps with annotation)
- Keep a **5-second pause** between scenarios (camera remains rolling)

---

## 6. Ground-Truth Annotation

After recording, the user should manually record the following ground-truth data in a spreadsheet or CSV:

### Per-Scenario Log

| Field | Example |
|---|---|
| Scenario ID | S1 |
| Video timestamp start | 0:00:05 |
| Video timestamp end | 0:01:05 |
| Number of people | 1 |
| Person IDs | P1 |

### Per-Person Zone Log

| Field | Example |
|---|---|
| Person ID | P1 |
| Zone entered | Z1 |
| Entry timestamp | 0:00:05 |
| Exit timestamp | 0:00:15 |
| Dwell duration (s) | 10 |
| Next zone | Z2 |
| Transition type | Z1→Z2 |

### Expected Ground-Truth Counts (per scenario)

| Scenario | Expected Footfall | Zones Visited | Transitions | Crossings | Occlusions |
|---|---|---|---|---|---|
| S1 | 1 person, all zones | Z1,Z2,Z3,Z4,Z5 | 4 | 0 | 0 |
| S2 | 3–4 people | Varies | Multiple | Possible | Possible |
| S3 | 1 person | 3+ zones | 2+ | 0 | 0 |
| S4 | 1 person | 1–2 zones | 1 | 0 | 0 |
| S5 | 1 person | Z2,Z3,Z2 | 3 (Z1→Z2→Z3→Z2→Z5) | 0 | 0 |
| S6 | 2 people | 1+ zones | Varies | 1 | 0 |
| S7 | 1 person | Boundary | 1–2 | 0 | 0 |
| S8 | 1 person | 2+ zones | 1+ | 0 | 1 |

---

## 7. File Naming Convention

```
controlled_YYYYMMDD_SXX.mp4
```

Example: `controlled_20260920_S01.mp4`

Or record as one continuous video: `controlled_20260920_full.mp4`

Store all files in: `data/controlled/`

---

## 8. Pre-Recording Checklist

- [ ] Location selected and permission obtained
- [ ] Camera mounted at elevated, stable position
- [ ] Full floor area visible in frame
- [ ] Zone boundary markers placed (optional but helpful)
- [ ] All participants briefed on scenarios
- [ ] Test recording of ~10s reviewed for framing and quality
- [ ] Recording started in landscape, 1080p, 25–30 FPS
- [ ] Ground-truth annotation template prepared

---

## 9. Post-Recording Checklist

- [ ] Video file(s) transferred to `data/controlled/`
- [ ] Video opens in VLC / OpenCV without corruption
- [ ] All 8 scenarios are present in the footage
- [ ] Ground-truth annotation spreadsheet completed
- [ ] Camera did not move during recording
- [ ] Floor is clearly visible throughout
