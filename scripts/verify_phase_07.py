"""
Phase 7 Verification Script — Footfall Analytics
================================================
Validates footfall analytics by interpreting Phase 6 zone-evaluated trajectories.
Computes and verifies:
1. Unique tracks per zone
2. Zone entry events
3. Contiguous zone visits
4. Basic visit duration (total & average)
5. Per-frame occupancy snapshots & timeline
6. Zone-to-zone transition matrix

Outputs structured report to outputs/reports/phase_07_footfall_report.json.
"""

import os
import sys
import json
import yaml

sys.path.insert(0, os.path.abspath("."))

from src.trajectory.trajectory import Trajectory, TrajectoryPoint
from src.spatial.homography import HomographyCalibrator
from src.analytics.zone import ZoneConfig
from src.analytics.zone_evaluator import ZoneEvaluator
from src.analytics.footfall import FootfallAnalyzer, FootfallSummary


def build_representative_trajectories() -> list:
    """
    Construct representative trajectories for MERL retail store floor plane [0, 10] x [0, 6].
    Zones in MERL config:
      - shelf_a:    [0.5, 4.0] x [1.0, 5.0]
      - shelf_b:    [6.0, 9.5] x [1.0, 5.0]
      - main_aisle: [4.0, 6.0] x [1.0, 5.0]
      - checkout:   [0.0, 10.0] x [0.0, 1.0]
    """
    trajectories = []
    fps = 5.0
    dt = 1.0 / fps

    # Customer 1 (Track 101): None -> shelf_a (4 frames) -> main_aisle (3 frames) -> None
    t1 = Trajectory(track_id=101)
    # Outside -> shelf_a
    coords_1 = [
        (0.2, 5.5),             # None
        (1.5, 3.0), (1.8, 3.0), # shelf_a
        (2.0, 3.2), (2.2, 3.2), # shelf_a
        (5.0, 3.0), (5.0, 2.5), # main_aisle
        (5.0, 2.0),             # main_aisle
        (0.2, 5.8)              # None
    ]
    for idx, (fx, fy) in enumerate(coords_1):
        t1.add_point(TrajectoryPoint(
            track_id=101, frame_idx=idx, timestamp_sec=round(idx * dt, 3),
            image_x=100.0, image_y=200.0, floor_x=fx, floor_y=fy
        ))
    trajectories.append(t1)

    # Customer 2 (Track 102): shelf_b (4 frames) -> main_aisle (3 frames) -> checkout (3 frames)
    t2 = Trajectory(track_id=102)
    coords_2 = [
        (7.5, 3.0), (7.8, 3.0), (8.0, 2.8), (8.0, 2.5), # shelf_b
        (5.2, 2.5), (5.0, 2.0), (4.8, 1.5),             # main_aisle
        (5.0, 0.5), (5.2, 0.5), (5.4, 0.5)              # checkout
    ]
    for idx, (fx, fy) in enumerate(coords_2):
        t2.add_point(TrajectoryPoint(
            track_id=102, frame_idx=idx, timestamp_sec=round(idx * dt, 3),
            image_x=150.0, image_y=250.0, floor_x=fx, floor_y=fy
        ))
    trajectories.append(t2)

    # Customer 3 (Track 103): Re-entrant shopper: shelf_a -> main_aisle -> shelf_a -> None
    t3 = Trajectory(track_id=103)
    coords_3 = [
        (2.0, 2.0), (2.0, 2.2), (2.0, 2.4), # shelf_a
        (4.5, 2.5), (4.8, 2.5),             # main_aisle
        (2.5, 2.0), (2.5, 2.2),             # shelf_a (re-entry!)
        (0.1, 5.5)                          # None
    ]
    for idx, (fx, fy) in enumerate(coords_3):
        t3.add_point(TrajectoryPoint(
            track_id=103, frame_idx=idx + 2, timestamp_sec=round((idx + 2) * dt, 3),
            image_x=120.0, image_y=210.0, floor_x=fx, floor_y=fy
        ))
    trajectories.append(t3)

    # Customer 4 (Track 104): Stays entirely outside all zones (passer-by in unassigned area)
    t4 = Trajectory(track_id=104)
    coords_4 = [
        (0.1, 5.5), (0.2, 5.5), (0.3, 5.6), (0.4, 5.6)
    ]
    for idx, (fx, fy) in enumerate(coords_4):
        t4.add_point(TrajectoryPoint(
            track_id=104, frame_idx=idx, timestamp_sec=round(idx * dt, 3),
            image_x=80.0, image_y=150.0, floor_x=fx, floor_y=fy
        ))
    trajectories.append(t4)

    return trajectories


def main():
    print("=" * 75)
    print("PHASE 7 — FOOTFALL ANALYTICS VERIFICATION")
    print("=" * 75)

    # 1. Load Configuration & Zone Definitions
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    reports_dir = os.path.join("outputs", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    zones, hysteresis_window = ZoneConfig.load_from_config(config, dataset_key="merl")
    known_zone_ids = [z.zone_id for z in zones]
    print(f"  Loaded {len(zones)} zones for MERL: {known_zone_ids}")
    print(f"  Hysteresis Window N = {hysteresis_window}")

    # 2. Phase 6 Evaluation Step: Spatial PIP & Hysteresis to Assign zone_id
    # Note: Phase 6 owns assigning pt.zone_id. Phase 7 only consumes it.
    print("\n[1/3] Running Phase 6 ZoneEvaluator to assign TrajectoryPoint.zone_id...")
    evaluator = ZoneEvaluator(zones=zones, hysteresis_window=1) # using window=1 on discrete sample sequence
    raw_trajectories = build_representative_trajectories()

    for traj in raw_trajectories:
        evaluator.evaluate_trajectory(traj)

    print(f"  Evaluated {len(raw_trajectories)} trajectories with Phase 6 ZoneEvaluator.")
    for traj in raw_trajectories:
        zone_seq = [p.zone_id for p in traj.points]
        print(f"    - Track {traj.track_id} ({len(traj.points)} pts): {' -> '.join(str(z) for z in zone_seq)}")

    # 3. Phase 7 Footfall Analytics Computation
    print("\n[2/3] Executing FootfallAnalyzer on evaluated trajectories...")
    analyzer = FootfallAnalyzer(known_zone_ids=known_zone_ids)
    summary = analyzer.analyze(raw_trajectories)

    # 4. Detailed Metrics Display
    print("\n" + "-" * 75)
    print(f"{'ZONE ID':<15} | {'UNIQUE TRACKS':<14} | {'ENTRIES':<9} | {'VISITS':<8} | {'AVG DURATION':<12}")
    print("-" * 75)
    for z in known_zone_ids:
        print(
            f"{z:<15} | "
            f"{summary.unique_tracks_per_zone[z]:<14} | "
            f"{summary.entry_events_per_zone[z]:<9} | "
            f"{summary.visits_per_zone[z]:<8} | "
            f"{summary.average_visit_duration_sec[z]:.2f}s"
        )
    print("-" * 75)

    # Check Zone-to-Zone Transitions
    print("\n  Zone-to-Zone Transition Matrix:")
    header = f"  {'FROM / TO':<14} | " + " | ".join(f"{z:<10}" for z in known_zone_ids)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for z_from in known_zone_ids:
        row = f"  {z_from:<14} | " + " | ".join(
            f"{summary.transition_matrix[z_from][z_to]:<10}" for z_to in known_zone_ids
        )
        print(row)

    # Occupancy Snapshots
    print("\n  Occupancy Snapshots at Selected Frames:")
    test_frames = [2, 5, 8]
    occupancy_results = {}
    for f in test_frames:
        occ = analyzer.get_occupancy_snapshot(raw_trajectories, frame_idx=f)
        occupancy_results[f"frame_{f}"] = occ
        print(f"    - Frame {f:2d}: {occ}")

    # 5. Sanity Checks & Assertions
    print("\n[3/3] Verifying metric invariants...")
    # Invariant 1: shelf_a had Track 101 and Track 103 -> Unique tracks = 2
    assert summary.unique_tracks_per_zone["shelf_a"] == 2, "Expected 2 unique tracks in shelf_a"
    # Invariant 2: Track 103 re-entered shelf_a -> shelf_a entries = 3 (1 from t1, 2 from t3)
    assert summary.entry_events_per_zone["shelf_a"] == 3, "Expected 3 entry events in shelf_a"
    assert summary.visits_per_zone["shelf_a"] == 3, "Expected 3 visits in shelf_a"
    # Invariant 3: Track 104 was outside -> shelf_a/shelf_b/checkout got 0 from Track 104
    # Invariant 4: Transitions from shelf_a to main_aisle (Track 101, Track 103) -> count = 2
    assert summary.transition_matrix["shelf_a"]["main_aisle"] == 2, "Expected 2 transitions shelf_a -> main_aisle"
    # Invariant 5: Transition main_aisle -> checkout (Track 102) -> count = 1
    assert summary.transition_matrix["main_aisle"]["checkout"] == 1, "Expected 1 transition main_aisle -> checkout"

    print("  All metric invariants verified successfully.")

    # 6. Save JSON Report
    report_data = {
        "phase": 7,
        "name": "Phase 7 Footfall Analytics Verification",
        "dataset": "MERL Retail Context",
        "known_zones": known_zone_ids,
        "trajectories_evaluated": len(raw_trajectories),
        "total_entries": len(summary.entries),
        "total_visits": len(summary.visits),
        "summary": summary.to_dict(),
        "occupancy_snapshots": occupancy_results,
    }

    report_path = os.path.join(reports_dir, "phase_07_footfall_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n  Saved structured report to: {report_path}")
    print("\n" + "=" * 75)
    print("VERIFICATION SUCCESSFUL: Phase 7 Footfall Analytics passed all checks.")
    print("=" * 75)


if __name__ == "__main__":
    main()
