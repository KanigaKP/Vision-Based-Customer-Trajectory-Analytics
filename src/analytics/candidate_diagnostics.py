"""
Vision-Based Customer Trajectory Analytics — Candidate Diagnostics Module.
========================================================================
Generates explainable, auditable diagnostics for exploratory retail layout
candidate selection without altering Phase 13 report schemas or inventing
unsupported recommendation claims.

CORE METHODOLOGICAL PRINCIPLES:
1. Descriptive Traffic Exploration Only:
   - Evaluates observed shopper movement patterns across configured shelf zones.
   - Does NOT infer customer purchasing intent, product demand, optimal layout,
     or commercial outcome.
2. Objective Sensitivity Comparison:
   - Evaluates the run across all 3 supported objectives ('observed_dwell',
     'sampled_points', 'visit_episodes').
   - Agreement among objectives is noted as descriptive alignment only and
     does NOT constitute empirical validation.
3. Transparent Absence & Neutrality:
   - Unobserved candidate zones are labelled 'Unobserved'; they are NEVER called
     'underperforming' or converted to synthetic zeroes.
   - When only one candidate zone has observations, explicitly states:
     "Selection is based on one observed candidate zone; comparative evidence
      for the other configured candidates is absent."
   - When no candidate zone has observations, returns an explicit
     no-supported-candidate state.
4. Footprint & Placement Boundaries:
   - Hotspot cell coordinates represent the highest-density spatial bin of
     sampled points; they do NOT represent an exact physical installation coordinate.
   - No placement into physical shelves, aisles, or fixtures is recommended without
     explicit CAD/fixture constraints.
5. Clip-Local Identity Scope:
   - Tracker identifiers are clip-local temporary track IDs, never distinct individuals.
"""

from dataclasses import dataclass, field, asdict
import math
import os
import json
from typing import Dict, List, Optional, Any, Sequence, Tuple, Union


SUPPORTED_OBJECTIVES = ("observed_dwell", "sampled_points", "visit_episodes")

OBJECTIVE_PRIMARY_METRIC = {
    "observed_dwell": "observed_dwell_span_sec",
    "sampled_points": "sampled_in_zone_points",
    "visit_episodes": "observed_visit_episodes",
}

OBJECTIVE_METRIC_LABELS = {
    "observed_dwell": "Observed Dwell Span (seconds)",
    "sampled_points": "Sampled In-Zone Points (count)",
    "visit_episodes": "Observed Visit Episodes (count)",
}


@dataclass
class CandidateZoneMetrics:
    """Observed traffic metrics for a single candidate shelf zone."""
    zone_id: str
    has_observations: bool
    observed_dwell_span_sec: Optional[float] = None
    sampled_in_zone_points: Optional[int] = None
    observed_visit_episodes: Optional[int] = None
    observed_runs: Optional[int] = None
    distinct_clip_local_tracks: Optional[int] = None
    hotspot_cell: Optional[Dict[str, Any]] = None
    contributing_track_ids: List[int] = field(default_factory=list)

    @property
    def is_evaluated(self) -> bool:
        return self.has_observations

    @property
    def distinct_tracks_count(self) -> Optional[int]:
        return self.distinct_clip_local_tracks

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["is_evaluated"] = self.is_evaluated
        d["distinct_tracks_count"] = self.distinct_tracks_count
        return d


@dataclass
class ObjectiveEvaluation:
    """Outcome of applying a specific ranking objective to the evaluated candidate zones."""
    objective: str
    status: str                         # 'selected', 'single_observed_zone', 'no_supported_candidate'
    selected_zone_id: Optional[str]
    selected_cell: Optional[Dict[str, Any]]
    ranking: List[str]                  # Zone IDs in descending priority order
    is_tie: bool                        # True if top rank shares identical primary metric
    tie_zones: List[str]                # Zone IDs sharing top primary metric
    lead_metric_name: str
    lead_metric_value: Optional[float]
    runner_up_zone_id: Optional[str]
    runner_up_metric_value: Optional[float]
    absolute_lead_difference: Optional[float]
    selection_rule: str
    selection_summary: str
    contributing_track_ids: List[int] = field(default_factory=list)
    contributing_points_count: int = 0

    @property
    def primary_metric_value(self) -> Optional[float]:
        return self.lead_metric_value

    @property
    def primary_metric_label(self) -> str:
        return OBJECTIVE_METRIC_LABELS.get(self.objective, self.lead_metric_name)

    @property
    def lead_difference(self) -> Optional[float]:
        return self.absolute_lead_difference

    @property
    def lead_percentage(self) -> Optional[float]:
        if self.lead_metric_value is not None and self.runner_up_metric_value is not None and self.lead_metric_value > 0:
            return round(((self.lead_metric_value - self.runner_up_metric_value) / self.lead_metric_value) * 100.0, 1)
        return None

    @property
    def tied_zone_ids(self) -> List[str]:
        return self.tie_zones

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["primary_metric_value"] = self.primary_metric_value
        d["primary_metric_label"] = self.primary_metric_label
        d["lead_difference"] = self.lead_difference
        d["lead_percentage"] = self.lead_percentage
        d["tied_zone_ids"] = self.tied_zone_ids
        return d


@dataclass
class CandidateDiagnosticsResult:
    """
    Versioned, self-contained diagnostics artifact detailing why a placement
    candidate was emitted, how alternatives performed, and whether objectives diverge.
    """
    schema_version: str = "1.0"
    run_id: str = ""
    dataset_name: str = ""
    candidate_zone_ids: List[str] = field(default_factory=list)
    candidate_zone_source: str = ""
    primary_objective: str = "observed_dwell"
    status: str = "candidate_selected"
    selected_zone_id: Optional[str] = None
    selected_cell: Optional[Dict[str, Any]] = None
    selection_summary: str = ""
    selection_rule: str = ""
    evaluated_candidate_zones: List[str] = field(default_factory=list)
    unobserved_candidate_zones: List[str] = field(default_factory=list)
    zone_metrics: Dict[str, CandidateZoneMetrics] = field(default_factory=dict)
    primary_evaluation: Optional[ObjectiveEvaluation] = None
    objective_comparisons: Dict[str, ObjectiveEvaluation] = field(default_factory=dict)
    objectives_agree: bool = True
    objectives_divergence_explanation: str = ""
    grid_bounds: Optional[Tuple[float, float, float, float]] = None
    cell_size: Optional[Union[float, Tuple[float, float]]] = None
    whole_run_point_accounting: Dict[str, Any] = field(default_factory=dict)
    missing_inputs: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.grid_bounds is not None:
            d["grid_bounds"] = list(self.grid_bounds)
        return d

    def save_json(self, output_path: str) -> str:
        """Save diagnostics to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return output_path


def compute_hotspot_with_track_lineage(
    points_with_tracks: Optional[Sequence[Tuple[float, float, int]]] = None,
    bounds: Tuple[float, float, float, float] = (0.0, 10.0, 0.0, 6.0),
    cell_size: Union[float, Tuple[float, float]] = 0.5,
    zone_id: Optional[str] = None,
    timeline_dict: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute within-zone hotspot cell with explicit contributing track lineage
    and tie status. Supports direct point sequences or extraction from a timeline dict.
    """
    min_x, max_x, min_y, max_y = (float(v) for v in bounds)
    if max_x <= min_x or max_y <= min_y:
        return None
    cs_x = float(cell_size[0]) if isinstance(cell_size, (tuple, list)) else float(cell_size)
    cs_y = float(cell_size[1]) if isinstance(cell_size, (tuple, list)) else float(cell_size)
    if cs_x <= 0 or cs_y <= 0:
        return None

    cols = max(1, int(round((max_x - min_x) / cs_x)))
    rows = max(1, int(round((max_y - min_y) / cs_y)))
    actual_cs_x = (max_x - min_x) / cols
    actual_cs_y = (max_y - min_y) / rows

    pts_to_process: List[Tuple[float, float, int]] = []
    if points_with_tracks is not None:
        pts_to_process = list(points_with_tracks)
    elif timeline_dict and "playback_mapping" in timeline_dict:
        for entry in timeline_dict.get("playback_mapping", []):
            for obs in entry.get("observations", []):
                if zone_id is None or obs.get("assigned_zone") == zone_id:
                    fp = obs.get("floor_point")
                    tid = obs.get("track_id", 0)
                    if fp and len(fp) >= 2:
                        pts_to_process.append((float(fp[0]), float(fp[1]), int(tid)))

    if not pts_to_process:
        return None

    cell_counts: Dict[Tuple[int, int], int] = {}
    cell_tracks: Dict[Tuple[int, int], set] = {}

    for item in pts_to_process:
        if len(item) >= 3:
            x, y, tid = item[0], item[1], item[2]
        else:
            x, y, tid = item[0], item[1], None

        if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
            continue
        if min_x <= x <= max_x and min_y <= y <= max_y:
            c = (cols - 1) if x >= max_x else int((x - min_x) / actual_cs_x)
            c = min(cols - 1, max(0, c))
            r = (rows - 1) if y >= max_y else int((y - min_y) / actual_cs_y)
            r = min(rows - 1, max(0, r))
            cell_counts[(r, c)] = cell_counts.get((r, c), 0) + 1
            if tid is not None:
                cell_tracks.setdefault((r, c), set()).add(int(tid))

    if not cell_counts:
        return None

    # Sort descending by count, then ascending by row, then col (deterministic existing rule)
    sorted_cells = sorted(cell_counts.items(), key=lambda it: (-it[1], it[0][0], it[0][1]))
    top_cell, max_n = sorted_cells[0]
    row, col = top_cell

    # Check for ties on count
    tied_cells = [c for c, n in sorted_cells if n == max_n]
    is_tied = len(tied_cells) > 1

    x0 = min_x + col * actual_cs_x
    y0 = min_y + row * actual_cs_y
    x1 = min(x0 + actual_cs_x, max_x)
    y1 = min(y0 + actual_cs_y, max_y)

    contributing_tracks = sorted(list(cell_tracks.get((row, col), set())))

    return {
        "grid_row": row,
        "grid_column": col,
        "x_min": round(x0, 4),
        "x_max": round(x1, 4),
        "y_min": round(y0, 4),
        "y_max": round(y1, 4),
        "center_x": round((x0 + x1) / 2.0, 4),
        "center_y": round((y0 + y1) / 2.0, 4),
        "sampled_points": max_n,
        "contributing_track_ids": contributing_tracks,
        "contributing_clip_local_tracks": contributing_tracks,
        "is_tie": is_tied,
        "is_tied_for_peak": is_tied,
        "tied_cell_count": len(tied_cells),
        "tied_cells_count": len(tied_cells),
        "unit": "configured floor units",
        "interpretation": "Highest sampled trajectory-point cell in this zone; not a validated demand or installation location.",
    }


def evaluate_single_objective(
    objective: str,
    shelf_ids: List[str],
    zone_metrics_dict: Dict[str, CandidateZoneMetrics],
) -> ObjectiveEvaluation:
    """Evaluate candidate ranking and selection under a single explicit objective."""
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"Unsupported objective: {objective}")

    metric_order = {
        "observed_dwell": ("observed_dwell_span_sec", "sampled_in_zone_points", "observed_visit_episodes"),
        "sampled_points": ("sampled_in_zone_points", "observed_dwell_span_sec", "observed_visit_episodes"),
        "visit_episodes": ("observed_visit_episodes", "observed_dwell_span_sec", "sampled_in_zone_points"),
    }[objective]

    primary_key = metric_order[0]

    rows = []
    for zid in shelf_ids:
        m = zone_metrics_dict.get(zid)
        dwell = m.observed_dwell_span_sec if m and m.observed_dwell_span_sec is not None else 0.0
        pts = m.sampled_in_zone_points if m and m.sampled_in_zone_points is not None else 0
        eps = m.observed_visit_episodes if m and m.observed_visit_episodes is not None else 0
        runs = m.observed_runs if m and m.observed_runs is not None else 0
        rows.append({
            "zone_id": zid,
            "observed_dwell_span_sec": dwell,
            "sampled_in_zone_points": pts,
            "observed_visit_episodes": eps,
            "observed_runs": runs,
            "has_observations": bool(m and m.has_observations),
        })

    # Sort descending by metric tuple, then ascending by zone_id
    rows.sort(key=lambda r: tuple(-r[k] for k in metric_order) + (r["zone_id"],))
    ranking = [r["zone_id"] for r in rows]

    active_rows = [r for r in rows if r["has_observations"] and any(r[k] > 0 for k in metric_order)]

    if not active_rows:
        return ObjectiveEvaluation(
            objective=objective,
            status="no_supported_candidate",
            selected_zone_id=None,
            selected_cell=None,
            ranking=ranking,
            is_tie=False,
            tie_zones=[],
            lead_metric_name=primary_key,
            lead_metric_value=None,
            runner_up_zone_id=None,
            runner_up_metric_value=None,
            absolute_lead_difference=None,
            selection_rule="No candidate shelf zone had observed activity; no candidate is emitted.",
            selection_summary="No candidate zone has supporting observations in this run.",
            contributing_track_ids=[],
            contributing_points_count=0,
        )

    selected_row = active_rows[0]
    selected_zid = selected_row["zone_id"]
    lead_val = selected_row[primary_key]

    # Check ties on primary metric
    top_ties = [r["zone_id"] for r in active_rows if abs(r[primary_key] - lead_val) < 1e-6]
    is_tie = len(top_ties) > 1

    selected_m = zone_metrics_dict.get(selected_zid)
    selected_cell = selected_m.hotspot_cell if selected_m else None
    c_tracks = selected_m.contributing_track_ids if selected_m else []
    c_pts = selected_m.sampled_in_zone_points if selected_m and selected_m.sampled_in_zone_points is not None else 0

    if len(active_rows) == 1:
        summary_text = (
            "Selection is based on one observed candidate zone; comparative evidence "
            "for the other configured candidates is absent."
        )
        rule_text = (
            "Only one shelf zone had observations; its hotspot is the configured cell "
            "with most sampled points (ties: row, then column)."
        )
        runner_up_zid = None
        runner_up_val = None
        lead_diff = None

        return ObjectiveEvaluation(
            objective=objective,
            status="single_observed_zone",
            selected_zone_id=selected_zid,
            selected_cell=selected_cell,
            ranking=ranking,
            is_tie=False,
            tie_zones=[],
            lead_metric_name=primary_key,
            lead_metric_value=lead_val,
            runner_up_zone_id=None,
            runner_up_metric_value=None,
            absolute_lead_difference=None,
            selection_rule=rule_text,
            selection_summary=summary_text,
            contributing_track_ids=c_tracks,
            contributing_points_count=c_pts,
        )

    # Multi-zone comparison
    runner_up_row = active_rows[1]
    runner_up_zid = runner_up_row["zone_id"]
    runner_up_val = runner_up_row[primary_key]
    lead_diff = round(lead_val - runner_up_val, 4)

    labels = {
        "observed_dwell_span_sec": "observed dwell span",
        "sampled_in_zone_points": "sampled in-zone points",
        "observed_visit_episodes": "observed visit episodes",
    }
    rule_text = (
        f"Among active shelf zones, prioritize {labels[primary_key]}; ties use other "
        f"observed metrics, then zone ID. No thresholds or learned weights."
    )
    summary_text = (
        f"Zone '{selected_zid}' leads zone '{runner_up_zid}' by {lead_diff} "
        f"{labels[primary_key]} ({lead_val} vs {runner_up_val})."
    )

    return ObjectiveEvaluation(
        objective=objective,
        status="selected",
        selected_zone_id=selected_zid,
        selected_cell=selected_cell,
        ranking=ranking,
        is_tie=is_tie,
        tie_zones=top_ties,
        lead_metric_name=primary_key,
        lead_metric_value=lead_val,
        runner_up_zone_id=runner_up_zid,
        runner_up_metric_value=runner_up_val,
        absolute_lead_difference=lead_diff,
        selection_rule=rule_text,
        selection_summary=summary_text,
        contributing_track_ids=c_tracks,
        contributing_points_count=c_pts,
    )



def generate_candidate_diagnostics(
    run_dict: Dict[str, Any],
    timeline_dict: Optional[Dict[str, Any]] = None,
    primary_objective: str = "observed_dwell",
    run_id: str = "",
) -> CandidateDiagnosticsResult:
    """
    Generate complete explainable candidate diagnostics for a run using existing
    stored summary data and optional timeline sidecar observations.
    """
    dataset_name = run_dict.get("dataset_name") or run_dict.get("metadata", {}).get("dataset_name", "unknown")
    scene_profile = run_dict.get("scene_profile") or {}

    # 1. Determine candidate zone IDs and source
    if isinstance(scene_profile, dict) and scene_profile.get("candidate_zone_ids"):
        candidate_zone_ids = list(scene_profile["candidate_zone_ids"])
        candidate_zone_source = "scene_profile.candidate_zone_ids"
    elif isinstance(scene_profile, dict) and scene_profile.get("zones"):
        conf_zids = [z["zone_id"] for z in scene_profile["zones"] if isinstance(z, dict) and "zone_id" in z]
        shelves = [z for z in conf_zids if str(z).startswith("shelf_")]
        if shelves:
            candidate_zone_ids = shelves
            candidate_zone_source = "scene_profile.zones:shelf_"
        else:
            candidate_zone_ids = conf_zids
            candidate_zone_source = "scene_profile.zones"
    else:
        # Check all zones known in run
        known_zones = []
        zone_counts = run_dict.get("zone_assignment_counts", {}).get("per_zone", {})
        if isinstance(zone_counts, dict):
            for k in zone_counts.keys():
                if k not in known_zones:
                    known_zones.append(k)
        dwell_zones = run_dict.get("dwell_analytics", {}).get("per_zone", {})
        if isinstance(dwell_zones, dict):
            for k in dwell_zones.keys():
                if k not in known_zones:
                    known_zones.append(k)
        # Filter shelf zones
        shelves = [z for z in known_zones if str(z).startswith("shelf_")]
        if shelves:
            candidate_zone_ids = shelves
            candidate_zone_source = "name_prefix:shelf_"
        else:
            candidate_zone_ids = known_zones if known_zones else ["shelf_a", "shelf_b"]
            candidate_zone_source = "default_fallback"

    # 2. Extract run geometry
    bounds_val = None
    density_raw = run_dict.get("density_analytics") or run_dict.get("spatial_density") or {}
    b_dict = density_raw.get("bounds")
    if isinstance(b_dict, dict) and all(k in b_dict for k in ("min_x", "max_x", "min_y", "max_y")):
        bounds_val = (float(b_dict["min_x"]), float(b_dict["max_x"]), float(b_dict["min_y"]), float(b_dict["max_y"]))
    elif isinstance(scene_profile, dict) and scene_profile.get("bounds"):
        bounds_val = tuple(float(v) for v in scene_profile["bounds"][:4])
    else:
        bounds_val = (0.0, 10.0, 0.0, 6.0)

    cell_size = density_raw.get("cell_size") or (scene_profile.get("cell_size") if isinstance(scene_profile, dict) else 0.5) or 0.5

    # 3. Whole run point accounting
    tot_pts = density_raw.get("total_evaluated_points")
    bin_pts = density_raw.get("total_binned_points")
    oob_pts = density_raw.get("out_of_bounds_count")
    untr_pts = density_raw.get("untransformed_count")
    conserved = (
        tot_pts == (bin_pts + oob_pts + untr_pts)
        if all(v is not None for v in (tot_pts, bin_pts, oob_pts, untr_pts))
        else False
    )

    whole_run_accounting = {
        "total_evaluated_points": tot_pts,
        "total_binned_points": bin_pts,
        "out_of_bounds_count": oob_pts,
        "untransformed_count": untr_pts,
        "total_points": tot_pts,
        "binned_points": bin_pts,
        "out_of_bounds_points": oob_pts,
        "untransformed_points": untr_pts,
        "conservation_verified": conserved,
    }

    # 4. Extract zone-assigned observations from timeline if available
    zone_pts_with_tracks: Dict[str, List[Tuple[float, float, int]]] = {zid: [] for zid in candidate_zone_ids}
    if timeline_dict and "playback_mapping" in timeline_dict:
        for entry in timeline_dict.get("playback_mapping", []):
            for obs in entry.get("observations", []):
                zid = obs.get("assigned_zone")
                fp = obs.get("floor_point")
                tid = obs.get("track_id")
                if zid in zone_pts_with_tracks and fp:
                    zone_pts_with_tracks[zid].append((float(fp[0]), float(fp[1]), int(tid) if tid is not None else 0))

    # 5. Extract per-zone metrics from run summary
    dwell_per_zone = run_dict.get("dwell_analytics", {}).get("per_zone", {})
    trans_per_zone = run_dict.get("direct_transitions", {}).get("per_zone", {})
    pts_per_zone = run_dict.get("zone_assignment_counts", {}).get("per_zone", {})
    existing_candidate_obj = run_dict.get("exploratory_layout_candidate") or run_dict.get("spatial_observations") or {}

    zone_metrics_dict: Dict[str, CandidateZoneMetrics] = {}
    evaluated_zones: List[str] = []
    unobserved_zones: List[str] = []

    for zid in candidate_zone_ids:
        # Point count
        pt_count = pts_per_zone.get(zid) if isinstance(pts_per_zone, dict) else None
        
        # Dwell metrics
        d_info = dwell_per_zone.get(zid, {}) if isinstance(dwell_per_zone, dict) else {}
        dwell_sec = d_info.get("observed_dwell_span_sec") if isinstance(d_info, dict) else getattr(d_info, "total_dwell_sec", None)
        episodes = None
        if isinstance(d_info, dict):
            episodes = d_info.get("total_episodes") if d_info.get("total_episodes") is not None else d_info.get("episodes")
        else:
            episodes = getattr(d_info, "total_episodes", None)

        unique_trks = None
        if isinstance(d_info, dict):
            unique_trks = d_info.get("distinct_tracks_count") if d_info.get("distinct_tracks_count") is not None else d_info.get("distinct_clip_local_tracks")
        else:
            unique_trks = getattr(d_info, "unique_tracks", None)

        # Transition metrics
        t_info = trans_per_zone.get(zid, {}) if isinstance(trans_per_zone, dict) else {}
        runs_count = t_info.get("observed_run_count") if isinstance(t_info, dict) else getattr(t_info, "observed_run_count", None)

        # Check if zone had any observed activity
        has_obs = bool(
            (pt_count is not None and pt_count > 0) or
            (dwell_sec is not None and dwell_sec > 0.0) or
            (episodes is not None and episodes > 0)
        )

        if has_obs:
            evaluated_zones.append(zid)
        else:
            unobserved_zones.append(zid)

        # Compute hotspot cell for this zone
        hotspot_cell = None
        if zone_pts_with_tracks.get(zid):
            hotspot_cell = compute_hotspot_with_track_lineage(
                zone_pts_with_tracks[zid], bounds=bounds_val, cell_size=cell_size
            )
        elif existing_candidate_obj.get("candidate_zone_id") == zid and existing_candidate_obj.get("candidate_hotspot"):
            # Fallback to recorded hotspot in summary
            hotspot_cell = existing_candidate_obj.get("candidate_hotspot")

        zone_tracks = sorted(list(set(tid for _, _, tid in zone_pts_with_tracks.get(zid, []) if tid > 0)))

        zone_metrics_dict[zid] = CandidateZoneMetrics(
            zone_id=zid,
            has_observations=has_obs,
            observed_dwell_span_sec=dwell_sec,
            sampled_in_zone_points=pt_count,
            observed_visit_episodes=episodes,
            observed_runs=runs_count,
            distinct_clip_local_tracks=unique_trks,
            hotspot_cell=hotspot_cell,
            contributing_track_ids=zone_tracks,
        )

    # 6. Evaluate all 3 supported objectives
    objective_results: Dict[str, ObjectiveEvaluation] = {}

    for obj in SUPPORTED_OBJECTIVES:
        objective_results[obj] = evaluate_single_objective(
            obj, candidate_zone_ids, zone_metrics_dict
        )

    primary_eval = objective_results.get(primary_objective, objective_results["observed_dwell"])

    # 7. Check objective agreement
    selected_zones_set = {
        res.selected_zone_id for res in objective_results.values() if res.selected_zone_id is not None
    }
    if not selected_zones_set:
        objectives_agree = True
        divergence_text = "All objectives agree: no candidate zone produced supporting observations in this run."
    elif len(selected_zones_set) == 1:
        objectives_agree = True
        the_zone = next(iter(selected_zones_set))
        divergence_text = (
            f"All 3 supported objectives prioritize '{the_zone}'. "
            f"Agreement among descriptive objectives reflects consistent relative traffic; "
            f"it does not constitute empirical validation or product demand."
        )
    else:
        objectives_agree = False
        parts = [f"'{obj}' selects '{res.selected_zone_id}'" for obj, res in objective_results.items()]
        divergence_text = (
            f"Objectives diverge across ranking criteria: {', '.join(parts)}. "
            f"Divergence occurs because dwell duration and sampled detection count reflect "
            f"different aspects of shopper movement (e.g., stationary pause vs. multi-point traversal)."
        )

    # 8. Missing inputs and limitations
    missing_inputs: List[str] = []
    if not evaluated_zones:
        missing_inputs.append("Observed shopper traffic is absent across all configured candidate zones.")
    elif len(evaluated_zones) < len(candidate_zone_ids):
        missing_inputs.append(
            f"Comparative observations absent for candidate zones: {', '.join(unobserved_zones)}."
        )
    if not timeline_dict:
        missing_inputs.append("Observation timeline sidecar is absent; individual point track IDs cannot be verified.")
    if run_dict.get("evidence_status", {}).get("is_eligible") is False:
        missing_inputs.append("Run evidence context is gated as ineligible for layout recommendation.")

    limitations = [
        "Descriptive Candidate Only: Emitted candidate reflects observed traffic frequency, not product demand or purchase conversion.",
        "Uncalibrated Scale: Coordinates reflect configured floor units, not surveyed metric metres.",
        "Boundary Truncation: Single-camera field-of-view limits observable zone transit.",
        "No Fixture Constraints: Cell bounds do not account for physical shelving, walking clearance, or structural pillars.",
        "Actionable Recommendations Withheld: Academic guardrail strictly withholds commercial layout changes.",
    ]

    overall_status = "candidate_selected" if primary_eval.status in ("selected", "single_observed_zone") else primary_eval.status

    return CandidateDiagnosticsResult(
        schema_version="1.0",
        run_id=run_id or run_dict.get("run_id", "current_run"),
        dataset_name=dataset_name,
        candidate_zone_ids=candidate_zone_ids,
        candidate_zone_source=candidate_zone_source,
        primary_objective=primary_objective,
        status=overall_status,
        selected_zone_id=primary_eval.selected_zone_id,
        selected_cell=primary_eval.selected_cell,
        selection_summary=primary_eval.selection_summary,
        selection_rule=primary_eval.selection_rule,
        evaluated_candidate_zones=evaluated_zones,
        unobserved_candidate_zones=unobserved_zones,
        zone_metrics=zone_metrics_dict,
        primary_evaluation=primary_eval,
        objective_comparisons=objective_results,
        objectives_agree=objectives_agree,
        objectives_divergence_explanation=divergence_text,
        grid_bounds=bounds_val,
        cell_size=cell_size,
        whole_run_point_accounting=whole_run_accounting,
        missing_inputs=missing_inputs,
        limitations=limitations,
    )
