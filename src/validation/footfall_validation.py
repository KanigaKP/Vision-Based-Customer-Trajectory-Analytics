"""
Footfall Event Validation Module
================================
Validates existing Phase 7 FootfallSummary outputs (zone entry counts,
zone-to-zone transitions) against independently supplied GroundTruthEvent records.

Architectural Rules & Methodological Boundaries:
1. Consumes Pre-Computed FootfallSummary:
   - Does NOT rerun FootfallAnalyzer or recompute entries/transitions from raw trajectories.
   - Evaluates aggregate event metrics directly from Phase 7 outputs.
2. Independent Identity Decoupling:
   - Does NOT require tracker IDs to equal ground-truth IDs.
   - Does NOT perform Re-ID, track rematching, or heuristic event pairing.
3. No Refined Dwell / Temporal Analytics:
   - Does NOT compute dwell distributions, temporal smoothing, gap-tolerant visits,
     or occupancy interpolation (reserved for Phase 9).
4. Descriptive Consistency Metrics:
   - Reports "event-count consistency" across evaluated zones and transitions.
   - Does NOT claim universal tracking or real-world footfall accuracy from synthetic fixtures.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import defaultdict

from src.analytics.footfall import FootfallSummary, ZoneEntryEvent
from src.validation.ground_truth import GroundTruthEvent, GroundTruthSequence


@dataclass
class EntryCountComparison:
    """
    Comparison between expected and predicted aggregate entry counts for a zone.

    Attributes:
        zone_id: Identifier of the retail zone.
        expected_count: Count of ground-truth entry events into this zone.
        predicted_count: Count of tracker-derived entry events from FootfallSummary.
        difference: Signed difference (predicted_count - expected_count).
        absolute_difference: Absolute difference abs(predicted_count - expected_count).
        is_match: True if expected_count == predicted_count.
    """
    zone_id: str
    expected_count: int
    predicted_count: int
    difference: int
    absolute_difference: int
    is_match: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntryCountComparison":
        """Reconstruct EntryCountComparison from dictionary."""
        return cls(
            zone_id=data["zone_id"],
            expected_count=int(data["expected_count"]),
            predicted_count=int(data["predicted_count"]),
            difference=int(data["difference"]),
            absolute_difference=int(data["absolute_difference"]),
            is_match=bool(data["is_match"]),
        )


@dataclass
class TransitionCountComparison:
    """
    Comparison between expected and predicted counts for a zone-to-zone transition.

    Attributes:
        from_zone_id: Origin zone identifier.
        to_zone_id: Destination zone identifier.
        expected_count: Count of ground-truth transition events between these zones.
        predicted_count: Count of tracker-derived transitions from FootfallSummary.
        difference: Signed difference (predicted_count - expected_count).
        absolute_difference: Absolute difference abs(predicted_count - expected_count).
        is_match: True if expected_count == predicted_count.
    """
    from_zone_id: str
    to_zone_id: str
    expected_count: int
    predicted_count: int
    difference: int
    absolute_difference: int
    is_match: bool

    @property
    def transition_key(self) -> str:
        """String representation of the transition pair 'from->to'."""
        return f"{self.from_zone_id}->{self.to_zone_id}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = asdict(self)
        d["transition_key"] = self.transition_key
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransitionCountComparison":
        """Reconstruct TransitionCountComparison from dictionary."""
        return cls(
            from_zone_id=data["from_zone_id"],
            to_zone_id=data["to_zone_id"],
            expected_count=int(data["expected_count"]),
            predicted_count=int(data["predicted_count"]),
            difference=int(data["difference"]),
            absolute_difference=int(data["absolute_difference"]),
            is_match=bool(data["is_match"]),
        )


@dataclass
class FootfallMetricDisagreement:
    """
    Structured record of a metric where expected and predicted aggregate counts differ.

    Attributes:
        metric_type: Category of metric ('entry_count' or 'transition_count').
        key: Zone identifier or transition key string ('from->to').
        expected_count: Ground-truth event count.
        predicted_count: Tracker-assigned count from FootfallSummary.
        difference: Signed difference (predicted - expected).
        absolute_difference: Magnitude of count difference.
        evidence_note: Descriptive note clarifying the discrepancy.
    """
    metric_type: str
    key: str
    expected_count: int
    predicted_count: int
    difference: int
    absolute_difference: int
    evidence_note: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FootfallMetricDisagreement":
        """Reconstruct FootfallMetricDisagreement from dictionary."""
        return cls(
            metric_type=data["metric_type"],
            key=data["key"],
            expected_count=int(data["expected_count"]),
            predicted_count=int(data["predicted_count"]),
            difference=int(data["difference"]),
            absolute_difference=int(data["absolute_difference"]),
            evidence_note=data["evidence_note"],
        )


@dataclass
class FootfallValidationResult:
    """
    Aggregate evaluation results comparing expected vs predicted footfall metrics.

    Attributes:
        dataset_name: Identifier string for sequence or evaluation run.
        expected_entry_counts: Ground-truth entry counts per zone.
        predicted_entry_counts: Tracker-derived entry counts per zone.
        entry_comparisons: List of per-zone EntryCountComparison records.
        expected_transition_counts: Ground-truth transition counts ('from->to' -> count).
        predicted_transition_counts: Tracker-derived transition counts ('from->to' -> count).
        transition_comparisons: List of TransitionCountComparison records.
        total_compared_metrics: Total count of evaluated metric comparisons.
        matched_metric_count: Count of comparisons where expected == predicted.
        mismatched_metric_count: Count of comparisons where expected != predicted.
        metric_consistency_rate: Descriptive proportion matched / total (or None if total == 0).
        evidence_status: 'FOOTFALL_COMPARISON_EVALUATED' or 'INSUFFICIENT_EVIDENCE'.
        evidence_note: Methodological scope explanation.
        disagreements: List of FootfallMetricDisagreement records.
    """
    dataset_name: str
    expected_entry_counts: Dict[str, int]
    predicted_entry_counts: Dict[str, int]
    entry_comparisons: List[EntryCountComparison]
    expected_transition_counts: Dict[str, int]
    predicted_transition_counts: Dict[str, int]
    transition_comparisons: List[TransitionCountComparison]
    total_compared_metrics: int
    matched_metric_count: int
    mismatched_metric_count: int
    metric_consistency_rate: Optional[float]
    evidence_status: str
    evidence_note: str
    disagreements: List[FootfallMetricDisagreement]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "expected_entry_counts": dict(self.expected_entry_counts),
            "predicted_entry_counts": dict(self.predicted_entry_counts),
            "entry_comparisons": [c.to_dict() for c in self.entry_comparisons],
            "expected_transition_counts": dict(self.expected_transition_counts),
            "predicted_transition_counts": dict(self.predicted_transition_counts),
            "transition_comparisons": [c.to_dict() for c in self.transition_comparisons],
            "total_compared_metrics": self.total_compared_metrics,
            "matched_metric_count": self.matched_metric_count,
            "mismatched_metric_count": self.mismatched_metric_count,
            "metric_consistency_rate": self.metric_consistency_rate,
            "evidence_status": self.evidence_status,
            "evidence_note": self.evidence_note,
            "disagreements": [d.to_dict() for d in self.disagreements],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FootfallValidationResult":
        """Reconstruct FootfallValidationResult from dictionary."""
        rate = data.get("metric_consistency_rate")
        return cls(
            dataset_name=data.get("dataset_name", "default"),
            expected_entry_counts={k: int(v) for k, v in data.get("expected_entry_counts", {}).items()},
            predicted_entry_counts={k: int(v) for k, v in data.get("predicted_entry_counts", {}).items()},
            entry_comparisons=[EntryCountComparison.from_dict(c) for c in data.get("entry_comparisons", [])],
            expected_transition_counts={k: int(v) for k, v in data.get("expected_transition_counts", {}).items()},
            predicted_transition_counts={k: int(v) for k, v in data.get("predicted_transition_counts", {}).items()},
            transition_comparisons=[
                TransitionCountComparison.from_dict(c) for c in data.get("transition_comparisons", [])
            ],
            total_compared_metrics=int(data["total_compared_metrics"]),
            matched_metric_count=int(data["matched_metric_count"]),
            mismatched_metric_count=int(data["mismatched_metric_count"]),
            metric_consistency_rate=float(rate) if rate is not None else None,
            evidence_status=data["evidence_status"],
            evidence_note=data["evidence_note"],
            disagreements=[FootfallMetricDisagreement.from_dict(d) for d in data.get("disagreements", [])],
        )


class FootfallValidator:
    """
    Validates aggregate Phase 7 FootfallSummary metrics against independent ground-truth events.
    """

    @classmethod
    def validate_summary(
        cls,
        summary: FootfallSummary,
        expected_events: List[GroundTruthEvent],
        dataset_name: str = "default",
        known_zones: Optional[List[str]] = None,
    ) -> FootfallValidationResult:
        """
        Compare Phase 7 FootfallSummary entry and transition counts against ground-truth events.

        Args:
            summary: FootfallSummary produced by FootfallAnalyzer.
            expected_events: List of GroundTruthEvent records.
            dataset_name: Identifier string.
            known_zones: Optional list of expected zone names to ensure representation.

        Returns:
            FootfallValidationResult with metric comparisons and disagreement records.

        Raises:
            TypeError: If summary is not a FootfallSummary or expected_events is not a list of GroundTruthEvents.
        """
        if not isinstance(summary, FootfallSummary):
            raise TypeError(f"summary must be a FootfallSummary. Got: {type(summary).__name__}")
        if not isinstance(expected_events, list):
            raise TypeError(f"expected_events must be a list. Got: {type(expected_events).__name__}")
        for idx, e in enumerate(expected_events):
            if not isinstance(e, GroundTruthEvent):
                raise TypeError(
                    f"All items in expected_events must be GroundTruthEvent instances. "
                    f"Item at index {idx} is {type(e).__name__}."
                )

        # Filter supported event types
        entry_events = [e for e in expected_events if e.event_type == "entry"]
        trans_events = [e for e in expected_events if e.event_type == "transition"]

        # If no usable ground-truth events provided, return INSUFFICIENT_EVIDENCE
        if len(entry_events) == 0 and len(trans_events) == 0:
            return FootfallValidationResult(
                dataset_name=dataset_name,
                expected_entry_counts={},
                predicted_entry_counts={},
                entry_comparisons=[],
                expected_transition_counts={},
                predicted_transition_counts={},
                transition_comparisons=[],
                total_compared_metrics=0,
                matched_metric_count=0,
                mismatched_metric_count=0,
                metric_consistency_rate=None,
                evidence_status="INSUFFICIENT_EVIDENCE",
                evidence_note=(
                    "No ground-truth entry or transition events provided. "
                    "Footfall metric consistency cannot be evaluated."
                ),
                disagreements=[],
            )

        # Gather all relevant zones
        all_zones: Set[str] = set(known_zones) if known_zones else set()
        all_zones.update(summary.entry_events_per_zone.keys())
        for e in entry_events:
            if e.to_zone_id:
                all_zones.add(e.to_zone_id)
        for e in trans_events:
            if e.from_zone_id:
                all_zones.add(e.from_zone_id)
            if e.to_zone_id:
                all_zones.add(e.to_zone_id)
        for z_from, targets in summary.transition_matrix.items():
            all_zones.add(z_from)
            for z_to in targets.keys():
                all_zones.add(z_to)

        sorted_zones = sorted(list(all_zones))

        # 1. Entry counts per zone
        expected_entries: Dict[str, int] = {z: 0 for z in sorted_zones}
        for e in entry_events:
            if e.to_zone_id in expected_entries:
                expected_entries[e.to_zone_id] += 1

        predicted_entries: Dict[str, int] = {
            z: summary.entry_events_per_zone.get(z, 0) for z in sorted_zones
        }

        entry_comparisons: List[EntryCountComparison] = []
        disagreements: List[FootfallMetricDisagreement] = []

        for z in sorted_zones:
            exp_c = expected_entries[z]
            pred_c = predicted_entries[z]
            diff = pred_c - exp_c
            abs_diff = abs(diff)
            match = (exp_c == pred_c)

            entry_comparisons.append(
                EntryCountComparison(
                    zone_id=z,
                    expected_count=exp_c,
                    predicted_count=pred_c,
                    difference=diff,
                    absolute_difference=abs_diff,
                    is_match=match,
                )
            )

            if not match:
                disagreements.append(
                    FootfallMetricDisagreement(
                        metric_type="entry_count",
                        key=z,
                        expected_count=exp_c,
                        predicted_count=pred_c,
                        difference=diff,
                        absolute_difference=abs_diff,
                        evidence_note=(
                            f"Entry count mismatch for zone '{z}': expected {exp_c}, "
                            f"predicted {pred_c} (diff={diff:+d}). Note: Descriptive aggregate difference; "
                            f"does not identify specific tracking or detection cause."
                        ),
                    )
                )

        # 2. Transition counts between zones
        expected_transitions: Dict[str, int] = {}
        predicted_transitions: Dict[str, int] = {}
        transition_comparisons: List[TransitionCountComparison] = []

        # Compare all pairs (z_from, z_to) with z_from != z_to
        for z_from in sorted_zones:
            for z_to in sorted_zones:
                if z_from == z_to:
                    continue
                pair_key = f"{z_from}->{z_to}"
                exp_c = sum(
                    1 for e in trans_events
                    if e.from_zone_id == z_from and e.to_zone_id == z_to
                )
                pred_c = summary.transition_matrix.get(z_from, {}).get(z_to, 0)
                diff = pred_c - exp_c
                abs_diff = abs(diff)
                match = (exp_c == pred_c)

                expected_transitions[pair_key] = exp_c
                predicted_transitions[pair_key] = pred_c

                transition_comparisons.append(
                    TransitionCountComparison(
                        from_zone_id=z_from,
                        to_zone_id=z_to,
                        expected_count=exp_c,
                        predicted_count=pred_c,
                        difference=diff,
                        absolute_difference=abs_diff,
                        is_match=match,
                    )
                )

                if not match:
                    disagreements.append(
                        FootfallMetricDisagreement(
                            metric_type="transition_count",
                            key=pair_key,
                            expected_count=exp_c,
                            predicted_count=pred_c,
                            difference=diff,
                            absolute_difference=abs_diff,
                            evidence_note=(
                                f"Transition count mismatch for '{pair_key}': expected {exp_c}, "
                                f"predicted {pred_c} (diff={diff:+d}). Note: Descriptive aggregate difference; "
                                f"does not identify specific tracking or trajectory cause."
                            ),
                        )
                    )

        total_compared = len(entry_comparisons) + len(transition_comparisons)
        matched_count = (
            sum(1 for c in entry_comparisons if c.is_match)
            + sum(1 for c in transition_comparisons if c.is_match)
        )
        mismatched_count = total_compared - matched_count
        rate = round(matched_count / total_compared, 4) if total_compared > 0 else None

        return FootfallValidationResult(
            dataset_name=dataset_name,
            expected_entry_counts=expected_entries,
            predicted_entry_counts=predicted_entries,
            entry_comparisons=entry_comparisons,
            expected_transition_counts=expected_transitions,
            predicted_transition_counts=predicted_transitions,
            transition_comparisons=transition_comparisons,
            total_compared_metrics=total_compared,
            matched_metric_count=matched_count,
            mismatched_metric_count=mismatched_count,
            metric_consistency_rate=rate,
            evidence_status="FOOTFALL_COMPARISON_EVALUATED",
            evidence_note=(
                f"Evaluated {total_compared} aggregate metrics ({len(entry_comparisons)} zone entry counts, "
                f"{len(transition_comparisons)} zone transition counts) across {len(sorted_zones)} zones. "
                f"Note: Descriptive event-count consistency on supplied data only; does not claim universal "
                f"tracking accuracy or real-world customer identity fidelity."
            ),
            disagreements=disagreements,
        )

    @classmethod
    def validate_sequence(
        cls,
        summary: FootfallSummary,
        sequence: GroundTruthSequence,
        known_zones: Optional[List[str]] = None,
    ) -> FootfallValidationResult:
        """
        Validate FootfallSummary against a GroundTruthSequence instance.

        Args:
            summary: FootfallSummary produced by FootfallAnalyzer.
            sequence: GroundTruthSequence containing events.
            known_zones: Optional list of known zone identifiers.

        Returns:
            FootfallValidationResult.
        """
        if not isinstance(sequence, GroundTruthSequence):
            raise TypeError(f"sequence must be a GroundTruthSequence. Got: {type(sequence).__name__}")
        return cls.validate_summary(
            summary=summary,
            expected_events=sequence.events,
            dataset_name=sequence.sequence_id,
            known_zones=known_zones,
        )


def validate_footfall(
    summary: FootfallSummary,
    expected_events: Union[List[GroundTruthEvent], GroundTruthSequence],
    dataset_name: str = "default",
    known_zones: Optional[List[str]] = None,
) -> FootfallValidationResult:
    """
    Convenience function to validate FootfallSummary against ground-truth events or sequence.
    """
    if isinstance(expected_events, GroundTruthSequence):
        return FootfallValidator.validate_sequence(
            summary=summary,
            sequence=expected_events,
            known_zones=known_zones,
        )
    return FootfallValidator.validate_summary(
        summary=summary,
        expected_events=expected_events,
        dataset_name=dataset_name,
        known_zones=known_zones,
    )
