"""
Analytics package for Vision-Based Customer Trajectory Analytics.
Phase 6: Spatial Zone Definition & Evaluation.
"""

from src.analytics.zone import Zone, ZoneConfig
from src.analytics.zone_evaluator import ZoneEvaluator
from src.analytics.footfall import (
    ZoneEntryEvent,
    ZoneVisit,
    FootfallSummary,
    FootfallAnalyzer,
)

__all__ = [
    "Zone",
    "ZoneConfig",
    "ZoneEvaluator",
    "ZoneEntryEvent",
    "ZoneVisit",
    "FootfallSummary",
    "FootfallAnalyzer",
]
