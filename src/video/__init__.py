"""
Video Processing Package
========================
Provides video validation, quality assessment, and lazy frame sampling modules.
"""

from src.video.inspector import VideoInspector, VideoMetadata
from src.video.quality import QualityEvaluator, QualityAssessment
from src.video.sampler import FrameSampler

__all__ = [
    "VideoInspector",
    "VideoMetadata",
    "QualityEvaluator",
    "QualityAssessment",
    "FrameSampler",
]
