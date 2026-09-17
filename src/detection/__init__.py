"""
Object Detection Package
========================
Provides YOLO11 person detection modules powered by ONNX Runtime CPU.
"""

from src.detection.detector import YOLO11PersonDetector
from src.detection.postprocess import (
    Detection,
    DetectionResult,
    letterbox,
    postprocess_yolo11_output
)

__all__ = [
    "YOLO11PersonDetector",
    "Detection",
    "DetectionResult",
    "letterbox",
    "postprocess_yolo11_output",
]
