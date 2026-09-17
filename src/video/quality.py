"""
Video Quality Evaluation Module
===============================
Evaluates video quality indicators (Resolution, FPS, Brightness, Sharpness)
against configurable engineering warning thresholds.

IMPORTANT METHODOLOGICAL NOTE:
The quality indicators and thresholds used here are configurable engineering
warning thresholds rather than scientifically validated universal limits.
They serve to flag potential input anomalies before downstream processing.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from src.video.inspector import VideoMetadata, VideoInspector


@dataclass
class QualityAssessment:
    """Container for video quality evaluation results."""
    file_name: str
    overall_status: str             # "PASS", "WARNING", "ERROR"
    resolution_status: str          # "PASS", "WARNING", "ERROR"
    fps_status: str                 # "PASS", "WARNING", "ERROR"
    brightness_status: str          # "PASS", "WARNING", "ERROR"
    sharpness_status: str           # "PASS", "WARNING", "ERROR"
    readability_status: str         # "PASS", "WARNING", "ERROR"
    
    width: int
    height: int
    fps: float
    brightness_mean: float          # Average brightness (0-255 scale)
    sharpness_laplacian_var: float  # Mean Laplacian variance (higher = sharper)
    
    warnings: List[str]
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QualityEvaluator:
    """Evaluates video quality against configurable warning thresholds."""

    DEFAULT_THRESHOLDS = {
        "min_width": 640,
        "min_height": 480,
        "min_fps": 10.0,
        "brightness": {
            "min_mean": 40.0,
            "max_mean": 230.0,
        },
        "sharpness": {
            "min_laplacian_var": 100.0,
        }
    }

    def __init__(self, thresholds_config: Optional[Dict[str, Any]] = None):
        self.config = thresholds_config if thresholds_config else self.DEFAULT_THRESHOLDS

    @staticmethod
    def calculate_frame_brightness(frame: np.ndarray) -> float:
        """Calculate mean brightness of a BGR frame using grayscale channel."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    @staticmethod
    def calculate_frame_sharpness(frame: np.ndarray) -> float:
        """Calculate image sharpness using the variance of the Laplacian operator."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def evaluate(self, video_path: str, max_sample_frames: int = 15) -> QualityAssessment:
        """
        Evaluate video file quality parameters.
        Sample up to `max_sample_frames` frames evenly across the video duration to compute average brightness and sharpness.
        """
        inspector = VideoInspector()
        meta: VideoMetadata = inspector.inspect(video_path)

        warnings: List[str] = []
        errors: List[str] = []

        if not meta.exists or not meta.is_readable or meta.status in ("NOT_FOUND", "INVALID", "CORRUPTED"):
            errors.append(f"Video unreadable: {meta.error_message}")
            return QualityAssessment(
                file_name=meta.file_name or "unknown",
                overall_status="ERROR",
                resolution_status="ERROR",
                fps_status="ERROR",
                brightness_status="ERROR",
                sharpness_status="ERROR",
                readability_status="ERROR",
                width=meta.width,
                height=meta.height,
                fps=meta.fps,
                brightness_mean=0.0,
                sharpness_laplacian_var=0.0,
                warnings=warnings,
                errors=errors
            )

        # 1. Resolution Check
        min_w = self.config.get("min_width", 640)
        min_h = self.config.get("min_height", 480)
        resolution_status = "PASS"
        if meta.width < min_w or meta.height < min_h:
            resolution_status = "WARNING"
            warnings.append(
                f"Resolution {meta.width}x{meta.height} is below recommended warning threshold {min_w}x{min_h}."
            )

        # 2. FPS Check
        min_fps = self.config.get("min_fps", 10.0)
        fps_status = "PASS"
        if meta.fps < min_fps:
            fps_status = "WARNING"
            warnings.append(f"Frame rate {meta.fps:.2f} FPS is below warning threshold {min_fps} FPS.")

        # 3. Frame Sampling for Brightness & Sharpness Metrics
        cap = cv2.VideoCapture(meta.file_path)
        brightness_list: List[float] = []
        sharpness_list: List[float] = []
        readability_status = "PASS"

        try:
            total_frames = meta.total_frames
            step = max(1, total_frames // max_sample_frames)
            
            for idx in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    readability_status = "WARNING"
                    warnings.append(f"Failed to read frame at position index {idx}.")
                    continue
                
                brightness_list.append(self.calculate_frame_brightness(frame))
                sharpness_list.append(self.calculate_frame_sharpness(frame))
                
                if len(brightness_list) >= max_sample_frames:
                    break
        except Exception as e:
            readability_status = "WARNING"
            warnings.append(f"Exception during frame metric evaluation: {e}")
        finally:
            cap.release()

        avg_brightness = float(np.mean(brightness_list)) if brightness_list else 0.0
        avg_sharpness = float(np.mean(sharpness_list)) if sharpness_list else 0.0

        # Brightness Status Check
        b_cfg = self.config.get("brightness", {})
        min_b = b_cfg.get("min_mean", 40.0)
        max_b = b_cfg.get("max_mean", 230.0)
        brightness_status = "PASS"
        if avg_brightness < min_b:
            brightness_status = "WARNING"
            warnings.append(f"Average brightness ({avg_brightness:.1f}) is low (below warning threshold {min_b}). Video may be underexposed.")
        elif avg_brightness > max_b:
            brightness_status = "WARNING"
            warnings.append(f"Average brightness ({avg_brightness:.1f}) is high (above warning threshold {max_b}). Video may be overexposed.")

        # Sharpness Status Check
        s_cfg = self.config.get("sharpness", {})
        min_s = s_cfg.get("min_laplacian_var", 100.0)
        sharpness_status = "PASS"
        if avg_sharpness < min_s:
            sharpness_status = "WARNING"
            warnings.append(f"Average sharpness / Laplacian variance ({avg_sharpness:.1f}) is low (below warning threshold {min_s}). Video may be blurry.")

        # Determine Overall Status
        if errors:
            overall_status = "ERROR"
        elif warnings:
            overall_status = "WARNING"
        else:
            overall_status = "PASS"

        return QualityAssessment(
            file_name=meta.file_name,
            overall_status=overall_status,
            resolution_status=resolution_status,
            fps_status=fps_status,
            brightness_status=brightness_status,
            sharpness_status=sharpness_status,
            readability_status=readability_status,
            width=meta.width,
            height=meta.height,
            fps=round(meta.fps, 2),
            brightness_mean=round(avg_brightness, 2),
            sharpness_laplacian_var=round(avg_sharpness, 2),
            warnings=warnings,
            errors=errors
        )
