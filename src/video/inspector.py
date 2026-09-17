"""
Video Inspection & Metadata Extraction Module
=============================================
Provides robust metadata extraction and graceful error handling for input video files.
"""

import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import cv2


@dataclass
class VideoMetadata:
    """Structured video metadata container."""
    file_path: str
    file_name: str
    file_size_bytes: int
    exists: bool
    valid_extension: bool
    is_readable: bool
    fourcc: str
    codec_name: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    sample_frame_accessible: bool
    status: str  # "VALID", "INVALID", "CORRUPTED", "NOT_FOUND"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VideoInspector:
    """Inspector for validating video files and extracting technical metadata."""

    DEFAULT_ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(self, allowed_extensions=None):
        self.allowed_extensions = set(allowed_extensions) if allowed_extensions else self.DEFAULT_ALLOWED_EXTENSIONS

    @staticmethod
    def _fourcc_to_string(fourcc_val: float) -> str:
        """Convert OpenCV FourCC float value to readable 4-character codec code."""
        try:
            val = int(fourcc_val)
            chars = [chr((val >> (8 * i)) & 0xFF) for i in range(4)]
            codec = "".join(chars).strip()
            # Clean non-printable characters
            clean_codec = "".join(c for c in codec if c.isprintable() and c != "\x00")
            return clean_codec if clean_codec else "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def inspect(self, video_path: str) -> VideoMetadata:
        """
        Inspect a video file path and extract metadata gracefully.
        Handles missing files, unsupported formats, and corrupted videos without crashing.
        """
        if not video_path:
            return VideoMetadata(
                file_path="",
                file_name="",
                file_size_bytes=0,
                exists=False,
                valid_extension=False,
                is_readable=False,
                fourcc="",
                codec_name="UNKNOWN",
                width=0,
                height=0,
                fps=0.0,
                total_frames=0,
                duration_seconds=0.0,
                sample_frame_accessible=False,
                status="NOT_FOUND",
                error_message="Video path is empty or None."
            )

        abs_path = os.path.abspath(video_path)
        file_name = os.path.basename(abs_path)

        if not os.path.isfile(abs_path):
            return VideoMetadata(
                file_path=abs_path,
                file_name=file_name,
                file_size_bytes=0,
                exists=False,
                valid_extension=False,
                is_readable=False,
                fourcc="",
                codec_name="UNKNOWN",
                width=0,
                height=0,
                fps=0.0,
                total_frames=0,
                duration_seconds=0.0,
                sample_frame_accessible=False,
                status="NOT_FOUND",
                error_message=f"File not found: {abs_path}"
            )

        file_size = os.path.getsize(abs_path)
        ext = os.path.splitext(file_name)[1].lower()
        valid_ext = ext in self.allowed_extensions

        if not valid_ext:
            return VideoMetadata(
                file_path=abs_path,
                file_name=file_name,
                file_size_bytes=file_size,
                exists=True,
                valid_extension=False,
                is_readable=False,
                fourcc="",
                codec_name="UNKNOWN",
                width=0,
                height=0,
                fps=0.0,
                total_frames=0,
                duration_seconds=0.0,
                sample_frame_accessible=False,
                status="INVALID",
                error_message=f"Unsupported file extension: {ext}. Allowed: {sorted(list(self.allowed_extensions))}"
            )

        # Attempt to open with OpenCV
        cap = None
        try:
            cap = cv2.VideoCapture(abs_path)
            if not cap.isOpened():
                return VideoMetadata(
                    file_path=abs_path,
                    file_name=file_name,
                    file_size_bytes=file_size,
                    exists=True,
                    valid_extension=True,
                    is_readable=False,
                    fourcc="",
                    codec_name="UNKNOWN",
                    width=0,
                    height=0,
                    fps=0.0,
                    total_frames=0,
                    duration_seconds=0.0,
                    sample_frame_accessible=False,
                    status="CORRUPTED",
                    error_message="OpenCV failed to open video file (corrupted or unreadable stream)."
                )

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc_raw = cap.get(cv2.CAP_PROP_FOURCC)
            codec_name = self._fourcc_to_string(fourcc_raw)

            # Test sample frame accessibility
            ret, frame = cap.read()
            sample_accessible = bool(ret and frame is not None and frame.size > 0)

            if total_frames <= 0 or width <= 0 or height <= 0 or not sample_accessible:
                return VideoMetadata(
                    file_path=abs_path,
                    file_name=file_name,
                    file_size_bytes=file_size,
                    exists=True,
                    valid_extension=True,
                    is_readable=sample_accessible,
                    fourcc=str(fourcc_raw),
                    codec_name=codec_name,
                    width=width,
                    height=height,
                    fps=fps,
                    total_frames=total_frames,
                    duration_seconds=0.0,
                    sample_frame_accessible=sample_accessible,
                    status="CORRUPTED",
                    error_message=f"Video stream contains invalid dimensions ({width}x{height}) or zero decodable frames."
                )

            duration = total_frames / fps if fps > 0 else 0.0

            return VideoMetadata(
                file_path=abs_path,
                file_name=file_name,
                file_size_bytes=file_size,
                exists=True,
                valid_extension=True,
                is_readable=True,
                fourcc=str(fourcc_raw),
                codec_name=codec_name,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_seconds=round(duration, 2),
                sample_frame_accessible=True,
                status="VALID",
                error_message=None
            )

        except Exception as e:
            return VideoMetadata(
                file_path=abs_path,
                file_name=file_name,
                file_size_bytes=file_size,
                exists=True,
                valid_extension=valid_ext,
                is_readable=False,
                fourcc="",
                codec_name="UNKNOWN",
                width=0,
                height=0,
                fps=0.0,
                total_frames=0,
                duration_seconds=0.0,
                sample_frame_accessible=False,
                status="CORRUPTED",
                error_message=f"Exception during video inspection: {type(e).__name__}: {str(e)}"
            )
        finally:
            if cap is not None:
                cap.release()
