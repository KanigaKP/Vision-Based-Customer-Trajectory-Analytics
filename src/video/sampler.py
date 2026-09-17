"""
Frame Sampler Module
====================
Provides memory-efficient, lazy frame sampling from video streams for downstream analysis.
"""

import os
from typing import Generator, Tuple, Optional
import numpy as np
import cv2
from src.video.inspector import VideoInspector, VideoMetadata


class FrameSampler:
    """
    Lazy frame sampler yielding individual sampled frames from a video file.
    Does NOT load the entire video into memory.
    """

    def __init__(self, target_fps: float = 5.0, max_sampled_frames: int = 1000):
        self.target_fps = target_fps
        self.max_sampled_frames = max_sampled_frames

    def sample_frames(
        self,
        video_path: str,
        override_target_fps: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Generator yielding (frame_index, timestamp_seconds, frame_bgr) from video_path.
        
        Parameters
        ----------
        video_path : str
            Path to video file.
        override_target_fps : Optional[float]
            Optional target sampling FPS overriding default.
        max_frames : Optional[int]
            Optional cap on max sampled frames.
        """
        inspector = VideoInspector()
        meta: VideoMetadata = inspector.inspect(video_path)

        if not meta.exists or not meta.is_readable or meta.total_frames <= 0:
            raise ValueError(f"Cannot sample frames: video unreadable or invalid ({meta.error_message})")

        target_fps = override_target_fps if override_target_fps is not None else self.target_fps
        max_limit = max_frames if max_frames is not None else self.max_sampled_frames

        native_fps = meta.fps if meta.fps > 0 else 30.0
        
        # Calculate step interval (number of native frames to jump per sample)
        if target_fps <= 0 or target_fps >= native_fps:
            step = 1
        else:
            step = max(1, int(round(native_fps / target_fps)))

        cap = cv2.VideoCapture(meta.file_path)
        sampled_count = 0

        try:
            current_frame_idx = 0
            while cap.isOpened() and sampled_count < max_limit:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                
                if current_frame_idx % step == 0:
                    timestamp_sec = round(current_frame_idx / native_fps, 3)
                    yield (current_frame_idx, timestamp_sec, frame)
                    sampled_count += 1
                
                current_frame_idx += 1
        finally:
            cap.release()
