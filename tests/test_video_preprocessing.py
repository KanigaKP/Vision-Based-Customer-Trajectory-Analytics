"""
Unit & Integration Tests for Video Preprocessing & Quality Verification
=======================================================================
Tests error conditions, invalid files, metadata extraction, quality evaluation,
and frame sampling using pytest.
"""

import os
import pytest
import numpy as np
import cv2

from src.video import VideoInspector, QualityEvaluator, FrameSampler


@pytest.fixture
def dummy_invalid_ext_file(tmp_path):
    """Fixture creating a file with an unsupported extension."""
    invalid_file = tmp_path / "sample.txt"
    invalid_file.write_text("This is not a video file.")
    return str(invalid_file)


@pytest.fixture
def dummy_corrupted_mp4_file(tmp_path):
    """Fixture creating a zero-byte or corrupt file with .mp4 extension."""
    corrupt_file = tmp_path / "corrupt_video.mp4"
    corrupt_file.write_bytes(b"CORRUPTED_HEADER_DATA_12345")
    return str(corrupt_file)


@pytest.fixture
def merl_test_video_path():
    """Fixture returning path to existing MERL test video."""
    path = os.path.join("data", "merl", "10_1_crop.mp4")
    if not os.path.isfile(path):
        pytest.skip(f"Test video {path} not found.")
    return path


# ============================================================
# 1. VideoInspector Tests
# ============================================================

def test_inspector_missing_video_path():
    inspector = VideoInspector()
    meta = inspector.inspect("non_existent_path_12345.mp4")
    
    assert not meta.exists
    assert not meta.is_readable
    assert meta.status == "NOT_FOUND"
    assert "File not found" in meta.error_message


def test_inspector_empty_path():
    inspector = VideoInspector()
    meta = inspector.inspect("")
    
    assert not meta.exists
    assert meta.status == "NOT_FOUND"
    assert "empty" in meta.error_message.lower()


def test_inspector_invalid_extension(dummy_invalid_ext_file):
    inspector = VideoInspector()
    meta = inspector.inspect(dummy_invalid_ext_file)
    
    assert meta.exists
    assert not meta.valid_extension
    assert meta.status == "INVALID"
    assert "Unsupported file extension" in meta.error_message


def test_inspector_corrupted_video(dummy_corrupted_mp4_file):
    inspector = VideoInspector()
    meta = inspector.inspect(dummy_corrupted_mp4_file)
    
    assert meta.exists
    assert meta.valid_extension
    assert not meta.is_readable
    assert meta.status == "CORRUPTED"
    assert "OpenCV failed" in meta.error_message or "invalid dimensions" in meta.error_message


def test_inspector_valid_video(merl_test_video_path):
    inspector = VideoInspector()
    meta = inspector.inspect(merl_test_video_path)
    
    assert meta.exists
    assert meta.valid_extension
    assert meta.is_readable
    assert meta.status == "VALID"
    assert meta.width == 920
    assert meta.height == 680
    assert meta.fps == 30.0
    assert meta.total_frames == 3796
    assert meta.duration_seconds > 100.0
    assert meta.sample_frame_accessible


# ============================================================
# 2. QualityEvaluator Tests
# ============================================================

def test_quality_evaluator_missing_file():
    evaluator = QualityEvaluator()
    assessment = evaluator.evaluate("missing_video.mp4")
    
    assert assessment.overall_status == "ERROR"
    assert len(assessment.errors) > 0


def test_quality_evaluator_valid_video(merl_test_video_path):
    evaluator = QualityEvaluator()
    assessment = evaluator.evaluate(merl_test_video_path, max_sample_frames=5)
    
    assert assessment.overall_status in ("PASS", "WARNING")
    assert assessment.width == 920
    assert assessment.height == 680
    assert assessment.brightness_mean > 0.0
    assert assessment.sharpness_laplacian_var > 0.0


def test_brightness_and_sharpness_math():
    # Create a synthetic black frame -> brightness 0
    black_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    b_black = QualityEvaluator.calculate_frame_brightness(black_frame)
    s_black = QualityEvaluator.calculate_frame_sharpness(black_frame)
    assert b_black == 0.0
    assert s_black == 0.0

    # Create a synthetic white frame -> brightness 255
    white_frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    b_white = QualityEvaluator.calculate_frame_brightness(white_frame)
    assert b_white == 255.0

    # Create a high-contrast checkerboard frame -> high Laplacian variance
    checkerboard = np.zeros((100, 100, 3), dtype=np.uint8)
    checkerboard[::2, ::2] = 255
    s_checker = QualityEvaluator.calculate_frame_sharpness(checkerboard)
    assert s_checker > 1000.0


# ============================================================
# 3. FrameSampler Tests
# ============================================================

def test_frame_sampler_valid_video(merl_test_video_path):
    sampler = FrameSampler(target_fps=5.0)
    samples = list(sampler.sample_frames(merl_test_video_path, max_frames=5))
    
    assert len(samples) == 5
    
    prev_idx = -1
    prev_time = -1.0
    for frame_idx, timestamp_sec, frame in samples:
        assert frame_idx > prev_idx
        assert timestamp_sec >= prev_time
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (680, 920, 3)
        prev_idx = frame_idx
        prev_time = timestamp_sec


def test_frame_sampler_invalid_video():
    sampler = FrameSampler()
    with pytest.raises(ValueError, match="Cannot sample frames"):
        list(sampler.sample_frames("non_existent_file.mp4"))
