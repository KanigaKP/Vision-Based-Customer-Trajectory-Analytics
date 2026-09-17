"""
Unit & Integration Tests for YOLO11 Person Detector Module
===========================================================
Tests model loading, preprocessing, post-processing, person filtering,
confidence thresholds, and detection execution using pytest.
"""

import os
import pytest
import numpy as np
import cv2

from src.detection import YOLO11PersonDetector, DetectionResult, Detection, letterbox


@pytest.fixture
def valid_model_path():
    """Fixture returning path to existing YOLO11 ONNX model asset."""
    path = os.path.join("data", "models", "yolo11n.onnx")
    if not os.path.isfile(path):
        pytest.skip(f"Model file {path} not found.")
    return path


@pytest.fixture
def known_test_frame_path():
    """Fixture returning path to MOT17 test frame 1."""
    path = os.path.join("data", "mot17", "train", "MOT17-04-FRCNN", "img1", "000001.jpg")
    if not os.path.isfile(path):
        pytest.skip(f"Test frame {path} not found.")
    return path


# ============================================================
# 1. Model Loading & Initialization Tests
# ============================================================

def test_missing_model_raises_error():
    with pytest.raises(FileNotFoundError, match="YOLO11 ONNX model file not found"):
        YOLO11PersonDetector(model_path="data/models/non_existent_model_12345.onnx")


def test_detector_initialization(valid_model_path):
    detector = YOLO11PersonDetector(model_path=valid_model_path, conf_threshold=0.40)
    meta = detector.get_metadata()
    
    assert detector.input_name == "images"
    assert detector.output_name == "output0"
    assert detector.input_width == 640
    assert detector.input_height == 640
    assert meta["conf_threshold"] == 0.40
    assert meta["target_class_id"] == 0
    assert meta["target_class_name"] == "person"
    assert "CPUExecutionProvider" in meta["providers"]


# ============================================================
# 2. Preprocessing & Letterboxing Tests
# ============================================================

def test_letterbox_preprocessing():
    # Test image 1920x1080
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    padded, ratio, (dw, dh) = letterbox(img, target_size=(640, 640))
    
    assert padded.shape == (640, 640, 3)
    assert ratio == pytest.approx(640 / 1920, rel=1e-3)
    assert dw == 0.0
    assert dh == pytest.approx((640 - 1080 * (640 / 1920)) / 2.0, abs=1.0)


def test_detector_preprocess_tensor_shape(valid_model_path):
    detector = YOLO11PersonDetector(model_path=valid_model_path)
    img_bgr = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    tensor, ratio, padding = detector.preprocess(img_bgr)
    
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


# ============================================================
# 3. Output Structure & Detection Execution Tests
# ============================================================

def test_detect_empty_image_raises_error(valid_model_path):
    detector = YOLO11PersonDetector(model_path=valid_model_path)
    with pytest.raises(ValueError, match="Input image is empty or None"):
        detector.detect(np.array([]))


def test_detection_output_structure(valid_model_path):
    detector = YOLO11PersonDetector(model_path=valid_model_path)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(img, frame_id=10, timestamp_sec=2.0)
    
    assert isinstance(result, DetectionResult)
    assert result.frame_id == 10
    assert result.timestamp_sec == 2.0
    assert result.image_width == 640
    assert result.image_height == 480
    assert result.raw_candidate_count == 8400
    assert isinstance(result.detections, list)
    assert result.preprocess_time_ms >= 0.0
    assert result.inference_time_ms > 0.0
    assert result.postprocess_time_ms >= 0.0
    assert result.total_time_ms > 0.0


# ============================================================
# 4. Person Class Filtering & Confidence Threshold Behavior
# ============================================================

def test_confidence_threshold_behavior(valid_model_path, known_test_frame_path):
    img = cv2.imread(known_test_frame_path)
    
    # Detector with high confidence threshold
    det_high = YOLO11PersonDetector(model_path=valid_model_path, conf_threshold=0.95)
    res_high = det_high.detect(img)
    
    # Detector with low confidence threshold
    det_low = YOLO11PersonDetector(model_path=valid_model_path, conf_threshold=0.20)
    res_low = det_low.detect(img)
    
    # Low threshold should detect equal or more candidates than high threshold
    assert res_low.person_count >= res_high.person_count
    
    for d in res_high.detections:
        assert d.confidence >= 0.95
        assert d.class_id == 0
        assert d.class_name == "person"

    for d in res_low.detections:
        assert d.confidence >= 0.20
        assert d.class_id == 0
        assert d.class_name == "person"


# ============================================================
# 5. End-to-End Detection on Known Test Frame
# ============================================================

def test_detection_on_known_test_frame(valid_model_path, known_test_frame_path):
    detector = YOLO11PersonDetector(model_path=valid_model_path, conf_threshold=0.40)
    img = cv2.imread(known_test_frame_path)
    result = detector.detect(img, frame_id=1, timestamp_sec=0.0)
    
    assert result.person_count > 0
    assert len(result.detections) == result.person_count
    
    for det in result.detections:
        assert len(det.bbox) == 4
        x1, y1, x2, y2 = det.bbox
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080
        assert det.confidence >= 0.40
        assert det.class_id == 0
        assert det.class_name == "person"


def test_annotate_image(valid_model_path, known_test_frame_path):
    detector = YOLO11PersonDetector(model_path=valid_model_path)
    img = cv2.imread(known_test_frame_path)
    result = detector.detect(img)
    
    annotated = detector.annotate_image(img, result.detections)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == img.shape
