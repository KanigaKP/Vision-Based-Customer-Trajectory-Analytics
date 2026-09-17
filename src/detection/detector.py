"""
YOLO11 Person Detector Module (ONNX Runtime CPU Engine)
======================================================
Provides single-frame and video stream person detection using a pre-converted
YOLO11 ONNX model executed via ONNX Runtime CPU.

DOES NOT require or import PyTorch for runtime inference.
"""

import os
import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
import onnxruntime as ort

from src.detection.postprocess import (
    Detection,
    DetectionResult,
    letterbox,
    postprocess_yolo11_output
)


class YOLO11PersonDetector:
    """
    ONNX Runtime-powered YOLO11 Person Detector.
    Strictly filters COCO person class (class ID 0).
    """

    DEFAULT_MODEL_PATH = "data/models/yolo11n.onnx"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        target_class_id: int = 0,
        target_class_name: str = "person"
    ):
        self.model_path = os.path.abspath(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_class_id = target_class_id
        self.target_class_name = target_class_name

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"YOLO11 ONNX model file not found at: {self.model_path}. "
                "Ensure data/models/yolo11n.onnx exists before instantiating detector."
            )

        # Initialize ONNX Runtime InferenceSession on CPU
        t0 = time.time()
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.load_time_ms = (time.time() - t0) * 1000.0

        # Inspect Inputs & Outputs
        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]
        
        self.input_name = self.input_info.name
        self.output_name = self.output_info.name
        
        # Expected input shape: [1, 3, 640, 640]
        self.input_width = self.input_info.shape[3] if len(self.input_info.shape) == 4 else 640
        self.input_height = self.input_info.shape[2] if len(self.input_info.shape) == 4 else 640

    def get_metadata(self) -> Dict[str, Any]:
        """Return model and session technical metadata."""
        return {
            "model_path": self.model_path,
            "load_time_ms": round(self.load_time_ms, 2),
            "input_name": self.input_name,
            "input_shape": list(self.input_info.shape),
            "output_name": self.output_name,
            "output_shape": list(self.output_info.shape),
            "providers": self.session.get_providers(),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "target_class_id": self.target_class_id,
            "target_class_name": self.target_class_name
        }

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Preprocess BGR image for YOLO11 ONNX inference.
        Applies letterboxing (640x640), BGR->RGB conversion, normalization [0, 1], and CHW layout.
        """
        # Letterbox to (input_width, input_height)
        padded_img, ratio, padding = letterbox(
            img_bgr, target_size=(self.input_width, self.input_height)
        )

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)

        # Normalize 0.0 - 1.0 and convert to float32
        tensor = img_rgb.astype(np.float32) / 255.0

        # HWC -> CHW -> NCHW (1, 3, H, W)
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor, ratio, padding

    def detect(
        self,
        img_bgr: np.ndarray,
        frame_id: Optional[int] = None,
        timestamp_sec: Optional[float] = None,
        conf_threshold_override: Optional[float] = None
    ) -> DetectionResult:
        """
        Run person detection on a single BGR image frame.
        Measures preprocessing, inference, and post-processing latency breakdown.
        """
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Input image is empty or None.")

        orig_h, orig_w = img_bgr.shape[:2]
        conf_thresh = conf_threshold_override if conf_threshold_override is not None else self.conf_threshold

        # Step 1: Preprocessing
        t0 = time.time()
        input_tensor, ratio, padding = self.preprocess(img_bgr)
        t_prep = (time.time() - t0) * 1000.0

        # Step 2: ONNX Runtime CPU Inference
        t1 = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        t_infer = (time.time() - t1) * 1000.0

        # Step 3: Post-processing (Person filtering & NMS)
        t2 = time.time()
        output_tensor = outputs[0]
        detections, raw_candidate_count = postprocess_yolo11_output(
            output_tensor=output_tensor,
            orig_shape=(orig_h, orig_w),
            ratio=ratio,
            padding=padding,
            conf_threshold=conf_thresh,
            iou_threshold=self.iou_threshold,
            target_class_id=self.target_class_id,
            target_class_name=self.target_class_name
        )
        t_post = (time.time() - t2) * 1000.0

        total_time = (time.time() - t0) * 1000.0

        return DetectionResult(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            image_width=orig_w,
            image_height=orig_h,
            raw_candidate_count=raw_candidate_count,
            person_count=len(detections),
            detections=detections,
            preprocess_time_ms=round(t_prep, 2),
            inference_time_ms=round(t_infer, 2),
            postprocess_time_ms=round(t_post, 2),
            total_time_ms=round(total_time, 2)
        )

    @staticmethod
    def annotate_image(
        img_bgr: np.ndarray,
        detections: List[Detection],
        box_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw bounding boxes and confidence labels on image for visual inspection."""
        annotated = img_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
            
            label = f"{det.class_name}: {det.confidence:.2f}"
            (w_label, h_label), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            y_text = max(y1, h_label + 10)
            cv2.rectangle(
                annotated,
                (x1, y_text - h_label - 5),
                (x1 + w_label + 10, y_text + baseline),
                box_color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y_text - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1
            )
        return annotated
