"""
YOLO11 ONNX Post-Processing & Preprocessing Utilities
=====================================================
Handles letterbox image transformation, coordinate mapping, and vectorised
Non-Maximum Suppression (NMS) for YOLO11 ONNX outputs.
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2


@dataclass
class Detection:
    """Individual person detection container."""
    bbox: List[float]       # [x1, y1, x2, y2] in original image pixel space
    confidence: float       # Confidence score [0.0, 1.0]
    class_id: int = 0       # COCO class ID (0 for person)
    class_name: str = "person"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionResult:
    """Structured detection result container for a single frame."""
    frame_id: Optional[int]
    timestamp_sec: Optional[float]
    image_width: int
    image_height: int
    raw_candidate_count: int
    person_count: int
    detections: List[Detection]
    preprocess_time_ms: float
    inference_time_ms: float
    postprocess_time_ms: float
    total_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["detections"] = [det.to_dict() for det in self.detections]
        return d


def letterbox(
    img: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize and pad image while preserving aspect ratio (letterboxing).
    
    Returns
    -------
    padded_img : np.ndarray
        640x640 RGB image.
    ratio : float
        Scaling factor r = target_size / original_size.
    padding : Tuple[float, float]
        Padding offsets (dw, dh) applied to width and height.
    """
    h_orig, w_orig = img.shape[:2]
    target_w, target_h = target_size

    # Scale ratio (new / old)
    r = min(target_w / w_orig, target_h / h_orig)

    # Compute unpadded dimensions
    new_unpad_w = int(round(w_orig * r))
    new_unpad_h = int(round(h_orig * r))

    # Compute padding offsets
    dw = (target_w - new_unpad_w) / 2.0
    dh = (target_h - new_unpad_h) / 2.0

    if (w_orig, h_orig) != (new_unpad_w, new_unpad_h):
        img_resized = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    padded_img = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return padded_img, r, (dw, dh)


def postprocess_yolo11_output(
    output_tensor: np.ndarray,
    orig_shape: Tuple[int, int],
    ratio: float,
    padding: Tuple[float, float],
    conf_threshold: float = 0.40,
    iou_threshold: float = 0.45,
    target_class_id: int = 0,
    target_class_name: str = "person"
) -> Tuple[List[Detection], int]:
    """
    Parse YOLO11 ONNX output tensor (shape [1, 84, 8400]), filter target class (person),
    apply Non-Maximum Suppression (NMS), and scale bounding boxes back to original image space.

    Returns
    -------
    detections : List[Detection]
        List of filtered and NMS-suppressed person detections.
    raw_candidates_count : int
        Number of candidate anchors before confidence filtering.
    """
    orig_h, orig_w = orig_shape
    dw, dh = padding

    # Squeeze batch dimension -> shape (84, 8400)
    predictions = np.squeeze(output_tensor)
    if predictions.ndim != 2 or predictions.shape[0] != 84:
        raise ValueError(f"Unexpected YOLO11 output tensor shape: {output_tensor.shape}. Expected (1, 84, 8400).")

    # Bounding boxes (cx, cy, w, h) in 640x640 space
    boxes_raw = predictions[:4, :].T  # (8400, 4)
    raw_candidates_count = boxes_raw.shape[0]

    # Target class confidence scores (row index 4 + target_class_id)
    # COCO person class ID 0 corresponds to row 4
    class_row_idx = 4 + target_class_id
    class_scores = predictions[class_row_idx, :]  # (8400,)

    # Step 1: Confidence threshold filtering
    mask = class_scores >= conf_threshold
    if not np.any(mask):
        return [], raw_candidates_count

    filtered_boxes = boxes_raw[mask]
    filtered_scores = class_scores[mask]

    # Convert [cx, cy, w, h] -> [x1, y1, x2, y2] in 640x640 letterboxed space
    cx = filtered_boxes[:, 0]
    cy = filtered_boxes[:, 1]
    w = filtered_boxes[:, 2]
    h = filtered_boxes[:, 3]

    x1_640 = cx - (w / 2.0)
    y1_640 = cy - (h / 2.0)
    x2_640 = cx + (w / 2.0)
    y2_640 = cy + (h / 2.0)

    # Step 2: Scale coordinates back to original image space
    x1_orig = np.clip((x1_640 - dw) / ratio, 0, orig_w)
    y1_orig = np.clip((y1_640 - dh) / ratio, 0, orig_h)
    x2_orig = np.clip((x2_640 - dw) / ratio, 0, orig_w)
    y2_orig = np.clip((y2_640 - dh) / ratio, 0, orig_h)

    # Convert to format required for OpenCV NMSBoxes: [x, y, width, height]
    nms_boxes = []
    nms_scores = []
    box_coords_orig = []

    for i in range(len(filtered_scores)):
        x1, y1, x2, y2 = x1_orig[i], y1_orig[i], x2_orig[i], y2_orig[i]
        bw = x2 - x1
        bh = y2 - y1
        if bw > 1 and bh > 1:  # Filter zero/degenerate box dimensions
            nms_boxes.append([float(x1), float(y1), float(bw), float(bh)])
            nms_scores.append(float(filtered_scores[i]))
            box_coords_orig.append([round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)])

    if not nms_boxes:
        return [], raw_candidates_count

    # Step 3: Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(
        bboxes=nms_boxes,
        scores=nms_scores,
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold
    )

    final_detections: List[Detection] = []
    if len(indices) > 0:
        # cv2.dnn.NMSBoxes can return flat array or 2D array depending on OpenCV version
        flat_indices = indices.flatten() if isinstance(indices, np.ndarray) else [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]
        
        for idx in flat_indices:
            bbox = box_coords_orig[idx]
            conf = nms_scores[idx]
            final_detections.append(
                Detection(
                    bbox=bbox,
                    confidence=round(conf, 4),
                    class_id=target_class_id,
                    class_name=target_class_name
                )
            )

    return final_detections, raw_candidates_count
