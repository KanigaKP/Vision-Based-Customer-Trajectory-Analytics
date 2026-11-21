"""
Independent Homography Validation Module
========================================
Evaluates image-to-floor homography transformations using independent held-out
calibration points that were strictly excluded from model estimation.

Key Principles:
1. Strict Role Separation:
   - 'training' points are used exclusively to compute the 3x3 homography matrix H.
   - 'held_out' points are used exclusively for independent out-of-sample error evaluation.
   - Held-out points are NEVER supplied to cv2.findHomography.
2. Diagnostic Separation:
   - Training reprojection error measures in-sample fitting quality.
   - Held-out error measures independent out-of-sample geometric consistency.
3. Explicit Evidence State:
   - When no held-out points are available, an explicit 'INSUFFICIENT_EVIDENCE'
     state is returned rather than manufacturing validation claims.
"""

from dataclasses import dataclass, asdict
import math
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from src.spatial.homography import HomographyCalibrator
from src.validation.ground_truth import (
    GroundTruthCalibrationPoint,
    GroundTruthSequence,
)


@dataclass
class HeldOutPointValidation:
    """
    Validation evaluation result for a single independent held-out calibration point.

    Attributes:
        label: Point identifier string (e.g. 'CP1').
        image_point: Observed image coordinates (image_x, image_y).
        reference_floor_point: Ground-truth reference floor coordinates (floor_x, floor_y).
        predicted_floor_point: Homography-predicted floor coordinates (floor_x, floor_y).
        error: Euclidean distance between predicted and reference floor coordinates.
        unit: Metric or grid unit string (e.g. 'metres').
    """
    label: str
    image_point: Tuple[float, float]
    reference_floor_point: Tuple[float, float]
    predicted_floor_point: Tuple[float, float]
    error: float
    unit: str = "metres"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "label": self.label,
            "image_point": list(self.image_point),
            "reference_floor_point": list(self.reference_floor_point),
            "predicted_floor_point": list(self.predicted_floor_point),
            "error": self.error,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HeldOutPointValidation":
        """Reconstruct HeldOutPointValidation from dictionary."""
        return cls(
            label=data["label"],
            image_point=tuple(data["image_point"]),
            reference_floor_point=tuple(data["reference_floor_point"]),
            predicted_floor_point=tuple(data["predicted_floor_point"]),
            error=float(data["error"]),
            unit=data.get("unit", "metres"),
        )


@dataclass
class HomographyValidationResult:
    """
    Aggregate evaluation results for homography transformation.

    Maintains strict separation between in-sample training reprojection error
    and out-of-sample independent held-out error.
    """
    training_point_count: int
    held_out_point_count: int
    training_reprojection_mean_error: float
    training_reprojection_max_error: float
    held_out_mean_error: Optional[float]
    held_out_max_error: Optional[float]
    held_out_results: List[HeldOutPointValidation]
    evidence_status: str
    evidence_note: str
    unit: str = "metres"
    dataset_name: str = "default"
    homography_matrix: Optional[List[List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "unit": self.unit,
            "training_point_count": self.training_point_count,
            "held_out_point_count": self.held_out_point_count,
            "training_reprojection_mean_error": self.training_reprojection_mean_error,
            "training_reprojection_max_error": self.training_reprojection_max_error,
            "held_out_mean_error": self.held_out_mean_error,
            "held_out_max_error": self.held_out_max_error,
            "evidence_status": self.evidence_status,
            "evidence_note": self.evidence_note,
            "homography_matrix": self.homography_matrix,
            "held_out_results": [r.to_dict() for r in self.held_out_results],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HomographyValidationResult":
        """Reconstruct HomographyValidationResult from dictionary."""
        held_out_results = [
            HeldOutPointValidation.from_dict(r) for r in data.get("held_out_results", [])
        ]
        return cls(
            training_point_count=data["training_point_count"],
            held_out_point_count=data["held_out_point_count"],
            training_reprojection_mean_error=float(data["training_reprojection_mean_error"]),
            training_reprojection_max_error=float(data["training_reprojection_max_error"]),
            held_out_mean_error=(
                float(data["held_out_mean_error"])
                if data.get("held_out_mean_error") is not None
                else None
            ),
            held_out_max_error=(
                float(data["held_out_max_error"])
                if data.get("held_out_max_error") is not None
                else None
            ),
            held_out_results=held_out_results,
            evidence_status=data["evidence_status"],
            evidence_note=data["evidence_note"],
            unit=data.get("unit", "metres"),
            dataset_name=data.get("dataset_name", "default"),
            homography_matrix=data.get("homography_matrix"),
        )


class HomographyValidator:
    """
    Validates homography transformations using Phase 8.1 GroundTruthCalibrationPoint records.

    Enforces that calibration points with role='held_out' are evaluated independently
    and never passed to the estimator.
    """

    @staticmethod
    def validate_calibration_points(
        calibration_points: List[GroundTruthCalibrationPoint],
        unit: str = "metres",
        dataset_name: str = "default",
    ) -> HomographyValidationResult:
        """
        Estimate a homography from role='training' points and evaluate against role='held_out' points.

        Args:
            calibration_points: List of GroundTruthCalibrationPoint instances.
            unit: Metric or coordinate unit string.
            dataset_name: Label identifier for the dataset or camera view.

        Returns:
            HomographyValidationResult containing separate training diagnostics and
            independent held-out validation metrics.

        Raises:
            ValueError: If fewer than 4 training points are supplied, or if points are degenerate.
        """
        training_pts = [cp for cp in calibration_points if cp.role == "training"]
        held_out_pts = [cp for cp in calibration_points if cp.role == "held_out"]

        if len(training_pts) < 4:
            raise ValueError(
                f"At least 4 valid training calibration points are required to fit a homography. "
                f"Got {len(training_pts)}."
            )

        # Build training coordinate sets (strictly excluding held-out points)
        train_image_pts = [[cp.image_x, cp.image_y] for cp in training_pts]
        train_floor_pts = [[cp.floor_x, cp.floor_y] for cp in training_pts]

        # Fit homography using ONLY training points via HomographyCalibrator
        calibrator = HomographyCalibrator(
            image_points=train_image_pts,
            floor_points=train_floor_pts,
            unit=unit,
            dataset_name=dataset_name,
        )

        matrix = calibrator.H.tolist() if calibrator.H is not None else None
        train_mean_err = calibrator.diagnostics.training_reprojection_mean_error
        train_max_err = calibrator.diagnostics.training_reprojection_max_error

        # If no held-out points are available, return explicit insufficient-evidence status
        if len(held_out_pts) == 0:
            return HomographyValidationResult(
                training_point_count=len(training_pts),
                held_out_point_count=0,
                training_reprojection_mean_error=train_mean_err,
                training_reprojection_max_error=train_max_err,
                held_out_mean_error=None,
                held_out_max_error=None,
                held_out_results=[],
                evidence_status="INSUFFICIENT_EVIDENCE",
                evidence_note=(
                    "No held-out calibration points available. Independent out-of-sample "
                    "validation cannot be performed."
                ),
                unit=unit,
                dataset_name=dataset_name,
                homography_matrix=matrix,
            )

        # Evaluate each held-out point independently using unrounded vectorized transform_points
        held_out_img_arr = np.array(
            [[hp.image_x, hp.image_y] for hp in held_out_pts],
            dtype=np.float32,
        )
        unrounded_pred_pts = calibrator.transform_points(held_out_img_arr)

        held_out_results: List[HeldOutPointValidation] = []
        errors: List[float] = []

        for idx, hp in enumerate(held_out_pts):
            pred_x = float(unrounded_pred_pts[idx, 0])
            pred_y = float(unrounded_pred_pts[idx, 1])
            ref_x = float(hp.floor_x)
            ref_y = float(hp.floor_y)
            # Calculate Euclidean error directly from full-precision unrounded predictions
            err = float(math.hypot(pred_x - ref_x, pred_y - ref_y))

            errors.append(err)
            held_out_results.append(
                HeldOutPointValidation(
                    label=hp.label,
                    image_point=(float(hp.image_x), float(hp.image_y)),
                    reference_floor_point=(ref_x, ref_y),
                    predicted_floor_point=(pred_x, pred_y),
                    error=err,
                    unit=unit,
                )
            )

        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))

        return HomographyValidationResult(
            training_point_count=len(training_pts),
            held_out_point_count=len(held_out_pts),
            training_reprojection_mean_error=train_mean_err,
            training_reprojection_max_error=train_max_err,
            held_out_mean_error=mean_err,
            held_out_max_error=max_err,
            held_out_results=held_out_results,
            evidence_status="HELD_OUT_EVALUATED",
            evidence_note=(
                f"Evaluated against {len(held_out_pts)} independent held-out check point(s). "
                f"Note: Evaluates geometric consistency on supplied points only; does not prove "
                f"universal physical accuracy without extensive real-world ground truth."
            ),
            unit=unit,
            dataset_name=dataset_name,
            homography_matrix=matrix,
        )

    @classmethod
    def validate_sequence(
        cls,
        sequence: GroundTruthSequence,
        unit: Optional[str] = None,
    ) -> HomographyValidationResult:
        """
        Validate homography calibration embedded within a GroundTruthSequence.

        Args:
            sequence: GroundTruthSequence instance containing calibration_points.
            unit: Optional unit override (defaults to 'metres' or metadata unit).

        Returns:
            HomographyValidationResult.
        """
        chosen_unit = unit or (sequence.metadata.get("unit", "metres") if sequence.metadata else "metres")
        return cls.validate_calibration_points(
            calibration_points=sequence.calibration_points,
            unit=chosen_unit,
            dataset_name=sequence.sequence_id,
        )


def validate_homography(
    target: Union[GroundTruthSequence, List[GroundTruthCalibrationPoint]],
    unit: str = "metres",
    dataset_name: str = "default",
) -> HomographyValidationResult:
    """
    Convenience function to perform independent homography validation on a sequence or points list.
    """
    if isinstance(target, GroundTruthSequence):
        return HomographyValidator.validate_sequence(target, unit=unit)
    return HomographyValidator.validate_calibration_points(
        calibration_points=target,
        unit=unit,
        dataset_name=dataset_name,
    )
