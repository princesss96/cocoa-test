"""
Severity estimation from YOLOv8 detection results.

Severity is computed as:
  severity = total_lesion_area / pod_area

Recommended: train YOLO with two classes: ["pod", "lesion"].
Fallback: if no pod box is found, pod_area = image_area.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass
class SeverityResult:
    severity: float
    lesion_area: float
    pod_area: float
    lesion_boxes_xyxy: List[List[float]]
    pod_box_xyxy: Optional[List[float]]


def _box_area_xyxy(xyxy: np.ndarray) -> np.ndarray:
    # xyxy: [...,4]
    w = np.clip(xyxy[..., 2] - xyxy[..., 0], 0, None)
    h = np.clip(xyxy[..., 3] - xyxy[..., 1], 0, None)
    return w * h


def compute_severity_from_ultralytics(
    yolo_result,
    class_names: List[str],
    image_shape_hw: Tuple[int, int],
    lesion_class_name: str = "lesion",
    pod_class_name: str = "pod",
) -> SeverityResult:
    """
    yolo_result: one element from ultralytics model.predict(...)
    class_names: list of class names in same order as YOLO training
    image_shape_hw: (H, W)
    """
    H, W = image_shape_hw
    image_area = float(H * W)

    boxes = yolo_result.boxes
    if boxes is None or len(boxes) == 0:
        return SeverityResult(
            severity=0.0,
            lesion_area=0.0,
            pod_area=image_area,
            lesion_boxes_xyxy=[],
            pod_box_xyxy=None,
        )

    xyxy = boxes.xyxy.cpu().numpy()  # [N,4]
    cls = boxes.cls.cpu().numpy().astype(int)  # [N]

    # Identify indices
    name_by_idx = {i: n for i, n in enumerate(class_names)}
    lesion_idx = [i for i, n in name_by_idx.items() if n == lesion_class_name]
    pod_idx = [i for i, n in name_by_idx.items() if n == pod_class_name]

    lesion_mask = np.isin(cls, lesion_idx) if len(lesion_idx) > 0 else np.zeros_like(cls, dtype=bool)
    pod_mask = np.isin(cls, pod_idx) if len(pod_idx) > 0 else np.zeros_like(cls, dtype=bool)

    lesion_boxes = xyxy[lesion_mask]
    pod_boxes = xyxy[pod_mask]

    lesion_area = float(_box_area_xyxy(lesion_boxes).sum()) if len(lesion_boxes) else 0.0

    pod_box = None
    if len(pod_boxes):
        # take largest pod area box
        areas = _box_area_xyxy(pod_boxes)
        j = int(np.argmax(areas))
        pod_box = pod_boxes[j].tolist()
        pod_area = float(areas[j])
        # avoid zero division
        pod_area = max(pod_area, 1.0)
    else:
        pod_area = max(image_area, 1.0)

    severity = float(np.clip(lesion_area / pod_area, 0.0, 1.0))

    return SeverityResult(
        severity=severity,
        lesion_area=lesion_area,
        pod_area=pod_area,
        lesion_boxes_xyxy=lesion_boxes.tolist() if len(lesion_boxes) else [],
        pod_box_xyxy=pod_box,
    )
