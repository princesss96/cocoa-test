"""
End-to-end inference:
- Load trained classifier checkpoint (Hybrid CNN–ViT)
- Predict class probabilities
- Run YOLOv8 detection for lesions (and optionally pod)
- Compute severity = lesion_area / pod_area
- Save annotated image

Example:
  python -m src.infer_cls_and_severity --cls_ckpt runs_cls/best.pt --yolo_ckpt runs/detect/train/weights/best.pt --image_path sample.jpg
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from ultralytics import YOLO

from .models import HybridClassifier
from .data import build_transforms
from .utils import get_device
from .ema import ModelEMA
from .severity import compute_severity_from_ultralytics


def load_classifier(ckpt_path: str, device: torch.device) -> Tuple[HybridClassifier, Dict[int, str], int]:
    ckpt = torch.load(ckpt_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    model = HybridClassifier(
        num_classes=num_classes,
        variant=ckpt.get("variant", "attn"),
        proj_dim=int(ckpt.get("proj_dim", 512)),
        dropout=float(ckpt.get("dropout", 0.2)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Prefer EMA weights if present
    if ckpt.get("ema_state") is not None:
        ema = ModelEMA(model, decay=0.999, device=device)
        ema.ema.load_state_dict(ckpt["ema_state"])
        ema.ema.eval()
        model = ema.ema  # type: ignore[assignment]

    img_size = int(ckpt.get("img_size", 224))
    return model, idx_to_class, img_size


@torch.no_grad()
def predict_class(model: torch.nn.Module, image_path: str, img_size: int, device: torch.device):
    _, eval_tf = build_transforms(img_size)
    img = Image.open(image_path).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred = int(np.argmax(probs))
    return pred, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_ckpt", type=str, required=True, help="Path to classifier checkpoint (best.pt).")
    parser.add_argument("--yolo_ckpt", type=str, required=True, help="Path to YOLOv8 weights (best.pt).")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--out_path", type=str, default="prediction.jpg", help="Where to save annotated output image.")
    args = parser.parse_args()

    device = get_device()

    # 1) Classification
    cls_model, idx_to_class, img_size = load_classifier(args.cls_ckpt, device)
    pred_idx, probs = predict_class(cls_model, args.image_path, img_size, device)
    pred_label = idx_to_class[pred_idx]

    # 2) YOLO detection + severity
    yolo = YOLO(args.yolo_ckpt)
    results = yolo.predict(source=args.image_path, conf=args.conf, verbose=False)
    res0 = results[0]
    # class names
    names = res0.names  # dict {idx: name}
    class_names = [names[i] for i in range(len(names))]

    img_bgr = cv2.imread(args.image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image_path}")
    H, W = img_bgr.shape[:2]

    sev = compute_severity_from_ultralytics(res0, class_names=class_names, image_shape_hw=(H, W))

    # Draw boxes (pod in green, lesion in red) + overlay text
    boxes = res0.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, confs):
            name = class_names[int(c)] if int(c) < len(class_names) else str(int(c))
            color = (0, 255, 0) if name == "pod" else (0, 0, 255)
            cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img_bgr, f"{name}:{cf:.2f}", (int(x1), max(10, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(img_bgr, f"Class: {pred_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img_bgr, f"Severity: {sev.severity:.3f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    cv2.imwrite(args.out_path, img_bgr)

    print("Prediction saved to:", args.out_path)
    print("Predicted class:", pred_label)
    print("Class probabilities:", {idx_to_class[i]: float(probs[i]) for i in range(len(probs))})
    print("Severity:", sev.severity, "| lesion_area:", sev.lesion_area, "| pod_area:", sev.pod_area)


if __name__ == "__main__":
    main()
