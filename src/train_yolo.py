"""
Train YOLOv8 for lesion (and optionally pod) detection.

Example:
  python -m src.train_yolo --data_yaml data_det/data.yaml --model yolov8n.pt --epochs 100 --imgsz 640
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_yaml", type=str, required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model (yolov8n.pt, yolov8s.pt, ...)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=str, default="runs", help="Ultralytics output project dir")
    parser.add_argument("--name", type=str, default="detect/train", help="Run name")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data_yaml, epochs=args.epochs, imgsz=args.imgsz, project=args.project, name=args.name)


if __name__ == "__main__":
    main()
