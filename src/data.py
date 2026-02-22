"""
Data loading utilities.

Supports:
- Folder datasets compatible with torchvision.datasets.ImageFolder:
  data_cls/train/<class_name>/*.jpg
  data_cls/val/<class_name>/*.jpg
  data_cls/test/<class_name>/*.jpg

- Optional CSV dataset (path,label) via CSVCocoaDataset without requiring pandas.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
form tochvision.transforms import InterpolationMode


@dataclass
class DataInfo:
    class_to_idx: dict
    idx_to_class: dict
    num_classes: int


def build_transforms(img_size: int = 224):
    # Stronger but still "realistic" for cocoa pod photos
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),

        # Keep vertical flip low (pods can be oriented in many ways, but not always)
        transforms.RandomVerticalFlip(p=0.1),

        # Small geometry changes (simulate camera tilt/shift)
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=15,
                translate=(0.05, 0.05),
                scale=(0.90, 1.10),
                shear=5
            )
        ], p=0.7),

        # Lighting variation (field conditions)
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),

        # Slight blur for motion / focus differences (optional)
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),

        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        # Helps generalization (optional)
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return train_tf, eval_tf

class CSVCocoaDataset(Dataset):
    """A minimal dataset that reads images from filepaths listed in a CSV (columns: path,label)."""
    def __init__(self, csv_path: str, transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, str]] = []

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError("CSV must contain columns: 'path' and 'label'")
            for row in reader:
                self.samples.append((row["path"], row["label"]))

        labels = sorted(list({label for _, label in self.samples}))
        self.class_to_idx = {c: i for i, c in enumerate(labels)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        y = self.class_to_idx[label]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def create_imagefolder_loaders(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], DataInfo]:
    train_tf, eval_tf = build_transforms(img_size)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)

    test_loader: Optional[DataLoader] = None
    if os.path.isdir(test_dir):
        test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    info = DataInfo(class_to_idx=class_to_idx, idx_to_class=idx_to_class, num_classes=len(class_to_idx))

    return train_loader, val_loader, test_loader, info
