"""
Utility helpers for training/evaluation.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Make runs more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: deterministic can reduce performance; enable if you really need strict reproducibility.
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class AverageMeter:
    """Track running averages (loss, accuracy, etc.)."""
    name: str
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def save_checkpoint(state: Dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(state, out_path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict:
    return torch.load(path, map_location=map_location)


def softmax_to_pred(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits -> predicted class indices."""
    return torch.argmax(logits, dim=1)


def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute common metrics: accuracy, macro-f1, precision, recall, Cohen's kappa."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm  # type: ignore[assignment]
    return metrics
