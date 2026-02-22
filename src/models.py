"""
Hybrid CNN–ViT models (CNN-only, ViT-only, concat, attention fusion).

Backbones are created with timm so you can swap in other models easily.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

import timm


@dataclass
class FusionOutput:
    logits: torch.Tensor
    alpha_beta: Optional[torch.Tensor] = None  # shape [B, 2] if attention fusion


class HybridClassifier(nn.Module):
    """
    Variants:
      - "cnn": CNN backbone only (ResNet50 by default)
      - "vit": ViT backbone only
      - "concat": concatenate projected CNN and ViT features
      - "attn": attention/gating fusion: F = α*F_cnn + β*F_vit (softmax weights)
    """
    def __init__(
        self,
        num_classes: int = 3,
        variant: str = "attn",
        cnn_name: str = "resnet50",
        vit_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        proj_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        variant = variant.lower().strip()
        if variant not in {"cnn", "vit", "concat", "attn"}:
            raise ValueError(f"Unknown variant '{variant}'. Use one of: cnn, vit, concat, attn")
        self.variant = variant

        # Feature extractors (no classification head)
        self.cnn = timm.create_model(cnn_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)

        cnn_dim = getattr(self.cnn, "num_features", None)
        vit_dim = getattr(self.vit, "num_features", None)
        if cnn_dim is None or vit_dim is None:
            raise RuntimeError("Backbones must expose .num_features (timm models usually do).")

        # Project both to a common embedding size
        self.cnn_proj = nn.Sequential(
            nn.LayerNorm(cnn_dim),
            nn.Linear(cnn_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.vit_proj = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if self.variant == "concat":
            head_in = 2 * proj_dim
        else:
            head_in = proj_dim

        # Attention/gating fusion (only used for "attn")
        self.gate = nn.Sequential(
            nn.Linear(2 * proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 2),
        )

        self.classifier = nn.Linear(head_in, num_classes)

    def freeze_backbones(self) -> None:
        for p in self.cnn.parameters():
            p.requires_grad_(False)
        for p in self.vit.parameters():
            p.requires_grad_(False)

    def unfreeze_backbones(self) -> None:
        for p in self.cnn.parameters():
            p.requires_grad_(True)
        for p in self.vit.parameters():
            p.requires_grad_(True)

    def forward(self, x: torch.Tensor, return_fusion: bool = False) -> FusionOutput | torch.Tensor:
        # Extract features
        cnn_feat = self.cnn(x)  # [B, cnn_dim]
        vit_feat = self.vit(x)  # [B, vit_dim]
        cnn_e = self.cnn_proj(cnn_feat)  # [B, proj_dim]
        vit_e = self.vit_proj(vit_feat)  # [B, proj_dim]

        alpha_beta: Optional[torch.Tensor] = None

        if self.variant == "cnn":
            fused = cnn_e
        elif self.variant == "vit":
            fused = vit_e
        elif self.variant == "concat":
            fused = torch.cat([cnn_e, vit_e], dim=1)
        else:  # "attn"
            h = torch.cat([cnn_e, vit_e], dim=1)
            w = F.softmax(self.gate(h), dim=1)  # [B,2]
            alpha = w[:, 0:1]
            beta = w[:, 1:2]
            fused = alpha * cnn_e + beta * vit_e
            alpha_beta = w

        logits = self.classifier(fused)

        if return_fusion:
            return FusionOutput(logits=logits, alpha_beta=alpha_beta)
        return logits
