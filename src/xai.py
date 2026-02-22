"""
Explainability utilities:
- Grad-CAM for the CNN branch
- Attention rollout heatmap for the ViT branch (hooking into timm ViT attention dropout)

Example:
  python -m src.xai --cls_ckpt runs_cls/best.pt --image_path sample.jpg --mode both --out_dir xai_out
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from .data import build_transforms
from .utils import get_device
from .models import HybridClassifier
from .ema import ModelEMA


def load_classifier(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[int, str], int]:
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

    if ckpt.get("ema_state") is not None:
        ema = ModelEMA(model, decay=0.999, device=device)
        ema.ema.load_state_dict(ckpt["ema_state"])
        ema.ema.eval()
        model = ema.ema  # type: ignore[assignment]

    img_size = int(ckpt.get("img_size", 224))
    return model, idx_to_class, img_size


def _prepare_input(image_path: str, img_size: int, device: torch.device):
    _, eval_tf = build_transforms(img_size)
    img_pil = Image.open(image_path).convert("RGB")
    x = eval_tf(img_pil).unsqueeze(0).to(device)
    return x


def _find_last_conv(module: torch.nn.Module) -> torch.nn.Module:
    last_conv = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in CNN backbone for Grad-CAM.")
    return last_conv


@torch.no_grad()
def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam


def overlay_heatmap(image_bgr: np.ndarray, heatmap_01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * heatmap_01)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return out


def gradcam_cnn(
    model: HybridClassifier,
    x: torch.Tensor,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """
    Compute Grad-CAM using the CNN backbone's last convolution.
    Returns a heatmap in [0,1] of shape (H,W) matching input spatial size.
    """
    device = x.device
    cnn = model.cnn

    last_conv = _find_last_conv(cnn)
    activations = None
    gradients = None

    def fwd_hook(_, __, output):
        nonlocal activations
        activations = output

    def bwd_hook(_, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = last_conv.register_forward_hook(fwd_hook)
    h2 = last_conv.register_full_backward_hook(bwd_hook)

    # Forward
    logits = model(x)  # [1,C]
    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1).item())

    # Backward w.r.t. chosen class score
    model.zero_grad(set_to_none=True)
    score = logits[:, target_class].sum()
    score.backward(retain_graph=True)

    h1.remove()
    h2.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Failed to capture activations/gradients for Grad-CAM.")

    # activations: [1, K, h, w], gradients: [1, K, h, w]
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1,K,1,1]
    cam = (weights * activations).sum(dim=1)  # [1,h,w]
    cam = F.relu(cam)
    cam = F.interpolate(cam.unsqueeze(1), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = _normalize_cam(cam)
    return cam


def vit_attention_rollout(
    model: HybridClassifier,
    x: torch.Tensor,
) -> np.ndarray:
    """
    Attention rollout heatmap for timm ViT backbone by hooking into each block's attn_drop.
    Returns a heatmap in [0,1] of shape (H,W) matching input spatial size.
    """
    vit = model.vit
    device = x.device

    attn_mats: List[torch.Tensor] = []

    hooks = []
    # Hook into attention dropout; its input is the attention matrix after softmax
    for blk in getattr(vit, "blocks", []):
        if hasattr(blk, "attn") and hasattr(blk.attn, "attn_drop"):
            def _hook(module, inputs, output):
                # inputs[0] is attention matrix: [B, heads, tokens, tokens]
                attn = inputs[0]
                attn_mats.append(attn.detach())

            hooks.append(blk.attn.attn_drop.register_forward_hook(_hook))

    # Forward to populate attn_mats
    _ = model(x)

    for h in hooks:
        h.remove()

    if len(attn_mats) == 0:
        raise RuntimeError("No attention matrices captured. Is your ViT backbone compatible with timm blocks?")

    # Rollout
    # attn_mats: list of [B, heads, T, T]
    attn = torch.stack(attn_mats, dim=0)  # [L,B,H,T,T]
    attn = attn.mean(dim=2)  # average heads -> [L,B,T,T]

    # Add residual (identity), normalize
    L, B, T, _ = attn.shape
    eye = torch.eye(T, device=device).unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
    attn = attn + eye
    attn = attn / attn.sum(dim=-1, keepdim=True)

    # Multiply matrices across layers
    joint = attn[0]
    for i in range(1, L):
        joint = attn[i].bmm(joint)

    # CLS token attention to patch tokens
    # joint: [B,T,T] -> take cls row 0, excluding cls col
    cls_attn = joint[:, 0, 1:]  # [B, T-1]
    cls_attn = cls_attn.reshape(B, int(np.sqrt(T - 1)), int(np.sqrt(T - 1)))  # [B, gh, gw]

    # Upsample to input resolution
    cls_attn = cls_attn.unsqueeze(1)  # [B,1,gh,gw]
    cls_attn = F.interpolate(cls_attn, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
    heat = cls_attn.squeeze().detach().cpu().numpy()
    heat = _normalize_cam(heat)
    return heat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_ckpt", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="both", choices=["gradcam", "vit_attn", "both"])
    parser.add_argument("--out_dir", type=str, default="xai_out")
    parser.add_argument("--target_class", type=int, default=-1, help="Optional fixed target class index. -1 = use model prediction.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()

    model, idx_to_class, img_size = load_classifier(args.cls_ckpt, device)
    x = _prepare_input(args.image_path, img_size, device)
    img_bgr = cv2.imread(args.image_path)
    if img_bgr is None:
        raise FileNotFoundError(args.image_path)

    # Predict label for display
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_class[pred_idx]
    target_class = None if args.target_class < 0 else args.target_class

    if args.mode in ("gradcam", "both"):
        cam = gradcam_cnn(model, x, target_class=target_class)
        out = overlay_heatmap(img_bgr, cam)
        cv2.putText(out, f"Grad-CAM (CNN) | pred={pred_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out_path = os.path.join(args.out_dir, "gradcam_cnn.jpg")
        cv2.imwrite(out_path, out)
        print("Saved:", out_path)

    if args.mode in ("vit_attn", "both"):
        heat = vit_attention_rollout(model, x)
        out = overlay_heatmap(img_bgr, heat)
        cv2.putText(out, f"Attention rollout (ViT) | pred={pred_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out_path = os.path.join(args.out_dir, "vit_attention_rollout.jpg")
        cv2.imwrite(out_path, out)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
