"""
Exponential Moving Average (EMA) of model parameters.

Useful for improving generalization stability.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[torch.device] = None):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        self.device = device
        if device is not None:
            self.ema.to(device=device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.ema.state_dict()
