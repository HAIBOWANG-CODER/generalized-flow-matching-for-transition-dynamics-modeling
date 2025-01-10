import torch
import torch.nn as nn
from typing import Tuple

from src.networks.unet_base import UNetModelWrapper


class BendUNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs.pop("bend_model", None)
        self.time_bend = True
        self.mainnet = UNetModelWrapper(bend_model=True, *args, **kwargs)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([x0, x1], dim=1)
        return self.mainnet(t, x)
