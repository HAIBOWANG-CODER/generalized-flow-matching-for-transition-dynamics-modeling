import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import math

from src.networks.unet_base import UNetModelWrapper


class VelocityUNet(nn.Module):
    def __init__(self, image_size: Tuple = (3, 16, 16)):
        super().__init__()
        self.image_size = image_size
        self.mainnet = UNetModelWrapper(
            image_size=self.image_size[-1],
            in_channels=self.image_size[0],
            out_channels=self.image_size[0],
        )

    def forward(self, x, t) -> torch.Tensor:
        print("VelocityUNet forward")
        print(x.shape)
        print(t.shape)
        return self.mainnet(x, t)
