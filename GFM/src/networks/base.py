import torch.nn as nn
import torch
from typing import List, Optional
from torchcfm.models.unet.unet import UNetModel


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "lrelu": nn.LeakyReLU,
    "softplus": nn.Softplus,
    "silu": nn.SiLU,
    "swish": swish,
}
"""
SimpleDenseNet：创建一个简单的多层全连接神经网络，支持不同激活函数和批量归一化。
DeepSetModel：在 SimpleDenseNet 的基础上增加了输出均值计算，适合处理集合数据。
InverseStandardScaler：用于对标准化数据进行逆变换，恢复数据的原始尺度。
这些类可以组合使用，构建复杂的神经网络模型，并处理不同类型的数据任务。
"""

class SimpleDenseNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        target_size: int,
        activation: str,
        batch_norm: bool = False,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        dims = [input_size, *hidden_dims, target_size]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(ACTIVATION_MAP[activation]())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)
        #print(self.model)

    def forward(self, x):
        return self.model(x)


class DeepSetModel(SimpleDenseNet):
    def __init__(
        self,
        input_size: int,
        target_size: int,
        activation: str,
        batch_norm: bool = False,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__(input_size, target_size, activation, batch_norm, hidden_dims)
        self.final_activation = ACTIVATION_MAP[activation]()

    def forward(self, x):
        x = super().forward(x)
        x = self.final_activation(x)
        x = x.mean(dim=0, keepdim=True)
        return x


class InverseStandardScaler(nn.Module):
    def __init__(self, mean, scale):
        super(InverseStandardScaler, self).__init__()
        # Register mean and std as buffers
        self.register_buffer("mean", torch.Tensor(mean))
        self.register_buffer("scale", torch.Tensor(scale))

    def forward(self, x):
        # Apply the inverse transformation
        return x * self.scale + self.mean
