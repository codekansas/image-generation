"""Defines a simple GAN model."""

from dataclasses import dataclass

import ml.api as ml
import torch
from torch import Tensor, nn


@dataclass
class GeneratorModelConfig(ml.BaseModelConfig):
    in_dim: int = ml.conf_field(1, help="Number of input dimensions")
    embed_dim: int = ml.conf_field(128, help="Embedding dimension")
    dim_scales: list[int] = ml.conf_field([1, 2, 4, 8], help="List of dimension scales")


@ml.register_model("generator", GeneratorModelConfig)
class GeneratorModel(ml.BaseModel[GeneratorModelConfig]):
    def __init__(self, config: GeneratorModelConfig) -> None:
        super().__init__(config)

        self.model = ml.UNet(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            dim_scales=config.dim_scales,
            use_time=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.model(x))


@dataclass
class DiscriminatorModelConfig(ml.BaseModelConfig):
    in_dim: int = ml.conf_field(1, help="Number of input dimensions")
    embed_dim: int = ml.conf_field(128, help="Embedding dimension")
    num_layers: int = ml.conf_field(2, help="Number of layers")


def cbr(in_c: int, out_c: int, kernel_size: int, stride: int, padding: int = 0) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2),
    )


@ml.register_model("discriminator", DiscriminatorModelConfig)
class DiscriminatorModel(ml.BaseModel[DiscriminatorModelConfig]):
    def __init__(self, config: DiscriminatorModelConfig) -> None:
        super().__init__(config)

        self.model = nn.Sequential(
            cbr(config.in_dim, config.embed_dim, 2, 2),
            cbr(config.embed_dim, config.embed_dim, 2, 2),
            cbr(config.embed_dim, config.embed_dim, 3, 1),
            nn.Conv2d(config.embed_dim, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
