"""Defines a model for image generation with diffusion.

This was largely taken from ``here <https://github.com/tonyduan/diffusion)>``_.
"""

from dataclasses import dataclass

import ml.api as ml
import torch
from torch import Tensor


@dataclass
class DiffusionModelConfig(ml.BaseModelConfig):
    in_dim: int = ml.conf_field(1, help="Number of input dimensions")
    embed_dim: int = ml.conf_field(128, help="Embedding dimension")
    dim_scales: list[int] = ml.conf_field([1, 2, 4, 8], help="List of dimension scales")


@ml.register_model("diffusion", DiffusionModelConfig)
class DiffusionModel(ml.BaseModel[DiffusionModelConfig]):
    def __init__(self, config: DiffusionModelConfig) -> None:
        super().__init__(config)

        self.model = ml.UNet(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            dim_scales=config.dim_scales,
            use_time=True,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.tanh(self.model(x, t))
