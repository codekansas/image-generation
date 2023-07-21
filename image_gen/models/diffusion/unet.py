"""Defines a UNet model for diffusion image generation.

This was largely taken from ``here <https://github.com/tonyduan/diffusion)>``_.
"""

from dataclasses import dataclass

import ml.api as ml
from omegaconf import MISSING
from torch import Tensor

from image_gen.models.modules.unet import UNet


@dataclass
class UNetModelConfig(ml.BaseModelConfig):
    in_dim: int = ml.conf_field(MISSING, help="Number of input dimensions")
    embed_dim: int = ml.conf_field(MISSING, help="Embedding dimension")
    dim_scales: list[int] = ml.conf_field(MISSING, help="List of dimension scales")


@ml.register_model("unet-diffusion", UNetModelConfig)
class UNetModel(ml.BaseModel[UNetModelConfig]):
    def __init__(self, config: UNetModelConfig) -> None:
        super().__init__(config)

        self.model = UNet(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            dim_scales=config.dim_scales,
            use_time=True,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.model(x, t)
