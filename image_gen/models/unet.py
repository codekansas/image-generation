"""Defines a template mode.

To use this, change the key from ``"template"`` to whatever your project
is called. Next, just override the ``forward`` model to whatever signature
your task expects, and you're good to go!
"""

import math
from dataclasses import dataclass
from typing import Sequence

import ml.api as ml
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import MISSING
from torch import Tensor, nn


def unsqueeze_to(tensor: Tensor, target_ndim: int) -> Tensor:
    assert tensor.ndim <= target_ndim
    while tensor.ndim < target_ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor


def unsqueeze_as(tensor: Tensor, target_tensor: Tensor) -> Tensor:
    assert tensor.ndim <= target_tensor.ndim
    while tensor.ndim < target_tensor.ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_length: int = 10000) -> None:
        super().__init__()

        self.register_buffer("embedding", self.make_embedding(dim, max_length))

    embedding: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding[x]

    @staticmethod
    def make_embedding(dim: int, max_length: int = 10000) -> Tensor:
        embedding = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(max_length / 2 / math.pi) / dim))
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        return embedding


class FFN(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()

        self.init_embed = nn.Linear(in_dim, embed_dim)
        self.time_embed = PositionalEmbedding(embed_dim)
        self.model = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, in_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.init_embed(x)
        t = self.time_embed(t)
        return self.model(x + t)


class BasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, time_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.mlp_time = nn.Sequential(
            nn.Linear(time_c, time_c),
            nn.ReLU(),
            nn.Linear(time_c, out_c),
        )

        self.shortcut = (
            nn.Identity()
            if in_c == out_c
            else nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_c),
            )
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out + unsqueeze_as(self.mlp_time(t), x))
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out + self.shortcut(x))
        return out


class SelfAttention2d(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.k_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.v_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.o_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)
        q = rearrange(q, "b (g c) h w -> (b g) c (h w)", g=self.num_heads)
        k = rearrange(k, "b (g c) h w -> (b g) c (h w)", g=self.num_heads)
        v = rearrange(v, "b (g c) h w -> (b g) c (h w)", g=self.num_heads)
        a = torch.einsum("b c s, b c t -> b s t", q, k) / self.dim**0.5
        a = self.dropout(torch.softmax(a, dim=-1))
        o = torch.einsum("b s t, b c t -> b c s", a, v)
        o = rearrange(o, "(b g) c (h w) -> b (g c) h w", g=self.num_heads, w=x.shape[-1])
        return x + self.o_conv(o)


class UNet(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, dim_scales: Sequence[int]) -> None:
        super().__init__()

        self.init_embed = nn.Conv2d(in_dim, embed_dim, 1)
        self.time_embed = PositionalEmbedding(embed_dim)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        all_dims = (embed_dim, *[embed_dim * s for s in dim_scales])

        for idx, (in_c, out_c) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            is_last = idx == len(all_dims) - 2
            self.down_blocks.extend(
                nn.ModuleList(
                    [
                        BasicBlock(in_c, in_c, embed_dim),
                        BasicBlock(in_c, in_c, embed_dim),
                        nn.Conv2d(in_c, out_c, 3, 2, 1) if not is_last else nn.Conv2d(in_c, out_c, 1),
                    ]
                )
            )

        for idx, (in_c, out_c, skip_c) in enumerate(zip(all_dims[::-1][:-1], all_dims[::-1][1:], all_dims[:-1][::-1])):
            is_last = idx == len(all_dims) - 2
            self.up_blocks.extend(
                nn.ModuleList(
                    [
                        BasicBlock(in_c + skip_c, in_c, embed_dim),
                        BasicBlock(in_c + skip_c, in_c, embed_dim),
                        nn.ConvTranspose2d(in_c, out_c, (2, 2), 2) if not is_last else nn.Conv2d(in_c, out_c, 1),
                    ]
                )
            )

        self.mid_blocks = nn.ModuleList(
            [
                BasicBlock(all_dims[-1], all_dims[-1], embed_dim),
                SelfAttention2d(all_dims[-1]),
                BasicBlock(all_dims[-1], all_dims[-1], embed_dim),
            ]
        )

        self.out_blocks = nn.ModuleList(
            [
                BasicBlock(embed_dim, embed_dim, embed_dim),
                nn.Conv2d(embed_dim, in_dim, 1, bias=True),
            ]
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.init_embed(x)
        t = self.time_embed(t)
        skip_conns = []
        residual = x.clone()

        for block in self.down_blocks:
            if isinstance(block, BasicBlock):
                x = block(x, t)
                skip_conns.append(x)
            else:
                x = block(x)
        for block in self.mid_blocks:
            if isinstance(block, BasicBlock):
                x = block(x, t)
            else:
                x = block(x)
        for block in self.up_blocks:
            if isinstance(block, BasicBlock):
                x = torch.cat((x, skip_conns.pop()), dim=1)
                x = block(x, t)
            else:
                x = block(x)

        x = x + residual
        for block in self.out_blocks:
            if isinstance(block, BasicBlock):
                x = block(x, t)
            else:
                x = block(x)
        return x


@dataclass
class UNetModelConfig(ml.BaseModelConfig):
    in_dim: int = ml.conf_field(MISSING, help="Number of input dimensions")
    embed_dim: int = ml.conf_field(MISSING, help="Embedding dimension")
    dim_scales: list[int] = ml.conf_field(MISSING, help="List of dimension scales")


@ml.register_model("unet", UNetModelConfig)
class UNetModel(ml.BaseModel[UNetModelConfig]):
    def __init__(self, config: UNetModelConfig) -> None:
        super().__init__(config)

        self.model = UNet(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            dim_scales=config.dim_scales,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.model(x, t)
