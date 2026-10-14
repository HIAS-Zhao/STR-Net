#!/usr/bin/env python3
"""PGSS-Tiny V1 model for full-resolution active fire segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralStem(nn.Module):
    def __init__(self, in_channels: int, width: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class PGSSTinyV1(nn.Module):
    """Lightweight full-resolution spectral-spatial active fire network."""

    def __init__(self, in_channels: int = 3, width: int = 16, head_width: int = 8) -> None:
        super().__init__()
        self.stem = SpectralStem(in_channels, width)
        self.spatial_d1 = DepthwiseResidualBlock(width, dilation=1)
        self.spatial_d2 = DepthwiseResidualBlock(width, dilation=2)
        self.head = nn.Sequential(
            nn.Conv2d(width, head_width, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_width, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.spatial_d1(x)
        x = self.spatial_d2(x)
        return self.head(x)


def forward_boundary_aligned_tile(
    model: PGSSTinyV1,
    tile_inputs: torch.Tensor,
    image_valid: torch.Tensor,
) -> torch.Tensor:
    """Run a tile with the same image-boundary semantics as dense inference.

    The stem is pointwise, but BatchNorm affine terms can turn a raw
    zero-padded pixel into a nonzero feature. Dense inference pads only the
    spatial convolutions, so image-outside features must be zeroed after the
    stem and before the spatial blocks.
    """
    if tile_inputs.ndim != 4 or image_valid.ndim != 4:
        raise ValueError("tile_inputs and image_valid must be 4D tensors")
    if image_valid.shape[0] != tile_inputs.shape[0] or image_valid.shape[1] != 1:
        raise ValueError("image_valid batch/channel shape mismatch")
    if image_valid.shape[-2:] != tile_inputs.shape[-2:]:
        raise ValueError("image_valid and tile_inputs spatial shape mismatch")
    # Keep the inference utility usable with the lightweight pointwise mock
    # models used by the sparse-path unit tests.
    if not all(hasattr(model, name) for name in ("stem", "spatial_d1", "spatial_d2", "head")):
        return model(tile_inputs)
    valid = image_valid.bool()
    features = model.stem(tile_inputs)
    features = features.masked_fill(~valid, 0.0)
    features = model.spatial_d1(features)
    # BatchNorm inside a residual block can make an all-zero border
    # nonzero; clear it before the next block so its convolution sees the
    # same zero padding as the full-image path.
    features = features.masked_fill(~valid, 0.0)
    features = model.spatial_d2(features)
    features = features.masked_fill(~valid, 0.0)
    return model.head(features)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def estimate_macs_per_pixel(in_channels: int = 3, width: int = 16, head_width: int = 8) -> int:
    stem = in_channels * width + width * width
    depthwise_blocks = 2 * (9 * width + width * width)
    head = width * head_width + head_width
    return stem + depthwise_blocks + head


if __name__ == "__main__":
    model = PGSSTinyV1(3)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(tuple(y.shape))
    print(count_parameters(model))
    print({"macs_per_pixel": estimate_macs_per_pixel(3)})
