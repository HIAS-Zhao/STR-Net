#!/usr/bin/env python3
"""Purely pointwise spectral MLP gate for two-stage PGSS inference."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralMLPGate(nn.Module):
    """B7/B6/B2 -> 16 -> 8 -> 1 using only biased 1x1 convolutions."""

    def __init__(self, in_channels: int = 3, hidden1: int = 16, hidden2: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden1, hidden2, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden2, 1, kernel_size=1, bias=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def prepare_gate_input(inputs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Explicitly zero invalid pixels without adding the mask as a model channel."""
    if valid.ndim != 4 or valid.shape[1] != 1:
        raise ValueError(f"valid must have shape [B,1,H,W], got {tuple(valid.shape)}")
    return inputs * valid.to(dtype=inputs.dtype)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def estimate_macs_per_pixel(in_channels: int = 3, hidden1: int = 16, hidden2: int = 8) -> int:
    return in_channels * hidden1 + hidden1 * hidden2 + hidden2


if __name__ == "__main__":
    model = SpectralMLPGate()
    output = model(torch.randn(2, 3, 256, 256))
    print({"output_shape": tuple(output.shape), **count_parameters(model)})
    print({"macs_per_pixel": estimate_macs_per_pixel()})
