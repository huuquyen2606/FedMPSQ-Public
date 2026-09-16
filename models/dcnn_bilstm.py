"""DCNN-BiLSTM for tabular network-flow classification."""

from __future__ import annotations

import torch
from torch import nn


SUPPORTED_NORMS = ("batch", "group", "layer")


def build_norm(kind: str, channels: int) -> nn.Module:
    """Return the configured normalisation layer for a Conv1d output.

    ``batch`` keeps running statistics, which federated averaging has to
    aggregate; under a strongly non-IID split those statistics differ so much
    between clients that averaging them corrupts the global model even while
    every client trains normally. ``group`` and ``layer`` normalise inside each
    sample instead, so they carry no buffers and nothing can be aggregated
    wrongly. ``layer`` is GroupNorm with a single group, which for a
    ``(N, C, L)`` activation is exactly layer normalisation.
    """
    if kind == "batch":
        return nn.BatchNorm1d(channels)
    if kind == "layer":
        return nn.GroupNorm(1, channels)
    if kind == "group":
        groups = next(g for g in range(min(32, channels), 0, -1) if channels % g == 0)
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"model.norm must be one of {SUPPORTED_NORMS}; got {kind!r}")


class ConvBlock(nn.Module):
    """One Conv1d block for tabular feature sequences."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
        use_pool: bool,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            build_norm(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ]
        if use_pool:
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DCNNBiLSTM(nn.Module):
    """Deep CNN plus BiLSTM for CICIoT2023 flow features.

    The input is a flat feature vector. It is interpreted as a one-dimensional
    feature sequence, processed by stacked Conv1d blocks, then passed through a
    bidirectional LSTM before classification.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        conv_channels: tuple[int, ...] = (64, 128, 128),
        kernel_size: int = 3,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        if norm not in SUPPORTED_NORMS:
            raise ValueError(f"model.norm must be one of {SUPPORTED_NORMS}; got {norm!r}")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        if not conv_channels:
            raise ValueError("conv_channels must not be empty")

        blocks: list[nn.Module] = []
        in_channels = 1
        for idx, out_channels in enumerate(conv_channels):
            blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    use_pool=idx < len(conv_channels) - 1,
                    norm=norm,
                )
            )
            in_channels = out_channels
        self.convolution = nn.Sequential(*blocks)
        self.bilstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected input shape [batch, features], got {tuple(x.shape)}")
        z = x.float().unsqueeze(1)
        z = self.convolution(z)
        z = z.transpose(1, 2)
        sequence, _ = self.bilstm(z)
        pooled = sequence.mean(dim=1)
        return self.classifier(pooled)

