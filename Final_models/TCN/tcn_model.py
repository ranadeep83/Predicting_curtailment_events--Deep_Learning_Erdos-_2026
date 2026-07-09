from __future__ import annotations

import torch
import torch.nn as nn

from Final_models.common import (
    evaluate_predictions,
    get_device,
    prepare_sequence_data,
    set_seed,
    train_torch_classifier,
)


DEFAULT_CONFIG = {
    "lookback": 48,
    "channels": (64, 64, 64, 64),
    "kernel_size": 3,
    "dropout": 0.3,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 10,
    "patience": 5,
    "focal_alpha": 0.75,
    "focal_gamma": 2.0,
    "seed": 1,
    "valid_start": "2025-07-01",
    "test_start": "2025-12-01",
}


class Chomp1d(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x if self.size == 0 else x[:, :, :-self.size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.layers(x) + self.shortcut(x))


class CurtailmentTCN(nn.Module):
    def __init__(self, input_dim, channels, kernel_size, dropout):
        super().__init__()
        blocks = []
        in_channels = input_dim
        for index, out_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=2 ** index,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.tcn = nn.Sequential(*blocks)
        self.fc = nn.Linear(channels[-1], 1)

    def forward(self, x):
        output = self.tcn(x.transpose(1, 2))
        return self.fc(output[:, :, -1]).squeeze(-1)


def train_tcn(data, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    config["channels"] = tuple(config["channels"])
    dataset = prepare_sequence_data(data, config)
    set_seed(int(config["seed"]))
    model = CurtailmentTCN(
        input_dim=dataset["train"]["X"].shape[2],
        channels=config["channels"],
        kernel_size=int(config["kernel_size"]),
        dropout=float(config["dropout"]),
    ).to(get_device())
    trained = train_torch_classifier(model, dataset, config)
    threshold, predictions, metrics = evaluate_predictions(dataset, trained)
    return {
        "model": trained["model"],
        "config": config,
        "threshold": threshold,
        "feature_columns": dataset["feature_columns"],
        "sequence_shape": dataset["train"]["X"].shape[1:],
        "validation": predictions["validation"],
        "test": predictions["test"],
        "metrics": metrics,
    }
