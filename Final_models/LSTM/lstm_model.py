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
    "lookback": 24,
    "hidden_dim": 64,
    "num_layers": 1,
    "dropout": 0.0,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 10,
    "patience": 5,
    "focal_alpha": 0.5,
    "focal_gamma": 2.0,
    "seed": 1,
    "valid_start": "2025-07-01",
    "test_start": "2025-12-01",
}


class CurtailmentLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(self.output_dropout(output[:, -1, :])).squeeze(-1)


def train_lstm(data, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    dataset = prepare_sequence_data(data, config)
    set_seed(int(config["seed"]))
    model = CurtailmentLSTM(
        input_dim=dataset["train"]["X"].shape[2],
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
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
