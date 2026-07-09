from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


CALENDAR_COLUMNS = [
    "Hour",
    "Day_of_week",
    "Month",
    "Is_weekend",
    "Hour_sin",
    "Hour_cos",
    "Day_of_week_sin",
    "Day_of_week_cos",
    "Month_sin",
    "Month_cos",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_sequence_data(data, config):
    frame = data.copy()
    if "timestamp_utc" in frame.columns:
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        frame = frame.set_index("timestamp_utc")
    else:
        frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()

    frame["Day_of_week_sin"] = np.sin(2 * np.pi * frame["Day_of_week"] / 7)
    frame["Day_of_week_cos"] = np.cos(2 * np.pi * frame["Day_of_week"] / 7)
    frame["Month_sin"] = np.sin(2 * np.pi * frame["Month"] / 12)
    frame["Month_cos"] = np.cos(2 * np.pi * frame["Month"] / 12)

    observed_columns = [column for column in frame.columns if column.startswith("observed_")]
    forecast_columns = [column for column in frame.columns if column.startswith("forecast_")]
    observed_by_name = {column.removeprefix("observed_"): column for column in observed_columns}
    forecast_by_name = {column.removeprefix("forecast_"): column for column in forecast_columns}
    weather_columns = list(observed_by_name)
    if set(weather_columns) != set(forecast_by_name):
        raise ValueError("Observed and forecast weather columns do not match")

    excluded = set(CALENDAR_COLUMNS + ["Negative_price", *observed_columns, *forecast_columns])
    smard_columns = [column for column in frame.columns if column not in excluded]
    feature_columns = CALENDAR_COLUMNS + smard_columns + weather_columns

    observed_weather = frame[[observed_by_name[name] for name in weather_columns]].copy()
    observed_weather.columns = weather_columns
    forecast_weather = frame[[forecast_by_name[name] for name in weather_columns]].copy()
    forecast_weather.columns = weather_columns

    past = pd.concat(
        [frame[CALENDAR_COLUMNS], frame[smard_columns], observed_weather],
        axis=1,
    ).reindex(columns=feature_columns)
    future = (
        frame[CALENDAR_COLUMNS]
        .join(forecast_weather)
        .reindex(columns=feature_columns)
    )

    valid_start = pd.Timestamp(config["valid_start"], tz="UTC")
    test_start = pd.Timestamp(config["test_start"], tz="UTC")
    train_rows = past[past.index < valid_start]
    feature_mean = train_rows.mean()
    feature_std = train_rows.std().replace(0, 1).fillna(1)
    past = ((past - feature_mean) / feature_std).fillna(0.0)
    future = ((future - feature_mean) / feature_std).fillna(0.0)

    lookback = int(config["lookback"])
    past_values = past.to_numpy(dtype=np.float32)
    future_values = future.to_numpy(dtype=np.float32)
    target = frame["Negative_price"].astype(np.float32)
    start_time = frame.index.min()
    sequences, labels, times = [], [], []

    for timestamp in target.index:
        past_start = timestamp - pd.Timedelta(hours=24 + lookback - 1)
        past_end = timestamp - pd.Timedelta(hours=24)
        start_position = int((past_start - start_time).total_seconds() // 3600)
        end_position = int((past_end - start_time).total_seconds() // 3600) + 1
        future_start = timestamp - pd.Timedelta(hours=23)
        future_start_position = int((future_start - start_time).total_seconds() // 3600)
        future_end_position = int((timestamp - start_time).total_seconds() // 3600) + 1

        if start_position < 0 or future_start_position < 0:
            continue
        past_sequence = past_values[start_position:end_position]
        future_sequence = future_values[future_start_position:future_end_position]
        if len(past_sequence) != lookback or len(future_sequence) != 24:
            continue

        sequences.append(np.vstack([past_sequence, future_sequence]))
        labels.append(target.loc[timestamp])
        times.append(timestamp)

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    times = pd.DatetimeIndex(times)
    masks = {
        "train": times < valid_start,
        "validation": (times >= valid_start) & (times < test_start),
        "test": times >= test_start,
    }
    dataset = {
        split: {
            "X": X[mask],
            "y": y[mask],
            "times": times[mask],
        }
        for split, mask in masks.items()
    }
    dataset["feature_columns"] = feature_columns
    dataset["feature_mean"] = feature_mean
    dataset["feature_std"] = feature_std
    return dataset


def train_torch_classifier(model, dataset, config):
    device = next(model.parameters()).device
    batch_size = int(config["batch_size"])

    def loader(split, shuffle):
        return DataLoader(
            TensorDataset(
                torch.tensor(dataset[split]["X"], dtype=torch.float32),
                torch.tensor(dataset[split]["y"], dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    train_loader = loader("train", True)
    validation_loader = loader("validation", False)
    test_loader = loader("test", False)

    y_train = dataset["train"]["y"]
    positives = y_train.sum()
    negatives = len(y_train) - positives
    pos_weight = torch.tensor(
        [negatives / max(positives, 1)],
        dtype=torch.float32,
        device=device,
    )

    def focal_loss(logits, targets):
        targets = targets.to(device=device, dtype=logits.dtype)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        p_t = targets * probabilities + (1 - targets) * (1 - probabilities)
        loss = (1 - p_t).pow(float(config["focal_gamma"])) * loss
        alpha = torch.as_tensor(
            float(config["focal_alpha"]),
            device=device,
            dtype=logits.dtype,
        )
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        return (alpha_t * loss).mean()

    def predict(data_loader):
        model.eval()
        probabilities, targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                logits = model(X_batch.to(device))
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
                targets.append(y_batch.numpy())
        return np.concatenate(probabilities), np.concatenate(targets)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    best_state = None
    best_pr_auc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(int(config["epochs"])):
        model.train()
        losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch.to(device))
            loss = focal_loss(logits, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        validation_prob, validation_true = predict(validation_loader)
        validation_pr_auc = average_precision_score(validation_true, validation_prob)

        if validation_pr_auc > best_pr_auc:
            best_pr_auc = validation_pr_auc
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(config["patience"]):
            break

    model.load_state_dict(best_state)
    validation_prob, validation_true = predict(validation_loader)
    test_prob, test_true = predict(test_loader)
    return {
        "model": model,
        "best_epoch": best_epoch,
        "validation_true": validation_true,
        "validation_prob": validation_prob,
        "test_true": test_true,
        "test_prob": test_prob,
    }


def evaluate_predictions(dataset, trained):
    y_valid = trained["validation_true"]
    valid_prob = trained["validation_prob"]
    precision, recall, thresholds = precision_recall_curve(y_valid, valid_prob)
    f1_values = 2 * precision * recall / (precision + recall + 1e-10)
    threshold = float(thresholds[int(np.argmax(f1_values[:-1]))])

    prediction_frames = {}
    metric_rows = []
    for split, true_key, prob_key in [
        ("validation", "validation_true", "validation_prob"),
        ("test", "test_true", "test_prob"),
    ]:
        y_true = trained[true_key]
        probabilities = trained[prob_key]
        predictions = (probabilities >= threshold).astype(int)
        prediction_frames[split] = pd.DataFrame({
            "timestamp": dataset[split]["times"],
            "y_true": y_true.astype(int),
            "y_prob": probabilities,
            "y_pred": predictions,
        })
        metric_rows.append({
            "split": split,
            "PR_AUC": average_precision_score(y_true, probabilities),
            "ROC_AUC": roc_auc_score(y_true, probabilities),
            "F1": f1_score(y_true, predictions, zero_division=0),
            "precision": precision_score(y_true, predictions, zero_division=0),
            "recall": recall_score(y_true, predictions, zero_division=0),
            "threshold": threshold,
        })

    return threshold, prediction_frames, pd.DataFrame(metric_rows).set_index("split")
