from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from Final_models.common import evaluate_predictions


DEFAULT_CONFIG = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_child_weight": 1,
    "colsample_bytree": 0.3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "n_jobs": 4,
    "seed": 42,
    "valid_start": "2025-07-01",
    "test_start": "2025-12-01",
    "tree_method": "hist",
}

CALENDAR_COLUMNS = [
    "Hour",
    "Day_of_week",
    "Month",
    "Is_weekend",
    "Hour_sin",
    "Hour_cos",
]


def train_xgboost(data, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    frame = data.copy()
    if "timestamp_utc" in frame.columns:
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        frame = frame.set_index("timestamp_utc")
    else:
        frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()

    observed_columns = [column for column in frame.columns if column.startswith("observed_")]
    forecast_columns = [column for column in frame.columns if column.startswith("forecast_")]
    forecast_by_name = {
        column.removeprefix("forecast_"): column
        for column in forecast_columns
    }
    weather = frame[list(forecast_by_name.values())].copy()
    weather.columns = list(forecast_by_name)
    weather_features = pd.concat([
        weather,
        weather.shift(1).add_suffix("_lag_1"),
        weather.shift(24).add_suffix("_lag_24"),
    ], axis=1)

    excluded = set(
        CALENDAR_COLUMNS
        + ["Negative_price", "Price", "Total_gen", "Residuals"]
        + observed_columns
        + forecast_columns
    )
    smard_columns = [column for column in frame.columns if column not in excluded]
    smard_features = pd.concat([
        frame[smard_columns].shift(24).add_suffix("_lag_24"),
        frame[smard_columns].shift(48).add_suffix("_lag_48"),
    ], axis=1)
    price_features = pd.concat([
        frame[["Price"]].shift(24).add_suffix("_lag_24"),
        frame[["Price"]].shift(48).add_suffix("_lag_48"),
    ], axis=1)

    features = pd.concat(
        [frame[CALENDAR_COLUMNS], weather_features, smard_features, price_features],
        axis=1,
    )
    model_data = features.join(frame["Negative_price"]).dropna()
    feature_columns = list(features.columns)
    valid_start = pd.Timestamp(config["valid_start"], tz="UTC")
    test_start = pd.Timestamp(config["test_start"], tz="UTC")
    masks = {
        "train": model_data.index < valid_start,
        "validation": (model_data.index >= valid_start) & (model_data.index < test_start),
        "test": model_data.index >= test_start,
    }
    dataset = {
        split: {
            "X": model_data.loc[mask, feature_columns],
            "y": model_data.loc[mask, "Negative_price"].to_numpy(dtype=np.float32),
            "times": model_data.index[mask],
        }
        for split, mask in masks.items()
    }

    y_train = dataset["train"]["y"]
    positives = y_train.sum()
    negatives = len(y_train) - positives
    model_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": negatives / max(positives, 1),
        "random_state": int(config["seed"]),
        **{
            key: config[key]
            for key in [
                "n_estimators",
                "max_depth",
                "min_child_weight",
                "colsample_bytree",
                "learning_rate",
                "subsample",
                "n_jobs",
                "tree_method",
            ]
        },
    }
    if config.get("device"):
        model_params["device"] = config["device"]
    model = XGBClassifier(**model_params)
    model.fit(dataset["train"]["X"], y_train)

    trained = {
        "validation_true": dataset["validation"]["y"],
        "validation_prob": model.predict_proba(dataset["validation"]["X"])[:, 1],
        "test_true": dataset["test"]["y"],
        "test_prob": model.predict_proba(dataset["test"]["X"])[:, 1],
    }
    threshold, predictions, metrics = evaluate_predictions(dataset, trained)
    return {
        "model": model,
        "config": config,
        "threshold": threshold,
        "feature_columns": feature_columns,
        "sequence_shape": (len(feature_columns),),
        "validation": predictions["validation"],
        "test": predictions["test"],
        "metrics": metrics,
    }
