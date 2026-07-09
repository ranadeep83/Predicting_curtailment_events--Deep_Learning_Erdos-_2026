# Final model interfaces

The three forecast-only model functions accept the finalized feature DataFrame
and an optional config dictionary:

```python
from Final_models import train_lstm, train_tcn, train_xgboost

result = train_tcn(data, {"seed": 1, "epochs": 2})
```

Each function returns the fitted `model`, merged `config`, validation-selected
`threshold`, `feature_columns`, training `history`, prediction frames for
`validation` and `test`, and a shared `metrics` table.

See `example.ipynb` for complete LSTM, TCN, and XGBoost runs.
