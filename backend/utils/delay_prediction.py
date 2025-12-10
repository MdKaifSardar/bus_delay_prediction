"""Prediction helpers separated out from the loader.

This module provides `predict_with_model` and the internal
`_payload_to_dataframe` helper. Keeping prediction logic here makes the
loader focused on model I/O and allows services to import prediction logic
directly from `backend.utils.predict` (re-exported via `backend.models`).
"""
from typing import Any, List
import pandas as pd
import xgboost as xgb


def _payload_to_dataframe(payload: Any, model: Any = None) -> pd.DataFrame:
    """Convert incoming JSON payload to pandas.DataFrame and align columns.

    Accepts a dict (single record), a list of dicts (batch), or a dict-of-lists
    in which case pandas constructs a DataFrame directly.
    If the model exposes `feature_names_in_` the DataFrame will be reindexed to
    that column order (missing columns become NaN).
    """
    if isinstance(payload, dict):
        if all(not isinstance(v, (list, tuple)) for v in payload.values()):
            df = pd.DataFrame([payload])
        else:
            df = pd.DataFrame(payload)
    elif isinstance(payload, list):
        df = pd.DataFrame(payload)
    else:
        raise ValueError("Unsupported JSON format")

    if model is not None and hasattr(model, "feature_names_in_"):
        required = list(model.feature_names_in_)
        df = df.reindex(columns=required)

    # pd.to_numeric(..., errors="ignore") is deprecated and will raise in
    # a future pandas release. Convert columns individually and catch
    # conversion errors to preserve the previous behavior (leave non-numeric
    # columns unchanged).
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            # Leave the column as-is if it can't be converted to numeric
            continue

    return df


def predict_with_model(model: Any, payload: Any) -> List[float]:
    """Run prediction using the provided model and JSON payload.

    Works with sklearn-like estimators exposing ``predict`` and raw
    ``xgboost.Booster`` instances (builds a DMatrix).
    Returns a plain Python list of predictions.
    """
    df = _payload_to_dataframe(payload, model=model)

    # sklearn-like wrappers
    if hasattr(model, "predict") and "Booster" not in type(model).__name__:
        preds = model.predict(df)
    else:
        dmat = xgb.DMatrix(df)
        preds = model.predict(dmat)

    if hasattr(preds, "tolist"):
        return preds.tolist()
    return [float(preds)]
