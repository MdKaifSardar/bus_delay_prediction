"""Prediction helpers used by the API and services.

This mirrors the logic from `delay_prediction.py` but provides a stable
module name `backend.utils.predict` which the `backend.models` package
re-exports. Having a predictable module name avoids import-time surprises
when the code is restructured.
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

    # Convert columns to numeric where possible. We attempt per-column
    # conversion and leave the column unchanged if conversion fails. This
    # preserves previous 'errors="ignore"' behavior without using the
    # deprecated API.
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
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
