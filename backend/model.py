"""Model utilities: loading and prediction helpers.

This module centralizes model loading and DataFrame preparation so the web
routes remain small and testable.
"""
from typing import Any, List
import os
import pickle

import pandas as pd
import xgboost as xgb


def default_model_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "best_xgb_model.pkl"))


def load_model(path: str = None) -> Any:
    """Load and return the pickled model from disk.

    Raises FileNotFoundError if the file is missing.
    """
    if path is None:
        path = default_model_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def _payload_to_dataframe(payload: Any, model: Any = None) -> pd.DataFrame:
    """Convert incoming JSON payload to pandas.DataFrame and align columns.

    payload may be a dict (single record) or a list of dicts (batch). If the
    model exposes `feature_names_in_` the DataFrame will be reindexed to those
    columns (missing columns will be NaN).
    """
    if isinstance(payload, dict):
        # Distinguish between dict-of-scalars (single record) and dict-of-lists
        if all(not isinstance(v, (list, tuple)) for v in payload.values()):
            df = pd.DataFrame([payload])
        else:
            df = pd.DataFrame(payload)
    elif isinstance(payload, list):
        df = pd.DataFrame(payload)
    else:
        raise ValueError("Unsupported JSON format")

    # Reindex to model's expected features if available
    if model is not None and hasattr(model, "feature_names_in_"):
        required = list(model.feature_names_in_)
        df = df.reindex(columns=required)

    # Try to coerce numeric-like values
    try:
        df = df.apply(pd.to_numeric, errors="ignore")
    except Exception:
        pass

    return df


def predict_with_model(model: Any, payload: Any) -> List[float]:
    """Run prediction using the provided model and JSON payload.

    Returns a plain Python list of predictions.
    """
    df = _payload_to_dataframe(payload, model=model)

    # sklearn-like wrappers
    if hasattr(model, "predict") and "Booster" not in type(model).__name__:
        preds = model.predict(df)
    else:
        # assume raw xgboost.Booster
        dmat = xgb.DMatrix(df)
        preds = model.predict(dmat)

    if hasattr(preds, "tolist"):
        return preds.tolist()
    return [float(preds)]
