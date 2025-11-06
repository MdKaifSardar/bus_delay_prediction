"""Model utilities: loading and prediction helpers.

This module centralizes model loading and DataFrame preparation so the web
routes remain small and testable.
"""
from typing import Any, List, Optional
import os
import pickle
import logging

import pandas as pd
import xgboost as xgb


# Module logger
logger = logging.getLogger(__name__)


# Simple in-memory cache for the loaded model to avoid reloading on every request
_CACHED_MODEL: Optional[Any] = None


def default_model_path() -> str:
    """Return the default path to the model file.

    Priority:
    1. If the environment variable MODEL_PATH is set, use that.
    2. Otherwise look for `best_xgb_model.pkl` one level above this package.
    """
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        return os.path.abspath(env_path)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "best_xgb_model.pkl"))


def load_model(path: str = None) -> Any:
    """Load and return the pickled model from disk.

    Raises:
      FileNotFoundError: when the model file does not exist.
      RuntimeError: when unpickling fails or an unexpected error occurs.
    """
    if path is None:
        path = default_model_path()

    if not os.path.exists(path):
        logger.error("Model file not found at %s", path)
        raise FileNotFoundError(f"Model file not found at {path}")

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        # Corrupt or incompatible pickle file
        logger.exception("Failed to unpickle model from %s", path)
        raise RuntimeError(f"Failed to load model from {path}: {e}") from e
    except Exception as e:
        # Catch-all for unexpected errors (permission, IO, etc.)
        logger.exception("Unexpected error while loading model from %s", path)
        raise RuntimeError(f"Unexpected error loading model from {path}: {e}") from e

    if model is None:
        logger.error("Model loaded from %s is None", path)
        raise RuntimeError(f"Model loaded from {path} is None")

    return model


def get_model(path: str = None, reload: bool = False) -> Any:
    """Return a cached model instance, loading it if needed.

    Parameters
    - path: optional path to override the default model location
    - reload: if True, force reloading from disk

    Raises the same exceptions as `load_model` on failure.
    """
    global _CACHED_MODEL
    if reload or _CACHED_MODEL is None:
        _CACHED_MODEL = load_model(path)
    return _CACHED_MODEL


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
