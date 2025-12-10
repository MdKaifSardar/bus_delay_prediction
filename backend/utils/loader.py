"""Model loader and prediction helpers.

This module contains the concrete model-loading logic and prediction helper
functions. It's placed under `backend/utils` to centralize utility code. The
public `backend.models` package re-exports the important symbols so callers
don't need to change their imports.
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
    2. Otherwise look for `best_xgb_model.pkl` inside the sibling `models` folder.
    """
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        return os.path.abspath(env_path)
    # Default to `backend/models/best_xgb_model.pkl` so the model file
    # is colocated with code that logically represents model artifacts.
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "best_xgb_model.pkl")
    )


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
# Prediction-related helpers were moved to `backend.utils.predict` to keep
# the loader focused on model I/O. Importing here would cause a circular
# dependency when re-exporting, so callers should import prediction helpers
# from `backend.utils.predict` or from the `backend.models` compatibility
# package which re-exports them.
