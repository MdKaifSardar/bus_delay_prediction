from typing import Any, List

from backend.models import get_model, predict_with_model


def run_prediction(payload: Any) -> List[float]:
    """Load cached model (or load if missing) and run prediction.

    This service abstracts model access and prediction logic so route handlers
    remain small and easy to test. It raises exceptions on failure which the
    API layer maps to HTTP responses.
    """
    model = get_model()
    return predict_with_model(model, payload)
