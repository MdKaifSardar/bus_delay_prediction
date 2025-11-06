from flask import Blueprint, current_app, request, jsonify

from .model import predict_with_model

bp = Blueprint("api", __name__)

# List of required features & an example body (kept here for easy reference)
REQUIRED_FEATURES = [
    "segment_id",
    "distance_km",
    "avg_speed_kmph",
    "traffic_mean",
    "traffic_std",
    "traffic_max",
    "traffic_p90",
    "rain_intensity",
    "temperature_celsius",
    "visibility_km",
    "num_signals",
    "num_stops",
    "is_holiday",
    "day_of_week",
    "hour_of_day",
]

EXAMPLE_SINGLE = {
    "segment_id": 3,
    "distance_km": 1.60,
    "avg_speed_kmph": 30.03,
    "traffic_mean": 1.48,
    "traffic_std": 0.14,
    "traffic_max": 1.82,
    "traffic_p90": 1.70,
    "rain_intensity": 2.53,
    "temperature_celsius": 28.52,
    "visibility_km": 6.07,
    "num_signals": 0,
    "num_stops": 2,
    "is_holiday": 1,
    "day_of_week": 6,
    "hour_of_day": 13,
}


@bp.route("/predict", methods=["POST"])
def predict():
    # Check readiness/loading state
    model = current_app.config.get("MODEL")
    load_error = current_app.config.get("LOAD_ERROR")
    model_loaded = current_app.config.get("MODEL_LOADED", False)

    if not model_loaded:
        # If there was a load error, surface it as a 500; otherwise indicate loading.
        if load_error:
            current_app.logger.error("Predict requested but model failed to load: %s", load_error)
            return jsonify({"error": "Model failed to load", "details": load_error}), 500
        current_app.logger.info("Predict requested while model is still loading")
        return jsonify({"error": "Model is still loading, try again shortly"}), 503

    try:
        payload = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "Invalid JSON", "details": str(e)}), 400

    if payload is None:
        return jsonify({"error": "Empty request body"}), 400

    try:
        preds = predict_with_model(model, payload)
        return jsonify({"predictions": preds})
    except ValueError as e:
        return jsonify({"error": "Unsupported JSON format", "details": str(e)}), 400
    except Exception as e:
        current_app.logger.exception("Prediction failed: %s", e)
        return jsonify({"error": "Prediction failed", "details": "Internal server error"}), 500


@bp.route("/", methods=["GET"])
def index():
    return (
        jsonify(
            {
                "message": "Bus delay API running",
                "endpoints": ["/predict"],
                "required_features": REQUIRED_FEATURES,
                "example_single_record": EXAMPLE_SINGLE,
                "notes": [
                    "POST JSON to /predict with Content-Type: application/json.",
                    "Accepts either a single object (as shown) or a list of such objects for batch predictions.",
                    "If the model exposes feature_names_in_, the server will reindex incoming data to that order; missing features become NaN (may cause prediction to fail).",
                    "Numeric-like strings will be coerced where possible.",
                ],
            }
        ),
        200,
    )


@bp.route("/favicon.ico")
def favicon():
    return "", 204


@bp.route("/ready", methods=["GET"])
def ready():
    """Readiness probe: returns 200 when model is loaded, 503 otherwise."""
    model_loaded = current_app.config.get("MODEL_LOADED", False)
    load_error = current_app.config.get("LOAD_ERROR")
    if model_loaded:
        return jsonify({"ready": True}), 200
    if load_error:
        return jsonify({"ready": False, "error": load_error}), 500
    return jsonify({"ready": False, "message": "model loading"}), 503
