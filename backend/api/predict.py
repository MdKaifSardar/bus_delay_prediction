from flask import current_app, request, jsonify
from . import bp

from backend.services.predict import run_prediction


@bp.route("/predict", methods=["POST"])
def predict():
    model_loaded = current_app.config.get("MODEL_LOADED", False)
    load_error = current_app.config.get("LOAD_ERROR")

    if not model_loaded:
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
        preds = run_prediction(payload)
        return jsonify({"predictions": preds})
    except ValueError as e:
        return jsonify({"error": "Unsupported JSON format", "details": str(e)}), 400
    except Exception as e:
        current_app.logger.exception("Prediction failed: %s", e)
        return jsonify({"error": "Prediction failed", "details": "Internal server error"}), 500
