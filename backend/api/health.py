from flask import current_app, jsonify
from . import bp


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


@bp.route("/favicon.ico")
def favicon():
    return "", 204
