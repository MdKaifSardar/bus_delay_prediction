from flask import Flask

from .model import load_model
from .api import bp as api_bp


def create_app(model_path: str = None) -> Flask:
    """Create and configure the Flask application.

    Loads the ML model at startup and stores it in app.config for access from
    route handlers. Registers the API blueprint from backend.api.
    """
    app = Flask(__name__)

    # Try to load model at startup and store any load error message so routes
    # can return informative errors if the model wasn't available.
    try:
        model = load_model(model_path)
        load_error = None
    except Exception as e:
        model = None
        load_error = str(e)

    app.config["MODEL"] = model
    app.config["LOAD_ERROR"] = load_error

    # Register routes
    app.register_blueprint(api_bp)

    return app
