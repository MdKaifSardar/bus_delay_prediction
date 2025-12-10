from flask import Flask
from threading import Thread
from typing import Optional

from .models import get_model
from .api import bp as api_bp


def _background_model_loader(app: Flask, path: Optional[str] = None) -> None:
    """Load the model in a background thread and record status in app.config.

    This keeps the process alive on platforms like Render even if the model
    is large or momentarily unavailable. The loader sets these config keys:
      - MODEL: the loaded model instance or None
      - LOAD_ERROR: string message if load failed, else None
      - MODEL_LOADED: True when model is ready, False otherwise
    """
    try:
        model = get_model(path=path, reload=True)
        app.logger.info("Model loaded successfully")
        app.config["MODEL"] = model
        app.config["LOAD_ERROR"] = None
        app.config["MODEL_LOADED"] = True
    except Exception as e:
        app.logger.exception("Model failed to load in background: %s", e)
        app.config["MODEL"] = None
        app.config["LOAD_ERROR"] = str(e)
        app.config["MODEL_LOADED"] = False


def create_app(model_path: str = None) -> Flask:
    """Create and configure the Flask application.

    The model is loaded in the background so the web process can start
    immediately. A readiness endpoint reports when the model is ready.
    """
    app = Flask(__name__)

    # Initialize model state; background loader will update these fields.
    app.config["MODEL"] = None
    app.config["LOAD_ERROR"] = None
    app.config["MODEL_LOADED"] = False

    # Register routes
    app.register_blueprint(api_bp)

    # Start background loader thread (daemon so it won't block shutdown)
    loader = Thread(target=_background_model_loader, args=(app, model_path), daemon=True)
    loader.start()

    return app
