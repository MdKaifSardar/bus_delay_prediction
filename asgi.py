"""ASGI wrapper for the Flask WSGI application.

Uvicorn is an ASGI server. Flask is WSGI. To run Flask under uvicorn we
wrap the Flask `app` with `asgiref.wsgi.WsgiToAsgi` and expose the variable
`app` for uvicorn to import.

This file should be simple and import-light so it can be used by the server
process directly.
"""
from asgiref.wsgi import WsgiToAsgi

# Import the Flask app from the root-level module `app.py` which defines
# `app = create_app()` at module scope.
from app import app as flask_app

# Expose ASGI app for uvicorn: `uvicorn asgi:app`
app = WsgiToAsgi(flask_app)
