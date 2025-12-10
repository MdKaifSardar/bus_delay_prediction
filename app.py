"""Small entrypoint that wires the refactored backend package.

This file is intentionally tiny: it imports create_app from the new
`backend` package and runs the returned Flask application. The real logic is
moved into `backend.models` and `backend.api` to keep things modular and easier
to test and extend.
"""

from backend import create_app


app = create_app()


if __name__ == "__main__":
    # Simple local run for development
    app.run(host="0.0.0.0", port=5000, debug=True)
