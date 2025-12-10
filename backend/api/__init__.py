from flask import Blueprint

# Central API blueprint. Individual route modules import this `bp` and
# register their handlers on it.
bp = Blueprint("api", __name__)

# Import route modules so they register on the blueprint
from . import index  # noqa: E402,F401
from . import predict  # noqa: E402,F401
from . import health  # noqa: E402,F401
