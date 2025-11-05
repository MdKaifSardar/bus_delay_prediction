from asgiref.wsgi import WsgiToAsgi
from app import app as flask_app

# Wrap the Flask (WSGI) app as an ASGI app for uvicorn
app = WsgiToAsgi(flask_app)
