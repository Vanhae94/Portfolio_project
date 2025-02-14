# app/routes/__init__.py
from app.routes.main import main_bp

def register_routes(app):
    app.register_blueprint(main_bp)
