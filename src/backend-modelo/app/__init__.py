from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')

    db.init_app(app)

    from app.routes.images import images_bp
    from app.routes.health import health_bp

    app.register_blueprint(images_bp, url_prefix='/api/v1')
    app.register_blueprint(health_bp, url_prefix='/api/v1')

    return app
