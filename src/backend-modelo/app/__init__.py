from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO

db = SQLAlchemy()
socketio = SocketIO(cors_allowed_origins="*")  

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')
    app.config['SECRET_KEY'] = 'secret!'

    db.init_app(app)
    socketio.init_app(app)  
    
    from app.routes.images import images_bp
    from app.routes.health import health_bp
    from app.routes.inference import inference_bp
    from app.routes.websocket_routes import socketio_bp

    app.register_blueprint(images_bp, url_prefix='/api/v1')
    app.register_blueprint(health_bp, url_prefix='/api/v1')
    app.register_blueprint(inference_bp, url_prefix='/api/v1')
    app.register_blueprint(socketio_bp)

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        db.session.remove()

    return app
