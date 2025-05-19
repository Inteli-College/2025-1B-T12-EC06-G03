from flask import Flask, jsonify
from routes.images import images_bp
from routes.health import health_bp

url_prefix = '/api/v1'

app = Flask(__name__)

app.register_blueprint(images_bp, url_prefix=url_prefix)
app.register_blueprint(health_bp, url_prefix=url_prefix)


if __name__ == '__main__':
    app.run(host='0.0.0.0' , port=5000)

