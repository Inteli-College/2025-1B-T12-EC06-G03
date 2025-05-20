from flask import Blueprint, request, jsonify
from app.utils.download_image import download_image_from_url
import os
import urllib.error
from app import db
from app.models.models import Imagem
from dotenv import load_dotenv

load_dotenv()

IMG_URL_PREFIX = os.getenv("IMG_URL_PREFIX")

inference_bp = Blueprint('inference', __name__, url_prefix='/api/v1')

@inference_bp.route('/infer-images', methods=['POST'])
def infer_images():
    payload = request.get_json()
    if not payload or 'image_ids' not in payload:
        return jsonify({"error": "Invalid payload."}), 400

    image_ids = payload['image_ids']
    if not isinstance(image_ids, list) or not all(isinstance(i, int) for i in image_ids):
        return jsonify({"error": "'image_ids' must be an integer array"}), 400

    images = Imagem.query.filter(Imagem.id.in_(image_ids)).all()

    result = []
    for img in images:
        result.append({
            "id": img.id,
            "caminho_arquivo": os.path.join(IMG_URL_PREFIX, img.caminho_arquivo),
            "processada": img.processada
        })

    print(result)
    return jsonify({"Status:": "imagens recebidas"}), 200