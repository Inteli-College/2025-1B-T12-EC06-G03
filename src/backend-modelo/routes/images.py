from flask import Blueprint, request, jsonify
import json

images_bp = Blueprint('images', __name__)

@images_bp.route('/image', methods=['POST'])
def download_image():
    """
    Endpoint to download an image from a given URL.
    """
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400

    url = data['url']

    ## Aqui vamos implementar o download da imagem    
    return jsonify({'message': 'Image downloaded successfully', 'url': url}), 200