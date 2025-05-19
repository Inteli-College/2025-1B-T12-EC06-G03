from flask import Blueprint, request, jsonify
import json

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def download_image():
    return jsonify({'status': 'ok'}), 200