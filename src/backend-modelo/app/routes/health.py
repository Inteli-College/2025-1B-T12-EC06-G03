from flask import Blueprint, request, jsonify
import json

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200