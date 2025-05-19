from flask import Blueprint, request, jsonify
from utils.download_image import download_image_from_url
import os
import urllib.error

images_bp = Blueprint('images', __name__)

@images_bp.route('/image', methods=['POST'])
def download_image():
    """
    Endpoint to download an image from a given URL.
    """
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'No URL provided'}), 400

        url = data['url']
        save_dir = 'images/'

        ## Validate URL format
        if not url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL format'}), 400

        ## Create directory if it doesn't exist to save images
        os.makedirs(save_dir, exist_ok=True)
        ## Extract file name from URL and create a full path
        file_name = os.path.basename(url)
        save_as = os.path.join(save_dir, file_name)

        ## Error handling for downloading the image
        try:
            download_image_from_url(url, save_as)
        except urllib.error.URLError as e:
            return jsonify({'error': f'Failed to download the image: {str(e)}'}), 400
        except urllib.error.HTTPError as e:
            return jsonify({'error': f'HTTP error occurred: {str(e)}'}), 400
        except OSError as e:
            return jsonify({'error': f'File system error: {str(e)}'}), 500

        return jsonify({'message': 'Image downloaded successfully', 'url': url, 'path': save_as}), 200

    except KeyError:
        return jsonify({'error': 'Invalid JSON payload'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
