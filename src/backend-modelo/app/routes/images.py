from flask import Blueprint, request, jsonify
from app.utils.download_image import download_image_from_url
import os
import urllib.error
from app import db
from app.models.models import Imagem

images_bp = Blueprint('images', __name__)

@images_bp.route('/image-download', methods=['POST'])
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

@images_bp.route('/get-images', methods=['GET'])
def get_images():
    """
    Endpoint to retrieve all images.
    """
    images = Imagem.query.all()
    result = []
    for img in images:
        result.append({
            "id": img.id,
            "fachada_id": img.fachada_id,
            "caminho_arquivo": img.caminho_arquivo,
            "nome_arquivo": img.nome_arquivo,
            "data_captura": img.data_captura.isoformat() if img.data_captura else None,
            "data_upload": img.data_upload.isoformat() if img.data_upload else None,
            "metadados": img.metadados,
            "processada": img.processada
        })
    db.session.close()
    return jsonify(result), 200

@images_bp.route('/get-image/<int:image_id>', methods=['GET'])
def get_image(image_id):
    """
    Endpoint to retrieve an image by its ID.
    """
    image = Imagem.query.get(image_id)
    if not image:
        db.session.close()
        return jsonify({'error': 'Image not found'}), 404

    result = {
        "id": image.id,
        "fachada_id": image.fachada_id,
        "caminho_arquivo": image.caminho_arquivo,
        "nome_arquivo": image.nome_arquivo,
        "data_captura": image.data_captura.isoformat() if image.data_captura else None,
        "data_upload": image.data_upload.isoformat() if image.data_upload else None,
        "metadados": image.metadados,
        "processada": image.processada
    }
    db.session.close()
    return jsonify(result), 200

@images_bp.route('/delete-image/<int:image_id>', methods=['DELETE'])
def delete_image(image_id):
    """
    Endpoint to delete an image by its ID.
    """
    try:
        image = Imagem.query.get(image_id)
        if not image:
            db.session.close()
            return jsonify({'error': 'Image not found'}), 404

        db.session.delete(image)
        db.session.commit()
        db.session.close()
        return jsonify({'message': 'Image deleted successfully'}), 200

    except FileNotFoundError as e:
        db.session.rollback()
        db.session.close()
        return jsonify({'error': f'File not found: {str(e)}'}), 404
    except PermissionError as e:
        db.session.rollback()
        db.session.close()
        return jsonify({'error': f'Permission denied: {str(e)}'}), 403
    except ValueError as e:
        db.session.rollback()
        db.session.close()
        return jsonify({'error': f'Value error: {str(e)}'}), 400
    except TypeError as e:
        db.session.rollback()
        db.session.close()
        return jsonify({'error': f'Type error: {str(e)}'}), 400
    except RuntimeError as e:
        db.session.rollback()
        db.session.close()
        return jsonify({'error': f'Runtime error: {str(e)}'}), 500
    except Exception as e:
        db.session.rollback()
        db.session.close()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@images_bp.route('/update-image/<int:image_id>', methods=['PUT'])
def update_image(image_id):
    """
    Endpoint to update an image by its ID.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        image = Imagem.query.get(image_id)
        if not image:
            return jsonify({'error': 'Image not found'}), 404

        for key, value in data.items():
            if hasattr(image, key):
                setattr(image, key, value)

        db.session.commit()
        return jsonify({'message': 'Image updated successfully'}), 200

    except FileNotFoundError as e:
        db.session.rollback()
        db.session.close()
        return jsonify({'error': f'File not found: {str(e)}'}), 404
    except PermissionError as e:
        db.session.rollback()
        db.session.close()
        return jsonify({'error': f'Permission denied: {str(e)}'}), 403
    except ValueError as e:
        db.session.rollback()
        db.session.close()
        return jsonify({'error': f'Value error: {str(e)}'}), 400
    except TypeError as e:
        db.session.rollback()
        db.session.close()
        return jsonify({'error': f'Type error: {str(e)}'}), 400
    except RuntimeError as e:
        db.session.rollback()
        db.session.close()
        return jsonify({'error': f'Runtime error: {str(e)}'}), 500
    except Exception as e:
        db.session.rollback()
        db.session.close()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@images_bp.route('/upload-image', methods=['POST'])
def upload_image():
    """
    Endpoint to upload an image.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the file to a directory
        save_dir = 'uploads/'
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file.filename)
        file.save(file_path)

        return jsonify({'message': 'File uploaded successfully', 'path': file_path}), 200

    except FileNotFoundError as e:
        return jsonify({'error': f'File not found: {str(e)}'}), 404
    except PermissionError as e:
        return jsonify({'error': f'Permission denied: {str(e)}'}), 403
    except ValueError as e:
        return jsonify({'error': f'Value error: {str(e)}'}), 400
    except TypeError as e:
        return jsonify({'error': f'Type error: {str(e)}'}), 400
    except RuntimeError as e:
        return jsonify({'error': f'Runtime error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500