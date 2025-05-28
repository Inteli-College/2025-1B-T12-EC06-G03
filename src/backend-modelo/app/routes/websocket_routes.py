from flask import Blueprint
from flask_socketio import emit, Namespace, disconnect
from app import socketio, db  
from app.models.models import Imagem
from ..utils.download_image import download_image_from_url
from ..utils.classify_image import Classifier
import os
import cv2
from dotenv import load_dotenv

load_dotenv()
IMG_URL_PREFIX = os.getenv("IMG_URL_PREFIX")

class InferenceNamespace(Namespace):
    def __init__(self, namespace=None):
        super().__init__(namespace)
        self.classifier = Classifier(
            cnn_model_path="app/inference_models/cnn_model.pt",
            yolo_model_path="app/inference_models/yolo.pt",
            class_map_path="app/inference_models/class_to_idx.json"
        )

    def on_connect(self):
        print("Cliente conectado.")

    def on_disconnect(self):
        print("Cliente desconectado.")

    def on_infer_images(self, data):
        # Validação do payload
        image_ids = data.get("image_ids")
        if not isinstance(image_ids, list) or not all(isinstance(i, int) for i in image_ids):
            emit("error", {"error": "'image_ids' deve ser uma lista de inteiros"})
            return

        # Busca no banco
        images = Imagem.query.filter(Imagem.id.in_(image_ids)).all()
        emit("status", {"message": f"{len(images)} imagens recebidas. Processando..."})

        results = []

        for img in images:
            image_url = os.path.join(IMG_URL_PREFIX, img.caminho_arquivo)
            local_path = os.path.join("images", img.caminho_arquivo)
            try:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                download_image_from_url(image_url, local_path)

                # Classificação
                image_array = cv2.imread(local_path)
                label, confidence, coords = self.classifier.classify(image_array)

                # Marca como processada
                img.processada = True
                db.session.commit()

                results.append({
                    "id": img.id,
                    "caminho": image_url,
                    "label": label,
                    "confidence": confidence,
                    "coords": coords,
                    "error": None
                })

            except Exception as e:
                db.session.rollback()
                results.append({
                    "id": img.id,
                    "caminho": image_url,
                    "label": None,
                    "confidence": None,
                    "coords": None,
                    "error": str(e)
                })

            finally:
                if os.path.exists(local_path):
                    os.remove(local_path)
                dir_path = os.path.dirname(local_path)
                while dir_path != "images" and os.path.exists(dir_path):
                    try:
                        os.rmdir(dir_path)  
                    except OSError:
                        break  
                    dir_path = os.path.dirname(dir_path)

        # Emite tudo de uma vez
        emit("results", {"results": results})
        emit("fim", {"message": "Processamento completo."})

        # Desconecta o cliente corretamente
        disconnect()

socketio_bp = Blueprint('socketio_bp', __name__)
socketio.on_namespace(InferenceNamespace("/ws/infer"))
