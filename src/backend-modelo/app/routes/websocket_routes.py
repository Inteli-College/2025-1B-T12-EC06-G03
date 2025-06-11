from flask import Blueprint
from flask_socketio import emit, Namespace, disconnect
from app import socketio, db  
from app.models.models import Imagem
from ..utils.download_image import download_image_from_url
from ..utils.classify_image import Classifier
import os
import cv2
import requests
import json
from datetime import datetime
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
        print("Cliente conectado ao namespace de inferência.")

    def on_disconnect(self):
        print("Cliente desconectado do namespace de inferência.")

    def save_fissura_to_java_backend(self, imagem_id, label, confidence, coords):
        """Salva a fissura classificada no banco de dados via backend Java"""
        try:
            # Validar se há uma detecção válida
            if not label or confidence is None:
                print(f"Detecção inválida para imagem {imagem_id}: label={label}, confidence={confidence}")
                return False
            
            # Preparar coordenadas no formato JSON correto
            coords_json = None
            if coords and len(coords) >= 4:
                coords_json = {
                    "x": int(coords[0]) if coords[0] is not None else 0,
                    "y": int(coords[1]) if coords[1] is not None else 0,
                    "width": int(coords[2] - coords[0]) if coords[2] is not None and coords[0] is not None else 0,
                    "height": int(coords[3] - coords[1]) if coords[3] is not None and coords[1] is not None else 0
                }
            
            # Determinar gravidade baseada na confiança
            if confidence >= 0.8:
                gravidade = "Alta"
            elif confidence >= 0.6:
                gravidade = "Média" 
            else:
                gravidade = "Baixa"
            
            # Payload para o backend Java - formato esperado pelo controller
            fissura_data = {
                "imagem": {"id": imagem_id},
                "tipo": label,
                "coordenadas": json.dumps(coords_json) if coords_json else None,
                "gravidade": gravidade,
                "confianca": float(confidence),
                "aprovado": False,
                "aprovadoPor": None,
                "dataDeteccao": datetime.now().isoformat()
            }
            
            print(f"Enviando fissura para backend Java: {fissura_data}")
            
            java_backend_url = "http://localhost:8080/api/fissura"
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(java_backend_url, json=fissura_data, headers=headers, timeout=30)
            
            if response.status_code in [200, 201]:
                print(f"Fissura salva com sucesso para imagem {imagem_id}")
                return True
            else:
                print(f"Erro ao salvar fissura: HTTP {response.status_code}")
                print(f"Resposta do servidor: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Erro de conexão com backend Java: {e}")
            return False
        except Exception as e:
            print(f"Erro inesperado ao salvar fissura: {e}")
            return False

    def on_infer_images(self, data):
        """Processa imagens para inferência de fissuras"""
        try:
            # Validação do payload
            image_ids = data.get("image_ids")
            if not isinstance(image_ids, list) or not all(isinstance(i, int) for i in image_ids):
                emit("error", {"error": "'image_ids' deve ser uma lista de inteiros"})
                return

            print(f"Iniciando processamento de {len(image_ids)} imagens: {image_ids}")
            
            # Busca no banco de dados Python (para obter caminhos das imagens)
            images = Imagem.query.filter(Imagem.id.in_(image_ids)).all()
            
            if not images:
                emit("error", {"error": "Nenhuma imagem encontrada com os IDs fornecidos"})
                return
                
            emit("status", {"message": f"{len(images)} imagens encontradas. Iniciando classificação..."})

            results = []
            processed_count = 0

            for img in images:
                try:
                    processed_count += 1
                    emit("status", {"message": f"Processando imagem {processed_count}/{len(images)} (ID: {img.id})..."})
                    
                    # Construir URL da imagem
                    image_url = os.path.join(IMG_URL_PREFIX, img.caminho_arquivo)
                    local_path = os.path.join("images", img.caminho_arquivo)
                    
                    # Criar diretório se não existir
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
                    # Download da imagem
                    download_image_from_url(image_url, local_path)
                    
                    if not os.path.exists(local_path):
                        raise Exception("Falha no download da imagem")

                    # Classificação usando o modelo
                    image_array = cv2.imread(local_path)
                    if image_array is None:
                        raise Exception("Não foi possível carregar a imagem")
                        
                    label, confidence, coords = self.classifier.classify(image_array)
                    
                    print(f"Classificação da imagem {img.id}: label={label}, confidence={confidence}, coords={coords}")

                    # Marcar como processada no banco Python
                    img.processada = True
                    db.session.commit()

                    # Salvar fissura no banco de dados Java (se detectada)
                    fissura_saved = False
                    if label and confidence is not None:
                        fissura_saved = self.save_fissura_to_java_backend(img.id, label, confidence, coords)

                    # Adicionar resultado
                    result_item = {
                        "id": img.id,
                        "caminho": image_url,
                        "label": label,
                        "confidence": float(confidence) if confidence is not None else None,
                        "coords": coords,
                        "severity": "Alta" if confidence and confidence >= 0.8 else "Média" if confidence and confidence >= 0.6 else "Baixa",
                        "fissura_saved": fissura_saved,
                        "error": None
                    }
                    
                    results.append(result_item)
                    print(f"Resultado para imagem {img.id}: {result_item}")

                except Exception as e:
                    error_msg = str(e)
                    print(f"Erro ao processar imagem {img.id}: {error_msg}")
                    
                    # Rollback em caso de erro
                    db.session.rollback()
                    
                    results.append({
                        "id": img.id,
                        "caminho": image_url if 'image_url' in locals() else "N/A",
                        "label": None,
                        "confidence": None,
                        "coords": None,
                        "severity": None,
                        "fissura_saved": False,
                        "error": error_msg
                    })

                finally:
                    # Limpeza de arquivos temporários
                    if 'local_path' in locals() and os.path.exists(local_path):
                        try:
                            os.remove(local_path)
                        except:
                            pass
                    
                    # Limpeza de diretórios vazios
                    if 'local_path' in locals():
                        dir_path = os.path.dirname(local_path)
                        while dir_path != "images" and os.path.exists(dir_path):
                            try:
                                os.rmdir(dir_path)
                                dir_path = os.path.dirname(dir_path)
                            except OSError:
                                break

            # Emitir resultados finais
            emit("results", {"results": results})
            
            # Estatísticas finais
            successful_classifications = len([r for r in results if r["label"] is not None and r["error"] is None])
            saved_fissuras = len([r for r in results if r["fissura_saved"]])
            
            final_message = f"Processamento completo! {successful_classifications}/{len(results)} imagens classificadas, {saved_fissuras} fissuras salvas no banco."
            emit("fim", {"message": final_message, "total_processed": len(results), "successful": successful_classifications, "fissuras_saved": saved_fissuras})
            
            print(final_message)

        except Exception as e:
            error_msg = f"Erro geral no processamento: {str(e)}"
            print(error_msg)
            emit("error", {"error": error_msg})
        
        finally:
            # Desconectar cliente após processamento
            disconnect()

# Registrar namespace no socketio
socketio_bp = Blueprint('socketio_bp', __name__)
socketio.on_namespace(InferenceNamespace("/ws/infer"))
