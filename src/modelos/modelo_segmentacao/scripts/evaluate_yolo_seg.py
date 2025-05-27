from ultralytics import YOLO

# Caminho para o modelo treinado
MODEL_PATH = '../runs/segmentation_model/weights/best.pt'

# Carregar o modelo
model = YOLO(MODEL_PATH)

# Avaliação no dataset de validação
metrics = model.val()

# Mostrar as métricas principais
print(metrics)