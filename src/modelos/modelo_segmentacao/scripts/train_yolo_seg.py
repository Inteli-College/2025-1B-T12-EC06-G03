from ultralytics import YOLO

# 🔧 Carregar o modelo pré-treinado de segmentação
model = YOLO('yolov8n-seg.pt')  # você pode trocar por yolov8s-seg.pt, yolov8m-seg.pt, etc.

# 🚀 Iniciar o treinamento
model.train(
    data='../data.yaml',  # Caminho para o arquivo de configuração
    epochs=50,             # Número de épocas (ajuste conforme seu dataset)
    imgsz=640,              # Tamanho das imagens (padrão 640)
    batch=8,                # Batch size (ajuste conforme sua GPU)
    project='../runs',    # Pasta onde os resultados serão salvos
    name='segmentation_model', # Nome do experimento
    save=True,
    save_period=10,         # Salva checkpoint a cada 10 épocas
    patience=20,            # Early stopping se não melhorar após 20 épocas
    pretrained=True         # Usar pesos pré-treinados (recomendado)
)
