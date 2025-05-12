from ultralytics import YOLO

# Carregar modelo base
model = YOLO('yolov8n.pt')  # pode trocar por yolov8s.pt ou yolov8m.pt

# Treinar com seu dataset
model.train(
    data='fissure.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    name='fissura-detector',
    project='runs/detect'
)