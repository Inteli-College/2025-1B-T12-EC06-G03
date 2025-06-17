from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

model.train(
    data='../data.yaml',
    epochs=200,
    imgsz=640,
    batch=8,
    project='../runs',
    name='segmentation_model',
    save=True,
    save_period=10,
    patience=50,
    pretrained=True
)
