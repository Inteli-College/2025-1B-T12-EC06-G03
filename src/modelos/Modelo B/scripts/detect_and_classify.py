import os
import cv2
import torch
import json
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from torch import nn

# CONFIGURAÇÕES
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CARREGAR MAPEAMENTO DE CLASSES USADO NO TREINO
with open("../models/class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# CARREGAR MODELO YOLO
yolo_model = YOLO('../yolo/runs/detect/fissura-detector/weights/best.pt')

# DEFINIÇÃO DO MODELO CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 30 * 30, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# CARREGAR MODELO CNN TREINADO
cnn_model = CNN().to(DEVICE)
cnn_model.load_state_dict(torch.load('../models/cnn_model.pt', map_location=DEVICE))
cnn_model.eval()

# TRANSFORMAÇÃO DA IMAGEM PARA A CNN
cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# CLASSIFICAÇÃO DE IMAGEM RECORTADA
def classificar_fissura(crop_bgr):
    resized = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE))
    tensor = cnn_transform(resized).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = cnn_model(tensor)
        pred = torch.argmax(output, dim=1).item()
        classe_bruta = idx_to_class[pred]
        return classe_bruta.split("_")[-1]  # remove prefixo, se houver

