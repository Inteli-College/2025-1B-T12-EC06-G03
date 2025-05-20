import os
import cv2
import torch
import json
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from torch import nn

# Configurações
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregamento dos modelos e mapeamentos
class Classifier:
    def __init__(self,
                 cnn_model_path: str = "../inference_models/cnn_model.pt",
                 yolo_model_path: str = "../inference_models/yolo.pt",
                 class_map_path: str = "../inference_models/class_to_idx.json"):
        # CNN
        self.cnn = self._load_cnn(cnn_model_path)
        # YOLO
        self.yolo = YOLO(yolo_model_path)
        # mapeamento idx -> classe
        with open(class_map_path, "r") as f:
            class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

        # Transformação para CNN
        self.cnn_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def _load_cnn(self, path: str) -> nn.Module:
        # Define arquitetura
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
                    nn.Linear(64, len(self.idx_to_class) if hasattr(self, 'idx_to_class') else 2)
                )

            def forward(self, x):
                return self.fc(self.conv(x))

        model = CNN().to(DEVICE)
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        return model

    def classify(self, image: np.ndarray) -> (str, float):
        """
        Executa detecção com YOLO e classificação com CNN.
        Retorna: (label, confidence)
        """
        # Pré-processamento de imagem
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        blur = cv2.GaussianBlur(eq, (0, 0), 3)
        sharpened = cv2.addWeighted(eq, 1.2, blur, -0.2, 0)
        img_yolo = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

        # Detecção YOLO
        results = self.yolo(img_yolo, conf=0.05)[0]
        if not results.boxes or len(results.boxes.xyxy) == 0:
            return "Nenhuma fissura detectada", 0.0

        # Seleciona a caixa com maior confiança
        confs = results.boxes.conf.cpu().numpy()
        idx_max = np.argmax(confs)
        x1, y1, x2, y2 = map(int, results.boxes.xyxy[idx_max])
        confidence = float(confs[idx_max])

        box = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

        # Crop para CNN
        crop = image[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return "Fissura muito pequena para análise", confidence

        # Transformação e classificação CNN
        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        tensor = self.cnn_transform(resized).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = self.cnn(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            label_key = self.idx_to_class.get(pred, f"classe_{pred}")
            label = label_key.split("_")[-1]
            conf_cnn = float(probs[pred])

        return label, conf_cnn, box

# API simplicada de uso

def classify_image_from_path(image_path: str) -> dict:
    """
    Lê imagem de disco, classifica e retorna resultados.
    { 'label': str, 'confidence': float }
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada em {image_path}")

    classifier = Classifier()
    label, confidence, coords = classifier.classify(img)
    return { 'label': label, 'confidence': confidence, 'coords': coords }

if __name__ == "__main__":
    image_path = "../../images/FT80.png"  
    result = classify_image_from_path(image_path)
    print(f"Resultado: {result}")