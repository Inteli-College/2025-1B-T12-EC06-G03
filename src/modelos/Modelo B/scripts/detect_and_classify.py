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

# FUNÇÃO PRINCIPAL: ANÁLISE COMPLETA DE UMA IMAGEM
def analisar_imagem(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return "Erro: imagem não encontrada."

    # Pré-processamento para YOLO
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (0, 0), 3)
    sharpened = cv2.addWeighted(eq, 1.2, blur, -0.2, 0)
    img_yolo = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    # Detecção com YOLO
    results = yolo_model(img_yolo, conf=0.05)[0]

    if not results.boxes or len(results.boxes) == 0:
        return "Nenhuma fissura detectada pelo YOLO."

    # Recorte com maior confiança
    confs = results.boxes.conf
    idx_max = torch.argmax(confs).item()
    x1, y1, x2, y2 = map(int, results.boxes.xyxy[idx_max])
    crop = img[y1:y2, x1:x2]

    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return "Fissura detectada, mas muito pequena para ser analisada."

    # Classificação
    label_cnn = classificar_fissura(crop)
    resultado_final = f"✅ Fissura detectada → Tipo: {label_cnn}"

    # Anotação e salvamento
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{label_cnn}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    output_path = os.path.join(OUTPUT_DIR, "resultado_final.png")
    cv2.imwrite(output_path, img)

    return resultado_final, output_path


# TESTE LOCAL DO SCRIPT
if __name__ == "__main__":
    print("🔍 Testando análise de imagem...\n")
    imagem_teste = "../exemplot.png"  # <- substitua pela sua imagem real

    resultado = analisar_imagem(imagem_teste)

    if isinstance(resultado, tuple):
        mensagem, caminho_img = resultado
        print(mensagem)
        print(f"📁 Imagem anotada salva em: {caminho_img}")

        # === Extra: salvar imagem apenas com a predição do YOLO e texto de confiança ===
        from ultralytics import YOLO
        import cv2

        # Recarrega imagem original
        img_original = cv2.imread(imagem_teste)

        # Pré-processa para o YOLO
        gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        blur = cv2.GaussianBlur(eq, (0, 0), 3)
        sharpened = cv2.addWeighted(eq, 1.2, blur, -0.2, 0)
        img_yolo = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

        # Executa novamente YOLO para obter coordenadas e confiança
        yolo_model = YOLO('../yolo/runs/detect/fissura-detector/weights/best.pt')
        results = yolo_model(img_yolo, conf=0.05)[0]

        if results.boxes and len(results.boxes) > 0:
            confs = results.boxes.conf
            idx_max = torch.argmax(confs).item()
            box = results.boxes.xyxy[idx_max]
            x1, y1, x2, y2 = map(int, box)
            conf = float(confs[idx_max]) * 100

            texto = f"fissura: {conf:.1f}%"
            cv2.rectangle(img_original, (x1, y1), (x2, y2), (0, 128, 255), 2)
            cv2.putText(img_original, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 128, 255), 2)

            path_saida = "outputs/resultado_yolo_bruto.png"
            cv2.imwrite(path_saida, img_original)
            print(f"📌 Imagem com box do YOLO salva em: {path_saida}")
        else:
            print("⚠️ YOLO não detectou nenhuma fissura para o salvamento adicional.")

    else:
        print(f"⚠️ Resultado: {resultado}")

