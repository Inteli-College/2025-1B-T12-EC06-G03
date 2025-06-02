import cv2
import numpy as np
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm

# 🚀 Carregar modelo treinado
model = YOLO('../runs/segmentation_model/weights/best.pt')

# Pasta com as imagens de teste
image_dir = '../yolo/images/test'
output_dir = '../yolo/output'

os.makedirs(output_dir, exist_ok=True)

# 🔗 Mapeamento das classes
class_map = {
    0: 'thermal',
    1: 'retraction'
}

# 🔍 Loop nas imagens
for image_name in tqdm(os.listdir(image_dir)):
    if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    output = image.copy()

    results = model(image, conf=0.1)
    
    # Dicionário para contar as classes detectadas
    class_counts = {}

    for result in results:
        if result.masks is None:
            print(f"Nenhuma detecção na imagem {image_name}")
            continue  # Pula para a próxima imagem

        masks = result.masks.data.cpu().numpy()       # Máscaras binárias
        class_ids = result.boxes.cls.cpu().numpy()      # Classes
        confidences = result.boxes.conf.cpu().numpy()     # Confiança

        for mask, class_id, conf in zip(masks, class_ids, confidences):
            # Incrementa o contador de classes
            cls_name = class_map.get(int(class_id), 'Desconhecido')
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            # 🔲 Processar a máscara
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            mask = (mask > 0.5).astype(np.uint8) * 255

            # 🟦 Encontrar contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                cv2.drawContours(output, [contour], -1, (0, 0, 255), 2)  # Vermelho

    # Determinar a classe mais predita
    if class_counts:
        most_predicted_class = max(class_counts, key=class_counts.get)
        # Adiciona o texto no canto superior esquerdo da imagem
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = most_predicted_class
        font_scale = 0.5
        thickness = 1
        margin = 10
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = output.shape[1] - text_width - margin
        y = output.shape[0] - margin
        cv2.putText(output, text, (x, y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    # Salvar resultado visual
    save_path = os.path.join(output_dir, image_name)
    cv2.imwrite(save_path, output)