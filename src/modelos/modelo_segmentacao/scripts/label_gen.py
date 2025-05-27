import os
import cv2
import numpy as np

# Diretórios principais
root_images = '../yolo/images'
root_masks = '../yolo/masks'
root_labels = '../yolo/labels'

# Mapeamento de classes
class_map = {
    'thermal': 0,
    'retraction': 1
}

# Dataset splits
dataset_splits = ['train', 'val']

for split in dataset_splits:
    images_split_dir = os.path.join(root_images, split)
    masks_split_dir = os.path.join(root_masks, split)
    labels_split_dir = os.path.join(root_labels, split)
    
    # Garante que o diretório de labels do split exista
    os.makedirs(labels_split_dir, exist_ok=True)
    
    for image_name in os.listdir(images_split_dir):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        image_path = os.path.join(images_split_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Imagem não pode ser lida: {image_name} ({split})")
            continue
        
        img_h, img_w = image.shape[:2]
        label_lines = []
        
        # Para cada classe, verifica se há máscara e adiciona as anotações
        for class_folder, class_id in class_map.items():
            mask_path = os.path.join(masks_split_dir, class_folder, image_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"⚠️ Máscara não encontrada para {image_name} na classe {class_folder} ({split})")
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 50:
                    continue  # ignora pequenos ruídos
                
                # Bounding box e normalização
                x, y, w, h = cv2.boundingRect(contour)
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # Normaliza os pontos do polígono
                polygon = contour.reshape(-1, 2)
                poly_norm = []
                for point in polygon:
                    px = point[0] / img_w
                    py = point[1] / img_h
                    poly_norm.extend([round(px, 6), round(py, 6)])
                
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} " + ' '.join(map(str, poly_norm))
                label_lines.append(line)
        
        # Salva o arquivo label na pasta do split sem subpastas adicionais
        label_path = os.path.join(labels_split_dir, image_name.rsplit('.', 1)[0] + '.txt')
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))
        
        print(f"✅ Label salvo: {label_path}")

print("🎯 Todos os labels foram gerados com sucesso!")
