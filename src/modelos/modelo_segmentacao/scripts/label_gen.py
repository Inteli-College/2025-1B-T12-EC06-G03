import os
import cv2
import numpy as np

ROOT_IMAGES = '../yolo/images'
ROOT_MASKS = '../yolo/masks'
ROOT_LABELS = '../yolo/labels'
DATASET_SPLITS = ['train', 'val']
CLASS_MAP = {
    'thermal': 0,
    'retraction': 1
}
MIN_CONTOUR_AREA = 50


def log(msg, level="INFO"):
    prefix = {
        "INFO": "[INFO]",
        "SUCCESS": "[SUCESSO]",
        "WARN": "[ATENÇÃO]",
        "ERROR": "[ERRO]"
    }.get(level, "[INFO]")
    print(f"{prefix} {msg}")


def generate_label_line(contour, class_id, img_w, img_h):
    x, y, w, h = cv2.boundingRect(contour)
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    polygon = contour.reshape(-1, 2)
    poly_norm = [
        f"{(px / img_w):.6f} {(py / img_h):.6f}"
        for px, py in polygon
    ]
    poly_flat = ' '.join(poly_norm)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} {poly_flat}"


for split in DATASET_SPLITS:
    images_dir = os.path.join(ROOT_IMAGES, split)
    masks_dir = os.path.join(ROOT_MASKS, split)
    labels_dir = os.path.join(ROOT_LABELS, split)

    if not os.path.exists(images_dir):
        log(f"Diretório de imagens não encontrado: {images_dir}", "ERROR")
        continue

    os.makedirs(labels_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            log(f"Imagem não pôde ser lida: {image_name} ({split})", "WARN")
            continue

        img_h, img_w = image.shape[:2]
        label_lines = []

        for class_folder, class_id in CLASS_MAP.items():
            mask_path = os.path.join(masks_dir, class_folder, image_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                log(f"Máscara não encontrada para {image_name} na classe '{class_folder}' ({split})", "WARN")
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [
                cnt for cnt in contours
                if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA
            ]

            if not valid_contours:
                log(f"Sem contornos relevantes para {image_name} na classe '{class_folder}'", "WARN")

            for contour in valid_contours:
                line = generate_label_line(contour, class_id, img_w, img_h)
                label_lines.append(line)

        label_filename = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)

        if label_lines:
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))
            log(f"Label salvo: {label_path}", "SUCCESS")
        else:
            log(f"Nenhuma anotação gerada para {image_name}. Label vazio não foi salvo.", "WARN")

log("Todos os labels foram gerados com sucesso!", "SUCCESS")
