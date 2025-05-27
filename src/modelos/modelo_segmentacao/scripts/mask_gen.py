import os
import cv2
import numpy as np
from skimage.filters import sato
from skimage import measure

# Datasets and classes
datasets = ['train', 'val', 'test']
classes = ['thermal', 'retraction']

for dataset in datasets:
    # Diretórios base
    root_images = os.path.join('../yolo/images', dataset)
    root_masks = os.path.join('../yolo/masks', dataset)

    # Garante que as pastas de máscara existem
    for class_name in classes:
        os.makedirs(os.path.join(root_masks, class_name), exist_ok=True)

    if not os.path.exists(root_images):
        print(f"⚠️ Images folder not found: {root_images}")
        continue

    for image_name in os.listdir(root_images):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        upper_name = image_name.upper()
        if 'FT' in upper_name:
            mask_subfolder = 'thermal'
        elif 'FR' in upper_name:
            mask_subfolder = 'retraction'
        else:
            print(f"⚠️ Skipping {image_name}. Name does not match class.")
            continue

        image_path = os.path.join(root_images, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Could not read {image_path}")
            continue

        # 🔸 Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 🔸 Suavização bilateral
        smoothed = cv2.bilateralFilter(gray, d=2, sigmaColor=25, sigmaSpace=25)

        # 🔸 CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(smoothed)

        # 🔸 Filtro Sato
        sato_filtered = sato(enhanced, sigmas=range(1,2), black_ridges=True)

        # 🔸 Normalização manual para [0, 255]
        sato_norm = (sato_filtered - np.min(sato_filtered)) / (np.max(sato_filtered) - np.min(sato_filtered))
        sato_uint8 = (sato_norm * 255).astype(np.uint8)

        # 🔸 Binarização
        _, mask = cv2.threshold(sato_uint8, 75, 255, cv2.THRESH_BINARY)

        # 🔸 Fechamento morfológico (opcional, mas geralmente ajuda)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 🔸 Remoção de pequenos objetos (muito eficaz contra ruído)
        mask_bool = mask.astype(bool)
        mask_clean = measure.label(mask_bool, connectivity=2)
        cleaned_mask = np.zeros_like(mask)

        # Ajuste este valor (min_size) conforme o tamanho dos ruídos
        min_size = 200  # pixels

        for region in measure.regionprops(mask_clean):
            if region.area >= min_size:
                for coord in region.coords:
                    cleaned_mask[coord[0], coord[1]] = 255

        # ✅ Salvar máscara
        mask_path = os.path.join(root_masks, mask_subfolder, image_name)
        cv2.imwrite(mask_path, cleaned_mask)
        print(f"✅ Mask saved: {mask_path}")

print("🎯 All masks have been generated successfully!")