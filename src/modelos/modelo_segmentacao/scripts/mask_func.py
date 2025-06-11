import os
import cv2
import numpy as np
from skimage.filters import sato
from skimage import measure


def generate_mask(image, initial_thresh=75, min_thresh=5, step=5):
    """
    Gera uma máscara binária a partir de uma imagem usando processamento de imagem.
    Se a máscara sair vazia, reduz o threshold gradualmente até min_thresh.

    Args:
        image (np.ndarray): imagem BGR.
        initial_thresh (int): valor inicial de threshold.
        min_thresh (int): valor mínimo de threshold a testar.
        step (int): decremento em cada iteração de fallback.

    Returns:
        np.ndarray: máscara binária uint8 (0 e 255).
    """
    if image is None:
        raise ValueError("Imagem inválida ou não carregada corretamente.")

    # 🔸 Grayscale + suavização + CLAHE + Sato + normalização
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(gray, d=2, sigmaColor=25, sigmaSpace=25)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)
    sato_filtered = sato(enhanced, sigmas=range(1, 2), black_ridges=True)
    sato_norm = (sato_filtered - np.min(sato_filtered)) / (np.max(sato_filtered) - np.min(sato_filtered))
    sato_uint8 = (sato_norm * 255).astype(np.uint8)

    # configurações de morfologia e tamanho mínimo
    kernel = np.ones((5, 5), np.uint8)
    min_size = 200  # pixels

    # loop de thresholds decrescentes
    for thresh in range(initial_thresh, min_thresh - 1, -step):
        _, mask = cv2.threshold(sato_uint8, thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # limpeza de pequenas regiões
        mask_bool = mask.astype(bool)
        labeled = measure.label(mask_bool, connectivity=2)
        cleaned = np.zeros_like(mask)
        for region in measure.regionprops(labeled):
            if region.area >= min_size:
                cleaned[tuple(region.coords.T)] = 255

        # se encontrou algo, já retorna
        if np.any(cleaned):
            return cleaned

    # se nenhum threshold produziu algo, retorna o mais limpo (vazio)
    return np.zeros_like(sato_uint8)



if __name__ == "__main__":
    # 🔥 Executa o pipeline completo para gerar as máscaras dos datasets

    datasets = ['train', 'val', 'test']
    classes = ['thermal', 'retraction']

    for dataset in datasets:
        root_images = os.path.join('../yolo/images', dataset)
        root_masks = os.path.join('../yolo/masks', dataset)

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

            try:
                mask = generate_mask(image)
            except Exception as e:
                print(f"❌ Error generating mask for {image_name}: {e}")
                continue

            mask_path = os.path.join(root_masks, mask_subfolder, image_name)
            cv2.imwrite(mask_path, mask)
            print(f"✅ Mask saved: {mask_path}")

    print("🎯 All masks have been generated successfully!")
