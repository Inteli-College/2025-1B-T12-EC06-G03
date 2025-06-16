import os
import cv2
import numpy as np
from skimage.filters import sato, frangi, meijering
from skimage import measure, morphology
from scipy import ndimage


def adaptive_threshold_analysis(image):
    mean_val = np.mean(image)
    std_val = np.std(image)
    adaptive_thresh = mean_val - 0.5 * std_val
    fallback_thresh = mean_val - 1.2 * std_val
    return max(20, min(120, int(adaptive_thresh))), max(10, min(80, int(fallback_thresh)))


def enhance_cracks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)
    sato_filtered = sato(enhanced, sigmas=range(1, 2), black_ridges=True)
    frangi_filtered = frangi(enhanced, sigmas=range(1, 4), black_ridges=True)
    meijering_filtered = meijering(enhanced, sigmas=range(1, 3), black_ridges=True)
    combined = (0.4 * sato_filtered + 0.4 * frangi_filtered + 0.2 * meijering_filtered)
    combined_norm = (combined - np.min(combined)) / (np.max(combined) - np.min(combined))
    combined_uint8 = (combined_norm * 255).astype(np.uint8)
    return combined_uint8, enhanced


def remove_noise_advanced(mask, min_size=200, aspect_ratio_threshold=3.0):
    labeled = measure.label(mask.astype(bool), connectivity=2)
    cleaned = np.zeros_like(mask)
    for region in measure.regionprops(labeled):
        area = region.area
        bbox = region.bbox
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        if height > 0 and width > 0:
            aspect_ratio = max(height, width) / min(height, width)
        else:
            aspect_ratio = 1
        solidity = region.solidity if region.solidity > 0 else 0
        extent = region.extent if region.extent > 0 else 0
        keep_region = (
            area >= min_size and
            (aspect_ratio >= aspect_ratio_threshold or area >= min_size * 2) and
            solidity < 0.8 and
            extent < 0.6
        )
        if keep_region:
            coords = region.coords
            cleaned[coords[:, 0], coords[:, 1]] = 255
    return cleaned


def post_process_morphology(mask):
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    smoothed = cv2.medianBlur(opened, 3)
    return smoothed


def generate_mask(image, min_size=150, aspect_ratio_threshold=2.5):
    if image is None:
        raise ValueError("Imagem inválida ou não carregada corretamente.")
    enhanced_image, original_enhanced = enhance_cracks(image)
    initial_thresh, fallback_thresh = adaptive_threshold_analysis(enhanced_image)
    thresholds = [initial_thresh, fallback_thresh, initial_thresh + 20, initial_thresh - 20]
    best_mask = None
    best_score = 0
    for thresh in thresholds:
        if thresh < 5 or thresh > 200:
            continue
        _, mask = cv2.threshold(enhanced_image, thresh, 255, cv2.THRESH_BINARY)
        mask = post_process_morphology(mask)
        cleaned = remove_noise_advanced(mask, min_size, aspect_ratio_threshold)
        score = evaluate_mask_quality(cleaned, original_enhanced)
        if score > best_score:
            best_score = score
            best_mask = cleaned.copy()
    if best_mask is None or np.sum(best_mask) < 1000:
        print("Tentando abordagem conservadora...")
        return conservative_approach(image, min_size)
    return best_mask


def evaluate_mask_quality(mask, original_image):
    if np.sum(mask) == 0:
        return 0
    labeled = measure.label(mask.astype(bool), connectivity=2)
    num_components = np.max(labeled)
    coverage = np.sum(mask > 0) / mask.size
    score = 0
    if num_components > 0:
        avg_component_size = np.sum(mask > 0) / num_components
        if avg_component_size > 300:
            score += 50
    if 0.01 < coverage < 0.15:
        score += 30
    elif coverage < 0.01:
        score -= 20
    elif coverage > 0.2:
        score -= 50
    return score


def conservative_approach(image, min_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    sato_filtered = sato(enhanced, sigmas=range(1, 2), black_ridges=True)
    sato_norm = (sato_filtered - np.min(sato_filtered)) / (np.max(sato_filtered) - np.min(sato_filtered))
    sato_uint8 = (sato_norm * 255).astype(np.uint8)
    _, mask = cv2.threshold(sato_uint8, 60, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = remove_noise_advanced(mask, min_size * 2, 4.0)
    return cleaned


if __name__ == "__main__":
    datasets = ['train', 'val', 'test']
    classes = ['thermal', 'retraction']
    stats = {'total': 0, 'success': 0, 'failed': 0}
    for dataset in datasets:
        root_images = os.path.join('../yolo/images', dataset)
        root_masks = os.path.join('../yolo/masks', dataset)
        for class_name in classes:
            os.makedirs(os.path.join(root_masks, class_name), exist_ok=True)
        if not os.path.exists(root_images):
            print(f"Pasta de imagens não encontrada: {root_images}")
            continue
        for image_name in os.listdir(root_images):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            stats['total'] += 1
            upper_name = image_name.upper()
            if 'FT' in upper_name:
                mask_subfolder = 'thermal'
            elif 'FR' in upper_name:
                mask_subfolder = 'retraction'
            else:
                print(f"Nome inválido para classificar: {image_name}")
                continue
            image_path = os.path.join(root_images, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Erro ao carregar imagem: {image_path}")
                stats['failed'] += 1
                continue
            try:
                mask = generate_mask(image)
                if np.sum(mask) < 500:
                    print(f"Máscara muito vazia: {image_name}, tentando abordagem alternativa...")
                    mask = conservative_approach(image, 100)
                mask_path = os.path.join(root_masks, mask_subfolder, image_name)
                cv2.imwrite(mask_path, mask)
                print(f"Máscara salva com sucesso: {mask_path} (pixels: {np.sum(mask > 0)})")
                stats['success'] += 1
            except Exception as e:
                print(f"Erro ao gerar máscara para {image_name}: {e}")
                stats['failed'] += 1
                continue
    print("Processamento finalizado.")
    print(f"Imagens processadas com sucesso: {stats['success']}/{stats['total']} | Falhas: {stats['failed']}")
