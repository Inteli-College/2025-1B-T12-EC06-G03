import os
import cv2
import numpy as np
from skimage.filters import sato, frangi, meijering
from skimage import measure, morphology
from scipy import ndimage


def adaptive_threshold_analysis(image):
    """
    Analisa a imagem para determinar os melhores parâmetros de threshold
    baseado nas características da imagem.
    """
    # Calcula estatísticas da imagem
    mean_val = np.mean(image)
    std_val = np.std(image)
    
    # Threshold adaptativo baseado na distribuição dos pixels
    adaptive_thresh = mean_val - 0.5 * std_val
    fallback_thresh = mean_val - 1.2 * std_val
    
    return max(20, min(120, int(adaptive_thresh))), max(10, min(80, int(fallback_thresh)))


def enhance_cracks(image):
    """
    Aplica múltiplos filtros para melhorar a detecção de fissuras.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Pré-processamento mais robusto
    # 1. Suavização bilateral mais suave
    bilateral = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    
    # 2. CLAHE com parâmetros otimizados
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)
    
    # 3. Combinação de filtros de detecção de linhas
    # Sato filter
    sato_filtered = sato(enhanced, sigmas=range(1, 2), black_ridges=True)
    
    # Frangi filter (melhor para estruturas tubulares/lineares)
    frangi_filtered = frangi(enhanced, sigmas=range(1, 4), black_ridges=True)
    
    # Meijering filter (alternativa para linhas finas)
    meijering_filtered = meijering(enhanced, sigmas=range(1, 3), black_ridges=True)
    
    # Combina os filtros com pesos
    combined = (0.4 * sato_filtered + 0.4 * frangi_filtered + 0.2 * meijering_filtered)
    
    # Normalização
    combined_norm = (combined - np.min(combined)) / (np.max(combined) - np.min(combined))
    combined_uint8 = (combined_norm * 255).astype(np.uint8)
    
    return combined_uint8, enhanced


def remove_noise_advanced(mask, min_size=200, aspect_ratio_threshold=3.0):
    """
    Remove ruído baseado em características geométricas das fissuras.
    """
    # Conectividade 8 para melhor detecção de componentes
    labeled = measure.label(mask.astype(bool), connectivity=2)
    cleaned = np.zeros_like(mask)
    
    for region in measure.regionprops(labeled):
        # Filtros baseados em características de fissuras
        area = region.area
        bbox = region.bbox
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        
        # Ratio de aspecto (fissuras tendem a ser longas e finas)
        if height > 0 and width > 0:
            aspect_ratio = max(height, width) / min(height, width)
        else:
            aspect_ratio = 1
        
        # Solidez (fissuras têm baixa solidez)
        solidity = region.solidity if region.solidity > 0 else 0
        
        # Extent (proporção da área da região em relação à bounding box)
        extent = region.extent if region.extent > 0 else 0
        
        # Critérios para manter a região
        keep_region = (
            area >= min_size and
            (aspect_ratio >= aspect_ratio_threshold or area >= min_size * 2) and
            solidity < 0.8 and  # Fissuras não são muito sólidas
            extent < 0.6        # Fissuras não preenchem completamente a bounding box
        )
        
        if keep_region:
            coords = region.coords
            cleaned[coords[:, 0], coords[:, 1]] = 255
    
    return cleaned


def post_process_morphology(mask):
    """
    Aplica operações morfológicas para conectar fissuras próximas e suavizar.
    """
    # Elemento estrutural para conectar fissuras próximas
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    
    # Fechamento para conectar pequenas quebras
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Abertura para remover pequenos ruídos
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    # Filtro mediano para suavizar
    smoothed = cv2.medianBlur(opened, 3)
    
    return smoothed


def generate_mask(image, min_size=150, aspect_ratio_threshold=2.5):
    """
    Versão melhorada da geração de máscara com múltiplas estratégias.
    
    Args:
        image (np.ndarray): imagem BGR.
        min_size (int): área mínima para a remoção de ruído.
        aspect_ratio_threshold (float): threshold para ratio de aspecto.

    Returns:
        np.ndarray: máscara binária uint8 (0 e 255).
    """
    if image is None:
        raise ValueError("Imagem inválida ou não carregada corretamente.")

    # 1. Melhoria na detecção de fissuras
    enhanced_image, original_enhanced = enhance_cracks(image)
    
    # 2. Threshold adaptativo baseado na imagem
    initial_thresh, fallback_thresh = adaptive_threshold_analysis(enhanced_image)
    
    # 3. Múltiplas tentativas de threshold
    thresholds = [initial_thresh, fallback_thresh, initial_thresh + 20, initial_thresh - 20]
    
    best_mask = None
    best_score = 0
    
    for thresh in thresholds:
        if thresh < 5 or thresh > 200:
            continue
            
        # Threshold
        _, mask = cv2.threshold(enhanced_image, thresh, 255, cv2.THRESH_BINARY)
        
        # Pós-processamento morfológico
        mask = post_process_morphology(mask)
        
        # Remoção avançada de ruído
        cleaned = remove_noise_advanced(mask, min_size, aspect_ratio_threshold)
        
        # Avalia a qualidade da máscara
        score = evaluate_mask_quality(cleaned, original_enhanced)
        
        if score > best_score:
            best_score = score
            best_mask = cleaned.copy()
    
    # Se não encontrou nada bom, tenta uma abordagem mais conservadora
    if best_mask is None or np.sum(best_mask) < 1000:
        print("⚠️ Tentando abordagem conservadora...")
        return conservative_approach(image, min_size)
    
    return best_mask


def evaluate_mask_quality(mask, original_image):
    """
    Avalia a qualidade da máscara baseado em características esperadas de fissuras.
    """
    if np.sum(mask) == 0:
        return 0
    
    # Número de componentes (preferível poucos componentes grandes)
    labeled = measure.label(mask.astype(bool), connectivity=2)
    num_components = np.max(labeled)
    
    # Proporção da imagem coberta (fissuras não devem cobrir muito)
    coverage = np.sum(mask > 0) / mask.size
    
    # Score baseado em múltiplos fatores
    score = 0
    
    # Penaliza muitos componentes pequenos (ruído)
    if num_components > 0:
        avg_component_size = np.sum(mask > 0) / num_components
        if avg_component_size > 300:  # Componentes maiores são melhores
            score += 50
    
    # Penaliza cobertura excessiva (muito ruído)
    if 0.01 < coverage < 0.15:  # Faixa ideal para fissuras
        score += 30
    elif coverage < 0.01:
        score -= 20  # Muito pouco
    elif coverage > 0.2:
        score -= 50  # Muito ruído
    
    return score


def conservative_approach(image, min_size):
    """
    Abordagem mais conservadora para casos difíceis.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Suavização mais agressiva
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # CLAHE mais suave
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Apenas Sato com parâmetros mais conservadores
    sato_filtered = sato(enhanced, sigmas=range(1, 2), black_ridges=True)
    sato_norm = (sato_filtered - np.min(sato_filtered)) / (np.max(sato_filtered) - np.min(sato_filtered))
    sato_uint8 = (sato_norm * 255).astype(np.uint8)
    
    # Threshold mais alto para reduzir ruído
    _, mask = cv2.threshold(sato_uint8, 60, 255, cv2.THRESH_BINARY)
    
    # Morfologia mais agressiva para limpeza
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Remoção de ruído mais rigorosa
    cleaned = remove_noise_advanced(mask, min_size * 2, 4.0)
    
    return cleaned


if __name__ == "__main__":
    # 🔥 Executa o pipeline completo para gerar as máscaras dos datasets

    datasets = ['train', 'val', 'test']
    classes = ['thermal', 'retraction']

    # Estatísticas para análise
    stats = {'total': 0, 'success': 0, 'failed': 0}

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

            stats['total'] += 1
            
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
                stats['failed'] += 1
                continue

            try:
                mask = generate_mask(image)
                
                # Verifica se a máscara tem conteúdo suficiente
                if np.sum(mask) < 500:  # Muito pouco conteúdo
                    print(f"⚠️ Mask too sparse for {image_name}, trying alternative...")
                    mask = conservative_approach(image, 100)
                
                mask_path = os.path.join(root_masks, mask_subfolder, image_name)
                cv2.imwrite(mask_path, mask)
                print(f"✅ Mask saved: {mask_path} (pixels: {np.sum(mask > 0)})")
                stats['success'] += 1
                
            except Exception as e:
                print(f"❌ Error generating mask for {image_name}: {e}")
                stats['failed'] += 1
                continue

    print("🎯 Processing completed!")
    print(f"📊 Statistics: {stats['success']}/{stats['total']} successful, {stats['failed']} failed")