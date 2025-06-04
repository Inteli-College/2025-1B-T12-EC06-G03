import os
from pathlib import Path
import numpy as np
import cv2
from skimage import io
from skimage.filters import sato
from skimage import measure
from skimage.exposure import rescale_intensity


# ------------------------------
# Parameters
# ------------------------------
INPUT_FOLDER = 'data/preprocessed'
OUTPUT_FOLDER = 'data/masks'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ------------------------------
# Sato Segmentation Function
# ------------------------------
def sato_segmentation(image):
    """
    Apply Sato filter-based segmentation with preprocessing and postprocessing
    tuned for crack-like linear structures.

    Args:
        image (np.ndarray): Input image (BGR or grayscale)

    Returns:
        binary_mask (np.ndarray): Binary mask (uint8, 0 and 255)
        sato_uint8 (np.ndarray): Grayscale Sato response (uint8)
    """
    if image is None:
        raise ValueError("Invalid image or failed to load.")

    # 🔸 Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 🔸 Bilateral Filter
    smoothed = cv2.bilateralFilter(gray, d=2, sigmaColor=25, sigmaSpace=25)

    # 🔸 CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)

    # 🔸 Sato Filter (sigma tuned for thin cracks)
    sato_response = sato(enhanced, sigmas=range(1, 2), black_ridges=True)

    # 🔸 Normalize Sato output to uint8
    sato_norm = (sato_response - np.min(sato_response)) / (np.max(sato_response) - np.min(sato_response))
    sato_uint8 = (sato_norm * 255).astype(np.uint8)

    # 🔸 Threshold to obtain binary mask
    _, binary_mask = cv2.threshold(sato_uint8, 75, 255, cv2.THRESH_BINARY)

    # 🔸 Morphological Closing (remove gaps in lines)
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # 🔸 Remove small objects (noise)
    mask_bool = binary_mask.astype(bool)
    mask_label = measure.label(mask_bool, connectivity=2)
    cleaned_mask = np.zeros_like(binary_mask)

    min_size = 200  # Minimum area of valid regions (pixels)
    for region in measure.regionprops(mask_label):
        if region.area >= min_size:
            for coord in region.coords:
                cleaned_mask[coord[0], coord[1]] = 255

    return cleaned_mask, sato_uint8


# ------------------------------
# Process All Images
# ------------------------------
def process_all_images(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    image_files = list(input_folder.glob('*.*'))

    for image_path in image_files:
        mask, sato_image = sato_segmentation(image_path)

        mask_output_path = output_folder / f"{image_path.stem}_mask.png"
        sato_output_path = output_folder / f"{image_path.stem}_sato.png"

        # Save the binary mask and the Sato-filtered image
        cv2.imwrite(str(mask_output_path), mask)
        cv2.imwrite(str(sato_output_path), sato_image)


        print(f"Segmented {image_path.name} → {mask_output_path.name}")


# ------------------------------
# Run
# ------------------------------
if __name__ == '__main__':
    process_all_images(INPUT_FOLDER, OUTPUT_FOLDER)