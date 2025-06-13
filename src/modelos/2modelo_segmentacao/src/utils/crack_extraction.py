import cv2
import os
from pathlib import Path
import numpy as np
from skimage import io


# ------------------------------
# Parameters
# ------------------------------
MASK_FOLDER = '../outputs/segmentation_results/masks'
ENHANCED_FOLDER = '../outputs/segmentation_results/bounding_boxes'
OUTPUT_FOLDER = '../src/data/crack_crops'

# Minimum area to consider a valid crack (filter small noise)
MIN_AREA = 200  

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ------------------------------
# Crack Region Extraction Function
# ------------------------------
def extract_cracks(mask, enhanced, output_folder, image_stem="image"):
    """
    Extract crack regions from mask and enhanced image, save crops,
    and return their bounding boxes.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crack_bboxes = []

    crack_id = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # Minimum area threshold
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        crack_bboxes.append((x, y, w, h))

        crop = enhanced[y:y + h, x:x + w]

        output_path = output_folder / f"{image_stem}_crack{crack_id}.png"
        cv2.imwrite(str(output_path), crop)

        print(f"✅ Saved crack {crack_id} from {image_stem} → {output_path.name}")
        crack_id += 1

    return crack_bboxes

# ------------------------------
# Process All Masks
# ------------------------------
def process_all_masks(mask_folder, enhanced_folder, output_folder):
    mask_folder = Path(mask_folder)
    output_folder = Path(output_folder)

    mask_files = list(mask_folder.glob('*_mask.png'))

    for mask_path in mask_files:
        extract_cracks(mask_path, enhanced_folder, output_folder)


# ------------------------------
# Run
# ------------------------------
if __name__ == '__main__':
    process_all_masks(MASK_FOLDER, ENHANCED_FOLDER, OUTPUT_FOLDER)
