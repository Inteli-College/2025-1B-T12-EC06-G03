import cv2
import os
from pathlib import Path
import numpy as np
from skimage import exposure


# ------------------------------
# Parameters
# ------------------------------
INPUT_FOLDER = 'data/raw_images'
OUTPUT_FOLDER = 'data/preprocessed'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------------------
# Enhancement Function
# ------------------------------
def enhance_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(contrast_enhanced, (3, 3), sigmaX=0)

    # Optional: Sharpen (Unsharp masking)
    sharpen = cv2.addWeighted(contrast_enhanced, 1.5, blurred, -0.5, 0)

    return sharpen


# ------------------------------
# Process All Images
# ------------------------------
def process_all_images(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    image_files = list(input_folder.glob('*.*'))

    for image_path in image_files:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load {image_path}")
            continue
        enhanced = enhance_image(image)

        output_path = output_folder / image_path.name
        cv2.imwrite(str(output_path), enhanced)

        print(f"Processed {image_path.name} → {output_path.name}")


# ------------------------------
# Run
# ------------------------------
if __name__ == '__main__':
    process_all_images(INPUT_FOLDER, OUTPUT_FOLDER)