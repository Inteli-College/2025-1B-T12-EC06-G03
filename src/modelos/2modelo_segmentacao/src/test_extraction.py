import cv2
from pathlib import Path
import os

from utils.preprocessing import enhance_image
from utils.segmentation import sato_segmentation
from utils.crack_extraction import extract_cracks


# -----------------------
# Configuration
# -----------------------
INPUT_FOLDER = Path('./data/test_images')
OUTPUT_FOLDER = Path('../outputs/segmentation_results')

# Subfolders for outputs
MASKS_FOLDER = OUTPUT_FOLDER / 'masks'
VISUALIZATIONS_FOLDER = OUTPUT_FOLDER / 'visualizations'
BBOX_FOLDER = OUTPUT_FOLDER / 'bounding_boxes'

# Create folders if not exist
for folder in [MASKS_FOLDER, VISUALIZATIONS_FOLDER, BBOX_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)


# -----------------------
# Run Extraction
# -----------------------
def run_crack_extraction():
    image_files = list(INPUT_FOLDER.glob('*.*'))

    if not image_files:
        print(f"No images found in {INPUT_FOLDER}")
        return

    print(f"Looking for images in {INPUT_FOLDER.resolve()}")
    print(f"Found {len(image_files)} images.")

    for img_path in image_files:
        print(f"Processing {img_path}")
        image = cv2.imread(str(img_path))

        if image is None:
            print(f"Failed to load {img_path}")
            continue

        # Enhancement
        enhanced = enhance_image(image)

        # Segmentation
        mask, _ = sato_segmentation(enhanced)

        # Save mask
        mask_filename = MASKS_FOLDER / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(mask_filename), mask)
        print(f"Saved mask to {mask_filename}")

        # Crack extraction (bounding boxes)
        crack_regions = extract_cracks(mask, enhanced, BBOX_FOLDER, image_stem=img_path.stem)

        # Visualize cracks
        vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        for bbox in crack_regions:
            x, y, w, h = bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        vis_filename = VISUALIZATIONS_FOLDER / f"{img_path.stem}_vis.png"
        cv2.imwrite(str(vis_filename), vis)
        print(f"Saved visualization to {vis_filename}")

    print("✅ Crack extraction completed.")


if __name__ == '__main__':
    run_crack_extraction()