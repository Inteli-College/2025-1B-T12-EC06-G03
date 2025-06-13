import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from utils.preprocessing import enhance_image
from utils.segmentation import sato_segmentation
from utils.crack_extraction import extract_cracks
from classifier import load_classifier, classify_crop

# -----------------------
# Configuration
# -----------------------
# Base project directory
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FOLDER = BASE_DIR / '2modelo_segmentacao' / 'src' / 'data' / 'test_images'
OUTPUT_FOLDER = BASE_DIR / '2modelo_segmentacao' / 'outputs'

# Segmentation subfolders
SEGMENT_FOLDER = OUTPUT_FOLDER / 'segmentation_results'
MASKS_FOLDER = SEGMENT_FOLDER / 'masks'
VISUALIZATIONS_FOLDER = SEGMENT_FOLDER / 'visualizations'
BBOX_FOLDER = SEGMENT_FOLDER / 'bounding_boxes'
# Predictions folder
PREDICTIONS_FOLDER = OUTPUT_FOLDER / 'predictions'
# Test set for evaluation (ground truth)
EVAL_TEST_FOLDER = BASE_DIR / 'modeloB' / 'data' / 'test'

# Create necessary directories
for folder in [MASKS_FOLDER, VISUALIZATIONS_FOLDER, BBOX_FOLDER, PREDICTIONS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# Load pretrained classifier
model, idx_to_class, device = load_classifier()
class_names = list(idx_to_class.values())

# -----------------------
# Run Extraction & Classification
# -----------------------
def run_crack_extraction():
    image_files = list(INPUT_FOLDER.glob('*.*'))
    if not image_files:
        print(f"No images found in {INPUT_FOLDER}")
        return

    print(f"Processing {len(image_files)} images from {INPUT_FOLDER}")
    for img_path in image_files:
        print(f"\n▶ Processing {img_path.name}")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ✖ Failed to load {img_path.name}")
            continue

        # Enhancement
        enhanced = enhance_image(image)

        # Segmentation
        mask, _ = sato_segmentation(enhanced)

        # Save mask
        mask_filename = MASKS_FOLDER / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(mask_filename), mask)

        # Crack extraction
        bboxes = extract_cracks(mask, enhanced, BBOX_FOLDER, image_stem=img_path.stem)
        print(f"  → Detected {len(bboxes)} cracks")

        # Prepare visualization and record predictions
        vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        predictions = []
        for (x, y, w, h) in bboxes:
            crop = enhanced[y:y+h, x:x+w]
            label = classify_crop(crop, model, idx_to_class, device)
            predictions.append({'bbox': (x, y, w, h), 'label': label})
            # Draw on visualization
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save visualization
        vis_filename = VISUALIZATIONS_FOLDER / f"{img_path.stem}_vis.png"
        cv2.imwrite(str(vis_filename), vis)
        # Save predictions
        pred_file = PREDICTIONS_FOLDER / f"{img_path.stem}_predictions.txt"
        with open(pred_file, 'w') as f:
            for p in predictions:
                x, y, w, h = p['bbox']
                f.write(f"{p['label']} bbox: {x},{y},{w},{h}\n")
        print(f"  ✓ Results saved for {img_path.name}")

    print("\n✅ Extraction and classification completed.")

# -----------------------
# Evaluation on Ground Truth Test Set
# -----------------------
def run_evaluation():
    print("\n=== Evaluating on Ground Truth Test Set ===")
    y_true, y_pred = [], []
    # Iterate true classes
    for class_name in class_names:
        folder = EVAL_TEST_FOLDER / class_name
        if not folder.exists():
            continue
        for img_path in folder.glob('*.*'):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            pred = classify_crop(img, model, idx_to_class, device)
            y_true.append(class_name)
            y_pred.append(pred)
    if not y_true:
        print("No ground truth images found for evaluation.")
        return

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Samples evaluated: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

# -----------------------
# Main
# -----------------------
if __name__ == '__main__':
    run_crack_extraction()
    run_evaluation()
