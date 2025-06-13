import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from classifier import load_classifier, classify_crop

# -----------------------
# Configuration
# -----------------------
# Path to test dataset from modeloB
ROOT = Path(__file__).resolve().parents[2]
TEST_DATA = ROOT / 'modeloB' / 'data' / 'test'

# Load model and mapping
model, idx_to_class, device = load_classifier()

# Ground truth and predictions
y_true = []
y_pred = []

# Iterate over class subfolders
for class_name in idx_to_class.values():
    class_folder = TEST_DATA / class_name
    if not class_folder.exists():
        continue
    for img_path in class_folder.glob('*.*'):
        # Read crop image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        # Classify
        pred = classify_crop(img, model, idx_to_class, device)
        y_true.append(class_name)
        y_pred.append(pred)

# Compute metrics
print("\n=== Classification Metrics on Test Set ===")
print(f"Number of samples: {len(y_true)}")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
print(f"F1-score: {f1_score(y_true, y_pred, average='weighted'):.4f}\n")
print("Detailed classification report:")
print(classification_report(y_true, y_pred))
