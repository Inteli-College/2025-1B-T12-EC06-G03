import cv2
import numpy as np
from pathlib import Path

YOLO_IMG    = Path("yolo_data/images/val/fissuras_de_retracao")
YOLO_LBL    = Path("yolo_data/labels/val/fissuras_de_retracao")
OVERLAY_DIR = Path("qc/overlays")
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

for img_path in YOLO_IMG.glob("*.jpg"):
    txt_path = YOLO_LBL / f"{img_path.stem}.txt"
    if not txt_path.exists():
        print(f"Aviso: sem anotação para {img_path.name}")
        continue

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    with open(txt_path) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            coords = np.array(parts[1:]).reshape(-1,2)
            pts = (coords * [w, h]).astype(int)
            cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)

    cv2.imwrite(str(OVERLAY_DIR / img_path.name), img)
