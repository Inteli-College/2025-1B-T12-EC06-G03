import cv2
import numpy as np
from pathlib import Path
import shutil

# Diretórios base (relativos a Modelo A/)
BASE          = Path(__file__).parent
DATASET_BASE  = BASE / "data" / "datasets" / "concrete" / "concreteCrackSegmentationDataset"
BW_DIR        = DATASET_BASE / "BW"   # máscaras BW
RGB_DIR       = DATASET_BASE / "rgb"  # imagens RGBM sem canal alpha
YOLO_IMG      = BASE / "yolo_data" / "images"
YOLO_LABELS   = BASE / "yolo_data" / "labels"

# Particionamento 80/20
all_imgs = sorted(RGB_DIR.glob("*.jpg"))
split_idx = int(0.8 * len(all_imgs))
splits = {
    "train": all_imgs[:split_idx],
    "val":   all_imgs[split_idx:],
}

def mask_to_txt(img_path: Path, split: str):
    # corresponde a máscara BW
    mask_path = BW_DIR / img_path.name
    mask      = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    h, w      = mask.shape

    # binariza e extrai contornos
    _, binm      = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _  = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # monta linhas YOLO-seg
    lines = []
    for cnt in contours:
        pts = cnt.squeeze().astype(float)
        # se for só um ponto, pula
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue
        # normaliza
        pts[:, 0] /= w
        pts[:, 1] /= h
        flat = pts.reshape(-1)
        vals = " ".join(f"{c:.6f}" for c in flat.tolist())
        lines.append(f"0 {vals}")

    # grava label
    lbl_out = YOLO_LABELS / split / img_path.name.replace(".jpg", ".txt")
    lbl_out.parent.mkdir(parents=True, exist_ok=True)
    lbl_out.write_text("\n".join(lines))

    # copia imagem RGB para pasta de treino/val
    img_out = YOLO_IMG / split / img_path.name
    img_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, img_out)

for split, imgs in splits.items():
    print(f"Processando {len(imgs)} imagens em '{split}'...")
    for img_path in imgs:
        mask_to_txt(img_path, split)

print("Concluído.")
