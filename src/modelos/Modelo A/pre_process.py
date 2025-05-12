import cv2
import numpy as np
from pathlib import Path

# Parâmetros de pré‑processamento
BLUR_KERNEL    = (3, 3)  # Reduzido para preservar mais detalhes
SHARPEN_KERNEL = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])  # Kernel mais agressivo

# Diretórios base
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def preprocess_crack_image(image_path: Path, output_path: Path):
    """
    Lê imagem em gray, aplica blur, sharpening agressivo 
    e salva em output_path.
    """
    # 1) Leitura em grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE) 
    if img is None:
        print(f"Aviso: não foi possível ler {image_path}")
        return

    # 2) Blur para reduzir ruído (mantendo detalhes)
    blurred = cv2.GaussianBlur(img, BLUR_KERNEL, 0)          

    # 3) Sharpening agressivo para destacar fissuras
    sharpened = cv2.filter2D(blurred, -1, SHARPEN_KERNEL)    

    # 4) Garante existência de pasta de saída
    output_path.parent.mkdir(parents=True, exist_ok=True)    

    # 5) Salva imagem processada com mesma extensão
    cv2.imwrite(str(output_path), sharpened)                 

def main():
    # Para cada subpasta em data/raw (tipos de fissura)
    for category_dir in RAW_DIR.iterdir():                   
        if not category_dir.is_dir():
            continue
        # Processa cada PNG dentro da subpasta
        for img_path in category_dir.glob("*.PNG"):          
            # Define saída em data/processed/<tipo>/<nome>
            out_path = PROCESSED_DIR / category_dir.name / img_path.name
            preprocess_crack_image(img_path, out_path)
            print(f"Processado: {img_path} → {out_path}")

if __name__ == "__main__":
    main()
