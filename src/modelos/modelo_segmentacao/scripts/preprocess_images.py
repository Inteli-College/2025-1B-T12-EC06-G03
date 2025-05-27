import os
import cv2

def preprocess_folder(folder_path):
    output_dir = os.path.join(folder_path, "processadas")
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(folder_path, fname)
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            # Converter para cinza
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Aplicar CLAHE (melhora contraste de forma mais suave)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(gray)

            # Realce de nitidez mais leve
            blurred = cv2.GaussianBlur(equalized, (0, 0), 3)
            sharpened = cv2.addWeighted(equalized, 1.2, blurred, -0.2, 0)

            # Salvar imagem processada
            output_path = os.path.join(output_dir, fname)
            cv2.imwrite(output_path, sharpened)

    print(f"✅ Processadas com CLAHE: {folder_path} → {output_dir}")

# --- Pré-processar imagens do YOLO ---
preprocess_folder("../yolo/images/train")
preprocess_folder("../yolo/images/val")
preprocess_folder("../yolo/images/test")



