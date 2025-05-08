import cv2
import numpy as np

def preprocess_crack_image(image_path, output_path):
    """
    Recebe uma imagem de uma rachadura, aplica filtros simples (blur, sharp, equalização de histograma e mediana) 
    e salva a imagem processada.

    :param image_path: Caminho para a imagem de entrada.
    :param output_path: Caminho para salvar a imagem processada.
    """
    # Carregar a imagem em escala de cinza
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Não foi possível carregar a imagem. Verifique o caminho fornecido.")

    # Aplicar filtro de desfoque (blur) para reduzir ruídos
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Criar um kernel para o filtro de nitidez (sharpen)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)

    # Salvar a imagem processada
    cv2.imwrite(output_path, sharpened)

if __name__ == "__main__":
    input_image_path = "data/raw/fissuras_termicas/FT58.PNG"
    output_image_path = "data/processed/fissuras_termicas/FT58.PNG"
    preprocess_crack_image(input_image_path, output_image_path)