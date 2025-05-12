from ultralytics import YOLO
import cv2

# Caminho para a imagem de entrada
image_path = 'data/processed/fissuras_de_retracao/FR3.PNG'

# Carregar o modelo treinado
model = YOLO('runs/train_seg/weights/best.pt')  # Substitua pelo caminho do seu modelo treinado

# Realizar a predição
results = model.predict(source=image_path, save=False, show=True, conf=0.25)

# Exibir a imagem com as segmentações
# O parâmetro show=True já exibe a imagem com as máscaras sobrepostas
# Se desejar salvar a imagem resultante, descomente a linha abaixo:
results[0].save(filename='imagem_segmentada.jpg')


