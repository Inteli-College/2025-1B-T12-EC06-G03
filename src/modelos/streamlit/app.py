import os
import cv2
import torch
import json
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from torch import nn

# Configurações
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(layout="wide", page_title="Classificação de Fissuras Estruturais", page_icon="👷🏽‍♂️")

# Custom CSS
st.markdown("""
    <style>
    /* Aumento geral da fonte */
    html, body, .stMarkdown, .stExpander, .stAlert, .stText {
        font-size: 24px !important;
    }
    
    /* Títulos principais */
    h1 {
        font-size: 3rem !important;
        margin-bottom: 2rem !important;
    }
    
    /* Títulos de seção */
    h2 {
        font-size: 2.5rem !important;
        margin-bottom: 1.8rem !important;
    }
    
    /* Texto em expanders */
    .stExpander .markdown-text-container {
        font-size: 1.4rem !important;
        line-height: 1.6 !important;
    }
    
    /* Resultados */
    .result-box h3 {
        font-size: 1.5rem !important;
    }
    
    .result-box p {
        font-size: 1.5rem !important;
        margin-top: 1rem !important;
    }
    
    /* Labels dos uploaders e colunas */
    .stFileUploader label,
    .column label {
        font-size: 1.6rem !important;
    }
    
    .stMarkdown p {
        line-height: 2em !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def carregar_modelos():
    with st.spinner('Carregando modelos...'):
        # Carregar mapeamento de classes
        with open("models/class_to_idx.json", "r") as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        # Carregar modelos
        yolo_model = YOLO('models/yolo.pt')
        cnn_model = CNN().to(DEVICE)
        cnn_model.load_state_dict(torch.load('models/cnn_model.pt', map_location=DEVICE))
        cnn_model.eval()
    
    return yolo_model, cnn_model, idx_to_class

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 30 * 30, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

def classificar_fissura(crop_bgr, cnn_model, idx_to_class):
    resized = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE))
    tensor = cnn_transform(resized).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = cnn_model(tensor)
        pred = torch.argmax(output, dim=1).item()
        classe_bruta = idx_to_class[pred]
        return classe_bruta.split("_")[-1]

cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def processar_imagem(img_array, yolo_model, cnn_model, idx_to_class):
    img = img_array.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (0, 0), 3)
    sharpened = cv2.addWeighted(eq, 1.2, blur, -0.2, 0)
    img_yolo = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    results = yolo_model(img_yolo, conf=0.05)[0]

    if not results.boxes or len(results.boxes) == 0:
        return "Nenhuma fissura detectada", None

    confs = results.boxes.conf
    idx_max = torch.argmax(confs).item()
    x1, y1, x2, y2 = map(int, results.boxes.xyxy[idx_max])
    crop = img[y1:y2, x1:x2]

    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return "Fissura muito pequena para análise", None

    label_cnn = classificar_fissura(crop, cnn_model, idx_to_class)
    resultado_final = f"Fissura do tipo {'retração' if label_cnn == 'retraction' else 'térmica'} detectada"

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{label_cnn}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return resultado_final, img

def main():
    st.title("Sistema de Detecção e Classificação de Fissuras Estruturais")
    
    tab1, tab2 = st.tabs(["Sobre o Projeto", "Análise de Imagens"])
    
    with tab1:
        with st.expander("Objetivo do Projeto"):
            st.markdown("""
                Este sistema utiliza redes neurais profundas para detecção e classificação automática de fissuras 
                em estruturas civis, auxiliando na manutenção preditiva e inspeções de segurança.
            """)
        
        with st.expander("Metodologia Técnica"):
            st.markdown("""
                - **YOLOv8**: Para detecção inicial de fissuras\n
                - **CNN Customizada**: Para classificação do tipo de fissura\n
                - **Pré-processamento**: CLAHE e sharpening para melhorar a detecção
            """)
        
        with st.expander("Benefícios"):
            st.markdown("""
                - Redução do tempo de inspeção\n
                - Precisão de detecção superior a 92%\n
                - Interface amigável para técnicos e engenheiros
            """)

    with tab2:
        uploaded_files = st.file_uploader("Selecione uma ou mais imagens para análise", 
                                       type=["png", "jpg", "jpeg"], 
                                       accept_multiple_files=True)
        
        if uploaded_files:
            yolo_model, cnn_model, idx_to_class = carregar_modelos()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f'Processando imagem {idx+1}/{len(uploaded_files)}...'):
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    opencv_image = cv2.imdecode(file_bytes, 1)
                    
                    resultado, imagem_annotada = processar_imagem(opencv_image, yolo_model, cnn_model, idx_to_class)
                    
                    # Container de resultado destacado
                    box_class = "result-box" if "detectada" in resultado else "result-box warning-box"
                    st.markdown(f"""
                        <div class="{box_class}">
                            <h3 style="color: #000000;">
                                Resultado da Imagem {idx+1}
                            </h3>
                            <p style="color: #000000; font-weight: bold; text-decoration: underline;">
                                {resultado}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Imagem Original**")
                        pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
                        st.image(pil_image.resize((400, int(400 * pil_image.size[1]/pil_image.size[0]))), 
                               use_container_width=False, width=800)
                    
                    with col2:
                        st.markdown(f"**Imagem Anotada**")
                        if imagem_annotada is not None:
                            pil_annotada = Image.fromarray(cv2.cvtColor(imagem_annotada, cv2.COLOR_BGR2RGB))
                            st.image(pil_annotada.resize((400, int(400 * pil_annotada.size[1]/pil_annotada.size[0]))), 
                                   use_container_width=False, width=800)
                        else:
                            st.warning("Não foi possível gerar anotações")
                    
                st.markdown("---")

if __name__ == "__main__":
    main()