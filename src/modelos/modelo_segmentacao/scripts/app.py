import streamlit as st
import os
from PIL import Image
import shutil
import numpy as np
import cv2
from ultralytics import YOLO


# Caminhos para o modelo e pastas de upload/output
MODEL_PATH = '../runs/segmentation_model/weights/best.pt'
UPLOAD_FOLDER = './yolo/uploads'
OUTPUT_FOLDER = './yolo/output'

CLASS_MAP = {
    0: "Térmica",
    1: "Retração"
}

# Carrega o modelo YOLO com cache para otimização
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

# Upload e tratamento das imagens de upload
def handle_upload():
    uploaded_files = st.file_uploader(
        "📤 Faça upload de uma ou mais imagens para predição:",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Limpa pastas anteriores
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        st.subheader('🔍 Imagens carregadas:')
        cols = st.columns(3)

        for idx, uploaded_file in enumerate(uploaded_files):
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            image = Image.open(uploaded_file)
            cols[idx % 3].image(image, caption=uploaded_file.name)

    return uploaded_files


# Função de predição de imagens
def run_prediction(model, uploaded_files):
    with st.spinner('Rodando modelo, aguarde...'):
        results = model.predict(
            source=UPLOAD_FOLDER,
            conf=0.1,
            save=False,
            stream=True
        )

        st.success('✅ Predição concluída!')

        st.subheader('🖼️ Resultados da Segmentação:')

        cols = st.columns(3)

        for idx, result in enumerate(results):
            img = result.orig_img.copy()
            output = img.copy()

            masks = result.masks
            class_counts = {}
            image_name = os.path.basename(result.path)

            if masks is not None:
                masks_data = masks.data.cpu().numpy()

                class_ids = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

                for mask, class_id, conf in zip(masks_data, class_ids, confidences):
                    cls_name = CLASS_MAP.get(int(class_id), 'Desconhecido')
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                    mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255

                    combined_mask = cv2.bitwise_or(combined_mask, mask_bin)

                color_mask = np.zeros_like(output, dtype=np.uint8)
                color_mask[:, :] = (255, 0, 0)  # Vermelho

                mask_rgb = cv2.merge([combined_mask] * 3)
                overlay = cv2.addWeighted(output, 1, cv2.bitwise_and(color_mask, mask_rgb), 0.4, 0)

                if class_counts:
                    most_predicted_class = max(class_counts, key=class_counts.get)
                    caption = f"Resultado {idx + 1} - {most_predicted_class} - {image_name}"
                else:
                    caption = f"Resultado {idx + 1} - Sem fissura detectada - {image_name}"

                cols[idx % 3].image(overlay, caption=caption)

            else:
                cols[idx % 3].image(
                    output,
                    caption=f"Resultado {idx + 1} - Sem fissura detectada - {image_name}"
                )



# Função de geração da página streamlit
def main():
    # Configuração da página
    st.set_page_config(page_title="Segmentação de Fissuras", layout="wide")
    st.title('🧠 Modelo de Segmentação de Fissuras')

    # CSS para estilização
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

    # Descrição da Aplicação
    st.markdown('''
    Plataforma intermediária para validação da primeira versão do modelo de detecção e **segmentação de fissuras térmicas e retrações** em imagens. 

    📂 Faça upload de imagens, e o modelo irá gerar a previsão segmentada destacando as fissuras detectadas.
    ''')

    # Carrega modelo
    model = load_model()

    # Upload e visualização
    uploaded_files = handle_upload()

    if uploaded_files:
        if st.button('🚀 Rodar predição'):
            run_prediction(model, uploaded_files)

if __name__ == '__main__':
    main()