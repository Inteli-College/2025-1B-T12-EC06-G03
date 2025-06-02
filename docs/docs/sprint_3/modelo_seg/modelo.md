---
title: Primeira versão do modelo de Segmentação de Fissuras
sidebar_position: 1
---

# Modelo de segmentação de Fissuras

## Introdução

&emsp;Este documento descreve o funcionamento de um sistema automatizado para **segmentação de fissuras** em imagens, com foco em aplicações na engenharia civil e inspeção de estruturas. O sistema utiliza um modelo **YOLOv8-seg** para **detecção e segmentação das fissuras** na imagem.

---

## Modelo YOLOv8-seg - Detecção de Fissura

&emsp;O modelo YOLOv8 é responsável por identificar a presença de **fissuras visíveis** em uma imagem. Ele foi treinado através da anotação automatizada em máscaras das imagens de input, marcando a localização e contorno da fissura.

## Desenvolvimento:

&emsp; O fluxo de treinamento do modelo de segmentação desenvolvido na Sprint 3 se adequa às requisições do modelo YOLOv8n-seg, utilizado para detecção e segmentação das fissuras. Para isso, para treinamento do modelo, seguem-se as etapas:

1. Inputs de imagens;

2. Geração de máscaras das imagens;

3. Geração de anotações para treinamento (com base nas imagens);

4. Treinamento do modelo;

5. Output das segmentações.

&emsp; Abaixo, detalha-se o funcionamento de cada etapa do processo de treinamento para encontrar os melhores pesos de funcionamento do modelo.

### Fluxo de treinamento do modelo

#### Geração de máscaras das imagens de input

&emsp; O arquivo `mask_gen.py` é responsável por gerar máscaras das imagens de input de treinamento. Abaixo, explora-se a função `generate_mask()` para geração de máscaras e as técnicas utilizadas para destacar as fissuras nas imagens e suavizar ruídos.

```python
def generate_mask(image):
    if image is None:
        raise ValueError("Imagem inválida ou não carregada corretamente.")

    # Transforma a imagem em escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Suavização bilateral
    smoothed = cv2.bilateralFilter(gray, d=2, sigmaColor=25, sigmaSpace=25)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)

    # Filtro Sato
    sato_filtered = sato(enhanced, sigmas=range(1, 2), black_ridges=True)

    # Normalização manual para [0, 255]
    sato_norm = (sato_filtered - np.min(sato_filtered)) / (np.max(sato_filtered) - np.min(sato_filtered))
    sato_uint8 = (sato_norm * 255).astype(np.uint8)

    # Binarização
    _, mask = cv2.threshold(sato_uint8, 75, 255, cv2.THRESH_BINARY)

    # Fechamento morfológico
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remoção de pequenos objetos
    mask_bool = mask.astype(bool)
    mask_clean = measure.label(mask_bool, connectivity=2)
    cleaned_mask = np.zeros_like(mask)

    min_size = 200  # Threshold de tamanho mínimo da fissura
    for region in measure.regionprops(mask_clean):
        if region.area >= min_size:
            for coord in region.coords:
                cleaned_mask[coord[0], coord[1]] = 255

    return cleaned_mask
```

&emsp; Primeiro, valida-se a imagem recebida pela função para geração da máscara no trecho abaixo:

```python
if image is None:
    raise ValueError("Imagem inválida ou não carregada corretamente.")

```

&emsp; Após validação de carregamento da imagem, realiza-se um tratamento inicial da imagem, transformando a imagem para escala de cinza e suavizando a imagem, para redução de ruídos. Aqui, é importante destacar o a utilização do filtro lateral para suavização, preservando os contornos da fissuras.

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

smoothed = cv2.bilateralFilter(gray, d=2, sigmaColor=25, sigmaSpace=25)
```

&emsp; Para destaque das fissuras, aplica-se a técnica de contraste CLAHE:

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(smoothed)
```

&emsp; Após isso, um filtro de identificação de estruras lineares é utilizado. O filtro Sato é aplicado para detecção de cristas e delineação das fissuras. Para futura binarização das cores da imagem (em preto e branco), normaliza-se o os valores dos pixels da imagem, retornando uma matriz que permite a visualização dos resultados e binarização. 

```python
sato_filtered = sato(enhanced, sigmas=range(1, 2), black_ridges=True)

sato_norm = (sato_filtered - np.min(sato_filtered)) / (np.max(sato_filtered) - np.min(sato_filtered))
sato_uint8 = (sato_norm * 255).astype(np.uint8)
```

&emsp; Converte-se a imagem para preto e branco, definindo o threshold de intensidade, isolando as estruturas identificadas pelo filtro Sato.

```python
_, mask = cv2.threshold(sato_uint8, 75, 255, cv2.THRESH_BINARY)
```

&emsp; Utiliza-se uma técnica de fechamento morfológico, conectando os fragmentos identificados anteriormente para continuidade da fissura identificada. 

```python
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```

&emsp; Finalmente, exclui-se os ruídos da imagem, reduzindo a quantidade de identificações falsas positivas e mantendo identificações com área maior do que 200 pixels.

```python
mask_bool = mask.astype(bool)
mask_clean = measure.label(mask_bool, connectivity=2)
cleaned_mask = np.zeros_like(mask)

min_size = 200
for region in measure.regionprops(mask_clean):
    if region.area >= min_size:
        for coord in region.coords:
            cleaned_mask[coord[0], coord[1]] = 255
```

#### Geração de máscaras das imagens de input

&emsp; O arquivo `label_gen.py` é responsável por gerar máscaras das imagens de input de treinamento. Abaixo, explora-se a função `generate_label_line()` para geração de máscaras e as técnicas utilizadas para destacar as fissuras nas imagens e suavizar ruídos.

```python
def generate_label_line(contour, class_id, img_w, img_h):
    """Gera uma linha de label no formato YOLO-seg."""
    x, y, w, h = cv2.boundingRect(contour)
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # Normaliza pontos do polígono
    polygon = contour.reshape(-1, 2)
    poly_norm = [
        f"{(px / img_w):.6f} {(py / img_h):.6f}"
        for px, py in polygon
    ]
    poly_flat = ' '.join(poly_norm)

    return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} {poly_flat}"
```

&emsp; Na função de geração das anotações para treinamento do modelo, destaca-se o trecho de extração e normalização dos pontos do polígono de contorno das fissuras:

```python
polygon = contour.reshape(-1, 2)
poly_norm = [
    f"{(px / img_w):.6f} {(py / img_h):.6f}"
    for px, py in polygon
]
```

&emsp; O retorno da função, por fim, gera uma string de formato ideal para treinamento do modelo **YOLOv8-seg**.

#### Treinamento do modelo YOLOv8-seg

&emsp; O arquivo `train_yolo_seg.py` é responsável por treinar o modelo **YOLOv8-seg**. Utiliza-se os dados de anotações gerados em `label_gen.py` para treinamento do modelo. Abaixo, o código de treinamento é apresentado: 

```python
from ultralytics import YOLO

# 🔧 Carregar o modelo pré-treinado de segmentação
model = YOLO('yolov8n-seg.pt')  # você pode trocar por yolov8s-seg.pt, yolov8m-seg.pt, etc.

# 🚀 Iniciar o treinamento
model.train(
    data='../data.yaml',  # Caminho para o arquivo de configuração
    epochs=200,             # Número de épocas
    imgsz=640,              # Tamanho das imagens
    batch=8,                # Batch size
    project='../runs',    # Pasta onde os resultados serão salvos
    name='segmentation_model', # Nome do experimento
    save=True,
    save_period=10,         # Salva checkpoint a cada 10 épocas
    patience=50,            # Early stopping se não melhorar após 20 épocas
    pretrained=True         # Usar pesos pré-treinados
)
```

&emsp; Os parâmetros de treinamento são definidos na chamada da função `model.train()`. A primeira versão do modelo foi treinado em 200 épocas, e os melhores pesos encontrados são salvos em `runs/segmentation_model`.

## Output de segmentações

&emsp; Para validar o modelo e as identificações das fissuras, o script `predict_yolo_seg.py` permite aplicar o modelo nas imagens de teste, seguindo o pipeline descrito abaixo

1. Recebe uma imagem original.
2. Aplica a geração de máscaras das imagens.
3. Anotações são feitas com base nas máscaras geradas.
4. O script retorna as coordenadas dos polígonos das fissuras.
5. O modelo YOLOv8-seg é treinado com base nas coordenadas identificadas.
6. As áreas das fissuras são identificadas nas imagens de input.
7. As coordenadas dos polígonos encontrados são classificados entre fissuras `térmicas` e `retração`.
8. As saídas, contendo a segmentação e classificação das fissuras na imagem são salvas.

* O diretório `../yolo/output/` recebe as imagens classificadas com fissuras segmentadas.

&emsp; Esse fluxo permite reaproveitamento modular, validação visual e explicabilidade.

---

## Como rodar o modelo

&emsp; Para teste do modelo de segmentação, basta instalar as dependências necessárias no diretório `../src/modelos/modelo_segmentacao` através do comando `pip install -r requirements.txt`.

&emsp; Após a instalação das dependências, basta entrar no diretório `../src/modelos/modelo_segmentacao/scripts` através do comando `cd scripts` e executar o comando de ativação da interface provisória do modelo através do comando `streamlit run app.py`. Ao executar o *script*, o aplicativo WEB será aberto em seu navegador padrão e, para teste do modelo, basta seguir o passo-a-passo na plataforma.

## Conclusão

&emsp; O modelo de segmentação desenvolvido na Sprint 3 é um piloto do modelo de segmentação a ser entrege ao final do projeto Athena, em parceria com o IPT e, portanto, passará por melhorias em seu fluxo de funcionamento. Futuramente, o grupo Athena explorará outras maneiras de treinamento, geração de máscaras e geração de anotações, visando melhoria da fidelidade das segmentações.

&emsp; O modelo de segmentação oferece mais informações quanto as fissuras quando comparado com o modelo de classificação. O fluxo de treinamento e predição das imagens utilizado nesta Sprint é apenas um das diversas maneiras possíveis para utilização do modelo **YOLOv8-seg**. Como validado pelos técnicos e engenheiros do IPT, o modelo de segmentação é interessante e possui significativo valor aos funcionários da instituição, especialmente quando consideradas as possíveis utilizações do modelo para cálculo de tamanho e espessura das fissuras.

---

## Bibliografia

- MARENGONI, Maurício; STRINGHINI, Stringhini. Tutorial: Introdução à visão computacional usando opencv. Revista de Informática Teórica e Aplicada, v. 16, n. 1, p. 125-160, 2009.
- HUSSAIN, Mazhar. YOLOv8 Real-time Instance Segmentation with Python. 2024.
